//  ModelStore.swift
//  Downloading a model to the phone, and keeping it there.
//
//  A GGUF is between half a gigabyte and eight. That size drives every
//  decision here:
//
//  * **Resumable.** A download that restarts from zero when the user
//    walks past a lift is not a download, it is a lottery. URLSession's
//    resume data survives suspension and relaunch.
//  * **Off the backup.** A re-downloadable four-gigabyte file must not
//    go into iCloud backup; Apple rejects apps that do this, and the
//    user's backup is not the place for something Hugging Face already
//    has.
//  * **Verified.** A truncated GGUF fails to load with an error that
//    reads like a broken model rather than a broken download, so the
//    size is checked before the file is offered as usable.

import Foundation

/// A model on disk.
struct InstalledModel: Codable, Sendable, Identifiable, Equatable {
    let repoID: String
    let filename: String
    let sizeBytes: Int
    let quant: String
    let installedAt: Date
    /// Read from the GGUF header after download; empty until then.
    var shape: StoredShape?

    var id: String { "\(repoID)/\(filename)" }
    var displayName: String {
        repoID.split(separator: "/").last.map(String.init) ?? repoID
    }
}

/// The parts of a `ModelShape` worth persisting.
///
/// Stored rather than re-read on every launch: parsing a GGUF header
/// means opening a multi-gigabyte file, and the shape does not change.
struct StoredShape: Codable, Sendable, Equatable {
    var parameters: Int
    var layers: Int
    var kvHeads: Int
    var headDim: Int
    var trainContext: Int
    var vocabSize: Int
    var embeddingDim: Int
    var tiedEmbeddings: Bool

    func modelShape(quant: String, fileBytes: Int, name: String) -> ModelShape {
        ModelShape(
            parameters: parameters, quant: quant, layers: layers, kvHeads: kvHeads,
            headDim: headDim, trainContext: trainContext, fileBytes: fileBytes,
            vocabSize: vocabSize, embeddingDim: embeddingDim,
            tiedEmbeddings: tiedEmbeddings, name: name
        )
    }
}

enum DownloadState: Sendable, Equatable {
    case idle
    case downloading(received: Int64, expected: Int64)
    case paused(received: Int64)
    case verifying
    case installed
    case failed(String)

    var fraction: Double {
        if case .downloading(let received, let expected) = self, expected > 0 {
            return Double(received) / Double(expected)
        }
        return 0
    }
}

enum ModelStoreError: LocalizedError {
    case notEnoughDisk(needed: Int64, free: Int64)
    case truncated(expected: Int64, got: Int64)
    case unauthorised
    case cancelled

    var errorDescription: String? {
        switch self {
        case .notEnoughDisk(let needed, let free):
            "This model needs \(ByteCountFormatter.string(fromByteCount: needed, countStyle: .file)) "
            + "and there is \(ByteCountFormatter.string(fromByteCount: free, countStyle: .file)) free."
        case .truncated(let expected, let got):
            "The download stopped early — got \(got) bytes of \(expected). "
            + "The file has been removed rather than left to fail as a broken model."
        case .unauthorised:
            "Hugging Face refused the download. If the model is gated, accept "
            + "its licence on the web with the account this token belongs to."
        case .cancelled:
            "Download cancelled."
        }
    }
}

/// Where models live, and how they get there.
@MainActor
final class ModelStore: NSObject, ObservableObject {
    @Published private(set) var installed: [InstalledModel] = []
    @Published private(set) var state: [String: DownloadState] = [:]

    private var tasks: [String: URLSessionDownloadTask] = [:]
    private var resumeData: [String: Data] = [:]
    private var token: String?
    private lazy var session: URLSession = {
        let config = URLSessionConfiguration.background(
            withIdentifier: "com.hypernix.hyperlink.models"
        )
        // A model download must survive the app being backgrounded;
        // that is the normal case for something that takes twenty
        // minutes, not the exception.
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true
        return URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }()

    /// `Application Support`, not `Documents`.
    ///
    /// Documents is user-visible in Files and is backed up. A model is
    /// neither user data nor worth putting in someone's iCloud backup.
    static var modelsDirectory: URL {
        let base = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        )[0].appendingPathComponent("Models", isDirectory: true)
        try? FileManager.default.createDirectory(
            at: base, withIntermediateDirectories: true
        )
        return base
    }

    func setToken(_ token: String?) { self.token = token }

    func url(for model: InstalledModel) -> URL {
        Self.modelsDirectory
            .appendingPathComponent(model.repoID.replacingOccurrences(of: "/", with: "__"))
            .appendingPathComponent(model.filename)
    }

    /// Free space on the volume the models live on.
    ///
    /// `volumeAvailableCapacityForImportantUsage` rather than
    /// `systemFreeSize`: the second counts space iOS will not actually
    /// give you, so a download starts and dies at 90%.
    static func freeBytes() -> Int64 {
        let values = try? modelsDirectory.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey]
        )
        return values?.volumeAvailableCapacityForImportantUsage ?? 0
    }

    func download(_ candidate: GGUFCandidate) throws {
        guard let url = candidate.downloadURL else { return }
        let needed = Int64(candidate.sizeBytes)
        let free = Self.freeBytes()
        // Checked before starting rather than discovered at 95%: a
        // twenty-minute cellular download that fails on disk space is
        // the worst version of this failure.
        if needed > 0, free > 0, needed + 256 * 1024 * 1024 > free {
            throw ModelStoreError.notEnoughDisk(needed: needed, free: free)
        }

        var request = URLRequest(url: url)
        if let token, !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        let task: URLSessionDownloadTask
        if let data = resumeData[candidate.id] {
            task = session.downloadTask(withResumeData: data)
            resumeData[candidate.id] = nil
        } else {
            task = session.downloadTask(with: request)
        }
        task.taskDescription = candidate.id
        tasks[candidate.id] = task
        state[candidate.id] = .downloading(received: 0, expected: needed)
        task.resume()
    }

    /// Pause, keeping what has arrived.
    func pause(_ candidateID: String) {
        guard let task = tasks[candidateID] else { return }
        task.cancel { [weak self] data in
            Task { @MainActor in
                guard let self else { return }
                if let data {
                    self.resumeData[candidateID] = data
                    self.state[candidateID] = .paused(received: Int64(data.count))
                } else {
                    self.state[candidateID] = .idle
                }
                self.tasks[candidateID] = nil
            }
        }
    }

    func cancel(_ candidateID: String) {
        tasks[candidateID]?.cancel()
        tasks[candidateID] = nil
        resumeData[candidateID] = nil
        state[candidateID] = .idle
    }

    func delete(_ model: InstalledModel) {
        try? FileManager.default.removeItem(at: url(for: model))
        installed.removeAll { $0.id == model.id }
        persist()
    }

    /// Total bytes models are using, for a settings screen that has to
    /// justify itself.
    var installedBytes: Int {
        installed.reduce(0) { $0 + $1.sizeBytes }
    }

    // MARK: - Persistence

    private var manifestURL: URL {
        Self.modelsDirectory.appendingPathComponent("installed.json")
    }

    func load() {
        guard
            let data = try? Data(contentsOf: manifestURL),
            let decoded = try? JSONDecoder().decode([InstalledModel].self, from: data)
        else { return }
        // Anything the manifest claims but the disk does not have is
        // dropped: a model deleted by iOS under storage pressure would
        // otherwise sit in the list and fail to load.
        installed = decoded.filter {
            FileManager.default.fileExists(atPath: url(for: $0).path)
        }
        if installed.count != decoded.count { persist() }
    }

    private func persist() {
        guard let data = try? JSONEncoder().encode(installed) else { return }
        try? data.write(to: manifestURL, options: .atomic)
    }

    fileprivate func finish(candidateID: String, temporary: URL, expected: Int64) {
        let parts = candidateID.split(separator: "/")
        guard parts.count >= 3 else { return }
        let repoID = parts.dropLast().joined(separator: "/")
        let filename = String(parts.last!)

        state[candidateID] = .verifying
        let attributes = try? FileManager.default.attributesOfItem(atPath: temporary.path)
        let actual = (attributes?[.size] as? NSNumber)?.int64Value ?? 0
        if expected > 0, actual < expected {
            // A truncated GGUF fails to load with an error that reads
            // like a broken model. Removing it keeps that confusion
            // from ever starting.
            try? FileManager.default.removeItem(at: temporary)
            state[candidateID] = .failed(
                ModelStoreError.truncated(expected: expected, got: actual)
                    .localizedDescription
            )
            return
        }

        let folder = Self.modelsDirectory
            .appendingPathComponent(repoID.replacingOccurrences(of: "/", with: "__"))
        try? FileManager.default.createDirectory(
            at: folder, withIntermediateDirectories: true
        )
        var destination = folder.appendingPathComponent(filename)
        try? FileManager.default.removeItem(at: destination)
        do {
            try FileManager.default.moveItem(at: temporary, to: destination)
        } catch {
            state[candidateID] = .failed(error.localizedDescription)
            return
        }

        // Keep it out of iCloud backup. Apple rejects apps that back up
        // re-downloadable data, and this is the definition of it.
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? destination.setResourceValues(values)

        let model = InstalledModel(
            repoID: repoID, filename: filename, sizeBytes: Int(actual),
            quant: ModelFit.quantFromFilename(filename), installedAt: Date(), shape: nil
        )
        installed.removeAll { $0.id == model.id }
        installed.append(model)
        persist()
        state[candidateID] = .installed
        tasks[candidateID] = nil
    }
}

extension ModelStore: URLSessionDownloadDelegate {
    nonisolated func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard let id = downloadTask.taskDescription else { return }
        Task { @MainActor in
            state[id] = .downloading(
                received: totalBytesWritten, expected: totalBytesExpectedToWrite
            )
        }
    }

    nonisolated func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        guard let id = downloadTask.taskDescription else { return }
        let expected = downloadTask.countOfBytesExpectedToReceive
        // The temporary file is deleted when this returns, so it has to
        // be moved somewhere durable synchronously — hopping to the
        // main actor first would race with that deletion.
        let staged = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try? FileManager.default.moveItem(at: location, to: staged)
        Task { @MainActor in
            finish(candidateID: id, temporary: staged, expected: expected)
        }
    }

    nonisolated func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        guard let id = task.taskDescription, let error else { return }
        let resume = (error as NSError)
            .userInfo[NSURLSessionDownloadTaskResumeData] as? Data
        Task { @MainActor in
            if let resume {
                resumeData[id] = resume
                state[id] = .paused(received: Int64(resume.count))
            } else if (error as NSError).code != NSURLErrorCancelled {
                state[id] = .failed(error.localizedDescription)
            }
            tasks[id] = nil
        }
    }
}
