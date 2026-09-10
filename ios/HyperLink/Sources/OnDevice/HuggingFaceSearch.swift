//  HuggingFaceSearch.swift
//  Finding GGUF models from the phone, with the user's own token.
//
//  This talks to huggingface.co directly rather than through a paired
//  machine, because the whole point of on-device mode is that there may
//  not be a machine. The token is the user's, lives in the Keychain,
//  and is sent to exactly one host.

import Foundation

/// One repository from a search.
struct HFModelSummary: Decodable, Sendable, Identifiable {
    let id: String
    let downloads: Int?
    let likes: Int?
    let gated: HFGated?
    let tags: [String]?
    let lastModified: String?

    var owner: String { id.split(separator: "/").first.map(String.init) ?? "" }
    var name: String { id.split(separator: "/").last.map(String.init) ?? id }

    /// A gated repository needs the user to have accepted its licence
    /// on the web first. Downloading returns 403 until they have, and
    /// saying so up front is the difference between a clear next step
    /// and an unexplained failure forty minutes in.
    var requiresLicenceAcceptance: Bool { gated?.isGated ?? false }
}

/// `gated` comes back as `false`, `"auto"` or `"manual"` — a bool in
/// one case and a string in the others. Decoding it as either type
/// alone fails on the other, and a failed decode loses the whole search
/// result over a field that is advisory.
enum HFGated: Decodable, Sendable {
    case no
    case auto
    case manual

    var isGated: Bool { self != .no }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let flag = try? container.decode(Bool.self) {
            self = flag ? .manual : .no
            return
        }
        switch try? container.decode(String.self) {
        case "auto": self = .auto
        case "manual": self = .manual
        default: self = .no
        }
    }
}

struct HFFileEntry: Decodable, Sendable {
    let rfilename: String
    let size: Int?
}

struct HFModelDetail: Decodable, Sendable {
    let id: String
    let siblings: [HFFileEntry]?
    let gated: HFGated?

    /// Just the GGUF files, largest first.
    ///
    /// Multi-part files (`-00001-of-00003.gguf`) are excluded: they need
    /// to be joined before use, and offering one as a download produces
    /// a file that cannot load. Naming that as a limit is better than
    /// silently listing something broken.
    var ggufCandidates: [GGUFCandidate] {
        (siblings ?? [])
            .filter { $0.rfilename.lowercased().hasSuffix(".gguf") }
            .filter { !$0.rfilename.contains(of: "-of-") }
            .map {
                GGUFCandidate(
                    repoID: id,
                    filename: $0.rfilename,
                    sizeBytes: $0.size ?? 0,
                    quant: ModelFit.quantFromFilename($0.rfilename)
                )
            }
            .sorted { $0.sizeBytes > $1.sizeBytes }
    }
}

/// One downloadable GGUF.
struct GGUFCandidate: Sendable, Identifiable, Equatable {
    let repoID: String
    let filename: String
    var sizeBytes: Int = 0
    var quant: String = ""

    var id: String { "\(repoID)/\(filename)" }

    var downloadURL: URL? {
        URL(string: "https://huggingface.co/\(repoID)/resolve/main/\(filename)")
    }

    var resolvedQuant: String {
        quant.isEmpty ? ModelFit.quantFromFilename(filename) : quant
    }
}

enum HFSearchError: LocalizedError {
    case badResponse(Int)
    case unauthorised
    case gated(String)
    case network(String)

    var errorDescription: String? {
        switch self {
        case .unauthorised:
            "Hugging Face rejected the token. Check it in Settings — a token "
            + "needs at least read access."
        case .gated(let repo):
            "\(repo) is gated. Accept its licence on huggingface.co with the "
            + "same account this token belongs to, then try again."
        case .badResponse(let code):
            "Hugging Face returned HTTP \(code)."
        case .network(let detail):
            "Could not reach Hugging Face: \(detail)"
        }
    }
}

/// Searches huggingface.co for GGUF models.
actor HuggingFaceSearch {
    private let session: URLSession
    private var token: String?

    init(session: URLSession = .shared) {
        self.session = session
    }

    /// The token is held here and sent to huggingface.co and nowhere
    /// else. It is never logged, never included in an error message,
    /// and never written anywhere but the Keychain.
    func setToken(_ token: String?) {
        self.token = token?.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var hasToken: Bool { !(token ?? "").isEmpty }

    /// Search for repositories carrying GGUF files.
    ///
    /// `filter=gguf` rather than a text match on "gguf": the tag is
    /// what Hugging Face indexes, and a text search returns every
    /// repository whose README mentions the format.
    func search(_ query: String, limit: Int = 30) async throws -> [HFModelSummary] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        var components = URLComponents(string: "https://huggingface.co/api/models")!
        components.queryItems = [
            URLQueryItem(name: "filter", value: "gguf"),
            URLQueryItem(name: "limit", value: String(max(1, min(limit, 100)))),
            URLQueryItem(name: "sort", value: trimmed.isEmpty ? "downloads" : "likes"),
            URLQueryItem(name: "direction", value: "-1"),
            URLQueryItem(name: "full", value: "false"),
        ]
        if !trimmed.isEmpty {
            components.queryItems?.append(URLQueryItem(name: "search", value: trimmed))
        }
        let data = try await get(components.url!)
        return try JSONDecoder().decode([HFModelSummary].self, from: data)
    }

    /// The files in one repository, so a candidate can be sized.
    func detail(repoID: String) async throws -> HFModelDetail {
        guard let url = URL(string: "https://huggingface.co/api/models/\(repoID)") else {
            throw HFSearchError.network("bad repository id")
        }
        return try JSONDecoder().decode(HFModelDetail.self, from: try await get(url))
    }

    private func get(_ url: URL) async throws -> Data {
        var request = URLRequest(url: url)
        request.timeoutInterval = 30
        if let token, !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        do {
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else { return data }
            switch http.statusCode {
            case 200..<300: return data
            case 401, 403:
                // 403 on a model endpoint is nearly always a gated
                // repository rather than a bad token, and the two need
                // different actions from the user.
                if http.statusCode == 403, url.path.contains("/api/models/") {
                    throw HFSearchError.gated(url.lastPathComponent)
                }
                throw HFSearchError.unauthorised
            default:
                throw HFSearchError.badResponse(http.statusCode)
            }
        } catch let error as HFSearchError {
            throw error
        } catch {
            throw HFSearchError.network(error.localizedDescription)
        }
    }
}

private extension String {
    /// `contains(_:)` on a String needs a Character or a StringProtocol
    /// depending on the overload, and the ambiguity has bitten here
    /// before. This is unambiguous.
    func contains(of needle: String) -> Bool {
        range(of: needle) != nil
    }
}
