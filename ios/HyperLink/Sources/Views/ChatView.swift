//  ChatView.swift
//  One conversation: bubbles, attachments, and the live stream.

import PhotosUI
import SwiftUI
// For `UIImage`, which the camera hands back. SwiftUI does not re-export
// UIKit, so this is not redundant however much it looks it.
import UIKit
import UniformTypeIdentifiers

struct ChatView: View {
    let sessionID: String

    @Environment(AppState.self) private var state
    @State private var draft = ""
    @State private var pendingAttachments: [Attachment] = []
    @State private var photoItem: PhotosPickerItem?
    @State private var showingFileImporter = false
    @State private var showingCodeImporter = false
    @State private var showingPhotoPicker = false
    @State private var showingCamera = false
    @State private var showingModelPicker = false
    @State private var showingRename = false
    @State private var isUploading = false

    private var session: ChatSession? {
        state.sessions.first { $0.sessionID == sessionID }
    }

    var body: some View {
        VStack(spacing: 0) {
            messageList
            if let error = state.lastError {
                errorBanner(error)
            }
            composer
        }
        .navigationTitle(session?.title ?? "Chat")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Menu {
                    Button {
                        showingRename = true
                    } label: {
                        Label("Rename", systemImage: "pencil")
                    }
                    Button {
                        showingModelPicker = true
                    } label: {
                        Label("Model", systemImage: "cpu")
                    }
                } label: {
                    Label("More", systemImage: "ellipsis.circle")
                }
            }
        }
        .sheet(isPresented: $showingModelPicker) {
            ModelPickerSheet(sessionID: sessionID, currentModel: session?.modelID ?? "")
        }
        .sheet(isPresented: $showingRename) {
            RenameChatSheet(
                sessionID: sessionID, currentTitle: session?.title ?? ""
            )
        }
        .photosPicker(
            isPresented: $showingPhotoPicker,
            selection: $photoItem,
            // Videos as well: the server takes any attachment, and
            // "photo or video" is what the menu row promises.
            matching: .any(of: [.images, .videos])
        )
        .fileImporter(
            isPresented: $showingCodeImporter,
            // Narrowed on purpose. A "code or text" picker that shows
            // every file on the phone is the "File" row again with a
            // different name.
            allowedContentTypes: [.sourceCode, .plainText, .json, .yaml, .xml, .propertyList],
            allowsMultipleSelection: false
        ) { result in
            Task { await attachFile(result) }
        }
        .fullScreenCover(isPresented: $showingCamera) {
            CameraCapture { image in
                Task { await attachCameraImage(image) }
            }
        }
        .task(id: sessionID) { await state.open(sessionID) }
        .onChange(of: photoItem) { _, item in
            guard let item else { return }
            Task { await attachPhoto(item) }
        }
        .fileImporter(
            isPresented: $showingFileImporter,
            allowedContentTypes: [.item],
            allowsMultipleSelection: false
        ) { result in
            Task { await attachFile(result) }
        }
    }

    // MARK: - Messages

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    // The system prompt is part of the conversation on
                    // the server but is not something to read as a
                    // message, so it is filtered out here rather than
                    // omitted server-side (where the model needs it).
                    ForEach(state.messages.filter { !$0.isSystem }) { message in
                        MessageBubble(message: message)
                            .id(message.messageID)
                    }
                    if !state.streamingText.isEmpty {
                        MessageBubble(
                            message: .local(
                                role: "assistant",
                                content: state.streamingText,
                                sessionID: sessionID
                            ),
                            isStreaming: true
                        )
                        .id("streaming")
                    } else if state.isSending {
                        HStack(spacing: 8) {
                            ProgressView().controlSize(.small)
                            Text("Thinking…").font(.caption).foregroundStyle(.secondary)
                        }
                        .padding(.horizontal)
                        .id("streaming")
                    }
                }
                .padding()
            }
            .onChange(of: state.messages.count) { _, _ in scrollToEnd(proxy) }
            .onChange(of: state.streamingText) { _, _ in scrollToEnd(proxy) }
        }
    }

    private func scrollToEnd(_ proxy: ScrollViewProxy) {
        // Anchor on the streaming bubble while one exists, otherwise the
        // last real message. Without the explicit anchor the view sticks
        // a few lines short of the bottom as text grows.
        withAnimation(.easeOut(duration: 0.15)) {
            if !state.streamingText.isEmpty || state.isSending {
                proxy.scrollTo("streaming", anchor: .bottom)
            } else if let last = state.messages.last {
                proxy.scrollTo(last.messageID, anchor: .bottom)
            }
        }
    }

    private func errorBanner(_ error: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
            Text(error).font(.caption)
            Spacer()
            Button("Dismiss") { state.lastError = nil }.font(.caption)
        }
        .padding(10)
        .background(Color.orange.opacity(0.15))
        .foregroundStyle(.primary)
    }

    // MARK: - Composer

    private var composer: some View {
        VStack(spacing: 8) {
            if !pendingAttachments.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(pendingAttachments) { attachment in
                            AttachmentChip(attachment: attachment) {
                                pendingAttachments.removeAll { $0.fileID == attachment.fileID }
                            }
                        }
                    }
                    .padding(.horizontal)
                }
            }

            HStack(alignment: .bottom, spacing: 8) {
                // One menu rather than a one-item menu beside a bare
                // photo button. The old shape offered two of the four
                // things people attach and gave no hint that the other
                // two were possible -- a document from Files, and the
                // camera, were both reachable by the server and by
                // nothing on screen.
                Menu {
                    Button {
                        showingPhotoPicker = true
                    } label: {
                        Label("Photo or video", systemImage: "photo.on.rectangle")
                    }
                    Button {
                        showingCamera = true
                    } label: {
                        Label("Take a photo", systemImage: "camera")
                    }
                    Divider()
                    Button {
                        showingFileImporter = true
                    } label: {
                        Label("File", systemImage: "doc")
                    }
                    Button {
                        showingCodeImporter = true
                    } label: {
                        Label("Code or text", systemImage: "chevron.left.forwardslash.chevron.right")
                    }
                } label: {
                    Image(systemName: "paperclip").font(.title3)
                        .accessibilityLabel("Attach")
                }
                .disabled(isUploading || state.isSending)

                TextField("Message", text: $draft, axis: .vertical)
                    .lineLimit(1...6)
                    .textFieldStyle(.roundedBorder)

                Button {
                    send()
                } label: {
                    Image(systemName: state.isSending ? "stop.circle.fill" : "arrow.up.circle.fill")
                        .font(.title)
                }
                .disabled(!state.isSending && draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && pendingAttachments.isEmpty)
            }
            .padding(.horizontal)
            .padding(.bottom, 8)

            if isUploading {
                ProgressView().controlSize(.small).padding(.bottom, 4)
            }
        }
        .background(.bar)
    }

    private func send() {
        if state.isSending {
            state.cancelStreaming()
            return
        }
        let text = draft
        let ids = pendingAttachments.map(\.fileID)
        draft = ""
        pendingAttachments = []
        let model = (session?.modelID).flatMap { $0.isEmpty ? nil : $0 }
        state.send(text: text, attachmentIDs: ids, modelID: model)
    }

    // MARK: - Attachments

    private func attachPhoto(_ item: PhotosPickerItem) async {
        isUploading = true
        defer { isUploading = false; photoItem = nil }
        guard let data = try? await item.loadTransferable(type: Data.self) else {
            state.lastError = "That photo could not be read."
            return
        }
        // The picker does not reliably give a filename; the extension is
        // cosmetic here anyway because the server sniffs the real type
        // from the bytes.
        let name = item.itemIdentifier.map { "photo-\($0.prefix(8)).jpg" } ?? "photo.jpg"
        if let attachment = await state.upload(data: data, filename: name, contentType: "image/jpeg") {
            pendingAttachments.append(attachment)
        }
    }

    private func attachCameraImage(_ image: UIImage) async {
        isUploading = true
        defer { isUploading = false }
        // 0.85 rather than 1.0: a full-quality capture from a modern
        // phone is several megabytes, and this is going over whatever
        // link the phone has to the PC -- often a Tailscale relay. The
        // difference is invisible to a model reading the picture.
        guard let data = image.jpegData(compressionQuality: 0.85) else {
            state.lastError = "That photo could not be encoded."
            return
        }
        let name = "camera-\(Int(Date().timeIntervalSince1970)).jpg"
        if let attachment = await state.upload(
            data: data, filename: name, contentType: "image/jpeg"
        ) {
            pendingAttachments.append(attachment)
        }
    }

    private func attachFile(_ result: Result<[URL], Error>) async {
        guard case let .success(urls) = result, let url = urls.first else { return }
        isUploading = true
        defer { isUploading = false }

        // A document-picker URL is security-scoped: without the
        // start/stop pair the read fails for anything outside the app's
        // own container, which is every file worth attaching.
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }

        guard let data = try? Data(contentsOf: url) else {
            state.lastError = "That file could not be read."
            return
        }
        let type = UTType(filenameExtension: url.pathExtension)?.preferredMIMEType ?? "application/octet-stream"
        if let attachment = await state.upload(data: data, filename: url.lastPathComponent, contentType: type) {
            pendingAttachments.append(attachment)
        }
    }
}

struct AttachmentChip: View {
    let attachment: Attachment
    let onRemove: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: attachment.isImage ? "photo" : (attachment.isText ? "doc.text" : "doc"))
            Text(attachment.filename).lineLimit(1).font(.caption)
            Button(action: onRemove) {
                Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(Color.secondary.opacity(0.15), in: Capsule())
    }
}
