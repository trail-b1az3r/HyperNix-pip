//  RenameChatSheet.swift
//  Giving a conversation a name.
//
//  The server has taken a title on `PATCH /hyperlink/sessions/{id}`
//  since HyperLink shipped. Nothing on the phone ever sent one, so every
//  chat kept whichever of its first sixty characters `autotitle` picked
//  out of the opening message — for ever, including the ones that opened
//  with "hey" or with a pasted stack trace.

import SwiftUI

struct RenameChatSheet: View {
    let sessionID: String
    let currentTitle: String

    @Environment(AppState.self) private var state
    @Environment(\.dismiss) private var dismiss
    @Environment(\.hyperLinkTheme) private var theme
    @State private var title: String = ""
    @State private var saving = false
    @FocusState private var focused: Bool

    /// What the server will accept. Matches `ChatSessionStore.update`,
    /// which trims and falls back to the old title on an empty string —
    /// so an empty box here would silently do nothing rather than fail,
    /// which is the worse of the two.
    private var trimmed: String {
        title.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var canSave: Bool {
        !trimmed.isEmpty && trimmed != currentTitle && !saving
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("Name", text: $title, axis: .vertical)
                        .lineLimit(1...3)
                        .focused($focused)
                        .submitLabel(.done)
                        .onSubmit { if canSave { save() } }
                } footer: {
                    Text("Stored on your PC, so every device sees it.")
                }

                if !currentTitle.isEmpty {
                    Section {
                        Button("Reset to the automatic name") {
                            // `ChatSessionStore.autotitle` re-titles from
                            // the first user message, but only for a
                            // session still called "" or "New chat" — so
                            // sending that exact string back is what
                            // re-arms it. It fires on the next turn
                            // rather than now, which is why the footer
                            // below says so instead of leaving somebody
                            // waiting for a name that has not been asked
                            // for yet.
                            title = "New chat"
                        }
                        .foregroundStyle(theme.accent)
                    } footer: {
                        Text("Takes effect with the next message.")
                    }
                }
            }
            .navigationTitle("Rename chat")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") { save() }
                        .disabled(!canSave)
                }
            }
        }
        .presentationDetents([.medium])
        .onAppear {
            title = currentTitle
            focused = true
        }
    }

    private func save() {
        saving = true
        Task {
            // The rename is optimistic in AppState, so this closes
            // immediately either way; a failure puts the old name back
            // and shows the error on the chat behind this sheet.
            await state.rename(sessionID, to: trimmed)
            dismiss()
        }
    }
}
