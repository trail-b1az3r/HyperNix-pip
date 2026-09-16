//  EditMessageSheet.swift
//  Changing what you asked, and re-asking it.
//
//  Why editing truncates
//  ---------------------
//  Everything below an edited message was written *in reply to the old
//  text*. Leaving it makes the conversation read as the model answering
//  a question nobody asked — and, worse, that same transcript is what
//  gets sent as context on the next turn, so the model would be told it
//  had said things it never said.
//
//  So the count is shown before anything happens. "Replacing this
//  removes 11 messages" is a decision; finding eleven messages gone
//  afterwards is a bug report.
//
//  Only your own messages
//  ----------------------
//  The server refuses to edit an assistant message and the menu does not
//  offer it. Rewriting what the model said turns the transcript into a
//  record of something that did not happen, which is a different feature
//  and not one anybody asked for.

import SwiftUI

struct EditMessageSheet: View {
    let message: ChatMessage
    @Environment(AppState.self) private var state
    @Environment(\.dismiss) private var dismiss

    @State private var draft: String = ""
    @State private var saving = false
    @FocusState private var focused: Bool

    private var wouldRemove: Int { state.editWouldRemove(message.messageID) }

    private var unchanged: Bool {
        draft.trimmingCharacters(in: .whitespacesAndNewlines)
            == message.content.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var empty: Bool {
        draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextEditor(text: $draft)
                        .frame(minHeight: 140)
                        .focused($focused)
                } header: {
                    Text("Your message")
                } footer: {
                    if empty {
                        Text("An empty message cannot be saved. Delete it instead.")
                            .foregroundStyle(.orange)
                    }
                }

                if wouldRemove > 0 {
                    Section {
                        Label(
                            "\(wouldRemove) later message\(wouldRemove == 1 ? "" : "s") "
                            + "will be removed",
                            systemImage: "exclamationmark.triangle"
                        )
                        .foregroundStyle(.orange)
                    } footer: {
                        Text(
                            "They were written in reply to what this message used "
                            + "to say. Keeping them would leave the model being told "
                            + "it had said things it never said."
                        )
                    }
                }
            }
            .navigationTitle("Edit")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button {
                        Task { await save() }
                    } label: {
                        if saving { ProgressView() } else { Text("Save") }
                    }
                    .disabled(saving || empty || unchanged)
                }
            }
            .task {
                draft = message.content
                focused = true
            }
        }
    }

    private func save() async {
        saving = true
        defer { saving = false }
        if await state.editMessage(message.messageID, to: draft) != nil {
            dismiss()
        }
    }
}
