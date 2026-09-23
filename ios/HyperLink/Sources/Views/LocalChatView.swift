//  LocalChatView.swift
//  Talking to the model loaded on this iPhone.
//
//  Kept on the phone, and deliberately not mixed into the server's
//  conversations: a thread that lives on the PC and a reply written on
//  the phone would each be half a record of the other. This one lives
//  in memory for as long as the screen does.

import SwiftUI

struct LocalChatView: View {
    @EnvironmentObject private var hub: OnDeviceHub
    @State private var turns: [(role: String, text: String)] = []
    @State private var draft = ""
    @State private var streaming = ""
    @State private var generating = false
    @State private var failure: String?

    var body: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(Array(turns.enumerated()), id: \.offset) { index, turn in
                            bubble(turn.text, mine: turn.role == "user").id(index)
                        }
                        if generating {
                            bubble(streaming.isEmpty ? "…" : streaming, mine: false).id("live")
                        }
                        if let failure {
                            Text(failure).font(.caption).foregroundStyle(.red)
                        }
                    }
                    .padding()
                }
                .onChange(of: streaming) { _, _ in proxy.scrollTo("live", anchor: .bottom) }
            }
            HStack(alignment: .bottom) {
                TextField("Message", text: $draft, axis: .vertical)
                    .lineLimit(1...5)
                    .textFieldStyle(.roundedBorder)
                Button {
                    generating ? stop() : send()
                } label: {
                    Image(systemName: generating ? "stop.circle.fill" : "arrow.up.circle.fill")
                        .font(.title)
                }
                .disabled(!generating && draft.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            .padding()
            .background(.bar)
        }
        .navigationTitle(hub.inference.loaded?.displayName ?? "On this iPhone")
        .navigationBarTitleDisplayMode(.inline)
    }

    private func bubble(_ text: String, mine: Bool) -> some View {
        HStack {
            if mine { Spacer(minLength: 40) }
            Text(text)
                .padding(10)
                .background(mine ? Color.accentColor.opacity(0.18) : Color.secondary.opacity(0.12))
                .clipShape(RoundedRectangle(cornerRadius: 14))
                .textSelection(.enabled)
            if !mine { Spacer(minLength: 40) }
        }
    }

    /// The whole conversation as one prompt. The runner takes a single
    /// prompt and applies the model's own chat template around it, so
    /// earlier turns are carried as text rather than lost after each
    /// reply — a model that forgets the previous message every turn is
    /// not a conversation.
    private func prompt(for text: String) -> String {
        let history = turns.suffix(12).map {
            ($0.role == "user" ? "User: " : "Assistant: ") + $0.text
        }
        return (history + ["User: " + text, "Assistant:"]).joined(separator: "\n")
    }

    private func send() {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, hub.inference.loaded != nil else {
            failure = "No model is loaded. Go back and load one."
            return
        }
        let full = prompt(for: text)
        turns.append((role: "user", text: text))
        draft = ""
        streaming = ""
        failure = nil
        generating = true
        Task {
            do {
                for try await token in hub.inference.generate(prompt: full) {
                    streaming += token.text
                }
            } catch {
                failure = error.localizedDescription
            }
            if !streaming.isEmpty {
                turns.append((role: "assistant", text: streaming))
            }
            streaming = ""
            generating = false
        }
    }

    private func stop() {
        Task { await hub.inference.cancel() }
    }
}
