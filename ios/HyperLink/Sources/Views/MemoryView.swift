//  MemoryView.swift
//  What the assistant knows about you, and how to change it.
//
//  Memories are facts carried between conversations — some you wrote,
//  some the model noticed. Both are shown, and both can be edited and
//  deleted, and that is the whole argument for this screen existing.
//
//  A model that remembers things about you and gives you no way to see
//  them is a model you cannot correct. "Why does it keep thinking I use
//  Windows" has to have an answer, and the answer has to be a row you
//  can swipe away. Automatic memories are marked as such, because the
//  first question about a fact you did not write is where it came from.

import SwiftUI

struct MemoryView: View {
    @Environment(AppState.self) private var state
    @State private var adding = false
    @State private var draft = ""
    @State private var editing: MemoryItem?

    private var pinned: [MemoryItem] { state.memories.filter(\.pinned) }
    private var loose: [MemoryItem] { state.memories.filter { !$0.pinned } }

    var body: some View {
        List {
            if state.memories.isEmpty {
                Section {
                    ContentUnavailableView {
                        Label("Nothing remembered yet", systemImage: "brain")
                    } description: {
                        Text(
                            state.settings.preferences.autoMemory
                            ? "The assistant will note things worth carrying between conversations. You can add your own below."
                            : "Automatic memory is off, so nothing is being noted. You can still add your own."
                        )
                    }
                }
            }

            if !pinned.isEmpty {
                Section {
                    ForEach(pinned) { memory in row(memory) }
                } header: {
                    Text("Pinned")
                } footer: {
                    Text("Pinned memories are never evicted to make room for newer ones.")
                }
            }

            if !loose.isEmpty {
                Section(pinned.isEmpty ? "Remembered" : "Everything else") {
                    ForEach(loose) { memory in row(memory) }
                }
            }

            Section {
                Button {
                    draft = ""
                    adding = true
                } label: {
                    Label("Remember something", systemImage: "plus.circle")
                }
            } footer: {
                Text("Anything here is available to every conversation, on every device signed in to this server.")
            }
        }
        .navigationTitle("Memory")
        .refreshable { await state.refreshMemories() }
        .task { await state.refreshMemories() }
        .sheet(isPresented: $adding) {
            MemoryEditor(title: "Remember", text: $draft) { text in
                await state.remember(text)
            }
        }
        .sheet(item: $editing) { memory in
            MemoryEditor(
                title: "Edit", text: .constant(memory.content), existing: memory
            ) { text in
                await state.updateMemory(memory.memoryID, content: text)
            }
        }
    }

    private func row(_ memory: MemoryItem) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(memory.content)
                .font(.callout)
            HStack(spacing: 6) {
                if memory.isAutomatic {
                    // The first question about a fact you did not write
                    // is where it came from.
                    Label("noticed by the assistant", systemImage: "sparkles")
                        .labelStyle(.titleAndIcon)
                }
                if !memory.category.isEmpty {
                    Text("·")
                    Text(memory.category)
                }
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
        .contentShape(Rectangle())
        .onTapGesture { editing = memory }
        .swipeActions(edge: .trailing) {
            Button(role: .destructive) {
                Task { await state.forget(memory.memoryID) }
            } label: {
                Label("Forget", systemImage: "trash")
            }
        }
        .swipeActions(edge: .leading) {
            Button {
                Task {
                    await state.updateMemory(memory.memoryID, pinned: !memory.pinned)
                }
            } label: {
                Label(memory.pinned ? "Unpin" : "Pin",
                      systemImage: memory.pinned ? "pin.slash" : "pin")
            }
            .tint(.orange)
        }
    }
}

private struct MemoryEditor: View {
    let title: String
    @Binding var text: String
    var existing: MemoryItem?
    let save: (String) async -> Bool

    @Environment(\.dismiss) private var dismiss
    @State private var draft = ""
    @State private var saving = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextEditor(text: $draft)
                        .frame(minHeight: 120)
                } footer: {
                    Text("One fact, not a conversation — something short enough to be worth carrying into every chat.")
                }
            }
            .navigationTitle(title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button {
                        Task {
                            saving = true
                            let ok = await save(draft)
                            saving = false
                            if ok { dismiss() }
                        }
                    } label: {
                        if saving { ProgressView() } else { Text("Save") }
                    }
                    .disabled(
                        saving
                        || draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    )
                }
            }
            .task { draft = existing?.content ?? text }
        }
    }
}
