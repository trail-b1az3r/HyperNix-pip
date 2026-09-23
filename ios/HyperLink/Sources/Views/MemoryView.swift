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
    @State private var search = ""
    /// The category being renamed, which also presents the prompt.
    @State private var renaming: String?
    @State private var renameDraft = ""
    @State private var organiseNote: String?
    @State private var organising = false

    private var filtered: [MemoryItem] {
        let needle = search.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !needle.isEmpty else { return state.memories }
        return state.memories.filter {
            $0.content.lowercased().contains(needle) || $0.category.lowercased().contains(needle)
        }
    }
    private var pinned: [MemoryItem] { filtered.filter(\.pinned) }
    /// Everything not pinned, by category: a list of forty facts is a
    /// pile, forty facts under Work, Tech and About you is an overview.
    private var grouped: [(name: String, items: [MemoryItem])] {
        let loose = filtered.filter { !$0.pinned }
        let byName = Dictionary(grouping: loose) { $0.category.isEmpty ? "General" : $0.category }
        return byName
            .map { (name: $0.key, items: $0.value) }
            .sorted { $0.items.count != $1.items.count ? $0.items.count > $1.items.count : $0.name < $1.name }
    }
    private var categoryNames: [String] {
        let known = ["About you", "Preferences", "Work", "Projects", "Tech", "Health", "Places", "Schedule", "General"]
        let used = Set(state.memories.map { $0.category.isEmpty ? "General" : $0.category })
        return Array(used.union(known)).sorted()
    }

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

            ForEach(grouped, id: \.name) { group in
                Section {
                    ForEach(group.items) { memory in row(memory) }
                } header: {
                    HStack {
                        Text(group.name)
                        Spacer()
                        Text("\(group.items.count)")
                    }
                    .contextMenu {
                        Button {
                            renameDraft = group.name
                            renaming = group.name
                        } label: {
                            Label("Rename category", systemImage: "pencil")
                        }
                    }
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
        .searchable(text: $search, prompt: "Search memories")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Menu {
                    Button {
                        Task { await organise() }
                    } label: {
                        Label("Organise into topics", systemImage: "folder.badge.gearshape")
                    }
                    .disabled(organising)
                } label: {
                    Label("Organise", systemImage: "square.grid.3x1.folder.badge.plus")
                }
            }
        }
        .alert("Rename category", isPresented: Binding(
            get: { renaming != nil }, set: { if !$0 { renaming = nil } }
        )) {
            TextField("Name", text: $renameDraft)
            Button("Rename") {
                guard let old = renaming else { return }
                renaming = nil
                Task { await state.renameMemoryCategory(from: old, to: renameDraft) }
            }
            Button("Cancel", role: .cancel) { renaming = nil }
        } message: {
            Text("Renaming to a category that exists merges the two.")
        }
        .alert("Memories organised", isPresented: Binding(
            get: { organiseNote != nil }, set: { if !$0 { organiseNote = nil } }
        )) {
            Button("OK", role: .cancel) { organiseNote = nil }
        } message: {
            Text(organiseNote ?? "")
        }
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
        .contextMenu {
            Menu {
                ForEach(categoryNames, id: \.self) { name in
                    Button(name) {
                        Task { await state.moveMemory(memory.memoryID, to: name) }
                    }
                    .disabled(name == memory.category)
                }
            } label: {
                Label("Move to", systemImage: "folder")
            }
            Button {
                Task { await state.updateMemory(memory.memoryID, pinned: !memory.pinned) }
            } label: {
                Label(memory.pinned ? "Unpin" : "Pin", systemImage: memory.pinned ? "pin.slash" : "pin")
            }
            Button(role: .destructive) {
                Task { await state.forget(memory.memoryID) }
            } label: {
                Label("Forget", systemImage: "trash")
            }
        }
    }

    private func organise() async {
        organising = true
        defer { organising = false }
        guard let result = await state.organiseMemories() else { return }
        if result.changes.isEmpty {
            organiseNote = "Everything is already under a topic."
        } else {
            let topics = Set(result.changes.map(\.to)).sorted().joined(separator: ", ")
            organiseNote = "Filed \(result.changes.count) memor\(result.changes.count == 1 ? "y" : "ies") under \(topics). "
                + "Categories you chose were left alone."
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
