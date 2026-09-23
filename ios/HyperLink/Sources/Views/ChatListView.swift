//  ChatListView.swift
//  The conversations, newest first. They live on the PC, so this is the
//  same list the desktop client sees.

import SwiftUI

struct ChatListView: View {
    @Environment(AppState.self) private var state
    /// The stack is owned here rather than by the tab, because "New
    /// chat" has to push the conversation it just created — and a
    /// NavigationStack one level up has no path this view can append to.
    @State private var path: [String] = []
    /// The chat being renamed, which is also what presents the sheet.
    /// An item-based sheet rather than a bool plus a separate "which
    /// one" -- those two can disagree, and the bug is a sheet that
    /// renames the wrong conversation.
    @State private var renaming: ChatSession?
    @Environment(\.hyperLinkTheme) private var theme

    var body: some View {
        NavigationStack(path: $path) {
            listBody
        }
    }

    @Environment(\.scenePhase) private var scenePhase

    private var server: String { state.currentServerID }
    private var hiddenIDs: Set<String> { state.privateChats.hiddenIDs(server: server) }
    private var visible: [ChatSession] { state.sessions.filter { !hiddenIDs.contains($0.sessionID) } }
    private var hiddenSessions: [ChatSession] { state.sessions.filter { hiddenIDs.contains($0.sessionID) } }

    private var listBody: some View {
        List {
            if visible.isEmpty && hiddenSessions.isEmpty && !state.isLoadingSessions {
                ContentUnavailableView {
                    Label("No conversations", systemImage: "bubble.left.and.bubble.right")
                } description: {
                    Text("Start one — it is stored on your PC, so you can pick it up on any device.")
                } actions: {
                    Button("New chat") { Task { await startChat() } }
                }
            }
            ForEach(visible) { session in
                row(session, hidden: false)
            }
            if !hiddenSessions.isEmpty {
                privateSection
            }
        }
        .navigationTitle("Chats")
        .navigationDestination(for: String.self) { sessionID in
            ChatView(sessionID: sessionID)
        }
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    Task { await startChat() }
                } label: {
                    Label("New chat", systemImage: "square.and.pencil")
                }
            }
        }
        .sheet(item: $renaming) { session in
            RenameChatSheet(
                sessionID: session.sessionID, currentTitle: session.title
            )
        }
        .refreshable { await state.refreshSessions() }
        .overlay {
            if state.isLoadingSessions && state.sessions.isEmpty {
                ProgressView()
            }
        }
        // Private stays private only while the phone is in the person's
        // hand: leaving the app locks it again.
        .onChange(of: scenePhase) { _, phase in
            if phase != .active { state.privateChats.lock() }
        }
        .onChange(of: state.sessions) { _, sessions in
            guard !state.isLoadingSessions, !sessions.isEmpty else { return }
            state.privateChats.prune(existing: Set(sessions.map(\.sessionID)), server: server)
        }
    }

    @ViewBuilder
    private var privateSection: some View {
        Section {
            if state.privateChats.isUnlocked {
                ForEach(hiddenSessions) { session in
                    row(session, hidden: true)
                }
            } else {
                Button {
                    Task { await state.privateChats.unlock() }
                } label: {
                    Label(
                        "\(hiddenSessions.count) private chat\(hiddenSessions.count == 1 ? "" : "s")",
                        systemImage: "lock.fill"
                    )
                }
                if let failure = state.privateChats.lastFailure {
                    Text(failure).font(.caption).foregroundStyle(.secondary)
                }
            }
        } header: {
            HStack {
                Text("Private")
                Spacer()
                if state.privateChats.isUnlocked {
                    Button("Lock") { state.privateChats.lock() }
                        .font(.caption)
                }
            }
        } footer: {
            Text("Hidden on this iPhone only, behind Face ID. The chat is still on your PC.")
        }
    }

    private func row(_ session: ChatSession, hidden: Bool) -> some View {
        NavigationLink(value: session.sessionID) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    if hidden {
                        Image(systemName: "lock.open").font(.caption).foregroundStyle(.secondary)
                    }
                    Text(session.title)
                        .font(.body)
                        .lineLimit(1)
                }
                HStack(spacing: 6) {
                    if !session.modelID.isEmpty {
                        Text(shortModelName(session.modelID))
                            .lineLimit(1)
                    }
                    Text("·")
                    Text("\(session.messageCount) message\(session.messageCount == 1 ? "" : "s")")
                    Text("·")
                    Text(relativeTime(session.updatedAt))
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
        .swipeActions(edge: .trailing) {
            Button(role: .destructive) {
                Task { await state.delete(session.sessionID) }
            } label: {
                Label("Delete", systemImage: "trash")
            }
            Button {
                renaming = session
            } label: {
                Label("Rename", systemImage: "pencil")
            }
            .tint(theme.accent)
        }
        .swipeActions(edge: .leading) {
            privacyButton(session, hidden: hidden)
                .tint(.indigo)
        }
        // And by long-press, because a swipe is not discoverable
        // and renaming is the one thing here somebody goes
        // looking for.
        .contextMenu {
            Button {
                renaming = session
            } label: {
                Label("Rename", systemImage: "pencil")
            }
            Button {
                Task { await state.retitle(session.sessionID) }
            } label: {
                Label("Rename with AI", systemImage: "sparkles")
            }
            privacyButton(session, hidden: hidden)
            Button(role: .destructive) {
                Task { await state.delete(session.sessionID) }
            } label: {
                Label("Delete", systemImage: "trash")
            }
        }
    }

    private func privacyButton(_ session: ChatSession, hidden: Bool) -> some View {
        Button {
            if hidden {
                state.privateChats.unhide(session.sessionID, server: server)
            } else {
                state.privateChats.hide(session.sessionID, server: server)
            }
        } label: {
            Label(hidden ? "Unhide" : "Hide with Face ID", systemImage: hidden ? "eye" : "eye.slash")
        }
    }

    private func startChat() async {
        guard let session = await state.newSession() else { return }
        path.append(session.sessionID)
    }
}

/// `lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf` is unreadable
/// in a list row; the last path component is the part that identifies it.
func shortModelName(_ modelID: String) -> String {
    modelID.split(separator: "/").last.map(String.init) ?? modelID
}

func relativeTime(_ timestamp: Double) -> String {
    let formatter = RelativeDateTimeFormatter()
    formatter.unitsStyle = .abbreviated
    return formatter.localizedString(for: Date(timeIntervalSince1970: timestamp), relativeTo: Date())
}
