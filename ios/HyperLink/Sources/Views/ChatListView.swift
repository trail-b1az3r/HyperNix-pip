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

    private var listBody: some View {
        List {
            if state.sessions.isEmpty && !state.isLoadingSessions {
                ContentUnavailableView {
                    Label("No conversations", systemImage: "bubble.left.and.bubble.right")
                } description: {
                    Text("Start one — it is stored on your PC, so you can pick it up on any device.")
                } actions: {
                    Button("New chat") { Task { await startChat() } }
                }
            }
            ForEach(state.sessions) { session in
                NavigationLink(value: session.sessionID) {
                    VStack(alignment: .leading, spacing: 3) {
                        Text(session.title)
                            .font(.body)
                            .lineLimit(1)
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
                // And by long-press, because a swipe is not discoverable
                // and renaming is the one thing here somebody goes
                // looking for.
                .contextMenu {
                    Button {
                        renaming = session
                    } label: {
                        Label("Rename", systemImage: "pencil")
                    }
                    Button(role: .destructive) {
                        Task { await state.delete(session.sessionID) }
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                }
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
