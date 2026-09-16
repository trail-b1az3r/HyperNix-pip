//  ServersView.swift
//  Every machine this phone is paired with, and one tap to switch.
//
//  The app stored exactly one pairing. Pairing with a laptop overwrote
//  the desktop, and getting back to the desktop meant pairing again —
//  finding a key or being on the right network at the right time. A
//  desktop and a laptop, or a machine at home and one at work, is not an
//  exotic setup, and re-pairing is not a way to switch between them.
//
//  Up to `SavedServers.maxServers` of them now, one selected. See
//  `SavedServers.swift` for where they live and why forgetting one
//  leaves the others signed in.

import SwiftUI

struct ServersView: View {
    @Environment(AppState.self) private var state
    @State private var switching: String?
    @State private var confirmingForget: SavedServer?

    var body: some View {
        List {
            Section {
                ForEach(state.savedServers.sorted(by: { $0.lastUsed > $1.lastUsed })) { server in
                    Button {
                        guard server.id != state.currentServerID else { return }
                        switching = server.id
                        Task {
                            await state.switchTo(serverID: server.id)
                            switching = nil
                        }
                    } label: {
                        ServerRow(
                            server: server,
                            isCurrent: server.id == state.currentServerID,
                            isSwitching: switching == server.id
                        )
                    }
                    .swipeActions(edge: .trailing) {
                        Button(role: .destructive) {
                            confirmingForget = server
                        } label: {
                            Label("Forget", systemImage: "trash")
                        }
                    }
                }
            } header: {
                Text("Paired machines")
            } footer: {
                Text(footerText)
            }

            if let uptime = state.uptime {
                Section("This server") {
                    LabeledContent("Up for", value: uptime.serverDescription)
                    LabeledContent("Machine up for", value: uptime.machineDescription)
                }
            }
        }
        .navigationTitle("Servers")
        .refreshable { await state.refreshAll() }
        .confirmationDialog(
            confirmingForget.map { "Forget \($0.displayName)?" } ?? "",
            isPresented: Binding(
                get: { confirmingForget != nil },
                set: { if !$0 { confirmingForget = nil } }
            ),
            titleVisibility: .visible
        ) {
            Button("Forget", role: .destructive) {
                guard let server = confirmingForget else { return }
                confirmingForget = nil
                Task { await state.forget(serverID: server.id) }
            }
            Button("Cancel", role: .cancel) { confirmingForget = nil }
        } message: {
            Text(
                "This phone stops being paired with that machine. Your "
                + "conversations stay on it, and the other servers here are "
                + "unaffected."
            )
        }
    }

    private var footerText: String {
        let used = state.savedServers.count
        let limit = SavedServers.maxServers
        if used >= limit {
            return "\(used) of \(limit). Pairing with another machine forgets "
                + "whichever of these you have not used for longest."
        }
        return "\(used) of \(limit). Tap one to switch — your conversations live "
            + "on each machine, so switching changes what you see."
    }
}

private struct ServerRow: View {
    let server: SavedServer
    let isCurrent: Bool
    let isSwitching: Bool

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(server.displayName)
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                HStack(spacing: 6) {
                    if let first = server.connection.endpoints.first {
                        Text(SavedServer.hostOf(first)).lineLimit(1)
                    }
                    if server.keyless {
                        Text("·")
                        Label("keyless", systemImage: "lock.open").labelStyle(.titleOnly)
                    }
                    if !server.connection.t1Version.isEmpty {
                        Text("·")
                        Text("v\(server.connection.t1Version)")
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            Spacer()
            if isSwitching {
                ProgressView()
            } else if isCurrent {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.tint)
                    .accessibilityLabel("current server")
            }
        }
    }
}
