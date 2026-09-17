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
    @State private var addingServer = false
    @State private var editingPort: SavedServer?

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
                    // Swiping right. The port is the one part of a
                    // pairing that changes for ordinary reasons — the
                    // server restarted somewhere else, or moved behind a
                    // different forward — and changing it used to mean
                    // re-pairing, which means finding a key again.
                    .swipeActions(edge: .leading) {
                        Button {
                            editingPort = server
                        } label: {
                            Label("Port", systemImage: "number")
                        }
                        .tint(.blue)
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
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    addingServer = true
                } label: {
                    Label("Add a server", systemImage: "plus")
                }
                .disabled(state.savedServers.count >= SavedServers.maxServers)
            }
        }
        .sheet(isPresented: $addingServer) {
            // The pairing screen, the same one shown when nothing is
            // paired at all. Reached from here it is an addition rather
            // than a replacement: PairingView goes through
            // SavedServers.remember, which keeps the others.
            //
            // No NavigationStack around it — PairingView brings its own,
            // and two would mean two navigation bars. `onDone` is what
            // tells it it is a sheet: it titles itself accordingly, puts
            // a Cancel in its own bar, and closes when a pairing lands.
            PairingView { addingServer = false }
        }
        .sheet(item: $editingPort) { server in
            PortEditor(server: server)
        }
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

/// Change the port on one saved machine.
///
/// Shows what every endpoint will become before committing, because a
/// pairing carries more than one — a LAN address and a tailnet name,
/// typically — and "change the port" has to mean all of them or the
/// failover list ends up half pointing at nothing.
private struct PortEditor: View {
    @Environment(AppState.self) private var state
    @Environment(\.dismiss) private var dismiss

    let server: SavedServer

    @State private var port = ""
    @State private var saving = false
    @State private var problem = ""

    private var parsed: Int? {
        guard let value = Int(port.trimmingCharacters(in: .whitespaces)) else {
            return nil
        }
        return (1...65535).contains(value) ? value : nil
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("Port", text: $port)
                        .keyboardType(.numberPad)
                } header: {
                    Text(server.displayName)
                } footer: {
                    Text(
                        "The machine is the same one — its key and its pinned "
                        + "identity are kept. Only where this phone looks for "
                        + "it changes."
                    )
                }

                if let wanted = parsed {
                    Section {
                        ForEach(server.connection.endpoints, id: \.self) { endpoint in
                            Text(SavedServers.replacingPort(in: endpoint, with: wanted))
                                .font(.caption.monospaced())
                                .foregroundStyle(.secondary)
                        }
                    } header: {
                        Text("Endpoints after this change")
                    }
                }

                if !problem.isEmpty {
                    Section {
                        Label(problem, systemImage: "exclamationmark.triangle")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    }
                }
            }
            .navigationTitle("Port")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") { Task { await save() } }
                        .disabled(parsed == nil || saving)
                }
            }
            .task {
                // Blank when the endpoints disagree, rather than showing
                // one of them: a field prefilled with 8080 for a record
                // whose tailnet endpoint is on 9000 would misstate what
                // Save is about to do.
                if let common = SavedServers.commonPort(of: server) {
                    port = "\(common)"
                }
            }
        }
    }

    @MainActor
    private func save() async {
        guard let wanted = parsed else { return }
        saving = true
        defer { saving = false }
        if await state.setPort(serverID: server.id, port: wanted) {
            dismiss()
        } else {
            problem = "That port could not be applied. It has to be between "
                + "1 and 65535, and the machine has to still be in this list."
        }
    }
}
