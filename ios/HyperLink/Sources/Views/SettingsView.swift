//  SettingsView.swift
//  Which PC this phone is talking to, and how to stop.
//
//  "Which PC" is now a choice rather than a fact: up to
//  `SavedServers.maxServers` of them, switched from `ServersView`.

import SwiftUI

struct SettingsView: View {
    @Environment(AppState.self) private var state
    @Environment(ThemeStore.self) private var themes
    @State private var confirmingUnpair = false
    /// Mirrors `AdminCredentialStore.persistsAcrossLaunches`. Held in
    /// view state only as a switch position — the credential itself
    /// never reaches a view.
    @State private var keepAdminCredentials = AdminCredentialStore.persistsAcrossLaunches

    /// What the runner row says on its right-hand side. Never blank: a
    /// row with nothing next to it reads as broken rather than idle.
    /// Which backend a message would reach, in words.
    private var answeringWith: String {
        let active = state.backends.active
        if active.isEmpty { return "nothing yet" }
        return state.backends.backends
            .first { $0.name == active }?.label ?? active
    }

    private var runnerSummary: String {
        guard state.runnerAvailable else { return "not available" }
        guard let model = state.runner.model else { return "nothing loaded" }
        return shortModelName(model.modelID)
    }

    var body: some View {
        List {
            Section("Paired with") {
                LabeledContent("Server", value: state.connection.serverName.isEmpty ? "—" : state.connection.serverName)
                LabeledContent("This device", value: state.connection.deviceName)
                if state.isKeyless {
                    Label(
                        "Connected without a key, because this network is trusted by the PC. Administrator actions are not available.",
                        systemImage: "lock.open"
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
                LabeledContent("T1 API", value: state.connection.t1Version.isEmpty ? "—" : "v" + state.connection.t1Version)
                if let status = state.serverStatus {
                    LabeledContent("HyperNix", value: status.hypernixVersion)
                    LabeledContent("Models registered", value: "\(status.modelCount)")
                    LabeledContent(
                        "LM Studio bridge",
                        value: status.lmstudioBridgeEnabled ? "on" : "off"
                    )
                }
                if let uptime = state.uptime {
                    // Worth a line because it answers a question people
                    // actually ask: a conversation that lost its context
                    // or a pairing that stopped working is usually a PC
                    // that rebooted, and nothing here used to say so.
                    LabeledContent("Server up for", value: uptime.serverDescription)
                    LabeledContent("Machine up for", value: uptime.machineDescription)
                }
            }

            Section {
                NavigationLink {
                    ServersView()
                } label: {
                    HStack {
                        Text("Servers")
                        Spacer()
                        Text("\(state.savedServers.count)")
                            .foregroundStyle(.secondary)
                    }
                }
            } footer: {
                Text(
                    "HyperLink remembers up to \(SavedServers.maxServers) machines "
                    + "and switches between them without re-pairing."
                )
            }

            Section {
                // What would actually answer a message right now. Worth
                // a line because the two failures look identical from
                // here and need opposite fixes: "no models on this
                // server" is solved by downloading one, "a model is
                // loaded but nothing is serving it" by turning
                // something on.
                LabeledContent("Answering with", value: answeringWith)
                NavigationLink {
                    RunnerView()
                } label: {
                    HStack {
                        Label("Runner", systemImage: "bolt")
                        Spacer()
                        Text(runnerSummary)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
                NavigationLink {
                    HardwareView()
                } label: {
                    Label("Hardware", systemImage: "gauge.with.dots.needle.67percent")
                }
                NavigationLink {
                    ServerUpdateView()
                } label: {
                    Label("Update the server", systemImage: "arrow.up.square")
                }
            } header: {
                Text("This machine")
            } footer: {
                Text(
                    "Load and unload models without touching the PC, see what its "
                    + "hardware is doing, and get the exact update commands for "
                    + "the Python it is actually running under."
                )
            }

            Section("Appearance") {
                NavigationLink {
                    ThemePickerView()
                } label: {
                    HStack {
                        Text("Theme")
                        Spacer()
                        Text(themes.theme.name)
                            .foregroundStyle(.secondary)
                        // The accent as a dot, so the current theme is
                        // legible from the settings list without opening
                        // the picker.
                        Circle()
                            .fill(themes.theme.accent)
                            .frame(width: 11, height: 11)
                            .accessibilityHidden(true)
                    }
                }
            }

            Section {
                ForEach(state.connection.endpoints, id: \.self) { endpoint in
                    HStack {
                        Text(endpoint)
                            .font(.system(.caption, design: .monospaced))
                            .lineLimit(1)
                        Spacer()
                        if endpoint.contains(".ts.net") || endpoint.contains("://100.") {
                            Image(systemName: "globe")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .accessibilityLabel("works away from home")
                        }
                    }
                }
            } header: {
                Text("Addresses tried, in order")
            } footer: {
                Text("The app uses whichever answers first. A globe marks the ones that keep working when you leave the house.")
            }

            Section {
                if state.connection.serverFingerprint.isEmpty {
                    Text("This server has not told the app its identity. Update it to HyperNix 0.72.4 or newer.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                } else {
                    Text(ServerIdentity.display(state.connection.serverFingerprint))
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                }
            } header: {
                Text("Server identity")
            } footer: {
                Text("Pinned when you paired. HyperLink checks it every time it reconnects, because an address can end up pointing at a different machine — and a machine can call itself anything it likes.")
            }

            if let warning = state.identityWarning {
                Section {
                    Label(warning, systemImage: "exclamationmark.shield")
                        .font(.callout)
                        .foregroundStyle(.red)
                    Button(role: .destructive) {
                        confirmingUnpair = true
                    } label: {
                        Text("Unpair and start again")
                    }
                }
            }

            Section {
                Toggle("Keep admin credentials", isOn: $keepAdminCredentials)
            } header: {
                Text("Administration")
            } footer: {
                Text(keepAdminCredentials
                     ? "An admin credential stays in the keychain until you unpair. Anyone who unlocks this phone can use it to administer your machine."
                     : "An admin credential is forgotten when HyperLink restarts. You will be asked for it again next time.")
            }

            if let error = state.connectionError {
                Section {
                    Label(error, systemImage: "wifi.exclamationmark")
                        .font(.callout)
                        .foregroundStyle(.orange)
                }
            }

            Section {
                Button {
                    Task { await state.refreshAll() }
                } label: {
                    Label("Check the connection", systemImage: "arrow.clockwise")
                }
                Button(role: .destructive) {
                    confirmingUnpair = true
                } label: {
                    Label("Unpair this device", systemImage: "minus.circle")
                }
            }
        }
        .navigationTitle("Server")
        .onChange(of: keepAdminCredentials) { _, keep in
            // Setting it to false deletes what is stored, immediately.
            // A switch that says "do not keep this" and leaves the
            // credential there until tomorrow is worse than no switch,
            // because it is believed.
            AdminCredentialStore.persistsAcrossLaunches = keep
        }
        .confirmationDialog(
            "Unpair this device?",
            isPresented: $confirmingUnpair,
            titleVisibility: .visible
        ) {
            Button("Unpair", role: .destructive) { Task { await state.unpair() } }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Your conversations stay on the PC. You will need a new pairing code to connect again.")
        }
    }
}
