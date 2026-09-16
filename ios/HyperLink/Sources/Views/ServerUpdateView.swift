//  ServerUpdateView.swift
//  The commands to update the PC, with the right interpreter in them.
//
//  "The T1 installed thinks it is running an older version." Half of
//  that was the installer printing a hand-maintained constant that had
//  gone stale. The other half is real: a server falls behind, and until
//  now finding out meant walking over to the machine.
//
//  Why this is a copy area and not a button
//  ----------------------------------------
//  Updating the package under a running server is a decision with a
//  restart attached. A phone button that did it silently is a phone
//  button that takes a machine down in the middle of somebody else's
//  conversation, and the failure modes — a venv that needs sudo, a
//  distribution that refuses to touch system packages — are ones a
//  person handles at a terminal and an API cannot.
//
//  What makes it worth having is the interpreter. `pip install -U
//  hypernix` on a machine with a system Python, a pyenv and the
//  service's own venv upgrades whichever is first on the path, prints
//  success, and leaves the server running exactly the version it was.
//  The server knows `sys.executable`; these commands name it.

import SwiftUI
import UIKit

struct ServerUpdateView: View {
    @Environment(AppState.self) private var state
    @State private var advice: UpgradeAdvice?
    @State private var failure: String?
    @State private var copied: String?

    var body: some View {
        List {
            if let installation = advice?.installation {
                Section("Running now") {
                    LabeledContent("HyperNix", value: installation.packageVersion)
                    LabeledContent("T1 API", value: "v" + installation.t1Version)
                    LabeledContent("Python", value: installation.pythonVersion)
                    if installation.editable {
                        Label(
                            "Development install — running from a checkout",
                            systemImage: "hammer"
                        )
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    } else if !installation.inVenv {
                        Label(
                            "Not in a virtual environment",
                            systemImage: "exclamationmark.triangle"
                        )
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    }
                }

                Section {
                    Text(installation.executable)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                } header: {
                    Text("The interpreter it runs under")
                } footer: {
                    Text(
                        "Every command below names this one. On a machine with "
                        + "more than one Python, a bare `pip install` upgrades a "
                        + "different installation and reports success."
                    )
                }
            }

            if let commands = advice?.commands, !commands.isEmpty {
                Section("Run these on the PC") {
                    ForEach(commands) { command in
                        CommandRow(command: command, copied: $copied)
                    }
                }
            }

            if let warnings = advice?.warnings, !warnings.isEmpty {
                Section("Worth knowing") {
                    ForEach(warnings, id: \.self) { warning in
                        Label(warning, systemImage: "info.circle")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            if let failure {
                Section {
                    ContentUnavailableView {
                        Label("Could not ask", systemImage: "wifi.exclamationmark")
                    } description: {
                        Text(failure)
                    }
                } footer: {
                    Text(
                        "A server older than 0.72.5 has no /hyperlink/upgrade "
                        + "endpoint. Update it with `pip install --upgrade "
                        + "hypernix` from whichever Python it runs under."
                    )
                }
            }
        }
        .navigationTitle("Update the server")
        .refreshable { await load() }
        .task { await load() }
    }

    private func load() async {
        do {
            advice = try await state.upgradeAdvice()
            failure = nil
        } catch {
            advice = nil
            failure = (error as? HyperLinkError)?.errorDescription
                ?? error.localizedDescription
        }
    }
}

/// One command, with the copy button that is the point of the screen.
private struct CommandRow: View {
    let command: ServerCommand
    @Binding var copied: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(command.label)
                    .font(.subheadline.weight(command.primary ? .semibold : .regular))
                Spacer()
                Button {
                    UIPasteboard.general.string = command.command
                    copied = command.command
                    // Long enough to read, short enough that it does not
                    // still say "Copied" when you come back to the screen.
                    Task {
                        try? await Task.sleep(for: .seconds(2))
                        if copied == command.command { copied = nil }
                    }
                } label: {
                    Label(
                        copied == command.command ? "Copied" : "Copy",
                        systemImage: copied == command.command
                            ? "checkmark" : "doc.on.doc"
                    )
                    .font(.caption)
                    .labelStyle(.titleAndIcon)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
            Text(command.command)
                .font(.system(.caption, design: .monospaced))
                .textSelection(.enabled)
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(.quaternary.opacity(0.4))
                )
            if !command.note.isEmpty {
                Text(command.note)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
    }
}
