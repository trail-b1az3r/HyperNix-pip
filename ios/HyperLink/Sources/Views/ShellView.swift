//  ShellView.swift
//  A command line on the server, from the phone.
//
//  Off unless the person running the server turned it on
//  (T1_HYPERLINK_SHELL=1): a paired phone is otherwise a chat client, and
//  this makes it a terminal on that machine. When it is off the screen
//  says so and how to change it, rather than offering a box that fails.
//
//  One command at a time, with its output, exit code and how long it
//  took. Not a live terminal: there is no stdin, so an interactive
//  program waits until the server's timeout and is then killed.

import SwiftUI
import UIKit

struct ShellView: View {
    @Environment(AppState.self) private var state
    @State private var command = ""
    @State private var cwd = ""
    @State private var running = false
    @State private var history: [ShellResult] = []
    @FocusState private var focused: Bool

    var body: some View {
        List {
            if let status = state.shellStatus {
                if status.enabled {
                    Section {
                        TextField("Command", text: $command, axis: .vertical)
                            .font(.system(.body, design: .monospaced))
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .focused($focused)
                            .submitLabel(.go)
                            .onSubmit { Task { await run() } }
                        TextField("Folder (default: the server's home)", text: $cwd)
                            .font(.system(.footnote, design: .monospaced))
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                        Button {
                            Task { await run() }
                        } label: {
                            if running {
                                HStack { ProgressView(); Text("Running…") }
                            } else {
                                Label("Run", systemImage: "play.fill")
                            }
                        }
                        .disabled(running || command.trimmingCharacters(in: .whitespaces).isEmpty)
                    } footer: {
                        Text("Runs as the server's own user. Stopped after \(Int(status.timeoutSeconds)) seconds. There is no keyboard input, so interactive programs will wait and then be stopped.")
                    }
                    ForEach(history) { result in
                        Section {
                            ShellResultView(result: result)
                        } header: {
                            Text(result.command).font(.system(.caption, design: .monospaced)).textCase(nil)
                        }
                    }
                } else {
                    Section {
                        ContentUnavailableView {
                            Label("The shell is off", systemImage: "terminal")
                        } description: {
                            Text("Whoever runs the server can turn it on: \(status.howToEnable). It lets this phone run commands on that machine, so it is off unless chosen.")
                        }
                    }
                }
            } else {
                Section {
                    ContentUnavailableView(
                        "Not available",
                        systemImage: "terminal",
                        description: Text("This server does not offer a shell. It may need updating.")
                    )
                }
            }
        }
        .navigationTitle("Shell")
        .task { await state.refreshShellStatus() }
        .refreshable { await state.refreshShellStatus() }
    }

    private func run() async {
        let text = command.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !running else { return }
        running = true
        defer { running = false }
        let folder = cwd.trimmingCharacters(in: .whitespacesAndNewlines)
        if let result = await state.runShell(text, cwd: folder.isEmpty ? nil : folder) {
            history.insert(result, at: 0)
            command = ""
            focused = true
        }
    }
}

private struct ShellResultView: View {
    let result: ShellResult

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                if result.timedOut {
                    Label("stopped after the time limit", systemImage: "clock.badge.exclamationmark")
                        .foregroundStyle(.orange)
                } else if let code = result.exitCode {
                    Label("exit \(code)", systemImage: code == 0 ? "checkmark.circle" : "xmark.octagon")
                        .foregroundStyle(code == 0 ? .green : .red)
                }
                Spacer()
                Text(String(format: "%.2fs", result.elapsedSeconds)).foregroundStyle(.secondary)
            }
            .font(.caption)
            if !result.stdout.isEmpty {
                output(result.stdout, error: false)
            }
            if !result.stderr.isEmpty {
                output(result.stderr, error: true)
            }
            if result.stdout.isEmpty && result.stderr.isEmpty {
                Text("No output.").font(.caption).foregroundStyle(.secondary)
            }
            if result.truncated {
                Text("Output was long and has been cut to its end.")
                    .font(.caption2).foregroundStyle(.secondary)
            }
        }
    }

    private func output(_ text: String, error: Bool) -> some View {
        ScrollView(.horizontal, showsIndicators: true) {
            Text(text)
                .font(.system(.footnote, design: .monospaced))
                .foregroundStyle(error ? .red : .primary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .contextMenu {
            Button {
                UIPasteboard.general.string = text
            } label: {
                Label("Copy", systemImage: "doc.on.doc")
            }
        }
    }
}
