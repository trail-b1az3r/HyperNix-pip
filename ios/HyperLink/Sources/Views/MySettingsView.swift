//  MySettingsView.swift
//  Who you are, and how you want to be answered.
//
//  Distinct from `SettingsView`, which is about the *machine*. This is
//  about the person: profile, bio, the system prompt every conversation
//  starts with, how hard the model should think, what happens when it
//  crashes, and what it is allowed to do.
//
//  All of it lives on the server. Two reasons, and the second decides
//  it: a person with a phone and a tablet is one person, and these are
//  *inputs to generation* — the prompt, the effort level and the context
//  bounds all have to be in the process that builds the request.
//
//  The bounds are the server's too
//  -------------------------------
//  The effort levels in the picker and the limits on the context fields
//  come from the server with the values. The app carrying its own copy
//  would mean offering a level this build rejects — a settings screen
//  that cannot save, with no way for the phone to know why.
//
//  Clamps are shown
//  ----------------
//  A context maximum of four million is not a preference; it is a number
//  that makes every reply fail two minutes later and somewhere
//  unrelated. The server lowers it and says so, and that note goes on
//  screen. Silently storing something other than what somebody typed is
//  how a settings screen becomes untrustworthy.

import SwiftUI

struct MySettingsView: View {
    @Environment(AppState.self) private var state

    @State private var displayName = ""
    @State private var bio = ""
    @State private var contextMinimum = ""
    @State private var contextMaximum = ""
    @State private var backupModel = ""
    @State private var notes: [String] = []
    @State private var saving = false
    @State private var confirmingReset = false

    private var preferences: UserPreferences { state.settings.preferences }

    var body: some View {
        List {
            profileSection
            promptSection
            effortSection
            contextSection
            reliabilitySection
            capabilitySection
            memorySection

            if !notes.isEmpty {
                Section("The server adjusted this") {
                    ForEach(notes, id: \.self) { note in
                        Label(note, systemImage: "info.circle")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    }
                }
            }

            Section {
                Button(role: .destructive) {
                    confirmingReset = true
                } label: {
                    Label("Reset everything", systemImage: "arrow.counterclockwise")
                }
            }
        }
        .navigationTitle("You")
        .refreshable { await load() }
        .task { await load() }
        .confirmationDialog(
            "Reset all your settings?",
            isPresented: $confirmingReset, titleVisibility: .visible
        ) {
            Button("Reset", role: .destructive) {
                Task {
                    await state.resetSettings()
                    await load()
                }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Your profile, prompt and preferences go back to the defaults. Your conversations and memories are untouched.")
        }
    }

    // MARK: - Sections

    private var profileSection: some View {
        Section {
            TextField("Name", text: $displayName)
                .onSubmit { Task { await save(.init(display_name: displayName)) } }
            TextField("A little about you", text: $bio, axis: .vertical)
                .lineLimit(3...8)
        } header: {
            Text("Profile")
        } footer: {
            Text("Both go at the top of every conversation, so you do not have to explain yourself again each time. The bio is sent on every single turn — a paragraph, not an essay.")
        }
    }

    private var promptSection: some View {
        Section {
            NavigationLink {
                SystemPromptView()
            } label: {
                HStack {
                    Text("System prompt")
                    Spacer()
                    Text(preferences.systemPrompt.isEmpty
                         ? "none"
                         : "\(preferences.systemPrompt.count) characters")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        } header: {
            Text("Instructions")
        } footer: {
            Text("Standing instructions for every conversation. A chat's own prompt comes after this one, so a conversation can override the default rather than fight it.")
        }
    }

    private var effortSection: some View {
        Section {
            Picker("Effort", selection: Binding(
                get: { preferences.effort },
                set: { level in Task { await save(.init(effort: level)) } }
            )) {
                ForEach(state.settings.effortLevels, id: \.self) { level in
                    Text(level.capitalized).tag(level)
                }
            }
            .pickerStyle(.menu)
        } header: {
            Text("How hard to think")
        } footer: {
            Text("A model with a real reasoning control is given this level by name. One without gets a matching temperature and answer length — which is an approximation, and is meant to be: it is scheduling, not thinking.")
        }
    }

    private var contextSection: some View {
        Section {
            TextField("Minimum (0 for no opinion)", text: $contextMinimum)
                .keyboardType(.numberPad)
            TextField("Maximum (0 for no opinion)", text: $contextMaximum)
                .keyboardType(.numberPad)
            Button("Apply context bounds") {
                Task {
                    await save(.init(
                        context_minimum: Int(contextMinimum) ?? 0,
                        context_maximum: Int(contextMaximum) ?? 0
                    ))
                }
            }
            .disabled(saving)
        } header: {
            Text("Context")
        } footer: {
            Text("How much history to send. Zero means “use whatever the model says it can do”, which is the right answer for almost everybody. The maximum is the useful half: it stops a long thread costing a full context window on every turn. This server accepts \(state.settings.contextFloor)–\(state.settings.contextCeiling).")
        }
    }

    private var reliabilitySection: some View {
        Section {
            TextField("Backup model", text: $backupModel)
                .autocorrectionDisabled()
                .textInputAutocapitalization(.never)
            Button("Set backup model") {
                Task { await save(.init(backup_model: backupModel)) }
            }
            .disabled(saving)
        } header: {
            Text("When the main model fails")
        } footer: {
            Text("Tried once if the model you asked for does not answer. Leave it empty to fail honestly instead — an answer from a model you did not choose is worse than no answer, unless you asked for one. When it is used, the message says so.")
        }
    }

    private var capabilitySection: some View {
        Section {
            Toggle("Let the model use tools", isOn: Binding(
                get: { preferences.toolsEnabled },
                set: { on in Task { await save(.init(tools_enabled: on)) } }
            ))
        } header: {
            Text("Tools")
        } footer: {
            Text("The model can create and edit files, run fish commands and make archives in its own workspace on the server — so “zip the logs and tell me what is in them” is one message. Off by default, because letting a model write files on your machine is not a default. What it does is recorded on each reply.")
        }
    }

    private var memorySection: some View {
        Section {
            Toggle("Remember things about me", isOn: Binding(
                get: { preferences.autoMemory },
                set: { on in Task { await save(.init(auto_memory: on)) } }
            ))
            NavigationLink {
                MemoryView()
            } label: {
                HStack {
                    Text("Memories")
                    Spacer()
                    Text("\(state.memories.count)")
                        .foregroundStyle(.secondary)
                }
            }
        } header: {
            Text("Memory")
        } footer: {
            Text("Facts carried between conversations. You can read, edit and delete every one of them — including the ones the model wrote itself.")
        }
    }

    // MARK: - Work

    private func load() async {
        await state.refreshSettings()
        await state.refreshMemories()
        displayName = preferences.displayName
        bio = preferences.bio
        contextMinimum = preferences.contextMinimum == 0
            ? "" : "\(preferences.contextMinimum)"
        contextMaximum = preferences.contextMaximum == 0
            ? "" : "\(preferences.contextMaximum)"
        backupModel = preferences.backupModel
    }

    private func save(_ patch: PreferencesPatch) async {
        saving = true
        defer { saving = false }
        notes = await state.saveSettings(patch)
    }
}

/// The long one, on its own screen because it is long.
struct SystemPromptView: View {
    @Environment(AppState.self) private var state
    @Environment(\.dismiss) private var dismiss
    @State private var draft = ""
    @State private var saving = false

    var body: some View {
        Form {
            Section {
                TextEditor(text: $draft)
                    .frame(minHeight: 240)
                    .font(.system(.body, design: .monospaced))
            } header: {
                Text("Standing instructions")
            } footer: {
                Text("\(draft.count) of \(state.settings.maxSystemPrompt) characters. Everything here is sent before every conversation, so it is charged for on every turn.")
            }
        }
        .navigationTitle("System prompt")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button {
                    Task {
                        saving = true
                        await state.saveSettings(.init(system_prompt: draft))
                        saving = false
                        dismiss()
                    }
                } label: {
                    if saving { ProgressView() } else { Text("Save") }
                }
                .disabled(saving || draft.count > state.settings.maxSystemPrompt)
            }
        }
        .task { draft = state.settings.preferences.systemPrompt }
    }
}
