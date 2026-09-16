//  RunnerView.swift
//  Loading a model on the PC, from the phone.
//
//  Before this, "switch model" meant walking over to the PC and opening
//  LM Studio. The server could hold forty GGUFs in ~/.hypernix/models
//  and serve none of them, and nothing in HyperLink could change that.
//
//  `/runner/*` owns a llama.cpp process, so load, unload and switch are
//  operations. This screen is the operations.
//
//  Two things it deliberately does not do
//  --------------------------------------
//  **It does not decide.** Every number here — the backends offered, the
//  layer split, whether a model fits — comes from the server, because
//  the server is the thing with the GPU. A phone that computed its own
//  layer count would be guessing about hardware it cannot see.
//
//  **It does not load without showing the plan first.** Loading evicts
//  whatever people are currently talking to. `/runner/plan` costs one
//  request and changes nothing, and seeing "41 of 81 on the GPU" before
//  committing is the difference between a decision and a surprise.

import SwiftUI

struct RunnerView: View {
    @Environment(AppState.self) private var state
    @State private var selected: CatalogueModel?
    @State private var confirmingUnload = false

    private var loadable: [CatalogueModel] {
        // Only models with a file on this machine. A model the bridge
        // borrowed from LM Studio has no path here, and the runner
        // cannot load what it cannot open.
        state.catalogue.models.filter { $0.runnable && !$0.path.isEmpty }
    }

    var body: some View {
        List {
            if !state.runnerAvailable {
                Section {
                    ContentUnavailableView {
                        Label("No runner here", systemImage: "bolt.slash")
                    } description: {
                        Text(
                            "This server does not offer HyperNix-managed inference. "
                            + "It needs HyperNix 0.72.5 or newer, with a built "
                            + "llama.cpp — until then, load models in LM Studio "
                            + "on the PC."
                        )
                    }
                }
            } else {
                runningSection
                loadableSection
            }

            if let error = state.runnerError {
                Section {
                    Label(error, systemImage: "exclamationmark.triangle")
                        .font(.callout)
                        .foregroundStyle(.orange)
                } footer: {
                    Text(
                        "Refusals here are usually about the machine rather than "
                        + "the request: no built llama.cpp, or a model that does "
                        + "not fit alongside what is already loaded."
                    )
                }
            }
        }
        .navigationTitle("Runner")
        .refreshable { await state.refreshRunner() }
        .task {
            await state.refreshRunner()
            await state.refreshModels()
        }
        .sheet(item: $selected) { model in
            LoadModelSheet(model: model)
        }
        .confirmationDialog(
            "Stop serving?", isPresented: $confirmingUnload, titleVisibility: .visible
        ) {
            Button("Unload", role: .destructive) {
                Task { await state.unloadModel() }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(
                "Anyone talking to this server stops getting answers until "
                + "something is loaded again."
            )
        }
    }

    @ViewBuilder
    private var runningSection: some View {
        Section("Running now") {
            if let model = state.runner.model {
                VStack(alignment: .leading, spacing: 4) {
                    Text(shortModelName(model.modelID)).font(.headline)
                    if let placement = model.placement {
                        Text(placement.layerSummary)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        if !placement.reason.isEmpty {
                            Text(placement.reason)
                                .font(.caption2)
                                .foregroundStyle(.tertiary)
                        }
                    }
                    HStack(spacing: 6) {
                        if model.contextLength > 0 {
                            Text("\(model.contextLength / 1024)k context")
                        }
                        if model.uptimeSeconds > 0 {
                            Text("·")
                            Text("up \(ServerUptime.describe(model.uptimeSeconds))")
                        }
                    }
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                }
                Button(role: .destructive) {
                    confirmingUnload = true
                } label: {
                    Label("Unload", systemImage: "eject")
                }
                .disabled(state.runnerBusy)
            } else {
                Text("Nothing loaded. Pick a model below.")
                    .foregroundStyle(.secondary)
            }

            if state.runnerBusy {
                HStack(spacing: 8) {
                    ProgressView()
                    Text("Working — a large model can take minutes to load.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    @ViewBuilder
    private var loadableSection: some View {
        Section {
            if loadable.isEmpty {
                Text(
                    "No GGUF files on this server to load. Drop one in "
                    + "~/.hypernix/models and run `hypernix-t1 index`."
                )
                .foregroundStyle(.secondary)
            }
            ForEach(loadable) { model in
                Button {
                    selected = model
                } label: {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(model.name.isEmpty
                                 ? shortModelName(model.modelID) : model.name)
                                .lineLimit(1)
                                .foregroundStyle(.primary)
                            if !model.summary.isEmpty {
                                Text(model.summary)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        Spacer()
                        if model.modelID == state.runner.model?.modelID {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(.tint)
                        }
                    }
                }
                .disabled(state.runnerBusy)
            }
        } header: {
            Text("On this server")
        } footer: {
            Text(
                "Loading one replaces whatever is running. You will see where "
                + "its layers go before anything happens."
            )
        }
    }
}

/// Pick the tuning, see the plan, then commit.
private struct LoadModelSheet: View {
    let model: CatalogueModel
    @Environment(AppState.self) private var state
    @Environment(\.dismiss) private var dismiss

    @State private var backend = "auto"
    @State private var automaticLayers = true
    @State private var gpuLayers: Double = 0
    @State private var totalLayers = ""
    @State private var contextLength = ""
    @State private var plan: RunnerPlan?
    @State private var planning = false

    private var backends: [String] {
        // The server's list, never a hard-coded one: a machine without
        // CUDA must not be offered CUDA, and only the server knows which
        // build it has.
        state.runner.backends.isEmpty ? ["auto"] : state.runner.backends
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Text(model.name.isEmpty ? shortModelName(model.modelID) : model.name)
                        .font(.headline)
                    if !model.summary.isEmpty {
                        Text(model.summary)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Section("Backend") {
                    Picker("Backend", selection: $backend) {
                        ForEach(backends, id: \.self) { name in
                            Text(label(for: name)).tag(name)
                        }
                    }
                    .pickerStyle(.menu)
                }

                Section {
                    Toggle("Decide for me", isOn: $automaticLayers)
                    if !automaticLayers {
                        Stepper(
                            "GPU layers: \(Int(gpuLayers))",
                            value: $gpuLayers, in: 0...200
                        )
                        TextField("Total layers (optional)", text: $totalLayers)
                            .keyboardType(.numberPad)
                    }
                } header: {
                    Text("Where the layers go")
                } footer: {
                    Text(automaticLayers
                         ? "The server fits as many layers on the GPU as its free VRAM allows, and the rest run on the CPU."
                         : "Your number is used as given. Without a total, a partial offload has no denominator — supply one if you know it.")
                }

                Section {
                    TextField("Context length (optional)", text: $contextLength)
                        .keyboardType(.numberPad)
                } footer: {
                    Text("Left empty, the model's own limit is used. A longer context costs VRAM that then cannot hold layers.")
                }

                if let placement = plan?.placement {
                    Section("What would happen") {
                        Text(placement.layerSummary)
                        if !placement.reason.isEmpty {
                            Text(placement.reason)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                Section {
                    Button {
                        Task { await check() }
                    } label: {
                        HStack {
                            if planning { ProgressView().padding(.trailing, 6) }
                            Text("Check where it fits")
                        }
                    }
                    .disabled(planning || state.runnerBusy)

                    Button {
                        Task {
                            if await state.loadModel(
                                modelID: model.modelID,
                                gpuLayers: automaticLayers ? nil : Int(gpuLayers),
                                backend: backend,
                                contextLength: Int(contextLength),
                                totalLayers: Int(totalLayers)
                            ) {
                                dismiss()
                            }
                        }
                    } label: {
                        HStack {
                            if state.runnerBusy { ProgressView().padding(.trailing, 6) }
                            Text(state.runner.loaded ? "Replace what is running" : "Load")
                        }
                    }
                    .disabled(state.runnerBusy)
                } footer: {
                    Text(state.runner.loaded
                         ? "Whatever is running now stops answering the moment this starts loading."
                         : "Nothing is running, so nothing is interrupted.")
                }
            }
            .navigationTitle("Load")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
            .task {
                // The context the indexer read, as the starting point —
                // rather than an empty field the person has to guess at.
                if contextLength.isEmpty && model.contextLimit > 0 {
                    contextLength = "\(model.contextLimit)"
                }
                await check()
            }
        }
    }

    private func check() async {
        planning = true
        defer { planning = false }
        plan = await state.planLoad(
            modelID: model.modelID,
            gpuLayers: automaticLayers ? nil : Int(gpuLayers),
            backend: backend,
            contextLength: Int(contextLength),
            totalLayers: Int(totalLayers)
        )
    }

    private func label(for backend: String) -> String {
        switch backend {
        case "auto": return "Whatever works (auto)"
        case "cuda": return "CUDA — NVIDIA"
        case "vulkan": return "Vulkan — AMD, Intel, NVIDIA"
        case "cpu": return "CPU only"
        case "hnx-cuda": return "HyperNix engine, CUDA"
        case "hnx-cpu": return "HyperNix engine, CPU"
        default: return backend
        }
    }
}
