//  ModelsView.swift
//  Every model the server can offer — not just the one LM Studio has open.
//
//  This screen used to show `/bridge/lmstudio/models`: whatever LM
//  Studio had loaded, and nothing else. A machine with forty GGUFs in
//  ~/.hypernix/models showed an empty list under a message telling the
//  user to go and open LM Studio, which was both wrong and the whole
//  complaint — the models were right there.
//
//  It now shows the merged catalogue, grouped by where each model came
//  from, and says what each source reported. That last part matters:
//  an empty list meant either "this server has no models" or "LM Studio
//  is not running", and one blank screen was shown for both.

import SwiftUI

struct ModelsView: View {
    @Environment(AppState.self) private var state
    @State private var showingDownload = false

    private var grouped: [(source: String, models: [CatalogueModel])] {
        // Registry, then LM Studio, then loose files: most-described
        // first, so the rows with the most to say about themselves are
        // not buried under bare filenames.
        let order = ["registry", "lmstudio", "local"]
        return order.compactMap { source in
            let models = state.catalogue.models.filter { $0.source == source }
            return models.isEmpty ? nil : (source, models)
        }
    }

    var body: some View {
        List {
            if state.catalogue.models.isEmpty {
                Section {
                    ContentUnavailableView {
                        Label("No models", systemImage: "cpu")
                    } description: {
                        Text(emptyExplanation)
                    }
                }
            }

            ForEach(grouped, id: \.source) { group in
                Section(sectionTitle(group.source)) {
                    ForEach(group.models) { model in
                        CatalogueRow(model: model)
                    }
                } footer: {
                    Text(sectionFooter(group.source))
                }
            }

            if !state.catalogue.unavailable.isEmpty {
                Section("Not reachable") {
                    ForEach(state.catalogue.unavailable) { source in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(source.label)
                            if !source.detail.isEmpty {
                                Text(source.detail)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                } footer: {
                    Text(
                        "A source that cannot be reached contributes nothing — "
                        + "which is not the same as having nothing to contribute."
                    )
                }
            }

            Section {
                Button {
                    showingDownload = true
                } label: {
                    Label("Add a model from Hugging Face", systemImage: "arrow.down.circle")
                }
            } footer: {
                Text("Paste a model page link, a direct download link, or both — HyperNix works out which files are actually needed.")
            }
        }
        .navigationTitle("Models")
        .refreshable { await state.refreshModels() }
        .task { await state.refreshModels() }
        .sheet(isPresented: $showingDownload) { ModelDownloadView() }
    }

    /// Why the list is empty, in the terms of whichever thing is
    /// actually wrong.
    private var emptyExplanation: String {
        let broken = state.catalogue.unavailable
        if !broken.isEmpty {
            let named = broken.map(\.label).joined(separator: ", ")
            return "\(named) could not be reached, so this list is incomplete. "
                + "Pull to refresh once it is back."
        }
        return "This server has no models registered, none in ~/.hypernix/models, "
            + "and nothing loaded in LM Studio. Add one below, or drop a GGUF in "
            + "the models folder and run `hypernix-t1 index`."
    }

    private func sectionTitle(_ source: String) -> String {
        switch source {
        case "registry": return "Registered on the server"
        case "lmstudio": return "Open in LM Studio"
        case "local": return "In the models folder"
        default: return source
        }
    }

    private func sectionFooter(_ source: String) -> String {
        switch source {
        case "registry":
            return "Indexed by `hypernix-t1 index`, with prices and context limits."
        case "lmstudio":
            return "Borrowed from LM Studio. The bridge cannot load one for you."
        case "local":
            return "GGUF files on the server's disk. Run `hypernix-t1 index` to "
                + "register them with their real context limits."
        default:
            return ""
        }
    }
}

/// One model in the catalogue.
private struct CatalogueRow: View {
    let model: CatalogueModel

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.name.isEmpty ? shortModelName(model.modelID) : model.name)
                    .lineLimit(1)
                if !model.summary.isEmpty {
                    Text(model.summary)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                if !model.alsoIn.isEmpty {
                    // One model known to two sources is one model. Saying
                    // so is what stops the same GGUF appearing twice and
                    // looking like a duplicate nobody cleaned up.
                    Text("also in \(model.alsoIn.joined(separator: ", "))")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                if !model.runnable && !model.detail.isEmpty {
                    Text(model.detail)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
            if model.loaded {
                Image(systemName: "circle.fill")
                    .font(.caption2)
                    .foregroundStyle(.green)
                    .accessibilityLabel("loaded")
            }
        }
    }
}

struct ModelPickerSheet: View {
    let sessionID: String
    let currentModel: String
    @Environment(AppState.self) private var state
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                if state.catalogue.models.isEmpty {
                    Text(
                        "No models on this server. Drop a GGUF in ~/.hypernix/models "
                        + "and run `hypernix-t1 index`, or load one in LM Studio."
                    )
                    .foregroundStyle(.secondary)
                }
                ForEach(state.catalogue.models) { model in
                    Button {
                        Task {
                            await state.setModel(model.modelID, for: sessionID)
                            dismiss()
                        }
                    } label: {
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(model.name.isEmpty
                                     ? shortModelName(model.modelID) : model.name)
                                    .lineLimit(1)
                                    .foregroundStyle(.primary)
                                Text(model.loaded
                                     ? model.sourceLabel
                                     : "\(model.sourceLabel) — not loaded yet")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            if model.modelID == currentModel {
                                Image(systemName: "checkmark").foregroundStyle(.tint)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Model")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .task { await state.refreshModels() }
        }
    }
}
