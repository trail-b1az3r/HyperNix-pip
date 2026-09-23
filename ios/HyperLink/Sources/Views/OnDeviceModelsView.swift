//  OnDeviceModelsView.swift
//  Models that run on this iPhone: find one, check it fits, download it,
//  load it, talk to it.
//
//  Every step here existed as code and none of it had a screen — search,
//  fit, download and the runner were written, tested in isolation, and
//  unreachable. This is the screen.
//
//  Fit is shown before the download, not after. A model that will not
//  fit is still listed, marked, and still downloadable — somebody may be
//  about to free memory — but nobody should spend twenty minutes of
//  cellular data to find out it cannot load.

import SwiftUI

struct OnDeviceModelsView: View {
    @EnvironmentObject private var hub: OnDeviceHub
    @ObservedObject private var store = ModelStore.shared

    @State private var query = ""
    @State private var results: [HFModelSummary] = []
    @State private var searching = false
    @State private var searchError: String?
    @State private var loadError: String?
    @State private var loading: String?
    @State private var chatting = false

    var body: some View {
        List {
            if !hub.inference.hasLocalEngine {
                Section {
                    Label(
                        "This build has no on-device engine, so models can be downloaded "
                        + "but not run. Build with ios/scripts/build_llama_xcframework.sh.",
                        systemImage: "exclamationmark.triangle"
                    )
                    .foregroundStyle(.orange)
                }
            }

            installedSection
            searchSection
        }
        .navigationTitle("On this iPhone")
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                NavigationLink {
                    OnDeviceSettingsView()
                } label: {
                    Label("On-device settings", systemImage: "slider.horizontal.3")
                }
            }
        }
        .searchable(text: $query, prompt: "Search Hugging Face for a GGUF")
        .onSubmit(of: .search) { Task { await runSearch() } }
        .navigationDestination(isPresented: $chatting) { LocalChatView() }
        .alert("Could not load", isPresented: Binding(
            get: { loadError != nil }, set: { if !$0 { loadError = nil } }
        )) {
            Button("OK", role: .cancel) { loadError = nil }
        } message: {
            Text(loadError ?? "")
        }
    }

    // MARK: - Installed

    @ViewBuilder
    private var installedSection: some View {
        Section {
            if store.installed.isEmpty {
                Text("Nothing downloaded yet. Search below for a small GGUF — a 1–4B model at Q4 is the right size for a phone.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            ForEach(store.installed) { model in
                installedRow(model)
            }
            .onDelete { offsets in
                for index in offsets {
                    let model = store.installed[index]
                    if hub.inference.loaded?.id == model.id {
                        Task { await hub.inference.unload() }
                    }
                    store.delete(model)
                }
            }
        } header: {
            Text("Downloaded")
        } footer: {
            if !store.installed.isEmpty {
                Text(ByteCountFormatter.string(fromByteCount: Int64(store.installedBytes),
                                               countStyle: .file) + " on this iPhone")
            }
        }
    }

    private func installedRow(_ model: InstalledModel) -> some View {
        let plan = hub.inference.plan(for: model)
        let isLoaded = hub.inference.loaded?.id == model.id
        return HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(model.displayName).lineLimit(1)
                HStack(spacing: 6) {
                    Text(model.quant.isEmpty ? "GGUF" : model.quant)
                    Text("·")
                    Text(ByteCountFormatter.string(fromByteCount: Int64(model.sizeBytes),
                                                   countStyle: .file))
                    Text("·")
                    FitBadge(verdict: plan.verdict)
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            Spacer()
            if loading == model.id {
                ProgressView()
            } else if isLoaded {
                Button("Chat") { chatting = true }
                    .buttonStyle(.borderedProminent)
            } else {
                Button("Load") { Task { await load(model) } }
                    .buttonStyle(.bordered)
                    .disabled(loading != nil)
            }
        }
        .swipeActions {
            if isLoaded {
                Button("Unload") { Task { await hub.inference.unload() } }
                    .tint(.orange)
            }
        }
    }

    private func load(_ model: InstalledModel) async {
        loading = model.id
        defer { loading = nil }
        do {
            try await hub.inference.load(model, from: store)
            chatting = true
        } catch {
            loadError = error.localizedDescription
        }
    }

    // MARK: - Search and download

    @ViewBuilder
    private var searchSection: some View {
        if searching {
            Section { ProgressView("Searching…") }
        } else if let searchError {
            Section { Text(searchError).foregroundStyle(.red) }
        } else if !results.isEmpty {
            Section("Results") {
                ForEach(results) { summary in
                    NavigationLink {
                        CandidateListView(summary: summary)
                    } label: {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(summary.name).lineLimit(1)
                            HStack(spacing: 6) {
                                Text(summary.owner)
                                if summary.requiresLicenceAcceptance {
                                    Text("·")
                                    Label("licence", systemImage: "lock")
                                        .labelStyle(.titleAndIcon)
                                }
                            }
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
    }

    private func runSearch() async {
        let text = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        searching = true
        searchError = nil
        defer { searching = false }
        do {
            results = try await hub.search.search(text)
            if results.isEmpty { searchError = "No GGUF repositories match \"\(text)\"." }
        } catch {
            searchError = error.localizedDescription
        }
    }
}

/// The GGUF files in one repository, each with its fit and a download.
private struct CandidateListView: View {
    let summary: HFModelSummary
    @EnvironmentObject private var hub: OnDeviceHub
    @ObservedObject private var store = ModelStore.shared
    @State private var candidates: [GGUFCandidate] = []
    @State private var failure: String?
    @State private var diskError: String?

    var body: some View {
        List {
            if summary.requiresLicenceAcceptance {
                Section {
                    Text("This model is gated. Accept its licence on huggingface.co and add a read token in On-device settings, or the download is refused.")
                        .font(.callout)
                }
            }
            if let failure {
                Section { Text(failure).foregroundStyle(.red) }
            }
            ForEach(candidates) { candidate in
                row(candidate)
            }
        }
        .navigationTitle(summary.name)
        .task { await loadDetail() }
        .alert("Not enough space", isPresented: Binding(
            get: { diskError != nil }, set: { if !$0 { diskError = nil } }
        )) {
            Button("OK", role: .cancel) { diskError = nil }
        } message: { Text(diskError ?? "") }
    }

    private func row(_ candidate: GGUFCandidate) -> some View {
        let shape = ModelShape(parameters: 0, quant: candidate.resolvedQuant, layers: 0,
                               kvHeads: 0, headDim: 0, fileBytes: candidate.sizeBytes,
                               name: candidate.filename)
        let plan = ModelFit.plan(shape: shape, memory: DeviceMemory.current(),
                                 context: hub.settings.contextLength,
                                 kvBits: hub.settings.kvCacheBits,
                                 backend: hub.settings.backend)
        let state = store.state[candidate.id] ?? .idle
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(candidate.filename).font(.callout).lineLimit(2)
                Spacer()
                FitBadge(verdict: plan.verdict)
            }
            HStack {
                Text(ByteCountFormatter.string(fromByteCount: Int64(candidate.sizeBytes),
                                               countStyle: .file))
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                controls(candidate, state: state)
            }
            if case .downloading = state {
                ProgressView(value: state.fraction)
            }
            if case .failed(let why) = state {
                Text(why).font(.caption).foregroundStyle(.red)
            }
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder
    private func controls(_ candidate: GGUFCandidate, state: DownloadState) -> some View {
        switch state {
        case .idle, .failed:
            Button("Download") { start(candidate) }.buttonStyle(.bordered)
        case .downloading:
            Button("Pause") { store.pause(candidate.id) }.buttonStyle(.bordered)
        case .paused:
            HStack {
                Button("Resume") { start(candidate) }.buttonStyle(.bordered)
                Button("Cancel", role: .destructive) { store.cancel(candidate.id) }
            }
        case .verifying:
            ProgressView()
        case .installed:
            Label("Downloaded", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
        }
    }

    private func start(_ candidate: GGUFCandidate) {
        do {
            try store.download(candidate)
        } catch {
            diskError = error.localizedDescription
        }
    }

    private func loadDetail() async {
        do {
            let detail = try await hub.search.detail(repoID: summary.id)
            candidates = detail.ggufCandidates
            if candidates.isEmpty {
                failure = "This repository has no GGUF files. Look for one whose name ends in -GGUF."
            }
        } catch {
            failure = error.localizedDescription
        }
    }
}

struct FitBadge: View {
    let verdict: FitVerdict

    var body: some View {
        switch verdict {
        case .runs: Label("fits", systemImage: "checkmark.circle").foregroundStyle(.green)
        case .tight: Label("tight", systemImage: "exclamationmark.circle").foregroundStyle(.orange)
        case .no: Label("too big", systemImage: "xmark.circle").foregroundStyle(.red)
        }
    }
}
