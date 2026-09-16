//  HardwareView.swift
//  What the PC is actually doing.
//
//  The question this answers is "is the server busy, or is my model just
//  slow?", which from six hundred miles away cannot be answered any
//  other way. The dashboards have sampled all of this for releases; this
//  is the same sampling with no terminal attached.
//
//  Admin or partial admin, server-side — it is a description of
//  somebody's hardware. An ordinary phone gets a 403 and this screen
//  says so plainly rather than showing an empty dashboard.
//
//  Nothing here invents a zero
//  ---------------------------
//  Every reading is optional and the server sends an `unavailable` list
//  naming what it could not read. A panel that renders a missing GPU
//  temperature as 0°C is worse than one that says it does not know: the
//  first is a confident wrong answer about hardware you cannot see.

import SwiftUI

struct HardwareView: View {
    @Environment(AppState.self) private var state
    @State private var snapshot: ServerHardware?
    @State private var refusal: String?
    @State private var loading = false

    var body: some View {
        List {
            if let refusal {
                Section {
                    ContentUnavailableView {
                        Label("Not available to you", systemImage: "lock")
                    } description: {
                        Text(refusal)
                    }
                }
            } else if let snapshot {
                readings(snapshot)
            } else if loading {
                Section { HStack { ProgressView(); Text("Sampling…") } }
            }
        }
        .navigationTitle("Hardware")
        .refreshable { await load() }
        .task { await load() }
    }

    @ViewBuilder
    private func readings(_ snapshot: ServerHardware) -> some View {
        Section("Machine") {
            if !snapshot.hostname.isEmpty {
                LabeledContent("Host", value: snapshot.hostname)
            }
            if !snapshot.platform.isEmpty {
                LabeledContent("Platform", value: snapshot.platform)
            }
            LabeledContent(
                "Up for", value: ServerUptime.describe(snapshot.uptimeSeconds)
            )
        }

        if let cpu = snapshot.cpu {
            Section("CPU") {
                if let percent = cpu.percent {
                    Meter(label: "In use", fraction: percent / 100)
                }
                if let count = cpu.coresLogical {
                    LabeledContent("Cores", value: "\(count)")
                }
                if let load = cpu.loadAverage, !load.isEmpty {
                    LabeledContent(
                        "Load average",
                        value: load.map { String(format: "%.2f", $0) }
                            .joined(separator: "  ")
                    )
                }
            }
        }

        if let memory = snapshot.memory {
            Section("Memory") {
                if let percent = memory.percent {
                    Meter(label: "In use", fraction: percent / 100)
                }
                LabeledContent("Total", value: bytes(memory.totalBytes))
                LabeledContent("Available", value: bytes(memory.availableBytes))
            }
        }

        if let swap = snapshot.swap, (swap.totalBytes ?? 0) > 0 {
            Section {
                if let percent = swap.percent {
                    Meter(label: "In use", fraction: percent / 100)
                }
                LabeledContent("Total", value: bytes(swap.totalBytes))
            } header: {
                Text("Swap")
            } footer: {
                Text("A model paging to swap is the usual reason a reply that was fast yesterday is slow today.")
            }
        }

        if !snapshot.gpus.isEmpty {
            Section("GPU") {
                ForEach(Array(snapshot.gpus.enumerated()), id: \.offset) { _, gpu in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(gpu.name.flatMap { $0.isEmpty ? nil : $0 } ?? "GPU")
                        if let used = gpu.memoryUsedBytes, let total = gpu.memoryTotalBytes,
                           total > 0 {
                            Meter(
                                label: "\(bytes(used)) of \(bytes(total))",
                                fraction: Double(used) / Double(total)
                            )
                        }
                        HStack(spacing: 8) {
                            if let busy = gpu.utilizationPercent {
                                Text("\(Int(busy))% busy")
                            }
                            if let temperature = gpu.temperatureC {
                                Text("\(Int(temperature))°C")
                            }
                        }
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    }
                }
            }
        }

        if !snapshot.disks.isEmpty {
            Section("Disks") {
                ForEach(Array(snapshot.disks.enumerated()), id: \.offset) { _, disk in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(disk.mount ?? "disk").font(.caption).lineLimit(1)
                        if let percent = disk.percent {
                            Meter(label: bytes(disk.freeBytes) + " free",
                                  fraction: percent / 100)
                        }
                    }
                }
            }
        }

        if !snapshot.unavailable.isEmpty {
            Section {
                ForEach(snapshot.unavailable, id: \.self) { missing in
                    Text(missing).font(.caption).foregroundStyle(.secondary)
                }
            } header: {
                Text("Could not be read")
            } footer: {
                Text("Named rather than shown as zero. A dashboard that reports an unreadable sensor as 0 is a confident wrong answer.")
            }
        }
    }

    private func bytes(_ value: Int?) -> String {
        guard let value, value > 0 else { return "—" }
        return ByteCountFormatter.string(fromByteCount: Int64(value), countStyle: .memory)
    }

    /// @MainActor because it mutates `@State`. `View.body` carries the
    /// annotation; the rest of the struct does not, so an async helper
    /// is nonisolated unless it says otherwise.
    @MainActor
    private func load() async {
        loading = true
        defer { loading = false }
        do {
            snapshot = try await state.hardware()
            refusal = nil
        } catch {
            snapshot = nil
            refusal = (error as? HyperLinkError)?.errorDescription
                ?? error.localizedDescription
        }
    }
}

/// A labelled bar. Clamped, because a server that reports 103% should
/// not draw outside its row.
private struct Meter: View {
    let label: String
    let fraction: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label).font(.caption)
                Spacer()
                Text("\(Int((min(max(fraction, 0), 1)) * 100))%")
                    .font(.caption)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            ProgressView(value: min(max(fraction, 0), 1))
                .tint(fraction > 0.9 ? .orange : .accentColor)
        }
    }
}
