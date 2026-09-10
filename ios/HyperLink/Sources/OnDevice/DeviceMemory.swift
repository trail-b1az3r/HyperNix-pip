//  DeviceMemory.swift
//  What this process may actually use — which is not the device's RAM.
//
//  This is the single most important file in on-device inference, and
//  the one where the obvious answer is wrong.
//
//  `ProcessInfo.processInfo.physicalMemory` returns the RAM the phone
//  has: 8 GB on an iPhone 15 Pro. An app may not use it. iOS gives each
//  process a jetsam limit well below that — commonly 2-3 GB for a
//  normal app on an 8 GB device — and exceeding it is not a swap, not a
//  slowdown, and not an exception. The process is killed. There is no
//  `catch` for it and no warning first.
//
//  So a "do you have enough RAM?" check written against physicalMemory
//  tells the user a 5 GB model fits, downloads it over twenty minutes
//  of cellular, and then dies partway through the first reply. The
//  number that matters is `os_proc_available_memory()`, which reports
//  what remains of *this process's* limit right now.
//
//  Mirrors hypernix/hyperlink/ondevice.py, which carries the same
//  arithmetic with tests against published model file sizes. When the
//  two disagree, that module is the reference.

import Foundation
import os

/// The memory this process may still use, and the device it is on.
struct DeviceMemory: Sendable, Equatable {
    /// From `os_proc_available_memory()`. The real budget.
    let availableBytes: Int
    /// From `ProcessInfo.physicalMemory`. Reported so the UI can
    /// explain the gap, never used to decide whether a model fits.
    let totalRAMBytes: Int
    /// Whether `com.apple.developer.kernel.increased-memory-limit` was
    /// *found* in the embedded provisioning profile.
    ///
    /// `false` means "not found" — App Store builds and the simulator
    /// carry no profile — so it is never evidence of absence. Recorded
    /// for reporting only: an estimate must never be inflated by a
    /// limit the process may not have been granted.
    let hasIncreasedLimit: Bool
    /// Performance cores. Efficiency cores are excluded deliberately —
    /// llama.cpp scheduling work onto them costs more in scheduler
    /// churn than the cores contribute.
    let performanceCores: Int
    let deviceModel: String

    static func current() -> DeviceMemory {
        DeviceMemory(
            availableBytes: Int(os_proc_available_memory()),
            totalRAMBytes: Int(ProcessInfo.processInfo.physicalMemory),
            hasIncreasedLimit: Self.entitlementFound,
            performanceCores: Self.performanceCoreCount,
            deviceModel: Self.hardwareIdentifier
        )
    }

    /// A snapshot taken again, for a caller watching pressure change.
    ///
    /// The budget is not static: it shrinks when other apps are
    /// running, so a model that fit when the user opened the list may
    /// not fit when they tap it. Re-checking immediately before a load
    /// is the difference between a refusal and a kill.
    func refreshed() -> DeviceMemory {
        DeviceMemory(
            availableBytes: Int(os_proc_available_memory()),
            totalRAMBytes: totalRAMBytes,
            hasIncreasedLimit: hasIncreasedLimit,
            performanceCores: performanceCores,
            deviceModel: deviceModel
        )
    }

    /// The gap the UI has to explain when it refuses something.
    var explainsGap: Bool { totalRAMBytes > availableBytes * 2 }

    /// Whether the increased-memory-limit entitlement could be *found*.
    ///
    /// Not "whether it is enabled". The distinction is real and this
    /// cannot close it.
    ///
    /// The obvious API — `SecTaskCreateFromSelf` and
    /// `SecTaskCopyValueForEntitlement` — is macOS-only. On iOS those
    /// are private SPI and not in scope, which is a compile error, not
    /// a runtime one:
    ///
    ///     error: cannot find 'SecTaskCreateFromSelf' in scope
    ///
    /// What iOS does offer is the embedded provisioning profile, which
    /// carries the entitlements the build was signed with. That is a
    /// real signal for development, ad-hoc and enterprise builds — and
    /// absent from App Store builds and usually from the simulator, so
    /// `false` means "not found", never "definitely not granted".
    ///
    /// Which is why nothing depends on it. It is reported, and the
    /// memory estimate is identical either way; a planner that gave
    /// itself headroom on the strength of this would be trusting a
    /// signal that is missing exactly where the app is most constrained.
    private static var entitlementFound: Bool {
        guard
            let url = Bundle.main.url(
                forResource: "embedded", withExtension: "mobileprovision"
            ),
            let raw = try? Data(contentsOf: url)
        else { return false }

        // The profile is CMS-signed, with the plist embedded as plain
        // XML inside the signature envelope. Slicing between the
        // markers is the documented-by-practice way to read it; there
        // is no public API that unwraps it.
        guard
            let start = raw.range(of: Data("<?xml".utf8)),
            let end = raw.range(of: Data("</plist>".utf8))
        else { return false }

        // Re-wrapped in a fresh Data: a slice keeps the parent's index
        // base, and PropertyListSerialization reads from zero.
        let plistData = Data(raw[start.lowerBound..<end.upperBound])
        guard
            let plist = try? PropertyListSerialization.propertyList(
                from: plistData, options: [], format: nil
            ) as? [String: Any],
            let entitlements = plist["Entitlements"] as? [String: Any]
        else { return false }

        return entitlements[
            "com.apple.developer.kernel.increased-memory-limit"
        ] as? Bool ?? false
    }

    private static var performanceCoreCount: Int {
        var count: Int32 = 0
        var size = MemoryLayout<Int32>.size
        // `hw.perflevel0.logicalcpu` is the performance cluster on
        // Apple silicon. On a device that does not report it, fall back
        // to activeProcessorCount minus the efficiency cluster, and
        // then to a conservative half.
        if sysctlbyname("hw.perflevel0.logicalcpu", &count, &size, nil, 0) == 0, count > 0 {
            return Int(count)
        }
        let all = ProcessInfo.processInfo.activeProcessorCount
        return max(1, all / 2)
    }

    private static var hardwareIdentifier: String {
        var info = utsname()
        uname(&info)
        return withUnsafePointer(to: &info.machine) { pointer in
            pointer.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) }
        }
    }
}
