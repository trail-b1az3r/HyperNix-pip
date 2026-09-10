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
    /// Whether `com.apple.developer.kernel.increased-memory-limit` is
    /// in the entitlements. Recorded for reporting only: an estimate
    /// must never be inflated by a limit the process may not have been
    /// granted at runtime.
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
            hasIncreasedLimit: Self.entitlementPresent,
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

    private static var entitlementPresent: Bool {
        // Read from the embedded provisioning profile rather than
        // assumed: the entitlement is granted per-application by Apple,
        // and a build that was not granted it will silently have a much
        // lower ceiling than a build that was.
        guard
            let task = SecTaskCreateFromSelf(nil),
            let value = SecTaskCopyValueForEntitlement(
                task, "com.apple.developer.kernel.increased-memory-limit" as CFString, nil
            ) as? Bool
        else { return false }
        return value
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
