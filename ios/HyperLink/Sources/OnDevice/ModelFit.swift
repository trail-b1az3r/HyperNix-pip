//  ModelFit.swift
//  Will this model run on this phone? Decided before the download.
//
//  A mirror of hypernix/hyperlink/ondevice.py. That module is the
//  reference: it carries the same constants and arithmetic with tests
//  against published Hugging Face file sizes, and
//  tests/test_hyperlink_ondevice_mirror.py asserts the constants here
//  still match it. Change one and the test says so.
//
//  Three things make the obvious answer wrong, and all three are here
//  rather than in a comment somewhere:
//
//  1. Total RAM is not the budget — see DeviceMemory.swift.
//  2. The KV cache is not small, and at long contexts exceeds the
//     weights.
//  3. On Apple silicon, moving layers to the GPU does not reduce
//     memory. Unified memory means a Metal buffer and a malloc come
//     from the same pool.

import Foundation

/// Backends that exist for GGUF on an iPhone.
///
/// There is no Neural Engine case, and its absence is deliberate rather
/// than an omission. See `ModelFit.aneExplanation`.
enum ComputeBackend: String, CaseIterable, Codable, Sendable {
    /// Accelerate/AMX on the performance cores.
    case cpu
    /// The GPU, via ggml-metal.
    case metal
    /// Metal for most layers, CPU for the rest.
    case hybrid

    var displayName: String {
        switch self {
        case .cpu: "CPU only"
        case .metal: "Metal (GPU)"
        case .hybrid: "Metal + CPU"
        }
    }

    var detail: String {
        switch self {
        case .cpu:
            "Runs on the performance cores through Accelerate. Slowest, and "
            + "the most predictable under thermal pressure."
        case .metal:
            "Runs on the GPU. Fastest on Apple silicon. Uses the same memory "
            + "pool as everything else, so it costs no extra RAM and saves none."
        case .hybrid:
            "Most layers on the GPU, the rest on the CPU. For a model that "
            + "nearly fits the GPU's working set."
        }
    }
}

enum FitVerdict: String, Codable, Sendable {
    case runs
    case tight
    case no

    var isUsable: Bool { self != .no }
}

/// A model's shape, from its GGUF metadata or a search listing.
struct ModelShape: Sendable, Equatable {
    var parameters: Int
    var quant: String
    var layers: Int
    /// The *key/value* head count, not the attention head count. A
    /// grouped-query model shares each KV head across several attention
    /// heads, so using the larger number overestimates the cache by the
    /// GQA ratio — 4x on Llama 3 — and refuses models that run fine.
    var kvHeads: Int
    var headDim: Int
    var trainContext: Int = 0
    /// The real file size, once known. Preferred over arithmetic.
    var fileBytes: Int = 0
    var vocabSize: Int = 0
    var embeddingDim: Int = 0
    var tiedEmbeddings: Bool = false
    var name: String = ""

    /// Bytes of weights.
    ///
    /// Not simply bits times parameters: llama.cpp keeps the embedding
    /// and output tensors at a higher precision than the file's name
    /// suggests, and for a small model those are a large fraction of
    /// it. Llama-3.2-1B has a 128k vocabulary over 2048 dimensions —
    /// 21% of its parameters — and sizing it flat under-counts by 8%.
    /// Under-counting is the direction that gets the process killed.
    var weightsBytes: Int {
        if fileBytes > 0 { return fileBytes }
        let bits = ModelFit.quantBits(quant)
        guard bits > 0, parameters > 0 else { return 0 }

        let totalBits: Double
        if vocabSize > 0, embeddingDim > 0, bits < ModelFit.embeddingTensorBits {
            let table = vocabSize * embeddingDim
            let special = min(table * (tiedEmbeddings ? 1 : 2), parameters)
            let rest = parameters - special
            totalBits = Double(special) * ModelFit.embeddingTensorBits + Double(rest) * bits
        } else {
            totalBits = Double(parameters) * bits
        }
        return Int(totalBits / 8 * ModelFit.sizeSafetyFactor)
    }

    /// Bytes the key/value cache needs for `context` tokens.
    ///
    /// `2` for the two caches, K and V. `kvBits` of 8 is llama.cpp's
    /// quantised cache, which roughly halves this and is often what
    /// makes a long context possible on a phone at all.
    func kvCacheBytes(context: Int, kvBits: Int = 16) -> Int {
        guard context > 0, layers > 0, kvHeads > 0, headDim > 0 else { return 0 }
        let perToken = 2 * layers * kvHeads * headDim * kvBits / 8
        return perToken * context
    }
}

/// What a fit check concluded, and why.
struct FitPlan: Sendable {
    let verdict: FitVerdict
    let context: Int
    let weightsBytes: Int
    let kvBytes: Int
    let overheadBytes: Int
    let totalBytes: Int
    let availableBytes: Int
    let backend: ComputeBackend
    let warnings: [String]
    let notes: [String]

    var headroomBytes: Int { availableBytes - totalBytes }
    var utilisation: Double {
        availableBytes > 0 ? Double(totalBytes) / Double(availableBytes) : .infinity
    }
}

enum ModelFit {
    /// Fraction of the budget a plan may use and still call itself
    /// "runs". Being 10% optimistic on a desktop means swapping; here
    /// it means the OS kills the process mid-sentence.
    static let safetyMargin = 0.85
    static let tightMargin = 0.95
    static let baseOverheadBytes = 320 * 1024 * 1024
    static let overheadBytesPerContextToken = 2048
    static let embeddingTensorBits = 6.56
    static let sizeSafetyFactor = 1.02

    /// The honest answer for anyone looking for a Neural Engine switch.
    static let aneExplanation = """
        The Neural Engine cannot run GGUF models. It is reachable only \
        through Core ML, and llama.cpp's Apple backend is Metal (GPU) with \
        Accelerate on the CPU path — there is no Core ML backend for LLM \
        inference. Running on the ANE would mean converting the model to \
        Core ML, which is a different file in a different format, not a \
        setting. Metal is the acceleration that exists here, and on Apple \
        silicon it is fast.
        """

    /// Effective bits per weight, including the scale and zero-point
    /// overhead that makes a "4-bit" model never 0.5 bytes per weight.
    static let quantTable: [String: Double] = [
        "IQ1_S": 1.56, "IQ1_M": 1.75, "IQ2_XXS": 2.06, "IQ2_XS": 2.31,
        "IQ2_S": 2.50, "IQ2_M": 2.70, "IQ3_XXS": 3.06, "IQ3_XS": 3.30,
        "IQ3_S": 3.44, "IQ3_M": 3.66, "IQ4_XS": 4.25, "IQ4_NL": 4.50,
        "Q2_K": 3.35, "Q3_K_S": 3.50, "Q3_K_M": 3.91, "Q3_K_L": 3.44,
        "Q4_0": 4.55, "Q4_1": 5.06, "Q4_K_S": 4.58, "Q4_K_M": 4.83,
        "Q5_0": 5.54, "Q5_1": 6.06, "Q5_K_S": 5.52, "Q5_K_M": 5.67,
        "Q6_K": 6.56, "Q8_0": 8.50,
        "F16": 16.0, "FP16": 16.0, "BF16": 16.0, "F32": 32.0, "FP32": 32.0,
    ]

    static func quantBits(_ name: String) -> Double {
        quantTable[name.uppercased()] ?? 0
    }

    /// The quantisation a GGUF filename advertises, or "".
    ///
    /// Hugging Face names files by convention rather than by rule, so
    /// this matches known names as whole tokens rather than parsing a
    /// structure that does not exist. Longest first, or `Q4_K_M` gets
    /// matched as `Q4_K` and carries the wrong bit width.
    static func quantFromFilename(_ filename: String) -> String {
        let upper = filename.uppercased()
        let boundary = CharacterSet.alphanumerics
        for name in quantTable.keys.sorted(by: { $0.count > $1.count }) {
            var searchStart = upper.startIndex
            while let found = upper.range(of: name, range: searchStart..<upper.endIndex) {
                let beforeOK = found.lowerBound == upper.startIndex
                    || !boundary.contains(upper.unicodeScalars[
                        upper.index(before: found.lowerBound)])
                let afterOK = found.upperBound == upper.endIndex
                    || !boundary.contains(upper.unicodeScalars[found.upperBound])
                if beforeOK && afterOK { return name }
                searchStart = found.upperBound
            }
        }
        return ""
    }

    static func runtimeOverheadBytes(context: Int) -> Int {
        baseOverheadBytes + max(0, context) * overheadBytesPerContextToken
    }

    /// Decide whether `shape` runs on `memory`, and say why not.
    static func plan(
        shape: ModelShape,
        memory: DeviceMemory,
        context: Int = 4096,
        kvBits: Int = 16,
        backend: ComputeBackend = .metal
    ) -> FitPlan {
        let context = max(0, context)
        let weights = shape.weightsBytes
        let kv = shape.kvCacheBytes(context: context, kvBits: kvBits)
        let overhead = runtimeOverheadBytes(context: context)
        let total = weights + kv + overhead

        var warnings: [String] = []
        var notes: [String] = []

        if memory.availableBytes <= 0 {
            warnings.append(
                "The device did not report an available-memory figure, so this "
                + "is unchecked."
            )
        }
        if weights <= 0 {
            warnings.append(
                "No size for this model — neither a file size nor a recognised "
                + "quantisation. This is not a fit answer."
            )
        }
        if memory.totalRAMBytes > 0, memory.availableBytes > 0,
           total <= memory.totalRAMBytes, total > memory.availableBytes {
            notes.append(
                "This model fits the device's \(gib(memory.totalRAMBytes)) of RAM "
                + "and does not fit this app's \(gib(memory.availableBytes)) limit. "
                + "iOS gives each process a limit well below total RAM, and "
                + "exceeding it is an immediate kill rather than a slowdown."
            )
            if !memory.hasIncreasedLimit {
                notes.append(
                    "The com.apple.developer.kernel.increased-memory-limit "
                    + "entitlement raises that ceiling; it is not enabled here."
                )
            }
        }
        if shape.trainContext > 0, context > shape.trainContext {
            notes.append(
                "Asking for \(context) tokens on a model trained for "
                + "\(shape.trainContext). It will run, and quality past the "
                + "trained length degrades in a way this estimate cannot see."
            )
        }
        if backend != .cpu {
            notes.append(
                "Apple silicon has unified memory, so moving layers to Metal "
                + "does not reduce this total — it buys speed, not headroom."
            )
        }
        if weights > 0, kv >= weights / 2 {
            let comparison = kv > weights ? "larger than" : "already comparable to"
            notes.append(
                "The KV cache (\(gib(kv))) is \(comparison) the weights "
                + "(\(gib(weights))) at this context length. Halving the context, "
                + "or an 8-bit cache, saves more than a smaller quantisation "
                + "would and costs nothing in the weights."
            )
        }

        let verdict: FitVerdict
        if memory.availableBytes <= 0 {
            verdict = .no
        } else if Double(total) <= Double(memory.availableBytes) * safetyMargin {
            verdict = .runs
        } else if Double(total) <= Double(memory.availableBytes) * tightMargin {
            verdict = .tight
            warnings.append(
                "This leaves very little headroom. It will probably run and may "
                + "be killed if anything else on the phone wants memory."
            )
        } else {
            verdict = .no
        }

        return FitPlan(
            verdict: verdict, context: context, weightsBytes: weights, kvBytes: kv,
            overheadBytes: overhead, totalBytes: total,
            availableBytes: memory.availableBytes, backend: backend,
            warnings: warnings, notes: notes
        )
    }

    /// The longest context that still earns a "runs" verdict.
    ///
    /// Usually the more useful question than "does it fit": a model
    /// that does not fit at 32k frequently runs at 4k, and refusing it
    /// outright hides one that would have been fine.
    static func largestContext(
        shape: ModelShape,
        memory: DeviceMemory,
        kvBits: Int = 16,
        ceiling: Int = 131_072,
        granularity: Int = 256
    ) -> Int {
        guard memory.availableBytes > 0, shape.weightsBytes > 0 else { return 0 }
        guard plan(shape: shape, memory: memory, context: 0, kvBits: kvBits)
            .verdict != .no else { return 0 }

        var low = 0
        var high = max(granularity, ceiling)
        var best = 0
        while low <= high {
            let mid = (low + high) / 2
            if plan(shape: shape, memory: memory, context: mid, kvBits: kvBits)
                .verdict == .runs {
                best = mid
                low = mid + granularity
            } else {
                high = mid - granularity
            }
            if high - low < granularity, best > 0 { break }
        }
        if shape.trainContext > 0 { best = min(best, shape.trainContext) }
        return (best / granularity) * granularity
    }

    private static func gib(_ value: Int) -> String {
        let gigabyte = 1024 * 1024 * 1024
        if value >= gigabyte {
            return String(format: "%.1f GiB", Double(value) / Double(gigabyte))
        }
        return String(format: "%.0f MiB", Double(value) / (1024 * 1024))
    }
}
