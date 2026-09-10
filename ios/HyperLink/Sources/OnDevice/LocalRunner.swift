//  LocalRunner.swift
//  Running a GGUF on the phone.
//
//  WHAT IS AND IS NOT HERE
//
//  Everything above llama.cpp is here and works: the memory guard, the
//  load and unload lifecycle, the pressure response, the settings, the
//  streaming interface the UI talks to, and a runner that answers
//  without a model so the whole path can be exercised.
//
//  llama.cpp itself is not. Linking it means adding a native target to
//  ios/project.yml that builds ggml with Metal for arm64-apple-ios, and
//  neither that build nor this file has been compiled — there is no
//  Xcode or Swift toolchain in the environment this was written in. The
//  interface below is what `LlamaRunner` must satisfy; see
//  wiki/HyperLink-OnDevice.md for the build wiring, and treat the
//  native side as specified rather than as done.
//
//  The split is deliberate rather than an excuse: the memory guard is
//  the part that decides whether the app survives, it is testable
//  without a model, and it is mirrored by
//  hypernix/hyperlink/ondevice.py with tests against real file sizes.

import Foundation

/// One token as it arrives.
struct GeneratedToken: Sendable {
    let text: String
    let isFinal: Bool
}

/// What a runner must do. Implemented by `LlamaRunner` against
/// llama.cpp, and by `EchoRunner` for a build without it.
protocol ModelRunner: Actor {
    func load(url: URL, shape: ModelShape, settings: RunnerSettings) async throws
    func unload() async
    var isLoaded: Bool { get async }
    func generate(
        prompt: String, systemPrompt: String, maxTokens: Int
    ) -> AsyncThrowingStream<GeneratedToken, Error>
    func cancel() async
}

/// The knobs llama.cpp is given, resolved from `OnDeviceSettings`.
struct RunnerSettings: Sendable, Equatable {
    var threads: Int
    var contextLength: Int
    var kvCacheBits: Int
    var backend: ComputeBackend
    /// How many layers go to Metal. On unified memory this changes
    /// speed and not footprint — see ModelFit.swift.
    var gpuLayers: Int
}

enum LocalRunnerError: LocalizedError {
    case willNotFit(FitPlan)
    case notBuiltIn
    case loadFailed(String)
    case noModel

    var errorDescription: String? {
        switch self {
        case .willNotFit(let plan):
            let need = ByteCountFormatter.string(
                fromByteCount: Int64(plan.totalBytes), countStyle: .memory
            )
            let have = ByteCountFormatter.string(
                fromByteCount: Int64(plan.availableBytes), countStyle: .memory
            )
            return "This model needs about \(need) and this app has \(have) to work "
                + "with. " + (plan.notes.first ?? "")
        case .notBuiltIn:
            return "This build has no local inference engine. On-device models "
                + "need a build with llama.cpp linked in."
        case .loadFailed(let detail):
            return "Could not load the model: \(detail)"
        case .noModel:
            return "No model is loaded."
        }
    }
}

/// Picks the engine this build actually has.
///
/// `HNX_LOCAL_LLAMA` is set by ios/vendor/LocalLlama.xcconfig, which
/// ios/scripts/build_llama_xcframework.sh writes once it has produced
/// the framework. Without it `LlamaRunner` is not compiled at all and
/// this returns the runner that explains why, rather than the app
/// failing to link.
enum RunnerFactory {
    static func make() -> any ModelRunner {
        #if HNX_LOCAL_LLAMA
        return LlamaRunner()
        #else
        return EchoRunner()
        #endif
    }

    /// Whether this build can actually run a model, for a settings
    /// screen that should say so rather than offering a download that
    /// leads nowhere.
    static var hasLocalEngine: Bool {
        #if HNX_LOCAL_LLAMA
        return true
        #else
        return false
        #endif
    }
}

/// Owns the loaded model and refuses to load one that will not fit.
@MainActor
final class LocalInference: ObservableObject {
    @Published private(set) var loaded: InstalledModel?
    @Published private(set) var lastPlan: FitPlan?
    @Published private(set) var isBusy = false

    private let runner: any ModelRunner
    private let settings: OnDeviceSettings

    init(runner: (any ModelRunner)? = nil, settings: OnDeviceSettings) {
        self.runner = runner ?? RunnerFactory.make()
        self.settings = settings
    }

    /// False in a build without the engine. The UI uses this to say so
    /// up front instead of letting someone download four gigabytes and
    /// then discover nothing can run it.
    var hasLocalEngine: Bool { RunnerFactory.hasLocalEngine }

    /// Check fit against the budget *now*, not when the list was drawn.
    ///
    /// The budget shrinks when other apps run, so a model that fit five
    /// minutes ago may not fit at the moment the user taps it. Re-asking
    /// immediately before the load is the difference between a refusal
    /// and a process kill.
    func plan(for model: InstalledModel) -> FitPlan {
        let memory = DeviceMemory.current()
        let shape = model.shape?.modelShape(
            quant: model.quant, fileBytes: model.sizeBytes, name: model.displayName
        ) ?? ModelShape(
            parameters: 0, quant: model.quant, layers: 0, kvHeads: 0, headDim: 0,
            fileBytes: model.sizeBytes, name: model.displayName
        )
        return ModelFit.plan(
            shape: shape, memory: memory,
            context: settings.contextLength, kvBits: settings.kvCacheBits,
            backend: settings.backend
        )
    }

    func load(_ model: InstalledModel, from store: ModelStore) async throws {
        let fit = plan(for: model)
        lastPlan = fit
        if fit.verdict == .no, settings.enforceMemoryCheck {
            throw LocalRunnerError.willNotFit(fit)
        }

        // Only one model at a time. Loading a second while the first is
        // resident is the fastest way to be killed, and on a phone
        // there is no case where two is better than one.
        await runner.unload()
        loaded = nil

        let memory = DeviceMemory.current()
        let shape = model.shape?.modelShape(
            quant: model.quant, fileBytes: model.sizeBytes, name: model.displayName
        ) ?? ModelShape(
            parameters: 0, quant: model.quant, layers: 0, kvHeads: 0, headDim: 0,
            fileBytes: model.sizeBytes, name: model.displayName
        )
        let resolved = RunnerSettings(
            threads: settings.resolvedThreads(memory: memory),
            contextLength: settings.contextLength,
            kvCacheBits: settings.kvCacheBits,
            backend: settings.backend,
            gpuLayers: settings.backend == .cpu ? 0 : shape.layers
        )
        try await runner.load(url: store.url(for: model), shape: shape, settings: resolved)
        loaded = model
    }

    func unload() async {
        await runner.unload()
        loaded = nil
    }

    func generate(prompt: String) -> AsyncThrowingStream<GeneratedToken, Error> {
        runner.generate(
            prompt: prompt,
            systemPrompt: settings.systemPrompt,
            maxTokens: 2048
        )
    }

    func cancel() async { await runner.cancel() }

    /// Drop the model when the system says memory is short.
    ///
    /// A resident model is by far the largest thing the app holds, so
    /// it is the right thing to give back — and giving it back
    /// voluntarily is much better than being killed, which loses the
    /// conversation too.
    func handleMemoryPressure() async {
        await unload()
    }

    /// Stop generating when the app goes to the background.
    func handleBackgrounding() async {
        guard settings.pauseInBackground else { return }
        await cancel()
    }
}

/// A runner for a build without llama.cpp.
///
/// Not a mock in a test target: it ships, so that a build lacking the
/// native engine degrades to a clear message in the UI rather than a
/// link error or a crash on first use.
actor EchoRunner: ModelRunner {
    private var model: InstalledModelStub?

    private struct InstalledModelStub {
        let url: URL
        let shape: ModelShape
    }

    var isLoaded: Bool { model != nil }

    func load(url: URL, shape: ModelShape, settings: RunnerSettings) async throws {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw LocalRunnerError.loadFailed("no file at \(url.lastPathComponent)")
        }
        model = InstalledModelStub(url: url, shape: shape)
    }

    func unload() async { model = nil }

    func cancel() async {}

    func generate(
        prompt: String, systemPrompt: String, maxTokens: Int
    ) -> AsyncThrowingStream<GeneratedToken, Error> {
        AsyncThrowingStream { continuation in
            guard model != nil else {
                continuation.finish(throwing: LocalRunnerError.noModel)
                return
            }
            continuation.yield(
                GeneratedToken(
                    text: LocalRunnerError.notBuiltIn.errorDescription ?? "",
                    isFinal: false
                )
            )
            continuation.yield(GeneratedToken(text: "", isFinal: true))
            continuation.finish()
        }
    }
}
