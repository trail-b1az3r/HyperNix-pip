//  LlamaRunner.swift
//  ModelRunner, over llama.cpp.
//
//  Compiled only when the engine is vendored — see
//  ios/vendor/LocalLlama.xcconfig. Without it this file is empty and
//  LocalInference uses EchoRunner, which says so.
//
//  Written against include/llama.h at b10883, the ref
//  native/ggml-hnx/build.sh pins and ios/scripts/build_llama_xcframework.sh
//  clones. That matters more than usual here: llama.cpp's C API churns,
//  and several of the names a reader might expect are gone.
//
//    llama_load_model_from_file   -> llama_model_load_from_file
//    llama_new_context_with_model -> llama_init_from_model
//    llama_free_model             -> llama_model_free
//    llama_n_ctx_train(model)     -> llama_model_n_ctx_train(model)
//    params.use_mmap / use_mlock  -> params.load_mode
//    llama_kv_cache_clear(ctx)    -> llama_memory_clear(llama_get_memory(ctx), _)
//
//  Tokenizer calls take a `const llama_vocab *` from
//  llama_model_get_vocab(model), not the model.

#if HNX_LOCAL_LLAMA

import Foundation
import llama

/// `llama_backend_init()`, run once per process.
///
/// A global `let` with a side-effecting initialiser is Swift's
/// once-only idiom: lazily initialised, and the runtime guarantees a
/// single thread-safe initialisation.
private let llamaBackendReady: Bool = {
    llama_backend_init()
    return true
}()

/// llama.cpp, behind the `ModelRunner` protocol.
///
/// An actor because llama_context is not thread-safe and generation is
/// long-running: serialising through the actor is what stops a second
/// request from corrupting the first one's KV cache.
actor LlamaRunner: ModelRunner {
    private var model: OpaquePointer?
    private var context: OpaquePointer?
    private var sampler: UnsafeMutablePointer<llama_sampler>?
    private var vocab: OpaquePointer?
    private var cancelled = false
    private var loadedContextLength: Int32 = 0

    var isLoaded: Bool { model != nil && context != nil }

    // MARK: - Lifecycle

    func load(url: URL, shape: ModelShape, settings: RunnerSettings) async throws {
        await unload()

        // Touching the global runs its initialiser exactly once, and
        // Swift guarantees that is thread-safe. The previous version
        // was a mutable `static var` guard, which is shared mutable
        // state across actor instances -- the thing strict concurrency
        // exists to catch.
        _ = llamaBackendReady

        var mparams = llama_model_default_params()
        // Negative means every layer. On unified memory this changes
        // speed and not footprint — see ModelFit.swift — so a CPU-only
        // choice is about thermal predictability, not saving RAM.
        mparams.n_gpu_layers = settings.backend == .cpu ? 0 : Int32(-1)

        // mmap, so the weights are file-backed and evictable rather than
        // dirty anonymous pages. On iOS that is the difference between
        // pages the kernel can reclaim under pressure and pages that
        // count fully against the jetsam limit. Emphatically not MLOCK:
        // pinning gigabytes on a phone is the fastest way to be killed.
        mparams.load_mode = LLAMA_LOAD_MODE_MMAP

        // The "JIT" part: read the rows of tensors the architecture
        // marks on demand rather than pulling the whole tensor up front.
        // AUTO applies it only to tensors over 4 GiB, which is the safe
        // default — full ON is a per-model decision and this is not the
        // place to make it for someone.
        mparams.lazy_mode = LLAMA_LAZY_MODE_AUTO

        let path = url.path
        guard let loaded = path.withCString({ llama_model_load_from_file($0, mparams) }) else {
            throw LocalRunnerError.loadFailed(
                "llama.cpp could not open \(url.lastPathComponent). If it was "
                + "downloaded recently the file may be truncated."
            )
        }
        model = loaded
        vocab = llama_model_get_vocab(loaded)

        var cparams = llama_context_default_params()
        // Clamped to what the model was trained for. Asking for more
        // than that allocates a cache the model cannot use and is a
        // straightforward way to be killed for nothing.
        let trained = llama_model_n_ctx_train(loaded)
        let wanted = Int32(max(256, settings.contextLength))
        loadedContextLength = trained > 0 ? min(wanted, trained) : wanted
        cparams.n_ctx = UInt32(loadedContextLength)
        cparams.n_batch = UInt32(min(512, loadedContextLength))
        cparams.n_threads = Int32(settings.threads)
        cparams.n_threads_batch = Int32(settings.threads)
        if settings.kvCacheBits == 8 {
            // Halves the cache, which at long contexts saves more than
            // a smaller quantisation would and costs nothing in the
            // weights.
            cparams.type_k = GGML_TYPE_Q8_0
            cparams.type_v = GGML_TYPE_Q8_0
        }
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO

        guard let ctx = llama_init_from_model(loaded, cparams) else {
            llama_model_free(loaded)
            model = nil
            vocab = nil
            throw LocalRunnerError.loadFailed(
                "could not create a context at \(loadedContextLength) tokens — "
                + "usually not enough memory"
            )
        }
        context = ctx
        sampler = Self.makeSampler()
    }

    func unload() async {
        if let sampler { llama_sampler_free(sampler) }
        sampler = nil
        if let context { llama_free(context) }
        context = nil
        if let model { llama_model_free(model) }
        model = nil
        vocab = nil
        loadedContextLength = 0
    }

    func cancel() async { cancelled = true }

    private static func makeSampler() -> UnsafeMutablePointer<llama_sampler>? {
        var sparams = llama_sampler_chain_default_params()
        sparams.no_perf = true
        guard let chain = llama_sampler_chain_init(sparams) else { return nil }
        // Order matters: the truncating samplers narrow the candidate
        // set, then temperature scales it, then dist draws. Putting
        // temp first would scale a distribution that top_p is about to
        // cut, which is not what either is documented to do.
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(40))
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(0.95, 1))
        llama_sampler_chain_add(chain, llama_sampler_init_temp(0.7))
        llama_sampler_chain_add(chain, llama_sampler_init_dist(LLAMA_DEFAULT_SEED))
        return chain
    }

    // MARK: - Generation

    nonisolated func generate(
        prompt: String, systemPrompt: String, maxTokens: Int
    ) -> AsyncThrowingStream<GeneratedToken, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    try await self.run(
                        prompt: prompt, systemPrompt: systemPrompt,
                        maxTokens: maxTokens, into: continuation
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func run(
        prompt: String,
        systemPrompt: String,
        maxTokens: Int,
        into continuation: AsyncThrowingStream<GeneratedToken, Error>.Continuation
    ) throws {
        guard let context, let vocab, let sampler else {
            throw LocalRunnerError.noModel
        }
        cancelled = false

        // Start from a clean cache. Carrying the previous turn's KV
        // across without its tokens produces confident nonsense, which
        // is much harder to diagnose than an empty context.
        llama_memory_clear(llama_get_memory(context), true)

        let text = formatted(prompt: prompt, systemPrompt: systemPrompt)
        var tokens = try tokenize(text, vocab: vocab, addSpecial: true)
        guard !tokens.isEmpty else { return }
        guard tokens.count < Int(loadedContextLength) else {
            throw LocalRunnerError.loadFailed(
                "the prompt is \(tokens.count) tokens and the context is "
                + "\(loadedContextLength). Raise the context length in "
                + "Settings, or shorten the conversation."
            )
        }

        var batch = tokens.withUnsafeMutableBufferPointer {
            llama_batch_get_one($0.baseAddress, Int32($0.count))
        }
        guard llama_decode(context, batch) == 0 else {
            throw LocalRunnerError.loadFailed("llama_decode failed on the prompt")
        }

        var produced = 0
        // Bytes rather than a String: token_to_piece emits UTF-8 that
        // can split a character across two tokens, and decoding each
        // piece alone turns an emoji into two replacement characters.
        var pending = [UInt8]()

        while produced < maxTokens, !cancelled {
            let next = llama_sampler_sample(sampler, context, -1)
            if llama_vocab_is_eog(vocab, next) { break }
            llama_sampler_accept(sampler, next)

            pending.append(contentsOf: piece(of: next, vocab: vocab))
            if let decoded = String(bytes: pending, encoding: .utf8) {
                pending.removeAll(keepingCapacity: true)
                if !decoded.isEmpty {
                    continuation.yield(GeneratedToken(text: decoded, isFinal: false))
                }
            }
            // If it did not decode, `pending` is holding an incomplete
            // sequence and the next token completes it.

            produced += 1
            if produced + tokens.count >= Int(loadedContextLength) { break }

            var one = next
            batch = withUnsafeMutablePointer(to: &one) {
                llama_batch_get_one($0, 1)
            }
            guard llama_decode(context, batch) == 0 else { break }
        }

        continuation.yield(GeneratedToken(text: "", isFinal: true))
    }

    /// Apply the model's own chat template when it has one.
    ///
    /// A GGUF usually carries the template it was trained with, and
    /// using it is the difference between a coherent assistant and one
    /// that ignores the system prompt. Falling back to a plain
    /// concatenation is for the models that ship without one.
    private func formatted(prompt: String, systemPrompt: String) -> String {
        guard let model, let template = llama_model_chat_template(model, nil) else {
            return systemPrompt.isEmpty ? prompt : systemPrompt + "\n\n" + prompt
        }

        var storage: [UnsafeMutablePointer<CChar>] = []
        defer { storage.forEach { free($0) } }
        func hold(_ value: String) -> UnsafePointer<CChar> {
            let copy = strdup(value)!
            storage.append(copy)
            return UnsafePointer(copy)
        }

        var messages: [llama_chat_message] = []
        if !systemPrompt.isEmpty {
            messages.append(
                llama_chat_message(role: hold("system"), content: hold(systemPrompt))
            )
        }
        messages.append(llama_chat_message(role: hold("user"), content: hold(prompt)))

        // Sized from the input rather than a fixed buffer: a long
        // conversation overflows a fixed one, and the API reports the
        // needed length as a negative return that has to be honoured.
        var buffer = [CChar](repeating: 0, count: max(4096, (prompt.utf8.count + systemPrompt.utf8.count) * 2))
        var written = messages.withUnsafeBufferPointer { messagePointer in
            llama_chat_apply_template(
                template, messagePointer.baseAddress, messages.count, true,
                &buffer, Int32(buffer.count)
            )
        }
        if written > Int32(buffer.count) {
            buffer = [CChar](repeating: 0, count: Int(written) + 1)
            written = messages.withUnsafeBufferPointer { messagePointer in
                llama_chat_apply_template(
                    template, messagePointer.baseAddress, messages.count, true,
                    &buffer, Int32(buffer.count)
                )
            }
        }
        guard written > 0 else {
            return systemPrompt.isEmpty ? prompt : systemPrompt + "\n\n" + prompt
        }
        return String(cString: buffer)
    }

    private func tokenize(
        _ text: String, vocab: OpaquePointer, addSpecial: Bool
    ) throws -> [llama_token] {
        let utf8 = Array(text.utf8)
        // Worst case is one token per byte plus the specials. Asking
        // with a negative count first would work too, but a single
        // over-allocated pass is cheaper than two.
        var tokens = [llama_token](repeating: 0, count: utf8.count + 8)
        let count = utf8.withUnsafeBufferPointer { bytes in
            bytes.baseAddress!.withMemoryRebound(to: CChar.self, capacity: utf8.count) { chars in
                llama_tokenize(
                    vocab, chars, Int32(utf8.count),
                    &tokens, Int32(tokens.count), addSpecial, true
                )
            }
        }
        guard count >= 0 else {
            throw LocalRunnerError.loadFailed("tokenisation failed")
        }
        return Array(tokens.prefix(Int(count)))
    }

    private func piece(of token: llama_token, vocab: OpaquePointer) -> [UInt8] {
        var buffer = [CChar](repeating: 0, count: 64)
        var written = llama_token_to_piece(
            vocab, token, &buffer, Int32(buffer.count), 0, false
        )
        if written < 0 {
            // Negative is "this many bytes needed", not an error.
            buffer = [CChar](repeating: 0, count: Int(-written))
            written = llama_token_to_piece(
                vocab, token, &buffer, Int32(buffer.count), 0, false
            )
        }
        guard written > 0 else { return [] }
        return buffer.prefix(Int(written)).map { UInt8(bitPattern: $0) }
    }
}

#endif
