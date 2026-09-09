#include "LocalEngine.h"

#include "ModelCatalogue.h"

#ifdef HNX_STUDIO_LOCAL_LLAMA
#include <llama.h>

#include <mutex>
#include <vector>
#endif

namespace hnx {

#ifdef HNX_STUDIO_LOCAL_LLAMA

namespace {

/// llama_backend_init is global and must happen once per process.
/// std::call_once rather than a bool: Load can be called from whichever
/// thread the UI hands it to, and two of them racing here would
/// initialise the backend twice.
std::once_flag g_backend_once;

void EnsureBackend() {
    std::call_once(g_backend_once, [] { llama_backend_init(); });
}

/// Turn one token into the text it stands for.
///
/// Two calls, because the first tells you how much room you need.
/// llama_token_to_piece writes no terminator and returns a *negative*
/// length when the buffer is short, which is easy to read as an error
/// and is not one.
std::string PieceFor(const llama_vocab* vocab, llama_token token) {
    char stack[128];
    int n = llama_token_to_piece(vocab, token, stack, sizeof(stack), 0, false);
    if (n >= 0) return std::string(stack, static_cast<std::size_t>(n));
    std::vector<char> heap(static_cast<std::size_t>(-n));
    n = llama_token_to_piece(vocab, token, heap.data(),
                             static_cast<int>(heap.size()), 0, false);
    if (n < 0) return std::string();
    return std::string(heap.data(), static_cast<std::size_t>(n));
}

std::vector<llama_token> Tokenize(const llama_vocab* vocab,
                                  const std::string& text, bool add_special) {
    // Same two-call shape: ask with no buffer to be told the count.
    const int upper = -llama_tokenize(vocab, text.c_str(),
                                      static_cast<int>(text.size()), nullptr, 0,
                                      add_special, true);
    if (upper <= 0) return {};
    std::vector<llama_token> tokens(static_cast<std::size_t>(upper));
    const int n = llama_tokenize(vocab, text.c_str(),
                                 static_cast<int>(text.size()), tokens.data(),
                                 static_cast<int>(tokens.size()), add_special,
                                 true);
    if (n < 0) return {};
    tokens.resize(static_cast<std::size_t>(n));
    return tokens;
}

}  // namespace

struct LocalEngine::State {
    llama_model* model = nullptr;
    llama_context* context = nullptr;
    LoadedModel info;
    std::mutex mutex;               // one generation at a time
};

LocalEngine::LocalEngine() : state_(new State) {}

LocalEngine::~LocalEngine() {
    Unload();
    delete state_;
}

bool LocalEngine::Available() { return true; }
std::string LocalEngine::UnavailableReason() { return ""; }

bool LocalEngine::Load(const std::string& path, const LoadOptions& options,
                       std::string* error) {
    Unload();
    EnsureBackend();

    // Read the header first. It costs a few kilobytes and it means a
    // file that is not a model, or is one this build cannot run, is
    // refused with a sentence rather than by llama.cpp printing to
    // stderr and returning null.
    const LocalModel described = ReadGguf(path);
    if (!described.ok) {
        if (error) *error = described.error.empty()
            ? "not a model file this build can read"
            : described.error;
        return false;
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = options.gpu_layers;
    // load_mode, not the use_mmap bool this had first: upstream replaced
    // the flag with an enum that also covers mlock and direct I/O. Only
    // touched when the caller asked for something other than the
    // default, so the default keeps tracking whatever upstream thinks
    // is right for the device.
    if (options.no_mmap) {
        model_params.load_mode = LLAMA_LOAD_MODE_NONE;
    }

    llama_model* model = llama_model_load_from_file(path.c_str(), model_params);
    if (model == nullptr) {
        if (error) {
            *error = "llama.cpp could not load " + described.name +
                     ". Its quantisation is " + described.quantisation +
                     "; a build without native/ggml-hnx applied cannot read "
                     "the HyperNix sub-bit types.";
        }
        return false;
    }

    llama_context_params context_params = llama_context_default_params();
    if (options.context_length > 0) {
        context_params.n_ctx = static_cast<std::uint32_t>(options.context_length);
    }
    if (options.threads > 0) {
        context_params.n_threads = options.threads;
        context_params.n_threads_batch = options.threads;
    }

    llama_context* context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        llama_model_free(model);
        if (error) {
            *error = "the model loaded but a context of " +
                     std::to_string(options.context_length) +
                     " tokens would not fit. Try a smaller context, or fewer "
                     "GPU layers.";
        }
        return false;
    }

    state_->model = model;
    state_->context = context;
    state_->info.path = path;
    state_->info.name = described.name;
    state_->info.architecture = described.architecture;
    state_->info.parameters = described.parameters;
    state_->info.context_length = static_cast<int>(llama_n_ctx(context));
    state_->info.gpu_layers = options.gpu_layers;
    state_->info.layers_total = llama_model_n_layer(model);
    return true;
}

void LocalEngine::Unload() {
    if (state_->context != nullptr) {
        llama_free(state_->context);
        state_->context = nullptr;
    }
    if (state_->model != nullptr) {
        llama_model_free(state_->model);
        state_->model = nullptr;
    }
    state_->info = LoadedModel();
}

bool LocalEngine::loaded() const { return state_->model != nullptr; }
LoadedModel LocalEngine::info() const { return state_->info; }

bool LocalEngine::Generate(const std::string& prompt,
                           const GenerateOptions& options,
                           const TokenCallback& on_token, std::string* error) {
    if (!loaded()) {
        if (error) *error = "no model is loaded";
        return false;
    }
    std::lock_guard<std::mutex> guard(state_->mutex);
    cancel_.store(false);

    const llama_vocab* vocab = llama_model_get_vocab(state_->model);
    std::vector<llama_token> tokens = Tokenize(vocab, prompt, true);
    if (tokens.empty()) {
        if (error) *error = "the prompt tokenised to nothing";
        return false;
    }

    const int context_size = static_cast<int>(llama_n_ctx(state_->context));
    if (static_cast<int>(tokens.size()) >= context_size) {
        if (error) {
            *error = "the prompt is " + std::to_string(tokens.size()) +
                     " tokens and the context is " +
                     std::to_string(context_size) +
                     ". Clear the conversation or load with a larger context.";
        }
        return false;
    }

    // Fresh KV cache per call. Studio sends the whole conversation each
    // time, so reusing it would prepend the last exchange to this one.
    llama_memory_clear(llama_get_memory(state_->context), true);

    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler* sampler = llama_sampler_chain_init(chain_params);
    if (options.temperature <= 0.0f) {
        // Not "temperature 0": a zero temperature is a division. Greedy
        // is what temperature 0 means, so that is what is built.
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(options.top_k));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(options.top_p, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(options.temperature));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(
            options.seed == 0 ? LLAMA_DEFAULT_SEED : options.seed));
    }

    bool ok = true;
    llama_batch batch = llama_batch_get_one(tokens.data(),
                                            static_cast<int32_t>(tokens.size()));
    int produced = 0;
    int used = static_cast<int>(tokens.size());

    while (produced < options.max_tokens) {
        if (llama_decode(state_->context, batch) != 0) {
            if (error) *error = "llama_decode failed";
            ok = false;
            break;
        }
        const llama_token next = llama_sampler_sample(sampler, state_->context, -1);
        if (llama_vocab_is_eog(vocab, next)) break;

        const std::string piece = PieceFor(vocab, next);
        if (!piece.empty() && on_token && !on_token(piece)) break;
        if (cancel_.load()) break;

        ++produced;
        ++used;
        if (used >= context_size) {
            // Stopping is the honest end. Silently dropping the oldest
            // tokens would make the model contradict what it just said,
            // and the caller cannot see that it happened.
            break;
        }
        batch = llama_batch_get_one(const_cast<llama_token*>(&next), 1);
    }

    llama_sampler_free(sampler);
    return ok;
}

void LocalEngine::Cancel() { cancel_.store(true); }

#else  // ---------------------------------------------------------------

struct LocalEngine::State {};

LocalEngine::LocalEngine() : state_(nullptr) {}
LocalEngine::~LocalEngine() = default;

bool LocalEngine::Available() { return false; }

std::string LocalEngine::UnavailableReason() {
    return "This Studio was built without local inference. Rebuild with "
           "-DSTUDIO_LOCAL_LLAMA=ON and -DLLAMA_ROOT=/path/to/llama.cpp to "
           "run models on this machine; connect to a HyperNix server "
           "otherwise.";
}

bool LocalEngine::Load(const std::string&, const LoadOptions&,
                       std::string* error) {
    if (error) *error = UnavailableReason();
    return false;
}

void LocalEngine::Unload() {}
bool LocalEngine::loaded() const { return false; }
LoadedModel LocalEngine::info() const { return LoadedModel(); }

bool LocalEngine::Generate(const std::string&, const GenerateOptions&,
                           const TokenCallback&, std::string* error) {
    if (error) *error = UnavailableReason();
    return false;
}

void LocalEngine::Cancel() {}

#endif

}  // namespace hnx
