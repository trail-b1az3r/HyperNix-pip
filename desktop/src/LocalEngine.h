/*
 * LocalEngine — running a model in this process, with no server.
 *
 * Studio was a client: chat went over HTTP to a HyperNix server, and a
 * laptop with a GGUF on it still needed something running somewhere.
 * This is the other path. Same conversation, same tool approval, same
 * workspace -- the model is just here instead of over there.
 *
 * Two builds, one interface
 * -------------------------
 * With llama.cpp available (``-DSTUDIO_LOCAL_LLAMA=ON``) this loads a
 * GGUF and generates. Without it, the same class compiles to a stub
 * whose ``Available()`` is false and whose ``Load()`` fails with a
 * message saying how to get the other one. That is deliberate: Studio
 * must still build and run on a machine with only Qt, because the
 * server path does not need llama.cpp and refusing to build without it
 * would make the harder dependency mandatory for everyone.
 *
 * Nothing here runs a command. The engine loads a file and does
 * arithmetic; the tool boundary is still ToolPolicy's, unchanged, and a
 * local model has exactly the same reach as a remote one -- which is to
 * say file operations inside the workspace, each one approved.
 *
 * Threading: Load and Generate block, and are meant to be called off
 * the UI thread. Cancel is the one exception -- it may be called from
 * any thread while Generate runs, and takes effect at the next token.
 */
#ifndef HNX_LOCAL_ENGINE_H
#define HNX_LOCAL_ENGINE_H

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>

namespace hnx {

struct LoadOptions {
    /// Layers to put on the GPU. 0 is CPU-only; -1 is "as many as fit".
    int gpu_layers = 0;
    /// Context window. 0 takes the model's own trained length.
    int context_length = 0;
    /// 0 lets llama.cpp pick from the hardware.
    int threads = 0;
    /// Keep the weights out of the page cache. Off by default: mmap is
    /// why a second load of the same model is instant.
    bool no_mmap = false;
};

struct GenerateOptions {
    int max_tokens = 512;
    float temperature = 0.7f;
    float top_p = 0.95f;
    int top_k = 40;
    /// 0 asks for a different answer each time; anything else repeats.
    std::uint32_t seed = 0;
};

/// What is currently loaded. Named LoadedModel, not ModelInfo, because
/// HyperLinkClient already has an hnx::ModelInfo for a model the *server*
/// offers -- and StudioBridge includes both headers.
struct LoadedModel {
    std::string path;
    std::string name;
    std::string architecture;
    std::uint64_t parameters = 0;
    int context_length = 0;
    int gpu_layers = 0;
    int layers_total = 0;
};

/// Called once per token with the piece of text it decodes to.
/// Return false to stop generating -- the same effect as Cancel, for a
/// caller that would rather decide in the callback.
using TokenCallback = std::function<bool(const std::string& piece)>;

class LocalEngine {
public:
    LocalEngine();
    ~LocalEngine();

    LocalEngine(const LocalEngine&) = delete;
    LocalEngine& operator=(const LocalEngine&) = delete;

    /// Whether this build can run a model at all. Compile-time.
    static bool Available();
    /// Why not, for the UI to show. Empty when Available().
    static std::string UnavailableReason();

    bool Load(const std::string& path, const LoadOptions& options,
              std::string* error);
    void Unload();
    bool loaded() const;
    LoadedModel info() const;

    /// Generate from `prompt`, calling `on_token` as text arrives.
    /// Returns false and sets `error` on failure. Blocks.
    bool Generate(const std::string& prompt, const GenerateOptions& options,
                  const TokenCallback& on_token, std::string* error);

    /// Stop the running Generate at the next token. Safe from any thread.
    void Cancel();

private:
    struct State;
    State* state_;
    std::atomic<bool> cancel_{false};
};

}  // namespace hnx

#endif  // HNX_LOCAL_ENGINE_H
