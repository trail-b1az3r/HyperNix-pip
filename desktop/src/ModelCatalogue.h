/*
 * ModelCatalogue — what models are on this disk, without loading any.
 *
 * Studio started as a client: it asked a server what models it had. This
 * is the other half, and the reason it exists is that a laptop with a
 * GGUF on it should not need a server running somewhere to use it.
 *
 * Everything here is read-only and bounded. A GGUF is a file somebody
 * downloaded, which makes its header attacker-controlled: every length
 * in it is a 64-bit number that this code would otherwise be told to
 * allocate. So each one is checked against what is left of the file
 * before anything is reserved, and the whole header is capped. A model
 * browser that can be made to allocate 16 exabytes by a malformed
 * download is a model browser that can be made to fall over by one.
 *
 * No Qt and no llama.cpp. Reading a GGUF header is parsing bytes, and
 * keeping it dependency-free means the parsing is testable with a
 * compiler and nothing else -- which matters more here than usual,
 * because the input is hostile by default.
 */
#ifndef HNX_MODEL_CATALOGUE_H
#define HNX_MODEL_CATALOGUE_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace hnx {

/// One model file on disk.
struct LocalModel {
    std::string path;
    /// The file's own name, less the .gguf.
    std::string file_name;
    /// `general.name` from the header when present, else file_name.
    std::string name;
    /// `general.architecture` — "llama", "qwen2", "hypernix0x-v2", ...
    std::string architecture;
    std::uint64_t bytes = 0;
    /// Summed from the tensor shapes, so it is exact rather than a
    /// guess from the file size.
    std::uint64_t parameters = 0;
    /// The type most of the weights are in, by tensor count. This is
    /// what a person means by "what quant is this".
    std::string quantisation;
    /// Every type present, and how many tensors use it.
    std::map<std::string, std::uint64_t> type_histogram;
    std::uint64_t context_length = 0;
    std::uint64_t tensor_count = 0;
    /// False when the header could not be read. `error` says why, and
    /// the entry is still listed -- a file that is there and broken is
    /// something the person wants to see, not something to hide.
    bool ok = false;
    std::string error;

    /// "7.2B", "486M". Empty when parameters is 0.
    std::string parameter_label() const;
    /// "4.1 GB".
    std::string size_label() const;
};

/// Read one GGUF's header. Never throws; check `ok`.
LocalModel ReadGguf(const std::string& path);

/// The directories searched when none are given.
///
/// $HNX_MODEL_PATH first (colon-separated, like $PATH), then the places
/// models actually land: HyperNix's own directory, the Hugging Face
/// cache, LM Studio's, and ~/models.
std::vector<std::string> DefaultSearchPaths();

/// Every .gguf under `roots`, headers read, sorted by name.
///
/// `max_depth` bounds the walk: a home directory handed in by accident
/// should cost a second, not a filesystem crawl. Symlinked directories
/// are not followed, which also settles the loop question.
std::vector<LocalModel> ScanForModels(const std::vector<std::string>& roots,
                                      int max_depth = 4);

/// The name ggml gives a type id, including the HyperNix sub-bit ones.
///
/// Its own table rather than a call into ggml: the catalogue must work
/// in a Studio built without llama.cpp, and a person browsing models
/// should still be told a file is IQ0.5_XXXL even when nothing in this
/// build could run it.
std::string GgufTypeName(std::uint32_t type);

}  // namespace hnx

#endif  // HNX_MODEL_CATALOGUE_H
