#include "ModelCatalogue.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace hnx {
namespace {

// A header this big is not a header. Real ones are tens of kilobytes;
// a 64 MB ceiling leaves room for a model with a very large tokenizer
// vocabulary in its metadata and still refuses a file that claims its
// header is the whole disk.
constexpr std::uint64_t kMaxHeaderBytes = 64ull * 1024 * 1024;
// Likewise: no real model has a billion tensors or metadata keys, and
// both counts are read straight from the file.
constexpr std::uint64_t kMaxTensors = 1ull << 20;
constexpr std::uint64_t kMaxKeys = 1ull << 20;
constexpr std::uint64_t kMaxStringBytes = 1ull << 26;
constexpr std::uint64_t kMaxArrayItems = 1ull << 24;

/// A cursor over the bytes, which refuses to read past the end.
///
/// The whole parser is written against this rather than against a
/// pointer, so "the file said 2^64" becomes a failed read instead of a
/// segfault. Every `Read` returns false at the end rather than throwing,
/// and callers stop at the first false.
class Cursor {
public:
    Cursor(const unsigned char* data, std::size_t size)
        : data_(data), size_(size) {}

    bool Skip(std::uint64_t n) {
        if (n > size_ - pos_) return false;
        pos_ += static_cast<std::size_t>(n);
        return true;
    }

    template <typename T>
    bool Read(T* out) {
        static_assert(std::is_trivially_copyable<T>::value, "raw read");
        if (sizeof(T) > size_ - pos_) return false;
        std::memcpy(out, data_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return true;
    }

    /// A GGUF string: a 64-bit length then that many bytes.
    bool ReadString(std::string* out) {
        std::uint64_t length = 0;
        if (!Read(&length)) return false;
        // Checked against what is *left*, not against a constant alone:
        // a length of 2^63 and a length of "one byte more than this
        // file" are the same mistake and both must fail here.
        if (length > kMaxStringBytes || length > size_ - pos_) return false;
        out->assign(reinterpret_cast<const char*>(data_ + pos_),
                    static_cast<std::size_t>(length));
        pos_ += static_cast<std::size_t>(length);
        return true;
    }

    std::size_t position() const { return pos_; }
    std::size_t remaining() const { return size_ - pos_; }

private:
    const unsigned char* data_;
    std::size_t size_;
    std::size_t pos_ = 0;
};

enum GgufValueType : std::uint32_t {
    kUint8 = 0, kInt8 = 1, kUint16 = 2, kInt16 = 3, kUint32 = 4,
    kInt32 = 5, kFloat32 = 6, kBool = 7, kString = 8, kArray = 9,
    kUint64 = 10, kInt64 = 11, kFloat64 = 12,
};

/// Bytes for a scalar type, or 0 for the ones that are not scalars.
std::size_t ScalarWidth(std::uint32_t type) {
    switch (type) {
        case kUint8: case kInt8: case kBool:   return 1;
        case kUint16: case kInt16:             return 2;
        case kUint32: case kInt32: case kFloat32: return 4;
        case kUint64: case kInt64: case kFloat64: return 8;
        default:                               return 0;
    }
}

/// A metadata value, kept only when it is one this cares about.
struct Value {
    bool is_string = false;
    std::string text;
    std::uint64_t number = 0;
};

bool ReadValue(Cursor* cursor, std::uint32_t type, Value* out, int depth);

/// Skip an array without keeping it. Vocabularies are arrays of a
/// hundred thousand strings and there is no reason to hold one.
bool SkipArray(Cursor* cursor, int depth) {
    if (depth > 4) return false;                 // nested arrays: not a thing
    std::uint32_t element_type = 0;
    std::uint64_t count = 0;
    if (!cursor->Read(&element_type)) return false;
    if (!cursor->Read(&count)) return false;
    if (count > kMaxArrayItems) return false;

    const std::size_t width = ScalarWidth(element_type);
    if (width > 0) {
        // One multiply, checked: count * width can overflow, and an
        // overflowed product that happens to be small would skip a
        // little and leave the cursor inside the array.
        if (count > (kMaxHeaderBytes / width)) return false;
        return cursor->Skip(count * width);
    }
    if (element_type == kString) {
        for (std::uint64_t i = 0; i < count; ++i) {
            std::string ignored;
            if (!cursor->ReadString(&ignored)) return false;
        }
        return true;
    }
    if (element_type == kArray) {
        for (std::uint64_t i = 0; i < count; ++i) {
            if (!SkipArray(cursor, depth + 1)) return false;
        }
        return true;
    }
    return false;
}

bool ReadValue(Cursor* cursor, std::uint32_t type, Value* out, int depth) {
    switch (type) {
        case kUint8:  { std::uint8_t v;  if (!cursor->Read(&v)) return false; out->number = v; return true; }
        case kInt8:   { std::int8_t v;   if (!cursor->Read(&v)) return false; out->number = static_cast<std::uint64_t>(v); return true; }
        case kUint16: { std::uint16_t v; if (!cursor->Read(&v)) return false; out->number = v; return true; }
        case kInt16:  { std::int16_t v;  if (!cursor->Read(&v)) return false; out->number = static_cast<std::uint64_t>(v); return true; }
        case kUint32: { std::uint32_t v; if (!cursor->Read(&v)) return false; out->number = v; return true; }
        case kInt32:  { std::int32_t v;  if (!cursor->Read(&v)) return false; out->number = static_cast<std::uint64_t>(v); return true; }
        case kFloat32:{ float v;         if (!cursor->Read(&v)) return false; out->number = static_cast<std::uint64_t>(v); return true; }
        case kBool:   { std::uint8_t v;  if (!cursor->Read(&v)) return false; out->number = v; return true; }
        case kUint64: { std::uint64_t v; if (!cursor->Read(&v)) return false; out->number = v; return true; }
        case kInt64:  { std::int64_t v;  if (!cursor->Read(&v)) return false; out->number = static_cast<std::uint64_t>(v); return true; }
        case kFloat64:{ double v;        if (!cursor->Read(&v)) return false; out->number = static_cast<std::uint64_t>(v); return true; }
        case kString: {
            out->is_string = true;
            return cursor->ReadString(&out->text);
        }
        case kArray:
            return SkipArray(cursor, depth);
        default:
            return false;                        // an unknown type ends it
    }
}

std::string Trim(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

}  // namespace

std::string GgufTypeName(std::uint32_t type) {
    switch (type) {
        case 0:  return "F32";
        case 1:  return "F16";
        case 2:  return "Q4_0";
        case 3:  return "Q4_1";
        case 6:  return "Q5_0";
        case 7:  return "Q5_1";
        case 8:  return "Q8_0";
        case 9:  return "Q8_1";
        case 10: return "Q2_K";
        case 11: return "Q3_K";
        case 12: return "Q4_K";
        case 13: return "Q5_K";
        case 14: return "Q6_K";
        case 15: return "Q8_K";
        case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS";
        case 19: return "IQ1_S";
        case 20: return "IQ4_NL";
        case 21: return "IQ3_S";
        case 22: return "IQ2_S";
        case 23: return "IQ4_XS";
        case 24: return "I8";
        case 25: return "I16";
        case 26: return "I32";
        case 27: return "I64";
        case 28: return "F64";
        case 29: return "IQ1_M";
        case 30: return "BF16";
        case 34: return "TQ1_0";
        case 35: return "TQ2_0";
        case 39: return "MXFP4";
        case 40: return "NVFP4";
        // HyperNix's own, from native/ggml-hnx. Named here as well as in
        // ggml so a Studio built with no llama.cpp still tells you what
        // a file is instead of printing a bare number.
        case 200: return "IQ0.9_L";
        case 201: return "IQ0.75_M";
        case 202: return "IQ0.5_XXXL";
        case 203: return "IQ0.25_UXL";
        case 204: return "INT1";
        default: return "type " + std::to_string(type);
    }
}

std::string LocalModel::parameter_label() const {
    if (parameters == 0) return "";
    char buffer[32];
    const double p = static_cast<double>(parameters);
    if (p >= 1e12)      std::snprintf(buffer, sizeof(buffer), "%.1fT", p / 1e12);
    else if (p >= 1e9)  std::snprintf(buffer, sizeof(buffer), "%.1fB", p / 1e9);
    else if (p >= 1e6)  std::snprintf(buffer, sizeof(buffer), "%.0fM", p / 1e6);
    else                std::snprintf(buffer, sizeof(buffer), "%.0fK", p / 1e3);
    return buffer;
}

std::string LocalModel::size_label() const {
    char buffer[32];
    const double b = static_cast<double>(bytes);
    if (b >= 1024.0 * 1024 * 1024)
        std::snprintf(buffer, sizeof(buffer), "%.1f GB", b / (1024.0 * 1024 * 1024));
    else if (b >= 1024.0 * 1024)
        std::snprintf(buffer, sizeof(buffer), "%.0f MB", b / (1024.0 * 1024));
    else
        std::snprintf(buffer, sizeof(buffer), "%.0f KB", b / 1024.0);
    return buffer;
}

LocalModel ReadGguf(const std::string& path) {
    LocalModel model;
    model.path = path;

    std::error_code error;
    const fs::path file(path);
    model.file_name = file.stem().string();
    model.name = model.file_name;
    const auto size = fs::file_size(file, error);
    if (error) {
        model.error = "cannot stat: " + error.message();
        return model;
    }
    model.bytes = static_cast<std::uint64_t>(size);

    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        model.error = "cannot open";
        return model;
    }

    // Only the header is read, not the weights: a 40 GB model must cost
    // a few kilobytes to list, or opening the browser reads every model
    // on the disk.
    const std::size_t want = static_cast<std::size_t>(
        std::min<std::uint64_t>(model.bytes, kMaxHeaderBytes));
    std::vector<unsigned char> buffer(want);
    stream.read(reinterpret_cast<char*>(buffer.data()),
                static_cast<std::streamsize>(want));
    buffer.resize(static_cast<std::size_t>(stream.gcount()));

    Cursor cursor(buffer.data(), buffer.size());

    char magic[4] = {0, 0, 0, 0};
    if (!cursor.Read(&magic) || std::memcmp(magic, "GGUF", 4) != 0) {
        model.error = "not a GGUF file";
        return model;
    }
    std::uint32_t version = 0;
    if (!cursor.Read(&version)) {
        model.error = "truncated header";
        return model;
    }
    if (version < 2 || version > 3) {
        model.error = "GGUF version " + std::to_string(version) +
                      " is not one this build reads";
        return model;
    }

    std::uint64_t tensor_count = 0;
    std::uint64_t key_count = 0;
    if (!cursor.Read(&tensor_count) || !cursor.Read(&key_count)) {
        model.error = "truncated header";
        return model;
    }
    if (tensor_count > kMaxTensors || key_count > kMaxKeys) {
        model.error = "header claims an implausible number of entries";
        return model;
    }
    model.tensor_count = tensor_count;

    // --- metadata -------------------------------------------------------
    std::string architecture;
    std::map<std::string, std::uint64_t> numbers;
    for (std::uint64_t i = 0; i < key_count; ++i) {
        std::string key;
        if (!cursor.ReadString(&key)) {
            model.error = "truncated metadata";
            return model;
        }
        std::uint32_t type = 0;
        if (!cursor.Read(&type)) {
            model.error = "truncated metadata";
            return model;
        }
        Value value;
        if (!ReadValue(&cursor, type, &value, 0)) {
            model.error = "unreadable metadata at key '" + key + "'";
            return model;
        }
        if (key == "general.architecture" && value.is_string) {
            architecture = Trim(value.text);
        } else if (key == "general.name" && value.is_string) {
            const std::string name = Trim(value.text);
            if (!name.empty()) model.name = name;
        } else if (!value.is_string) {
            numbers[key] = value.number;
        }
    }
    model.architecture = architecture;

    // Context length is stored per-architecture, so the key is only
    // known once the architecture is.
    if (!architecture.empty()) {
        const auto found = numbers.find(architecture + ".context_length");
        if (found != numbers.end()) model.context_length = found->second;
    }

    // --- tensors --------------------------------------------------------
    // Read for two things a person actually wants: the exact parameter
    // count, and which quantisation the weights are really in. Both are
    // otherwise guesses -- parameter count from file size is wrong by
    // whatever the quantisation is, and `general.file_type` is one
    // number for a file that often mixes types.
    std::uint64_t parameters = 0;
    for (std::uint64_t i = 0; i < tensor_count; ++i) {
        std::string tensor_name;
        if (!cursor.ReadString(&tensor_name)) {
            model.error = "truncated tensor table";
            return model;
        }
        std::uint32_t dimensions = 0;
        if (!cursor.Read(&dimensions) || dimensions > 4) {
            model.error = "tensor '" + tensor_name + "' has an impossible shape";
            return model;
        }
        std::uint64_t elements = 1;
        for (std::uint32_t d = 0; d < dimensions; ++d) {
            std::uint64_t extent = 0;
            if (!cursor.Read(&extent)) {
                model.error = "truncated tensor table";
                return model;
            }
            // Saturating rather than wrapping: a corrupt shape should
            // make the count obviously wrong, not quietly small.
            if (extent != 0 && elements > (UINT64_MAX / extent)) {
                elements = UINT64_MAX;
                break;
            }
            elements *= extent;
        }
        std::uint32_t type = 0;
        std::uint64_t offset = 0;
        if (!cursor.Read(&type) || !cursor.Read(&offset)) {
            model.error = "truncated tensor table";
            return model;
        }
        if (parameters <= UINT64_MAX - elements) parameters += elements;
        model.type_histogram[GgufTypeName(type)] += 1;
    }
    model.parameters = parameters;

    // The dominant type by tensor count. Ties broken by name so the
    // answer does not depend on map iteration order.
    std::uint64_t best = 0;
    for (const auto& entry : model.type_histogram) {
        // F32 is skipped unless it is all there is: every quantised
        // model keeps its norms and biases in F32, and there are enough
        // of them to win a count while telling you nothing.
        if (entry.first == "F32" && model.type_histogram.size() > 1) continue;
        if (entry.second > best) {
            best = entry.second;
            model.quantisation = entry.first;
        }
    }
    if (model.quantisation.empty() && !model.type_histogram.empty()) {
        model.quantisation = model.type_histogram.begin()->first;
    }

    model.ok = true;
    return model;
}

std::vector<std::string> DefaultSearchPaths() {
    std::vector<std::string> paths;

    if (const char* configured = std::getenv("HNX_MODEL_PATH")) {
        std::stringstream stream(configured);
        std::string item;
        while (std::getline(stream, item, ':')) {
            if (!item.empty()) paths.push_back(item);
        }
    }

    const char* home = std::getenv("HOME");
    if (home == nullptr) {
#ifdef _WIN32
        home = std::getenv("USERPROFILE");
#endif
    }
    if (home != nullptr) {
        const fs::path base(home);
        for (const char* suffix : {
                 ".hypernix/models",
                 ".cache/huggingface/hub",
                 ".cache/lm-studio/models",
                 ".local/share/models",
                 "models",
             }) {
            paths.push_back((base / suffix).string());
        }
    }
    return paths;
}

std::vector<LocalModel> ScanForModels(const std::vector<std::string>& roots,
                                      int max_depth) {
    std::vector<LocalModel> found;
    std::vector<std::string> seen;

    for (const std::string& root : roots) {
        std::error_code error;
        if (!fs::is_directory(root, error)) continue;

        // skip_permission_denied, because a home directory with one
        // unreadable folder in it should still list the models beside
        // it. And no symlink following: it settles directory loops, and
        // a link into / would turn a scan into a filesystem crawl.
        auto options = fs::directory_options::skip_permission_denied;
        fs::recursive_directory_iterator iterator(root, options, error);
        if (error) continue;

        for (auto entry = iterator; entry != fs::recursive_directory_iterator();
             entry.increment(error)) {
            if (error) break;
            if (entry.depth() > max_depth) {
                entry.disable_recursion_pending();
                continue;
            }
            if (entry->is_symlink(error)) {
                entry.disable_recursion_pending();
                continue;
            }
            if (!entry->is_regular_file(error)) continue;

            const fs::path& path = entry->path();
            if (path.extension() != ".gguf") continue;
            // A sharded model is many files; only the first names the
            // model, and loading any other one fails confusingly.
            const std::string stem = path.stem().string();
            if (stem.size() > 12) {
                const std::string tail = stem.substr(stem.size() - 12);
                if (tail.compare(0, 1, "-") == 0 &&
                    tail.find("-of-") != std::string::npos &&
                    tail.compare(1, 5, "00001") != 0) {
                    continue;
                }
            }

            const std::string canonical = fs::weakly_canonical(path, error).string();
            const std::string key = error ? path.string() : canonical;
            if (std::find(seen.begin(), seen.end(), key) != seen.end()) continue;
            seen.push_back(key);

            found.push_back(ReadGguf(path.string()));
        }
    }

    std::sort(found.begin(), found.end(),
              [](const LocalModel& a, const LocalModel& b) {
                  if (a.name != b.name) return a.name < b.name;
                  return a.path < b.path;
              });
    return found;
}

}  // namespace hnx
