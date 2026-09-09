// model_catalogue_test.cpp — reading GGUF headers, including hostile ones.
//
// The catalogue's input is a file somebody downloaded, so most of what
// is checked here is not "does it parse a good file" but "what does it
// do with a bad one". Every length in a GGUF header is a 64-bit number
// the parser would otherwise be told to allocate, and the interesting
// cases are the ones where that number is a lie.
//
//   g++ -std=c++17 model_catalogue_test.cpp ../src/ModelCatalogue.cpp -o t && ./t

#include "../src/ModelCatalogue.h"

#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int failures = 0;
int checks = 0;

void Check(bool condition, const std::string& what) {
    checks++;
    std::printf(condition ? "  ok   %s\n" : "  FAIL %s\n", what.c_str());
    if (!condition) failures++;
}

/// Builds a GGUF header byte by byte, so a test can make one wrong in
/// exactly one way.
class Builder {
public:
    void U32(std::uint32_t v) { Raw(&v, sizeof(v)); }
    void U64(std::uint64_t v) { Raw(&v, sizeof(v)); }
    void Str(const std::string& s) {
        U64(s.size());
        bytes_.insert(bytes_.end(), s.begin(), s.end());
    }
    /// A length that does not match what follows: the whole point.
    void RawLength(std::uint64_t claimed) { U64(claimed); }
    void Magic(const char* four = "GGUF") { bytes_.insert(bytes_.end(), four, four + 4); }

    void KeyString(const std::string& key, const std::string& value) {
        Str(key);
        U32(8);                                  // string
        Str(value);
    }
    void KeyU32(const std::string& key, std::uint32_t value) {
        Str(key);
        U32(4);                                  // uint32
        U32(value);
    }
    void Tensor(const std::string& name, std::vector<std::uint64_t> dims,
                std::uint32_t type) {
        Str(name);
        U32(static_cast<std::uint32_t>(dims.size()));
        for (std::uint64_t d : dims) U64(d);
        U32(type);
        U64(0);                                  // offset
    }

    void Raw(const void* data, std::size_t n) {
        const auto* p = static_cast<const unsigned char*>(data);
        bytes_.insert(bytes_.end(), p, p + n);
    }

    std::string Write(const fs::path& path) const {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(bytes_.data()),
                  static_cast<std::streamsize>(bytes_.size()));
        out.close();
        return path.string();
    }

private:
    std::vector<unsigned char> bytes_;
};

/// A small, valid, two-tensor model.
Builder GoodModel() {
    Builder b;
    b.Magic();
    b.U32(3);                                    // version
    b.U64(2);                                    // tensors
    b.U64(3);                                    // metadata keys
    b.KeyString("general.architecture", "llama");
    b.KeyString("general.name", "Tiny Test");
    b.KeyU32("llama.context_length", 4096);
    b.Tensor("blk.0.attn_q.weight", {512, 512}, 202);   // IQ0.5_XXXL
    b.Tensor("output_norm.weight", {512}, 0);           // F32
    return b;
}

}  // namespace

int main() {
    const fs::path root =
        fs::temp_directory_path() / ("hnx-cat-" + std::to_string(::getpid()));
    fs::create_directories(root);

    std::printf("\nreading a good file\n");
    {
        const std::string path = GoodModel().Write(root / "tiny.gguf");
        const hnx::LocalModel m = hnx::ReadGguf(path);
        Check(m.ok, "it parses");
        Check(m.error.empty(), "with no error");
        Check(m.name == "Tiny Test", "general.name wins over the file name");
        Check(m.architecture == "llama", "the architecture comes through");
        Check(m.context_length == 4096, "and the per-arch context length");
        Check(m.tensor_count == 2, "both tensors counted");
        // 512*512 + 512. Summed from the shapes, so it is exact rather
        // than inferred from the file size, which the quantisation makes
        // meaningless.
        Check(m.parameters == 512 * 512 + 512, "parameters summed from shapes");
        Check(m.parameter_label() == "263K",
              "and labelled, rounded to nearest (262656 -> 263K)");
        Check(m.quantisation == "IQ0.5_XXXL",
              "the dominant type is the quantisation, not the F32 norms");
    }

    std::printf("\nthe HyperNix types are named without llama.cpp\n");
    {
        Check(hnx::GgufTypeName(200) == "IQ0.9_L", "200");
        Check(hnx::GgufTypeName(201) == "IQ0.75_M", "201");
        Check(hnx::GgufTypeName(202) == "IQ0.5_XXXL", "202");
        Check(hnx::GgufTypeName(203) == "IQ0.25_UXL", "203");
        Check(hnx::GgufTypeName(204) == "INT1", "204");
        Check(hnx::GgufTypeName(8) == "Q8_0", "and upstream's still work");
        Check(hnx::GgufTypeName(999).find("999") != std::string::npos,
              "an unknown type says its number rather than lying");
    }

    std::printf("\nfiles that are not what they claim\n");
    {
        Builder b;
        b.Magic("JUNK");
        b.U32(3);
        const hnx::LocalModel m = hnx::ReadGguf(b.Write(root / "notgguf.gguf"));
        Check(!m.ok, "a bad magic is refused");
        Check(m.error.find("not a GGUF") != std::string::npos, "and says so");
        Check(m.bytes > 0, "but the file is still described");
    }
    {
        Builder b;
        b.Magic();
        b.U32(99);
        b.U64(0);
        b.U64(0);
        const hnx::LocalModel m = hnx::ReadGguf(b.Write(root / "future.gguf"));
        Check(!m.ok, "an unknown version is refused");
        Check(m.error.find("99") != std::string::npos, "naming the version");
    }
    {
        Builder b;
        b.Magic();
        b.U32(3);
        b.U64(2);
        b.U64(3);
        // Header ends here: the counts promise more than the file has.
        const hnx::LocalModel m = hnx::ReadGguf(b.Write(root / "short.gguf"));
        Check(!m.ok, "a truncated header is refused");
    }

    std::printf("\nlengths that are lies\n");
    {
        // The one that matters: a string that says it is 2^63 bytes.
        // Nothing may be reserved on the strength of that number.
        Builder b;
        b.Magic();
        b.U32(3);
        b.U64(0);
        b.U64(1);
        b.RawLength(0x7FFFFFFFFFFFFFFFull);
        b.Raw("x", 1);
        const hnx::LocalModel m = hnx::ReadGguf(b.Write(root / "huge-key.gguf"));
        Check(!m.ok, "an impossible key length is refused, not allocated");
    }
    {
        Builder b;
        b.Magic();
        b.U32(3);
        b.U64(0xFFFFFFFFFFFFFFFFull);            // tensors
        b.U64(0);
        const hnx::LocalModel m = hnx::ReadGguf(b.Write(root / "huge-count.gguf"));
        Check(!m.ok, "an impossible tensor count is refused");
        Check(m.error.find("implausible") != std::string::npos, "and says why");
    }
    {
        Builder b;
        b.Magic();
        b.U32(3);
        b.U64(0);
        b.U64(1);
        b.Str("big.array");
        b.U32(9);                                // array
        b.U32(4);                                // of uint32
        b.U64(0xFFFFFFFFFFFFFFFFull);            // this many
        const hnx::LocalModel m = hnx::ReadGguf(b.Write(root / "huge-array.gguf"));
        Check(!m.ok, "an impossible array length is refused");
    }
    {
        Builder b;
        b.Magic();
        b.U32(3);
        b.U64(1);
        b.U64(0);
        b.Str("t");
        b.U32(9999);                             // dimensions
        const hnx::LocalModel m = hnx::ReadGguf(b.Write(root / "huge-dims.gguf"));
        Check(!m.ok, "an impossible tensor rank is refused");
    }

    std::printf("\nscanning\n");
    {
        const fs::path tree = root / "tree";
        fs::create_directories(tree / "a" / "b");
        GoodModel().Write(tree / "one.gguf");
        GoodModel().Write(tree / "a" / "two.gguf");
        GoodModel().Write(tree / "a" / "b" / "three.gguf");
        std::ofstream(tree / "notes.txt") << "not a model";

        const auto models = hnx::ScanForModels({tree.string()});
        Check(models.size() == 3, "every .gguf beneath the root is found");
        bool any_txt = false;
        for (const auto& m : models) {
            if (m.path.find(".txt") != std::string::npos) any_txt = true;
        }
        Check(!any_txt, "and nothing that is not one");

        const auto shallow = hnx::ScanForModels({tree.string()}, 1);
        Check(shallow.size() == 2, "max_depth bounds the walk");

        const auto missing = hnx::ScanForModels({(root / "nope").string()});
        Check(missing.empty(), "a directory that is not there is not an error");
    }
    {
        // A scan must not be turned into a filesystem crawl by a link.
        const fs::path tree = root / "linked";
        fs::create_directories(tree / "real");
        GoodModel().Write(tree / "real" / "m.gguf");
        std::error_code ignored;
        fs::create_directory_symlink(tree / "real", tree / "loop", ignored);
        const auto models = hnx::ScanForModels({tree.string()});
        Check(models.size() == 1, "a symlinked directory is not descended");
    }

    std::printf("\nlabels\n");
    {
        hnx::LocalModel m;
        m.parameters = 7'241'000'000ull;
        m.bytes = 4'400'000'000ull;
        Check(m.parameter_label() == "7.2B", "billions");
        Check(m.size_label().find("GB") != std::string::npos, "gigabytes");
        hnx::LocalModel empty;
        Check(empty.parameter_label().empty(),
              "an unknown parameter count is blank, not '0K'");
    }

    fs::remove_all(root);
    std::printf("\n%d checks, %d failures\n\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
