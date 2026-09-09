// local_engine_test.cpp — running a model here, or saying honestly that
// this build cannot.
//
// The same test binary is built in both configurations and checks
// different things in each. Without llama.cpp it checks the stub is
// honest: Available() false, a reason a person can act on, and every
// entry point failing rather than pretending. With llama.cpp, and with
// a model handed to it in $HNX_TEST_MODEL, it loads and generates.
//
// Testing only the configuration you happen to build is how the other
// one rots, and the stub is the one nearly every CI machine gets.
//
//   ctest --test-dir build -R local_engine
//   HNX_TEST_MODEL=/path/to/tiny.gguf ./build/local_engine_test

#include "../src/LocalEngine.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

namespace {

int failures = 0;
int checks = 0;

void Check(bool condition, const std::string& what) {
    checks++;
    std::printf(condition ? "  ok   %s\n" : "  FAIL %s\n", what.c_str());
    if (!condition) failures++;
}

void Skip(const std::string& what) {
    std::printf("  skip %s\n", what.c_str());
}

}  // namespace

int main() {
    std::printf("\nwhat this build can do\n");
    const bool available = hnx::LocalEngine::Available();
    std::printf("  (local inference is %s)\n", available ? "ON" : "OFF");

    hnx::LocalEngine engine;
    std::string error;

    if (!available) {
        std::printf("\nthe stub is honest about it\n");
        Check(!hnx::LocalEngine::UnavailableReason().empty(),
              "there is a reason, not just a false");
        const std::string reason = hnx::LocalEngine::UnavailableReason();
        Check(reason.find("STUDIO_LOCAL_LLAMA") != std::string::npos,
              "and it names the flag that turns it on");
        Check(reason.find("server") != std::string::npos,
              "and the other way to get a model");

        Check(!engine.Load("/tmp/anything.gguf", {}, &error),
              "Load fails rather than half-succeeding");
        Check(error == reason, "with that same reason");
        Check(!engine.loaded(), "nothing is loaded");

        error.clear();
        Check(!engine.Generate("hi", {}, [](const std::string&) { return true; },
                               &error),
              "Generate fails too");
        Check(!error.empty(), "and says why");

        engine.Cancel();
        engine.Unload();
        Check(true, "Cancel and Unload are safe to call anyway");

        std::printf("\n%d checks, %d failures\n\n", checks, failures);
        return failures == 0 ? 0 : 1;
    }

    std::printf("\nrefusing what is not a model\n");
    Check(!engine.Load("/etc/hostname", {}, &error), "a text file is refused");
    Check(error.find("GGUF") != std::string::npos,
          "and told it is not a GGUF, before llama.cpp is asked");
    Check(!engine.Load("/nonexistent/model.gguf", {}, &error),
          "a missing file is refused");

    const char* model = std::getenv("HNX_TEST_MODEL");
    if (model == nullptr) {
        Skip("loading a model (set HNX_TEST_MODEL to a small .gguf)");
        std::printf("\n%d checks, %d failures\n\n", checks, failures);
        return failures == 0 ? 0 : 1;
    }

    std::printf("\nloading\n");
    hnx::LoadOptions options;
    options.context_length = 256;
    options.threads = 2;
    if (!engine.Load(model, options, &error)) {
        std::printf("  FAIL could not load %s: %s\n", model, error.c_str());
        std::printf("\n%d checks, %d failures\n\n", checks, ++failures);
        return 1;
    }
    Check(engine.loaded(), "it loads");
    const hnx::LoadedModel info = engine.info();
    Check(!info.name.empty(), "and knows what it loaded");
    Check(info.parameters > 0, "with a parameter count from the header");
    Check(info.context_length > 0, "and a context length");

    std::printf("\ngenerating\n");
    int tokens = 0;
    std::string text;
    hnx::GenerateOptions gen;
    gen.max_tokens = 16;
    gen.seed = 7;
    Check(engine.Generate("The capital of France is", gen,
                          [&](const std::string& piece) {
                              ++tokens;
                              text += piece;
                              return true;
                          }, &error),
          "it generates");
    Check(tokens > 0, "the callback ran per token, streaming");
    Check(!text.empty(), "and the pieces decode to text");

    std::printf("\ncontrol\n");
    {
        // Greedy has to be reproducible, or nothing built on it can be
        // tested. It is also what temperature 0 must mean: a literal
        // zero temperature is a division.
        std::string a, b;
        hnx::GenerateOptions greedy;
        greedy.max_tokens = 8;
        greedy.temperature = 0.0f;
        engine.Generate("Hello", greedy,
                        [&](const std::string& p) { a += p; return true; }, &error);
        engine.Generate("Hello", greedy,
                        [&](const std::string& p) { b += p; return true; }, &error);
        Check(a == b, "temperature 0 gives the same answer twice");
    }
    {
        int seen = 0;
        hnx::GenerateOptions many;
        many.max_tokens = 200;
        engine.Generate("Once upon", many,
                        [&](const std::string&) { return ++seen < 3; }, &error);
        Check(seen == 3, "returning false from the callback stops it");
    }
    {
        // From another thread, which is where it comes from: the UI
        // thread cannot be the one blocked inside Generate.
        int seen = 0;
        hnx::GenerateOptions many;
        many.max_tokens = 400;
        std::thread canceller([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            engine.Cancel();
        });
        engine.Generate("Tell me a very long story about", many,
                        [&](const std::string&) { ++seen; return true; }, &error);
        canceller.join();
        Check(seen < 400, "Cancel from another thread stops it");
    }

    std::printf("\nrefusing what will not fit\n");
    {
        std::string huge;
        for (int i = 0; i < 4000; ++i) huge += "word ";
        error.clear();
        Check(!engine.Generate(huge, gen,
                               [](const std::string&) { return true; }, &error),
              "a prompt longer than the context is refused");
        // Silently dropping the oldest tokens would make the model
        // contradict what it just said, with nothing to show for it.
        Check(error.find("context") != std::string::npos,
              "and says so rather than truncating");
    }

    std::printf("\nunloading\n");
    engine.Unload();
    Check(!engine.loaded(), "it unloads");
    error.clear();
    Check(!engine.Generate("x", gen, [](const std::string&) { return true; },
                           &error),
          "and generating afterwards is an error, not a crash");
    engine.Unload();
    Check(true, "unloading twice is safe");

    std::printf("\n%d checks, %d failures\n\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
