// tool_runner_test.cpp — the filesystem half, against a real filesystem.
//
// ToolPolicy's tests are pure and cover the lexical escapes. These cover
// what only a real disk can show: a symlink inside the workspace that
// points out of it, a create that would clobber, a read that would drag
// 40 MB into a conversation, and the "nobody approved this" path that a
// signature cannot enforce on its own.
//
//   g++ -std=c++17 tool_runner_test.cpp ../src/ToolRunner.cpp
//       ../src/ToolPolicy.cpp -o t && ./t

#include "../src/ToolRunner.h"

#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace fs = std::filesystem;

namespace {

int failures = 0;
int checks = 0;

void Check(bool condition, const std::string& what) {
    checks++;
    if (condition) {
        std::printf("  ok   %s\n", what.c_str());
    } else {
        std::printf("  FAIL %s\n", what.c_str());
        failures++;
    }
}

// A workspace, and an "outside" beside it, both under one temporary root
// that is removed at the end.
struct Sandbox {
    fs::path root;
    fs::path workspace;
    fs::path outside;

    Sandbox() {
        root = fs::temp_directory_path() /
               ("hnx-studio-test-" + std::to_string(::getpid()));
        fs::remove_all(root);
        workspace = root / "project";
        outside = root / "elsewhere";
        fs::create_directories(workspace / "src");
        fs::create_directories(outside);
        Write(workspace / "README.md", "# hello\n");
        Write(workspace / "src" / "main.cpp", "int main() { return 0; }\n");
        Write(outside / "secrets.txt", "hunter2\n");
    }

    ~Sandbox() {
        std::error_code error;
        fs::remove_all(root, error);
    }

    static void Write(const fs::path& path, const std::string& text) {
        std::ofstream stream(path, std::ios::binary);
        stream << text;
    }

    std::string Read(const fs::path& path) const {
        std::ifstream stream(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(stream), {});
    }
};

void TestReading(const Sandbox& box, const hnx::ToolRunner& runner) {
    std::printf("reading\n");

    const hnx::ToolResult ok = runner.ReadFile("README.md");
    Check(ok.ok, "a file inside the workspace reads");
    Check(ok.output == "# hello\n", "and the contents come back");

    const hnx::ToolResult missing = runner.ReadFile("nope.md");
    Check(!missing.ok, "a missing file fails");
    Check(missing.output.find("No readable file") != std::string::npos,
          "with a message a model can act on");

    const hnx::ToolResult escaped = runner.ReadFile("../elsewhere/secrets.txt");
    Check(!escaped.ok, "a path outside the workspace is refused");

    // The one only a real filesystem can show: a link inside the
    // workspace pointing out of it. Every string check in the world
    // passes this.
    std::error_code error;
    fs::create_symlink(box.outside / "secrets.txt", box.workspace / "notes.txt",
                       error);
    if (!error) {
        const hnx::ToolResult linked = runner.ReadFile("notes.txt");
        Check(!linked.ok, "a symlink out of the workspace is refused");
        Check(linked.output.find("link out of the workspace") != std::string::npos,
              "and says so, rather than 'no such file'");
    } else {
        std::printf("  skip symlink checks (%s)\n", error.message().c_str());
    }

    // A symlinked *directory* is the same escape one level up, and the
    // leaf inside it never appears in the path that gets checked.
    fs::create_directory_symlink(box.outside, box.workspace / "linkdir", error);
    if (!error) {
        Check(!runner.ReadFile("linkdir/secrets.txt").ok,
              "and so is a symlinked directory");
    }

    // Too large: refused with the size named, not truncated. A truncated
    // file that reads as complete is how a model comes to reason
    // confidently about code that is not there.
    std::string big(hnx::ToolRunner::kMaxReadBytes + 1024, 'x');
    Sandbox::Write(box.workspace / "huge.txt", big);
    const hnx::ToolResult oversized = runner.ReadFile("huge.txt");
    Check(!oversized.ok, "an oversized file is refused");
    Check(oversized.output.find("larger than") != std::string::npos,
          "and the limit is named");
    Check(oversized.output.find("specific part") != std::string::npos,
          "with what to do instead");
}

void TestListing(const Sandbox& box, const hnx::ToolRunner& runner) {
    std::printf("listing\n");
    Sandbox::Write(box.workspace / ".env", "TOKEN=secret\n");

    const hnx::ToolResult listed = runner.ListDirectory(".");
    Check(listed.ok, "the workspace lists");
    Check(listed.output.find("README.md") != std::string::npos,
          "ordinary files appear");
    Check(listed.output.find("src/") != std::string::npos,
          "directories are marked with a slash");
    // Marked, not hidden: someone scanning the tree should see that a
    // file is one Studio will not read, and why.
    Check(listed.output.find(".env") != std::string::npos,
          "a credentials file still appears");
    Check(listed.output.find("not readable") != std::string::npos,
          "labelled as unreadable rather than quietly omitted");

    Check(!runner.ListDirectory("../elsewhere").ok,
          "a directory outside is refused");
    Check(!runner.ReadFile(".env").ok, "and reading it is still refused");
}

void TestWriting(const Sandbox& box, const hnx::ToolRunner& runner) {
    std::printf("writing\n");

    // The check a signature cannot make on its own. Passing approved as
    // a parameter means forgetting to ask is a refusal rather than a
    // silent success.
    const hnx::ToolResult unapproved =
        runner.WriteFile("README.md", "clobbered\n", false);
    Check(!unapproved.ok, "an unapproved write does nothing");
    Check(unapproved.output.find("nobody approved") != std::string::npos,
          "and says why");
    Check(box.Read(box.workspace / "README.md") == "# hello\n",
          "the file is untouched");

    const hnx::ToolResult approved =
        runner.WriteFile("README.md", "# changed\n", true);
    Check(approved.ok, "an approved write happens");
    Check(box.Read(box.workspace / "README.md") == "# changed\n",
          "and the contents are replaced");
    // Written via a temporary and renamed, so a crash halfway leaves the
    // old file rather than half a new one. The temporary must not be
    // left behind.
    Check(!fs::exists(box.workspace / "README.md.hnx-tmp"),
          "no scratch file is left behind");

    Check(!runner.WriteFile("../elsewhere/secrets.txt", "x\n", true).ok,
          "an approved write outside the workspace is still refused");
}

void TestCreating(const Sandbox& box, const hnx::ToolRunner& runner) {
    std::printf("creating\n");

    Check(!runner.CreateFile("new.txt", "x\n", false).ok,
          "an unapproved create does nothing");
    Check(!fs::exists(box.workspace / "new.txt"), "and no file appears");

    Check(runner.CreateFile("new.txt", "hello\n", true).ok,
          "an approved create happens");
    Check(box.Read(box.workspace / "new.txt") == "hello\n", "with the contents");

    // Refused rather than silently becoming a write: the user agreed to
    // "create a new file", and clobbering an existing one is not that.
    const hnx::ToolResult again = runner.CreateFile("new.txt", "other\n", true);
    Check(!again.ok, "creating over an existing file is refused");
    Check(again.output.find("write_file") != std::string::npos,
          "and points at the tool that shows a diff");
    Check(box.Read(box.workspace / "new.txt") == "hello\n", "the file is intact");

    Check(runner.CreateFile("deep/nested/file.txt", "x\n", true).ok,
          "missing parent directories are created");
}

void TestDeleting(const Sandbox& box, const hnx::ToolRunner& runner) {
    std::printf("deleting\n");
    Sandbox::Write(box.workspace / "throwaway.txt", "x\n");

    Check(!runner.DeleteFile("throwaway.txt", false).ok,
          "an unapproved delete does nothing");
    Check(fs::exists(box.workspace / "throwaway.txt"), "the file survives");

    Check(runner.DeleteFile("throwaway.txt", true).ok, "an approved one happens");
    Check(!fs::exists(box.workspace / "throwaway.txt"), "and the file is gone");

    // A single file only. remove_all on a directory a model named is how
    // one wrong path becomes a lost project, and there is no undo.
    const hnx::ToolResult directory = runner.DeleteFile("src", true);
    Check(!directory.ok, "a directory is refused");
    Check(fs::exists(box.workspace / "src" / "main.cpp"),
          "and its contents are still there");

    Check(!runner.DeleteFile("../elsewhere/secrets.txt", true).ok,
          "and a path outside is refused");
    Check(fs::exists(box.outside / "secrets.txt"), "leaving it in place");
}

}  // namespace

int main() {
    std::printf("ToolRunner\n\n");
    const Sandbox box;
    const hnx::ToolRunner runner(hnx::ToolPolicy(box.workspace.string()));

    TestReading(box, runner);
    TestListing(box, runner);
    TestWriting(box, runner);
    TestCreating(box, runner);
    TestDeleting(box, runner);

    std::printf("\n%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
