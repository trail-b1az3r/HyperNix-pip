// ToolPolicy.h — what a model is allowed to do to your machine.
//
// Studio lets a model edit files, create them, and search the web. That
// is the useful part and it is also the whole risk: the thing deciding
// which file to write is a language model, and the text steering it can
// come from a document, a web page, or a repository that somebody else
// wrote.
//
// The stated constraint for this project is that host-provided
// applications must not be executed automatically without user
// confirmation and security validation, and that discovery and
// connection stay separate from arbitrary code execution. This class is
// where that is enforced, and it is a class rather than a setting on
// purpose: a policy you can switch off in Preferences is a policy that
// gets switched off.
//
// Three rules, in order
// --------------------
// 1. **There is no shell tool.** Not disabled by default — absent. No
//    tool in Studio runs a command, spawns a process, or evaluates
//    anything. A model can ask for a file to be written and a person can
//    agree; there is no path by which a model runs code. That is the one
//    guarantee worth more than every other check here, and the only way
//    to keep it is not to write the feature.
//
// 2. **Every path is resolved and must land inside the workspace.**
//    Resolved, not string-checked: `..`, symlinks and absolute paths all
//    get their chance, and a check on the string a model supplied is a
//    check on the wrong thing.
//
// 3. **Anything that changes the machine, or leaves it, needs a human.**
//    Writes, creations, deletions and web searches all return
//    NeedsApproval. Reads inside the workspace are the only thing that
//    happens on the model's word alone, and even those are refused for
//    the handful of files whose contents are credentials.
//
// The verdict is data, not an action: Decide() returns what should
// happen and why, and something else does it. That split is what makes
// this testable without a filesystem, a Qt event loop, or a model — and
// the tests are the only reason to believe any of it.

#ifndef HYPERNIX_STUDIO_TOOL_POLICY_H
#define HYPERNIX_STUDIO_TOOL_POLICY_H

#include <string>
#include <vector>

namespace hnx {

// What a model asked for.
enum class ToolKind {
    ReadFile,
    WriteFile,     // overwrite an existing file
    CreateFile,    // a path that does not exist yet
    DeleteFile,
    ListDirectory,
    WebSearch,
    // Anything the model named that Studio does not implement. Kept as a
    // case rather than an error so an unknown tool is *refused* rather
    // than falling through some default.
    Unknown,
};

enum class Verdict {
    Allow,           // read-only, inside the workspace, not sensitive
    NeedsApproval,   // a person has to say yes, every time
    Deny,            // not offered to the user at all
};

struct Request {
    ToolKind kind = ToolKind::Unknown;
    // As the model wrote it. Never used as a path without resolution.
    std::string path;
    std::string query;      // WebSearch
    std::size_t bytes = 0;  // WriteFile / CreateFile payload size
};

struct Decision {
    Verdict verdict = Verdict::Deny;
    // Shown to the user verbatim, so it has to say what will happen in
    // terms of their machine and not in terms of this class.
    std::string reason;
    // The resolved absolute path, when there is one. What would actually
    // be touched -- which is the thing to show, because it can differ
    // from what the model asked for.
    std::string resolved;
    // True when the decision turned on something worth naming in the
    // prompt: a path that escaped, a file that holds credentials.
    bool suspicious = false;
};

class ToolPolicy {
public:
    // `workspace` is the directory the user chose. Everything is
    // relative to it and nothing outside it is reachable.
    explicit ToolPolicy(std::string workspace);

    // The whole policy. Pure: no filesystem access beyond path
    // normalisation, no I/O, no time. Given the same request and the
    // same workspace it always answers the same thing, which is what
    // makes it possible to test the refusals rather than hope for them.
    Decision Decide(const Request& request) const;

    // Normalise `path` against the workspace and return it, or an empty
    // string if it lands outside.
    //
    // Lexical: `.` and `..` are resolved textually and the result is
    // required to be under the workspace. Symlinks are *not* followed
    // here -- that needs the filesystem, and the caller checks it before
    // acting (see ToolRunner). Both checks are needed: this one refuses
    // `../../etc/passwd` without touching the disk, and the caller's
    // refuses a symlink inside the workspace pointing out of it.
    std::string Resolve(const std::string& path) const;

    // Whether a resolved path is one whose contents are a credential.
    // Public because the UI marks these in the file tree.
    static bool IsSensitive(const std::string& resolved);

    // The tool names Studio implements, for the model's tool list. A
    // model cannot ask for a tool that is not here, and nothing here
    // runs a command.
    static std::vector<std::string> ToolNames();

    static ToolKind KindFromName(const std::string& name);

    const std::string& workspace() const { return workspace_; }

private:
    std::string workspace_;
};

}  // namespace hnx

#endif  // HYPERNIX_STUDIO_TOOL_POLICY_H
