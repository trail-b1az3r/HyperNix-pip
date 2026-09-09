// ToolRunner.h — carrying out what a person approved.
//
// ToolPolicy decides; this acts. The split is deliberate: the decision
// is pure and testable without a filesystem, and the part that touches
// the disk is small enough to read in one sitting.
//
// The second boundary check lives here
// ------------------------------------
// ToolPolicy resolves paths lexically, which catches `../../etc/passwd`
// without touching the disk. It cannot catch a *symlink* inside the
// workspace that points outside it -- that needs the filesystem, and a
// pure decision function should not have one.
//
// So every operation re-checks the real path here, after resolution and
// immediately before acting. Both checks are needed and neither is
// redundant: the lexical one refuses an escape without a syscall, and
// this one refuses `workspace/notes -> /etc` regardless of how the path
// was spelled.
//
// It is also the check that closes the gap between deciding and doing.
// A path approved a moment ago can be a symlink now, and the answer to
// that is to look again at the moment of the write rather than to trust
// the earlier look.

#ifndef HYPERNIX_STUDIO_TOOL_RUNNER_H
#define HYPERNIX_STUDIO_TOOL_RUNNER_H

#include <string>
#include <vector>

#include "ToolPolicy.h"

namespace hnx {

struct ToolResult {
    bool ok = false;
    // What the model is told. On failure this is the whole message, so
    // it has to be useful to a model that will try again -- "outside the
    // workspace" tells it to stop, "no such file" tells it to look.
    std::string output;
    // What the UI shows. Separate from `output` because the useful
    // detail for a person is often the resolved path, which is noise to
    // the model.
    std::string detail;
    std::string resolved;
};

class ToolRunner {
public:
    ToolRunner(ToolPolicy policy);

    // Read a file. Refuses anything the policy did not Allow.
    ToolResult ReadFile(const std::string& path) const;
    ToolResult ListDirectory(const std::string& path) const;

    // The three that change the disk. `approved` must be true, and it is
    // the caller's job to have actually asked -- this cannot verify that
    // a human said yes, so it takes it as a parameter and refuses when
    // it is false. A signature that could not express "not approved"
    // would make forgetting to ask a silent success.
    ToolResult WriteFile(const std::string& path, const std::string& contents,
                         bool approved) const;
    ToolResult CreateFile(const std::string& path, const std::string& contents,
                          bool approved) const;
    ToolResult DeleteFile(const std::string& path, bool approved) const;

    // The largest file Studio will read into a conversation. A model
    // does not benefit from 40 MB of minified JavaScript and the tokens
    // are not free, so this refuses with a clear reason rather than
    // truncating -- a truncated file that looks complete is worse than
    // no file.
    static constexpr std::size_t kMaxReadBytes = 512 * 1024;

    // Whether `resolved` is really inside the workspace once symlinks
    // are followed. Public so the file tree can mark an escaping link.
    bool IsTrulyInside(const std::string& resolved) const;

    const ToolPolicy& policy() const { return policy_; }

private:
    ToolResult Refuse(const Decision& decision) const;
    ToolResult CheckedResolve(const Request& request, Verdict required,
                              std::string* resolved) const;

    ToolPolicy policy_;
};

}  // namespace hnx

#endif  // HYPERNIX_STUDIO_TOOL_RUNNER_H
