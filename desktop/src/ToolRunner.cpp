// ToolRunner.cpp — see ToolRunner.h.

#include "ToolRunner.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <system_error>

namespace fs = std::filesystem;

namespace hnx {
namespace {

std::string ReadAll(const fs::path& path, std::size_t limit, bool* too_big) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return {};
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    std::string contents = buffer.str();
    if (contents.size() > limit) {
        *too_big = true;
        return {};
    }
    return contents;
}

}  // namespace

ToolRunner::ToolRunner(ToolPolicy policy) : policy_(std::move(policy)) {}

bool ToolRunner::IsTrulyInside(const std::string& resolved) const {
    std::error_code error;
    // weakly_canonical rather than canonical: the path may not exist yet
    // (CreateFile), and canonical throws for a missing one. Weakly
    // resolves the existing prefix and appends the rest, which is
    // exactly the semantics wanted -- a symlinked parent directory is
    // followed even when the leaf is new.
    const fs::path real = fs::weakly_canonical(fs::path(resolved), error);
    if (error) return false;
    const fs::path root = fs::weakly_canonical(fs::path(policy_.workspace()), error);
    if (error) return false;

    // Component-wise, not string prefix: "/home/dev/project-backup"
    // starts with "/home/dev/project" as a string and is a different
    // directory.
    auto real_it = real.begin();
    for (auto root_it = root.begin(); root_it != root.end(); ++root_it, ++real_it) {
        if (real_it == real.end() || *real_it != *root_it) return false;
    }
    return true;
}

ToolResult ToolRunner::Refuse(const Decision& decision) const {
    ToolResult result;
    result.ok = false;
    result.output = decision.reason;
    result.detail = decision.reason;
    result.resolved = decision.resolved;
    return result;
}

ToolResult ToolRunner::CheckedResolve(const Request& request, Verdict required,
                                      std::string* resolved) const {
    const Decision decision = policy_.Decide(request);
    if (decision.verdict != required) {
        return Refuse(decision);
    }
    // The filesystem check, after the lexical one and immediately before
    // acting. A symlink inside the workspace pointing out of it passes
    // every string test there is.
    if (!IsTrulyInside(decision.resolved)) {
        Decision escaped = decision;
        escaped.reason =
            "That path is a link out of the workspace. Only files that really "
            "live under " + policy_.workspace() + " are reachable.";
        escaped.suspicious = true;
        return Refuse(escaped);
    }
    *resolved = decision.resolved;
    ToolResult ok;
    ok.ok = true;
    ok.resolved = decision.resolved;
    return ok;
}

ToolResult ToolRunner::ReadFile(const std::string& path) const {
    Request request;
    request.kind = ToolKind::ReadFile;
    request.path = path;

    std::string resolved;
    ToolResult gate = CheckedResolve(request, Verdict::Allow, &resolved);
    if (!gate.ok) return gate;

    std::error_code error;
    if (!fs::is_regular_file(fs::path(resolved), error) || error) {
        gate.ok = false;
        gate.output = "No readable file at " + path + ".";
        gate.detail = gate.output;
        return gate;
    }

    bool too_big = false;
    const std::string contents = ReadAll(resolved, kMaxReadBytes, &too_big);
    if (too_big) {
        gate.ok = false;
        // Refused with the size named rather than truncated: a truncated
        // file that reads as complete is how a model comes to reason
        // confidently about code that is not there.
        gate.output = "That file is larger than " +
                      std::to_string(kMaxReadBytes / 1024) +
                      " KB. Ask for a specific part of it instead.";
        gate.detail = gate.output;
        return gate;
    }
    gate.output = contents;
    gate.detail = "read " + std::to_string(contents.size()) + " bytes from " +
                  resolved;
    return gate;
}

ToolResult ToolRunner::ListDirectory(const std::string& path) const {
    Request request;
    request.kind = ToolKind::ListDirectory;
    request.path = path.empty() ? "." : path;

    std::string resolved;
    ToolResult gate = CheckedResolve(request, Verdict::Allow, &resolved);
    if (!gate.ok) return gate;

    std::error_code error;
    if (!fs::is_directory(fs::path(resolved), error) || error) {
        gate.ok = false;
        gate.output = "No directory at " + request.path + ".";
        gate.detail = gate.output;
        return gate;
    }

    std::vector<std::string> entries;
    for (const fs::directory_entry& entry :
         fs::directory_iterator(resolved, fs::directory_options::skip_permission_denied,
                                error)) {
        const std::string name = entry.path().filename().string();
        std::error_code kind_error;
        const bool is_dir = entry.is_directory(kind_error);
        std::string line = is_dir ? name + "/" : name;
        // Marked rather than hidden: a person scanning the tree should
        // see that a file is one Studio will not read, and why.
        if (ToolPolicy::IsSensitive(entry.path().string())) {
            line += "    (credentials — not readable)";
        } else if (!IsTrulyInside(entry.path().string())) {
            line += "    (link out of the workspace — not reachable)";
        }
        entries.push_back(line);
    }
    std::sort(entries.begin(), entries.end());

    std::ostringstream out;
    for (const std::string& line : entries) out << line << "\n";
    gate.output = out.str();
    gate.detail = std::to_string(entries.size()) + " entries in " + resolved;
    return gate;
}

ToolResult ToolRunner::WriteFile(const std::string& path,
                                 const std::string& contents,
                                 bool approved) const {
    Request request;
    request.kind = ToolKind::WriteFile;
    request.path = path;
    request.bytes = contents.size();

    if (!approved) {
        Decision refusal = policy_.Decide(request);
        refusal.reason = "Not written: nobody approved it.";
        return Refuse(refusal);
    }

    std::string resolved;
    ToolResult gate = CheckedResolve(request, Verdict::NeedsApproval, &resolved);
    if (!gate.ok) return gate;

    // Written to a temporary in the same directory and renamed over the
    // original: a crash or a full disk halfway through leaves the old
    // file intact rather than a half-written one. Same directory
    // because rename is only atomic within a filesystem.
    const fs::path target(resolved);
    const fs::path scratch = target.parent_path() /
                             (target.filename().string() + ".hnx-tmp");
    {
        std::ofstream stream(scratch, std::ios::binary | std::ios::trunc);
        if (!stream) {
            gate.ok = false;
            gate.output = "Could not write to " + resolved + ".";
            gate.detail = gate.output;
            return gate;
        }
        stream << contents;
    }
    std::error_code error;
    fs::rename(scratch, target, error);
    if (error) {
        fs::remove(scratch, error);
        gate.ok = false;
        gate.output = "Could not replace " + resolved + ".";
        gate.detail = gate.output;
        return gate;
    }
    gate.output = "Wrote " + std::to_string(contents.size()) + " bytes.";
    gate.detail = "wrote " + resolved;
    return gate;
}

ToolResult ToolRunner::CreateFile(const std::string& path,
                                  const std::string& contents,
                                  bool approved) const {
    Request request;
    request.kind = ToolKind::CreateFile;
    request.path = path;
    request.bytes = contents.size();

    if (!approved) {
        Decision refusal = policy_.Decide(request);
        refusal.reason = "Not created: nobody approved it.";
        return Refuse(refusal);
    }

    std::string resolved;
    ToolResult gate = CheckedResolve(request, Verdict::NeedsApproval, &resolved);
    if (!gate.ok) return gate;

    std::error_code error;
    if (fs::exists(fs::path(resolved), error)) {
        // Refused rather than silently becoming a write. The user
        // approved "create a new file", and overwriting an existing one
        // is a different thing than what they agreed to.
        gate.ok = false;
        gate.output = resolved +
                      " already exists. Use write_file to change it, which "
                      "shows the diff first.";
        gate.detail = gate.output;
        return gate;
    }

    fs::create_directories(fs::path(resolved).parent_path(), error);
    std::ofstream stream(resolved, std::ios::binary);
    if (!stream) {
        gate.ok = false;
        gate.output = "Could not create " + resolved + ".";
        gate.detail = gate.output;
        return gate;
    }
    stream << contents;
    gate.output = "Created with " + std::to_string(contents.size()) + " bytes.";
    gate.detail = "created " + resolved;
    return gate;
}

ToolResult ToolRunner::DeleteFile(const std::string& path, bool approved) const {
    Request request;
    request.kind = ToolKind::DeleteFile;
    request.path = path;

    if (!approved) {
        Decision refusal = policy_.Decide(request);
        refusal.reason = "Not deleted: nobody approved it.";
        return Refuse(refusal);
    }

    std::string resolved;
    ToolResult gate = CheckedResolve(request, Verdict::NeedsApproval, &resolved);
    if (!gate.ok) return gate;

    std::error_code error;
    // A single file only. remove_all on a directory a model named is how
    // one wrong path becomes a lost project, and there is no undo.
    if (fs::is_directory(fs::path(resolved), error)) {
        gate.ok = false;
        gate.output =
            "That is a directory. Studio deletes single files only -- remove a "
            "directory yourself if you mean to.";
        gate.detail = gate.output;
        return gate;
    }
    if (!fs::remove(fs::path(resolved), error) || error) {
        gate.ok = false;
        gate.output = "Could not delete " + resolved + ".";
        gate.detail = gate.output;
        return gate;
    }
    gate.output = "Deleted.";
    gate.detail = "deleted " + resolved;
    return gate;
}

}  // namespace hnx
