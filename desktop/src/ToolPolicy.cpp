// ToolPolicy.cpp — see ToolPolicy.h.
//
// Plain C++17 with no Qt: this is the security-critical half of Studio
// and it is worth being able to compile and test it with nothing but a
// compiler. The Qt layer calls into it; it never calls back.

#include "ToolPolicy.h"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace hnx {
namespace {

// Files whose contents are credentials. Reading one and handing it to a
// model is exfiltration whether or not the model is hosted, because the
// transcript outlives the session -- and a model that has been steered
// by a hostile document will ask for exactly these.
//
// Matched on the final path component, case-insensitively, plus a few
// directory names. Not a comprehensive list of every secret that has
// ever been in a file: it is the set that turns up in a source tree,
// which is what a workspace is.
const char* const kSensitiveNames[] = {
    ".env", ".env.local", ".env.production", ".netrc", ".npmrc", ".pypirc",
    "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa",
    "credentials", "credentials.json", "secrets.json", "service-account.json",
    ".git-credentials", ".htpasswd", "shadow", "master.key",
};

const char* const kSensitiveDirs[] = {
    ".ssh", ".gnupg", ".aws", ".kube", ".docker", ".hypernix",
};

std::string ToLower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text;
}

std::vector<std::string> Split(const std::string& path) {
    std::vector<std::string> parts;
    std::string part;
    std::istringstream stream(path);
    while (std::getline(stream, part, '/')) {
        if (!part.empty()) parts.push_back(part);
    }
    return parts;
}

std::string Join(const std::vector<std::string>& parts) {
    std::string out;
    for (const std::string& part : parts) {
        out += "/";
        out += part;
    }
    return out.empty() ? "/" : out;
}

// Collapse "." and "..". Returns false when the path climbs above the
// root it started from, which is the case worth refusing rather than
// clamping: clamping "../../etc/passwd" to the workspace root would turn
// an escape attempt into a successful read of something else.
bool Normalise(const std::vector<std::string>& parts,
               std::vector<std::string>* out) {
    for (const std::string& part : parts) {
        if (part == ".") continue;
        if (part == "..") {
            if (out->empty()) return false;
            out->pop_back();
            continue;
        }
        out->push_back(part);
    }
    return true;
}

bool StartsWith(const std::vector<std::string>& path,
                const std::vector<std::string>& prefix) {
    if (path.size() < prefix.size()) return false;
    return std::equal(prefix.begin(), prefix.end(), path.begin());
}

}  // namespace

ToolPolicy::ToolPolicy(std::string workspace) : workspace_(std::move(workspace)) {
    // A trailing slash would make the prefix comparison in Resolve
    // depend on how the caller spelled the workspace.
    while (workspace_.size() > 1 && workspace_.back() == '/') {
        workspace_.pop_back();
    }
}

std::string ToolPolicy::Resolve(const std::string& path) const {
    if (path.empty()) return {};
    // A NUL byte truncates the path at the syscall boundary, so
    // "safe.txt\0/../../etc/passwd" passes a string check and opens
    // something else. Refused outright rather than trimmed.
    if (path.find('\0') != std::string::npos) return {};

    std::vector<std::string> root;
    if (!Normalise(Split(workspace_), &root)) return {};

    std::vector<std::string> combined;
    if (!path.empty() && path.front() == '/') {
        // An absolute path is taken at face value and then required to
        // be inside the workspace. Not rejected out of hand: the model
        // is shown absolute paths in the file tree, so asking for one
        // back is the normal case.
        if (!Normalise(Split(path), &combined)) return {};
    } else {
        combined = root;
        if (!Normalise(Split(path), &combined)) return {};
    }

    if (!StartsWith(combined, root)) return {};
    return Join(combined);
}

bool ToolPolicy::IsSensitive(const std::string& resolved) {
    const std::vector<std::string> parts = Split(resolved);
    if (parts.empty()) return false;

    const std::string name = ToLower(parts.back());
    for (const char* candidate : kSensitiveNames) {
        if (name == candidate) return true;
    }
    // A private key is usually id_* with no extension, but .pem and .key
    // are the general case and both hold one often enough to ask about.
    const auto dot = name.rfind('.');
    if (dot != std::string::npos) {
        const std::string extension = name.substr(dot);
        if (extension == ".pem" || extension == ".key" || extension == ".p12" ||
            extension == ".pfx" || extension == ".keystore") {
            return true;
        }
    }
    // Any component, not just the parent: .ssh/known_hosts/../id_rsa has
    // already been normalised by the time this is called, but a nested
    // .aws/cli/cache is still credentials.
    for (const std::string& part : parts) {
        const std::string lowered = ToLower(part);
        for (const char* candidate : kSensitiveDirs) {
            if (lowered == candidate) return true;
        }
    }
    return false;
}

std::vector<std::string> ToolPolicy::ToolNames() {
    // Note what is not here: no shell, no exec, no eval, no "run tests",
    // no package install. A model can ask for a file to be written and a
    // person can agree; there is no path by which a model runs code.
    // Adding one would remove the only guarantee in this file that does
    // not depend on a check being correct.
    return {
        "read_file", "list_directory", "write_file", "create_file",
        "delete_file", "web_search",
    };
}

ToolKind ToolPolicy::KindFromName(const std::string& name) {
    if (name == "read_file") return ToolKind::ReadFile;
    if (name == "write_file") return ToolKind::WriteFile;
    if (name == "create_file") return ToolKind::CreateFile;
    if (name == "delete_file") return ToolKind::DeleteFile;
    if (name == "list_directory") return ToolKind::ListDirectory;
    if (name == "web_search") return ToolKind::WebSearch;
    return ToolKind::Unknown;
}

Decision ToolPolicy::Decide(const Request& request) const {
    Decision decision;

    if (request.kind == ToolKind::Unknown) {
        decision.verdict = Verdict::Deny;
        decision.reason =
            "Studio does not have that tool. Nothing in Studio runs commands, "
            "so a request to run one is refused rather than offered.";
        decision.suspicious = true;
        return decision;
    }

    if (request.kind == ToolKind::WebSearch) {
        if (request.query.empty()) {
            decision.verdict = Verdict::Deny;
            decision.reason = "An empty search asks for nothing.";
            return decision;
        }
        // Approval every time, not once per session. A search sends
        // whatever the model chose to put in the query to a third party,
        // and the model chose it from a conversation that may include the
        // contents of your files.
        decision.verdict = Verdict::NeedsApproval;
        decision.reason =
            "This sends the search text to a web search service. Read it "
            "before agreeing: the model wrote it, and it may contain "
            "something from this conversation.";
        return decision;
    }

    decision.resolved = Resolve(request.path);
    if (decision.resolved.empty()) {
        decision.verdict = Verdict::Deny;
        decision.suspicious = true;
        decision.reason =
            "That path is outside the workspace. Only files under " +
            workspace_ + " are reachable, and this one resolves elsewhere.";
        return decision;
    }

    const bool sensitive = IsSensitive(decision.resolved);

    switch (request.kind) {
        case ToolKind::ReadFile:
        case ToolKind::ListDirectory:
            if (sensitive) {
                // Refused, not escalated to approval. There is no
                // legitimate reason for a model to need the contents of
                // a private key, and an approval prompt for one is a
                // prompt people learn to click through.
                decision.verdict = Verdict::Deny;
                decision.suspicious = true;
                decision.reason =
                    "That file holds credentials. Studio will not read it into "
                    "a conversation -- open it yourself if you need to.";
                return decision;
            }
            decision.verdict = Verdict::Allow;
            decision.reason = "Reading a file inside the workspace.";
            return decision;

        case ToolKind::WriteFile:
            decision.verdict = Verdict::NeedsApproval;
            decision.suspicious = sensitive;
            decision.reason = sensitive
                ? "This would overwrite a file that holds credentials."
                : "This overwrites an existing file. The change is shown as a "
                  "diff before anything is written.";
            return decision;

        case ToolKind::CreateFile:
            decision.verdict = Verdict::NeedsApproval;
            decision.suspicious = sensitive;
            decision.reason = sensitive
                ? "This would create a file in a location that holds "
                  "credentials."
                : "This creates a new file in the workspace.";
            return decision;

        case ToolKind::DeleteFile:
            decision.verdict = Verdict::NeedsApproval;
            // Always worth naming: a delete is the one action here with
            // nothing to undo it, whatever the path.
            decision.suspicious = true;
            decision.reason =
                "This deletes a file. Studio has no undo for it -- the file is "
                "gone unless it is in version control.";
            return decision;

        case ToolKind::WebSearch:
        case ToolKind::Unknown:
            break;  // handled above
    }

    decision.verdict = Verdict::Deny;
    decision.reason = "Refused: no rule covers this request.";
    decision.suspicious = true;
    return decision;
}

}  // namespace hnx
