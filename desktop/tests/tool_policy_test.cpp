// tool_policy_test.cpp — the refusals, checked.
//
// ToolPolicy is the part of Studio that stands between a language model
// and somebody's filesystem. Everything else in the app is a
// convenience; this is the part where being wrong costs someone their
// files or their keys.
//
// It has no Qt dependency precisely so this can be built with a compiler
// and nothing else, and run everywhere -- including on the machine it
// was written on, which is where the escapes below were found.
//
//   g++ -std=c++17 -O2 tool_policy_test.cpp ../src/ToolPolicy.cpp -o t && ./t

#include "../src/ToolPolicy.h"

#include <cstdio>
#include <string>

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

hnx::Request Read(const std::string& path) {
    hnx::Request request;
    request.kind = hnx::ToolKind::ReadFile;
    request.path = path;
    return request;
}

hnx::Request Write(const std::string& path) {
    hnx::Request request;
    request.kind = hnx::ToolKind::WriteFile;
    request.path = path;
    return request;
}

const char* VerdictName(hnx::Verdict verdict) {
    switch (verdict) {
        case hnx::Verdict::Allow: return "Allow";
        case hnx::Verdict::NeedsApproval: return "NeedsApproval";
        case hnx::Verdict::Deny: return "Deny";
    }
    return "?";
}

// --- the sandbox ---------------------------------------------------------

void TestPathsStayInside() {
    std::printf("the workspace boundary\n");
    const hnx::ToolPolicy policy("/home/dev/project");

    Check(policy.Decide(Read("src/main.cpp")).verdict == hnx::Verdict::Allow,
          "a relative path inside is allowed");
    Check(policy.Decide(Read("/home/dev/project/src/main.cpp")).verdict ==
              hnx::Verdict::Allow,
          "an absolute path inside is allowed");
    Check(policy.Decide(Read("./src/../src/main.cpp")).verdict ==
              hnx::Verdict::Allow,
          "a path that wanders and comes back is allowed");

    // The escapes. Each of these is what a model steered by a hostile
    // document asks for, and each has to be refused on the resolved
    // path rather than on the string.
    const char* escapes[] = {
        "../secrets.txt",
        "../../etc/passwd",
        "src/../../../../etc/shadow",
        "/etc/passwd",
        "/home/dev/other-project/notes.md",
        // Sibling with a shared prefix: the string starts with the
        // workspace path, so a prefix check on the raw string lets it
        // through.
        "/home/dev/project-backup/secrets",
        "/home/dev/project/../project-evil/x",
    };
    for (const char* path : escapes) {
        const hnx::Decision decision = policy.Decide(Read(path));
        Check(decision.verdict == hnx::Verdict::Deny,
              std::string("refused: ") + path);
        Check(decision.suspicious,
              std::string("and flagged as suspicious: ") + path);
    }

    // A NUL truncates the path at the syscall boundary, so a string
    // check passes and open() sees something shorter.
    std::string poisoned = "safe.txt";
    poisoned.push_back('\0');
    poisoned += "/../../etc/passwd";
    Check(policy.Decide(Read(poisoned)).verdict == hnx::Verdict::Deny,
          "an embedded NUL is refused");

    Check(policy.Decide(Read("")).verdict == hnx::Verdict::Deny,
          "an empty path is refused");

    // The resolved path is what gets shown to the user, and it can
    // differ from what the model asked for -- which is the point of
    // showing it.
    const hnx::Decision resolved = policy.Decide(Read("./src/./main.cpp"));
    Check(resolved.resolved == "/home/dev/project/src/main.cpp",
          "the resolved path is reported, normalised");
}

void TestTrailingSlashesDoNotMatter() {
    std::printf("workspace spelling\n");
    const hnx::ToolPolicy bare("/home/dev/project");
    const hnx::ToolPolicy slashed("/home/dev/project/");

    Check(bare.Resolve("a.txt") == slashed.Resolve("a.txt"),
          "a trailing slash on the workspace changes nothing");
    Check(slashed.Decide(Read("../x")).verdict == hnx::Verdict::Deny,
          "and does not open an escape");
}

// --- credentials ---------------------------------------------------------

void TestCredentialsAreNotReadable() {
    std::printf("files that hold credentials\n");
    const hnx::ToolPolicy policy("/home/dev/project");

    const char* secrets[] = {
        ".env", "config/.env.production", ".netrc", "deploy/id_rsa",
        "keys/server.pem", "certs/client.p12", "secrets.json",
        ".ssh/config", ".aws/credentials", "nested/.gnupg/pubring.kbx",
        "SECRETS.JSON",            // case
        ".git-credentials",
    };
    for (const char* path : secrets) {
        const hnx::Decision decision = policy.Decide(Read(path));
        // Denied, not escalated to approval: there is no legitimate
        // reason for a model to need a private key, and a prompt for
        // one is a prompt people learn to click through.
        Check(decision.verdict == hnx::Verdict::Deny,
              std::string("not readable: ") + path);
    }

    Check(policy.Decide(Read("README.md")).verdict == hnx::Verdict::Allow,
          "an ordinary file still is");
    Check(policy.Decide(Read("environment.ts")).verdict == hnx::Verdict::Allow,
          "and a name that merely looks like one is not caught");
    Check(policy.Decide(Read("src/keyboard.cpp")).verdict == hnx::Verdict::Allow,
          "nor is a file with 'key' in the middle of its name");

    // Writing one is a different question: overwriting a .env is a
    // plausible thing to ask for and a terrible thing to do silently,
    // so it needs approval and is flagged.
    const hnx::Decision overwrite = policy.Decide(Write(".env"));
    Check(overwrite.verdict == hnx::Verdict::NeedsApproval,
          "overwriting one asks first");
    Check(overwrite.suspicious, "and is flagged when it does");
}

// --- everything that changes the machine ---------------------------------

void TestMutationsNeedAHuman() {
    std::printf("changes to the machine\n");
    const hnx::ToolPolicy policy("/home/dev/project");

    const hnx::ToolKind mutating[] = {
        hnx::ToolKind::WriteFile,
        hnx::ToolKind::CreateFile,
        hnx::ToolKind::DeleteFile,
    };
    for (hnx::ToolKind kind : mutating) {
        hnx::Request request;
        request.kind = kind;
        request.path = "src/main.cpp";
        const hnx::Decision decision = policy.Decide(request);
        Check(decision.verdict == hnx::Verdict::NeedsApproval,
              std::string("needs approval, verdict was ") +
                  VerdictName(decision.verdict));
        Check(!decision.reason.empty(),
              "and says what will happen, in the user's terms");
    }

    hnx::Request remove;
    remove.kind = hnx::ToolKind::DeleteFile;
    remove.path = "notes.md";
    Check(policy.Decide(remove).suspicious,
          "a delete is always flagged -- there is no undo for it");

    // Read-only is the only thing that happens on the model's word.
    Check(policy.Decide(Read("src/main.cpp")).verdict == hnx::Verdict::Allow,
          "a read inside the workspace does not");
    hnx::Request list;
    list.kind = hnx::ToolKind::ListDirectory;
    list.path = "src";
    Check(policy.Decide(list).verdict == hnx::Verdict::Allow,
          "nor does a directory listing");
}

void TestWebSearchAsksEveryTime() {
    std::printf("web search\n");
    const hnx::ToolPolicy policy("/home/dev/project");

    hnx::Request search;
    search.kind = hnx::ToolKind::WebSearch;
    search.query = "how to fix a segfault in main.cpp";
    const hnx::Decision decision = policy.Decide(search);

    // Every time, not once per session: the model writes the query, and
    // it writes it from a conversation that may contain the contents of
    // your files.
    Check(decision.verdict == hnx::Verdict::NeedsApproval,
          "a search asks first");
    Check(decision.reason.find("model wrote it") != std::string::npos,
          "and says who wrote the text being sent");

    search.query.clear();
    Check(policy.Decide(search).verdict == hnx::Verdict::Deny,
          "an empty search is refused");
}

// --- the tool surface ----------------------------------------------------

void TestThereIsNoShell() {
    std::printf("the tool surface\n");
    const std::vector<std::string> names = hnx::ToolPolicy::ToolNames();

    // The guarantee that does not depend on a check being correct.
    const char* forbidden[] = {
        "shell", "exec", "run", "bash", "sh", "eval", "spawn", "system",
        "run_command", "terminal", "install", "npm", "pip",
    };
    for (const char* banned : forbidden) {
        bool present = false;
        for (const std::string& name : names) {
            if (name.find(banned) != std::string::npos) present = true;
        }
        Check(!present, std::string("no tool named anything like '") + banned + "'");
    }

    Check(names.size() == 6, "six tools, all of them file or search");
    Check(hnx::ToolPolicy::KindFromName("read_file") == hnx::ToolKind::ReadFile,
          "the names map to kinds");

    // An unknown tool is refused rather than falling through a default.
    Check(hnx::ToolPolicy::KindFromName("shell") == hnx::ToolKind::Unknown,
          "a tool that does not exist is Unknown");
    const hnx::ToolPolicy policy("/home/dev/project");
    hnx::Request unknown;
    unknown.kind = hnx::ToolKind::Unknown;
    const hnx::Decision decision = policy.Decide(unknown);
    Check(decision.verdict == hnx::Verdict::Deny, "and Unknown is denied");
    Check(decision.reason.find("runs commands") != std::string::npos,
          "with a reason that says why there is no such tool");
}

}  // namespace

int main() {
    std::printf("ToolPolicy\n\n");
    TestPathsStayInside();
    TestTrailingSlashesDoNotMatter();
    TestCredentialsAreNotReadable();
    TestMutationsNeedAHuman();
    TestWebSearchAsksEveryTime();
    TestThereIsNoShell();

    std::printf("\n%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
