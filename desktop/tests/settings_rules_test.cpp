// settings_rules_test.cpp — the settings validation, checked.
//
// Two rules that matter, and one that only looks cosmetic.
//
// The context clamp decides what a model is asked for. Get it wrong
// upward and the load fails with a KV cache error that reads as a bug
// in Studio; get it wrong downward and every conversation truncates
// early for no stated reason. Both are silent, which is why the clamp
// reports *why* it changed a number.
//
// The shell name looks cosmetic and is not. The value is written into
// generated scripts that get committed, run by CI, and pasted into
// Dockerfiles — so a "shell name" carrying an argument or an operator
// is a setting being turned into an execution, somewhere Studio cannot
// see.
//
//   c++ -std=c++17 -O1 -Wall -Wextra -Werror settings_rules_test.cpp -o t && ./t

#include "../src/SettingsRules.h"

#include <cstdio>
#include <string>

namespace {

int failures = 0;
int checks = 0;

void Check(bool condition, const std::string& what) {
    checks++;
    if (!condition) {
        failures++;
        std::printf("  FAIL: %s\n", what.c_str());
    }
}

using namespace hnx::rules;

void ShellNames() {
    // The defaults, which are the whole point of the pair.
    Check(IsPlausibleShell(kDefaultShell), "fish is a shell");
    Check(IsPlausibleShell(kDefaultCodingShell), "bash is a shell");
    Check(std::string(kDefaultShell) == "fish",
          "the interactive default is fish");
    Check(std::string(kDefaultCodingShell) == "bash",
          "the coding default is bash -- generated scripts run where "
          "fish is not installed");

    Check(IsPlausibleShell("zsh"), "zsh");
    Check(IsPlausibleShell("pwsh"), "pwsh");
    Check(IsPlausibleShell("  bash  "), "surrounding space is a person, not an attack");

    // Not names.
    Check(!IsPlausibleShell(""), "empty");
    Check(!IsPlausibleShell("   "), "whitespace only");
    Check(!IsPlausibleShell("/bin/bash"), "a path is not a name");
    Check(!IsPlausibleShell("bash -c"), "an argument is not a name");
    Check(!IsPlausibleShell("bash;rm -rf /"), "an operator is not a name");
    Check(!IsPlausibleShell("bash && curl x"), "chained commands");
    Check(!IsPlausibleShell("$(whoami)"), "a substitution");
    Check(!IsPlausibleShell("`id`"), "a backtick substitution");
    Check(!IsPlausibleShell("bash\nrm"), "a newline");
    Check(!IsPlausibleShell(".."), "a directory is not a shell");
    Check(!IsPlausibleShell(std::string(40, 'a')), "absurdly long");

    // Falling back rather than accepting nonsense.
    Check(ShellOrDefault("/bin/sh -c evil", "bash") == "bash",
          "a rejected name becomes the default");
    Check(ShellOrDefault("  zsh ", "bash") == "zsh",
          "an accepted name comes back trimmed");
}

void ContextClamping() {
    std::string why;

    // Inside the range, nothing happens and nothing is said.
    Check(ClampContext(8192, 0, &why) == 8192, "a sane value is kept");
    Check(why.empty(), "and says nothing about it");

    // Below the floor.
    Check(ClampContext(10, 0, &why) == kMinimumContext, "raised to the floor");
    Check(!why.empty(), "and says why it was raised");
    Check(ClampContext(0, 0) == kMinimumContext, "zero is raised too");
    Check(ClampContext(-5000, 0) == kMinimumContext,
          "a hand-edited negative becomes the floor, not a crash");

    // Above the ceiling.
    Check(ClampContext(99999999, 0, &why) == kMaximumContext, "capped");
    Check(!why.empty(), "and says why");

    // The model's own ceiling wins when it is lower. This is the one
    // that otherwise fails at load with a message about the KV cache.
    Check(ClampContext(32768, 4096, &why) == 4096, "the model's limit wins");
    Check(why.find("4096") != std::string::npos,
          "and the reason names the model's number");
    Check(why.find("32768") != std::string::npos,
          "and what was asked for");

    // A model ceiling of 0 means "not loaded yet" -- nothing to clamp to.
    Check(ClampContext(32768, 0) == 32768, "no model means no model clamp");

    // A model bigger than the request does not raise it.
    Check(ClampContext(4096, 131072) == 4096,
          "a roomy model does not inflate the request");

    // The two ceilings together: the request is absurd AND the model is
    // small. The model's number is the one that matters to the reader.
    Check(ClampContext(99999999, 8192, &why) == 8192, "both ceilings");
    Check(why.find("8192") != std::string::npos, "the model's one is reported");
}

void Defaults() {
    Check(kDefaultContext >= kMinimumContext, "the default is not below the floor");
    Check(kDefaultContext <= kMaximumContext, "nor above the ceiling");
    Check(ClampContext(kDefaultContext, 0) == kDefaultContext,
          "the default survives its own clamp");
}

}  // namespace

int main() {
    ShellNames();
    ContextClamping();
    Defaults();
    std::printf("%d checks, %d failures\n", checks, failures);
    std::printf("interactive shell is %s, coding shell is %s\n",
                kDefaultShell, kDefaultCodingShell);
    return failures == 0 ? 0 : 1;
}
