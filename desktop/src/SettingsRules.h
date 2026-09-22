// SettingsRules.h — the settings validation, with no Qt in it.
//
// Split out from StudioSettings for the same reason ToolPolicy has no
// Qt dependency: this is the part where being wrong is expensive, and a
// rule that needs a Qt event loop to exercise is a rule that gets
// tested by hand once and then never again.
//
// Everything here is std::string and int. StudioSettings is the
// QObject that persists these and exposes them to QML; this is what
// decides whether a value is allowed.

#ifndef HYPERNIX_STUDIO_SETTINGS_RULES_H
#define HYPERNIX_STUDIO_SETTINGS_RULES_H

#include <string>

namespace hnx {
namespace rules {

// Smallest context that can hold a system prompt and one turn.
inline constexpr int kMinimumContext = 512;
// Past this, no model here accepts it and the failure arrives as a KV
// cache error rather than as a complaint about the setting.
inline constexpr int kMaximumContext = 1048576;
inline constexpr int kDefaultContext = 8192;

// fish where a human types; bash where a machine runs it. See
// StudioSettings.h for why those are two settings rather than one.
inline const char* kDefaultShell = "fish";
inline const char* kDefaultCodingShell = "bash";

/// Is *name* a shell name we are willing to write a script for?
///
/// A shell here is a name, not a command. Anything carrying a path
/// separator, an argument, a quote or a shell operator is somebody
/// turning a setting into an execution — and the answer is no even
/// though Studio executes nothing, because the value is written into
/// generated scripts that are run elsewhere.
inline bool IsPlausibleShell(const std::string& name) {
    // Trim first: " bash " is a person, not an attack.
    const auto first = name.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return false;                       // empty or all whitespace
    }
    const auto last = name.find_last_not_of(" \t\r\n");
    const std::string trimmed = name.substr(first, last - first + 1);

    if (trimmed.empty() || trimmed.size() > 32) {
        return false;
    }
    for (const unsigned char character : trimmed) {
        const bool ok = (character >= 'a' && character <= 'z') ||
                        (character >= 'A' && character <= 'Z') ||
                        (character >= '0' && character <= '9') ||
                        character == '-' || character == '_' ||
                        character == '.';
        if (!ok) {
            return false;
        }
    }
    // A name that is only dots is `.` or `..` — a directory, not a shell.
    if (trimmed.find_first_not_of('.') == std::string::npos) {
        return false;
    }
    return true;
}

/// The trimmed name, or the default when *name* is not one.
inline std::string ShellOrDefault(const std::string& name,
                                  const std::string& fallback) {
    if (!IsPlausibleShell(name)) {
        return fallback;
    }
    const auto first = name.find_first_not_of(" \t\r\n");
    const auto last = name.find_last_not_of(" \t\r\n");
    return name.substr(first, last - first + 1);
}

/// What a context of *wanted* becomes, given a model ceiling.
///
/// Clamped, never rejected: a caller asking for more than a model has
/// should get the model's maximum, not an error. *why* is filled with
/// the reason when the answer differs from the request, so the UI can
/// say which of the two ceilings was hit rather than silently showing a
/// different number.
///
/// *model_maximum* of 0 means "not known yet" — before a model is
/// loaded there is nothing to clamp against.
inline int ClampContext(int wanted, int model_maximum, std::string* why = nullptr) {
    if (why != nullptr) {
        why->clear();
    }
    int value = wanted;

    if (value < kMinimumContext) {
        value = kMinimumContext;
        if (why != nullptr) {
            *why = "A context below " + std::to_string(kMinimumContext) +
                   " cannot hold a system prompt and one turn, so it was "
                   "raised to the minimum.";
        }
    } else if (value > kMaximumContext) {
        value = kMaximumContext;
        if (why != nullptr) {
            *why = std::to_string(wanted) +
                   " is past what any model here accepts; using " +
                   std::to_string(kMaximumContext) + ".";
        }
    }

    if (model_maximum > 0 && value > model_maximum) {
        if (why != nullptr) {
            *why = "This model tops out at " + std::to_string(model_maximum) +
                   " tokens, so " + std::to_string(wanted) +
                   " was reduced to that.";
        }
        value = model_maximum;
    }
    return value;
}

}  // namespace rules
}  // namespace hnx

#endif  // HYPERNIX_STUDIO_SETTINGS_RULES_H
