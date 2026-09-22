// StudioSettings.h — what Studio remembers between launches.
//
// Studio had no settings at all: every preference was a default
// compiled in, so "use a bigger context" meant editing a constant and
// rebuilding. This is the store, and it is deliberately small — three
// things worth persisting, validated on the way in and on the way out.
//
// Validated on the way *out* as well, because QSettings reads an INI
// file a person can edit, and a context limit of -1 or a shell of
// `rm -rf /` has to become a sane default rather than reach anything.
//
// The shells
// ----------
// Two of them, because they are for different jobs.
//
// `shell` is the interactive one — fish by default. It is what a
// person is sitting in front of, and fish's completions and history
// are the reason to pick it.
//
// `codingShell` is the one generated scripts target — bash by default,
// and that is not a preference. A script Studio writes gets committed,
// run by CI, run on a server, and pasted into a Dockerfile; fish is not
// installed in any of those places, and fish's syntax for the things
// scripts do (`set -x`, test brackets, `$status` vs `$?`) is different
// enough that a fish script fails on a POSIX shell in ways that look
// like the script is wrong rather than the interpreter.
//
// So: fish where a human types, bash where a machine runs it. Both
// configurable; neither the same setting.
//
// There is still no shell *tool*. Studio does not run commands — see
// ToolPolicy.h for why that is a guarantee rather than a default. These
// settings say which shell to write *for*, not which shell to execute
// in.

#ifndef HYPERNIX_STUDIO_SETTINGS_H
#define HYPERNIX_STUDIO_SETTINGS_H

#include <QtCore/QObject>
#include <QtCore/QString>
#include <QtCore/QStringList>

namespace hnx {

class StudioSettings : public QObject {
    Q_OBJECT

    Q_PROPERTY(QString shell READ shell WRITE setShell NOTIFY changed)
    Q_PROPERTY(QString codingShell READ codingShell WRITE setCodingShell
                   NOTIFY changed)
    Q_PROPERTY(int contextLimit READ contextLimit WRITE setContextLimit
                   NOTIFY changed)
    Q_PROPERTY(QStringList knownShells READ knownShells CONSTANT)
    Q_PROPERTY(int minimumContext READ minimumContext CONSTANT)
    Q_PROPERTY(int maximumContext READ maximumContext CONSTANT)

public:
    explicit StudioSettings(QObject* parent = nullptr);

    // Defaults. Named rather than inline so the tests can assert on the
    // same constants the code uses, instead of on a repeated literal.
    static const char* kDefaultShell;         // "fish"
    static const char* kDefaultCodingShell;   // "bash"
    static constexpr int kDefaultContext = 8192;

    // A context below this cannot hold a system prompt and one turn; a
    // context above it is past what any model here accepts, and asking
    // for it fails at load with a message about KV cache rather than
    // about the setting.
    static constexpr int kMinimumContext = 512;
    static constexpr int kMaximumContext = 1048576;

    QString shell() const { return shell_; }
    QString codingShell() const { return codingShell_; }
    int contextLimit() const { return contextLimit_; }

    int minimumContext() const { return kMinimumContext; }
    int maximumContext() const { return kMaximumContext; }

    // The shells offered in the picker. A person may type another; this
    // is the list, not the law.
    QStringList knownShells() const;

    Q_INVOKABLE void setShell(const QString& value);
    Q_INVOKABLE void setCodingShell(const QString& value);
    Q_INVOKABLE void setContextLimit(int value);
    Q_INVOKABLE void resetToDefaults();

    // Clamped, never rejected: a caller asking for more context than a
    // model has should get the model's maximum, not an error. Returning
    // the reason too, so the UI can say why it did not get what it
    // asked for rather than silently showing a different number.
    static int clampContext(int wanted, int modelMaximum, QString* why = nullptr);

    // Whether *name* is a shell we are willing to write a script for.
    // An empty or whitespace-only name is not; neither is one carrying
    // an argument or a path separator, because "which shell" is a name
    // and anything else is somebody trying to make it a command.
    static bool isPlausibleShell(const QString& name);

signals:
    void changed();

private:
    void load();
    void save();

    QString shell_;
    QString codingShell_;
    int contextLimit_ = kDefaultContext;
};

}  // namespace hnx

#endif  // HYPERNIX_STUDIO_SETTINGS_H
