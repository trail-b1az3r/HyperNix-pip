#include "StudioSettings.h"

#include <QtCore/QSettings>

#include "SettingsRules.h"

#include <algorithm>
#include <string>

namespace hnx {

const char* StudioSettings::kDefaultShell = "fish";
const char* StudioSettings::kDefaultCodingShell = "bash";

namespace {

constexpr char kOrganisation[] = "HyperNix";
constexpr char kApplication[] = "Studio";
constexpr char kShellKey[] = "shell/interactive";
constexpr char kCodingShellKey[] = "shell/coding";
constexpr char kContextKey[] = "model/contextLimit";

}  // namespace

StudioSettings::StudioSettings(QObject* parent) : QObject(parent) {
    load();
}

QStringList StudioSettings::knownShells() const {
    // fish first because it is the interactive default; bash second
    // because it is the coding one. The rest are there so a person on
    // zsh does not have to type it.
    return {"fish", "bash", "zsh", "sh", "dash", "ksh", "nu", "pwsh"};
}

bool StudioSettings::isPlausibleShell(const QString& name) {
    // Delegated so the rule has exactly one definition, and that one
    // is in a header with no Qt in it that the tests compile directly.
    return rules::IsPlausibleShell(name.trimmed().toStdString());
}

int StudioSettings::clampContext(int wanted, int modelMaximum, QString* why) {
    std::string reason;
    const int value = rules::ClampContext(wanted, modelMaximum, &reason);
    if (why != nullptr) {
        *why = QString::fromStdString(reason);
    }
    return value;
}

void StudioSettings::setShell(const QString& value) {
    if (!isPlausibleShell(value)) {
        return;
    }
    const QString trimmed = value.trimmed();
    if (trimmed == shell_) {
        return;
    }
    shell_ = trimmed;
    save();
    emit changed();
}

void StudioSettings::setCodingShell(const QString& value) {
    if (!isPlausibleShell(value)) {
        return;
    }
    const QString trimmed = value.trimmed();
    if (trimmed == codingShell_) {
        return;
    }
    codingShell_ = trimmed;
    save();
    emit changed();
}

void StudioSettings::setContextLimit(int value) {
    const int clamped = clampContext(value, 0);
    if (clamped == contextLimit_) {
        return;
    }
    contextLimit_ = clamped;
    save();
    emit changed();
}

void StudioSettings::resetToDefaults() {
    shell_ = QString::fromLatin1(kDefaultShell);
    codingShell_ = QString::fromLatin1(kDefaultCodingShell);
    contextLimit_ = kDefaultContext;
    save();
    emit changed();
}

void StudioSettings::load() {
    QSettings store(QString::fromLatin1(kOrganisation),
                    QString::fromLatin1(kApplication));

    // Validated on the way out, not just on the way in: QSettings reads
    // an INI file a person can edit, so a hand-written context of -1 or
    // a shell of "rm -rf /" has to become a default rather than reach
    // anything.
    const QString storedShell =
        store.value(QString::fromLatin1(kShellKey)).toString();
    shell_ = isPlausibleShell(storedShell)
                 ? storedShell.trimmed()
                 : QString::fromLatin1(kDefaultShell);

    const QString storedCoding =
        store.value(QString::fromLatin1(kCodingShellKey)).toString();
    codingShell_ = isPlausibleShell(storedCoding)
                       ? storedCoding.trimmed()
                       : QString::fromLatin1(kDefaultCodingShell);

    bool ok = false;
    const int storedContext =
        store.value(QString::fromLatin1(kContextKey)).toInt(&ok);
    contextLimit_ = ok ? clampContext(storedContext, 0) : kDefaultContext;
}

void StudioSettings::save() {
    QSettings store(QString::fromLatin1(kOrganisation),
                    QString::fromLatin1(kApplication));
    store.setValue(QString::fromLatin1(kShellKey), shell_);
    store.setValue(QString::fromLatin1(kCodingShellKey), codingShell_);
    store.setValue(QString::fromLatin1(kContextKey), contextLimit_);
}

}  // namespace hnx
