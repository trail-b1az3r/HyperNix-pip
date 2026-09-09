// StudioBridge.h — the one object QML talks to.
//
// Everything the UI needs, on the main thread, as properties and
// invokables. QML never touches HyperLinkClient or ToolRunner directly:
// a view that could call ToolRunner could call it without asking, and
// the whole design here is that a tool runs only after a person agrees.
//
// The approval flow, which is the point of this class
// ---------------------------------------------------
//   1. HyperLinkClient emits toolCallRequested. Nothing has happened.
//   2. This asks ToolPolicy. A Deny never reaches the user at all —
//      there is nothing to approve about reading /etc/shadow, and a
//      prompt for it is a prompt people learn to click through.
//   3. An Allow (read, list, inside the workspace, not credentials)
//      runs immediately. That is the only case that does.
//   4. A NeedsApproval becomes a `pendingApproval` the QML dialog
//      renders, with the resolved path and a diff. Nothing runs until
//      approveTool() is called, and approveTool() is the only path from
//      here into ToolRunner's mutating methods.
//
// There is no "approve all", no "remember this", and no timeout that
// defaults to yes. Each of those would be the same feature: a way for a
// tool call to happen without anyone reading it.

#ifndef HYPERNIX_STUDIO_BRIDGE_H
#define HYPERNIX_STUDIO_BRIDGE_H

#include <QtCore/QJsonArray>
#include <QtCore/QJsonObject>
#include <QtCore/QObject>
#include <QtCore/QString>
#include <QtCore/QVariantList>
#include <QtCore/QVariantMap>

#include <memory>

#include "HyperLinkClient.h"
#include "LocalSession.h"
#include "ToolRunner.h"

namespace hnx {

class StudioBridge : public QObject {
    Q_OBJECT

    Q_PROPERTY(bool connected READ connected NOTIFY statusChanged)
    Q_PROPERTY(QString serverName READ serverName NOTIFY statusChanged)
    Q_PROPERTY(QString serverVersion READ serverVersion NOTIFY statusChanged)
    Q_PROPERTY(QString serverFingerprint READ serverFingerprint
                   NOTIFY statusChanged)
    Q_PROPERTY(QString status READ status NOTIFY statusChanged)
    Q_PROPERTY(QString workspace READ workspace NOTIFY workspaceChanged)

    Q_PROPERTY(QVariantList models READ models NOTIFY modelsChanged)
    Q_PROPERTY(QString activeModel READ activeModel NOTIFY modelsChanged)
    Q_PROPERTY(QVariantList gpus READ gpus NOTIFY resourcesChanged)
    Q_PROPERTY(double cpuPercent READ cpuPercent NOTIFY resourcesChanged)
    Q_PROPERTY(int ramUsedMb READ ramUsedMb NOTIFY resourcesChanged)
    Q_PROPERTY(int ramTotalMb READ ramTotalMb NOTIFY resourcesChanged)

    Q_PROPERTY(QVariantList messages READ messages NOTIFY messagesChanged)
    Q_PROPERTY(bool busy READ busy NOTIFY busyChanged)
    // Null when nothing is waiting. The dialog binds its visibility to
    // this, so a pending call cannot be dismissed by navigating away.
    Q_PROPERTY(QVariantMap pendingApproval READ pendingApproval
                   NOTIFY pendingApprovalChanged)

    // "server" or "local". Studio began as a client and the server path
    // is unchanged; this decides which one send() goes down.
    Q_PROPERTY(QString source READ source WRITE setSource NOTIFY sourceChanged)
    Q_PROPERTY(bool localAvailable READ localAvailable CONSTANT)
    Q_PROPERTY(QString localUnavailableReason READ localUnavailableReason
                   CONSTANT)
    Q_PROPERTY(QVariantList localModels READ localModels NOTIFY localChanged)
    Q_PROPERTY(bool localLoaded READ localLoaded NOTIFY localChanged)
    Q_PROPERTY(QVariantMap localInfo READ localInfo NOTIFY localChanged)
    Q_PROPERTY(QStringList modelFolders READ modelFolders NOTIFY localChanged)
    // True when a model can answer, whichever way. What the composer
    // binds to, so it does not have to know which mode it is in.
    Q_PROPERTY(bool ready READ ready NOTIFY readyChanged)

public:
    explicit StudioBridge(QObject* parent = nullptr);
    ~StudioBridge() override;

    bool connected() const { return connected_; }
    QString serverName() const { return serverName_; }
    QString serverVersion() const { return serverVersion_; }
    QString serverFingerprint() const;
    QString status() const { return status_; }
    QString workspace() const;
    QVariantList models() const;
    QString activeModel() const { return activeModel_; }
    QVariantList gpus() const;
    double cpuPercent() const;
    int ramUsedMb() const;
    int ramTotalMb() const;
    QVariantList messages() const { return messages_; }
    bool busy() const { return busy_; }
    QVariantMap pendingApproval() const { return pendingApproval_; }

    QString source() const { return source_; }
    void setSource(const QString& value);
    bool localAvailable() const { return LocalSession::available(); }
    QString localUnavailableReason() const {
        return LocalSession::unavailableReason();
    }
    QVariantList localModels() const { return local_.models(); }
    bool localLoaded() const { return local_.loaded(); }
    QVariantMap localInfo() const { return local_.info(); }
    QStringList modelFolders() const { return modelFolders_; }
    bool ready() const;

public slots:
    // Connection. `key` is a T2S key; Studio never asks for an admin
    // one, because nothing it does needs administration.
    void connectTo(const QString& address, const QString& key);
    void disconnectFromServer();
    void chooseWorkspace(const QString& path);

    void refresh();
    void switchModel(const QString& modelId, int gpuLayers);
    void send(const QString& text);
    void clearConversation();
    void searchModels(const QString& query);
    void download(const QString& repoId, const QString& filename);

    // The only route from the UI into a mutating tool.
    void approveTool();
    void rejectTool();

    // Local models. All no-ops in a build without llama.cpp, except
    // scanning -- the catalogue is always there, so a Studio that
    // cannot run a model can still tell you what is on the disk and
    // why it cannot run it.
    void scanLocalModels();
    void addModelFolder(const QString& path);
    void removeModelFolder(const QString& path);
    void loadLocalModel(const QString& path, int gpuLayers, int contextLength);
    void unloadLocalModel();
    void stopGenerating();

    // For the file tree. Read-only and policy-checked like everything
    // else, so a view cannot list its way outside the workspace.
    QVariantList listWorkspace(const QString& relative);
    QString readWorkspaceFile(const QString& relative);

    // Human-readable, grouped in eights, matching what
    // `hypernix-t1 status` prints so the two can be compared by eye.
    QString displayFingerprint() const;

signals:
    void statusChanged();
    void workspaceChanged();
    void modelsChanged();
    void resourcesChanged();
    void messagesChanged();
    void busyChanged();
    void pendingApprovalChanged();
    void sourceChanged();
    void localChanged();
    void readyChanged();
    // A mismatch is not a toast. The UI puts the app into a blocked
    // state until the user re-pairs or dismisses deliberately, because
    // "something else is answering at this address" is not information
    // to scroll past.
    void identityMismatch(const QString& expected, const QString& actual);
    void error(const QString& what, const QString& detail);

private:
    void appendMessage(const QString& role, const QString& text,
                       const QString& detail = QString());
    void onToolCall(const QString& name, const QJsonObject& arguments,
                    const QString& callId);
    void runApproved(const QString& name, const QJsonObject& arguments,
                     const QString& callId, bool approved);
    void reportToolResult(const QString& callId, const ToolResult& result);
    void setStatus(const QString& text);
    void setBusy(bool value);
    QJsonArray toolSchema() const;
    QJsonArray transcript() const;

    void sendToServer(const QString& text);
    void sendToLocal(const QString& text);
    QString localPrompt() const;
    void loadModelFolders();
    void saveModelFolders();

    HyperLinkClient client_;
    LocalSession local_;
    std::unique_ptr<ToolRunner> runner_;

    QString source_ = "server";
    QStringList modelFolders_;
    // Where the streaming reply is being assembled, so each token can
    // extend the message already on screen instead of appending a new
    // one per token.
    int streamingIndex_ = -1;

    bool connected_ = false;
    bool busy_ = false;
    QString serverName_;
    QString serverVersion_;
    QString status_ = "Not connected";
    QString activeModel_;
    QVariantList messages_;

    QVariantMap pendingApproval_;
    QString pendingName_;
    QJsonObject pendingArguments_;
    QString pendingCallId_;
};

}  // namespace hnx

#endif  // HYPERNIX_STUDIO_BRIDGE_H
