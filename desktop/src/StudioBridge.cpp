// StudioBridge.cpp — see StudioBridge.h.

#include "StudioBridge.h"

#include <QtCore/QDir>
#include <QtCore/QFileInfo>
#include <QtCore/QJsonDocument>
#include <QtCore/QJsonValue>
#include <QtCore/QSettings>
#include <QtCore/QUrl>

namespace hnx {
namespace {

QString FromLocalFileUrl(const QString& maybeUrl) {
    // QML's FolderDialog hands back a file:// URL; a user typing a path
    // hands back a path. Both have to work, and QUrl leaves a plain
    // path alone.
    const QUrl url(maybeUrl);
    return url.isLocalFile() ? url.toLocalFile() : maybeUrl;
}

QVariantMap ToMap(const ModelInfo& info) {
    QVariantMap map;
    map["id"] = info.id;
    map["displayName"] = info.displayName.isEmpty() ? info.id : info.displayName;
    map["architecture"] = info.architecture;
    map["quantisation"] = info.quantisation;
    map["parametersB"] = info.parametersB;
    map["contextLimit"] = info.contextLimit;
    map["loaded"] = info.loaded;
    map["backend"] = info.backend;
    return map;
}

QVariantMap ToMap(const GpuInfo& gpu) {
    QVariantMap map;
    map["index"] = gpu.index;
    map["vendor"] = gpu.vendor;
    map["name"] = gpu.name;
    map["framework"] = gpu.framework;
    // -1 travels through to QML and the views render "—" for it. A
    // measurement the vendor tool declined to give is not zero, and a
    // full bar of nothing is a lie a dashboard tells confidently.
    map["memoryTotalMb"] = gpu.memoryTotalMb;
    map["memoryUsedMb"] = gpu.memoryUsedMb;
    map["utilisation"] = gpu.utilisation;
    return map;
}

}  // namespace

StudioBridge::StudioBridge(QObject* parent) : QObject(parent) {
    connect(&client_, &HyperLinkClient::connected, this,
            [this](const QString& name, const QString& version, bool keyless) {
                connected_ = true;
                serverName_ = name;
                serverVersion_ = version;
                setStatus(keyless
                              ? name + " — this network is trusted, no key needed"
                              : name + " — connected");
                emit statusChanged();
                refresh();
            });

    connect(&client_, &HyperLinkClient::identityMismatch, this,
            [this](const QString& expected, const QString& actual) {
                // Disconnected outright, not warned about. The credential
                // is already withheld by the client; dropping the
                // connection is what stops the next action from being
                // sent to whatever is answering there.
                connected_ = false;
                setStatus("This address is answering for a different machine");
                emit statusChanged();
                emit identityMismatch(expected, actual);
            });

    connect(&client_, &HyperLinkClient::modelsChanged, this,
            [this]() { emit modelsChanged(); });
    connect(&client_, &HyperLinkClient::resourcesChanged, this,
            [this]() { emit resourcesChanged(); });
    connect(&client_, &HyperLinkClient::failed, this,
            [this](const QString& what, const QString& detail) {
                setBusy(false);
                emit error(what, detail);
            });

    connect(&client_, &HyperLinkClient::chatFinished, this,
            [this](const QJsonObject& reply) {
                setBusy(false);
                const QString text = reply.value("content").toString();
                if (!text.isEmpty()) appendMessage("assistant", text);
            });

    connect(&client_, &HyperLinkClient::toolCallRequested, this,
            &StudioBridge::onToolCall);

    connect(&client_, &HyperLinkClient::searchResults, this,
            [this](const QJsonArray& results) {
                setBusy(false);
                QStringList names;
                for (const QJsonValue& value : results) {
                    names << value.toObject().value("filename").toString();
                }
                appendMessage("system",
                              names.isEmpty()
                                  ? "No downloadable files at that link."
                                  : "Found: " + names.join(", "));
            });

    // The workspace persists; the key does not. A T2S key is a
    // credential for somebody's machine and Studio has no business
    // holding one across launches without being asked -- the same
    // reasoning as HyperLink's admin credential store.
    QSettings settings("HyperNix", "Studio");
    const QString saved = settings.value("workspace").toString();
    if (!saved.isEmpty()) chooseWorkspace(saved);
}

StudioBridge::~StudioBridge() = default;

QString StudioBridge::serverFingerprint() const {
    return client_.pinnedFingerprint();
}

QString StudioBridge::displayFingerprint() const {
    const QString value = client_.pinnedFingerprint();
    QStringList groups;
    for (int i = 0; i < value.size(); i += 8) groups << value.mid(i, 8);
    return groups.join(' ');
}

QString StudioBridge::workspace() const {
    return runner_ ? QString::fromStdString(runner_->policy().workspace())
                   : QString();
}

QVariantList StudioBridge::models() const {
    QVariantList out;
    for (const ModelInfo& info : client_.models()) out.append(ToMap(info));
    return out;
}

QVariantList StudioBridge::gpus() const {
    QVariantList out;
    for (const GpuInfo& gpu : client_.gpus()) out.append(ToMap(gpu));
    return out;
}

double StudioBridge::cpuPercent() const { return client_.cpuPercent(); }
int StudioBridge::ramUsedMb() const { return client_.ramUsedMb(); }
int StudioBridge::ramTotalMb() const { return client_.ramTotalMb(); }

void StudioBridge::setStatus(const QString& text) {
    status_ = text;
    emit statusChanged();
}

void StudioBridge::setBusy(bool value) {
    if (busy_ == value) return;
    busy_ = value;
    emit busyChanged();
}

void StudioBridge::connectTo(const QString& address, const QString& key) {
    if (!HyperLinkClient::looksLikeKey(key)) {
        emit error("That does not look like a key",
                   "A T2S key starts with T2S_ (or T2_ / T1_). Check the paste "
                   "kept every character — they contain punctuation that some "
                   "terminals eat.\n\nOn the server:\n"
                   "  gkey create -v v2short --scopes read,write");
        return;
    }
    QSettings settings("HyperNix", "Studio");
    // The fingerprint is not a secret -- it is a public identifier the
    // server hands to any authenticated caller, like a certificate
    // fingerprint -- so it lives in settings, where it can be shown.
    client_.setPinnedFingerprint(
        settings.value("fingerprint/" + address).toString());
    client_.setServer(address, key);
    setStatus("Connecting to " + HyperLinkClient::normaliseAddress(address));
    client_.connectToServer();
}

void StudioBridge::disconnectFromServer() {
    client_.setServer(QString(), QString());
    connected_ = false;
    serverName_.clear();
    serverVersion_.clear();
    setStatus("Not connected");
    emit statusChanged();
}

void StudioBridge::chooseWorkspace(const QString& path) {
    const QString local = FromLocalFileUrl(path);
    const QFileInfo info(local);
    if (!info.isDir()) {
        emit error("Not a folder", local + " is not a directory.");
        return;
    }
    // The absolute, symlink-resolved path. Storing what the user typed
    // would make every later boundary check depend on the spelling.
    const QString canonical = info.canonicalFilePath();
    runner_ = std::make_unique<ToolRunner>(ToolPolicy(canonical.toStdString()));

    QSettings settings("HyperNix", "Studio");
    settings.setValue("workspace", canonical);
    emit workspaceChanged();
}

void StudioBridge::refresh() {
    if (!connected_) return;
    client_.refreshModels();
    client_.refreshResources();
    if (!client_.pinnedFingerprint().isEmpty()) {
        QSettings settings("HyperNix", "Studio");
        settings.setValue("fingerprint/" + client_.address(),
                          client_.pinnedFingerprint());
    }
}

void StudioBridge::switchModel(const QString& modelId, int gpuLayers) {
    if (!connected_) {
        emit error("Not connected", "Connect to a server first.");
        return;
    }
    activeModel_ = modelId;
    emit modelsChanged();
    setStatus("Loading " + modelId);
    client_.loadModel(modelId, gpuLayers);
}

void StudioBridge::appendMessage(const QString& role, const QString& text,
                                 const QString& detail) {
    QVariantMap message;
    message["role"] = role;
    message["text"] = text;
    message["detail"] = detail;
    messages_.append(message);
    emit messagesChanged();
}

QJsonArray StudioBridge::transcript() const {
    QJsonArray out;
    for (const QVariant& value : messages_) {
        const QVariantMap message = value.toMap();
        const QString role = message.value("role").toString();
        // "system" entries are Studio's own notes to the user -- what a
        // tool did, what a search found -- and are not part of the
        // conversation the model sees. Sending them would have the model
        // reading the app's UI text as though a person had written it.
        if (role != "user" && role != "assistant" && role != "tool") continue;
        QJsonObject entry;
        entry["role"] = role;
        entry["content"] = message.value("text").toString();
        out.append(entry);
    }
    return out;
}

QJsonArray StudioBridge::toolSchema() const {
    QJsonArray out;
    if (!runner_) return out;   // no workspace, no file tools
    for (const std::string& name : ToolPolicy::ToolNames()) {
        QJsonObject tool;
        tool["name"] = QString::fromStdString(name);
        out.append(tool);
    }
    return out;
}

void StudioBridge::send(const QString& text) {
    if (!connected_) {
        emit error("Not connected", "Connect to a server first.");
        return;
    }
    if (activeModel_.isEmpty()) {
        emit error("No model chosen",
                   "Pick a model in the Models tab. The server has to load one "
                   "before it can answer.");
        return;
    }
    appendMessage("user", text);
    setBusy(true);
    client_.sendChat(activeModel_, transcript(), toolSchema());
}

void StudioBridge::clearConversation() {
    messages_.clear();
    emit messagesChanged();
}

void StudioBridge::searchModels(const QString& query) {
    if (!connected_) {
        emit error("Not connected", "Connect to a server first.");
        return;
    }
    setBusy(true);
    client_.searchHuggingFace(query);
}

void StudioBridge::download(const QString& repoId, const QString& filename) {
    if (!connected_) return;
    client_.downloadModel(repoId, {filename});
}

// --- the approval flow ---------------------------------------------------

void StudioBridge::onToolCall(const QString& name, const QJsonObject& arguments,
                              const QString& callId) {
    if (!runner_) {
        ToolResult refusal;
        refusal.output =
            "No workspace is open, so there are no files to work with. Open "
            "one from the sidebar.";
        reportToolResult(callId, refusal);
        return;
    }

    Request request;
    request.kind = ToolPolicy::KindFromName(name.toStdString());
    request.path = arguments.value("path").toString().toStdString();
    request.query = arguments.value("query").toString().toStdString();
    request.bytes =
        static_cast<std::size_t>(arguments.value("contents").toString().size());

    const Decision decision = runner_->policy().Decide(request);

    if (decision.verdict == Verdict::Deny) {
        // Never shown as a prompt. There is nothing to approve about
        // reading a private key, and a dialog for it is a dialog people
        // learn to click through -- which would then be there for the
        // request that mattered.
        ToolResult refusal;
        refusal.output = decision.reason;
        refusal.detail = decision.reason;
        refusal.resolved = decision.resolved;
        appendMessage("system", "Refused: " + QString::fromStdString(decision.reason));
        reportToolResult(callId, refusal);
        return;
    }

    if (decision.verdict == Verdict::Allow) {
        runApproved(name, arguments, callId, true);
        return;
    }

    // NeedsApproval. Nothing happens until approveTool().
    pendingName_ = name;
    pendingArguments_ = arguments;
    pendingCallId_ = callId;

    pendingApproval_.clear();
    pendingApproval_["tool"] = name;
    pendingApproval_["reason"] = QString::fromStdString(decision.reason);
    // The resolved path, not the one the model wrote. They can differ,
    // and the one that matters is what would actually be touched.
    pendingApproval_["path"] = QString::fromStdString(decision.resolved);
    pendingApproval_["requestedPath"] = arguments.value("path").toString();
    pendingApproval_["query"] = arguments.value("query").toString();
    pendingApproval_["contents"] = arguments.value("contents").toString();
    pendingApproval_["suspicious"] = decision.suspicious;
    // The current contents, so the dialog can show a diff rather than
    // just "this will be overwritten".
    if (request.kind == ToolKind::WriteFile) {
        pendingApproval_["existing"] =
            readWorkspaceFile(arguments.value("path").toString());
    }
    emit pendingApprovalChanged();
}

void StudioBridge::approveTool() {
    if (pendingCallId_.isEmpty()) return;
    const QString name = pendingName_;
    const QJsonObject arguments = pendingArguments_;
    const QString callId = pendingCallId_;

    pendingApproval_.clear();
    pendingName_.clear();
    pendingArguments_ = QJsonObject();
    pendingCallId_.clear();
    emit pendingApprovalChanged();

    runApproved(name, arguments, callId, true);
}

void StudioBridge::rejectTool() {
    if (pendingCallId_.isEmpty()) return;
    const QString callId = pendingCallId_;
    pendingApproval_.clear();
    pendingName_.clear();
    pendingArguments_ = QJsonObject();
    pendingCallId_.clear();
    emit pendingApprovalChanged();

    ToolResult refusal;
    // Said as the user's decision, not as an error. A model told "you
    // are not permitted" tries a different route; one told "the person
    // said no" asks what they would prefer.
    refusal.output = "The person using Studio declined this.";
    refusal.detail = refusal.output;
    appendMessage("system", "Declined.");
    reportToolResult(callId, refusal);
}

void StudioBridge::runApproved(const QString& name, const QJsonObject& arguments,
                               const QString& callId, bool approved) {
    if (!runner_) return;
    const QString path = arguments.value("path").toString();
    const QString contents = arguments.value("contents").toString();
    ToolResult result;

    switch (ToolPolicy::KindFromName(name.toStdString())) {
        case ToolKind::ReadFile:
            result = runner_->ReadFile(path.toStdString());
            break;
        case ToolKind::ListDirectory:
            result = runner_->ListDirectory(path.toStdString());
            break;
        case ToolKind::WriteFile:
            result = runner_->WriteFile(path.toStdString(),
                                        contents.toStdString(), approved);
            break;
        case ToolKind::CreateFile:
            result = runner_->CreateFile(path.toStdString(),
                                         contents.toStdString(), approved);
            break;
        case ToolKind::DeleteFile:
            result = runner_->DeleteFile(path.toStdString(), approved);
            break;
        case ToolKind::WebSearch:
            // Routed through the server, which is where the search
            // credential lives -- a desktop client holding one would put
            // it somewhere it does not need to be.
            client_.searchHuggingFace(arguments.value("query").toString());
            result.ok = true;
            result.output = "Searching.";
            break;
        case ToolKind::Unknown:
            result.output = "Studio does not have that tool.";
            break;
    }

    if (!result.detail.isEmpty()) appendMessage("system", result.detail);
    reportToolResult(callId, result);
}

void StudioBridge::reportToolResult(const QString& callId,
                                    const ToolResult& result) {
    // The result goes back as a `tool` turn and the conversation
    // continues, so the model can act on what it learned -- including on
    // a refusal, which is information it should have rather than a dead
    // end it retries.
    QVariantMap message;
    message["role"] = "tool";
    message["text"] = result.output;
    message["detail"] = callId;
    messages_.append(message);
    emit messagesChanged();

    if (connected_ && !activeModel_.isEmpty()) {
        setBusy(true);
        client_.sendChat(activeModel_, transcript(), toolSchema());
    }
}

// --- read-only helpers for the file tree ---------------------------------

QVariantList StudioBridge::listWorkspace(const QString& relative) {
    QVariantList out;
    if (!runner_) return out;
    const ToolResult result = runner_->ListDirectory(relative.toStdString());
    if (!result.ok) return out;
    for (const QString& line :
         QString::fromStdString(result.output).split('\n', Qt::SkipEmptyParts)) {
        QVariantMap entry;
        const int marker = line.indexOf("    (");
        entry["name"] = marker < 0 ? line : line.left(marker);
        entry["note"] = marker < 0 ? QString() : line.mid(marker).trimmed();
        entry["directory"] = entry["name"].toString().endsWith('/');
        out.append(entry);
    }
    return out;
}

QString StudioBridge::readWorkspaceFile(const QString& relative) {
    if (!runner_) return {};
    const ToolResult result = runner_->ReadFile(relative.toStdString());
    return result.ok ? QString::fromStdString(result.output) : QString();
}

}  // namespace hnx
