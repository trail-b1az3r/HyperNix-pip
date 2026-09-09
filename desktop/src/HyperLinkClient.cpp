// HyperLinkClient.cpp — see HyperLinkClient.h.

#include "HyperLinkClient.h"

#include <QtCore/QJsonDocument>
#include <QtCore/QJsonValue>
#include <QtCore/QUrl>

namespace hnx {
namespace {

// The T1 error envelope. Every failure carries a stable code, and
// surfacing it is what lets the UI say "your machine has no model
// loaded" instead of "something went wrong".
QString describeError(const QByteArray& body, QNetworkReply* reply) {
    const QJsonObject envelope =
        QJsonDocument::fromJson(body).object().value("error").toObject();
    const QString message = envelope.value("message").toString();
    if (!message.isEmpty()) {
        const QJsonObject details = envelope.value("details").toObject();
        const QString remedy = details.value("remedy").toString();
        return remedy.isEmpty() ? message : message + "\n\n" + remedy;
    }
    return reply->errorString();
}

}  // namespace

HyperLinkClient::HyperLinkClient(QObject* parent) : QObject(parent) {}

QString HyperLinkClient::normaliseAddress(const QString& address) {
    QString text = address.trimmed();
    if (text.isEmpty()) return text;
    if (!text.contains("://")) text.prepend("http://");
    while (text.endsWith('/')) text.chop(1);
    // A bare host with no port almost always means the port the server
    // advertises; adding it beats a connection refused on 80.
    const QUrl url(text);
    if (url.port() == -1 && url.scheme() == "http") text += ":8000";
    return text;
}

bool HyperLinkClient::looksLikeKey(const QString& candidate) {
    const QString text = candidate.trimmed();
    if (!text.startsWith("T2S_") && !text.startsWith("T2_") &&
        !text.startsWith("T1_")) {
        return false;
    }
    return text.size() >= 20;
}

void HyperLinkClient::setServer(const QString& address, const QString& key) {
    address_ = normaliseAddress(address);
    key_ = key.trimmed();
}

QNetworkRequest HyperLinkClient::request(const QString& path) const {
    QNetworkRequest out(QUrl(address_ + path));
    out.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    out.setRawHeader("Accept", "application/json");
    out.setRawHeader("User-Agent", "HyperNix-Studio/0.1");
    if (!key_.isEmpty()) {
        out.setRawHeader("Authorization", ("Bearer " + key_).toUtf8());
    }
    // Redirects are not followed. A redirect from an authenticated
    // request would resend the Authorization header to whatever host the
    // response named, which is a credential handed to somewhere the user
    // never chose.
    out.setAttribute(QNetworkRequest::RedirectPolicyAttribute,
                     QNetworkRequest::ManualRedirectPolicy);
    return out;
}

QNetworkReply* HyperLinkClient::get(const QString& path) {
    return network_.get(request(path));
}

QNetworkReply* HyperLinkClient::post(const QString& path,
                                     const QJsonObject& body) {
    return network_.post(request(path),
                         QJsonDocument(body).toJson(QJsonDocument::Compact));
}

void HyperLinkClient::reportError(const QString& what, QNetworkReply* reply) {
    emit failed(what, describeError(reply->readAll(), reply));
    reply->deleteLater();
}

void HyperLinkClient::connectToServer() {
    QNetworkReply* reply = get("/hyperlink/endpoints");
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        if (reply->error() != QNetworkReply::NoError) {
            reportError("Could not reach the server", reply);
            return;
        }
        const QJsonObject body =
            QJsonDocument::fromJson(reply->readAll()).object();
        reply->deleteLater();

        const QString reported = body.value("server_fingerprint").toString();
        // An empty fingerprint is a server older than 0.72.4, not a
        // mismatch. Treating it as one would refuse to talk to every
        // server that has not been updated yet, which is how a check
        // like this gets removed in the next release.
        if (!reported.isEmpty() && !pinnedFingerprint_.isEmpty() &&
            reported != pinnedFingerprint_) {
            emit identityMismatch(pinnedFingerprint_, reported);
            return;
        }
        if (pinnedFingerprint_.isEmpty()) pinnedFingerprint_ = reported;

        emit connected(body.value("server_name").toString(),
                       body.value("t1_version").toString(),
                       body.value("keyless_available_here").toBool());
    });
}

void HyperLinkClient::refreshModels() {
    QNetworkReply* reply = get("/models");
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        if (reply->error() != QNetworkReply::NoError) {
            reportError("Could not list models", reply);
            return;
        }
        const QJsonArray entries =
            QJsonDocument::fromJson(reply->readAll()).object()
                .value("models").toArray();
        reply->deleteLater();

        models_.clear();
        for (const QJsonValue& value : entries) {
            const QJsonObject entry = value.toObject();
            ModelInfo info;
            info.id = entry.value("model_id").toString();
            info.displayName = entry.value("display_name").toString();
            info.architecture = entry.value("architecture").toString();
            info.quantisation = entry.value("quantisation").toString();
            info.parametersB = entry.value("total_parameters").toDouble();
            info.contextLimit = entry.value("context_limit").toInt();
            info.backend = entry.value("backend").toString();
            models_.append(info);
        }
        emit modelsChanged();
    });
}

void HyperLinkClient::refreshResources() {
    QNetworkReply* reply = get("/training/resources");
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        if (reply->error() != QNetworkReply::NoError) {
            // 403 is the normal answer for a T2S key on a server that is
            // not in trusted-network mode. Reported as "not available"
            // rather than as a failure: the panel is a nicety, and an
            // error banner for a permission the app never claimed to
            // have would be noise on every launch.
            const int status =
                reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
            if (status != 403 && status != 401) {
                reportError("Could not read machine resources", reply);
            } else {
                reply->deleteLater();
            }
            gpus_.clear();
            cpuPercent_ = -1.0;
            emit resourcesChanged();
            return;
        }
        const QJsonObject body =
            QJsonDocument::fromJson(reply->readAll()).object();
        reply->deleteLater();

        gpus_.clear();
        for (const QJsonValue& value : body.value("gpus").toArray()) {
            const QJsonObject card = value.toObject();
            GpuInfo gpu;
            gpu.index = card.value("index").toInt();
            gpu.vendor = card.value("vendor").toString();
            gpu.name = card.value("name").toString();
            gpu.framework = card.value("framework").toString();
            // -1 rather than 0 for a measurement the vendor tool did not
            // give: nvidia-smi answers "[N/A]" for several of these on
            // consumer cards, and 0 would render as a full bar of
            // nothing.
            gpu.memoryTotalMb = card.value("memory_total_mb").isNull()
                                    ? -1 : card.value("memory_total_mb").toInt();
            gpu.memoryUsedMb = card.value("memory_used_mb").isNull()
                                   ? -1 : card.value("memory_used_mb").toInt();
            gpu.utilisation = card.value("utilization_pct").isNull()
                                  ? -1.0 : card.value("utilization_pct").toDouble();
            gpus_.append(gpu);
        }
        cpuPercent_ = body.value("cpu_percent").isNull()
                          ? -1.0 : body.value("cpu_percent").toDouble();
        ramUsedMb_ = body.value("ram_used_mb").isNull()
                         ? -1 : body.value("ram_used_mb").toInt();
        ramTotalMb_ = body.value("ram_total_mb").isNull()
                          ? -1 : body.value("ram_total_mb").toInt();
        emit resourcesChanged();
    });
}

void HyperLinkClient::sendChat(const QString& modelId,
                               const QJsonArray& messages,
                               const QJsonArray& tools) {
    QJsonObject body;
    body["model"] = modelId;
    body["messages"] = messages;
    if (!tools.isEmpty()) body["tools"] = tools;

    QNetworkReply* reply = post("/inference/chat", body);
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        if (reply->error() != QNetworkReply::NoError) {
            reportError("The model did not answer", reply);
            return;
        }
        const QJsonObject answer =
            QJsonDocument::fromJson(reply->readAll()).object();
        reply->deleteLater();

        // A tool call is *requested*, never performed here. The
        // signal goes to the UI, which asks the user, and only then does
        // anything reach ToolRunner. Nothing in this class touches the
        // filesystem, which is what makes "a model cannot act on its own
        // word" a property of the structure rather than of a check.
        const QJsonArray calls = answer.value("tool_calls").toArray();
        for (const QJsonValue& value : calls) {
            const QJsonObject call = value.toObject();
            emit toolCallRequested(call.value("name").toString(),
                                   call.value("arguments").toObject(),
                                   call.value("id").toString());
        }
        emit chatFinished(answer);
    });
}

void HyperLinkClient::loadModel(const QString& modelId, int gpuLayers) {
    QJsonObject body;
    body["model_id"] = modelId;
    // -1 means "let the server decide", which is almost always right: it
    // is the machine that knows how much VRAM is free, and a number
    // chosen on the client is a guess about somebody else's GPU.
    if (gpuLayers >= 0) body["gpu_layers"] = gpuLayers;

    QNetworkReply* reply = post("/inference/load", body);
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        if (reply->error() != QNetworkReply::NoError) {
            reportError("Could not switch model", reply);
            return;
        }
        reply->deleteLater();
        refreshModels();
    });
}

void HyperLinkClient::searchHuggingFace(const QString& query) {
    QJsonObject body;
    // The server resolves this, not Studio. It is the machine that will
    // download the weights and the one that holds the HF token, and
    // sending a token from a desktop client would put it somewhere it
    // does not need to be.
    body["page_url"] = query;
    body["prefer"] = "strict";

    QNetworkReply* reply = post("/hyperlink/models/resolve", body);
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        if (reply->error() != QNetworkReply::NoError) {
            reportError("Could not resolve that model", reply);
            return;
        }
        const QJsonObject body =
            QJsonDocument::fromJson(reply->readAll()).object();
        reply->deleteLater();
        emit searchResults(body.value("files").toArray());
    });
}

void HyperLinkClient::downloadModel(const QString& repoId,
                                    const QStringList& filenames) {
    QJsonObject body;
    body["repo_id"] = repoId;
    QJsonArray names;
    for (const QString& name : filenames) names.append(name);
    body["filenames"] = names;

    QNetworkReply* reply = post("/hyperlink/models/download", body);
    connect(reply, &QNetworkReply::finished, this, [this, reply]() {
        if (reply->error() != QNetworkReply::NoError) {
            reportError("Could not start the download", reply);
            return;
        }
        reply->deleteLater();
        // The server runs it as a job; the models list is where it will
        // appear when it finishes.
        refreshModels();
    });
}

}  // namespace hnx
