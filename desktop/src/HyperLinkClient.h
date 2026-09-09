// HyperLinkClient.h — Studio's side of the T1 API.
//
// The same surface HyperLink on the phone uses, for the same reason: the
// server is the machine with the GPU and the models, and Studio is a
// client of it whether it is running on that machine or across a
// tailnet. There is no second protocol and no local-only path — a
// desktop that talked to a local server differently from a remote one
// would be two apps sharing a window.
//
// Credentials
// -----------
// Studio authenticates with a **T2S key**, which is limited by
// construction to reading and non-admin writing. That is the right
// credential for this app: Studio switches models, chats, reads and
// writes files in a workspace, and searches. None of that is
// administration, and a tool that does not need admin should not hold an
// admin key where a compromised workspace could reach it.
//
// So there is no code here that sends an admin credential, and no
// endpoint called that requires one. Training controls, key rotation and
// pairing are deliberately absent: they belong to `waiter` and
// `hypernix-t1` on the machine itself.
//
// Server identity
// ---------------
// `/hyperlink/endpoints` reports a fingerprint. Studio pins it the first
// time it connects and compares it afterwards, for the same reason
// HyperLink does: the address Studio reaches can end up pointing at a
// different machine, and the server *name* authenticates nothing —
// anything on a LAN or a tailnet can claim one.

#ifndef HYPERNIX_STUDIO_HYPERLINK_CLIENT_H
#define HYPERNIX_STUDIO_HYPERLINK_CLIENT_H

#include <QtCore/QJsonArray>
#include <QtCore/QJsonObject>
#include <QtCore/QObject>
#include <QtCore/QString>
#include <QtCore/QStringList>
#include <QtNetwork/QNetworkAccessManager>
#include <QtNetwork/QNetworkReply>

namespace hnx {

// One model as the server's registry describes it.
struct ModelInfo {
    QString id;
    QString displayName;
    QString architecture;
    QString quantisation;
    double parametersB = 0.0;
    int contextLimit = 0;
    bool loaded = false;
    // Where it would run. From the server's GPU abstraction, so this is
    // what the server actually detected rather than what the app hopes.
    QString backend;
};

struct GpuInfo {
    int index = 0;
    QString vendor;
    QString name;
    QString framework;      // cuda | rocm | mps | xpu | cpu
    int memoryTotalMb = -1;
    int memoryUsedMb = -1;
    double utilisation = -1.0;
};

class HyperLinkClient : public QObject {
    Q_OBJECT

public:
    explicit HyperLinkClient(QObject* parent = nullptr);

    // `address` may be a bare host, a host:port, or a full URL; it is
    // normalised the same way HyperLink's client does it.
    void setServer(const QString& address, const QString& key);

    QString address() const { return address_; }
    bool hasCredential() const { return !key_.isEmpty(); }

    // The pinned fingerprint, empty until the first successful connect.
    QString pinnedFingerprint() const { return pinnedFingerprint_; }
    void setPinnedFingerprint(const QString& value) { pinnedFingerprint_ = value; }

public slots:
    // GET /hyperlink/endpoints — connectivity, identity, and whether
    // this network can go keyless.
    void connectToServer();
    // GET /models, and /bridge/lmstudio/models for what is loaded.
    void refreshModels();
    // GET /training/resources — the GPU/CPU/RAM panel. Read-only, and
    // the one training endpoint a T2S key can reach on a trusted
    // network; a 403 here is normal and reported as "not available"
    // rather than as an error.
    void refreshResources();
    // POST /inference/chat. The governed surface, not the raw bridge:
    // /inference applies the registry, the routing cascade and the
    // quota, and /bridge/lmstudio applies none of them.
    void sendChat(const QString& modelId, const QJsonArray& messages,
                  const QJsonArray& tools);
    // POST /models/{id}/load via the server's own switching endpoint.
    void loadModel(const QString& modelId, int gpuLayers);
    // POST /hyperlink/models/resolve — a Hugging Face page or file link
    // turned into something downloadable.
    void searchHuggingFace(const QString& query);
    void downloadModel(const QString& repoId, const QStringList& filenames);

signals:
    void connected(const QString& serverName, const QString& t1Version,
                   bool keylessAvailable);
    // Emitted instead of connected() when the server at this address
    // reports a different fingerprint than the one pinned. Not a
    // warning to be dismissed: Studio refuses to send the key.
    void identityMismatch(const QString& expected, const QString& actual);
    void modelsChanged();
    void resourcesChanged();
    void chatChunk(const QString& text);
    void chatFinished(const QJsonObject& reply);
    void toolCallRequested(const QString& name, const QJsonObject& arguments,
                           const QString& callId);
    void searchResults(const QJsonArray& results);
    void failed(const QString& what, const QString& detail);

public:
    QList<ModelInfo> models() const { return models_; }
    QList<GpuInfo> gpus() const { return gpus_; }
    double cpuPercent() const { return cpuPercent_; }
    int ramUsedMb() const { return ramUsedMb_; }
    int ramTotalMb() const { return ramTotalMb_; }

    static QString normaliseAddress(const QString& address);
    // A shape check only, to catch a paste that lost characters before
    // it costs a round trip. The server decides whether a key is real.
    static bool looksLikeKey(const QString& candidate);

private:
    QNetworkRequest request(const QString& path) const;
    QNetworkReply* get(const QString& path);
    QNetworkReply* post(const QString& path, const QJsonObject& body);
    void reportError(const QString& what, QNetworkReply* reply);

    QNetworkAccessManager network_;
    QString address_;
    QString key_;
    QString pinnedFingerprint_;

    QList<ModelInfo> models_;
    QList<GpuInfo> gpus_;
    double cpuPercent_ = -1.0;
    int ramUsedMb_ = -1;
    int ramTotalMb_ = -1;
};

}  // namespace hnx

#endif  // HYPERNIX_STUDIO_HYPERLINK_CLIENT_H
