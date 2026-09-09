/*
 * LocalSession — LocalEngine on a worker thread, with Qt signals.
 *
 * LocalEngine::Load and ::Generate block: loading a 7B model takes
 * seconds and generating takes as long as it takes. Neither may happen
 * on the GUI thread, or the window stops repainting and the desktop
 * offers to kill it.
 *
 * So the engine lives on a QThread and this is the only thing that
 * touches it. Requests go in as queued slot calls, tokens come back as
 * queued signals, and the one method that crosses threads on purpose is
 * cancel() -- LocalEngine::Cancel is the atomic that exists for it.
 */
#ifndef HNX_LOCAL_SESSION_H
#define HNX_LOCAL_SESSION_H

#include <QtCore/QObject>
#include <QtCore/QString>
#include <QtCore/QThread>
#include <QtCore/QVariantList>

#include <memory>

#include "LocalEngine.h"
#include "ModelCatalogue.h"

namespace hnx {

/// Lives on the worker thread. Not used directly by the UI.
class LocalWorker : public QObject {
    Q_OBJECT

public:
    LocalWorker();
    ~LocalWorker() override;

    /// Safe from the GUI thread: the only method that is.
    void cancel();

public slots:
    void scan(const QStringList& roots);
    void load(const QString& path, int gpuLayers, int contextLength);
    void unload();
    void generate(const QString& prompt, int maxTokens, double temperature);

signals:
    void scanned(const QVariantList& models);
    void loaded(const QVariantMap& info);
    void loadFailed(const QString& reason);
    void token(const QString& piece);
    void finished();
    void failed(const QString& reason);

private:
    std::unique_ptr<LocalEngine> engine_;
};

/// The GUI-thread handle. Owns the thread and the worker.
class LocalSession : public QObject {
    Q_OBJECT

public:
    explicit LocalSession(QObject* parent = nullptr);
    ~LocalSession() override;

    static bool available();
    static QString unavailableReason();

    bool loaded() const { return loaded_; }
    QVariantMap info() const { return info_; }
    QVariantList models() const { return models_; }
    QString modelPath() const { return modelPath_; }

    void scan(const QStringList& roots);
    void load(const QString& path, int gpuLayers, int contextLength);
    void unload();
    void generate(const QString& prompt, int maxTokens, double temperature);
    void cancel();

    /// Where to look when the user has not said. Qt-side so it can add
    /// the platform's own data directory to the catalogue's list.
    static QStringList defaultRoots();

signals:
    void scanned();
    void loadedChanged();
    void loadFailed(const QString& reason);
    void token(const QString& piece);
    void finished();
    void failed(const QString& reason);

private:
    QThread thread_;
    LocalWorker* worker_;              // owned by the thread
    bool loaded_ = false;
    QVariantMap info_;
    QVariantList models_;
    QString modelPath_;
};

}  // namespace hnx

#endif  // HNX_LOCAL_SESSION_H
