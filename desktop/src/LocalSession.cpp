#include "LocalSession.h"

#include <QtCore/QDir>
#include <QtCore/QStandardPaths>
#include <QtCore/QVariantMap>

namespace hnx {
namespace {

QVariantMap ToMap(const LocalModel& model) {
    QVariantMap map;
    map["id"] = QString::fromStdString(model.path);
    map["path"] = QString::fromStdString(model.path);
    map["name"] = QString::fromStdString(model.name);
    map["fileName"] = QString::fromStdString(model.file_name);
    map["architecture"] = QString::fromStdString(model.architecture);
    map["quantisation"] = QString::fromStdString(model.quantisation);
    map["parameters"] = QString::fromStdString(model.parameter_label());
    map["size"] = QString::fromStdString(model.size_label());
    map["contextLength"] = static_cast<qulonglong>(model.context_length);
    map["ok"] = model.ok;
    map["error"] = QString::fromStdString(model.error);
    map["local"] = true;
    return map;
}

QVariantMap ToMap(const LoadedModel& info) {
    QVariantMap map;
    map["path"] = QString::fromStdString(info.path);
    map["name"] = QString::fromStdString(info.name);
    map["architecture"] = QString::fromStdString(info.architecture);
    map["parameters"] = static_cast<qulonglong>(info.parameters);
    map["contextLength"] = info.context_length;
    map["gpuLayers"] = info.gpu_layers;
    map["layersTotal"] = info.layers_total;
    return map;
}

}  // namespace

// --- the worker ---------------------------------------------------------

LocalWorker::LocalWorker() : engine_(new LocalEngine) {}
LocalWorker::~LocalWorker() = default;

void LocalWorker::cancel() { engine_->Cancel(); }

void LocalWorker::scan(const QStringList& roots) {
    std::vector<std::string> paths;
    for (const QString& root : roots) paths.push_back(root.toStdString());
    QVariantList out;
    for (const LocalModel& model : ScanForModels(paths)) out.append(ToMap(model));
    emit scanned(out);
}

void LocalWorker::load(const QString& path, int gpuLayers, int contextLength) {
    LoadOptions options;
    options.gpu_layers = gpuLayers;
    options.context_length = contextLength;
    std::string error;
    if (!engine_->Load(path.toStdString(), options, &error)) {
        emit loadFailed(QString::fromStdString(error));
        return;
    }
    emit loaded(ToMap(engine_->info()));
}

void LocalWorker::unload() { engine_->Unload(); }

void LocalWorker::generate(const QString& prompt, int maxTokens,
                           double temperature) {
    GenerateOptions options;
    options.max_tokens = maxTokens;
    options.temperature = static_cast<float>(temperature);
    std::string error;
    const bool ok = engine_->Generate(
        prompt.toStdString(), options,
        [this](const std::string& piece) {
            // One signal per token. Queued across the thread boundary,
            // so the UI paints them as they arrive rather than in one
            // lump at the end -- which is the whole point of streaming.
            emit token(QString::fromStdString(piece));
            return true;
        },
        &error);
    if (!ok) {
        emit failed(QString::fromStdString(error));
        return;
    }
    emit finished();
}

// --- the handle ---------------------------------------------------------

LocalSession::LocalSession(QObject* parent)
    : QObject(parent), worker_(new LocalWorker) {
    worker_->moveToThread(&thread_);
    connect(&thread_, &QThread::finished, worker_, &QObject::deleteLater);

    connect(worker_, &LocalWorker::scanned, this,
            [this](const QVariantList& models) {
                models_ = models;
                emit scanned();
            });
    connect(worker_, &LocalWorker::loaded, this,
            [this](const QVariantMap& info) {
                info_ = info;
                modelPath_ = info.value("path").toString();
                loaded_ = true;
                emit loadedChanged();
            });
    connect(worker_, &LocalWorker::loadFailed, this,
            [this](const QString& reason) {
                loaded_ = false;
                info_.clear();
                modelPath_.clear();
                emit loadedChanged();
                emit loadFailed(reason);
            });
    connect(worker_, &LocalWorker::token, this, &LocalSession::token);
    connect(worker_, &LocalWorker::finished, this, &LocalSession::finished);
    connect(worker_, &LocalWorker::failed, this, &LocalSession::failed);

    thread_.start();
}

LocalSession::~LocalSession() {
    // Cancel first: quitting a thread that is inside Generate would
    // wait for the whole generation, and closing the window during a
    // long answer should not take a minute.
    if (worker_ != nullptr) worker_->cancel();
    thread_.quit();
    thread_.wait();
}

bool LocalSession::available() { return LocalEngine::Available(); }

QString LocalSession::unavailableReason() {
    return QString::fromStdString(LocalEngine::UnavailableReason());
}

void LocalSession::scan(const QStringList& roots) {
    QMetaObject::invokeMethod(worker_, "scan", Qt::QueuedConnection,
                              Q_ARG(QStringList, roots));
}

void LocalSession::load(const QString& path, int gpuLayers, int contextLength) {
    QMetaObject::invokeMethod(worker_, "load", Qt::QueuedConnection,
                              Q_ARG(QString, path), Q_ARG(int, gpuLayers),
                              Q_ARG(int, contextLength));
}

void LocalSession::unload() {
    loaded_ = false;
    info_.clear();
    modelPath_.clear();
    emit loadedChanged();
    QMetaObject::invokeMethod(worker_, "unload", Qt::QueuedConnection);
}

void LocalSession::generate(const QString& prompt, int maxTokens,
                            double temperature) {
    QMetaObject::invokeMethod(worker_, "generate", Qt::QueuedConnection,
                              Q_ARG(QString, prompt), Q_ARG(int, maxTokens),
                              Q_ARG(double, temperature));
}

void LocalSession::cancel() {
    // Direct, not queued: the worker thread is *inside* Generate, so a
    // queued call would sit in its event queue until the generation it
    // is meant to stop had finished. LocalEngine::Cancel is an atomic
    // for exactly this.
    if (worker_ != nullptr) worker_->cancel();
}

QStringList LocalSession::defaultRoots() {
    QStringList roots;
    for (const std::string& path : DefaultSearchPaths()) {
        roots << QString::fromStdString(path);
    }
    // Where Qt says this platform keeps application data, which is not
    // $HOME/.local/share everywhere.
    const QString data =
        QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    if (!data.isEmpty()) roots << QDir(data).filePath("models");
    roots.removeDuplicates();
    return roots;
}

}  // namespace hnx
