// main.cpp — HyperNix Studio.
//
// A desktop client for a HyperNix server: switch models, chat, work on
// code in a folder you chose, and let the model edit files in it with
// your agreement. Qt 6 with QML, because the UI is the point and QML is
// what makes a modern one affordable to write.
//
// What Studio deliberately is not
// -------------------------------
// It does not run a model itself. The server has the GPU, the registry,
// the quota and the routing cascade, and re-implementing any of that in
// a desktop app would mean two things to keep in agreement. Studio is a
// client of the T1 API, over HyperLink's endpoints, exactly as the phone
// app is.
//
// It does not administer anything. It authenticates with a T2S key,
// which is limited to reading and non-admin writing by construction, and
// no code path here needs more. Key management, training controls and
// pairing belong to `waiter` and `hypernix-t1` on the machine itself.
//
// It cannot run a command. There is no shell tool, no exec, no "run the
// tests" button that shells out. That is the one guarantee in this app
// that does not depend on a check being correct, and the only way to
// keep it is not to write the feature.

#include <QtGui/QGuiApplication>
#include <QtQml/QQmlApplicationEngine>
#include <QtQml/QQmlContext>
#include <QtQuickControls2/QQuickStyle>

#include "StudioBridge.h"

int main(int argc, char* argv[]) {
    QGuiApplication app(argc, argv);
    app.setApplicationName("HyperNix Studio");
    app.setOrganizationName("HyperNix");
    app.setApplicationVersion("0.1.0");

    // Basic, not Fusion: Fusion imitates a desktop toolkit and looks
    // like a 2012 Qt app. Basic is unstyled, which is what makes a
    // custom look possible -- everything visual here is in qml/Theme.qml
    // and the components that read it.
    QQuickStyle::setStyle("Basic");

    hnx::StudioBridge bridge;

    QQmlApplicationEngine engine;
    engine.rootContext()->setContextProperty("studio", &bridge);
    engine.load(QUrl("qrc:/qml/Main.qml"));
    if (engine.rootObjects().isEmpty()) return 1;

    return app.exec();
}
