import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

// Main.qml — the window, and the four things Studio does.
//
// A sidebar and a stack, which is the shape LM Studio uses and the right
// one here: the four views are peers rather than a hierarchy, and a
// person switching between chat and code should not lose their place in
// either.
ApplicationWindow {
    id: window
    width: 1280
    height: 840
    minimumWidth: 900
    minimumHeight: 600
    visible: true
    title: "HyperNix Studio"
    color: Theme.base

    // A mismatch takes the whole window. "Something else is answering at
    // this address" is not information to scroll past, and a toast is
    // exactly a way to scroll past something.
    property bool identityBlocked: false
    property string identityExpected: ""
    property string identityActual: ""

    Connections {
        target: studio
        function onIdentityMismatch(expected, actual) {
            window.identityExpected = expected
            window.identityActual = actual
            window.identityBlocked = true
        }
        function onError(what, detail) {
            errorBanner.show(what, detail)
        }
    }

    RowLayout {
        anchors.fill: parent
        spacing: 0
        enabled: !window.identityBlocked

        Sidebar {
            id: sidebar
            Layout.preferredWidth: Theme.sidebar
            Layout.fillHeight: true
            onOpenWorkspace: workspaceDialog.open()
        }

        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0

            ErrorBanner { id: errorBanner; Layout.fillWidth: true }

            StackLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                currentIndex: sidebar.currentIndex

                ChatView {}
                CodeView {}
                ModelsView {}
                MachineView {}
            }
        }
    }

    FolderDialog {
        id: workspaceDialog
        title: "Choose a folder for the model to work in"
        onAccepted: studio.chooseWorkspace(selectedFolder)
    }

    // Nothing runs until this is answered.
    ToolApproval {
        anchors.fill: parent
        visible: studio.pendingApproval && studio.pendingApproval.tool !== undefined
    }

    // The blocking state. Deliberately offers only two ways out, and
    // neither of them is "continue anyway": there is no safe way to
    // continue talking to a machine that is not the one you paired with.
    Rectangle {
        anchors.fill: parent
        color: Qt.rgba(0, 0, 0, 0.82)
        visible: window.identityBlocked

        ColumnLayout {
            anchors.centerIn: parent
            width: Math.min(560, parent.width - 80)
            spacing: Theme.gapLarge

            Label {
                text: "This is not the machine you connected to before"
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontTitle
                font.weight: Font.DemiBold
                color: Theme.text
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
            }

            Label {
                text: "The server at this address reports a different identity " +
                      "than the one Studio recorded. That happens when a router " +
                      "hands your machine's old address to something else — or " +
                      "when something is pretending to be it.\n\n" +
                      "Studio has not sent your key."
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontBody
                color: Theme.textDim
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
            }

            // Both fingerprints, so it can be compared against what
            // `hypernix-t1 status` prints on the machine itself.
            GridLayout {
                columns: 2
                columnSpacing: Theme.gap
                Label {
                    text: "expected"
                    color: Theme.textFaint
                    font.family: Theme.fontFamily
                    font.pixelSize: Theme.fontSmall
                }
                Label {
                    text: window.identityExpected
                    color: Theme.text
                    font.family: Theme.monoFamily
                    font.pixelSize: Theme.fontSmall
                }
                Label {
                    text: "answered"
                    color: Theme.textFaint
                    font.family: Theme.fontFamily
                    font.pixelSize: Theme.fontSmall
                }
                Label {
                    text: window.identityActual
                    color: Theme.accentHi
                    font.family: Theme.monoFamily
                    font.pixelSize: Theme.fontSmall
                }
            }

            RowLayout {
                spacing: Theme.gap
                StudioButton {
                    text: "Disconnect"
                    primary: true
                    onClicked: {
                        studio.disconnectFromServer()
                        window.identityBlocked = false
                    }
                }
                StudioButton {
                    text: "How do I check?"
                    onClicked: Qt.openUrlExternally(
                        "https://github.com/minerofthesoal/hypernix-pip/blob/main/wiki/T1-API.md#finding-a-server-and-knowing-which-one-it-is")
                }
            }
        }
    }
}
