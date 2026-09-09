import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// CodeView.qml — the workspace, as files.
//
// A tree on the left and the file on the right. Read-only: editing here
// would be a second way to change a file, and the whole approval flow
// exists because there should be exactly one. What this is for is
// *seeing* what the model is working on, and checking what it did.
//
// Files the model cannot touch are shown and labelled rather than
// hidden. Someone scanning a tree should be able to see that `.env` is
// there and that Studio will not read it, which is a different and more
// useful fact than its absence.
Item {
    id: root
    property string openPath: ""
    property string openContents: ""
    property string currentDir: "."

    function openFile(name) {
        var path = currentDir === "." ? name : currentDir + "/" + name
        openPath = path
        openContents = studio.readWorkspaceFile(path)
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        Rectangle {
            Layout.fillWidth: true
            implicitHeight: 52
            color: Theme.surface

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: Theme.gapLarge
                anchors.rightMargin: Theme.gapLarge
                spacing: Theme.gap

                Label {
                    text: root.openPath.length > 0 ? root.openPath : "Code"
                    color: Theme.text
                    font.family: root.openPath.length > 0
                                 ? Theme.monoFamily : Theme.fontFamily
                    font.pixelSize: root.openPath.length > 0
                                    ? Theme.fontBody : Theme.fontTitle
                    font.weight: Font.DemiBold
                    Layout.fillWidth: true
                    elide: Text.ElideMiddle
                }

                Label {
                    text: "read-only — changes go through the model, with your approval"
                    color: Theme.textFaint
                    font.family: Theme.fontFamily
                    font.pixelSize: 11
                }
            }

            Rectangle {
                anchors.bottom: parent.bottom
                width: parent.width; height: 1
                color: Theme.border
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0

            // --- the tree ---
            Rectangle {
                Layout.preferredWidth: 280
                Layout.fillHeight: true
                color: Theme.surface

                Rectangle {
                    anchors.right: parent.right
                    width: 1; height: parent.height
                    color: Theme.border
                }

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: Theme.gap
                    spacing: 4

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 6
                        StudioButton {
                            text: "↑"
                            implicitWidth: 30
                            enabled: root.currentDir !== "."
                            onClicked: {
                                var parts = root.currentDir.split("/")
                                parts.pop()
                                root.currentDir = parts.length ? parts.join("/") : "."
                                tree.model = studio.listWorkspace(root.currentDir)
                            }
                        }
                        Label {
                            text: root.currentDir
                            color: Theme.textFaint
                            font.family: Theme.monoFamily
                            font.pixelSize: 11
                            Layout.fillWidth: true
                            elide: Text.ElideMiddle
                        }
                    }

                    ListView {
                        id: tree
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        model: studio.workspace.length > 0
                               ? studio.listWorkspace(root.currentDir) : []
                        ScrollBar.vertical: ScrollBar {}

                        delegate: Rectangle {
                            required property var modelData
                            width: ListView.view.width
                            implicitHeight: 28
                            radius: Theme.radiusSmall
                            color: entryHover.hovered ? Theme.hover : "transparent"
                            // Not clickable when Studio would refuse it
                            // anyway: offering to open a file that
                            // cannot be opened is a dead end dressed up
                            // as a feature.
                            readonly property bool reachable:
                                modelData.note === undefined ||
                                modelData.note.length === 0

                            HoverHandler { id: entryHover }
                            TapHandler {
                                enabled: parent.reachable
                                onTapped: {
                                    var name = modelData.name
                                    if (modelData.directory) {
                                        var clean = name.substring(0, name.length - 1)
                                        root.currentDir = root.currentDir === "."
                                            ? clean : root.currentDir + "/" + clean
                                        tree.model = studio.listWorkspace(root.currentDir)
                                    } else {
                                        root.openFile(name)
                                    }
                                }
                            }

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.leftMargin: 8
                                anchors.rightMargin: 8
                                spacing: 0

                                Label {
                                    text: modelData.name
                                    color: parent.parent.reachable
                                           ? (modelData.directory
                                              ? Theme.textDim : Theme.text)
                                           : Theme.textFaint
                                    font.family: Theme.monoFamily
                                    font.pixelSize: Theme.fontSmall
                                    Layout.fillWidth: true
                                    elide: Text.ElideMiddle
                                }
                                Label {
                                    visible: !parent.parent.reachable
                                    text: modelData.note
                                    color: Theme.warn
                                    font.family: Theme.fontFamily
                                    font.pixelSize: 9
                                    Layout.fillWidth: true
                                    elide: Text.ElideRight
                                }
                            }
                        }

                        Label {
                            anchors.centerIn: parent
                            visible: studio.workspace.length === 0
                            text: "No workspace.\nChoose a folder in the sidebar."
                            color: Theme.textFaint
                            font.family: Theme.fontFamily
                            font.pixelSize: Theme.fontSmall
                            horizontalAlignment: Text.AlignHCenter
                        }
                    }
                }
            }

            // --- the file ---
            ScrollView {
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true

                TextArea {
                    readOnly: true
                    text: root.openContents
                    color: Theme.text
                    font.family: Theme.monoFamily
                    font.pixelSize: Theme.fontSmall
                    selectByMouse: true
                    wrapMode: TextArea.NoWrap
                    // Never rich text: this is the contents of somebody's
                    // file, and a file that happens to contain markup
                    // would render as markup.
                    textFormat: TextEdit.PlainText
                    background: Rectangle { color: Theme.base }

                    Label {
                        anchors.centerIn: parent
                        visible: root.openPath.length === 0
                        text: "Pick a file to read it."
                        color: Theme.textFaint
                        font.family: Theme.fontFamily
                        font.pixelSize: Theme.fontBody
                    }
                }
            }
        }
    }
}
