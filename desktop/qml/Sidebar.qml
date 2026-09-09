import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// Sidebar.qml — where you are, what you are connected to, and the
// workspace the model may touch.
//
// The workspace is here rather than buried in a settings screen on
// purpose: it is the boundary on everything the model can do to the
// disk, and a boundary you cannot see is one you forget you set.
Rectangle {
    id: root
    property int currentIndex: 0
    signal openWorkspace()

    color: Theme.surface

    Rectangle {
        anchors.right: parent.right
        width: 1
        height: parent.height
        color: Theme.border
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: Theme.gap
        spacing: Theme.gap

        // --- the mark ---
        RowLayout {
            spacing: 10
            Layout.topMargin: 4

            // The three staggered bars, drawn rather than shipped as an
            // image: it is nine coordinates, and a Canvas keeps it
            // crisp at any scale factor.
            Canvas {
                width: 26; height: 26
                onPaint: {
                    var ctx = getContext("2d")
                    ctx.reset()
                    var s = width / 64
                    var bars = [
                        [[6,52],[16,40],[42,40],[32,52], Theme.text],
                        [[12,34],[22,22],[48,22],[38,34], Theme.textDim],
                        [[18,16],[28,4],[54,4],[44,16], Theme.accent]
                    ]
                    for (var i = 0; i < bars.length; i++) {
                        var bar = bars[i]
                        ctx.fillStyle = bar[4]
                        ctx.beginPath()
                        ctx.moveTo(bar[0][0]*s, bar[0][1]*s)
                        for (var j = 1; j < 4; j++) ctx.lineTo(bar[j][0]*s, bar[j][1]*s)
                        ctx.closePath()
                        ctx.fill()
                    }
                }
            }

            Label {
                text: "Studio"
                color: Theme.text
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontTitle
                font.weight: Font.DemiBold
            }
        }

        // --- the four views ---
        Repeater {
            model: [
                { label: "Chat",    hint: "talk to a model" },
                { label: "Code",    hint: "work on a folder" },
                { label: "Models",  hint: "switch and download" },
                { label: "Machine", hint: "GPU, CPU, memory" }
            ]

            delegate: Rectangle {
                required property int index
                required property var modelData

                Layout.fillWidth: true
                implicitHeight: Theme.rowHeight
                radius: Theme.radiusSmall
                color: root.currentIndex === index
                       ? Theme.surfaceAlt
                       : (hover.hovered ? Theme.hover : "transparent")
                Behavior on color { ColorAnimation { duration: Theme.anim } }

                // A left bar rather than a filled row for the selection:
                // it reads at a glance without the accent competing with
                // everything else on the screen.
                Rectangle {
                    anchors.left: parent.left
                    anchors.verticalCenter: parent.verticalCenter
                    width: 2
                    height: parent.height * 0.55
                    radius: 1
                    color: Theme.accent
                    visible: root.currentIndex === index
                }

                Label {
                    anchors.verticalCenter: parent.verticalCenter
                    anchors.left: parent.left
                    anchors.leftMargin: 14
                    text: modelData.label
                    color: root.currentIndex === index ? Theme.text : Theme.textDim
                    font.family: Theme.fontFamily
                    font.pixelSize: Theme.fontBody
                }

                HoverHandler { id: hover }
                TapHandler { onTapped: root.currentIndex = index }
            }
        }

        Item { Layout.fillHeight: true }

        // --- the workspace ---
        ColumnLayout {
            Layout.fillWidth: true
            spacing: 4

            Label {
                text: "WORKSPACE"
                color: Theme.textFaint
                font.family: Theme.fontFamily
                font.pixelSize: 10
                font.letterSpacing: 1
            }
            Label {
                text: studio.workspace.length > 0
                      ? studio.workspace
                      : "none — the model has no files"
                color: studio.workspace.length > 0 ? Theme.textDim : Theme.textFaint
                font.family: Theme.monoFamily
                font.pixelSize: Theme.fontSmall
                Layout.fillWidth: true
                wrapMode: Text.WrapAnywhere
                maximumLineCount: 3
                elide: Text.ElideMiddle
            }
            StudioButton {
                text: studio.workspace.length > 0 ? "Change folder" : "Choose a folder"
                Layout.fillWidth: true
                onClicked: root.openWorkspace()
            }
        }

        Rectangle { Layout.fillWidth: true; height: 1; color: Theme.border }

        // --- the connection ---
        ColumnLayout {
            Layout.fillWidth: true
            spacing: 4

            RowLayout {
                spacing: 6
                Rectangle {
                    width: 7; height: 7; radius: 3.5
                    color: studio.connected ? Theme.ok : Theme.textFaint
                }
                Label {
                    text: studio.status
                    color: Theme.textDim
                    font.family: Theme.fontFamily
                    font.pixelSize: Theme.fontSmall
                    Layout.fillWidth: true
                    elide: Text.ElideRight
                }
            }

            // The pinned identity, shown rather than hidden. It is not a
            // secret, and it is what someone compares against
            // `hypernix-t1 status` when they want to be sure.
            Label {
                text: studio.displayFingerprint()
                visible: studio.connected && studio.displayFingerprint().length > 0
                color: Theme.textFaint
                font.family: Theme.monoFamily
                font.pixelSize: 10
                Layout.fillWidth: true
                elide: Text.ElideRight
            }

            // Hidden in local mode: there is nothing to connect to, and
            // a connect form on a machine running its own model is an
            // invitation to think one is required.
            ConnectPanel {
                Layout.fillWidth: true
                visible: !studio.connected && studio.source !== "local"
            }
        }
    }
}
