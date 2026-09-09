import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// ModelsView.qml — what is available, what is loaded, and getting more.
//
// GPU layers is the one number worth exposing and it defaults to "let
// the server decide", because the server is the machine that knows how
// much VRAM is actually free. A number chosen here is a guess about
// somebody else's GPU, and a wrong one is an out-of-memory error minutes
// into a load.
Item {
    id: root
    property int gpuLayers: -1

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: Theme.gapLarge
        spacing: Theme.gap

        RowLayout {
            Layout.fillWidth: true
            spacing: Theme.gap

            Label {
                text: "Models"
                color: Theme.text
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontDisplay
                font.weight: Font.DemiBold
            }
            Item { Layout.fillWidth: true }
            StudioButton {
                text: "Refresh"
                enabled: studio.connected
                onClicked: studio.refresh()
            }
        }

        // --- offloading ---
        Rectangle {
            Layout.fillWidth: true
            implicitHeight: offload.implicitHeight + Theme.gap * 2
            radius: Theme.radius
            color: Theme.surface
            border.width: 1
            border.color: Theme.border

            ColumnLayout {
                id: offload
                anchors.fill: parent
                anchors.margins: Theme.gap
                spacing: 6

                RowLayout {
                    Layout.fillWidth: true
                    Label {
                        text: "GPU offloading"
                        color: Theme.text
                        font.family: Theme.fontFamily
                        font.pixelSize: Theme.fontBody
                        font.weight: Font.DemiBold
                    }
                    Item { Layout.fillWidth: true }
                    Label {
                        text: root.gpuLayers < 0
                              ? "server decides"
                              : root.gpuLayers + " layers"
                        color: root.gpuLayers < 0 ? Theme.ok : Theme.text
                        font.family: Theme.monoFamily
                        font.pixelSize: Theme.fontSmall
                    }
                }

                Slider {
                    Layout.fillWidth: true
                    from: -1
                    to: 99
                    stepSize: 1
                    value: root.gpuLayers
                    onMoved: root.gpuLayers = Math.round(value)

                    background: Rectangle {
                        x: parent.leftPadding
                        y: parent.topPadding + parent.availableHeight / 2 - 2
                        width: parent.availableWidth
                        height: 4
                        radius: 2
                        color: Theme.surfaceAlt
                        Rectangle {
                            width: parent.parent.visualPosition * parent.width
                            height: parent.height
                            radius: 2
                            color: Theme.accent
                        }
                    }
                    handle: Rectangle {
                        x: parent.leftPadding + parent.visualPosition *
                           (parent.availableWidth - width)
                        y: parent.topPadding + parent.availableHeight / 2 - height / 2
                        width: 16; height: 16; radius: 8
                        color: Theme.text
                        border.width: 1
                        border.color: Theme.border
                    }
                }

                Label {
                    text: "All the way left leaves it to the server, which is " +
                          "usually right — it is the machine that knows how " +
                          "much VRAM is free. 0 is CPU only."
                    color: Theme.textFaint
                    font.family: Theme.fontFamily
                    font.pixelSize: 11
                    Layout.fillWidth: true
                    wrapMode: Text.WordWrap
                }
            }
        }

        // --- the list ---
        ListView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            spacing: 6
            model: studio.models
            ScrollBar.vertical: ScrollBar {}

            delegate: Rectangle {
                required property var modelData
                width: ListView.view.width
                implicitHeight: 66
                radius: Theme.radius
                color: hover.hovered ? Theme.hover : Theme.surface
                border.width: 1
                border.color: modelData.id === studio.activeModel
                              ? Theme.accent : Theme.border
                Behavior on color { ColorAnimation { duration: Theme.anim } }

                HoverHandler { id: hover }

                RowLayout {
                    anchors.fill: parent
                    anchors.margins: Theme.gap
                    spacing: Theme.gap

                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 2
                        Label {
                            text: modelData.displayName
                            color: Theme.text
                            font.family: Theme.fontFamily
                            font.pixelSize: Theme.fontBody
                            font.weight: Font.DemiBold
                            Layout.fillWidth: true
                            elide: Text.ElideRight
                        }
                        Label {
                            text: {
                                var bits = []
                                if (modelData.parametersB > 0)
                                    bits.push(modelData.parametersB.toFixed(1) + "B")
                                if (modelData.quantisation)
                                    bits.push(modelData.quantisation)
                                if (modelData.contextLimit > 0)
                                    bits.push((modelData.contextLimit / 1024).toFixed(0) +
                                              "K ctx")
                                if (modelData.backend)
                                    bits.push(modelData.backend)
                                return bits.join("  ·  ")
                            }
                            color: Theme.textFaint
                            font.family: Theme.monoFamily
                            font.pixelSize: 11
                            Layout.fillWidth: true
                            elide: Text.ElideRight
                        }
                    }

                    StudioButton {
                        text: modelData.id === studio.activeModel ? "Loaded" : "Load"
                        primary: modelData.id !== studio.activeModel
                        enabled: modelData.id !== studio.activeModel
                        onClicked: studio.switchModel(modelData.id, root.gpuLayers)
                    }
                }
            }

            // An empty list is either "not connected" or "the server has
            // no models", and those need different actions.
            Label {
                anchors.centerIn: parent
                visible: parent.count === 0
                text: studio.connected
                      ? "The server's registry is empty.\nRun `hypernix-t1 index` on it."
                      : "Connect to a server to see its models."
                color: Theme.textFaint
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontBody
                horizontalAlignment: Text.AlignHCenter
            }
        }

        // --- Hugging Face ---
        RowLayout {
            Layout.fillWidth: true
            spacing: Theme.gap

            StudioField {
                id: hfLink
                Layout.fillWidth: true
                placeholderText: "a Hugging Face model page or file link"
                mono: true
                enabled: studio.connected
                onAccepted: fetch.clicked()
            }
            StudioButton {
                id: fetch
                text: "Find"
                enabled: studio.connected && hfLink.text.length > 0 && !studio.busy
                // Resolved by the server, which is the machine that will
                // hold the weights and the one with the HF token.
                onClicked: studio.searchModels(hfLink.text)
            }
        }
    }
}
