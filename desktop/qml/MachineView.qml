import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// MachineView.qml — what the server's hardware is doing.
//
// From /training/resources, which is the server's own GPU abstraction,
// so this shows what the machine actually detected rather than what the
// app hopes is there. NVIDIA, AMD and CPU-only all arrive through the
// same shape.
//
// A measurement the vendor tool declined to give arrives as -1 and
// renders as "—". Never 0: nvidia-smi answers "[N/A]" for utilisation on
// several consumer cards, and a full bar of nothing is a lie a dashboard
// tells confidently.
Item {
    Timer {
        // Two seconds while this view is showing, and not at all
        // otherwise: each tick shells out to nvidia-smi on the server,
        // and polling a machine that nobody is looking at is a
        // background process someone eventually has to explain.
        interval: 2000
        running: parent.visible && studio.connected
        repeat: true
        onTriggered: studio.refresh()
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: Theme.gapLarge
        spacing: Theme.gap

        Label {
            text: "Machine"
            color: Theme.text
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fontDisplay
            font.weight: Font.DemiBold
        }

        Label {
            text: studio.serverName.length > 0
                  ? studio.serverName + "  ·  T1 v" + studio.serverVersion
                  : "not connected"
            color: Theme.textFaint
            font.family: Theme.monoFamily
            font.pixelSize: Theme.fontSmall
        }

        // --- GPUs ---
        Repeater {
            model: studio.gpus
            delegate: Rectangle {
                required property var modelData
                Layout.fillWidth: true
                implicitHeight: card.implicitHeight + Theme.gap * 2
                radius: Theme.radius
                color: Theme.surface
                border.width: 1
                border.color: Theme.border

                ColumnLayout {
                    id: card
                    anchors.fill: parent
                    anchors.margins: Theme.gap
                    spacing: 8

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8
                        Label {
                            text: "GPU " + modelData.index
                            color: Theme.textFaint
                            font.family: Theme.monoFamily
                            font.pixelSize: Theme.fontSmall
                        }
                        Label {
                            text: modelData.name
                            color: Theme.text
                            font.family: Theme.fontFamily
                            font.pixelSize: Theme.fontBody
                            font.weight: Font.DemiBold
                            Layout.fillWidth: true
                            elide: Text.ElideRight
                        }
                        // The compute stack, from the server's Vendor
                        // mapping: cuda, rocm, mps, xpu or cpu.
                        Rectangle {
                            implicitWidth: framework.implicitWidth + 14
                            implicitHeight: 20
                            radius: 10
                            color: Theme.surfaceAlt
                            border.width: 1
                            border.color: Theme.border
                            Label {
                                id: framework
                                anchors.centerIn: parent
                                text: modelData.framework
                                color: Theme.textDim
                                font.family: Theme.monoFamily
                                font.pixelSize: 10
                            }
                        }
                    }

                    Meter {
                        Layout.fillWidth: true
                        label: "VRAM"
                        used: modelData.memoryUsedMb
                        total: modelData.memoryTotalMb
                        unit: "MB"
                    }
                    Meter {
                        Layout.fillWidth: true
                        label: "Utilisation"
                        used: modelData.utilisation
                        total: modelData.utilisation < 0 ? -1 : 100
                        unit: "%"
                    }
                }
            }
        }

        // --- CPU and RAM ---
        Rectangle {
            Layout.fillWidth: true
            implicitHeight: host.implicitHeight + Theme.gap * 2
            radius: Theme.radius
            color: Theme.surface
            border.width: 1
            border.color: Theme.border

            ColumnLayout {
                id: host
                anchors.fill: parent
                anchors.margins: Theme.gap
                spacing: 8

                Meter {
                    Layout.fillWidth: true
                    label: "CPU"
                    used: studio.cpuPercent
                    total: studio.cpuPercent < 0 ? -1 : 100
                    unit: "%"
                }
                Meter {
                    Layout.fillWidth: true
                    label: "RAM"
                    used: studio.ramUsedMb
                    total: studio.ramTotalMb
                    unit: "MB"
                }
            }
        }

        Label {
            visible: studio.connected && studio.gpus.length === 0
            text: "No GPU reported. Either this machine has none, or the " +
                  "server declined the request — reading machine resources " +
                  "needs an admin key, or trusted-network mode turned on."
            color: Theme.textFaint
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fontSmall
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
        }

        Item { Layout.fillHeight: true }
    }
}
