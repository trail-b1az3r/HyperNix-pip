import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts

// ModelsView.qml — what is available, what is loaded, and getting more.
//
// Two sources, one view. "This machine" lists the GGUF files on the
// disk and runs one here; "Server" is what Studio has always done. The
// switch is at the top because it changes what everything below means.
//
// GPU layers is the one number worth exposing. On the server it
// defaults to "let the server decide", because that is the machine that
// knows how much VRAM is free. Locally it defaults to CPU-only, which
// is the setting that always works -- an offload guess that does not
// fit is an out-of-memory minutes into a load, and this machine's GPU
// may be busy with a training run.
Item {
    id: root
    property int gpuLayers: local ? 0 : -1
    readonly property bool local: studio.source === "local"

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
                enabled: root.local || studio.connected
                onClicked: root.local ? studio.scanLocalModels() : studio.refresh()
            }
        }

        // --- where models come from ---
        RowLayout {
            Layout.fillWidth: true
            spacing: 0

            Repeater {
                model: [
                    { key: "local", label: "This machine" },
                    { key: "server", label: "Server" }
                ]
                delegate: Rectangle {
                    required property var modelData
                    readonly property bool active: studio.source === modelData.key
                    Layout.fillWidth: true
                    implicitHeight: 34
                    color: active ? Theme.accent : Theme.surface
                    border.width: 1
                    border.color: active ? Theme.accent : Theme.border
                    Behavior on color { ColorAnimation { duration: Theme.anim } }

                    Label {
                        anchors.centerIn: parent
                        text: modelData.label
                        color: parent.active ? "#ffffff" : Theme.text
                        font.family: Theme.fontFamily
                        font.pixelSize: Theme.fontBody
                        font.weight: parent.active ? Font.DemiBold : Font.Normal
                    }
                    TapHandler { onTapped: studio.source = modelData.key }
                }
            }
        }

        // Said once, plainly, rather than left for a failed load to
        // explain. A build without llama.cpp can still list what is on
        // the disk -- reading a header needs no inference -- so the
        // models below are real even here, and only Load is refused.
        Rectangle {
            Layout.fillWidth: true
            visible: root.local && !studio.localAvailable
            implicitHeight: noLocal.implicitHeight + Theme.gap * 2
            radius: Theme.radius
            color: Theme.surfaceAlt
            border.width: 1
            border.color: Theme.border
            Label {
                id: noLocal
                anchors.fill: parent
                anchors.margins: Theme.gap
                text: studio.localUnavailableReason
                color: Theme.textFaint
                font.family: Theme.fontFamily
                font.pixelSize: 11
                wrapMode: Text.WordWrap
            }
        }

        // --- where to look ---
        RowLayout {
            Layout.fillWidth: true
            visible: root.local
            spacing: Theme.gap

            Label {
                text: studio.modelFolders.length > 0
                      ? studio.modelFolders.length + " folder(s) added, plus the usual places"
                      : "Looking in ~/.hypernix/models, the Hugging Face cache and ~/models"
                color: Theme.textFaint
                font.family: Theme.fontFamily
                font.pixelSize: 11
                Layout.fillWidth: true
                elide: Text.ElideRight
            }
            StudioButton {
                text: "Add folder"
                onClicked: folderDialog.open()
            }
        }

        FolderDialog {
            id: folderDialog
            title: "A folder with .gguf files in it"
            onAccepted: studio.addModelFolder(selectedFolder)
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
                              ? (root.local ? "as many as fit" : "server decides")
                              : (root.gpuLayers === 0
                                 ? "CPU only" : root.gpuLayers + " layers")
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
                    text: root.local
                          ? "0 is CPU only, and it is the setting that always " +
                            "works. Raise it to put that many layers on the " +
                            "GPU — if they do not fit, the load fails minutes " +
                            "in. All the way left lets llama.cpp fit as many " +
                            "as it can."
                          : "All the way left leaves it to the server, which is " +
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
            model: root.local ? studio.localModels : studio.models
            ScrollBar.vertical: ScrollBar {}

            delegate: Rectangle {
                required property var modelData
                width: ListView.view.width
                implicitHeight: 66
                radius: Theme.radius
                color: hover.hovered ? Theme.hover : Theme.surface
                border.width: 1
                readonly property bool isActive: root.local
                    ? (studio.localLoaded &&
                       modelData.path === studio.localInfo.path)
                    : modelData.id === studio.activeModel
                border.color: isActive ? Theme.accent : Theme.border
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
                            // A server model calls it displayName and a
                            // local one calls it name. Read whichever is
                            // there rather than teaching one side the
                            // other's spelling: undefined here would bind
                            // as an empty label and look like a model with
                            // no name.
                            text: modelData.displayName !== undefined
                                  ? modelData.displayName
                                  : (modelData.name !== undefined
                                     ? modelData.name : "")
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
                                if (root.local) {
                                    if (modelData.ok === false) {
                                        // Listed rather than hidden: a
                                        // file that is there and broken
                                        // is something you want to see.
                                        return "unreadable — " + modelData.error
                                    }
                                    if (modelData.parameters)
                                        bits.push(modelData.parameters)
                                    if (modelData.quantisation)
                                        bits.push(modelData.quantisation)
                                    if (modelData.contextLength > 0)
                                        bits.push((modelData.contextLength / 1024)
                                                  .toFixed(0) + "K ctx")
                                    if (modelData.size)
                                        bits.push(modelData.size)
                                    if (modelData.architecture)
                                        bits.push(modelData.architecture)
                                    return bits.join("  ·  ")
                                }
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
                        text: parent.parent.isActive
                              ? (root.local ? "Unload" : "Loaded") : "Load"
                        primary: !parent.parent.isActive
                        enabled: root.local
                                 ? (studio.localAvailable &&
                                    modelData.ok !== false && !studio.busy)
                                 : !parent.parent.isActive
                        onClicked: {
                            if (!root.local) {
                                studio.switchModel(modelData.id, root.gpuLayers)
                            } else if (parent.parent.isActive) {
                                studio.unloadLocalModel()
                            } else {
                                studio.loadLocalModel(modelData.path,
                                                      root.gpuLayers, 0)
                            }
                        }
                    }
                }
            }

            // An empty list is either "not connected" or "the server has
            // no models", and those need different actions.
            Label {
                anchors.centerIn: parent
                visible: parent.count === 0
                text: root.local
                      ? "No .gguf files found.\nAdd a folder above, or put one " +
                        "in ~/.hypernix/models."
                      : (studio.connected
                         ? "The server's registry is empty.\nRun `hypernix-t1 index` on it."
                         : "Connect to a server to see its models.")
                color: Theme.textFaint
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontBody
                horizontalAlignment: Text.AlignHCenter
            }
        }

        // --- Hugging Face ---
        // Server-side: it is the machine that will hold the weights and
        // the one with the token. Hidden rather than disabled in local
        // mode, because a greyed-out box invites a click that cannot be
        // explained.
        RowLayout {
            Layout.fillWidth: true
            visible: !root.local
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
