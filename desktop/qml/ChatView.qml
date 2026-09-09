import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// ChatView.qml — the conversation.
//
// Four kinds of turn, and they look different on purpose: what you said,
// what the model said, what a tool did, and what Studio is telling you.
// A transcript that renders all four the same is one where "I wrote a
// file" and "the model claims it wrote a file" are indistinguishable.
Item {
    id: root

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // --- header ---
        Rectangle {
            Layout.fillWidth: true
            implicitHeight: 52
            color: Theme.surface

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: Theme.gapLarge
                anchors.rightMargin: Theme.gapLarge
                spacing: Theme.gap

                ColumnLayout {
                    spacing: 0
                    Label {
                        // Whichever model is actually going to answer.
                        // Binding to activeModel alone showed "No model
                        // loaded" over a working local conversation.
                        readonly property string current:
                            studio.source === "local"
                            ? (studio.localLoaded
                               ? (studio.localInfo.name || "") : "")
                            : studio.activeModel
                        text: current.length > 0 ? current : "No model loaded"
                        color: current.length > 0 ? Theme.text : Theme.textFaint
                        font.family: Theme.fontFamily
                        font.pixelSize: Theme.fontBody
                        font.weight: Font.DemiBold
                    }
                    Label {
                        text: studio.workspace.length > 0
                              ? "can edit files in " + studio.workspace
                              : "no workspace — the model has no file tools"
                        color: Theme.textFaint
                        font.family: Theme.fontFamily
                        font.pixelSize: 11
                    }
                }

                Item { Layout.fillWidth: true }

                StudioButton {
                    text: "Clear"
                    enabled: studio.messages.length > 0
                    onClicked: studio.clearConversation()
                }
            }

            Rectangle {
                anchors.bottom: parent.bottom
                width: parent.width; height: 1
                color: Theme.border
            }
        }

        // --- transcript ---
        ListView {
            id: transcript
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            model: studio.messages
            spacing: Theme.gap
            topMargin: Theme.gapLarge
            bottomMargin: Theme.gapLarge
            leftMargin: Theme.gapLarge
            rightMargin: Theme.gapLarge

            // Follows the tail only when the user is already at the
            // tail. Scrolling back to read something and being yanked
            // forward by the next token is the single most irritating
            // thing a chat UI can do.
            property bool atTail: true
            onContentYChanged: atTail = (contentY >= contentHeight - height - 40)
            onCountChanged: if (atTail) positionViewAtEnd()

            delegate: MessageBubble {
                width: transcript.width - Theme.gapLarge * 2
                role: modelData.role
                text: modelData.text
                detail: modelData.detail
            }

            ScrollBar.vertical: ScrollBar {}
        }

        // --- composer ---
        Rectangle {
            Layout.fillWidth: true
            implicitHeight: composer.implicitHeight + Theme.gap * 2
            color: Theme.surface

            Rectangle {
                width: parent.width; height: 1
                color: Theme.border
            }

            RowLayout {
                id: composer
                anchors.fill: parent
                anchors.margins: Theme.gap
                anchors.leftMargin: Theme.gapLarge
                anchors.rightMargin: Theme.gapLarge
                spacing: Theme.gap

                ScrollView {
                    Layout.fillWidth: true
                    implicitHeight: Math.min(140, Math.max(34, input.implicitHeight))

                    TextArea {
                        id: input
                        // `ready` rather than `connected`: in local
                        // mode there is no server to connect to, and
                        // gating on one is what made the composer dead
                        // with a model loaded and running.
                        placeholderText: studio.ready
                                         ? "Ask, or describe a change to make"
                                         : (studio.source === "local"
                                            ? "Load a model in the Models tab"
                                            : "Connect to a server first")
                        enabled: studio.ready && !studio.busy
                        color: Theme.text
                        placeholderTextColor: Theme.textFaint
                        font.family: Theme.fontFamily
                        font.pixelSize: Theme.fontBody
                        wrapMode: TextArea.Wrap
                        selectByMouse: true
                        leftPadding: 10
                        rightPadding: 10
                        topPadding: 7

                        background: Rectangle {
                            radius: Theme.radiusSmall
                            color: Theme.surfaceAlt
                            border.width: 1
                            border.color: input.activeFocus ? Theme.accent : Theme.border
                            Behavior on border.color {
                                ColorAnimation { duration: Theme.anim }
                            }
                        }

                        // Enter sends, Shift+Enter is a newline. The
                        // opposite of what a text editor does, and what
                        // every chat app does, because this is a chat.
                        Keys.onReturnPressed: function (event) {
                            if (event.modifiers & Qt.ShiftModifier) {
                                event.accepted = false
                                return
                            }
                            send.clicked()
                        }
                    }
                }

                StudioButton {
                    id: send
                    // Turns into Stop while a local model is generating.
                    // A long answer on a slow machine is a minute of
                    // watching, and there has to be a way out of it that
                    // is not closing the window.
                    readonly property bool stopping:
                        studio.busy && studio.source === "local"
                    text: stopping ? "Stop" : (studio.busy ? "…" : "Send")
                    primary: true
                    enabled: stopping ||
                             (studio.ready && !studio.busy &&
                              input.text.trim().length > 0)
                    onClicked: {
                        if (stopping) {
                            studio.stopGenerating()
                            return
                        }
                        studio.send(input.text.trim())
                        input.text = ""
                    }
                }
            }
        }
    }
}
