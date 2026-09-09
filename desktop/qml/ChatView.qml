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
                        text: studio.activeModel.length > 0
                              ? studio.activeModel : "No model loaded"
                        color: studio.activeModel.length > 0
                               ? Theme.text : Theme.textFaint
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
                        placeholderText: studio.connected
                                         ? "Ask, or describe a change to make"
                                         : "Connect to a server first"
                        enabled: studio.connected && !studio.busy
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
                    text: studio.busy ? "…" : "Send"
                    primary: true
                    enabled: studio.connected && !studio.busy &&
                             input.text.trim().length > 0
                    onClicked: {
                        studio.send(input.text.trim())
                        input.text = ""
                    }
                }
            }
        }
    }
}
