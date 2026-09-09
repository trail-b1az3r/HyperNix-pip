import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// MessageBubble.qml — one turn, rendered as what it is.
//
// The four roles are visually distinct because conflating them hides the
// thing that matters. A `tool` turn is a fact about the filesystem; an
// `assistant` turn is a model's claim about it. Rendering both as prose
// in the same bubble is how somebody comes to believe a file was written
// because a model said it was.
Rectangle {
    id: root
    property string role: "user"
    property string text: ""
    property string detail: ""

    readonly property bool isUser: role === "user"
    readonly property bool isTool: role === "tool"
    readonly property bool isNote: role === "system"

    implicitHeight: content.implicitHeight + Theme.gap * 2
    radius: Theme.radius
    color: isUser ? Theme.surfaceAlt
                  : (isNote ? "transparent" : Theme.surface)
    border.width: isNote ? 0 : 1
    border.color: isUser ? Theme.border
                         : (isTool ? Qt.rgba(0.25, 0.64, 0.42, 0.35) : Theme.border)

    ColumnLayout {
        id: content
        anchors.fill: parent
        anchors.margins: Theme.gap
        spacing: 5

        RowLayout {
            spacing: 6
            visible: !root.isNote

            Rectangle {
                width: 5; height: 5; radius: 2.5
                color: root.isUser ? Theme.textDim
                                   : (root.isTool ? Theme.ok : Theme.accent)
            }
            Label {
                text: root.isUser ? "you"
                                  : (root.isTool ? "tool" : "model")
                color: Theme.textFaint
                font.family: Theme.fontFamily
                font.pixelSize: 10
                font.letterSpacing: 0.6
            }
        }

        Label {
            text: root.text
            // Monospace for tool output because it is file contents and
            // directory listings, where alignment carries meaning.
            font.family: root.isTool ? Theme.monoFamily : Theme.fontFamily
            font.pixelSize: root.isNote ? Theme.fontSmall : Theme.fontBody
            color: root.isNote ? Theme.textFaint : Theme.text
            Layout.fillWidth: true
            wrapMode: Text.Wrap
            textFormat: Text.PlainText   // never rich text: the content
                                         // is a model's output and a
                                         // file's contents, and both
                                         // would happily contain markup
            // Selectable, because the first thing anyone wants to do
            // with a code block is copy it.
            TextEdit {
                anchors.fill: parent
                visible: false
            }
        }
    }

    // Plain-text selection, without making the label editable.
    TextEdit {
        anchors.fill: parent
        anchors.margins: Theme.gap
        text: root.text
        readOnly: true
        selectByMouse: true
        opacity: 0
        font.family: root.isTool ? Theme.monoFamily : Theme.fontFamily
        font.pixelSize: root.isNote ? Theme.fontSmall : Theme.fontBody
        wrapMode: TextEdit.Wrap
    }
}
