import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// ToolApproval.qml — the dialog that has to be answered.
//
// This is the security surface of the whole app, and its design is
// mostly a list of things it does not have:
//
//   no "approve all"        — every call is a separate decision
//   no "remember this"      — a remembered yes is a yes nobody read
//   no timeout              — nothing defaults to approval
//   no click-outside-to-yes — dismissing is declining
//   no default focus on the affirmative button
//
// Each of those would be the same feature under a different name: a way
// for a file to be written without anyone having looked. The one
// convenience it does have is showing the resolved path and the actual
// diff, because the failure mode that matters is not "someone clicked
// yes" but "someone clicked yes without knowing what they agreed to".
Rectangle {
    id: root
    color: Qt.rgba(0, 0, 0, 0.72)

    readonly property var call: studio.pendingApproval
    readonly property bool suspicious: field("suspicious") === true
    readonly property bool isSearch: field("tool") === "web_search"
    readonly property bool isDelete: field("tool") === "delete_file"

    // Every read of `call` goes through this. See the note above: an
    // undefined field leaves a QString property at its previous value,
    // and in this dialog the previous value is the last file somebody
    // was asked about.
    function field(name) {
        if (!call || call[name] === undefined || call[name] === null) return ""
        return call[name]
    }

    // Swallows every click that is not on the panel. Not a way to
    // dismiss: clicking away from a dialog is how people close things
    // they have not read, and here that would be an answer.
    TapHandler {}

    Rectangle {
        anchors.centerIn: parent
        width: Math.min(720, parent.width - 80)
        height: Math.min(implicitHeight, parent.height - 80)
        implicitHeight: panel.implicitHeight + Theme.gapLarge * 2
        radius: Theme.radius
        color: Theme.surface
        border.width: 1
        border.color: root.suspicious ? Theme.accent : Theme.border

        ColumnLayout {
            id: panel
            anchors.fill: parent
            anchors.margins: Theme.gapLarge
            spacing: Theme.gap

            // --- what is being asked ---
            RowLayout {
                spacing: 8
                Rectangle {
                    width: 7; height: 7; radius: 3.5
                    color: root.suspicious ? Theme.accent : Theme.warn
                }
                Label {
                    text: {
                        switch (root.field("tool")) {
                        case "write_file":  return "Change a file?"
                        case "create_file": return "Create a file?"
                        case "delete_file": return "Delete a file?"
                        case "web_search":  return "Search the web?"
                        default:            return "Allow this?"
                        }
                    }
                    color: Theme.text
                    font.family: Theme.fontFamily
                    font.pixelSize: Theme.fontTitle
                    font.weight: Font.DemiBold
                }
                Item { Layout.fillWidth: true }
                Label {
                    text: root.field("tool")
                    color: Theme.textFaint
                    font.family: Theme.monoFamily
                    font.pixelSize: Theme.fontSmall
                }
            }

            Label {
                text: root.field("reason")
                color: Theme.textDim
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontBody
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
            }

            // --- the resolved path ---
            //
            // The path that would actually be touched, not the one the
            // model wrote. They can differ, and when they do that is the
            // most important thing on the screen.
            ColumnLayout {
                visible: !root.isSearch && root.field("tool").length > 0
                Layout.fillWidth: true
                spacing: 2

                Label {
                    text: "FILE"
                    color: Theme.textFaint
                    font.family: Theme.fontFamily
                    font.pixelSize: 10
                    font.letterSpacing: 1
                }
                Label {
                    text: root.field("path")
                    color: Theme.text
                    font.family: Theme.monoFamily
                    font.pixelSize: Theme.fontSmall
                    Layout.fillWidth: true
                    wrapMode: Text.WrapAnywhere
                }
                Label {
                    // Only when the model asked for something that
                    // resolved elsewhere -- a `..`, or a different
                    // spelling. Worth seeing.
                    visible: root.field("requestedPath").length > 0 &&
                             root.field("path").indexOf(
                                 root.field("requestedPath")) < 0
                    text: "the model asked for: " + root.field("requestedPath")
                    color: Theme.warn
                    font.family: Theme.monoFamily
                    font.pixelSize: 10
                    Layout.fillWidth: true
                    wrapMode: Text.WrapAnywhere
                }
            }

            // --- the search text ---
            ColumnLayout {
                visible: root.isSearch
                Layout.fillWidth: true
                spacing: 2
                Label {
                    text: "SEARCH TEXT — written by the model"
                    color: Theme.textFaint
                    font.family: Theme.fontFamily
                    font.pixelSize: 10
                    font.letterSpacing: 1
                }
                Rectangle {
                    Layout.fillWidth: true
                    implicitHeight: queryText.implicitHeight + 16
                    radius: Theme.radiusSmall
                    color: Theme.surfaceAlt
                    border.width: 1
                    border.color: Theme.border
                    Label {
                        id: queryText
                        anchors.fill: parent
                        anchors.margins: 8
                        text: root.field("query")
                        color: Theme.text
                        font.family: Theme.monoFamily
                        font.pixelSize: Theme.fontSmall
                        wrapMode: Text.WordWrap
                        textFormat: Text.PlainText
                    }
                }
            }

            // --- the contents ---
            //
            // What would be written, in full and scrollable. A dialog
            // that summarises this ("142 lines") is a dialog that asks
            // someone to approve something they cannot see.
            ColumnLayout {
                visible: !root.isSearch && !root.isDelete &&
                         root.field("contents").length > 0
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 2

                Label {
                    text: root.field("existing").length > 0
                          ? "NEW CONTENTS — replacing " +
                            root.field("existing").split("\n").length + " lines"
                          : "CONTENTS"
                    color: Theme.textFaint
                    font.family: Theme.fontFamily
                    font.pixelSize: 10
                    font.letterSpacing: 1
                }

                ScrollView {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 220
                    clip: true

                    TextArea {
                        readOnly: true
                        text: root.field("contents")
                        color: Theme.text
                        font.family: Theme.monoFamily
                        font.pixelSize: Theme.fontSmall
                        selectByMouse: true
                        wrapMode: TextArea.NoWrap
                        textFormat: TextEdit.PlainText

                        background: Rectangle {
                            radius: Theme.radiusSmall
                            color: Theme.surfaceAlt
                            border.width: 1
                            border.color: Theme.border
                        }
                    }
                }
            }

            // --- the answer ---
            RowLayout {
                Layout.fillWidth: true
                Layout.topMargin: 4
                spacing: Theme.gap

                Label {
                    text: root.isDelete
                          ? "There is no undo for this."
                          : "Studio asks every time."
                    color: root.isDelete ? Theme.accentHi : Theme.textFaint
                    font.family: Theme.fontFamily
                    font.pixelSize: Theme.fontSmall
                    Layout.fillWidth: true
                    wrapMode: Text.WordWrap
                }

                // Decline first, and focused. The affirmative button is
                // never the default: a dialog answered by reflex should
                // be answered "no".
                StudioButton {
                    text: "No"
                    focus: true
                    Keys.onReturnPressed: studio.rejectTool()
                    onClicked: studio.rejectTool()
                }
                StudioButton {
                    text: root.isDelete ? "Delete it"
                                        : (root.isSearch ? "Search" : "Do it")
                    danger: root.isDelete || root.suspicious
                    primary: !root.isDelete && !root.suspicious
                    onClicked: studio.approveTool()
                }
            }
        }
    }

    // Escape declines. The one keyboard shortcut, and it goes the safe
    // way.
    Keys.onEscapePressed: studio.rejectTool()
    focus: visible
}
