import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// SettingsView.qml — the three things Studio remembers.
//
// Small on purpose. Studio had no settings at all, and the answer to
// that is not forty of them; it is the handful whose defaults are
// genuinely wrong for somebody.
//
// The two shells are separate fields rather than one because they are
// for different jobs: `shell` is what a person types in, `codingShell`
// is what generated scripts target. A script Studio writes gets
// committed, run by CI and pasted into a Dockerfile, and fish is not
// installed in any of those — so the coding default is bash even for
// somebody whose own shell is fish. The footer says so, because
// otherwise the two fields look like the same setting duplicated.
Flickable {
    id: root

    property var settings: bridge.settings

    contentHeight: column.implicitHeight + Theme.gap * 2
    clip: true

    ColumnLayout {
        id: column
        width: root.width
        anchors.margins: Theme.gap
        spacing: Theme.gap * 2

        Label {
            text: "Settings"
            font.pixelSize: 22
            font.weight: Font.DemiBold
            color: Theme.text
            Layout.leftMargin: Theme.gap
            Layout.topMargin: Theme.gap
        }

        // --- shells ---
        GroupBox {
            title: "Shells"
            Layout.fillWidth: true
            Layout.leftMargin: Theme.gap
            Layout.rightMargin: Theme.gap

            ColumnLayout {
                anchors.fill: parent
                spacing: Theme.gap

                RowLayout {
                    Layout.fillWidth: true
                    Label {
                        text: "Yours"
                        color: Theme.text
                        Layout.preferredWidth: 140
                    }
                    ComboBox {
                        id: interactiveShell
                        Layout.fillWidth: true
                        editable: true
                        model: root.settings.knownShells
                        currentIndex: model.indexOf(root.settings.shell)
                        editText: root.settings.shell
                        onAccepted: root.settings.setShell(editText)
                        onActivated: root.settings.setShell(textAt(index))
                    }
                }

                Label {
                    text: "The shell you sit in front of. Studio does not run "
                          + "anything here — this is what it writes for."
                    wrapMode: Text.WordWrap
                    font.pixelSize: 12
                    color: Theme.textDim
                    Layout.fillWidth: true
                }

                RowLayout {
                    Layout.fillWidth: true
                    Label {
                        text: "For scripts"
                        color: Theme.text
                        Layout.preferredWidth: 140
                    }
                    ComboBox {
                        id: codingShell
                        Layout.fillWidth: true
                        editable: true
                        model: root.settings.knownShells
                        currentIndex: model.indexOf(root.settings.codingShell)
                        editText: root.settings.codingShell
                        onAccepted: root.settings.setCodingShell(editText)
                        onActivated: root.settings.setCodingShell(textAt(index))
                    }
                }

                Label {
                    text: "What generated scripts target. Kept separate from "
                          + "yours because a script gets committed, run by CI "
                          + "and pasted into a Dockerfile — and fish is not "
                          + "installed in any of those."
                    wrapMode: Text.WordWrap
                    font.pixelSize: 12
                    color: Theme.textDim
                    Layout.fillWidth: true
                }
            }
        }

        // --- context ---
        GroupBox {
            title: "Allowed context"
            Layout.fillWidth: true
            Layout.leftMargin: Theme.gap
            Layout.rightMargin: Theme.gap

            ColumnLayout {
                anchors.fill: parent
                spacing: Theme.gap

                RowLayout {
                    Layout.fillWidth: true
                    Slider {
                        id: contextSlider
                        Layout.fillWidth: true
                        from: root.settings.minimumContext
                        to: 131072
                        stepSize: 512
                        snapMode: Slider.SnapAlways
                        value: root.settings.contextLimit
                        onMoved: root.settings.setContextLimit(Math.round(value))
                    }
                    Label {
                        text: root.settings.contextLimit + " tokens"
                        color: Theme.text
                        Layout.preferredWidth: 120
                        horizontalAlignment: Text.AlignRight
                    }
                }

                Label {
                    text: "How much context to ask a model for. A model with "
                          + "less than this gets asked for what it has — "
                          + "asking for more fails at load with an error "
                          + "about the KV cache, which reads as a bug rather "
                          + "than a setting."
                    wrapMode: Text.WordWrap
                    font.pixelSize: 12
                    color: Theme.textDim
                    Layout.fillWidth: true
                }

                // Shown only when the last load did not get what it
                // asked for. A person who typed 32768 and got 8192
                // needs to be told which ceiling they hit.
                Label {
                    visible: bridge.contextNote.length > 0
                    text: bridge.contextNote
                    wrapMode: Text.WordWrap
                    font.pixelSize: 12
                    color: Theme.warn
                    Layout.fillWidth: true
                }
            }
        }

        Button {
            text: "Reset to defaults"
            Layout.leftMargin: Theme.gap
            Layout.bottomMargin: Theme.gap
            onClicked: root.settings.resetToDefaults()
        }
    }
}
