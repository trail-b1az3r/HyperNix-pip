import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// ConnectPanel.qml — an address and a T2S key.
//
// A T2S key and not an admin one, said in the UI rather than only in the
// code: Studio switches models, chats and edits files in a folder you
// chose, and none of that is administration. A tool that does not need
// admin should not be asking for an admin credential, and a person
// pasting one into a desktop app deserves to be told it is the wrong
// kind.
ColumnLayout {
    id: root
    spacing: 6

    StudioField {
        id: address
        Layout.fillWidth: true
        placeholderText: "desktop.tailnet.ts.net:8000"
        mono: true
    }

    StudioField {
        id: key
        Layout.fillWidth: true
        placeholderText: "T2S_…"
        mono: true
        // Never autocapitalised, never autocorrected: a T2S key is
        // case-sensitive and full of punctuation, and "helpfully"
        // changing either turns a correct key into a wrong one.
        echoMode: TextInput.Password
        onAccepted: connectNow.clicked()
    }

    Label {
        text: "A read/write key, not an admin one:\ngkey create -v v2short --scopes read,write"
        color: Theme.textFaint
        font.family: Theme.monoFamily
        font.pixelSize: 10
        Layout.fillWidth: true
        wrapMode: Text.WordWrap
    }

    StudioButton {
        id: connectNow
        text: "Connect"
        primary: true
        Layout.fillWidth: true
        enabled: address.text.length > 0 && key.text.length > 0
        onClicked: {
            studio.connectTo(address.text, key.text)
            // Cleared from the field the moment it is handed over. It
            // lives in the client for the session and nowhere a
            // screenshot or a crash dump would find it.
            key.text = ""
        }
    }
}
