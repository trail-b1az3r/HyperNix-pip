import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// ErrorBanner.qml — a failure, with what to do about it.
//
// Shows the server's own message, which is the point: every T1 failure
// carries a stable code and a remedy, and surfacing them is what lets
// this say "your machine has no model loaded" instead of "something
// went wrong". It does not auto-dismiss — a banner that vanishes is a
// banner nobody read.
Rectangle {
    id: banner
    property string what: ""
    property string detail: ""

    function show(newWhat, newDetail) {
        what = newWhat
        detail = newDetail
        visible = true
    }

    visible: false
    implicitHeight: visible ? content.implicitHeight + Theme.gap * 2 : 0
    color: Qt.rgba(0.78, 0.10, 0.18, 0.14)

    Rectangle {
        anchors.left: parent.left
        width: 3
        height: parent.height
        color: Theme.accent
    }

    RowLayout {
        id: content
        anchors.fill: parent
        anchors.margins: Theme.gap
        anchors.leftMargin: Theme.gap + 3
        spacing: Theme.gap

        ColumnLayout {
            Layout.fillWidth: true
            spacing: 2
            Label {
                text: banner.what
                color: Theme.text
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontBody
                font.weight: Font.DemiBold
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
            }
            Label {
                text: banner.detail
                visible: banner.detail.length > 0
                color: Theme.textDim
                font.family: Theme.fontFamily
                font.pixelSize: Theme.fontSmall
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
            }
        }

        StudioButton {
            text: "Dismiss"
            implicitWidth: 88
            onClicked: banner.visible = false
        }
    }
}
