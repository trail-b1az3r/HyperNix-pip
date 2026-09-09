import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// Meter.qml — one measurement, or an honest dash.
//
// `used` or `total` below zero means the vendor tool declined to answer,
// and that renders as "—" with an empty track. Not a zero bar: a card
// that does not report utilisation would otherwise show 0% forever and
// read as an idle GPU.
ColumnLayout {
    id: root
    property string label: ""
    property real used: -1
    property real total: -1
    property string unit: ""

    readonly property bool known: used >= 0 && total > 0
    readonly property real fraction: known ? Math.min(1, used / total) : 0

    spacing: 3

    RowLayout {
        Layout.fillWidth: true
        Label {
            text: root.label
            color: Theme.textDim
            font.family: Theme.fontFamily
            font.pixelSize: Theme.fontSmall
        }
        Item { Layout.fillWidth: true }
        Label {
            text: root.known
                  ? (root.unit === "%"
                     ? root.used.toFixed(0) + "%"
                     : root.used.toFixed(0) + " / " + root.total.toFixed(0) +
                       " " + root.unit)
                  : "—"
            color: root.known ? Theme.text : Theme.textFaint
            font.family: Theme.monoFamily
            font.pixelSize: Theme.fontSmall
        }
    }

    Rectangle {
        Layout.fillWidth: true
        implicitHeight: 5
        radius: 2.5
        color: Theme.surfaceAlt

        Rectangle {
            width: parent.width * root.fraction
            height: parent.height
            radius: 2.5
            // The accent above 85%: the one place a colour change means
            // "this is about to be a problem" rather than decoration.
            color: root.fraction > 0.85 ? Theme.accent : Theme.textDim
            Behavior on width { NumberAnimation { duration: Theme.anim * 2 } }
            Behavior on color { ColorAnimation { duration: Theme.anim } }
        }
    }
}
