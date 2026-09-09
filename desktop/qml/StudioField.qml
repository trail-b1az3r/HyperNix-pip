import QtQuick
import QtQuick.Controls

// StudioField.qml — a text field that matches the rest of the app.
TextField {
    id: control
    property bool mono: false

    implicitHeight: 34
    color: Theme.text
    placeholderTextColor: Theme.textFaint
    font.family: mono ? Theme.monoFamily : Theme.fontFamily
    font.pixelSize: Theme.fontBody
    leftPadding: 10
    rightPadding: 10
    selectByMouse: true

    background: Rectangle {
        radius: Theme.radiusSmall
        color: Theme.surfaceAlt
        // The focused border is the accent, which is the only place in
        // the app where a colour marks "you are here" rather than
        // "something needs attention".
        border.width: 1
        border.color: control.activeFocus ? Theme.accent : Theme.border
        Behavior on border.color { ColorAnimation { duration: Theme.anim } }
    }
}
