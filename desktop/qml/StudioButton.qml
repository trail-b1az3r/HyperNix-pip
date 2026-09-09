import QtQuick
import QtQuick.Controls

// StudioButton.qml — one button, so there is one button.
//
// Qt's default Button is a grey rounded rectangle from another decade,
// and restyling it at each call site is how an app ends up with nine
// slightly different buttons.
Button {
    id: control
    // The accent is the mark's red and it is the only saturated colour
    // in the app, so exactly one button per screen should be primary.
    // More than one and none of them reads as the thing to do.
    property bool primary: false
    property bool danger: false

    implicitHeight: 34
    implicitWidth: Math.max(96, contentItem.implicitWidth + 28)
    font.family: Theme.fontFamily
    font.pixelSize: Theme.fontBody
    font.weight: primary ? Font.DemiBold : Font.Normal

    contentItem: Text {
        text: control.text
        font: control.font
        color: control.primary || control.danger ? "#ffffff" : Theme.text
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        opacity: control.enabled ? 1.0 : 0.4
    }

    background: Rectangle {
        radius: Theme.radiusSmall
        color: {
            if (control.danger)
                return control.hovered ? Qt.lighter(Theme.accent, 1.2) : Theme.accent
            if (control.primary)
                return control.hovered ? Theme.accentHi : Theme.accent
            return control.hovered ? Theme.hover : Theme.surfaceAlt
        }
        border.width: control.primary || control.danger ? 0 : 1
        border.color: Theme.border
        opacity: control.enabled ? 1.0 : 0.4

        Behavior on color { ColorAnimation { duration: Theme.anim } }
    }
}
