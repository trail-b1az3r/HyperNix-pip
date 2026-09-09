pragma Singleton
import QtQuick

// Theme.qml — one place for every colour, size and duration.
//
// A singleton rather than constants scattered through the views, because
// the alternative is what every Qt app that looks dated actually is: a
// dozen hard-coded #2b2b2b that drifted apart. Change a value here and
// the whole app moves.
//
// The palette is HyperNix's own, from assets/logo-new/README.md — the
// accent is the mark's red, #c8192e, and it is the only saturated colour
// in the app. Everything else is a step on a neutral ramp. That is what
// makes an interface read as one thing: one accent, used sparingly, and
// contrast doing the rest of the work.
QtObject {
    // --- surfaces, darkest to lightest ---
    readonly property color base:      "#0e0f11"   // the window
    readonly property color surface:   "#16181b"   // panels
    readonly property color surfaceAlt:"#1e2125"   // cards, inputs
    readonly property color hover:     "#262a2f"
    readonly property color border:    "#2e3338"

    // --- text ---
    readonly property color text:      "#f2f2f0"   // the mark's off-white
    readonly property color textDim:   "#9aa0a6"
    readonly property color textFaint: "#6b7178"

    // --- the accent, and the two states that earn a colour of their own ---
    readonly property color accent:    "#c8192e"
    readonly property color accentHi:  "#e8394d"
    readonly property color warn:      "#e0a030"
    readonly property color ok:        "#3fa46a"

    // --- geometry ---
    readonly property int radius:      10
    readonly property int radiusSmall: 6
    readonly property int gap:         12
    readonly property int gapLarge:    20
    readonly property int sidebar:     248
    readonly property int rowHeight:   38

    // --- type ---
    // A stack rather than one family: Inter is what the website uses and
    // is worth asking for, and the fallbacks are what actually exists on
    // a stock Linux install.
    readonly property string fontFamily: "Inter, Cantarell, Ubuntu, DejaVu Sans, sans-serif"
    readonly property string monoFamily: "JetBrains Mono, Fira Code, DejaVu Sans Mono, monospace"
    readonly property int fontSmall:   12
    readonly property int fontBody:    14
    readonly property int fontTitle:   18
    readonly property int fontDisplay: 26

    // --- motion ---
    // 140ms: long enough to be seen as movement, short enough that a
    // person clicking through the app never waits for it.
    readonly property int anim: 140
}
