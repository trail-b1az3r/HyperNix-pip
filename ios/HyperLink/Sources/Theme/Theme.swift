//  Theme.swift
//  HyperLink — colour, everywhere.
//
//  Eight themes, one definition, read by the phone app and by CarPlay.
//
//  Why a definition rather than a tint
//  -----------------------------------
//  `.tint(.accentColor)` was the whole of HyperLink's colour, which gets
//  you a blue button and a blue button. A theme here is five colours and
//  a name, because the three places colour actually carries meaning in
//  this app each need a different one: the person's bubble, the model's
//  bubble, and the connection indicator. A single accent cannot say all
//  three.
//
//  The bubbles are the reason this is not cosmetic. A transcript is two
//  speakers alternating, and the only thing separating them is which
//  side and which colour — so a theme whose two bubble colours are close
//  together is a theme that makes the app harder to read. Every theme
//  below is checked in `tests/test_hyperlink_ios_wiring.py` — the text on
//  each bubble against that bubble, at WCAG AA, and the two bubbles
//  against each other.
//
//  Not CarPlay
//  -----------
//  None of this reaches the car. CarPlay templates have no tint and no
//  colour an app can set: the system owns the whole appearance, and it
//  should, because it is the one deciding what is legible at speed. A
//  `carPlayTint` here was written before that was checked, and has been
//  removed rather than left as a property nothing reads.

import SwiftUI

/// One colour scheme.
struct HyperLinkTheme: Identifiable, Hashable, Sendable {
    let id: String
    let name: String
    /// The one-line description shown under the name in the picker.
    let note: String

    /// Buttons, links, the selected tab.
    let accent: Color
    /// The person's own messages.
    let userBubble: Color
    /// The model's messages.
    let assistantBubble: Color
    /// Connected, streaming, healthy.
    let good: Color
    /// Disconnected, failed, stopped.
    let bad: Color

    /// Text on top of `userBubble`. Computed rather than stored because
    /// getting it wrong is the difference between a readable bubble and
    /// an unreadable one, and a stored field is a field somebody can
    /// forget to set when they add a theme.
    var userBubbleText: Color { Self.readableText(on: userBubble) }
    var assistantBubbleText: Color { Self.readableText(on: assistantBubble) }

    /// Black or white, whichever the eye can read on `background`.
    ///
    /// Measured rather than thresholded. The first version compared
    /// luminance against 0.6, on the reasoning that a mid-tone reads
    /// darker than its luminance suggests and so dark text should start
    /// above the midpoint. That is backwards: the crossover where black
    /// and white contrast equally is at a luminance of 0.179, well
    /// *below* the midpoint, so a 0.6 threshold put white text on every
    /// mid-tone. It did it to this app's own default — white on HyperNix
    /// green came out at 3.37:1, under the 4.5:1 WCAG asks for, on the
    /// most-seen colour pair in the app.
    ///
    /// Comparing the two ratios needs no constant and cannot drift when
    /// somebody adds a theme.
    static func readableText(on background: Color) -> Color {
        background.contrastRatio(against: .black)
            >= background.contrastRatio(against: .white) ? .black : .white
    }
}

extension Color {
    /// WCAG relative luminance, for the text-colour decision above.
    var relativeLuminance: Double {
        #if canImport(UIKit)
        var red: CGFloat = 0, green: CGFloat = 0, blue: CGFloat = 0, alpha: CGFloat = 0
        UIColor(self).getRed(&red, green: &green, blue: &blue, alpha: &alpha)
        func channel(_ value: CGFloat) -> Double {
            let v = Double(value)
            return v <= 0.03928 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4)
        }
        return 0.2126 * channel(red) + 0.7152 * channel(green) + 0.0722 * channel(blue)
        #else
        return 0.5
        #endif
    }

    /// Contrast ratio against another colour, 1...21.
    func contrastRatio(against other: Color) -> Double {
        let a = relativeLuminance + 0.05
        let b = other.relativeLuminance + 0.05
        return max(a, b) / min(a, b)
    }

    /// `#rrggbb`, because a theme reads better written the way it would
    /// be written anywhere else.
    init(hex: UInt32) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xff) / 255,
            green: Double((hex >> 8) & 0xff) / 255,
            blue: Double(hex & 0xff) / 255,
            opacity: 1
        )
    }
}

extension HyperLinkTheme {
    /// The themes that ship.
    ///
    /// Eight, and deliberately not eighty: every one has to stay readable
    /// in both appearances and in a car, and a list nobody can check is a
    /// list where one of them is unreadable.
    static let all: [HyperLinkTheme] = [
        HyperLinkTheme(
            id: "hypernix",
            name: "HyperNix",
            note: "The default. Green on slate, the same palette as the dashboards.",
            accent: Color(hex: 0x2F9E6B),
            userBubble: Color(hex: 0x2F9E6B),
            assistantBubble: Color(hex: 0x39414E),
            good: Color(hex: 0x4ADE80),
            bad: Color(hex: 0xE05555)
        ),
        HyperLinkTheme(
            id: "midnight",
            name: "Midnight",
            note: "Deep blue. Easiest of these on a phone at night.",
            accent: Color(hex: 0x4C7DF0),
            userBubble: Color(hex: 0x2F4FA8),
            assistantBubble: Color(hex: 0x2A3040),
            good: Color(hex: 0x62C6F5),
            bad: Color(hex: 0xF07171)
        ),
        HyperLinkTheme(
            id: "ember",
            name: "Ember",
            note: "Warm orange. The highest-contrast one here, and the best in sunlight.",
            accent: Color(hex: 0xE2622E),
            userBubble: Color(hex: 0xC94F1E),
            assistantBubble: Color(hex: 0x3A332E),
            good: Color(hex: 0xF0A94B),
            bad: Color(hex: 0xD93A3A)
        ),
        HyperLinkTheme(
            id: "terminal",
            name: "Terminal",
            note: "Phosphor green on black, for people who were always going to pick this one.",
            accent: Color(hex: 0x33D17A),
            userBubble: Color(hex: 0x1F7A48),
            assistantBubble: Color(hex: 0x1A1D1A),
            good: Color(hex: 0x33D17A),
            bad: Color(hex: 0xE0524F)
        ),
        HyperLinkTheme(
            id: "paper",
            name: "Paper",
            note: "Light, low-saturation. The only one that stays light in dark mode.",
            accent: Color(hex: 0x4A6572),
            // Deeper than a light theme looks like it wants. At 0xD6E2E8
            // this sat 1.13:1 from the assistant bubble beside it, which
            // in a transcript is two bubbles of the same colour with only
            // the side of the screen telling them apart.
            userBubble: Color(hex: 0xA8C6D6),
            assistantBubble: Color(hex: 0xF0EDE6),
            good: Color(hex: 0x3F7D5A),
            // Darker than the red in the other themes. On a light
            // background both status colours are dark, and at 0xA83A2E
            // this one sat 1.30:1 from the green beside it — red and
            // green telling each other apart by hue alone, which for a
            // red-green colourblind reader is not telling them apart.
            bad: Color(hex: 0x8C2118)
        ),
        HyperLinkTheme(
            id: "plum",
            name: "Plum",
            note: "Purple and grey. Quiet.",
            accent: Color(hex: 0x8A63C4),
            userBubble: Color(hex: 0x6B4A9E),
            assistantBubble: Color(hex: 0x36313E),
            good: Color(hex: 0xB49AE0),
            // 1.74:1 from the green rather than 1.34:1. Same reason as
            // Paper's.
            bad: Color(hex: 0xC7566A)
        ),
        HyperLinkTheme(
            id: "sea",
            name: "Sea",
            note: "Teal. Middle of the road in the best sense.",
            accent: Color(hex: 0x1E8C8C),
            userBubble: Color(hex: 0x17706F),
            assistantBubble: Color(hex: 0x2C3838),
            good: Color(hex: 0x4FC3C3),
            bad: Color(hex: 0xE0725F)
        ),
        HyperLinkTheme(
            id: "mono",
            name: "Mono",
            note: "No colour at all. For a car dashboard, or for not liking colour.",
            accent: Color(hex: 0x7A7A7A),
            userBubble: Color(hex: 0x4A4A4A),
            assistantBubble: Color(hex: 0x2B2B2B),
            good: Color(hex: 0xBFBFBF),
            bad: Color(hex: 0x8A8A8A)
        ),
    ]

    static let fallback = all[0]

    static func named(_ id: String) -> HyperLinkTheme {
        all.first { $0.id == id } ?? fallback
    }
}

/// Which theme is on, remembered across launches.
///
/// `UserDefaults` rather than the Keychain: a colour preference is not a
/// secret, and putting it in the Keychain would mean it survived an
/// uninstall, which is the wrong behaviour for a setting.
@Observable
final class ThemeStore {
    private static let key = "hyperlink.theme"

    private(set) var theme: HyperLinkTheme

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        let stored = defaults.string(forKey: Self.key) ?? HyperLinkTheme.fallback.id
        self.theme = HyperLinkTheme.named(stored)
    }

    private let defaults: UserDefaults

    func select(_ theme: HyperLinkTheme) {
        self.theme = theme
        defaults.set(theme.id, forKey: Self.key)
    }

    func select(id: String) {
        select(HyperLinkTheme.named(id))
    }
}

// MARK: - Reading it from a view

private struct ThemeKey: EnvironmentKey {
    static let defaultValue = HyperLinkTheme.fallback
}

extension EnvironmentValues {
    var hyperLinkTheme: HyperLinkTheme {
        get { self[ThemeKey.self] }
        set { self[ThemeKey.self] = newValue }
    }
}

extension View {
    /// Apply a theme to this view and everything under it.
    func hyperLinkTheme(_ theme: HyperLinkTheme) -> some View {
        environment(\.hyperLinkTheme, theme).tint(theme.accent)
    }
}
