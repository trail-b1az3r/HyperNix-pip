//  ThemePickerView.swift
//  Choosing a theme, by looking at it.
//
//  A list of names would be a list of words: nobody knows what "Plum"
//  looks like until they pick it, and picking to find out means leaving
//  the screen and coming back. So each row is a two-bubble preview in
//  that theme's own colours — the same two bubbles the transcript uses,
//  because that is where a theme is actually seen.

import SwiftUI

struct ThemePickerView: View {
    @Environment(ThemeStore.self) private var store
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        List {
            Section {
                ForEach(HyperLinkTheme.all) { theme in
                    Button {
                        store.select(theme)
                    } label: {
                        ThemeRow(theme: theme, selected: theme.id == store.theme.id)
                    }
                    .buttonStyle(.plain)
                }
            } footer: {
                Text(
                    "Every theme keeps its text readable against its own "
                    + "bubbles. CarPlay is not themed — the car decides its "
                    + "own colours, being the one judging what is legible at "
                    + "speed."
                )
            }
        }
        .navigationTitle("Theme")
        .navigationBarTitleDisplayMode(.inline)
    }
}

private struct ThemeRow: View {
    let theme: HyperLinkTheme
    let selected: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(theme.name)
                    .font(.body.weight(selected ? .semibold : .regular))
                Spacer()
                if selected {
                    Image(systemName: "checkmark")
                        .foregroundStyle(theme.accent)
                        .accessibilityLabel("Selected")
                }
            }

            // The preview. Two bubbles, the way they appear in a
            // transcript, so what is on this row is what will be on that
            // screen.
            HStack {
                Text("How do I…")
                    .font(.caption)
                    .padding(.horizontal, 9)
                    .padding(.vertical, 5)
                    .background(theme.assistantBubble, in: .rect(cornerRadius: 11))
                    .foregroundStyle(theme.assistantBubbleText)
                Spacer(minLength: 24)
                Text("Like this.")
                    .font(.caption)
                    .padding(.horizontal, 9)
                    .padding(.vertical, 5)
                    .background(theme.userBubble, in: .rect(cornerRadius: 11))
                    .foregroundStyle(theme.userBubbleText)
            }

            Text(theme.note)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
        .contentShape(.rect)
        // One announcement rather than five fragments, and it says what
        // the preview shows to somebody who cannot see it.
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(theme.name). \(theme.note)")
        .accessibilityAddTraits(selected ? [.isButton, .isSelected] : .isButton)
    }
}
