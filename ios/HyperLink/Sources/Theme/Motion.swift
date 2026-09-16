//  Motion.swift
//  How things move, in one place.
//
//  The app animated almost nothing: bubbles appeared, lists snapped, and
//  the only transition was the pairing screen. The result reads as
//  cheap, and worse, it hides information — a message that simply *is*
//  there gives no sense of whether it arrived now or was always there.
//
//  Three rules, because animation is easy to get embarrassing.
//
//  **One vocabulary.** Every duration and curve is named here. Views
//  that each pick their own spring produce an app where two things doing
//  the same thing move differently, which reads as a bug nobody can
//  point at.
//
//  **Springs, not ease curves.** A spring interrupted mid-flight
//  continues from where it is; an `easeInOut` restarts. Chat is all
//  interruptions — a message lands while the list is still settling from
//  the last one — so the ease curve is the wrong tool almost everywhere
//  here.
//
//  **Reduce Motion is not a suggestion.** Somebody who turns it on gets
//  vestibular symptoms from parallax and scale. Every helper here
//  collapses to a cross-fade, which is what that setting means: not "no
//  feedback", but "no movement".

import SwiftUI

enum Motion {
    /// The default for anything appearing, leaving or rearranging.
    ///
    /// Slightly under-damped, so it settles with the smallest possible
    /// overshoot — enough to read as physical, not enough to bounce.
    static let standard = Animation.spring(response: 0.34, dampingFraction: 0.82)

    /// For things that must feel instant: a button's own state, a toggle.
    /// Past about 200ms a control stops feeling connected to the finger.
    static let snappy = Animation.spring(response: 0.22, dampingFraction: 0.9)

    /// For larger changes — a screen swapping, a sheet's content
    /// replacing itself. Long enough to be followed by the eye.
    static let deliberate = Animation.spring(response: 0.45, dampingFraction: 0.85)

    /// A token arriving. Nearly instant on purpose: at several per
    /// second, anything slower queues up behind itself and the text
    /// visibly lags the model.
    static let token = Animation.easeOut(duration: 0.12)

    /// The animation to use, honouring Reduce Motion.
    ///
    /// Returns a plain fade rather than `nil` when motion is reduced:
    /// the change still needs to be *noticed*, it just must not move.
    static func respecting(_ reduced: Bool, _ animation: Animation) -> Animation {
        reduced ? .easeInOut(duration: 0.2) : animation
    }
}

/// How a message bubble arrives.
///
/// Up and in, because that is the direction a new message comes from in
/// a bottom-anchored list — a bubble that faded in place would read as
/// having always been there, which is precisely the information a chat
/// needs to convey.
struct MessageArrival: ViewModifier {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    func body(content: Content) -> some View {
        content.transition(
            reduceMotion
                ? .opacity
                : .asymmetric(
                    insertion: .move(edge: .bottom)
                        .combined(with: .opacity)
                        .combined(with: .scale(scale: 0.97, anchor: .bottom)),
                    removal: .opacity
                )
        )
    }
}

/// A row settling into a list.
struct RowArrival: ViewModifier {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    func body(content: Content) -> some View {
        content.transition(
            reduceMotion ? .opacity : .opacity.combined(with: .move(edge: .top))
        )
    }
}

/// The caret while tokens arrive.
///
/// A slow opacity pulse rather than a spinner: a spinner says "waiting",
/// and this is the opposite of waiting — text is arriving, and the caret
/// is where the next character will be.
struct ThinkingCaret: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var dim = false

    var body: some View {
        Text("▌")
            .font(.caption)
            .foregroundStyle(.secondary)
            .opacity(dim ? 0.25 : 1)
            .animation(
                reduceMotion
                    ? nil
                    : .easeInOut(duration: 0.65).repeatForever(autoreverses: true),
                value: dim
            )
            .onAppear { dim = true }
            .accessibilityHidden(true)
    }
}

extension View {
    /// A message bubble arriving in the thread.
    func messageArrival() -> some View { modifier(MessageArrival()) }

    /// A row arriving in a list.
    func rowArrival() -> some View { modifier(RowArrival()) }
}
