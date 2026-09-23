//  SwipeActions.swift
//  Swipe left on a message to edit or resend it.
//
//  Why not `.swipeActions`
//  -----------------------
//  SwiftUI's `.swipeActions` only works on the rows of a `List`. The chat
//  is a `ScrollView` of a `LazyVStack` — for the scroll-to-bottom and the
//  streaming bubble — and on anything that is not a `List` row the
//  modifier compiles and does nothing at all. So this is a drag gesture.
//
//  Scrolling still has to work
//  ---------------------------
//  A drag gesture on a view inside a ScrollView takes the touch from it
//  unless it is careful. This one is *simultaneous*, and only moves once
//  the drag is clearly horizontal and to the left — a vertical scroll
//  that wanders a few points sideways stays a scroll.
//
//  And the actions have to be reachable without the gesture
//  --------------------------------------------------------
//  VoiceOver users cannot swipe a bubble. Every action here is also an
//  accessibility action, so the same Edit and Resend are one rotor
//  gesture away.

import SwiftUI

struct SwipeAction: Identifiable {
    let id = UUID()
    let title: String
    let systemImage: String
    let tint: Color
    let perform: () -> Void
}

private struct SwipeToReveal: ViewModifier {
    let rowID: String
    let actions: [SwipeAction]
    @Binding var openRowID: String?

    @State private var offset: CGFloat = 0
    @State private var tracking = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var settle: Animation { Motion.respecting(reduceMotion, Motion.snappy) }

    private let buttonWidth: CGFloat = 76
    private var revealWidth: CGFloat { buttonWidth * CGFloat(actions.count) }

    func body(content: Content) -> some View {
        ZStack(alignment: .trailing) {
            if offset < 0 {
                HStack(spacing: 0) {
                    ForEach(actions) { action in
                        Button {
                            close()
                            action.perform()
                        } label: {
                            VStack(spacing: 4) {
                                Image(systemName: action.systemImage)
                                Text(action.title).font(.caption2)
                            }
                            .frame(width: buttonWidth)
                            .frame(maxHeight: .infinity)
                            .foregroundStyle(.white)
                            .background(action.tint)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .transition(.opacity)
            }
            content
                .offset(x: offset)
        }
        .contentShape(Rectangle())
        .simultaneousGesture(
            DragGesture(minimumDistance: 18, coordinateSpace: .local)
                .onChanged { value in
                    let dx = value.translation.width
                    let dy = value.translation.height
                    // Horizontal and leftward, by a clear margin — or
                    // closing one that is already open.
                    guard tracking || (abs(dx) > abs(dy) * 1.6) else { return }
                    tracking = true
                    let base: CGFloat = openRowID == rowID ? -revealWidth : 0
                    // A little rubber band past the buttons, never to the right.
                    offset = min(0, max(-revealWidth * 1.25, base + dx))
                }
                .onEnded { _ in
                    guard tracking else { return }
                    tracking = false
                    withAnimation(settle) {
                        if offset < -revealWidth / 2 {
                            offset = -revealWidth
                            openRowID = rowID
                        } else {
                            offset = 0
                            if openRowID == rowID { openRowID = nil }
                        }
                    }
                }
        )
        .onChange(of: openRowID) { _, current in
            // Another message was opened: this one closes. One open at a
            // time, as every list on the platform behaves.
            if current != rowID, offset != 0 {
                withAnimation(settle) { offset = 0 }
            }
        }
        .accessibilityActions {
            ForEach(actions) { action in
                Button(action.title, action: action.perform)
            }
        }
    }

    private func close() {
        withAnimation(settle) { offset = 0 }
        if openRowID == rowID { openRowID = nil }
    }
}

extension View {
    /// Swipe left to reveal *actions*. Nothing happens with none.
    @ViewBuilder
    func swipeToReveal(id: String, actions: [SwipeAction], openRowID: Binding<String?>) -> some View {
        if actions.isEmpty {
            self
        } else {
            modifier(SwipeToReveal(rowID: id, actions: actions, openRowID: openRowID))
        }
    }
}
