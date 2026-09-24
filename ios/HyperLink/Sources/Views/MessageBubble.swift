//  MessageBubble.swift
//  One message, with fenced code blocks rendered as code.
//
//  A model answering a programming question puts most of its useful
//  output inside ``` fences. Rendering that as body text — proportional,
//  wrapped, with no way to copy just the code — is the difference
//  between an app you can work in and one you read on.
//
//  Everything *between* the fences is markdown, and is rendered as such
//  by `MarkdownText`. It used to be plain `Text`, so a numbered list
//  arrived as one wrapped paragraph with the numbers buried in it.

import SwiftUI
import UIKit

struct MessageBubble: View {
    let message: ChatMessage
    var isStreaming: Bool = false
    /// Drawn on the last bubble of a run from one speaker, as Messages
    /// does: a run of three replies reads as one turn, not three.
    var showsTail: Bool = true

    @Environment(\.hyperLinkTheme) private var theme

    var body: some View {
        VStack(alignment: message.isUser ? .trailing : .leading, spacing: 4) {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(Array(MessageSegment.parse(message.content).enumerated()), id: \.offset) { _, segment in
                    switch segment {
                    case let .text(body):
                        // Markdown, not plain text: headings, lists and
                        // **bold** are how every instruction-tuned model
                        // writes, and rendering them literally left the
                        // structure the model produced as the hardest
                        // part of the reply to read. See `Markdown.swift`.
                        MarkdownText(source: body)
                    case let .code(language, body):
                        CodeBlockView(language: language, code: body)
                    }
                }
                if !message.attachmentIDs.isEmpty {
                    Label(
                        "\(message.attachmentIDs.count) attachment\(message.attachmentIDs.count == 1 ? "" : "s")",
                        systemImage: "paperclip"
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
                if isStreaming {
                    // A caret while the tokens arrive: cheaper to read
                    // than a spinner, and it disappears the moment the
                    // real message replaces this bubble. It pulses
                    // rather than spins, because a spinner says
                    // "waiting" and this is the opposite — text is
                    // arriving, and the caret is where the next
                    // character will be.
                    ThinkingCaret()
                }
            }
            .padding(12)
            // The theme's two bubble colours at full strength, rather
            // than one accent at 18% and the system grey at 12%. Those
            // two were close enough in several appearances that the only
            // thing separating the speakers was which side they were on.
            .background(
                message.isUser ? theme.userBubble : theme.assistantBubble,
                in: BubbleShape(isUser: message.isUser, showsTail: showsTail)
            )
            // Room for the tail, which is drawn outside the bubble's own
            // rectangle and would otherwise be clipped by the screen edge.
            .padding(message.isUser ? .trailing : .leading, 5)
            .foregroundStyle(
                message.isUser ? theme.userBubbleText : theme.assistantBubbleText
            )
            .frame(maxWidth: .infinity, alignment: message.isUser ? .trailing : .leading)

            if message.isAssistant && !message.modelID.isEmpty && !isStreaming {
                Text(shortModelName(message.modelID))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    // In line with the bubble, which sits 5pt in to leave
                    // room for its tail, rather than under the tail.
                    .padding(.leading, 5)
            }
        }
        .frame(maxWidth: .infinity, alignment: message.isUser ? .trailing : .leading)
    }
}

/// A rounded bubble with the tail Messages draws on the speaker's side.
///
/// The tail is part of the shape, not an overlay, so it takes the same
/// fill, the same theme colour and the same accessibility contrast as
/// the bubble it belongs to.
///
/// Joined with `union`, not `addPath`. The tail is drawn for the right
/// and mirrored for the left, and mirroring reverses the direction it
/// winds. With `addPath`, the assistant's tail then wound against the
/// bubble, and under SwiftUI's non-zero fill the part where they
/// overlapped cancelled out: the assistant's bubble had a dark triangle
/// punched through its bottom corner with a sliver of tail around it,
/// while the user's, winding the same way as its bubble, looked right.
/// A union fills the outline of both, whichever way either winds.
struct BubbleShape: Shape {
    var isUser: Bool
    var showsTail: Bool
    var radius: CGFloat = 17

    func path(in rect: CGRect) -> Path {
        let path = Path(roundedRect: rect, cornerRadius: radius, style: .continuous)
        guard showsTail, rect.height > radius else { return path }
        // Drawn for the right-hand side and mirrored for the left, so the
        // two speakers' tails are the same shape.
        let bottom = rect.maxY
        let edge = isUser ? rect.maxX : rect.minX
        let out: CGFloat = isUser ? 1 : -1
        var tail = Path()
        tail.move(to: CGPoint(x: edge - out * 10, y: bottom - radius))
        tail.addQuadCurve(
            to: CGPoint(x: edge + out * 5, y: bottom),
            control: CGPoint(x: edge - out * 1, y: bottom - 3)
        )
        tail.addQuadCurve(
            to: CGPoint(x: edge - out * radius, y: bottom - 1),
            control: CGPoint(x: edge - out * 6, y: bottom + 1)
        )
        tail.closeSubpath()
        return path.union(tail)
    }
}

/// Where the server summarised older messages to keep the thread inside
/// the model's context. The originals are still above it; this says the
/// model is now sent the summary instead, and shows it on request.
struct CompactionMarker: View {
    let message: ChatMessage
    @State private var expanded = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        VStack(spacing: 6) {
            Button {
                withAnimation(Motion.respecting(reduceMotion, Motion.snappy)) { expanded.toggle() }
            } label: {
                HStack(spacing: 8) {
                    Rectangle().frame(height: 1).foregroundStyle(.quaternary)
                    Label("Earlier messages summarised", systemImage: "rectangle.compress.vertical")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize()
                    Rectangle().frame(height: 1).foregroundStyle(.quaternary)
                }
            }
            .buttonStyle(.plain)
            .accessibilityHint(expanded ? "Hides the summary" : "Shows what the model is now sent")
            if expanded {
                Text(message.content)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
                    .background(Color.secondary.opacity(0.12), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
            }
        }
        .padding(.vertical, 4)
    }
}

/// A message split into prose and fenced code.
enum MessageSegment {
    case text(String)
    case code(language: String, body: String)

    /// Split on ``` fences.
    ///
    /// An unterminated fence — which is every partially-streamed code
    /// block — is treated as a code block running to the end of what has
    /// arrived. Waiting for the closing fence would make a streamed
    /// answer flip from prose to code when it completes, which looks
    /// like a rendering bug.
    static func parse(_ content: String) -> [MessageSegment] {
        guard content.contains("```") else {
            return content.isEmpty ? [] : [.text(content)]
        }
        var segments: [MessageSegment] = []
        var inCode = false
        var language = ""
        var buffer: [String] = []

        func flush() {
            let body = buffer.joined(separator: "\n")
            buffer.removeAll()
            if inCode {
                segments.append(.code(language: language, body: body))
            } else {
                let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { segments.append(.text(trimmed)) }
            }
        }

        for line in content.components(separatedBy: .newlines) {
            if line.hasPrefix("```") {
                flush()
                if !inCode {
                    language = String(line.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                } else {
                    language = ""
                }
                inCode.toggle()
                continue
            }
            buffer.append(line)
        }
        flush()
        return segments
    }
}

struct CodeBlockView: View {
    let language: String
    let code: String
    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(language.isEmpty ? "code" : language)
                    .font(.caption2.weight(.medium))
                    .foregroundStyle(.secondary)
                Spacer()
                Button {
                    UIPasteboard.general.string = code
                    copied = true
                    Task {
                        try? await Task.sleep(for: .seconds(1.5))
                        copied = false
                    }
                } label: {
                    Label(copied ? "Copied" : "Copy", systemImage: copied ? "checkmark" : "doc.on.doc")
                        .font(.caption2)
                }
                .buttonStyle(.borderless)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)

            Divider()

            // Code must not be re-wrapped: a wrapped line changes what
            // the code appears to say. Horizontal scrolling is the only
            // honest way to show a long line on a phone.
            ScrollView(.horizontal, showsIndicators: true) {
                Text(code)
                    .font(.system(.footnote, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .background(Color.black.opacity(0.06), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(Color.secondary.opacity(0.25))
        )
    }
}
