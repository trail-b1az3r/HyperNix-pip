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
                in: RoundedRectangle(cornerRadius: 14, style: .continuous)
            )
            .foregroundStyle(
                message.isUser ? theme.userBubbleText : theme.assistantBubbleText
            )
            .frame(maxWidth: .infinity, alignment: message.isUser ? .trailing : .leading)

            if message.isAssistant && !message.modelID.isEmpty && !isStreaming {
                Text(shortModelName(message.modelID))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .frame(maxWidth: .infinity, alignment: message.isUser ? .trailing : .leading)
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
