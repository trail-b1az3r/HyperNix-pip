//  Markdown.swift
//  Rendering what the model actually writes.
//
//  Models write markdown. Not occasionally — it is the house style of
//  every instruction-tuned model there is, and asking one for anything
//  structured gets back headings, bullets and **bold**. The prose half
//  of a message was rendered with plain `Text`, so all of that arrived
//  as literal asterisks and hashes: a numbered list came out as one
//  wrapped paragraph with `1.` and `2.` buried in the middle of it, and
//  the structure the model had gone to the trouble of producing was the
//  hardest part to read.
//
//  Code fences are already handled a level up, in `MessageSegment` —
//  this is everything between them.
//
//  Two decisions worth writing down
//  --------------------------------
//  **Not `Text(LocalizedStringKey)`.** SwiftUI's `Text("**hi**")` does
//  parse a little markdown, but it does it by treating the string as a
//  *localization key*: model output goes through the app's string
//  catalogue, `%@` in a reply becomes a format specifier, and the
//  subset it handles has no headings and no lists. It is the wrong tool
//  wearing the right shape.
//
//  **Inline-only parsing, block structure by hand.** `AttributedString`
//  can parse full markdown, but `.full` collapses newlines and drops
//  the block structure into a single run, which is the opposite of what
//  is wanted here: the blocks are exactly what needs laying out as
//  separate views. So blocks are split first and each one's *inline*
//  markup — bold, italic, `code`, links — is parsed with
//  `.inlineOnlyPreservingWhitespace`.
//
//  Half-written markdown
//  ---------------------
//  Every one of these is parsed mid-stream, so `**bol` arrives before
//  `**bold**` does. Unterminated markup renders as the literal
//  characters, which is the right answer: text that flickers between
//  bold and not as the tokens land is worse than text that is briefly
//  plain, and text that *vanishes* while the model finishes a token
//  would look like a bug.

import SwiftUI

/// One block of a markdown document.
enum MarkdownBlock: Equatable {
    case heading(level: Int, text: String)
    case paragraph(String)
    /// `indent` is nesting depth, `marker` is the bullet or number to
    /// show. An ordered list keeps its own numbers rather than being
    /// renumbered, because a model writing `1.` three times in a row is
    /// making a point about steps being interchangeable often enough
    /// that silently fixing it is a change of meaning.
    case listItem(indent: Int, marker: String, text: String)
    case quote(String)
    case rule

    /// Split *source* into blocks.
    static func parse(_ source: String) -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []
        var paragraph: [String] = []

        func flushParagraph() {
            let joined = paragraph.joined(separator: "\n")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            paragraph.removeAll()
            if !joined.isEmpty { blocks.append(.paragraph(joined)) }
        }

        for rawLine in source.components(separatedBy: .newlines) {
            let line = rawLine
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            if trimmed.isEmpty {
                flushParagraph()
                continue
            }
            if isRule(trimmed) {
                flushParagraph()
                blocks.append(.rule)
                continue
            }
            if let heading = heading(trimmed) {
                flushParagraph()
                blocks.append(heading)
                continue
            }
            if let item = listItem(line) {
                flushParagraph()
                blocks.append(item)
                continue
            }
            if trimmed.hasPrefix(">") {
                flushParagraph()
                let body = String(trimmed.dropFirst())
                    .trimmingCharacters(in: .whitespaces)
                blocks.append(.quote(body))
                continue
            }
            paragraph.append(line)
        }
        flushParagraph()
        return blocks
    }

    /// `***`, `---` or `___`, three or more, and nothing else.
    ///
    /// The length test is what keeps `--` in the middle of a sentence
    /// and an em-dash-happy model from punching a horizontal rule
    /// through the reply.
    private static func isRule(_ line: String) -> Bool {
        guard line.count >= 3 else { return false }
        for character in "-*_" where line.allSatisfy({ $0 == character }) {
            return true
        }
        return false
    }

    private static func heading(_ line: String) -> MarkdownBlock? {
        var level = 0
        var index = line.startIndex
        while index < line.endIndex, line[index] == "#", level < 6 {
            level += 1
            index = line.index(after: index)
        }
        guard level > 0, index < line.endIndex, line[index] == " " else { return nil }
        let text = String(line[index...]).trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty else { return nil }
        return .heading(level: level, text: text)
    }

    private static func listItem(_ line: String) -> MarkdownBlock? {
        let leading = line.prefix { $0 == " " || $0 == "\t" }
        // Tabs count as four, which is what every editor that writes
        // them into markdown means by one.
        let indent = leading.reduce(0) { $0 + ($1 == "\t" ? 4 : 1) } / 2
        let rest = line.dropFirst(leading.count)

        for bullet in ["- ", "* ", "+ "] where rest.hasPrefix(bullet) {
            let text = String(rest.dropFirst(2)).trimmingCharacters(in: .whitespaces)
            guard !text.isEmpty else { return nil }
            return .listItem(indent: indent, marker: "•", text: text)
        }

        // `1.` / `1)` — the digits are kept rather than renumbered.
        let digits = rest.prefix { $0.isNumber }
        guard !digits.isEmpty, digits.count <= 3 else { return nil }
        let afterDigits = rest.dropFirst(digits.count)
        guard let separator = afterDigits.first, separator == "." || separator == ")",
              afterDigits.dropFirst().hasPrefix(" ")
        else { return nil }
        let text = String(afterDigits.dropFirst(2))
            .trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty else { return nil }
        return .listItem(indent: indent, marker: "\(digits)\(separator)", text: text)
    }

    /// Inline markup parsed, block markers already removed.
    ///
    /// Falls back to the literal string rather than throwing: a reply
    /// that cannot be parsed must still be readable, and half-written
    /// markup arrives on every single stream.
    static func inline(_ text: String) -> AttributedString {
        (try? AttributedString(
            markdown: text,
            options: .init(
                allowsExtendedAttributes: true,
                interpretedSyntax: .inlineOnlyPreservingWhitespace,
                failurePolicy: .returnPartiallyParsedIfPossible
            )
        )) ?? AttributedString(text)
    }
}

/// A block of model prose, laid out as markdown.
struct MarkdownText: View {
    let source: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(Array(MarkdownBlock.parse(source).enumerated()), id: \.offset) { _, block in
                view(for: block)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func view(for block: MarkdownBlock) -> some View {
        switch block {
        case let .heading(level, text):
            Text(MarkdownBlock.inline(text))
                .font(Self.headingFont(level))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                // A heading in a chat bubble is a *relative* size, not a
                // page title: `.largeTitle` inside a 280pt bubble is a
                // single word per line.
                .padding(.top, level <= 2 ? 4 : 2)

        case let .paragraph(text):
            Text(MarkdownBlock.inline(text))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)

        case let .listItem(indent, marker, text):
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text(marker)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                Text(MarkdownBlock.inline(text))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            // Capped: a model that indents eight levels deep should not
            // push its own text off the side of the bubble.
            .padding(.leading, CGFloat(min(indent, 4)) * 14)

        case let .quote(text):
            HStack(alignment: .top, spacing: 8) {
                Rectangle()
                    .frame(width: 3)
                    .foregroundStyle(.tertiary)
                Text(MarkdownBlock.inline(text))
                    .italic()
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .fixedSize(horizontal: false, vertical: true)

        case .rule:
            Divider().padding(.vertical, 2)
        }
    }

    private static func headingFont(_ level: Int) -> Font {
        switch level {
        case 1: return .title3.bold()
        case 2: return .headline
        case 3: return .subheadline.bold()
        default: return .body.bold()
        }
    }
}
