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

import Foundation
import SwiftUI

/// How a table column is aligned (GFM's `:--`, `:-:`, `--:`).
enum TableAlignment: Equatable {
    case none, leading, center, trailing
}

/// A footnote, numbered in the order it is first referred to.
struct Footnote: Equatable {
    let number: Int
    let label: String
    let text: String
}

/// One block of a markdown document.
///
/// 0.72.8a3: CommonMark, GitHub Flavored Markdown and Markdown Extra —
/// setext headings, `{#id}` attributes, task lists, tables, fenced and
/// indented code, definition lists, link references, footnotes and
/// abbreviations, and bare URLs as links. The web page renders the same
/// syntax (`hyperlink/web/markdown.js`).
enum MarkdownBlock: Equatable {
    case heading(level: Int, text: String)
    case paragraph(String)
    /// `indent` is nesting depth, `marker` is the bullet or number to
    /// show. An ordered list keeps its own numbers rather than being
    /// renumbered, because a model writing `1.` three times in a row is
    /// making a point about steps being interchangeable often enough
    /// that silently fixing it is a change of meaning.
    case listItem(indent: Int, marker: String, text: String)
    /// A GFM task: `- [ ] to do`, `- [x] done`.
    case task(indent: Int, checked: Bool, text: String)
    case quote(String)
    case rule
    /// Fenced or indented code. Fences are usually split off a level
    /// up, in `MessageSegment`; one that reaches here is still code.
    case code(language: String, text: String)
    case table(header: [String], alignments: [TableAlignment], rows: [[String]])
    /// Markdown Extra: a term, then `: ` lines that define it.
    case definition(term: String, details: [String])
    case footnotes([Footnote])

    /// Split *source* into blocks.
    static func parse(_ source: String) -> [MarkdownBlock] {
        var document = MarkdownDocument(source)
        return document.blocks()
    }

    /// `***`, `---` or `___`, three or more, and nothing else (spaces
    /// between allowed, as CommonMark allows `* * *`).
    ///
    /// The length test is what keeps `--` in the middle of a sentence
    /// and an em-dash-happy model from punching a horizontal rule
    /// through the reply.
    static func isRule(_ line: String) -> Bool {
        let compact = line.filter { $0 != " " && $0 != "\t" }
        guard compact.count >= 3, let first = compact.first, "-*_".contains(first) else { return false }
        return compact.allSatisfy { $0 == first }
    }

    static func heading(_ line: String) -> MarkdownBlock? {
        var level = 0
        var index = line.startIndex
        while index < line.endIndex, line[index] == "#", level < 6 {
            level += 1
            index = line.index(after: index)
        }
        guard level > 0, index < line.endIndex, line[index] == " " else { return nil }
        var text = String(line[index...]).trimmingCharacters(in: .whitespaces)
        // A closing run of #s is decoration, not text.
        while text.hasSuffix("#") { text.removeLast() }
        text = MarkdownDocument.withoutAttributes(text.trimmingCharacters(in: .whitespaces))
        guard !text.isEmpty else { return nil }
        return .heading(level: level, text: text)
    }

    static func listItem(_ line: String) -> MarkdownBlock? {
        let leading = line.prefix { $0 == " " || $0 == "\t" }
        // Tabs count as four, which is what every editor that writes
        // them into markdown means by one.
        let indent = leading.reduce(0) { $0 + ($1 == "\t" ? 4 : 1) } / 2
        let rest = line.dropFirst(leading.count)

        for bullet in ["- ", "* ", "+ "] where rest.hasPrefix(bullet) {
            let text = String(rest.dropFirst(2)).trimmingCharacters(in: .whitespaces)
            guard !text.isEmpty else { return nil }
            for (box, checked) in [("[ ] ", false), ("[x] ", true), ("[X] ", true)] where text.hasPrefix(box) {
                let task = String(text.dropFirst(4)).trimmingCharacters(in: .whitespaces)
                if !task.isEmpty { return .task(indent: indent, checked: checked, text: task) }
            }
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

/// One reply, read in two passes: the definitions (link references,
/// footnotes, abbreviations) first, then the blocks, with each block's
/// text resolved to the inline markdown `AttributedString` understands.
struct MarkdownDocument {
    private var lines: [String]
    private var references: [String: String] = [:]
    private var notes: [String: String] = [:]
    private var noteOrder: [String] = []

    init(_ source: String) {
        lines = source.replacingOccurrences(of: "\r\n", with: "\n").components(separatedBy: "\n")
        collect()
    }

    private static func label(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespaces).lowercased()
            .components(separatedBy: .whitespaces).filter { !$0.isEmpty }.joined(separator: " ")
    }

    private static func match(_ pattern: String, _ text: String) -> [String]? {
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let found = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text))
        else { return nil }
        return (0..<found.numberOfRanges).map { index in
            Range(found.range(at: index), in: text).map { String(text[$0]) } ?? ""
        }
    }

    /// `{#id .class}` at the end of a heading (Markdown Extra), removed.
    static func withoutAttributes(_ text: String) -> String {
        guard let found = match(#"^(.*?)\s*\{\s*(?:[#.][\w-]+\s*)+\}\s*$"#, text) else { return text }
        return found[1]
    }

    private mutating func collect() {
        var kept: [String] = []
        var fence: String?
        var index = 0
        while index < lines.count {
            let line = lines[index]
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if let open = fence {
                if trimmed.hasPrefix(open) { fence = nil }
                kept.append(line)
                index += 1
                continue
            }
            if trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~") {
                fence = String(trimmed.prefix(3))
                kept.append(line)
                index += 1
                continue
            }
            if let note = Self.match(#"^ {0,3}\[\^([^\]\s]+)\]:\s?(.*)$"#, line) {
                var body = [note[2]]
                while index + 1 < lines.count, lines[index + 1].hasPrefix("    ") {
                    index += 1
                    body.append(String(lines[index].dropFirst(4)))
                }
                notes[Self.label(note[1])] = body.joined(separator: "\n")
                index += 1
                continue
            }
            if let reference = Self.match(#"^ {0,3}\[([^\]^][^\]]*)\]:\s*<?(\S+?)>?(?:\s+["'(].*["')])?\s*$"#, line) {
                references[Self.label(reference[1])] = reference[2]
                index += 1
                continue
            }
            if Self.match(#"^ {0,3}\*\[[^\]]+\]:"#, line) != nil {      // an abbreviation: nothing to show
                index += 1
                continue
            }
            kept.append(line)
            index += 1
        }
        lines = kept
    }

    /// A block's text with references, footnote marks, bare URLs and the
    /// harmless inline HTML resolved; code spans left exactly as they are.
    mutating func resolve(_ text: String) -> String {
        var out = ""
        let parts = text.components(separatedBy: "`")
        for (index, part) in parts.enumerated() {
            if index > 0 { out += "`" }
            out += index % 2 == 1 && index < parts.count - 1 ? part : resolveProse(part)
        }
        return out
    }

    private mutating func resolveProse(_ text: String) -> String {
        var result = text
        // Footnote marks, numbered in the order they are first seen.
        while let found = Self.match(#"\[\^([^\]\s]+)\]"#, result) {
            let key = Self.label(found[1])
            guard notes[key] != nil else { break }
            if !noteOrder.contains(key) { noteOrder.append(key) }
            let number = (noteOrder.firstIndex(of: key) ?? 0) + 1
            result = result.replacingOccurrences(of: found[0], with: Self.superscript(number))
        }
        // [text][ref], [text][] and [ref], when the reference exists.
        if let regex = try? NSRegularExpression(pattern: #"\[([^\]\[]+)\](?:\[([^\]]*)\])?(?!\()"#) {
            let source = result
            var rebuilt = ""
            var last = source.startIndex
            for found in regex.matches(in: source, range: NSRange(source.startIndex..., in: source)) {
                guard let whole = Range(found.range, in: source),
                      let textRange = Range(found.range(at: 1), in: source) else { continue }
                let label = Range(found.range(at: 2), in: source).map { String(source[$0]) } ?? ""
                let key = Self.label(label.isEmpty ? String(source[textRange]) : label)
                guard let url = references[key] else { continue }
                rebuilt += source[last..<whole.lowerBound]
                rebuilt += "[\(source[textRange])](\(url))"
                last = whole.upperBound
            }
            rebuilt += source[last...]
            result = rebuilt
        }
        // Bare URLs become autolinks (GFM); not ones already in a link.
        if let regex = try? NSRegularExpression(
            pattern: #"(?<![<(\[/\w])(https?://|www\.)[^\s<>()\[\]]*[^\s<>()\[\].,:;!?'\x22*_~]"#) {
            let range = NSRange(result.startIndex..., in: result)
            result = regex.stringByReplacingMatches(in: result, range: range, withTemplate: "<$0>")
                .replacingOccurrences(of: "<www.", with: "<http://www.")
        }
        // <kbd>x</kbd> reads as code; the other harmless tags as their text.
        for (open, close, replacement) in [("<kbd>", "</kbd>", "`"), ("<code>", "</code>", "`")] {
            result = result.replacingOccurrences(of: open, with: replacement)
                .replacingOccurrences(of: close, with: replacement)
        }
        for tag in ["sup", "sub", "mark", "ins", "u", "small", "abbr", "q", "cite", "var", "samp"] {
            result = result.replacingOccurrences(of: "<\(tag)>", with: "")
                .replacingOccurrences(of: "</\(tag)>", with: "")
        }
        for tag in ["del", "s"] {
            result = result.replacingOccurrences(of: "<\(tag)>", with: "~~")
                .replacingOccurrences(of: "</\(tag)>", with: "~~")
        }
        return result.replacingOccurrences(of: "<br>", with: "\n").replacingOccurrences(of: "<br/>", with: "\n")
    }

    private static func superscript(_ number: Int) -> String {
        let digits: [Character: Character] = ["0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
                                              "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"]
        return String(String(number).compactMap { digits[$0] })
    }

    private static func cells(_ line: String) -> [String] {
        var text = line.trimmingCharacters(in: .whitespaces)
        if text.hasPrefix("|") { text.removeFirst() }
        if text.hasSuffix("|"), !text.hasSuffix("\\|") { text.removeLast() }
        var cells: [String] = []
        var current = ""
        var inCode = false
        var previous: Character = " "
        let backslash: Character = "\u{5C}"
        for character in text {
            if character == "`" { inCode.toggle() }
            if character == "|", !inCode, previous != backslash {
                cells.append(current.trimmingCharacters(in: .whitespaces))
                current = ""
            } else if character == "|", previous == backslash {
                current.removeLast()
                current.append("|")
            } else {
                current.append(character)
            }
            previous = character
        }
        cells.append(current.trimmingCharacters(in: .whitespaces))
        return cells
    }

    private static func isDelimiterRow(_ line: String) -> Bool {
        let row = cells(line)
        return !row.isEmpty && row.allSatisfy { cell in
            var core = cell
            if core.hasPrefix(":") { core.removeFirst() }
            if core.hasSuffix(":") { core.removeLast() }
            return !core.isEmpty && core.allSatisfy { $0 == "-" }
        }
    }

    private static func alignment(_ cell: String) -> TableAlignment {
        switch (cell.hasPrefix(":"), cell.hasSuffix(":")) {
        case (true, true): return .center
        case (true, false): return .leading
        case (false, true): return .trailing
        default: return .none
        }
    }

    private static func isFence(_ trimmed: String) -> Bool {
        trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~")
    }

    private static func definitionLine(_ line: String) -> String? {
        guard let found = match(#"^ {0,3}:\s+(.*)$"#, line) else { return nil }
        return found[1]
    }

    mutating func blocks() -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []
        var paragraph: [String] = []

        func flushParagraph(into blocks: inout [MarkdownBlock], _ paragraph: inout [String],
                            _ document: inout MarkdownDocument) {
            let joined = paragraph.joined(separator: "\n")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            paragraph.removeAll()
            if !joined.isEmpty { blocks.append(.paragraph(document.resolve(joined))) }
        }

        var index = 0
        while index < lines.count {
            let line = lines[index]
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            if trimmed.isEmpty {
                flushParagraph(into: &blocks, &paragraph, &self)
                index += 1
                continue
            }
            if Self.isFence(trimmed) {
                flushParagraph(into: &blocks, &paragraph, &self)
                let fence = String(trimmed.prefix(3))
                let language = String(trimmed.dropFirst(3)).trimmingCharacters(in: CharacterSet(charactersIn: " {}."))
                var body: [String] = []
                index += 1
                while index < lines.count,
                      !lines[index].trimmingCharacters(in: .whitespaces).hasPrefix(fence) {
                    body.append(lines[index])
                    index += 1
                }
                index += 1
                blocks.append(.code(language: language, text: body.joined(separator: "\n")))
                continue
            }
            if paragraph.isEmpty, line.hasPrefix("    "), MarkdownBlock.listItem(line) == nil {
                var body: [String] = []
                while index < lines.count,
                      lines[index].hasPrefix("    ") || lines[index].trimmingCharacters(in: .whitespaces).isEmpty {
                    body.append(String(lines[index].dropFirst(min(4, lines[index].count))))
                    index += 1
                }
                while body.last?.trimmingCharacters(in: .whitespaces).isEmpty == true { body.removeLast() }
                blocks.append(.code(language: "", text: body.joined(separator: "\n")))
                continue
            }
            // Setext: a one-line paragraph underlined with === or ---.
            if paragraph.count == 1, trimmed.count >= 3,
               trimmed.allSatisfy({ $0 == "=" }) || trimmed.allSatisfy({ $0 == "-" }) {
                let text = MarkdownDocument.withoutAttributes(paragraph[0].trimmingCharacters(in: .whitespaces))
                paragraph.removeAll()
                blocks.append(.heading(level: trimmed.first == "=" ? 1 : 2, text: resolve(text)))
                index += 1
                continue
            }
            if MarkdownBlock.isRule(trimmed) {
                flushParagraph(into: &blocks, &paragraph, &self)
                blocks.append(.rule)
                index += 1
                continue
            }
            if case let .heading(level, text)? = MarkdownBlock.heading(trimmed) {
                flushParagraph(into: &blocks, &paragraph, &self)
                blocks.append(.heading(level: level, text: resolve(text)))
                index += 1
                continue
            }
            if trimmed.contains("|"), index + 1 < lines.count, Self.isDelimiterRow(lines[index + 1]) {
                let header = Self.cells(line)
                let alignments = Self.cells(lines[index + 1]).map(Self.alignment)
                if header.count == alignments.count {
                    flushParagraph(into: &blocks, &paragraph, &self)
                    index += 2
                    var rows: [[String]] = []
                    while index < lines.count, lines[index].contains("|"),
                          !lines[index].trimmingCharacters(in: .whitespaces).isEmpty {
                        let row = Self.cells(lines[index])
                        rows.append((0..<header.count).map { $0 < row.count ? resolve(row[$0]) : "" })
                        index += 1
                    }
                    blocks.append(.table(header: header.map { resolve($0) }, alignments: alignments, rows: rows))
                    continue
                }
            }
            if let item = MarkdownBlock.listItem(line) {
                flushParagraph(into: &blocks, &paragraph, &self)
                var text: String
                switch item {
                case let .listItem(_, _, body), let .task(_, _, body): text = body
                default: text = ""
                }
                index += 1
                // Lines indented under the item continue it.
                while index < lines.count, lines[index].hasPrefix("  "),
                      MarkdownBlock.listItem(lines[index]) == nil,
                      !lines[index].trimmingCharacters(in: .whitespaces).isEmpty {
                    text += "\n" + lines[index].trimmingCharacters(in: .whitespaces)
                    index += 1
                }
                switch item {
                case let .listItem(indent, marker, _):
                    blocks.append(.listItem(indent: indent, marker: marker, text: resolve(text)))
                case let .task(indent, checked, _):
                    blocks.append(.task(indent: indent, checked: checked, text: resolve(text)))
                default: break
                }
                continue
            }
            if trimmed.hasPrefix(">") {
                flushParagraph(into: &blocks, &paragraph, &self)
                var body: [String] = []
                while index < lines.count {
                    var quoted = lines[index].trimmingCharacters(in: .whitespaces)
                    guard quoted.hasPrefix(">") else { break }
                    while quoted.hasPrefix(">") {
                        quoted = String(quoted.dropFirst()).trimmingCharacters(in: .whitespaces)
                    }
                    body.append(quoted)
                    index += 1
                }
                blocks.append(.quote(resolve(body.joined(separator: "\n")
                    .trimmingCharacters(in: .whitespacesAndNewlines))))
                continue
            }
            // A term (the paragraph so far) followed by `: definition` lines.
            if let first = Self.definitionLine(line), paragraph.count >= 1 {
                let term = resolve(paragraph.joined(separator: " ").trimmingCharacters(in: .whitespaces))
                paragraph.removeAll()
                var details = [resolve(first)]
                index += 1
                while index < lines.count, let more = Self.definitionLine(lines[index]) {
                    details.append(resolve(more))
                    index += 1
                }
                blocks.append(.definition(term: term, details: details))
                continue
            }
            paragraph.append(line)
            index += 1
        }
        flushParagraph(into: &blocks, &paragraph, &self)
        if !noteOrder.isEmpty {
            let numbered = noteOrder.enumerated().map { offset, key in
                Footnote(number: offset + 1, label: key, text: resolve(notes[key] ?? ""))
            }
            blocks.append(.footnotes(numbered))
        }
        return blocks
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

        case let .task(indent, checked, text):
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Image(systemName: checked ? "checkmark.square.fill" : "square")
                    .foregroundStyle(checked ? Color.accentColor : .secondary)
                    .accessibilityLabel(checked ? "Done" : "Not done")
                Text(MarkdownBlock.inline(text))
                    .strikethrough(checked, color: .secondary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(.leading, CGFloat(min(indent, 4)) * 14)

        case let .code(_, text):
            ScrollView(.horizontal, showsIndicators: false) {
                Text(text)
                    .font(.system(.callout, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(8)
            }
            .background(.quaternary.opacity(0.6), in: RoundedRectangle(cornerRadius: 8))

        case let .table(header, alignments, rows):
            ScrollView(.horizontal, showsIndicators: false) {
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 6) {
                    GridRow {
                        ForEach(Array(header.enumerated()), id: \.offset) { column, cell in
                            Text(MarkdownBlock.inline(cell))
                                .bold()
                                .gridColumnAlignment(Self.alignment(alignments, column))
                        }
                    }
                    Divider()
                    ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                        GridRow {
                            ForEach(Array(row.enumerated()), id: \.offset) { _, cell in
                                Text(MarkdownBlock.inline(cell)).textSelection(.enabled)
                            }
                        }
                    }
                }
                .padding(.vertical, 2)
            }

        case let .definition(term, details):
            VStack(alignment: .leading, spacing: 2) {
                Text(MarkdownBlock.inline(term)).bold()
                ForEach(Array(details.enumerated()), id: \.offset) { _, detail in
                    Text(MarkdownBlock.inline(detail))
                        .textSelection(.enabled)
                        .padding(.leading, 14)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

        case let .footnotes(notes):
            VStack(alignment: .leading, spacing: 2) {
                Divider().padding(.vertical, 2)
                ForEach(notes, id: \.number) { note in
                    Text(MarkdownBlock.inline("\(note.number). \(note.text)"))
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }
        }
    }

    private static func alignment(_ alignments: [TableAlignment], _ column: Int) -> HorizontalAlignment {
        guard column < alignments.count else { return .leading }
        switch alignments[column] {
        case .center: return .center
        case .trailing: return .trailing
        default: return .leading
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
