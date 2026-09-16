//  MarkdownTests.swift
//  Rendering what the model actually writes.
//
//  Models write markdown — it is the house style of every
//  instruction-tuned model there is. The prose half of a message was
//  rendered with plain `Text`, so a numbered list arrived as one wrapped
//  paragraph with `1.` and `2.` buried in the middle of it.
//
//  Two things these tests are really about.
//
//  **Not eating text.** Every case below checks that the *content*
//  survives, not just that a block was recognised. A parser that drops a
//  line it does not understand is worse than no parser, because the
//  failure is silent and the missing sentence was the answer.
//
//  **Half-written markup.** Every one of these is parsed mid-stream, so
//  `**bol` arrives before `**bold**` does and `1` arrives before `1. `.
//  Nothing may vanish, flicker, or be promoted to a block on the
//  strength of a prefix that is still being typed.

import XCTest
@testable import HyperLink

final class MarkdownTests: XCTestCase {

    // MARK: - Blocks

    func testAHeadingIsAHeading() {
        XCTAssertEqual(
            MarkdownBlock.parse("## Results"),
            [.heading(level: 2, text: "Results")]
        )
    }

    func testTheHashesDecideTheLevel() {
        XCTAssertEqual(
            MarkdownBlock.parse("#### Deep"),
            [.heading(level: 4, text: "Deep")]
        )
    }

    func testAHashWithNoSpaceIsNotAHeading() {
        /// `#1` and `#hashtag` are ordinary text, and a model writing
        /// about C preprocessor directives writes `#include` a lot.
        XCTAssertEqual(MarkdownBlock.parse("#1 in the list"), [.paragraph("#1 in the list")])
        XCTAssertEqual(MarkdownBlock.parse("#include <stdio.h>"),
                       [.paragraph("#include <stdio.h>")])
    }

    func testABulletIsAListItem() {
        XCTAssertEqual(
            MarkdownBlock.parse("- first"),
            [.listItem(indent: 0, marker: "•", text: "first")]
        )
    }

    func testEveryBulletCharacterWorks() {
        for bullet in ["-", "*", "+"] {
            XCTAssertEqual(
                MarkdownBlock.parse("\(bullet) item"),
                [.listItem(indent: 0, marker: "•", text: "item")],
                "\(bullet) was not read as a bullet"
            )
        }
    }

    func testAnOrderedListKeepsItsOwnNumbers() {
        /// A model that writes `1.` three times in a row is making a
        /// point about the steps being interchangeable often enough
        /// that renumbering them silently is a change of meaning.
        XCTAssertEqual(
            MarkdownBlock.parse("1. one\n1. also one"),
            [
                .listItem(indent: 0, marker: "1.", text: "one"),
                .listItem(indent: 0, marker: "1.", text: "also one"),
            ]
        )
    }

    func testAClosingParenthesisCountsToo() {
        XCTAssertEqual(
            MarkdownBlock.parse("2) second"),
            [.listItem(indent: 0, marker: "2)", text: "second")]
        )
    }

    func testIndentationNests() {
        let blocks = MarkdownBlock.parse("- top\n    - under")
        XCTAssertEqual(blocks.count, 2)
        guard case let .listItem(indent, _, _) = blocks[1] else {
            return XCTFail("the indented line was not a list item")
        }
        XCTAssertGreaterThan(indent, 0)
    }

    func testAQuoteIsAQuote() {
        XCTAssertEqual(MarkdownBlock.parse("> careful"), [.quote("careful")])
    }

    func testARuleIsARule() {
        for rule in ["---", "***", "___", "-----"] {
            XCTAssertEqual(MarkdownBlock.parse(rule), [.rule], "\(rule) was not a rule")
        }
    }

    func testTwoDashesAreNotARule() {
        /// An em-dash-happy model should not punch a horizontal rule
        /// through its own reply.
        XCTAssertEqual(MarkdownBlock.parse("--"), [.paragraph("--")])
    }

    func testAMixedDividerIsNotARule() {
        XCTAssertEqual(MarkdownBlock.parse("-*-"), [.paragraph("-*-")])
    }

    // MARK: - Nothing is dropped

    func testABlankLineSeparatesParagraphs() {
        XCTAssertEqual(
            MarkdownBlock.parse("first\n\nsecond"),
            [.paragraph("first"), .paragraph("second")]
        )
    }

    func testWrappedLinesStayOneParagraph() {
        XCTAssertEqual(
            MarkdownBlock.parse("a sentence\nthat wrapped"),
            [.paragraph("a sentence\nthat wrapped")]
        )
    }

    func testEveryLineOfAMixedReplySurvives() {
        /// The real shape of a model's answer, and the check that no
        /// branch above silently swallows a line.
        let reply = """
        # Findings

        Two things stood out:

        1. The first one
        2. The second one

        > Worth checking.

        ---

        Done.
        """
        let blocks = MarkdownBlock.parse(reply)
        let rendered = blocks.map { block -> String in
            switch block {
            case let .heading(_, text): return text
            case let .paragraph(text): return text
            case let .listItem(_, _, text): return text
            case let .quote(text): return text
            case .rule: return ""
            }
        }.joined(separator: " ")

        for fragment in ["Findings", "Two things stood out:", "The first one",
                         "The second one", "Worth checking.", "Done."] {
            XCTAssertTrue(rendered.contains(fragment), "lost: \(fragment)")
        }
    }

    func testAnEmptyStringIsNoBlocks() {
        XCTAssertTrue(MarkdownBlock.parse("").isEmpty)
        XCTAssertTrue(MarkdownBlock.parse("\n\n   \n").isEmpty)
    }

    // MARK: - Half-written markup

    func testABareNumberIsNotYetAList() {
        /// `1` arrives before `1. ` does, and a paragraph that jumps
        /// into a list and back out again as the tokens land is worse
        /// than one that waits.
        XCTAssertEqual(MarkdownBlock.parse("1"), [.paragraph("1")])
        XCTAssertEqual(MarkdownBlock.parse("1."), [.paragraph("1.")])
    }

    func testABareDashIsNotYetAList() {
        XCTAssertEqual(MarkdownBlock.parse("-"), [.paragraph("-")])
    }

    func testAHeadingWithNoTextYetIsNotAHeading() {
        XCTAssertEqual(MarkdownBlock.parse("## "), [.paragraph("##")])
    }

    func testUnterminatedEmphasisRendersAsItsOwnCharacters() {
        /// Not empty, and not bold-until-it-changes-its-mind.
        let partial = MarkdownBlock.inline("this is **bol")
        XCTAssertTrue(String(partial.characters).contains("bol"))
    }

    func testInlineMarkupIsParsed() {
        let bold = MarkdownBlock.inline("a **word** here")
        /// The asterisks are consumed; the word is not.
        let text = String(bold.characters)
        XCTAssertTrue(text.contains("word"))
        XCTAssertFalse(text.contains("**"))
    }

    func testInlineParsingNeverReturnsNothing() {
        /// Whatever it is handed, something readable comes back.
        ///
        /// No image syntax in this list on purpose: `![alt](url)` parses
        /// to an attachment run with no characters, which is correct and
        /// would fail an "it is not empty" check for the wrong reason.
        for input in ["", "[", "**", "a | b | c", "$%^&*", "~~~", "> "] {
            let parsed = MarkdownBlock.inline(input)
            XCTAssertEqual(
                String(parsed.characters).isEmpty, input.isEmpty,
                "input \(input.debugDescription) came back empty"
            )
        }
    }
}
