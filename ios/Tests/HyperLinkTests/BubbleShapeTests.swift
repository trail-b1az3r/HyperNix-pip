//  BubbleShapeTests.swift
//  The tail joins the bubble; it never punches a hole in it.
//
//  Reported from an iPhone: the assistant's bubble had a dark triangle
//  with a light rim at its bottom-left corner. The tail is drawn for the
//  right and mirrored for the left, mirroring reverses its winding, and
//  under the non-zero fill the overlap with the bubble cancelled out.
//  The user's side, winding with its bubble, never showed it.

import SwiftUI
import XCTest
@testable import HyperLink

final class BubbleShapeTests: XCTestCase {
    private let rect = CGRect(x: 0, y: 0, width: 200, height: 60)

    func testEveryPointOfTheBubbleIsFilledForBothSpeakers() {
        let bubble = Path(roundedRect: rect, cornerRadius: 17, style: .continuous)
        for isUser in [true, false] {
            let shape = BubbleShape(isUser: isUser, showsTail: true).path(in: rect)
            var checked = 0
            for x in stride(from: 0.5, to: 200, by: 1) {
                for y in stride(from: 40.5, to: 60, by: 1) {
                    let point = CGPoint(x: x, y: y)
                    guard bubble.contains(point) else { continue }
                    checked += 1
                    XCTAssertTrue(shape.contains(point),
                                  "\(isUser ? "user" : "assistant") bubble has a hole at \(point)")
                }
            }
            XCTAssertGreaterThan(checked, 1000)
        }
    }

    func testTheTailStillSticksOutOnTheSpeakersSide() {
        let user = BubbleShape(isUser: true, showsTail: true).path(in: rect)
        let assistant = BubbleShape(isUser: false, showsTail: true).path(in: rect)
        XCTAssertTrue(user.contains(CGPoint(x: rect.maxX + 3, y: rect.maxY - 0.5)))
        XCTAssertTrue(assistant.contains(CGPoint(x: rect.minX - 3, y: rect.maxY - 0.5)))
        XCTAssertFalse(user.contains(CGPoint(x: rect.minX - 3, y: rect.maxY - 0.5)))
    }

    func testNoTailIsJustTheBubble() {
        let plain = BubbleShape(isUser: false, showsTail: false).path(in: rect)
        XCTAssertFalse(plain.contains(CGPoint(x: rect.minX - 3, y: rect.maxY - 0.5)))
        XCTAssertTrue(plain.contains(CGPoint(x: rect.midX, y: rect.midY)))
    }
}
