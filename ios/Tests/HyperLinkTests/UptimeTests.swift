//  UptimeTests.swift
//  How long the server has been up, in words.
//
//  The question this answers on screen is not "how many seconds": it is
//  "did the PC reboot?" — which is the usual reason a conversation lost
//  its context or a pairing stopped working, and which nothing in the
//  app used to say.
//
//  So the formatting rules are about *not* being precise: days and hours
//  once it has been days, never seconds, and a phrase rather than a zero
//  when there is nothing to report.

import XCTest
@testable import HyperLink

final class UptimeTests: XCTestCase {
    func testDaysAndHours() {
        XCTAssertEqual(ServerUptime.describe(3 * 86_400 + 4 * 3_600), "3d 4h")
    }

    func testWholeDaysDropTheHours() {
        XCTAssertEqual(ServerUptime.describe(3 * 86_400), "3d")
    }

    func testHoursAndMinutes() {
        XCTAssertEqual(ServerUptime.describe(4 * 3_600 + 12 * 60), "4h 12m")
    }

    func testMinutesAlone() {
        XCTAssertEqual(ServerUptime.describe(12 * 60), "12m")
    }

    func testSecondsAreNotWorthSaying() {
        /// "41 seconds" is precision nobody asked for about a machine
        /// six hundred miles away.
        XCTAssertEqual(ServerUptime.describe(41), "just now")
    }

    func testNothingKnownIsNotZero() {
        /// A server too old to report it, or a platform that cannot be
        /// asked. "0m" would be a claim that it just restarted, which is
        /// the one thing this field exists to tell people.
        XCTAssertEqual(ServerUptime.describe(nil), "unknown")
        XCTAssertEqual(ServerUptime.describe(0), "unknown")
        XCTAssertEqual(ServerUptime.describe(-5), "unknown")
    }
}
