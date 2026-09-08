//  AddressAdviceTests.swift
//  The addresses that cannot work, and whether the app says so.
//
//  Three of the addresses a person naturally types are unreachable by
//  construction, and iOS describes all three in terms of its own policy
//  rather than the situation:
//
//    127.0.0.1:8000        -> "Could not connect to the server."
//    100.109.195.71:8000   -> "...App Transport Security policy requires
//    http://host:8000          the use of a secure connection."
//
//  The first reads as though the PC is down. The other two are about
//  the phone, name no address that would work, and are identical to
//  each other despite having different fixes. All three were reported
//  from a real device in that form.
//
//  The 100.x case is the one worth being precise about: ATS exception
//  domains are matched against domain names and do not apply to IP
//  literals at all, so no Info.plist entry can permit it. Only
//  NSAllowsArbitraryLoads would, and that would permit insecure HTTP to
//  the whole internet. The fix is the MagicDNS name, which the server
//  already prints above the numeric address.

import XCTest
@testable import HyperLink

final class LoopbackAddressTests: XCTestCase {
    func testLoopbackIsRefused() {
        XCTAssertTrue(AddressCheck.advice(for: "127.0.0.1:8000").isRefusal)
        XCTAssertTrue(AddressCheck.advice(for: "localhost:8000").isRefusal)
        XCTAssertTrue(AddressCheck.advice(for: "127.0.0.1").isRefusal)
    }

    func testItSaysTheAddressIsThePhone() {
        let message = AddressCheck.advice(for: "127.0.0.1:8000").message ?? ""
        XCTAssertTrue(message.contains("this iPhone"), message)
    }

    func testItNamesTheCommandThatPrintsAWorkingAddress() {
        let message = AddressCheck.advice(for: "127.0.0.1:8000").message ?? ""
        XCTAssertTrue(message.contains("waiter hyperlink pair"), message)
    }

    func testAnyOf127IsLoopback() {
        XCTAssertTrue(AddressCheck.isLoopback("127.0.0.1"))
        XCTAssertTrue(AddressCheck.isLoopback("127.1.2.3"))
        XCTAssertFalse(AddressCheck.isLoopback("128.0.0.1"))
    }
}

final class LocalNetworkAddressTests: XCTestCase {
    // NSAllowsLocalNetworking covers these, so they must not be refused.
    func testPrivateAddressesArePermitted() {
        for address in ["192.168.1.95:8000", "10.0.0.4:8000", "172.16.3.9:8000"] {
            XCTAssertEqual(AddressCheck.advice(for: address), .ok, address)
        }
    }

    func testDotLocalIsPermitted() {
        XCTAssertEqual(AddressCheck.advice(for: "desktop.local:8000"), .ok)
    }

    func testTheBoundariesOfRFC1918() {
        XCTAssertTrue(AddressCheck.isPrivateIPv4("172.16.0.1"))
        XCTAssertTrue(AddressCheck.isPrivateIPv4("172.31.255.254"))
        XCTAssertFalse(AddressCheck.isPrivateIPv4("172.15.0.1"))
        XCTAssertFalse(AddressCheck.isPrivateIPv4("172.32.0.1"))
    }
}

final class TailscaleAddressTests: XCTestCase {
    func testTheMagicDNSNameIsPermitted() {
        // The Info.plist exception domain covers this; refusing it here
        // would block the one address that does work over a tailnet.
        XCTAssertEqual(
            AddressCheck.advice(for: "http://blazeindustries.taile82951.ts.net:8000"),
            .ok
        )
    }

    func testTheBareTailscaleIPIsRefused() {
        XCTAssertTrue(AddressCheck.advice(for: "100.109.195.71:8000").isRefusal)
    }

    func testItExplainsThatAnIPCannotBeExcepted() {
        let message = AddressCheck.advice(for: "100.109.195.71:8000").message ?? ""
        XCTAssertTrue(message.contains("bare IP"), message)
    }

    func testItOffersTheMagicDNSNameAsTheFix() {
        let message = AddressCheck.advice(for: "100.109.195.71:8000").message ?? ""
        XCTAssertTrue(message.contains(".ts.net"), message)
    }

    func testTheCGNATRangeBoundaries() {
        XCTAssertTrue(AddressCheck.isTailscaleIPv4("100.64.0.1"))
        XCTAssertTrue(AddressCheck.isTailscaleIPv4("100.127.255.254"))
        XCTAssertFalse(AddressCheck.isTailscaleIPv4("100.63.0.1"))
        XCTAssertFalse(AddressCheck.isTailscaleIPv4("100.128.0.1"))
    }

    func testTailscaleSpaceIsNotTreatedAsPrivate() {
        // The distinction the whole thing turns on: NSAllowsLocalNetworking
        // covers RFC 1918 and not 100.64/10, which is why a LAN address
        // works and a tailnet address does not.
        XCTAssertFalse(AddressCheck.isPrivateIPv4("100.109.195.71"))
    }
}

final class PublicAddressTests: XCTestCase {
    func testPlainHTTPToAPublicHostIsRefused() {
        XCTAssertTrue(AddressCheck.advice(for: "http://example.com:8000").isRefusal)
    }

    func testHTTPSIsAlwaysPermitted() {
        XCTAssertEqual(AddressCheck.advice(for: "https://example.com"), .ok)
        XCTAssertEqual(AddressCheck.advice(for: "https://100.109.195.71:8000"), .ok)
    }

    func testAnEmptyAddressIsNotRefusedHere() {
        // Left to the existing "enter an address" validation rather than
        // duplicated into a second message that says something else.
        XCTAssertEqual(AddressCheck.advice(for: ""), .ok)
    }
}

final class FailureAdviceTests: XCTestCase {
    private func error(_ code: Int, _ description: String) -> NSError {
        NSError(
            domain: NSURLErrorDomain, code: code,
            userInfo: [NSLocalizedDescriptionKey: description]
        )
    }

    func testATSIsExplainedRatherThanRepeated() {
        let ats = error(
            NSURLErrorAppTransportSecurityRequiresSecureConnection,
            "The resource could not be loaded because the App Transport Security "
                + "policy requires the use of a secure connection."
        )

        let text = FailureAdvice.explain(ats, address: "100.109.195.71:8000")

        XCTAssertTrue(text.contains(".ts.net"), text)
        XCTAssertTrue(text.contains("local network"), text)
    }

    func testARefusedConnectionSaysNothingAnswered() {
        let refused = error(NSURLErrorCannotConnectToHost, "Could not connect to the server.")

        let text = FailureAdvice.explain(refused, address: "192.168.1.95:8000")

        XCTAssertTrue(text.contains("Nothing answered"), text)
        XCTAssertTrue(text.contains("hypernix-t1 status"), text)
    }

    func testATimeoutMentionsBothNetworks() {
        let timedOut = error(NSURLErrorTimedOut, "The request timed out.")

        let text = FailureAdvice.explain(timedOut, address: "desktop.ts.net:8000")

        XCTAssertTrue(text.contains("tailscale status"), text)
    }

    func testAnUnresolvableHostSaysSo() {
        let notFound = error(NSURLErrorCannotFindHost, "A server with the specified hostname could not be found.")

        let text = FailureAdvice.explain(notFound, address: "desktop.ts.net:8000")

        XCTAssertTrue(text.contains("MagicDNS"), text)
    }

    func testAnUnrelatedErrorIsPassedThroughUnchanged() {
        // A server-side refusal — a wrong pairing code — must not be
        // rewritten into network advice.
        let other = error(0, "That pairing code is not valid.")

        let text = FailureAdvice.explain(other, address: "192.168.1.95:8000")

        XCTAssertEqual(text, "That pairing code is not valid.")
    }
}
