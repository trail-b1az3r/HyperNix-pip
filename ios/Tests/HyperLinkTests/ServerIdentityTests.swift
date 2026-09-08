//  ServerIdentityTests.swift
//  Noticing that the address you reached is a different machine.
//
//  The app reaches its server at whichever of several addresses answers
//  first, and those addresses move: a DHCP lease is reassigned, a
//  tailnet name is transferred, and the app will happily try an address
//  that some other machine now answers on.
//
//  Comparing the server *name* does not catch that. Names are chosen by
//  whoever set the machine up and advertised in the clear; on a home LAN
//  or a tailnet anything can call itself `desktop`. So the check is a
//  fingerprint the server derives from a random seed it keeps to itself,
//  and these tests are about the three cases that decide whether that
//  check is worth having: a match, a genuine mismatch, and the two
//  "nothing to compare" cases that must not be reported as either.

import XCTest
@testable import HyperLink

final class ServerIdentityTests: XCTestCase {
    private let pinned = "a1b2c3d4e5f60718293a4b5c6d7e8f90"

    func testAMatchIsAMatch() {
        XCTAssertEqual(
            ServerIdentity.check(reported: pinned, pinned: pinned), .matches
        )
    }

    func testADifferentFingerprintIsAMismatch() {
        let other = "ffffffffffffffffffffffffffffffff"

        XCTAssertEqual(
            ServerIdentity.check(reported: other, pinned: pinned),
            .mismatch(expected: pinned, actual: other)
        )
    }

    /// A server from before 0.72.4 sends no fingerprint. Reading that as
    /// a mismatch would put a red banner in front of every existing
    /// pairing the moment the app updated — which is how a security
    /// check comes to be switched off in the release after it lands.
    func testAnOlderServerIsUnknownNotAMismatch() {
        XCTAssertEqual(ServerIdentity.check(reported: "", pinned: pinned), .unknown)
    }

    /// A pairing made before this existed has nothing pinned. Refusing
    /// to connect to a server the user has always used, because this
    /// app version has no record of it, would be the app breaking
    /// itself.
    func testAPairingWithNothingPinnedIsUnknown() {
        XCTAssertEqual(ServerIdentity.check(reported: pinned, pinned: ""), .unknown)
    }

    func testOnlyAMatchAuthorisesAdminCredentials() {
        XCTAssertTrue(IdentityVerdict.matches.isSafeForAdminCredentials)
        // Including .unknown. "We have never checked" is not "we
        // checked and it was fine", and the admin credential is the one
        // that costs someone their machine rather than their chat
        // history.
        XCTAssertFalse(IdentityVerdict.unknown.isSafeForAdminCredentials)
        XCTAssertFalse(
            IdentityVerdict.mismatch(expected: "a", actual: "b")
                .isSafeForAdminCredentials
        )
    }

    func testOnlyAMismatchIsWorthTellingTheUserAbout() {
        XCTAssertNil(IdentityVerdict.matches.message)
        XCTAssertNil(IdentityVerdict.unknown.message)
        XCTAssertNotNil(IdentityVerdict.mismatch(expected: "a", actual: "b").message)
    }
}

final class ServerIdentityPinningTests: XCTestCase {
    /// The rule that makes the whole check meaningful: a mismatch must
    /// never quietly become a re-pin. If it did, every "something else
    /// is answering here" would be recorded as the new truth and the
    /// warning would fire exactly once, ever.
    func testAnExistingPinIsNeverOverwritten() {
        let kept = ServerIdentity.pinning(current: "aaaa", reported: "bbbb")

        XCTAssertEqual(kept, "aaaa")
    }

    func testABlankPinIsFilledIn() {
        XCTAssertEqual(ServerIdentity.pinning(current: "", reported: "bbbb"), "bbbb")
    }

    func testABlankReportLeavesTheBlankAlone() {
        XCTAssertEqual(ServerIdentity.pinning(current: "", reported: ""), "")
    }
}

final class FingerprintDisplayTests: XCTestCase {
    /// Grouped so a person can compare it against what the server
    /// printed without losing their place in the middle.
    func testItIsGroupedInEights() {
        let display = ServerIdentity.display("a1b2c3d4e5f60718293a4b5c6d7e8f90")

        XCTAssertEqual(display, "a1b2c3d4 e5f60718 293a4b5c 6d7e8f90")
    }

    func testAShortValueDoesNotCrash() {
        XCTAssertEqual(ServerIdentity.display("abc"), "abc")
        XCTAssertEqual(ServerIdentity.display(""), "")
    }
}

final class EndpointsDecodingTests: XCTestCase {
    private func decode(_ json: String) throws -> EndpointsResponse {
        try JSONDecoder().decode(EndpointsResponse.self, from: Data(json.utf8))
    }

    /// The compatibility case that matters: a server that predates every
    /// field added in 0.72.4 must still decode, or upgrading the app
    /// unpairs everyone whose PC has not been updated yet.
    func testAPre0724ReplyStillDecodes() throws {
        let reply = try decode(#"""
        {"server_name": "desktop", "t1_version": "1.0.26",
         "endpoints": [], "tailscale": false, "reachable_off_lan": false}
        """#)

        XCTAssertEqual(reply.serverName, "desktop")
        XCTAssertEqual(reply.serverFingerprint, "")
        XCTAssertFalse(reply.trustedNetwork)
        XCTAssertFalse(reply.keylessAvailableHere)
    }

    func testTheNewFieldsAreRead() throws {
        let reply = try decode(#"""
        {"server_name": "desktop", "t1_version": "1.0.26", "endpoints": [],
         "tailscale": true, "reachable_off_lan": true,
         "server_fingerprint": "a1b2c3d4e5f60718293a4b5c6d7e8f90",
         "trusted_network": true, "keyless_available_here": true,
         "origin_trust": "tailnet"}
        """#)

        XCTAssertEqual(reply.serverFingerprint, "a1b2c3d4e5f60718293a4b5c6d7e8f90")
        XCTAssertTrue(reply.keylessAvailableHere)
        XCTAssertEqual(reply.originTrust, "tailnet")
    }

    /// "The server allows keyless connections" and "this phone can make
    /// one" are different questions, and only the second is actionable.
    /// A public origin reaching a server in trusted mode gets the first
    /// true and the second false.
    func testTrustedNetworkOnDoesNotImplyKeylessHere() throws {
        let reply = try decode(#"""
        {"server_name": "desktop", "t1_version": "1.0.26", "endpoints": [],
         "tailscale": false, "reachable_off_lan": false,
         "trusted_network": true, "keyless_available_here": false,
         "origin_trust": "public"}
        """#)

        XCTAssertTrue(reply.trustedNetwork)
        XCTAssertFalse(reply.keylessAvailableHere)
    }
}

final class DiscoveredPeerTests: XCTestCase {
    private func decode(_ json: String) throws -> PeersResponse {
        try JSONDecoder().decode(PeersResponse.self, from: Data(json.utf8))
    }

    /// Every row says, in the data the app parses, that nothing about it
    /// has been authenticated. A peer that calls itself "desktop" is
    /// still just a candidate.
    func testAPeerIsNeverDecodedAsVerified() throws {
        let reply = try decode(#"""
        {"peers": [{"name": "laptop.tail1234.ts.net", "address": "100.64.0.2",
                    "url": "http://laptop.tail1234.ts.net:8000", "online": true,
                    "reachable": true, "t1_version": "1.0.26",
                    "server_name": "desktop", "detail": "", "os": "linux",
                    "verified": false}],
         "count": 1, "reachable": 1, "tailscale": true, "detail": ""}
        """#)

        XCTAssertEqual(reply.peers.count, 1)
        XCTAssertFalse(reply.peers[0].verified)
    }

    /// A malicious server could send `verified: true`. It changes
    /// nothing: the app has no code path that treats a peer as trusted,
    /// and pairing is still what establishes what a machine is. This
    /// test exists to record that the field is display metadata, not a
    /// decision — if it ever gates anything, this is the test to come
    /// back to.
    func testTheVerifiedFlagIsNotAnAuthorisation() throws {
        let reply = try decode(#"""
        {"peers": [{"name": "evil.ts.net", "address": "100.64.0.9", "url": "",
                    "online": true, "reachable": true, "verified": true}],
         "count": 1, "reachable": 1, "tailscale": true, "detail": ""}
        """#)

        // Decoded faithfully, and still nothing but a row in a list.
        XCTAssertTrue(reply.peers[0].verified)
        XCTAssertEqual(reply.peers[0].displayName, "evil")
    }

    func testDisplayNameFallsBackToTheAddress() throws {
        let reply = try decode(#"""
        {"peers": [{"name": "", "address": "100.64.0.3", "url": "",
                    "online": true, "reachable": true}],
         "count": 1, "reachable": 1, "tailscale": true, "detail": ""}
        """#)

        XCTAssertEqual(reply.peers[0].displayName, "100.64.0.3")
    }

    func testAnEmptyReplyDecodes() throws {
        let reply = try decode(#"{"detail": "tailscale is not installed."}"#)

        XCTAssertTrue(reply.peers.isEmpty)
        XCTAssertFalse(reply.tailscale)
        XCTAssertEqual(reply.detail, "tailscale is not installed.")
    }
}

final class ServerConnectionPersistenceTests: XCTestCase {
    /// The upgrade case. Every install that predates 0.72.4 has a stored
    /// record with no `serverFingerprint`, and Swift's synthesised
    /// decoder throws on a missing key whether or not the property has a
    /// default — the default belongs to the memberwise initialiser, not
    /// the decoder. Without a hand-written `init(from:)` this record
    /// fails to decode, `restore()` swallows the error, and the app
    /// comes up signed out with nothing anywhere saying why.
    func testARecordFromBeforeTheFingerprintExistedStillDecodes() throws {
        let stored = #"""
        {"endpoints": ["http://desktop.ts.net:8000"], "serverName": "desktop",
         "t1Version": "1.0.26", "deviceID": "dev-1", "deviceName": "iPhone"}
        """#

        let connection = try JSONDecoder().decode(
            ServerConnection.self, from: Data(stored.utf8)
        )

        XCTAssertEqual(connection.deviceID, "dev-1")
        XCTAssertEqual(connection.serverFingerprint, "")
        XCTAssertTrue(connection.isConfigured)
    }

    func testARoundTripKeepsTheFingerprint() throws {
        let original = ServerConnection(
            endpoints: ["http://desktop.ts.net:8000"], serverName: "desktop",
            t1Version: "1.0.26", deviceID: "dev-1", deviceName: "iPhone",
            serverFingerprint: "a1b2c3d4e5f60718293a4b5c6d7e8f90"
        )

        let restored = try JSONDecoder().decode(
            ServerConnection.self, from: JSONEncoder().encode(original)
        )

        XCTAssertEqual(restored, original)
    }

    /// Connecting with a T2S key creates no device record on the server —
    /// the key is the credential — so `deviceID` is empty. Requiring one
    /// meant those connections failed `restore()` and the app signed
    /// itself out on every restart.
    func testAKeyBasedConnectionWithNoDeviceIDIsStillRestorable() {
        let connection = ServerConnection(
            endpoints: ["http://desktop.ts.net:8000"], serverName: "desktop",
            t1Version: "1.0.26", deviceID: "", deviceName: "iPhone"
        )

        XCTAssertTrue(connection.isConfigured)
    }

    func testNoAddressIsNotRestorable() {
        XCTAssertFalse(ServerConnection.empty.isConfigured)
    }
}
