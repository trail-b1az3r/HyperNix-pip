//  SavedServersTests.swift
//  Keeping more than one machine, and not losing the one you had.
//
//  The app stored exactly one pairing. Pairing with a laptop overwrote
//  the desktop's record and left its token behind under the one keychain
//  account there was, so getting back to the desktop meant pairing
//  again.
//
//  The risky half of fixing that is not the list — it is the update.
//  Every install in the world has a record in the old shape, and an
//  update that started reading a new key would come up unpaired on every
//  device at once: the worst possible way to ship a feature about
//  *keeping* connections. `testTheExistingPairingSurvivesTheUpdate` is
//  the one that matters here.
//
//  Each test gets a throwaway `UserDefaults` suite, so nothing here
//  touches the app's own. The keychain is not stubbed and these tests do
//  not assert on it: a keychain call from a test bundle behaves
//  differently per signing configuration, and a test that passes because
//  `SecItemAdd` quietly failed is worse than no test.

import XCTest
@testable import HyperLink

final class SavedServersTests: XCTestCase {
    private var defaults: UserDefaults!
    private var suiteName: String!

    override func setUp() {
        super.setUp()
        suiteName = "hyperlink.tests.\(UUID().uuidString)"
        defaults = UserDefaults(suiteName: suiteName)
    }

    override func tearDown() {
        defaults.removePersistentDomain(forName: suiteName)
        defaults = nil
        super.tearDown()
    }

    private func connection(
        name: String, address: String, fingerprint: String = ""
    ) -> ServerConnection {
        ServerConnection(
            endpoints: [address], serverName: name, t1Version: "1.0.26.9.2.3",
            deviceID: "dev-\(name)", deviceName: "iPhone",
            serverFingerprint: fingerprint
        )
    }

    // MARK: - The list

    func testAServerIsRemembered() {
        SavedServers.remember(
            connection: connection(name: "desktop", address: "http://192.168.1.10:8000"),
            keyless: false, token: "tok", defaults: defaults
        )
        XCTAssertEqual(SavedServers.all(defaults: defaults).count, 1)
        XCTAssertEqual(SavedServers.selected(defaults: defaults)?.displayName, "desktop")
    }

    /// The whole bug: this used to be one record and one key.
    func testASecondServerDoesNotReplaceTheFirst() {
        SavedServers.remember(
            connection: connection(name: "desktop", address: "http://192.168.1.10:8000"),
            keyless: false, token: "a", defaults: defaults
        )
        SavedServers.remember(
            connection: connection(name: "laptop", address: "http://192.168.1.11:8000"),
            keyless: false, token: "b", defaults: defaults
        )
        let names = Set(SavedServers.all(defaults: defaults).map(\.displayName))
        XCTAssertEqual(names, ["desktop", "laptop"])
    }

    func testTheNewestIsSelected() {
        SavedServers.remember(
            connection: connection(name: "desktop", address: "http://192.168.1.10:8000"),
            keyless: false, token: "a", defaults: defaults
        )
        SavedServers.remember(
            connection: connection(name: "laptop", address: "http://192.168.1.11:8000"),
            keyless: false, token: "b", defaults: defaults
        )
        XCTAssertEqual(SavedServers.selected(defaults: defaults)?.displayName, "laptop")
    }

    func testSwitchingChangesTheSelection() {
        let desktop = SavedServers.remember(
            connection: connection(name: "desktop", address: "http://192.168.1.10:8000"),
            keyless: false, token: "a", defaults: defaults
        )
        SavedServers.remember(
            connection: connection(name: "laptop", address: "http://192.168.1.11:8000"),
            keyless: false, token: "b", defaults: defaults
        )
        XCTAssertTrue(SavedServers.select(id: desktop.id, defaults: defaults))
        XCTAssertEqual(SavedServers.selected(defaults: defaults)?.id, desktop.id)
    }

    func testSelectingSomethingThatIsNotThereFails() {
        XCTAssertFalse(SavedServers.select(id: "nonsense", defaults: defaults))
    }

    // MARK: - One machine, one record

    func testRePairingTheSameMachineUpdatesItsRecord() {
        /// Two rows for one desktop, one of them with a dead token, is
        /// not a feature.
        let first = SavedServers.remember(
            connection: connection(
                name: "desktop", address: "http://192.168.1.10:8000",
                fingerprint: "abc123"
            ),
            keyless: false, token: "a", defaults: defaults
        )
        let again = SavedServers.remember(
            connection: connection(
                name: "desktop renamed", address: "http://desktop.ts.net:8000",
                fingerprint: "abc123"
            ),
            keyless: false, token: "b", defaults: defaults
        )
        XCTAssertEqual(first.id, again.id)
        XCTAssertEqual(SavedServers.all(defaults: defaults).count, 1)
        XCTAssertEqual(
            SavedServers.selected(defaults: defaults)?.displayName, "desktop renamed"
        )
    }

    func testTheFingerprintBeatsTheAddress() {
        /// A machine that moved from the LAN to a tailnet is the same
        /// machine, and pairing with it again must not make a second row.
        let lan = SavedServers.remember(
            connection: connection(
                name: "desktop", address: "http://192.168.1.10:8000",
                fingerprint: "abc123"
            ),
            keyless: false, token: "a", defaults: defaults
        )
        let tailnet = SavedServers.remember(
            connection: connection(
                name: "desktop", address: "http://desktop.ts.net:8000",
                fingerprint: "abc123"
            ),
            keyless: false, token: "a", defaults: defaults
        )
        XCTAssertEqual(lan.id, tailnet.id)
    }

    func testTwoMachinesOnOneAddressAreStillTwoMachines() {
        /// Same address, different fingerprints: a DHCP lease that moved.
        SavedServers.remember(
            connection: connection(
                name: "old", address: "http://192.168.1.10:8000", fingerprint: "aaa"
            ),
            keyless: false, token: "a", defaults: defaults
        )
        SavedServers.remember(
            connection: connection(
                name: "new", address: "http://192.168.1.10:8000", fingerprint: "bbb"
            ),
            keyless: false, token: "b", defaults: defaults
        )
        XCTAssertEqual(SavedServers.all(defaults: defaults).count, 2)
    }

    // MARK: - The limit

    func testTheLimitIsThirtyTwo() {
        XCTAssertEqual(SavedServers.maxServers, 32)
    }

    func testPairingPastTheLimitDropsTheLeastRecentlyUsed() {
        for index in 0..<(SavedServers.maxServers + 4) {
            SavedServers.remember(
                connection: connection(
                    name: "server-\(index)",
                    address: "http://10.0.0.\(index):8000",
                    fingerprint: "fp-\(index)"
                ),
                keyless: false, token: "tok-\(index)", defaults: defaults
            )
        }
        let kept = SavedServers.all(defaults: defaults)
        XCTAssertEqual(kept.count, SavedServers.maxServers)
        // The four oldest went; the newest is still selected.
        XCTAssertFalse(kept.contains { $0.displayName == "server-0" })
        XCTAssertTrue(kept.contains { $0.displayName == "server-35" })
    }

    // MARK: - Forgetting one

    func testForgettingOneLeavesTheOthers() {
        let desktop = SavedServers.remember(
            connection: connection(name: "desktop", address: "http://192.168.1.10:8000"),
            keyless: false, token: "a", defaults: defaults
        )
        SavedServers.remember(
            connection: connection(name: "laptop", address: "http://192.168.1.11:8000"),
            keyless: false, token: "b", defaults: defaults
        )
        SavedServers.remove(id: desktop.id, defaults: defaults)
        XCTAssertEqual(
            SavedServers.all(defaults: defaults).map(\.displayName), ["laptop"]
        )
    }

    func testForgettingTheSelectedOneFallsBackToAnother() {
        SavedServers.remember(
            connection: connection(name: "desktop", address: "http://192.168.1.10:8000"),
            keyless: false, token: "a", defaults: defaults
        )
        let laptop = SavedServers.remember(
            connection: connection(name: "laptop", address: "http://192.168.1.11:8000"),
            keyless: false, token: "b", defaults: defaults
        )
        SavedServers.remove(id: laptop.id, defaults: defaults)
        XCTAssertEqual(SavedServers.selected(defaults: defaults)?.displayName, "desktop")
    }

    func testForgettingTheLastOneLeavesNothingSelected() {
        let only = SavedServers.remember(
            connection: connection(name: "desktop", address: "http://192.168.1.10:8000"),
            keyless: false, token: "a", defaults: defaults
        )
        SavedServers.remove(id: only.id, defaults: defaults)
        XCTAssertNil(SavedServers.selected(defaults: defaults))
    }

    // MARK: - The update

    func testTheExistingPairingSurvivesTheUpdate() {
        /// The one that matters. Every install has a record in the old
        /// single-pairing shape; an update that did not carry it across
        /// would sign every user out at once.
        let existing = connection(
            name: "desktop", address: "http://192.168.1.10:8000", fingerprint: "abc"
        )
        defaults.set(try! JSONEncoder().encode(existing), forKey: PairingStore.connectionKey)
        defaults.set(false, forKey: PairingStore.keylessKey)

        let migrated = SavedServers.all(defaults: defaults)
        XCTAssertEqual(migrated.count, 1)
        XCTAssertEqual(migrated.first?.displayName, "desktop")
        XCTAssertEqual(migrated.first?.connection.serverFingerprint, "abc")
        XCTAssertEqual(SavedServers.selected(defaults: defaults)?.id, migrated.first?.id)
    }

    func testTheMigratedRecordKeepsTheLegacyKeychainAccount() {
        /// Its token is under `default`, where it has always been.
        /// Moving it would mean a keychain write and delete during an
        /// app update, and a failure halfway through loses a credential
        /// for no gain.
        let existing = connection(name: "desktop", address: "http://192.168.1.10:8000")
        defaults.set(try! JSONEncoder().encode(existing), forKey: PairingStore.connectionKey)

        let migrated = SavedServers.all(defaults: defaults).first
        XCTAssertEqual(migrated?.tokenAccount, TokenStore.legacyAccount)
    }

    func testANewRecordGetsItsOwnKeychainAccount() {
        let fresh = SavedServers.remember(
            connection: connection(name: "laptop", address: "http://192.168.1.11:8000"),
            keyless: false, token: "b", defaults: defaults
        )
        XCTAssertEqual(fresh.tokenAccount, fresh.id)
        XCTAssertNotEqual(fresh.tokenAccount, TokenStore.legacyAccount)
    }

    func testTheMigrationRunsOnce() {
        /// It runs before every read, so running twice would re-create a
        /// record the user had just forgotten.
        let existing = connection(name: "desktop", address: "http://192.168.1.10:8000")
        defaults.set(try! JSONEncoder().encode(existing), forKey: PairingStore.connectionKey)

        let migrated = SavedServers.all(defaults: defaults).first!
        SavedServers.remove(id: migrated.id, defaults: defaults)
        XCTAssertTrue(SavedServers.all(defaults: defaults).isEmpty)
    }

    func testAFreshInstallMigratesNothing() {
        XCTAssertTrue(SavedServers.all(defaults: defaults).isEmpty)
        XCTAssertNil(SavedServers.selected(defaults: defaults))
    }

    // MARK: - Labels

    func testAServerWithNoNameFallsBackToItsHost() {
        let nameless = SavedServer(
            connection: ServerConnection(
                endpoints: ["http://192.168.1.10:8000"], serverName: "",
                t1Version: "", deviceID: "", deviceName: ""
            ),
            keyless: true
        )
        XCTAssertEqual(nameless.displayName, "192.168.1.10")
    }

    func testAServerWithNothingAtAllStillHasALabel() {
        /// An empty row in a list of servers is unclickable and
        /// unexplainable.
        let empty = SavedServer(connection: .empty, keyless: true)
        XCTAssertFalse(empty.displayName.isEmpty)
    }

    // MARK: - Changing the port

    //  The port is the one part of a pairing that changes for ordinary
    //  reasons: the server restarted somewhere else, or moved behind a
    //  different forward. Before this, the only way to follow it was to
    //  pair again — which means finding a key again, and which threw
    //  away the fingerprint that says this is the same machine.

    private func multiHomed(name: String) -> ServerConnection {
        ServerConnection(
            endpoints: ["http://192.168.1.10:8000", "http://desktop.tail1234.ts.net:8000"],
            serverName: name, t1Version: "1.0.26.9.2.3",
            deviceID: "dev-\(name)", deviceName: "iPhone",
            serverFingerprint: "fp-\(name)"
        )
    }

    func testTheNewPortReachesEveryEndpoint() {
        // A pairing carries a LAN address *and* a tailnet name. Changing
        // one of them leaves the failover list half pointing at nothing,
        // and the half that still works is whichever one is tried first.
        SavedServers.remember(
            connection: multiHomed(name: "desktop"), keyless: false,
            token: "tok", defaults: defaults
        )
        let id = SavedServers.all(defaults: defaults)[0].id

        XCTAssertTrue(SavedServers.setPort(id: id, port: 9000, defaults: defaults))

        let endpoints = SavedServers.all(defaults: defaults)[0].connection.endpoints
        XCTAssertEqual(endpoints, [
            "http://192.168.1.10:9000",
            "http://desktop.tail1234.ts.net:9000",
        ])
    }

    func testTheMachineIsStillTheSameMachine() {
        // The point of editing rather than re-pairing: the fingerprint
        // and the keychain account survive, so the pinned identity is
        // not thrown away and the token is not asked for again.
        SavedServers.remember(
            connection: multiHomed(name: "desktop"), keyless: false,
            token: "tok", defaults: defaults
        )
        let before = SavedServers.all(defaults: defaults)[0]

        XCTAssertTrue(SavedServers.setPort(id: before.id, port: 9000, defaults: defaults))

        let after = SavedServers.all(defaults: defaults)[0]
        XCTAssertEqual(after.id, before.id)
        XCTAssertEqual(after.tokenAccount, before.tokenAccount)
        XCTAssertEqual(after.connection.serverFingerprint, before.connection.serverFingerprint)
        XCTAssertEqual(after.connection.serverName, before.connection.serverName)
    }

    func testTheOtherServersAreUntouched() {
        SavedServers.remember(
            connection: connection(name: "desktop", address: "http://192.168.1.10:8000"),
            keyless: false, token: "a", defaults: defaults
        )
        SavedServers.remember(
            connection: connection(name: "laptop", address: "http://192.168.1.11:8000"),
            keyless: false, token: "b", defaults: defaults
        )
        let desktop = SavedServers.all(defaults: defaults).first { $0.displayName == "desktop" }!

        XCTAssertTrue(SavedServers.setPort(id: desktop.id, port: 9000, defaults: defaults))

        let laptop = SavedServers.all(defaults: defaults).first { $0.displayName == "laptop" }!
        XCTAssertEqual(laptop.connection.endpoints, ["http://192.168.1.11:8000"])
    }

    func testAnImpossiblePortIsRefused() {
        SavedServers.remember(
            connection: connection(name: "desktop", address: "http://192.168.1.10:8000"),
            keyless: false, token: "a", defaults: defaults
        )
        let id = SavedServers.all(defaults: defaults)[0].id

        XCTAssertFalse(SavedServers.setPort(id: id, port: 0, defaults: defaults))
        XCTAssertFalse(SavedServers.setPort(id: id, port: 65_536, defaults: defaults))
        XCTAssertEqual(
            SavedServers.all(defaults: defaults)[0].connection.endpoints,
            ["http://192.168.1.10:8000"]
        )
    }

    func testAMachineThatIsNotThereFails() {
        XCTAssertFalse(SavedServers.setPort(id: "no-such-id", port: 9000, defaults: defaults))
    }

    func testTheCommonPortIsOfferedAsTheStartingPoint() {
        SavedServers.remember(
            connection: multiHomed(name: "desktop"), keyless: false,
            token: "tok", defaults: defaults
        )
        let server = SavedServers.all(defaults: defaults)[0]
        XCTAssertEqual(SavedServers.commonPort(of: server), 8000)
    }

    func testEndpointsThatDisagreeOfferNothing() {
        // A field prefilled with 8080 for a record whose tailnet
        // endpoint is on 9000 would misstate what Save is about to do.
        let mixed = ServerConnection(
            endpoints: ["http://192.168.1.10:8080", "http://desktop.ts.net:9000"],
            serverName: "desktop", t1Version: "1", deviceID: "d",
            deviceName: "iPhone", serverFingerprint: ""
        )
        SavedServers.remember(
            connection: mixed, keyless: false, token: "tok", defaults: defaults
        )
        let server = SavedServers.all(defaults: defaults)[0]
        XCTAssertNil(SavedServers.commonPort(of: server))
    }

    func testARewriteKeepsTheSchemeAndThePath() {
        XCTAssertEqual(
            SavedServers.replacingPort(in: "https://desktop.ts.net:8000", with: 9000),
            "https://desktop.ts.net:9000"
        )
    }

    func testSomethingThatIsNotAURLIsLeftAlone() {
        // Better a record that still points where it did than one
        // rewritten into a string nothing can parse.
        XCTAssertEqual(
            SavedServers.replacingPort(in: "not a url", with: 9000),
            "not a url"
        )
    }

    func testAnEndpointWithNoPortGainsOne() {
        XCTAssertEqual(
            SavedServers.replacingPort(in: "http://192.168.1.10", with: 9000),
            "http://192.168.1.10:9000"
        )
    }
}
