//  SavedServers.swift
//  More than one machine, and a way back to each of them.
//
//  The app stored exactly one pairing: a single `ServerConnection` under
//  one UserDefaults key and a single token under one keychain account.
//  Pairing with a second machine overwrote the first, and getting back
//  to it meant pairing again — re-entering a key, or being on the right
//  network at the right time. People have a desktop and a laptop, or a
//  machine at home and one at work, and that is not an exotic setup.
//
//  So: a list, up to `maxServers` of them, one of which is selected.
//
//  Where things live
//  -----------------
//  The *connections* are in UserDefaults, as one encoded array. They are
//  addresses, names and fingerprints — not secrets, and they should not
//  survive an uninstall.
//
//  Each server's *token* is in the keychain under its own account, keyed
//  by the record's `id`. That is what makes forgetting one server leave
//  the others signed in.
//
//  Not signing everybody out
//  -------------------------
//  Every existing install has a pairing in the old single-record shape,
//  with its token under the keychain's legacy `default` account. An
//  update that simply started reading a new key would come up unpaired
//  on every device at once, which is the worst possible way to ship a
//  feature about *keeping* connections. `migrateIfNeeded()` reads the
//  old record, writes it as the first entry of the list under a stable
//  id, and adopts the legacy token account for it.

import Foundation

/// One server the app can return to.
struct SavedServer: Codable, Equatable, Sendable, Identifiable {
    /// Stable for the life of the record. The keychain account for this
    /// server's token, and what `select` and `remove` take.
    let id: String
    var connection: ServerConnection
    /// True when this pairing presented no credential on purpose,
    /// because the server was in trusted-network mode.
    var keyless: Bool
    /// Seconds since the epoch. Used for ordering and for evicting the
    /// least recently used when the list is full.
    var lastUsed: Double

    /// The legacy keychain account, kept by whichever record the
    /// migration adopted. New records use their own `id`.
    var usesLegacyToken: Bool = false

    init(
        id: String = UUID().uuidString,
        connection: ServerConnection,
        keyless: Bool,
        lastUsed: Double = Date().timeIntervalSince1970,
        usesLegacyToken: Bool = false
    ) {
        self.id = id
        self.connection = connection
        self.keyless = keyless
        self.lastUsed = lastUsed
        self.usesLegacyToken = usesLegacyToken
    }

    /// Decoded by hand for the same reason `ServerConnection` is: a
    /// property default is used by the memberwise initialiser, not by
    /// the synthesised decoder, so a record written before a field
    /// existed would throw and be silently dropped.
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decodeIfPresent(String.self, forKey: .id) ?? UUID().uuidString
        connection = try container.decode(ServerConnection.self, forKey: .connection)
        keyless = try container.decodeIfPresent(Bool.self, forKey: .keyless) ?? false
        lastUsed = try container.decodeIfPresent(Double.self, forKey: .lastUsed) ?? 0
        usesLegacyToken = try container.decodeIfPresent(
            Bool.self, forKey: .usesLegacyToken
        ) ?? false
    }

    /// The keychain account this server's token is under.
    var tokenAccount: String { usesLegacyToken ? TokenStore.legacyAccount : id }

    /// What to show in a list. The server's own name when it has one,
    /// and otherwise the address — never an empty row.
    var displayName: String {
        let name = connection.serverName.trimmingCharacters(in: .whitespacesAndNewlines)
        if !name.isEmpty { return name }
        return connection.endpoints.first.map(Self.hostOf) ?? "Unnamed server"
    }

    /// The host out of an endpoint URL, for a label. Falls back to the
    /// whole string rather than to nothing.
    static func hostOf(_ endpoint: String) -> String {
        URL(string: endpoint)?.host ?? endpoint
    }

    /// Whether two records are the same machine.
    ///
    /// The fingerprint when both have one — it is derived from a seed
    /// the server keeps, so it survives the machine changing address.
    /// Otherwise the first endpoint, which is the best available answer
    /// for a server too old to have a fingerprint.
    func isSameServer(as other: SavedServer) -> Bool {
        let mine = connection.serverFingerprint
        let theirs = other.connection.serverFingerprint
        if !mine.isEmpty && !theirs.isEmpty { return mine == theirs }
        guard let a = connection.endpoints.first, let b = other.connection.endpoints.first
        else { return false }
        return a == b
    }
}

enum SavedServers {
    /// The cap the request named. Not arbitrary: the list is one
    /// UserDefaults value read at launch, and 32 connections of a few
    /// hundred bytes each is nothing, while an unbounded list is a
    /// keychain entry per server with nothing ever cleaning up.
    static let maxServers = 32

    static let listKey = "hyperlink.servers"
    static let selectedKey = "hyperlink.servers.selected"
    static let migratedKey = "hyperlink.servers.migrated"

    // MARK: - Reading

    static func all(defaults: UserDefaults = .standard) -> [SavedServer] {
        migrateIfNeeded(defaults: defaults)
        return stored(defaults: defaults)
    }

    private static func stored(defaults: UserDefaults) -> [SavedServer] {
        guard let data = defaults.data(forKey: listKey),
              let list = try? JSONDecoder().decode([SavedServer].self, from: data)
        else { return [] }
        return list
    }

    /// The selected server, or the most recently used one when the
    /// selection points at a record that is gone.
    static func selected(defaults: UserDefaults = .standard) -> SavedServer? {
        let list = all(defaults: defaults)
        if let id = defaults.string(forKey: selectedKey),
           let match = list.first(where: { $0.id == id }) {
            return match
        }
        return list.max(by: { $0.lastUsed < $1.lastUsed })
    }

    // MARK: - Writing

    /// Add or update a server, and select it. Returns the stored record.
    ///
    /// Re-pairing with a machine already in the list updates that record
    /// rather than adding a second one: two rows for one desktop, one of
    /// them with a dead token, is not a feature.
    @discardableResult
    static func remember(
        connection: ServerConnection,
        keyless: Bool,
        token: String?,
        defaults: UserDefaults = .standard
    ) -> SavedServer {
        migrateIfNeeded(defaults: defaults)
        var list = stored(defaults: defaults)
        let candidate = SavedServer(connection: connection, keyless: keyless)

        var record: SavedServer
        if let index = list.firstIndex(where: { $0.isSameServer(as: candidate) }) {
            record = list[index]
            record.connection = connection
            record.keyless = keyless
            record.lastUsed = Date().timeIntervalSince1970
            list[index] = record
        } else {
            record = candidate
            list.append(record)
        }

        list = evictOverLimit(list)
        // Only write the token once the record is definitely staying:
        // a keychain entry for a server evicted in the same breath is
        // exactly the orphan this limit exists to prevent.
        if list.contains(where: { $0.id == record.id }) {
            if let token, !token.isEmpty {
                TokenStore.save(token, account: record.tokenAccount)
            } else if keyless {
                // A keyless record holds no credential, and any token
                // left from a previous pairing with this machine would
                // be presented the next time the flag was wrong.
                TokenStore.delete(account: record.tokenAccount)
            }
            // Otherwise: no token and not deliberately keyless, which is
            // a re-save of a known server (a fingerprint being pinned, a
            // refreshed address list). Leave the credential alone rather
            // than deleting a working one because this call did not
            // happen to carry it.
        }
        write(list, defaults: defaults)
        defaults.set(record.id, forKey: selectedKey)
        return record
    }

    /// Make *id* the current server. Returns false when it is not there.
    @discardableResult
    static func select(id: String, defaults: UserDefaults = .standard) -> Bool {
        var list = all(defaults: defaults)
        guard let index = list.firstIndex(where: { $0.id == id }) else { return false }
        list[index].lastUsed = Date().timeIntervalSince1970
        write(list, defaults: defaults)
        defaults.set(id, forKey: selectedKey)
        return true
    }

    /// Forget one server, its token included, leaving the others alone.
    static func remove(id: String, defaults: UserDefaults = .standard) {
        var list = all(defaults: defaults)
        guard let index = list.firstIndex(where: { $0.id == id }) else { return }
        TokenStore.delete(account: list[index].tokenAccount)
        list.remove(at: index)
        write(list, defaults: defaults)
        if defaults.string(forKey: selectedKey) == id {
            defaults.set(
                list.max(by: { $0.lastUsed < $1.lastUsed })?.id, forKey: selectedKey
            )
        }
    }

    /// Forget every server. What a full sign-out does.
    static func removeAll(defaults: UserDefaults = .standard) {
        for server in all(defaults: defaults) {
            TokenStore.delete(account: server.tokenAccount)
        }
        defaults.removeObject(forKey: listKey)
        defaults.removeObject(forKey: selectedKey)
    }

    private static func write(_ list: [SavedServer], defaults: UserDefaults) {
        guard let data = try? JSONEncoder().encode(list) else { return }
        defaults.set(data, forKey: listKey)
    }

    /// Drop the least recently used until the list fits, deleting each
    /// evicted server's token on the way out.
    private static func evictOverLimit(_ list: [SavedServer]) -> [SavedServer] {
        guard list.count > maxServers else { return list }
        let ordered = list.sorted { $0.lastUsed > $1.lastUsed }
        for evicted in ordered.dropFirst(maxServers) {
            TokenStore.delete(account: evicted.tokenAccount)
        }
        return Array(ordered.prefix(maxServers))
    }

    // MARK: - Migration

    /// Carry the single stored pairing into the list, once.
    ///
    /// Runs before every read, and does nothing after the first time.
    /// The alternative — migrating at launch from one call site — breaks
    /// the moment an App Intent or the CarPlay scene reads the list in a
    /// process where that call site never ran, which is exactly the
    /// situation `PairingStore` was created for.
    static func migrateIfNeeded(defaults: UserDefaults = .standard) {
        guard !defaults.bool(forKey: migratedKey) else { return }
        defaults.set(true, forKey: migratedKey)

        guard
            let data = defaults.data(forKey: PairingStore.connectionKey),
            let connection = try? JSONDecoder().decode(ServerConnection.self, from: data),
            connection.isConfigured
        else { return }

        let keyless = defaults.bool(forKey: PairingStore.keylessKey)
        // The legacy token stays exactly where it is, under the `default`
        // account. Moving it would mean a write and a delete against the
        // keychain during an app update, and a failure halfway through
        // that loses somebody's credential for no gain.
        let migrated = SavedServer(
            connection: connection,
            keyless: keyless,
            lastUsed: Date().timeIntervalSince1970,
            usesLegacyToken: true
        )
        write([migrated], defaults: defaults)
        defaults.set(migrated.id, forKey: selectedKey)
    }
}
