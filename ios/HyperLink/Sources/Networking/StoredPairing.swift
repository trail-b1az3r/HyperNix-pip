//  StoredPairing.swift
//  Where the pairing lives, for everything that needs to rebuild it.
//
//  `AppState` restores this at launch. So does the CarPlay scene, and so
//  does every App Intent — and those two run in situations where
//  `AppState` does not exist at all: an intent runs in a separate
//  process, and the CarPlay scene can connect while the phone app has
//  never been opened.
//
//  This was `AppState`'s own private keys and private restore logic,
//  which is why the first draft of the intents invented a `PairingStore`
//  that did not exist. One place, three readers.
//
//  What is where, and why
//  ----------------------
//  The *token* is in the keychain (`TokenStore`); the connection — which
//  addresses, which device, which server fingerprint — is in
//  `UserDefaults`. That split is deliberate: a list of LAN addresses is
//  not a secret and does not want keychain semantics (surviving an
//  uninstall), while the credential is and does.

import Foundation

/// A pairing read back from disk.
struct StoredPairing: Sendable {
    let connection: ServerConnection
    /// nil for a deliberate keyless connection. See `keyless`.
    let token: String?
    /// True when this pairing presented no credential on purpose,
    /// because the server was in trusted-network mode.
    let keyless: Bool

    var endpoints: [String] { connection.endpoints }
}

enum PairingStore {
    static let connectionKey = "hyperlink.connection"
    static let keylessKey = "hyperlink.connection.keyless"

    /// The stored pairing, or nil when there is not a usable one.
    ///
    /// "Usable" is doing work here. A stored connection with no
    /// credential is only restorable if it was keyless on purpose;
    /// otherwise the token has been lost — a keychain reset, a restore
    /// from backup — and coming up "paired" would mean every request
    /// 401s with nothing on screen explaining why.
    static func load(defaults: UserDefaults = .standard) -> StoredPairing? {
        guard
            let data = defaults.data(forKey: connectionKey),
            let connection = try? JSONDecoder().decode(ServerConnection.self, from: data),
            connection.isConfigured
        else { return nil }
        let token = TokenStore.load()
        let keyless = defaults.bool(forKey: keylessKey)
        guard token != nil || keyless else { return nil }
        return StoredPairing(connection: connection, token: token, keyless: keyless)
    }

    static func save(
        connection: ServerConnection, keyless: Bool,
        defaults: UserDefaults = .standard
    ) {
        if let data = try? JSONEncoder().encode(connection) {
            defaults.set(data, forKey: connectionKey)
        }
        defaults.set(keyless, forKey: keylessKey)
    }

    /// Forget the pairing. The token is `TokenStore`'s to delete — this
    /// clears only what lives in `UserDefaults`, so a caller that wants
    /// a full sign-out does both.
    static func clear(defaults: UserDefaults = .standard) {
        defaults.removeObject(forKey: connectionKey)
        defaults.removeObject(forKey: keylessKey)
    }

    /// A client configured from the stored pairing, or nil when there
    /// is none. What the intents and the CarPlay scene use instead of
    /// reaching for `AppState`.
    static func client() async -> HyperLinkClient? {
        guard let pairing = load(), !pairing.endpoints.isEmpty else { return nil }
        let client = HyperLinkClient()
        await client.configure(
            endpoints: pairing.endpoints,
            token: pairing.token,
            keyless: pairing.keyless
        )
        return client
    }
}
