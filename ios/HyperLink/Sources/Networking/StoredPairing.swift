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
//
//  One pairing, or one of several
//  ------------------------------
//  There is now a list of up to 32 of them in `SavedServers`, and this
//  type is the *selected* one. Everything above still holds — the
//  intents and the CarPlay scene want "whichever server I am on", not a
//  picker — so `PairingStore` keeps its shape and reads through to the
//  list underneath. `connectionKey` and `keylessKey` survive as what the
//  migration reads, not as where anything is written.

import Foundation

/// A pairing read back from disk.
struct StoredPairing: Sendable {
    /// Which saved server this is. Empty only for a pairing built in
    /// memory rather than read back from the list.
    var id: String = ""
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
        guard let server = SavedServers.selected(defaults: defaults),
              server.connection.isConfigured
        else { return nil }
        let token = TokenStore.load(account: server.tokenAccount)
        guard token != nil || server.keyless else { return nil }
        return StoredPairing(
            id: server.id, connection: server.connection,
            token: token, keyless: server.keyless
        )
    }

    /// Remember this pairing and make it the current one.
    ///
    /// The token argument is new and it matters: with a keychain account
    /// per server, saving a connection without saying which credential
    /// goes with it cannot work. Passing nil leaves whatever is already
    /// stored for that server alone, which is what a re-connect to a
    /// known machine wants.
    static func save(
        connection: ServerConnection, keyless: Bool,
        token: String? = nil,
        defaults: UserDefaults = .standard
    ) {
        SavedServers.remember(
            connection: connection, keyless: keyless, token: token, defaults: defaults
        )
    }

    /// Forget the *selected* pairing, its token included.
    ///
    /// This used to leave the token to the caller, because there was one
    /// token and clearing it was a separate decision. With an account
    /// per server, leaving it behind is an orphaned credential nothing
    /// will ever look at again.
    static func clear(defaults: UserDefaults = .standard) {
        guard let server = SavedServers.selected(defaults: defaults) else { return }
        SavedServers.remove(id: server.id, defaults: defaults)
    }

    /// Forget every server. A full sign-out.
    static func clearAll(defaults: UserDefaults = .standard) {
        SavedServers.removeAll(defaults: defaults)
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
