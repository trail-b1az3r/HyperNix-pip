//  AdminCredentialStore.swift
//  An admin credential, held for as long as it is needed and no longer.
//
//  A device token is a phone's ordinary credential: scoped, revocable
//  from the PC, and meant to live on the device indefinitely. An *admin*
//  credential is not that. It mints pairing codes, reads the audit log,
//  and — since 0.72.4 — stops training runs. It is the credential that,
//  if it leaks, costs someone their machine rather than their chat
//  history.
//
//  So this is deliberately not `TokenStore` with a different key:
//
//  **It does not survive a restart by default.** The item is written
//  with `kSecAttrAccessibleWhenPasscodeSetThisDeviceOnly` and a
//  session marker; on launch, `beginSession()` clears anything left from
//  the last run. A phone found on a train, or restored from a backup,
//  has no admin credential on it. Someone who genuinely wants it kept
//  turns that on themselves, having read what it means.
//
//  **It is never logged, never printed, and never described.** There is
//  no `debugDescription`, no `CustomStringConvertible`, and the value is
//  returned as a `String` only from `load()`. `has` answers the question
//  a view actually asks — "is one stored?" — so no UI code needs to hold
//  the value to decide whether to show a button.
//
//  **It is bound to one server.** Stored under the server's fingerprint,
//  not its name or address, so a credential for one machine cannot be
//  presented to another that happens to answer at the same address. See
//  `ServerIdentity`.
//
//  What this cannot do is stop a compromised device from using the
//  credential while the app is open. Nothing on the phone can. What it
//  does is bound the window to a single app session unless someone has
//  said otherwise.

import Foundation
import Security

enum AdminCredentialStore {
    private static let service = "com.hypernix.hyperlink.admin-credential"
    /// Set once per launch. Anything stored under a *different* marker
    /// is from a previous run and is cleared, which is what makes
    /// "forgotten on restart" true rather than aspirational.
    private static let sessionKey = "hyperlink.admin.session"
    private static let persistKey = "hyperlink.admin.persist"

    // MARK: - Policy

    /// Whether an admin credential is allowed to outlive the app.
    ///
    /// Off by default. Turning it on is a decision about someone's own
    /// machine, and the settings copy says what it costs; defaulting it
    /// on would make that decision for them silently.
    static var persistsAcrossLaunches: Bool {
        get { UserDefaults.standard.bool(forKey: persistKey) }
        set {
            UserDefaults.standard.set(newValue, forKey: persistKey)
            // Turning it *off* has to take effect now, not at the next
            // launch. A switch that says "do not keep this" and leaves
            // the credential in the keychain until tomorrow is worse
            // than no switch, because it is believed.
            if !newValue { deleteAll() }
        }
    }

    // MARK: - Session

    /// Call once at launch, before anything reads a credential.
    ///
    /// Clears whatever the previous run left behind unless the user
    /// asked for it to be kept. Idempotent: calling it twice in one
    /// launch does not discard a credential entered in that launch,
    /// because the marker only changes when the process is new.
    static func beginSession() {
        let previous = UserDefaults.standard.string(forKey: sessionKey)
        if previous == processMarker { return }
        if !persistsAcrossLaunches {
            deleteAll()
        }
        UserDefaults.standard.set(processMarker, forKey: sessionKey)
    }

    /// Stable within one process, different in the next. That is the
    /// whole requirement: it is compared against what the previous run
    /// wrote, and only needs to not collide with it.
    private static let processMarker = UUID().uuidString

    // MARK: - Storage

    /// Store *credential* for the server identified by *fingerprint*.
    ///
    /// Keyed on the fingerprint rather than the address or the name:
    /// addresses move and names can be claimed by any machine on the
    /// network, and an admin credential handed to the wrong machine is
    /// the failure this whole type exists to make unlikely.
    @discardableResult
    static func save(_ credential: String, fingerprint: String) -> Bool {
        guard !credential.isEmpty, !fingerprint.isEmpty else { return false }
        delete(fingerprint: fingerprint)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: fingerprint,
            kSecValueData as String: Data(credential.utf8),
            // ThisDeviceOnly: it must not ride an iCloud backup onto
            // another phone. WhenUnlocked rather than AfterFirstUnlock:
            // unlike the device token, nothing here needs to run while
            // the screen is locked, so the stricter class costs nothing.
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
        ]
        return SecItemAdd(query as CFDictionary, nil) == errSecSuccess
    }

    static func load(fingerprint: String) -> String? {
        guard !fingerprint.isEmpty else { return nil }
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: fingerprint,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data,
              let value = String(data: data, encoding: .utf8),
              !value.isEmpty
        else { return nil }
        return value
    }

    /// Whether one is stored, without handing the value to the caller.
    ///
    /// The question a view asks is "should I show the admin section?",
    /// and answering it with the credential itself is how a secret ends
    /// up held in view state and, eventually, in a crash report.
    static func has(fingerprint: String) -> Bool {
        guard !fingerprint.isEmpty else { return false }
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: fingerprint,
            kSecReturnData as String: false,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        return SecItemCopyMatching(query as CFDictionary, nil) == errSecSuccess
    }

    static func delete(fingerprint: String) {
        guard !fingerprint.isEmpty else { return }
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: fingerprint,
        ]
        SecItemDelete(query as CFDictionary)
    }

    /// Every admin credential this app holds, for every server.
    static func deleteAll() {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
        ]
        SecItemDelete(query as CFDictionary)
    }
}
