//  TokenStore.swift
//  The device token, in the Keychain.
//
//  The token is a bearer credential for someone's home machine: it goes
//  in the Keychain, not UserDefaults, and with
//  `kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly`.
//
//  Both halves of that constant are deliberate. *AfterFirstUnlock*
//  rather than *WhenUnlocked* because a notification-driven refresh can
//  run with the screen locked and would otherwise fail. *ThisDeviceOnly*
//  because the token identifies one physical device to the server: if
//  it rode an iCloud backup to a new phone, two devices would share one
//  identity and revoking either would revoke both.
//
//  One account per saved server
//  ----------------------------
//  There used to be exactly one account, `default`, because there was
//  exactly one pairing. With up to 32 saved servers each needs its own,
//  or forgetting one machine signs you out of all of them — so the
//  account is the `SavedServer`'s id, and `legacyAccount` is kept for
//  the record the migration adopted. The no-argument calls still mean
//  `legacyAccount`, which is what makes an app built before this keep
//  working against a keychain written after it.

import Foundation
import Security

enum TokenStore {
    private static let service = "com.hypernix.hyperlink.device-token"

    /// The account every token used before there was more than one
    /// server. Still the home of whichever pairing the migration
    /// adopted — see `SavedServers.migrateIfNeeded`.
    static let legacyAccount = "default"

    static func save(_ token: String, account: String = legacyAccount) {
        let data = Data(token.utf8)
        // SecItemUpdate cannot create, and SecItemAdd cannot replace, so
        // delete-then-add is the standard shape for "upsert" here.
        //
        // `delete(account:)`, not `delete()`: with an account per saved
        // server, the defaulted call would clear the legacy account and
        // sign the user out of a different machine than the one being
        // saved.
        delete(account: account)
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly,
        ]
        SecItemAdd(query as CFDictionary, nil)
    }

    static func load(account: String = legacyAccount) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data,
              let token = String(data: data, encoding: .utf8),
              !token.isEmpty
        else { return nil }
        return token
    }

    static func delete(account: String = legacyAccount) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        SecItemDelete(query as CFDictionary)
    }
}
