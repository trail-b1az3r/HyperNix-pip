//  PrivateChats.swift
//  Chats hidden behind Face ID.
//
//  Hiding is about who is holding the phone, so it lives on the phone:
//  the server still has the chat, a second paired phone still sees it,
//  and nothing is encrypted by this. It keeps a chat out of the list
//  until the person holding the phone proves they are the owner — which
//  is the thing people mean when they hand their phone to someone.
//
//  Unlocking lasts until the app leaves the foreground. A private section
//  that stays open after the phone is passed across the table is not
//  private.

import Foundation
import LocalAuthentication
import Observation

@MainActor
@Observable
final class PrivateChats {
    /// Session ids hidden on this phone, per server: the same id on two
    /// servers is two different chats.
    private(set) var hidden: [String: Set<String>] = [:]
    /// Whether the hidden section is showing right now.
    private(set) var isUnlocked = false
    /// Why the last unlock did not happen, in words.
    private(set) var lastFailure: String?

    private let defaults: UserDefaults
    private static let storageKey = "hyperlink.privateChats.v1"

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        if let data = defaults.data(forKey: Self.storageKey),
           let decoded = try? JSONDecoder().decode([String: [String]].self, from: data) {
            hidden = decoded.mapValues(Set.init)
        }
    }

    func isHidden(_ sessionID: String, server: String) -> Bool {
        hidden[server]?.contains(sessionID) ?? false
    }

    func hiddenIDs(server: String) -> Set<String> {
        hidden[server] ?? []
    }

    func hide(_ sessionID: String, server: String) {
        hidden[server, default: []].insert(sessionID)
        save()
    }

    func unhide(_ sessionID: String, server: String) {
        hidden[server]?.remove(sessionID)
        save()
    }

    /// Forget ids the server no longer has, so a deleted chat does not
    /// linger as an invisible entry here.
    func prune(existing: Set<String>, server: String) {
        guard let current = hidden[server] else { return }
        let kept = current.intersection(existing)
        if kept != current {
            hidden[server] = kept
            save()
        }
    }

    /// Face ID, Touch ID, or the passcode when neither is set up. Never
    /// a bypass when the device has no protection at all: then the
    /// section cannot be opened, and the person is told why.
    @discardableResult
    func unlock(reason: String = "Show your private chats") async -> Bool {
        let context = LAContext()
        context.localizedCancelTitle = "Cancel"
        var error: NSError?
        guard context.canEvaluatePolicy(.deviceOwnerAuthentication, error: &error) else {
            lastFailure = "Set a passcode on this iPhone to use private chats."
            isUnlocked = false
            return false
        }
        do {
            let ok = try await context.evaluatePolicy(.deviceOwnerAuthentication, localizedReason: reason)
            isUnlocked = ok
            lastFailure = ok ? nil : "Not unlocked."
            return ok
        } catch {
            isUnlocked = false
            lastFailure = (error as? LAError)?.code == .userCancel ? nil : error.localizedDescription
            return false
        }
    }

    func lock() {
        isUnlocked = false
    }

    private func save() {
        let plain = hidden.mapValues { Array($0).sorted() }
        if let data = try? JSONEncoder().encode(plain) {
            defaults.set(data, forKey: Self.storageKey)
        }
    }
}
