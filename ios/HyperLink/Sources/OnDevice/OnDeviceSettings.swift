//  OnDeviceSettings.swift
//  What the user gets to decide about local inference.

import Foundation
import Security

/// Everything tunable about running a model on the phone.
///
/// Persisted in UserDefaults except the Hugging Face token, which is a
/// credential and lives in the Keychain.
@MainActor
final class OnDeviceSettings: ObservableObject {
    @Published var backend: ComputeBackend {
        didSet { defaults.set(backend.rawValue, forKey: Keys.backend) }
    }

    /// Threads llama.cpp is allowed. 0 means "decide for me", which
    /// resolves to the performance-core count.
    ///
    /// Exposed because the right answer is not obvious: more threads is
    /// faster until it is not, and on a phone the ceiling is thermal
    /// rather than computational. A sustained generation on every core
    /// throttles within a minute or two and ends up slower than the
    /// same work on fewer.
    @Published var threads: Int {
        didSet { defaults.set(threads, forKey: Keys.threads) }
    }

    @Published var contextLength: Int {
        didSet { defaults.set(contextLength, forKey: Keys.context) }
    }

    /// 8 halves the KV cache at a small quality cost, and is frequently
    /// what makes a long context possible on a phone at all.
    @Published var kvCacheBits: Int {
        didSet { defaults.set(kvCacheBits, forKey: Keys.kvBits) }
    }

    @Published var systemPrompt: String {
        didSet { defaults.set(systemPrompt, forKey: Keys.systemPrompt) }
    }

    /// Stop generating when the app is backgrounded.
    ///
    /// Default on. iOS gives a backgrounded app a few seconds of
    /// runtime, and a model mid-generation is holding gigabytes — the
    /// most likely thing to be jetsammed. Finishing the sentence is not
    /// worth losing the conversation.
    @Published var pauseInBackground: Bool {
        didSet { defaults.set(pauseInBackground, forKey: Keys.pauseInBackground) }
    }

    /// Refuse to load a model the fit check says will not run.
    ///
    /// Default on, and overridable, because the estimate is an estimate:
    /// someone who knows their device better should be able to try. The
    /// warning stays either way.
    @Published var enforceMemoryCheck: Bool {
        didSet { defaults.set(enforceMemoryCheck, forKey: Keys.enforceMemory) }
    }

    private let defaults: UserDefaults

    private enum Keys {
        static let backend = "ondevice.backend"
        static let threads = "ondevice.threads"
        static let context = "ondevice.context"
        static let kvBits = "ondevice.kvBits"
        static let systemPrompt = "ondevice.systemPrompt"
        static let pauseInBackground = "ondevice.pauseInBackground"
        static let enforceMemory = "ondevice.enforceMemory"
    }

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        backend = ComputeBackend(
            rawValue: defaults.string(forKey: Keys.backend) ?? ""
        ) ?? .metal
        threads = defaults.integer(forKey: Keys.threads)
        let storedContext = defaults.integer(forKey: Keys.context)
        contextLength = storedContext > 0 ? storedContext : 4096
        let storedKV = defaults.integer(forKey: Keys.kvBits)
        kvCacheBits = storedKV == 8 ? 8 : 16
        systemPrompt = defaults.string(forKey: Keys.systemPrompt) ?? ""
        // `object(forKey:)` rather than `bool(forKey:)`: the second
        // returns false for a key that was never set, which would make
        // both of these default to off instead of on.
        pauseInBackground = defaults.object(forKey: Keys.pauseInBackground) as? Bool ?? true
        enforceMemoryCheck = defaults.object(forKey: Keys.enforceMemory) as? Bool ?? true
    }

    /// Threads to actually use, resolving 0.
    func resolvedThreads(memory: DeviceMemory) -> Int {
        threads > 0 ? min(threads, 16) : max(1, memory.performanceCores)
    }

    /// The thread counts worth offering.
    static func threadOptions(memory: DeviceMemory) -> [Int] {
        let ceiling = max(2, memory.performanceCores)
        return [0] + Array(1...ceiling)
    }

    // MARK: - The Hugging Face token

    /// Kept in the Keychain, not UserDefaults.
    ///
    /// A UserDefaults plist is readable from a file-system backup and
    /// from any process with the container. A token that can download
    /// gated models on the user's account belongs behind the Keychain's
    /// access controls, with `ThisDeviceOnly` so it does not travel to
    /// a restored backup on someone else's phone.
    var huggingFaceToken: String? {
        get { Self.readToken() }
        set { Self.writeToken(newValue) }
    }

    private static let tokenAccount = "huggingface.token"
    private static let tokenService = "com.hypernix.hyperlink"

    private static func readToken() -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: tokenService,
            kSecAttrAccount as String: tokenAccount,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        guard
            SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
            let data = item as? Data,
            let value = String(data: data, encoding: .utf8),
            !value.isEmpty
        else { return nil }
        return value
    }

    private static func writeToken(_ token: String?) {
        let base: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: tokenService,
            kSecAttrAccount as String: tokenAccount,
        ]
        SecItemDelete(base as CFDictionary)
        guard
            let token,
            case let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines),
            !trimmed.isEmpty,
            let data = trimmed.data(using: .utf8)
        else { return }
        var insert = base
        insert[kSecValueData as String] = data
        insert[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        SecItemAdd(insert as CFDictionary, nil)
    }
}
