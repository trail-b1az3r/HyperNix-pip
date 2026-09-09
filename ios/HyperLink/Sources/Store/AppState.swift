//  AppState.swift
//  The app's single source of truth.
//
//  One `@Observable` object on the main actor, holding the connection,
//  the session list, and the messages of whichever session is open.
//  Everything that talks to the network goes through here rather than
//  from a view, for one reason: a streamed answer arrives over several
//  seconds and must survive the user scrolling, rotating, or switching
//  tabs, none of which a view's lifetime does.

import Foundation
import Observation

@MainActor
@Observable
final class AppState {
    // MARK: - Connection

    private(set) var connection: ServerConnection = .empty
    private(set) var isPaired: Bool = false
    private(set) var serverStatus: ServerStatus?
    private(set) var connectionError: String?
    /// Set when the server at the current address reported a different
    /// fingerprint than the one pinned at pairing time. Rendered as a
    /// banner; nothing clears it but re-pairing, because the honest
    /// answer to "something else is answering here" is not a toast.
    private(set) var identityWarning: String?
    /// Whether this phone's current network can connect to the paired
    /// server without a key, as that server just reported it.
    private(set) var keylessAvailableHere = false
    /// True when this connection presented no credential at all. Kept
    /// separate from "we have no token" because the app has to be able
    /// to tell a deliberate keyless connection from a lost one — and
    /// because a keyless caller never has admin rights server-side, so
    /// nothing admin-shaped should be offered for one.
    private(set) var isKeyless = false

    // MARK: - Content

    private(set) var sessions: [ChatSession] = []
    private(set) var messages: [ChatMessage] = []
    private(set) var openSessionID: String?
    private(set) var availableModels: [BridgeModel] = []

    // MARK: - Transient UI state

    private(set) var isSending = false
    private(set) var isLoadingSessions = false
    /// The assistant text accumulated so far in the current stream. The
    /// chat view renders this as a live bubble; it is cleared when the
    /// real persisted message arrives, so the bubble never appears twice.
    private(set) var streamingText: String = ""
    var lastError: String?

    private let client = HyperLinkClient()
    private var streamTask: Task<Void, Never>?
    private static let connectionKey = "hyperlink.connection"
    private static let keylessKey = "hyperlink.connection.keyless"

    init() { restore() }

    // MARK: - Persistence

    private func restore() {
        guard
            let data = UserDefaults.standard.data(forKey: Self.connectionKey),
            let saved = try? JSONDecoder().decode(ServerConnection.self, from: data),
            saved.isConfigured
        else { return }
        let token = TokenStore.load()
        let wasKeyless = UserDefaults.standard.bool(forKey: Self.keylessKey)
        // A stored connection with no credential is only restorable if
        // it was keyless on purpose. Otherwise the token has been lost
        // — a keychain reset, a restore from backup — and coming up
        // "paired" would mean every request 401s with no explanation.
        guard token != nil || wasKeyless else { return }
        connection = saved
        isKeyless = wasKeyless
        isPaired = true
        Task {
            await client.configure(
                endpoints: saved.endpoints, token: token, keyless: wasKeyless
            )
        }
    }

    private func persist() {
        if let data = try? JSONEncoder().encode(connection) {
            UserDefaults.standard.set(data, forKey: Self.connectionKey)
        }
        UserDefaults.standard.set(isKeyless, forKey: Self.keylessKey)
    }

    // MARK: - Pairing

    /// Connect with a T2S key rather than a pairing code.
    ///
    /// Same end state as `pair`: a credential in the keychain, a ranked
    /// endpoint list, and `isPaired`. The difference is that there is no
    /// redeem step and no device record on the server — the key itself is
    /// the credential, so signing out here simply forgets it rather than
    /// revoking a device.
    func connect(address: String, key: String, deviceName: String) async -> Bool {
        connectionError = nil
        // Before the request: three addresses people reasonably type
        // cannot work, and iOS reports all three in words about its own
        // policy rather than about the address. Saying so here beats
        // sending a request that is guaranteed to fail.
        if let refusal = AddressCheck.advice(for: address).message {
            connectionError = refusal
            return false
        }
        let credential = key.trimmingCharacters(in: .whitespacesAndNewlines)
        do {
            let discovered = try await HyperLinkClient.connect(address: address, key: credential)
            var endpoints = [HyperLinkClient.normalize(address)]
            for endpoint in discovered.endpoints where !endpoints.contains(endpoint.url) {
                endpoints.append(endpoint.url)
            }
            connection = ServerConnection(
                endpoints: endpoints,
                serverName: discovered.serverName,
                t1Version: discovered.t1Version,
                deviceID: "",
                deviceName: deviceName,
                serverFingerprint: discovered.serverFingerprint
            )
            keylessAvailableHere = discovered.keylessAvailableHere
            isKeyless = false
            TokenStore.save(credential)
            persist()
            await client.configure(endpoints: endpoints, token: credential)
            isPaired = true
            await refreshAll()
            return true
        } catch {
            connectionError = FailureAdvice.explain(error, address: address)
            return false
        }
    }

    /// Connect with no credential, to a server in trusted-network mode.
    ///
    /// The server decides whether this is allowed — its operator has to
    /// have turned the mode on, and it classifies the connection as
    /// loopback, LAN or tailnet before answering. A public origin is
    /// refused whatever the configuration says, so this is not a way to
    /// reach a machine over the internet by leaving the key out.
    ///
    /// Nothing is written to the keychain here, because there is nothing
    /// to write. Signing out is forgetting an address.
    func connectKeyless(address: String, deviceName: String) async -> Bool {
        connectionError = nil
        if let refusal = AddressCheck.advice(for: address).message {
            connectionError = refusal
            return false
        }
        do {
            let discovered = try await HyperLinkClient.connectKeyless(address: address)
            var endpoints = [HyperLinkClient.normalize(address)]
            for endpoint in discovered.endpoints where !endpoints.contains(endpoint.url) {
                endpoints.append(endpoint.url)
            }
            connection = ServerConnection(
                endpoints: endpoints,
                serverName: discovered.serverName,
                t1Version: discovered.t1Version,
                deviceID: "",
                deviceName: deviceName,
                serverFingerprint: discovered.serverFingerprint
            )
            isKeyless = true
            keylessAvailableHere = discovered.keylessAvailableHere
            // No token is stored, and any token from a previous pairing
            // is cleared: leaving one behind would mean a "keyless"
            // connection quietly presenting somebody else's credential
            // the next time the flag was wrong.
            TokenStore.delete()
            persist()
            await client.configure(endpoints: endpoints, token: nil, keyless: true)
            isPaired = true
            await refreshAll()
            return true
        } catch {
            connectionError = FailureAdvice.explain(error, address: address)
            return false
        }
    }

    func pair(address: String, code: String, deviceName: String) async -> Bool {
        connectionError = nil
        if let refusal = AddressCheck.advice(for: address).message {
            connectionError = refusal
            return false
        }
        do {
            let (redeemed, discovered) = try await HyperLinkClient.pair(
                address: address, code: code, deviceName: deviceName
            )
            // The typed address goes first: it is known to work right
            // now. The server's ranked list follows, so the app can
            // still reach the PC after the phone leaves the network the
            // pairing happened on.
            var endpoints = [HyperLinkClient.normalize(address)]
            for endpoint in discovered?.endpoints ?? [] where !endpoints.contains(endpoint.url) {
                endpoints.append(endpoint.url)
            }
            keylessAvailableHere = discovered?.keylessAvailableHere ?? false
            connection = ServerConnection(
                endpoints: endpoints,
                serverName: redeemed.serverName,
                t1Version: redeemed.t1Version,
                deviceID: redeemed.deviceID,
                deviceName: redeemed.name,
                // Trust on first use, and this is that first use:
                // someone is standing at the PC reading a six-character
                // code off its screen, which is the one moment in this
                // app's life with independent evidence of which machine
                // it is talking to.
                serverFingerprint: discovered?.serverFingerprint ?? ""
            )
            isKeyless = false
            TokenStore.save(redeemed.deviceToken)
            persist()
            await client.configure(endpoints: endpoints, token: redeemed.deviceToken)
            isPaired = true
            await refreshAll()
            return true
        } catch {
            connectionError = FailureAdvice.explain(error, address: address)
            return false
        }
    }

    /// Sign out. The server-side revoke is attempted but not required:
    /// a phone being wiped on a train has no route to the PC, and
    /// refusing to clear the local token in that case would leave the
    /// credential on the device — the opposite of what was asked for.
    func unpair() async {
        let deviceID = connection.deviceID
        if !deviceID.isEmpty {
            try? await client.unpairSelf(deviceID: deviceID)
        }
        TokenStore.delete()
        // The admin credential is scoped to this server's fingerprint,
        // so signing out of the server is the moment it stops being
        // something this phone should be holding.
        AdminCredentialStore.delete(fingerprint: connection.serverFingerprint)
        UserDefaults.standard.removeObject(forKey: Self.connectionKey)
        await client.configure(endpoints: [], token: nil)
        connection = .empty
        isPaired = false
        identityWarning = nil
        keylessAvailableHere = false
        isKeyless = false
        UserDefaults.standard.removeObject(forKey: Self.keylessKey)
        sessions = []
        messages = []
        openSessionID = nil
        availableModels = []
        serverStatus = nil
    }

    /// Called when any request comes back with a revoked/invalid token.
    private func handle(_ error: Error) {
        if let hyperlinkError = error as? HyperLinkError {
            lastError = hyperlinkError.errorDescription
            if hyperlinkError.requiresRepairing {
                Task { await unpair() }
            }
        } else {
            lastError = error.localizedDescription
        }
    }

    // MARK: - Refresh

    func refreshAll() async {
        async let status: Void = refreshStatus()
        async let list: Void = refreshSessions()
        async let models: Void = refreshModels()
        async let identity: Void = verifyIdentity()
        _ = await (status, list, models, identity)
    }

    /// Re-check that the address we reached is still the machine we
    /// paired with, and note whether this network can go keyless.
    ///
    /// Run on every refresh rather than only at launch, because the
    /// interesting case is the phone moving: home wifi, a friend's wifi,
    /// a tailnet, back again — each a different route to what is
    /// supposed to be the same PC, and one of them may not be.
    ///
    /// A failure to *reach* the server is not an identity problem and
    /// must not raise one. "Something else is answering here" and "the
    /// PC is asleep" are different, and only the first deserves a
    /// banner nothing but re-pairing clears.
    func verifyIdentity() async {
        guard isPaired, let reply = try? await client.endpoints() else { return }
        keylessAvailableHere = reply.keylessAvailableHere

        let verdict = ServerIdentity.check(
            reported: reply.serverFingerprint, pinned: connection.serverFingerprint
        )
        identityWarning = verdict.message
        if case .mismatch = verdict { return }

        // Fill in a blank pin — a pairing made before the server had a
        // fingerprint — without ever overwriting one that exists.
        let pinned = ServerIdentity.pinning(
            current: connection.serverFingerprint, reported: reply.serverFingerprint
        )
        if pinned != connection.serverFingerprint {
            connection.serverFingerprint = pinned
            persist()
        }
    }

    /// Other machines on the paired server's tailnet.
    ///
    /// Candidates, not servers. Nothing in the returned list has been
    /// authenticated and none of the names in it means anything — a
    /// person picks one and pairs with it, which is what proves what it
    /// is. Admin-only server-side, so an ordinary phone gets a 403 and
    /// the caller shows nothing rather than an error: not being an
    /// admin is the normal case, not a fault.
    func discoverPeers() async -> [DiscoveredPeer] {
        guard isPaired, let reply = try? await client.peers() else { return [] }
        return reply.peers.filter { $0.reachable }
    }

    func refreshStatus() async {
        do {
            serverStatus = try await client.status()
            connectionError = nil
        } catch {
            connectionError = (error as? HyperLinkError)?.errorDescription ?? error.localizedDescription
        }
    }

    func refreshSessions() async {
        isLoadingSessions = true
        defer { isLoadingSessions = false }
        do {
            sessions = try await client.sessions()
        } catch {
            handle(error)
        }
    }

    func refreshModels() async {
        // A server with no LM Studio configured is a normal state, not
        // an error: the model picker just shows nothing to pick.
        availableModels = (try? await client.bridgeModels())?.models ?? []
    }

    // MARK: - Sessions

    func newSession(modelID: String = "") async -> ChatSession? {
        do {
            let session = try await client.createSession(modelID: modelID)
            sessions.insert(session, at: 0)
            await open(session.sessionID)
            return session
        } catch {
            handle(error)
            return nil
        }
    }

    func open(_ sessionID: String) async {
        openSessionID = sessionID
        messages = []
        streamingText = ""
        do {
            messages = try await client.messages(in: sessionID)
        } catch {
            handle(error)
        }
    }

    func delete(_ sessionID: String) async {
        do {
            try await client.deleteSession(sessionID)
            sessions.removeAll { $0.sessionID == sessionID }
            if openSessionID == sessionID {
                openSessionID = nil
                messages = []
            }
        } catch {
            handle(error)
        }
    }

    // MARK: - Sending

    /// Send a turn and stream the reply.
    ///
    /// The user's bubble is shown immediately from a locally-built
    /// message, then replaced by the persisted one when the `start`
    /// frame arrives. Matching on `seq == -1` is what keeps the two from
    /// both being on screen — the local placeholder is the only message
    /// that can have a negative sequence.
    func send(text: String, attachmentIDs: [String] = [], modelID: String? = nil) {
        guard let sessionID = openSessionID, !isSending else { return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty || !attachmentIDs.isEmpty else { return }

        // `isSending` flips here, synchronously, not inside the task.
        // Setting it in the task left a window where two quick taps both
        // passed the guard and sent the same message twice.
        isSending = true
        streamingText = ""
        lastError = nil
        messages.append(
            .local(role: "user", content: trimmed, sessionID: sessionID, attachments: attachmentIDs)
        )

        // The task is owned by the state, not by the view that started
        // it: a streamed answer must survive the user leaving the
        // screen, and `cancelStreaming()` needs something to cancel.
        streamTask = Task { [weak self] in
            await self?.runTurn(
                sessionID: sessionID, text: trimmed, attachmentIDs: attachmentIDs, modelID: modelID
            )
        }
    }

    private func runTurn(
        sessionID: String, text: String, attachmentIDs: [String], modelID: String?
    ) async {
        // Whatever happens below, the composer must come back. An early
        // return that leaves `isSending` true is a permanently stuck UI.
        defer {
            isSending = false
            streamTask = nil
        }
        do {
            let request = try await client.streamingChatRequest(
                sessionID: sessionID,
                content: text,
                attachmentIDs: attachmentIDs,
                modelID: modelID
            )
            for try await event in SSEStream.events(for: request) {
                switch event {
                case .start:
                    // The server has the message; drop the placeholder
                    // and take the authoritative copy on the next reload.
                    break
                case let .delta(piece):
                    streamingText += piece
                case .done:
                    streamingText = ""
                    messages = (try? await client.messages(in: sessionID)) ?? messages
                    await refreshSessionSummary()
                case let .failed(_, message):
                    lastError = message
                    // Whatever streamed before the failure was persisted
                    // server-side, so reload rather than keeping a
                    // half-message that only exists on the phone.
                    streamingText = ""
                    messages = (try? await client.messages(in: sessionID)) ?? messages
                }
            }
            // A stream that ends without a `done` frame (the PC slept,
            // the tunnel dropped) still has a persisted partial reply.
            if !streamingText.isEmpty {
                streamingText = ""
                messages = (try? await client.messages(in: sessionID)) ?? messages
            }
        } catch {
            if !(error is CancellationError) {
                handle(error)
            }
            // Drop the optimistic bubble; the authoritative history is
            // whatever the server has, and it is reloaded below.
            messages.removeAll { $0.seq == -1 }
            messages = (try? await client.messages(in: sessionID)) ?? messages
        }
    }

    private func refreshSessionSummary() async {
        guard let updated = try? await client.sessions() else { return }
        sessions = updated
    }

    /// Stop a streamed answer.
    ///
    /// The server persists whatever streamed before the disconnect, so
    /// the history is reloaded rather than trusting what is on screen —
    /// a partial answer that exists only on the phone would vanish on
    /// the next refresh and look like data loss.
    func cancelStreaming() {
        streamTask?.cancel()
        streamTask = nil
        isSending = false
        streamingText = ""
        guard let sessionID = openSessionID else { return }
        Task { messages = (try? await client.messages(in: sessionID)) ?? messages }
    }

    /// Switch which model answers in this session, from here on.
    func setModel(_ modelID: String, for sessionID: String) async {
        do {
            let updated = try await client.setModel(modelID, for: sessionID)
            if let index = sessions.firstIndex(where: { $0.sessionID == sessionID }) {
                sessions[index] = updated
            }
        } catch {
            handle(error)
        }
    }

    // MARK: - Attachments

    func upload(data: Data, filename: String, contentType: String) async -> Attachment? {
        guard let sessionID = openSessionID else { return nil }
        do {
            return try await client.upload(
                data: data, filename: filename, contentType: contentType, sessionID: sessionID
            )
        } catch {
            handle(error)
            return nil
        }
    }

    func attachmentData(_ fileID: String) async -> Data? {
        try? await client.attachmentData(fileID)
    }

    // MARK: - Model downloads

    func resolveModel(pageURL: String, fileURL: String, prefer: String) async throws -> ResolvedModel {
        try await client.resolveModel(pageURL: pageURL, fileURL: fileURL, prefer: prefer)
    }
}
