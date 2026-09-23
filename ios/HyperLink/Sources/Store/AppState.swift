//  AppState.swift
//  The app's single source of truth.
//
//  One `@Observable` object on the main actor, holding the connection,
//  the session list, and the messages of whichever session is open.
//  Everything that talks to the network goes through here rather than
//  from a view, for one reason: a streamed answer arrives over several
//  seconds and must survive the user scrolling, rotating, or switching
//  tabs, none of which a view's lifetime does.

import AppIntents
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

    /// Every machine this phone is paired with, most recently used
    /// first. Up to `SavedServers.maxServers` of them.
    private(set) var savedServers: [SavedServer] = []

    /// The id of the server currently connected, or "" when none is.
    private(set) var currentServerID: String = ""

    // MARK: - Content

    private(set) var sessions: [ChatSession] = []
    private(set) var messages: [ChatMessage] = []
    private(set) var openSessionID: String?
    /// Every model the server can offer, from every source it has —
    /// the registry, the LM Studio bridge, and the GGUFs on disk.
    ///
    /// This replaced `availableModels`, which held `/bridge/lmstudio/models`
    /// and nothing else: one source of three, and the only one that
    /// needs a second application to be running. A machine with forty
    /// models in ~/.hypernix/models showed an empty picker.
    private(set) var catalogue: ModelCatalogue = .empty

    /// How long the server and its machine have been up. Nil until the
    /// first refresh, and on a server too old to answer.
    private(set) var uptime: ServerUptime?
    /// What the server is running versus what is installed on it.
    /// Separate from `serverStatus`, whose version is whatever the
    /// process imported at start and so cannot report its own
    /// staleness.
    private(set) var serverVersion: ServerVersion?

    /// What the HyperNix runner is running, if anything.
    ///
    /// `.unknown` rather than nil while it has not been asked, so a
    /// view can tell "no model loaded" from "this server has no runner"
    /// — the first is a button to press and the second is not.
    private(set) var runner: RunnerStatus = .unknown
    /// False on a server too old to have `/runner/*`, or one where it is
    /// switched off. Nothing about the runner is offered when it is.
    private(set) var runnerAvailable = false
    /// True while a load or unload is in flight. A 70B coming off a
    /// spinning disk takes minutes, and a screen that does not say so
    /// looks broken.
    private(set) var runnerBusy = false
    /// What the last load or unload said when it refused.
    var runnerError: String?

    /// Reload one conversation's history from the server.
    ///
    /// What makes backgrounding survivable. The server keeps generating
    /// and persists the reply as it goes, so a phone that was suspended
    /// mid-answer does not need the socket it lost — it needs to ask
    /// again. See `BackgroundSession` for why that is the whole trick.
    func reload(sessionID: String) async {
        guard let refreshed = try? await client.messages(in: sessionID) else { return }
        if openSessionID == sessionID {
            messages = refreshed
            // A stream that was cut off mid-flight leaves this set, and
            // a live bubble next to the real persisted reply is the same
            // text twice.
            streamingText = ""
            isSending = false
        }
        sessions = (try? await client.sessions()) ?? sessions
    }

    // MARK: - Settings

    /// This person's settings and the bounds this server accepts.
    ///
    /// The bounds travel with the values so the app never carries its
    /// own copy of a list the server owns — an effort level the phone
    /// offers and the server rejects is a settings screen that cannot
    /// save.
    private(set) var settings: PreferencesEnvelope = .empty
    /// What could answer a message here, and what would right now.
    private(set) var backends: BackendList = .empty
    private(set) var memories: [MemoryItem] = []

    // MARK: - Transient UI state

    private(set) var isSending = false
    private(set) var isLoadingSessions = false
    /// The assistant text accumulated so far in the current stream. The
    /// chat view renders this as a live bubble; it is cleared when the
    /// real persisted message arrives, so the bubble never appears twice.
    private(set) var streamingText: String = ""
    /// The id the server gave this generation, when it sent one. Lets
    /// Stop name exactly what to stop — which matters with two devices
    /// open on one conversation, where "whatever is running here" would
    /// stop the other phone's answer.
    private var streamingGenerationID: String?
    var lastError: String?

    private let client = HyperLinkClient()
    private var streamTask: Task<Void, Never>?

    init() { restore() }

    // MARK: - Persistence
    //
    // The reading and writing themselves live in `PairingStore`, not
    // here, because this is no longer the only reader: an App Intent
    // runs in its own process and the CarPlay scene can connect while
    // the phone app has never been opened, and neither of those has an
    // `AppState` to ask. What stays here is what to *do* with a restored
    // pairing — which is this object's own state.

    private func restore() {
        savedServers = SavedServers.all()
        guard let pairing = PairingStore.load() else { return }
        currentServerID = pairing.id
        connection = pairing.connection
        isKeyless = pairing.keyless
        isPaired = true
        Task {
            await client.configure(
                endpoints: pairing.endpoints,
                token: pairing.token,
                keyless: pairing.keyless
            )
        }
    }

    /// Write the current connection to the saved-server list.
    ///
    /// *token* goes with it, rather than being written separately by the
    /// caller: each saved server keeps its credential under its own
    /// keychain account, so "save this token" is not a question that can
    /// be answered without knowing which server it belongs to. Passing
    /// nil on a non-keyless save leaves the existing credential alone,
    /// which is what a re-save of a known machine wants.
    private func persist(token: String? = nil) {
        PairingStore.save(connection: connection, keyless: isKeyless, token: token)
        savedServers = SavedServers.all()
        currentServerID = SavedServers.selected()?.id ?? ""
    }

    // MARK: - More than one server

    /// Switch to another paired machine.
    ///
    /// Everything on screen belongs to the server it came from —
    /// sessions, messages, the model list — so it is all cleared before
    /// the new one is loaded rather than left to be replaced piecemeal.
    /// A half-swapped view showing one machine's chats under another
    /// machine's name is worse than an empty one for the second it takes
    /// to fill.
    @discardableResult
    func switchTo(serverID: String) async -> Bool {
        guard serverID != currentServerID else { return true }
        guard SavedServers.select(id: serverID),
              let pairing = PairingStore.load()
        else {
            savedServers = SavedServers.all()
            return false
        }

        // Anything in flight belongs to the machine being left.
        streamTask?.cancel()
        streamTask = nil
        isSending = false
        streamingText = ""
        sessions = []
        messages = []
        openSessionID = nil
        catalogue = .empty
        uptime = nil
        settings = .empty
        backends = .empty
        memories = []
        runner = .unknown
        runnerAvailable = false
        runnerError = nil
        serverStatus = nil
        identityWarning = nil
        lastError = nil

        currentServerID = pairing.id
        connection = pairing.connection
        isKeyless = pairing.keyless
        isPaired = true
        savedServers = SavedServers.all()
        await client.configure(
            endpoints: pairing.endpoints, token: pairing.token, keyless: pairing.keyless
        )
        await refreshAll()
        return true
    }

    /// Follow one machine to a different port.
    ///
    /// The port is the only part of a pairing that changes for ordinary
    /// reasons — the server restarted somewhere else, or moved behind a
    /// different forward — and the only way to follow it used to be to
    /// pair again, which means finding a key again and throwing away
    /// the fingerprint that says this is the same machine.
    ///
    /// Editing the machine you are currently talking to has to
    /// reconnect, or the app goes on using the old port and the change
    /// looks like it did nothing.
    @discardableResult
    func setPort(serverID: String, port: Int) async -> Bool {
        guard SavedServers.setPort(id: serverID, port: port) else { return false }
        savedServers = SavedServers.all()
        guard serverID == currentServerID else { return true }
        guard let pairing = PairingStore.load() else { return true }
        connection = pairing.connection
        await client.configure(
            endpoints: pairing.endpoints, token: pairing.token, keyless: pairing.keyless
        )
        await refreshAll()
        return true
    }

    /// Forget one machine without signing out of the others.
    ///
    /// Forgetting the one currently connected falls back to whichever
    /// was used most recently, or to the pairing screen when that was
    /// the last one.
    func forget(serverID: String) async {
        let wasCurrent = serverID == currentServerID
        SavedServers.remove(id: serverID)
        savedServers = SavedServers.all()
        guard wasCurrent else { return }
        if let next = SavedServers.selected() {
            currentServerID = ""
            await switchTo(serverID: next.id)
        } else {
            currentServerID = ""
            connection = .empty
            isPaired = false
            isKeyless = false
            sessions = []
            messages = []
            openSessionID = nil
            catalogue = .empty
            uptime = nil
            settings = .empty
            backends = .empty
            memories = []
            runner = .unknown
            runnerAvailable = false
            runnerError = nil
            serverStatus = nil
            await client.configure(endpoints: [], token: nil)
        }
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
            persist(token: credential)
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
            // with this machine is cleared by `persist` — leaving one
            // behind would mean a "keyless" connection quietly
            // presenting a credential the next time the flag was wrong.
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
            persist(token: redeemed.deviceToken)
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
        // The token goes with the record: `PairingStore.clear()` deletes
        // this server's keychain entry and leaves the other saved
        // servers' alone. A bare `TokenStore.delete()` here would clear
        // the legacy account, which after the migration belongs to
        // whichever server was paired first — not necessarily this one.
        //
        // The admin credential is scoped to this server's fingerprint,
        // so signing out of the server is the moment it stops being
        // something this phone should be holding.
        AdminCredentialStore.delete(fingerprint: connection.serverFingerprint)
        PairingStore.clear()
        savedServers = SavedServers.all()
        await client.configure(endpoints: [], token: nil)
        connection = .empty
        isPaired = false
        identityWarning = nil
        keylessAvailableHere = false
        isKeyless = false
        sessions = []
        messages = []
        openSessionID = nil
        catalogue = .empty
        uptime = nil
        settings = .empty
        backends = .empty
        memories = []
        runner = .unknown
        runnerAvailable = false
        runnerError = nil
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
        async let clock: Void = refreshUptime()
        async let engine: Void = refreshRunner()
        async let mine: Void = refreshSettings()
        async let built: Void = refreshVersion()
        _ = await (status, list, models, identity, clock, engine, mine, built)
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
            // Siri learns chat names from ChatEntityQuery; this is what
            // tells it the list changed (see HyperLinkIntents.swift).
            HyperLinkShortcuts.updateAppShortcutParameters()
        } catch {
            handle(error)
        }
    }

    func refreshModels() async {
        // One request, not two. The catalogue already contains what the
        // LM Studio bridge would have returned, marked as coming from
        // it, so asking the bridge separately was a second round trip
        // for a subset of the answer.
        //
        // A server with nothing to offer is a normal state, not an
        // error: the picker shows the reason instead of a red line.
        if let merged = try? await client.modelCatalogue() {
            catalogue = merged
            HyperLinkShortcuts.updateAppShortcutParameters()
        }
    }

    /// How long the server has been up.
    ///
    /// Silent on failure by design: this is an ornament next to the
    /// server name, and a server too old to have the endpoint 404s.
    /// Turning that into a visible error would put a red line on screen
    /// about a feature nobody asked for.
    func refreshUptime() async {
        uptime = try? await client.uptime()
    }

    /// What is running here, and whether it is what is installed.
    ///
    /// `try?` for the same reason as uptime: a server too old to have
    /// /version is not an error worth a red line, it just leaves the
    /// extra detail off the Server page. The version already shown
    /// there comes from /status and does not depend on this.
    func refreshVersion() async {
        let found = try? await client.version()
        serverVersion = found
        guard let found else { return }

        // The version the *process* is running, not the one pip has on
        // disk. A server upgraded and not restarted is still answering
        // with the old code, and it is the old code the app has to be
        // compatible with — showing the installed number would say the
        // upgrade had taken effect when it has not.
        let running = found.hypernix
        guard !running.isEmpty else { return }
        guard connection.hypernixVersion != running
                || connection.hypernixStale != found.stale else { return }

        connection.hypernixVersion = running
        connection.hypernixStale = found.stale
        // Persisted so the servers list shows it before the next
        // connection has had a chance to ask — the list is the screen
        // people check to see which machine is on which version, and it
        // is usually opened while not connected to most of them.
        SavedServers.remember(
            connection: connection, keyless: isKeyless, token: nil
        )
        savedServers = SavedServers.all()
    }

    /// What the server is running, and how to update it.
    ///
    /// Throws rather than returning nil: a server too old to have the
    /// endpoint is exactly the server somebody opened this screen to
    /// update, and silently showing nothing would be the least helpful
    /// possible response to that.
    func upgradeAdvice() async throws -> UpgradeAdvice {
        try await client.upgradeAdvice()
    }

    /// One hardware sample.
    ///
    /// Throws rather than returning nil, unlike the other refreshes:
    /// this one is opened deliberately from a menu, and "you are not an
    /// admin on this server" is the answer to show rather than an empty
    /// screen. The distinction is the whole reason it is not cached in
    /// `AppState` — a reading is true for a second, and a stale one on
    /// screen is worse than a spinner.
    func hardware() async throws -> ServerHardware {
        try await client.hardware()
    }

    // MARK: - Settings

    /// Load the settings and what can answer.
    ///
    /// Silent on failure, like the other background refreshes: a server
    /// too old for `/hyperlink/preferences` 404s, and a red line about
    /// a feature nobody asked for is worse than the defaults.
    func refreshSettings() async {
        if let envelope = try? await client.preferences() {
            settings = envelope
        }
        if let list = try? await client.backends() {
            backends = list
        }
    }

    /// Save some settings. Returns the notes the server sent back.
    ///
    /// The notes are the point of returning anything: a context maximum
    /// of four million is lowered, and a settings screen that did not
    /// say so would be one nobody could trust afterwards.
    @discardableResult
    func saveSettings(_ patch: PreferencesPatch) async -> [String] {
        do {
            settings = try await client.savePreferences(patch)
            // What answers can change with them — switching backend, or
            // naming a backup model, changes what a message would reach.
            if let list = try? await client.backends() { backends = list }
            return settings.notes
        } catch {
            handle(error)
            return []
        }
    }

    @discardableResult
    func resetSettings() async -> Bool {
        do {
            settings = try await client.resetPreferences()
            return true
        } catch {
            handle(error)
            return false
        }
    }

    // MARK: - Memory

    func refreshMemories() async {
        memories = (try? await client.memories())?.memories ?? memories
    }

    @discardableResult
    func remember(_ content: String) async -> Bool {
        let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        do {
            try await client.rememberFact(trimmed)
            await refreshMemories()
            return true
        } catch {
            handle(error)
            return false
        }
    }

    @discardableResult
    func updateMemory(_ memoryID: String, content: String? = nil, pinned: Bool? = nil) async -> Bool {
        do {
            try await client.editMemory(memoryID, content: content, pinned: pinned)
            await refreshMemories()
            return true
        } catch {
            handle(error)
            return false
        }
    }

    @discardableResult
    func forget(_ memoryID: String) async -> Bool {
        do {
            try await client.forgetMemory(memoryID)
            // Removed locally first so the row goes immediately, then
            // reconciled — a list that waits for a round trip before a
            // delete looks stuck.
            memories.removeAll { $0.memoryID == memoryID }
            await refreshMemories()
            return true
        } catch {
            handle(error)
            return false
        }
    }

    // MARK: - Organising memory

    private(set) var memoryCategories: [MemoryCategory] = []

    func refreshMemoryCategories() async {
        memoryCategories = (try? await client.memoryCategories()) ?? memoryCategories
    }

    /// File loose and model-written memories under topics. With
    /// `dryRun` it only says what would move.
    func organiseMemories(dryRun: Bool = false) async -> MemoryOrganiseResult? {
        do {
            let result = try await client.organiseMemories(dryRun: dryRun)
            if !dryRun {
                await refreshMemories()
                await refreshMemoryCategories()
            }
            return result
        } catch {
            handle(error)
            return nil
        }
    }

    @discardableResult
    func renameMemoryCategory(from old: String, to new: String) async -> Bool {
        let trimmed = new.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed != old else { return false }
        do {
            try await client.renameMemoryCategory(from: old, to: trimmed)
            await refreshMemories()
            await refreshMemoryCategories()
            return true
        } catch {
            handle(error)
            return false
        }
    }

    @discardableResult
    func moveMemory(_ memoryID: String, to category: String) async -> Bool {
        do {
            try await client.moveMemory(memoryID, to: category)
            await refreshMemories()
            await refreshMemoryCategories()
            return true
        } catch {
            handle(error)
            return false
        }
    }

    // MARK: - Titles and compression

    /// Ask the model to name this chat again.
    @discardableResult
    func retitle(_ sessionID: String) async -> Bool {
        do {
            _ = try await client.retitle(sessionID)
            await refreshSessionSummary()
            return true
        } catch {
            handle(error)
            return false
        }
    }

    /// Summarise the older part of a conversation now. Every message
    /// stays in the transcript; the model is sent the summary instead.
    func compress(_ sessionID: String) async -> CompactResult? {
        do {
            let result = try await client.compress(sessionID)
            messages = (try? await client.messages(in: sessionID)) ?? messages
            return result
        } catch {
            handle(error)
            return nil
        }
    }

    // MARK: - Images

    /// Whether the model a chat would use can see a photo: true, false,
    /// or nil when nothing says either way. The session's model, or
    /// whatever is loaded when the session has not picked one.
    func modelSupportsImages(_ modelID: String) -> Bool? {
        let wanted = modelID.isEmpty
            ? catalogue.models.first(where: { $0.loaded })?.modelID ?? ""
            : modelID
        guard !wanted.isEmpty else { return nil }
        return catalogue.models.first(where: { $0.modelID == wanted })?.supportsImages
    }

    // MARK: - Shell

    /// Nil until asked, and on a server too old to have the route.
    private(set) var shellStatus: ShellStatus?

    func refreshShellStatus() async {
        shellStatus = try? await client.shellStatus()
    }

    func runShell(_ command: String, cwd: String? = nil) async -> ShellResult? {
        let trimmed = command.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        do {
            return try await client.runShell(trimmed, cwd: cwd)
        } catch {
            handle(error)
            return nil
        }
    }

    // MARK: - Private chats

    /// Chats hidden behind Face ID. Kept on this phone only — hiding is
    /// about who can pick the phone up, not about the server.
    let privateChats = PrivateChats()

    // MARK: - The runner

    /// Ask what the server is running.
    ///
    /// A 404 or a 403 here is not an error worth showing: a server too
    /// old for `/runner/*`, or a caller without the rights to see it,
    /// is a normal state and the answer is to offer nothing rather than
    /// to put a red line on screen.
    func refreshRunner() async {
        if let status = try? await client.runnerStatus() {
            runner = status
            runnerAvailable = true
        } else {
            runner = .unknown
            runnerAvailable = false
        }
    }

    /// Where this model's layers would go. Changes nothing.
    func planLoad(
        modelID: String, gpuLayers: Int?, backend: String,
        contextLength: Int?, totalLayers: Int?
    ) async -> RunnerPlan? {
        try? await client.runnerPlan(RunnerLoadRequest(
            model_id: modelID, gpu_layers: gpuLayers, backend: backend,
            context_length: contextLength, total_layers: totalLayers
        ))
    }

    /// Load a model on the server, replacing whatever was running.
    ///
    /// Unlike the read paths, a failure here *is* shown. Somebody just
    /// asked for a specific thing to happen to a shared machine, and
    /// "there is no built llama.cpp" or "this model does not fit" are
    /// both things they can act on.
    @discardableResult
    func loadModel(
        modelID: String, gpuLayers: Int? = nil, backend: String = "auto",
        contextLength: Int? = nil, totalLayers: Int? = nil
    ) async -> Bool {
        runnerBusy = true
        runnerError = nil
        defer { runnerBusy = false }
        do {
            runner = try await client.runnerLoad(RunnerLoadRequest(
                model_id: modelID, gpu_layers: gpuLayers, backend: backend,
                context_length: contextLength, total_layers: totalLayers
            ))
            runnerAvailable = true
            // The catalogue's `loaded` flags are now stale — the model
            // that was answering a moment ago is not the one answering
            // now, and a picker still showing the old green dot is how
            // somebody talks to the wrong model.
            await refreshModels()
            return true
        } catch {
            runnerError = (error as? HyperLinkError)?.errorDescription
                ?? error.localizedDescription
            return false
        }
    }

    /// Stop serving. Unloading nothing is a success.
    @discardableResult
    func unloadModel() async -> Bool {
        runnerBusy = true
        runnerError = nil
        defer { runnerBusy = false }
        do {
            runner = try await client.runnerUnload()
            await refreshModels()
            return true
        } catch {
            runnerError = (error as? HyperLinkError)?.errorDescription
                ?? error.localizedDescription
            return false
        }
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

    /// Rename a conversation.
    ///
    /// Optimistic: the row changes as the sheet closes and is put back
    /// if the server refuses. A rename that waits for a round trip feels
    /// broken on a phone that is four hops and a Tailscale relay away
    /// from the machine holding the chat.
    @discardableResult
    func rename(_ sessionID: String, to title: String) async -> Bool {
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        let index = sessions.firstIndex { $0.sessionID == sessionID }
        let previous = index.map { sessions[$0].title }
        if let index { sessions[index].title = trimmed }
        do {
            let updated = try await client.rename(sessionID, to: trimmed)
            if let index { sessions[index] = updated }
            return true
        } catch {
            if let index, let previous { sessions[index].title = previous }
            handle(error)
            return false
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

    /// Ask the model to answer the thread's last message again.
    ///
    /// No optimistic bubble: the message being answered is already on
    /// screen. Used after an edit and by resend, both of which leave the
    /// thread ending on your message.
    func regenerateLast(modelID: String? = nil) {
        guard let sessionID = openSessionID, !isSending else { return }
        isSending = true
        streamingText = ""
        lastError = nil
        streamTask = Task { [weak self] in
            await self?.runTurn(
                sessionID: sessionID, text: "", attachmentIDs: [], modelID: modelID,
                regenerate: true
            )
        }
    }

    /// Ask one of your messages again, as it is.
    ///
    /// An edit with the same text, then a regenerate: the edit removes
    /// everything after the message — the old reply included — so the
    /// new answer replaces it rather than stacking under it. Returns the
    /// number of later messages removed, or nil when refused.
    @discardableResult
    func resendMessage(_ message: ChatMessage) async -> Int? {
        await editMessage(message.messageID, to: message.content)
    }

    private func runTurn(
        sessionID: String, text: String, attachmentIDs: [String], modelID: String?,
        regenerate: Bool = false
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
                modelID: modelID,
                regenerate: regenerate
            )
            for try await event in SSEStream.events(for: request) {
                switch event {
                case let .start(_, _, generationID):
                    // The server has the message; drop the placeholder
                    // and take the authoritative copy on the next
                    // reload. Keep the generation id, which is what
                    // Stop names.
                    streamingGenerationID = generationID.isEmpty ? nil : generationID
                case let .delta(piece):
                    streamingText += piece
                case .done:
                    streamingGenerationID = nil
                    streamingText = ""
                    messages = (try? await client.messages(in: sessionID)) ?? messages
                    await refreshSessionSummary()
                    // The model may have remembered something during
                    // this turn. The Memories screen loads once when it
                    // appears, so one kept alive in another tab would go
                    // on showing what it knew before this conversation.
                    if settings.preferences.autoMemory {
                        await refreshMemories()
                    }
                case .title:
                    // The server named a new chat after its first reply.
                    // The list carries titles, so refresh it rather than
                    // patching one entry and drifting from the server.
                    await refreshSessionSummary()
                case .compacted:
                    // The summary is a message in the thread; the reload
                    // at `done` has already brought it in.
                    break
                case let .failed(_, message):
                    streamingGenerationID = nil
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

    /// Stop a streamed answer — on the server as well as here.
    ///
    /// Cancelling the read task is what makes the button feel instant,
    /// and on its own it is half the job: nothing told the server, so
    /// the model finished the whole answer into a socket nobody was
    /// reading. On a shared machine that is somebody else's GPU for a
    /// minute and a half; on a metered deployment it is a bill for
    /// output the person explicitly asked not to have.
    ///
    /// Order matters. The local cancel goes first because it is
    /// instant and cannot fail, so the UI responds now rather than
    /// after a round trip; the server call follows in its own task and
    /// its failure is not shown — a Stop that reached the server late,
    /// or reached a server too old to have the endpoint, still stopped
    /// the thing the person was looking at.
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
        let generationID = streamingGenerationID
        streamingGenerationID = nil
        Task {
            // `_ =` rather than a bare `try?`: the call is
            // @discardableResult, but `try?` wraps it in an Optional and
            // an unused Optional is a warning in its own right.
            _ = try? await client.stopGeneration(
                sessionID: sessionID, generationID: generationID
            )
            messages = (try? await client.messages(in: sessionID)) ?? messages
        }
    }

    // MARK: - Edit mode

    /// Rewrite one of your own messages.
    ///
    /// Everything after it goes, because everything after it was a
    /// reply to the old text. The caller is expected to have said so
    /// first — `removedCount` on the result is what it costs, and
    /// `editWouldRemove` answers the same question before anything
    /// happens.
    ///
    /// Returns the number of messages dropped, or nil when the edit was
    /// refused.
    @discardableResult
    func editMessage(_ messageID: String, to content: String) async -> Int? {
        guard let sessionID = openSessionID else { return nil }
        do {
            let result = try await client.editMessage(
                sessionID: sessionID, messageID: messageID, content: content
            )
            // Reloaded rather than patched in place: the server just
            // deleted an unknown number of rows, and reconstructing that
            // locally is how a phone ends up showing a conversation the
            // server does not have.
            messages = (try? await client.messages(in: sessionID)) ?? messages
            await refreshSessionSummary()
            // An edit is a question asked differently, so it is asked.
            // It used to stop here, leaving the thread ending on an
            // unanswered message and the person wondering whether the
            // edit had worked.
            regenerateLast()
            return result.removedCount
        } catch {
            handle(error)
            return nil
        }
    }

    /// How many messages an edit at *messageID* would remove.
    ///
    /// Computed from what is on screen rather than asked of the server:
    /// it is a confirmation prompt, it has to be instant, and being off
    /// by one because a message arrived mid-prompt is not worth a round
    /// trip.
    func editWouldRemove(_ messageID: String) -> Int {
        guard let index = messages.firstIndex(where: { $0.messageID == messageID })
        else { return 0 }
        return messages.count - index - 1
    }

    /// Remove one message, leaving the rest of the conversation.
    ///
    /// Not a truncation, unlike an edit: deleting is usually about
    /// removing something that should not be stored — a pasted key, a
    /// name — and taking the thread with it would make people keep the
    /// secret rather than lose the conversation.
    @discardableResult
    func deleteMessage(_ messageID: String) async -> Bool {
        guard let sessionID = openSessionID else { return false }
        do {
            try await client.deleteMessage(sessionID: sessionID, messageID: messageID)
            messages.removeAll { $0.messageID == messageID }
            return true
        } catch {
            handle(error)
            return false
        }
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
