//  HyperLinkClient.swift
//  The one place that knows how to talk to a HyperNix T1 API server.
//
//  Two things here are not boilerplate and are worth reading before
//  changing anything:
//
//  1. **Endpoint failover.** A home server has several addresses — a LAN
//     IP that is fast at home and dead everywhere else, and a Tailscale
//     name that works anywhere and is a little slower. The phone cannot
//     know which network it is on (and asking iOS is unreliable and
//     racy), so the client simply tries the addresses in the order the
//     server ranked them and keeps the first that answers, re-testing
//     from the top when the current one fails. That is why `baseURL` is
//     a computed property over a *list* rather than a stored string.
//
//  2. **Errors are typed from the server's own envelope.** Every T1
//     failure carries a stable `code`; surfacing "MODEL_UNAVAILABLE"
//     as `.serverError(code:message:)` is what lets the chat view say
//     "your PC has no model loaded" instead of "something went wrong".

import Foundation

enum HyperLinkError: LocalizedError, Sendable {
    case notConfigured
    case noReachableEndpoint([String])
    case unauthorized(String)
    case serverError(code: String, message: String, status: Int)
    case transport(String)
    case decoding(String)

    var errorDescription: String? {
        switch self {
        case .notConfigured:
            return "No server paired yet."
        case let .noReachableEndpoint(tried):
            return tried.isEmpty
                ? "No server address to try."
                : "Could not reach your PC at any known address (\(tried.joined(separator: ", "))). "
                    + "Check it is awake, and that Tailscale is on if you are away from home."
        case let .unauthorized(message):
            return message
        case let .serverError(_, message, _):
            return message
        case let .transport(message):
            return message
        case let .decoding(message):
            return "The server sent something this app could not read: \(message)"
        }
    }

    /// True when re-pairing is the only way forward — the app clears its
    /// stored token and returns to the pairing screen on this.
    var requiresRepairing: Bool {
        if case let .serverError(code, _, status) = self {
            return status == 401 && (code.hasPrefix("AUTH_") || code == "AUTH_REVOKED_KEY")
        }
        if case .unauthorized = self { return true }
        return false
    }
}

/// Immutable connection settings. Stored in UserDefaults; the token
/// itself lives in the Keychain and is injected at use.
struct ServerConnection: Codable, Equatable, Sendable {
    var endpoints: [String]
    var serverName: String
    var t1Version: String
    var deviceID: String
    var deviceName: String
    /// The server's identity, pinned when this pairing was made. Not a
    /// secret — it is a public identifier the server hands to any
    /// authenticated caller, like a certificate fingerprint — so it
    /// lives here with the rest of the connection rather than in the
    /// keychain. See `ServerIdentity`.
    var serverFingerprint: String = ""

    static let empty = ServerConnection(
        endpoints: [], serverName: "", t1Version: "", deviceID: "", deviceName: "",
        serverFingerprint: ""
    )

    init(
        endpoints: [String], serverName: String, t1Version: String,
        deviceID: String, deviceName: String, serverFingerprint: String = ""
    ) {
        self.endpoints = endpoints
        self.serverName = serverName
        self.t1Version = t1Version
        self.deviceID = deviceID
        self.deviceName = deviceName
        self.serverFingerprint = serverFingerprint
    }

    /// Written out by hand because the synthesised one would not do
    /// this. A property default is used by the *memberwise* initialiser,
    /// not by the decoder: the synthesised `init(from:)` calls
    /// `decode(_:forKey:)` and throws when the key is absent, defaulted
    /// property or not.
    ///
    /// The stored record on every existing install was written before
    /// `serverFingerprint` existed. With the synthesised decoder,
    /// `restore()` would throw, its `try?` would swallow it, and the app
    /// would come up unpaired — every user silently signed out by an
    /// update, with nothing in any log to say why.
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        endpoints = try container.decodeIfPresent([String].self, forKey: .endpoints) ?? []
        serverName = try container.decodeIfPresent(String.self, forKey: .serverName) ?? ""
        t1Version = try container.decodeIfPresent(String.self, forKey: .t1Version) ?? ""
        deviceID = try container.decodeIfPresent(String.self, forKey: .deviceID) ?? ""
        deviceName = try container.decodeIfPresent(String.self, forKey: .deviceName) ?? ""
        serverFingerprint = try container.decodeIfPresent(
            String.self, forKey: .serverFingerprint
        ) ?? ""
    }

    /// Enough of a record to reconnect with.
    ///
    /// An address is the requirement; a `deviceID` is not. Connecting
    /// with a T2S key produces no device record on the server — the key
    /// *is* the credential — so it leaves `deviceID` empty, and
    /// requiring one here meant a key-based connection failed
    /// `restore()` and the app came up signed out after every restart.
    /// Whether there is a credential is `TokenStore`'s question, and
    /// `restore()` asks it separately.
    var isConfigured: Bool { !endpoints.isEmpty }
}

actor HyperLinkClient {
    private var endpoints: [String]
    private var preferredIndex: Int = 0
    private var token: String?
    /// Connected to a server in trusted-network mode with no credential.
    ///
    /// A separate flag rather than "send the token if we happen to have
    /// one": that would make a lost token look like a deliberate
    /// keyless connection, and the difference matters when deciding
    /// what the app is allowed to offer. Keyless callers never get admin
    /// rights server-side, so an admin screen must not be shown for one.
    private var keyless: Bool = false
    private let session: URLSession
    private let decoder = JSONDecoder()
    private let encoder = JSONEncoder()

    /// The app's own version, sent as the User-Agent and recorded on the
    /// device record so `waiter hyperlink devices` can show which build
    /// a phone is running when something misbehaves.
    static let appVersion: String =
        (Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String) ?? "1.0.26"

    init(endpoints: [String] = [], token: String? = nil) {
        self.endpoints = endpoints
        self.token = token

        let config = URLSessionConfiguration.default
        // A home PC on a tailnet across cellular is slower than a CDN
        // and a 70B first token is slower still, so the resource timeout
        // is generous. The *request* timeout stays short: a dead address
        // must be ruled out quickly for failover to be worth having.
        config.timeoutIntervalForRequest = 20
        config.timeoutIntervalForResource = 600
        config.waitsForConnectivity = false
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        config.allowsExpensiveNetworkAccess = true
        config.allowsConstrainedNetworkAccess = true
        self.session = URLSession(configuration: config)
    }

    // MARK: - Configuration

    func configure(endpoints: [String], token: String?, keyless: Bool = false) {
        self.endpoints = endpoints
        self.token = token
        self.keyless = keyless
        self.preferredIndex = 0
    }

    func setToken(_ token: String?) { self.token = token }

    var currentEndpoint: String? {
        guard !endpoints.isEmpty else { return nil }
        return endpoints[min(preferredIndex, endpoints.count - 1)]
    }

    // MARK: - Request plumbing

    private func makeRequest(
        base: String,
        path: String,
        method: String,
        body: Data?,
        contentType: String?,
        authenticated: Bool,
        timeout: TimeInterval
    ) throws -> URLRequest {
        guard let url = URL(string: base.hasSuffix("/") ? String(base.dropLast()) + path : base + path) else {
            throw HyperLinkError.transport("Not a usable server address: \(base)")
        }
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.httpBody = body
        request.timeoutInterval = timeout
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("HyperLink-iOS/\(Self.appVersion)", forHTTPHeaderField: "User-Agent")
        if let contentType {
            request.setValue(contentType, forHTTPHeaderField: "Content-Type")
        }
        if authenticated && !keyless {
            guard let token, !token.isEmpty else { throw HyperLinkError.notConfigured }
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    /// Run a request against each endpoint in turn until one answers.
    ///
    /// "Answers" means the HTTP layer completed — a 404 is an answer and
    /// stops failover, because the address is clearly a server and
    /// trying the next one would just produce the same 404 more slowly.
    /// Only a transport failure moves on.
    private func send(
        path: String,
        method: String = "GET",
        body: Data? = nil,
        contentType: String? = "application/json",
        authenticated: Bool = true,
        timeout: TimeInterval = 20
    ) async throws -> Data {
        guard !endpoints.isEmpty else { throw HyperLinkError.notConfigured }

        var tried: [String] = []
        var lastTransportError: Error?

        // Start at the endpoint that worked last time, then wrap around.
        let order = (0..<endpoints.count).map { (preferredIndex + $0) % endpoints.count }
        for index in order {
            let base = endpoints[index]
            tried.append(base)
            do {
                let request = try makeRequest(
                    base: base,
                    path: path,
                    method: method,
                    body: body,
                    contentType: contentType,
                    authenticated: authenticated,
                    timeout: timeout
                )
                let (data, response) = try await session.data(for: request)
                preferredIndex = index
                try Self.check(response: response, data: data)
                return data
            } catch let error as HyperLinkError {
                // A `.transport` failure means this address is not
                // usable (unparseable, wrong scheme) — that is exactly
                // what failover is for. Anything else is an answer from
                // a real server and trying the next address would just
                // produce the same answer more slowly.
                if case .transport = error {
                    lastTransportError = error
                    continue
                }
                throw error
            } catch {
                lastTransportError = error        // dead address; try the next
                continue
            }
        }
        if let lastTransportError, endpoints.count == 1 {
            throw HyperLinkError.transport(lastTransportError.localizedDescription)
        }
        throw HyperLinkError.noReachableEndpoint(tried)
    }

    private static func check(response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard !(200..<300).contains(http.statusCode) else { return }
        if let envelope = try? JSONDecoder().decode(APIErrorEnvelope.self, from: data) {
            throw HyperLinkError.serverError(
                code: envelope.error.code,
                message: envelope.error.message,
                status: http.statusCode
            )
        }
        if http.statusCode == 401 || http.statusCode == 403 {
            throw HyperLinkError.unauthorized(
                "This device is no longer paired with that server. Pair it again."
            )
        }
        throw HyperLinkError.serverError(
            code: "HTTP_\(http.statusCode)",
            message: "The server returned HTTP \(http.statusCode).",
            status: http.statusCode
        )
    }

    private func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        do {
            return try decoder.decode(type, from: data)
        } catch {
            throw HyperLinkError.decoding(String(describing: error))
        }
    }

    private func get<T: Decodable>(_ path: String, as type: T.Type, timeout: TimeInterval = 20) async throws -> T {
        try decode(type, from: await send(path: path, timeout: timeout))
    }

    private func post<T: Decodable>(
        _ path: String,
        body: some Encodable,
        as type: T.Type,
        authenticated: Bool = true,
        timeout: TimeInterval = 120
    ) async throws -> T {
        let data = try encoder.encode(body)
        return try decode(
            type,
            from: await send(
                path: path, method: "POST", body: data,
                authenticated: authenticated, timeout: timeout
            )
        )
    }

    // MARK: - Pairing (the only unauthenticated calls)

    /// Redeem a pairing code against one specific address.
    ///
    /// Deliberately not routed through `send`: at pairing time there is
    /// exactly one address — the one the user typed — and no token, so
    /// failover has nothing to fail over to and a confusing
    /// "tried 1 address" error would replace a precise one.
    static func pair(
        address: String,
        code: String,
        deviceName: String
    ) async throws -> (PairRedeemResponse, EndpointsResponse?) {
        let base = normalize(address)
        let client = HyperLinkClient(endpoints: [base], token: nil)
        let payload = PairRedeemRequest(
            code: code.uppercased().filter { $0.isLetter || $0.isNumber },
            deviceName: deviceName,
            platform: "ios",
            appVersion: appVersion
        )
        let redeemed: PairRedeemResponse = try await client.post(
            "/hyperlink/pair/redeem", body: payload, as: PairRedeemResponse.self,
            authenticated: false, timeout: 20
        )
        // Now that there is a token, ask the server for its full address
        // list. This is what makes the app work away from home without
        // the user ever typing a Tailscale name.
        await client.setToken(redeemed.deviceToken)
        // Optional on purpose: a phone that redeemed a code has a
        // working credential, and failing the whole pairing because the
        // follow-up address list did not come back would throw away a
        // single-use code over something recoverable.
        let discovered = try? await client.endpoints()
        return (redeemed, discovered)
    }

    /// Connect with a T2S key instead of redeeming a pairing code.
    ///
    /// Pairing needs someone at the PC: it is an admin operation, and a
    /// T2S key is never an admin. That is a problem when the PC is not
    /// to hand, or when a pairing code expired mid-setup — which is the
    /// case this exists for.
    ///
    /// A T2S key is 26 body characters precisely so it can be typed on a
    /// phone, and the server accepts one as a first-class HyperLink
    /// credential. There is no redeem step: the key *is* the credential,
    /// so this verifies it works and asks the server for its address
    /// list, which is the same thing pairing does after redeeming.
    ///
    /// Nothing is uppercased or stripped here, unlike a pairing code. A
    /// T2S key is case-sensitive and contains punctuation; normalising it
    /// would turn a correct key into a wrong one.
    static func connect(
        address: String,
        key: String
    ) async throws -> EndpointsResponse {
        let base = normalize(address)
        let client = HyperLinkClient(endpoints: [base], token: key.trimmingCharacters(in: .whitespacesAndNewlines))
        return try await client.endpoints()
    }

    /// Connect with no credential at all, to a server in trusted-network
    /// mode.
    ///
    /// The server decides, not the app: it answers only if its operator
    /// turned the mode on *and* classifies this connection as loopback,
    /// LAN or tailnet. A public origin is refused however the server is
    /// configured, so this cannot be used to reach a machine over the
    /// internet by leaving the key out.
    ///
    /// Throws `.unauthorized` when the mode is off, which is the common
    /// case and reads correctly in the pairing screen: keyless is
    /// something the person at the PC has to enable first.
    static func connectKeyless(address: String) async throws -> EndpointsResponse {
        let client = HyperLinkClient(endpoints: [normalize(address)], token: nil)
        await client.configure(
            endpoints: [normalize(address)], token: nil, keyless: true
        )
        return try await client.endpoints()
    }

    /// Does this look like a key the server will accept as a credential?
    ///
    /// A shape check only, to catch a paste that lost characters before
    /// it costs a round trip. The server decides whether the key is real.
    static func looksLikeKey(_ candidate: String) -> Bool {
        let text = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
        guard text.hasPrefix("T2S_") || text.hasPrefix("T2_") || text.hasPrefix("T1_") else {
            return false
        }
        // Shortest real key: "T2S_" plus a 26-character body, two
        // lowercase, five specials, a slash, a digit and "-<level>".
        return text.count >= 20
    }

    /// `desktop:8000` → `http://desktop:8000`, and strip a trailing slash.
    static func normalize(_ address: String) -> String {
        var text = address.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return text }
        if !text.contains("://") { text = "http://" + text }
        while text.hasSuffix("/") { text.removeLast() }
        // A bare host with no port almost always means the default the
        // server advertises; adding it beats a connection refused on 80.
        if let url = URL(string: text), url.port == nil, url.scheme == "http" {
            text += ":8000"
        }
        return text
    }

    // MARK: - Endpoints, status, identity

    func endpoints() async throws -> EndpointsResponse {
        try await get("/hyperlink/endpoints", as: EndpointsResponse.self, timeout: 10)
    }

    func status() async throws -> ServerStatus {
        try await get("/status", as: ServerStatus.self, timeout: 10)
    }

    /// Other machines on the paired server's tailnet.
    ///
    /// Admin-only server-side, so this throws 403 for an ordinary device
    /// token — which is correct: a phone's credential for one server is
    /// not authority to enumerate every machine its owner runs.
    ///
    /// The timeout is long because the server probes each peer, and a
    /// sleeping laptop costs the probe budget before it is given up on.
    func peers() async throws -> PeersResponse {
        try await get("/hyperlink/peers", as: PeersResponse.self, timeout: 30)
    }

    func whoami() async throws -> DeviceSummary {
        try await get("/hyperlink/devices/me", as: DeviceResponse.self, timeout: 10).device
    }

    /// Sign out: revoke this device's own token, server-side.
    func unpairSelf(deviceID: String) async throws {
        _ = try await send(path: "/hyperlink/devices/\(deviceID)", method: "DELETE", timeout: 15)
    }

    // MARK: - Sessions

    func sessions() async throws -> [ChatSession] {
        try await get("/hyperlink/sessions?limit=100", as: SessionListResponse.self).sessions
    }

    func createSession(title: String = "", modelID: String = "", systemPrompt: String = "") async throws -> ChatSession {
        struct Body: Encodable {
            let title: String
            let model_id: String
            let system_prompt: String
        }
        return try await post(
            "/hyperlink/sessions",
            body: Body(title: title, model_id: modelID, system_prompt: systemPrompt),
            as: SessionResponse.self,
            timeout: 20
        ).session
    }

    func setModel(_ modelID: String, for sessionID: String) async throws -> ChatSession {
        struct Body: Encodable {
            let model_id: String
            let backend: String
        }
        let data = try encoder.encode(Body(model_id: modelID, backend: "lmstudio"))
        let response = try await send(
            path: "/hyperlink/sessions/\(sessionID)", method: "PATCH", body: data, timeout: 20
        )
        return try decode(SessionResponse.self, from: response).session
    }

    func deleteSession(_ sessionID: String) async throws {
        _ = try await send(path: "/hyperlink/sessions/\(sessionID)", method: "DELETE", timeout: 15)
    }

    func messages(in sessionID: String, afterSeq: Int = 0) async throws -> [ChatMessage] {
        try await get(
            "/hyperlink/sessions/\(sessionID)/messages?after_seq=\(afterSeq)",
            as: MessageListResponse.self
        ).messages
    }

    // MARK: - Chat

    func chat(
        sessionID: String,
        content: String,
        attachmentIDs: [String] = [],
        modelID: String? = nil
    ) async throws -> ChatTurnResponse {
        struct Body: Encodable {
            let content: String
            let attachment_ids: [String]
            let model_id: String?
        }
        return try await post(
            "/hyperlink/sessions/\(sessionID)/chat",
            body: Body(content: content, attachment_ids: attachmentIDs, model_id: modelID),
            as: ChatTurnResponse.self,
            timeout: 600
        )
    }

    /// Build the request for a streaming turn. The stream itself is
    /// driven by `SSEStream`, which needs the `URLRequest` rather than a
    /// decoded result — so this hands one back instead of doing the call.
    func streamingChatRequest(
        sessionID: String,
        content: String,
        attachmentIDs: [String],
        modelID: String?
    ) throws -> URLRequest {
        guard let base = currentEndpoint else { throw HyperLinkError.notConfigured }
        struct Body: Encodable {
            let content: String
            let attachment_ids: [String]
            let model_id: String?
            let stream: Bool
        }
        let body = try encoder.encode(
            Body(content: content, attachment_ids: attachmentIDs, model_id: modelID, stream: true)
        )
        var request = try makeRequest(
            base: base,
            path: "/hyperlink/sessions/\(sessionID)/chat/stream",
            method: "POST",
            body: body,
            contentType: "application/json",
            authenticated: true,
            timeout: 600
        )
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        return request
    }

    // MARK: - Attachments

    /// Multipart upload, built by hand.
    ///
    /// `URLSession` has no multipart builder and the body is small and
    /// well-specified, so hand-rolling it is less code than a dependency
    /// — but note the exact CRLFs: a `\n` where the spec says `\r\n`
    /// produces a 422 from Starlette that reads like a server bug.
    func upload(data: Data, filename: String, contentType: String, sessionID: String) async throws -> Attachment {
        let boundary = "hyperlink.\(UUID().uuidString)"
        var body = Data()
        func append(_ string: String) { body.append(Data(string.utf8)) }

        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"session_id\"\r\n\r\n")
        append("\(sessionID)\r\n")
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"file\"; filename=\"\(filename)\"\r\n")
        append("Content-Type: \(contentType)\r\n\r\n")
        body.append(data)
        append("\r\n--\(boundary)--\r\n")

        let response = try await send(
            path: "/hyperlink/files",
            method: "POST",
            body: body,
            contentType: "multipart/form-data; boundary=\(boundary)",
            timeout: 300
        )
        return try decode(AttachmentResponse.self, from: response).file
    }

    func attachmentData(_ fileID: String) async throws -> Data {
        try await send(path: "/hyperlink/files/\(fileID)", timeout: 120)
    }

    // MARK: - Models

    func bridgeModels() async throws -> BridgeModelsResponse {
        try await get("/bridge/lmstudio/models", as: BridgeModelsResponse.self, timeout: 30)
    }

    func resolveModel(pageURL: String, fileURL: String, prefer: String) async throws -> ResolvedModel {
        try await post(
            "/hyperlink/models/resolve",
            body: HFResolveRequest(
                pageURL: pageURL, fileURL: fileURL, prefer: prefer, includeVision: true
            ),
            as: ResolvedModel.self,
            timeout: 60
        )
    }

    // MARK: - Sync (0.72.4.post9)

    /// Changes since `cursor`, as a bounded page.
    ///
    /// Loop until `more` is false, carrying `cursor` forward from each
    /// page. Do **not** advance it by the page size: a filtered feed
    /// skips rows, and guessing would skip real changes with them.
    func sync(
        cursor: Int,
        limit: Int = 100,
        sessionID: String? = nil,
        entities: [String]? = nil
    ) async throws -> SyncPage {
        var path = "/hyperlink/sync?cursor=\(cursor)&limit=\(limit)"
        if let sessionID, !sessionID.isEmpty {
            path += "&session_id=\(sessionID)"
        }
        if let entities, !entities.isEmpty {
            path += "&entities=\(entities.joined(separator: ","))"
        }
        return try await get(path, as: SyncPage.self)
    }

    /// Claim an idempotency key before sending a turn.
    ///
    /// The phone cannot tell "the server never saw it" from "the server
    /// saw it and the reply was lost", so it retries — and without a
    /// key that produces two identical messages and two replies, one of
    /// which cost real tokens for nothing. Mint the id **before** the
    /// first attempt and reuse it on every retry.
    func claim(clientMsgID: String) async throws -> SyncClaim {
        try await post(
            "/hyperlink/sync/claim",
            body: SyncClaimRequest(clientMsgID: clientMsgID),
            as: SyncClaim.self,
            timeout: 20
        )
    }

    /// An id for one outbound turn.
    ///
    /// A UUID rather than a counter: a counter resets when the app is
    /// reinstalled, and a reused key would silently return an old
    /// turn's answer instead of sending the new one.
    static func newClientMessageID() -> String {
        "cmsg_" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24)
    }

    // MARK: - Push notifications (0.72.4.post9)

    /// Register this device's APNs token.
    ///
    /// Safe to call on every launch — iOS hands the app a token each
    /// time, and the server updates rather than duplicating. The
    /// response carries a fingerprint; the token is never returned.
    func registerPush(
        token: Data,
        bundleID: String,
        environment: String = "production",
        events: [String]? = nil
    ) async throws -> PushRegistration {
        let hex = token.map { String(format: "%02x", $0) }.joined()
        let response: PushRegistrationResponse = try await post(
            "/hyperlink/push",
            body: PushRegisterRequest(
                token: hex, platform: "ios", bundleID: bundleID,
                environment: environment, events: events
            ),
            as: PushRegistrationResponse.self,
            timeout: 20
        )
        return response.registration
    }

    func pushRegistrations(includeDisabled: Bool = false) async throws -> PushRegistrationList {
        try await get(
            "/hyperlink/push?include_disabled=\(includeDisabled)",
            as: PushRegistrationList.self
        )
    }

    /// What this server can notify about.
    ///
    /// Fetched rather than hard-coded, so a server that gains an event
    /// kind can offer it without an App Store release.
    func pushEventCatalogue() async throws -> [String] {
        try await get("/hyperlink/push/events", as: PushEventCatalogue.self).events
    }

    func setPushEvents(
        registrationID: String, events: [String]
    ) async throws -> PushRegistration {
        let data = try encoder.encode(PushEventsRequest(events: events))
        let response: PushRegistrationResponse = try decode(
            PushRegistrationResponse.self,
            from: await send(
                path: "/hyperlink/push/\(registrationID)", method: "PATCH",
                body: data, timeout: 20
            )
        )
        return response.registration
    }

    func unregisterPush(registrationID: String) async throws {
        _ = try await send(
            path: "/hyperlink/push/\(registrationID)", method: "DELETE", timeout: 20
        )
    }

    // MARK: - Search (0.72.4.post9)

    /// Search this account's sessions and messages.
    ///
    /// Check `capped` before presenting the result as complete: the
    /// server bounds its scan, and a silent partial answer is what
    /// makes someone conclude a conversation is gone.
    func search(
        _ query: String,
        limit: Int = 25,
        sessionID: String? = nil,
        includeArchived: Bool = false,
        roles: [String]? = nil
    ) async throws -> SearchResults {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            // The server rejects an empty `q` with a 422. Answering
            // locally keeps an empty search box from producing an error
            // banner on every keystroke.
            return SearchResults(hits: [], scanned: 0, capped: false, terms: [])
        }
        let escaped = trimmed.addingPercentEncoding(
            withAllowedCharacters: .urlQueryAllowed
        ) ?? trimmed
        var path = "/hyperlink/search?q=\(escaped)&limit=\(limit)"
        if let sessionID, !sessionID.isEmpty {
            path += "&session_id=\(sessionID)"
        }
        if includeArchived {
            path += "&include_archived=true"
        }
        if let roles, !roles.isEmpty {
            path += "&roles=\(roles.joined(separator: ","))"
        }
        return try await get(path, as: SearchResults.self, timeout: 30)
    }
}
