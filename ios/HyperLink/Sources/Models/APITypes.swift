//  APITypes.swift
//  The wire types, mirroring hypernix/t1api/schemas.py.
//
//  Every one of these is Decodable-only unless the app actually sends
//  it. Making a response type Encodable "for symmetry" is how a field
//  the server never reads ends up in a request body, so the direction
//  is part of the type.
//
//  Field names match the Python side exactly and CodingKeys convert
//  snake_case, rather than relying on .convertFromSnakeCase on the
//  decoder: the conversion is lossy in both directions for names like
//  `sha256` and `t1_version`, and a silent mismatch shows up as a
//  decode failure at runtime instead of a compile error.

import Foundation

// MARK: - Envelope

/// The shape every T1 error comes back in.
struct APIErrorEnvelope: Decodable, Sendable {
    struct Detail: Decodable, Sendable {
        let code: String
        let message: String
    }
    let error: Detail
    let requestID: String?

    enum CodingKeys: String, CodingKey {
        case error
        case requestID = "request_id"
    }
}

// MARK: - Discovery / pairing

struct ServerEndpoint: Decodable, Hashable, Sendable {
    let url: String
    let kind: String
    let priority: Int
    let note: String

    /// True for the addresses that keep working when the phone leaves
    /// the house — what the pairing screen tells the user about.
    var worksOffLAN: Bool { kind.hasPrefix("tailscale") || kind == "configured" }
}

struct EndpointsResponse: Decodable, Sendable {
    let serverName: String
    let t1Version: String
    let endpoints: [ServerEndpoint]
    let tailscale: Bool
    let reachableOffLAN: Bool
    /// The server's stable identity. Defaulted rather than required: a
    /// server from before 0.72.4 does not send one, and an app that
    /// failed to decode its reply would break every existing pairing on
    /// upgrade. `ServerIdentity` treats an empty value as "unknown".
    let serverFingerprint: String
    /// Whether this server accepts keyless connections from trusted
    /// networks, and whether *this* phone's current network is one.
    /// The second is the only one the app can act on.
    let trustedNetwork: Bool
    let keylessAvailableHere: Bool
    let originTrust: String

    enum CodingKeys: String, CodingKey {
        case serverName = "server_name"
        case t1Version = "t1_version"
        case endpoints, tailscale
        case reachableOffLAN = "reachable_off_lan"
        case serverFingerprint = "server_fingerprint"
        case trustedNetwork = "trusted_network"
        case keylessAvailableHere = "keyless_available_here"
        case originTrust = "origin_trust"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        serverName = try container.decodeIfPresent(String.self, forKey: .serverName) ?? ""
        t1Version = try container.decodeIfPresent(String.self, forKey: .t1Version) ?? ""
        endpoints = try container.decodeIfPresent([ServerEndpoint].self, forKey: .endpoints) ?? []
        tailscale = try container.decodeIfPresent(Bool.self, forKey: .tailscale) ?? false
        reachableOffLAN = try container.decodeIfPresent(Bool.self, forKey: .reachableOffLAN) ?? false
        serverFingerprint = try container.decodeIfPresent(String.self, forKey: .serverFingerprint) ?? ""
        trustedNetwork = try container.decodeIfPresent(Bool.self, forKey: .trustedNetwork) ?? false
        keylessAvailableHere = try container.decodeIfPresent(Bool.self, forKey: .keylessAvailableHere) ?? false
        originTrust = try container.decodeIfPresent(String.self, forKey: .originTrust) ?? ""
    }
}

struct PairRedeemRequest: Encodable, Sendable {
    let code: String
    let deviceName: String
    let platform: String
    let appVersion: String

    enum CodingKeys: String, CodingKey {
        case code
        case deviceName = "device_name"
        case platform
        case appVersion = "app_version"
    }
}

struct PairRedeemResponse: Decodable, Sendable {
    let deviceID: String
    let deviceToken: String
    let name: String
    let scopes: [String]
    let serverName: String
    let t1Version: String

    enum CodingKeys: String, CodingKey {
        case deviceID = "device_id"
        case deviceToken = "device_token"
        case name, scopes
        case serverName = "server_name"
        case t1Version = "t1_version"
    }
}

struct DeviceSummary: Decodable, Identifiable, Sendable {
    let deviceID: String
    let name: String
    let platform: String
    let lastSeen: Double?
    let revoked: Bool

    var id: String { deviceID }

    enum CodingKeys: String, CodingKey {
        case deviceID = "device_id"
        case name, platform, revoked
        case lastSeen = "last_seen"
    }
}

struct DeviceResponse: Decodable, Sendable {
    let device: DeviceSummary
}

// MARK: - Status

struct ServerStatus: Decodable, Sendable {
    let environment: String
    let t1APIVersion: String
    let hypernixVersion: String
    let modelCount: Int
    let lmstudioBridgeEnabled: Bool
    let hyperlinkEnabled: Bool

    enum CodingKeys: String, CodingKey {
        case environment
        case t1APIVersion = "t1_api_version"
        case hypernixVersion = "hypernix_version"
        case modelCount = "model_count"
        case lmstudioBridgeEnabled = "lmstudio_bridge_enabled"
        case hyperlinkEnabled = "hyperlink_enabled"
    }
}

// MARK: - Sessions and messages

struct ChatSession: Decodable, Identifiable, Hashable, Sendable {
    let sessionID: String
    /// `var` alone among these, so a rename can show immediately and be
    /// put back if the server refuses. Waiting for a round trip feels
    /// broken on a phone four hops and a relay away from the machine
    /// holding the chat.
    var title: String
    let modelID: String
    let backend: String
    let createdAt: Double
    let updatedAt: Double
    let archived: Bool
    let messageCount: Int

    var id: String { sessionID }

    enum CodingKeys: String, CodingKey {
        case sessionID = "session_id"
        case title
        case modelID = "model_id"
        case backend
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case archived
        case messageCount = "message_count"
    }
}

struct SessionResponse: Decodable, Sendable { let session: ChatSession }

struct SessionListResponse: Decodable, Sendable {
    let sessions: [ChatSession]
    let count: Int
}

struct ChatMessage: Decodable, Identifiable, Hashable, Sendable {
    let messageID: String
    let sessionID: String
    let seq: Int
    let role: String
    var content: String
    let modelID: String
    let attachmentIDs: [String]
    let createdAt: Double
    let inputTokens: Int
    let outputTokens: Int

    var id: String { messageID }
    var isUser: Bool { role == "user" }
    var isAssistant: Bool { role == "assistant" }
    var isSystem: Bool { role == "system" }

    enum CodingKeys: String, CodingKey {
        case messageID = "message_id"
        case sessionID = "session_id"
        case seq, role, content
        case modelID = "model_id"
        case attachmentIDs = "attachment_ids"
        case createdAt = "created_at"
        case inputTokens = "input_tokens"
        case outputTokens = "output_tokens"
    }

    /// A locally-constructed message, for the bubble shown while the
    /// server has not answered yet. `seq` is negative so it can never
    /// collide with a real one and always sorts last.
    static func local(role: String, content: String, sessionID: String, attachments: [String] = []) -> ChatMessage {
        ChatMessage(
            messageID: "local-\(UUID().uuidString)",
            sessionID: sessionID,
            seq: -1,
            role: role,
            content: content,
            modelID: "",
            attachmentIDs: attachments,
            createdAt: Date().timeIntervalSince1970,
            inputTokens: 0,
            outputTokens: 0
        )
    }
}

struct MessageListResponse: Decodable, Sendable {
    let sessionID: String
    let messages: [ChatMessage]
    let count: Int

    enum CodingKeys: String, CodingKey {
        case sessionID = "session_id"
        case messages, count
    }
}

struct ChatTurnResponse: Decodable, Sendable {
    let sessionID: String
    let userMessage: ChatMessage
    let assistantMessage: ChatMessage
    let modelID: String
    let backend: String

    enum CodingKeys: String, CodingKey {
        case sessionID = "session_id"
        case userMessage = "user_message"
        case assistantMessage = "assistant_message"
        case modelID = "model_id"
        case backend
    }
}

// MARK: - Attachments

struct Attachment: Decodable, Identifiable, Hashable, Sendable {
    let fileID: String
    let filename: String
    let contentType: String
    let sizeBytes: Int
    let isImage: Bool
    let isText: Bool

    var id: String { fileID }

    enum CodingKeys: String, CodingKey {
        case fileID = "file_id"
        case filename
        case contentType = "content_type"
        case sizeBytes = "size_bytes"
        case isImage = "is_image"
        case isText = "is_text"
    }
}

struct AttachmentResponse: Decodable, Sendable { let file: Attachment }

// MARK: - LM Studio bridge

struct BridgeModel: Decodable, Identifiable, Hashable, Sendable {
    let modelID: String
    let kind: String
    let loaded: Bool
    let quantization: String
    let maxContextLength: Int
    let supportsVision: Bool

    var id: String { modelID }

    enum CodingKeys: String, CodingKey {
        case modelID = "model_id"
        case kind, loaded, quantization
        case maxContextLength = "max_context_length"
        case supportsVision = "supports_vision"
    }
}

struct BridgeModelsResponse: Decodable, Sendable {
    let baseURL: String
    let models: [BridgeModel]
    let count: Int
    let loadedCount: Int

    enum CodingKeys: String, CodingKey {
        case baseURL = "base_url"
        case models, count
        case loadedCount = "loaded_count"
    }
}

// MARK: - Hugging Face resolution

struct GGUFFile: Decodable, Identifiable, Hashable, Sendable {
    let filename: String
    let url: String
    let sizeBytes: Int
    let role: String
    let partIndex: Int
    let partTotal: Int

    var id: String { filename }

    enum CodingKeys: String, CodingKey {
        case filename, url, role
        case sizeBytes = "size_bytes"
        case partIndex = "part_index"
        case partTotal = "part_total"
    }
}

struct ResolvedModel: Decodable, Sendable {
    let repoID: String
    let revision: String
    let quantization: String
    let gated: Bool
    let totalBytes: Int
    let totalSizeHuman: String
    let fileCount: Int
    let isSplit: Bool
    let hasVision: Bool
    let primaryFile: String
    let files: [GGUFFile]
    let warnings: [String]
    let license: String

    enum CodingKeys: String, CodingKey {
        case repoID = "repo_id"
        case revision, quantization, gated, files, warnings, license
        case totalBytes = "total_bytes"
        case totalSizeHuman = "total_size_human"
        case fileCount = "file_count"
        case isSplit = "is_split"
        case hasVision = "has_vision"
        case primaryFile = "primary_file"
    }
}

struct HFResolveRequest: Encodable, Sendable {
    let pageURL: String
    let fileURL: String
    let prefer: String
    let includeVision: Bool

    enum CodingKeys: String, CodingKey {
        case pageURL = "page_url"
        case fileURL = "file_url"
        case prefer
        case includeVision = "include_vision"
    }
}


/// One machine on the server's tailnet — a *candidate*, never a server
/// this app has verified.
///
/// `verified` is decoded and always false; it is carried rather than
/// dropped so the type cannot be mistaken for something authenticated.
/// `name` and `serverName` are what that machine calls itself, which
/// any machine on the network can claim: they are for a person picking
/// from a list, and never for the decision to connect. Identity comes
/// from the fingerprint, after a credential has been presented.
struct DiscoveredPeer: Decodable, Hashable, Sendable, Identifiable {
    let name: String
    let address: String
    let url: String
    let online: Bool
    let reachable: Bool
    let t1Version: String
    let serverName: String
    let detail: String
    let os: String
    let verified: Bool

    var id: String { address }

    /// What to show in a list. The tailnet name if there is one, since
    /// that is what the person recognises, falling back to the address.
    var displayName: String {
        let short = name.split(separator: ".").first.map(String.init) ?? ""
        return short.isEmpty ? address : short
    }

    enum CodingKeys: String, CodingKey {
        case name, address, url, online, reachable, detail, os, verified
        case t1Version = "t1_version"
        case serverName = "server_name"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decodeIfPresent(String.self, forKey: .name) ?? ""
        address = try container.decodeIfPresent(String.self, forKey: .address) ?? ""
        url = try container.decodeIfPresent(String.self, forKey: .url) ?? ""
        online = try container.decodeIfPresent(Bool.self, forKey: .online) ?? false
        reachable = try container.decodeIfPresent(Bool.self, forKey: .reachable) ?? false
        t1Version = try container.decodeIfPresent(String.self, forKey: .t1Version) ?? ""
        serverName = try container.decodeIfPresent(String.self, forKey: .serverName) ?? ""
        detail = try container.decodeIfPresent(String.self, forKey: .detail) ?? ""
        os = try container.decodeIfPresent(String.self, forKey: .os) ?? ""
        verified = try container.decodeIfPresent(Bool.self, forKey: .verified) ?? false
    }
}

struct PeersResponse: Decodable, Sendable {
    let peers: [DiscoveredPeer]
    let count: Int
    let reachable: Int
    let tailscale: Bool
    let detail: String

    enum CodingKeys: String, CodingKey {
        case peers, count, reachable, tailscale, detail
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        peers = try container.decodeIfPresent([DiscoveredPeer].self, forKey: .peers) ?? []
        count = try container.decodeIfPresent(Int.self, forKey: .count) ?? 0
        reachable = try container.decodeIfPresent(Int.self, forKey: .reachable) ?? 0
        tailscale = try container.decodeIfPresent(Bool.self, forKey: .tailscale) ?? false
        detail = try container.decodeIfPresent(String.self, forKey: .detail) ?? ""
    }
}

// MARK: - Sync, notifications and search (0.72.4.post9)

/// One change from `GET /hyperlink/sync`.
///
/// `sessionID`, `deviceID` and `payload` are optional because the server
/// omits them when they do not apply — a device change carries no
/// session id. Decoding them as non-optional with a default would turn
/// "not applicable" into an empty string the UI has to special-case.
struct SyncChange: Decodable, Sendable, Identifiable {
    let seq: Int
    let kind: String
    let entity: String
    let entityID: String
    let createdAt: Double
    let sessionID: String?
    let deviceID: String?

    var id: Int { seq }

    /// Kinds the server may send. Compared as strings rather than
    /// decoded into an enum: an unrecognised kind from a newer server
    /// must survive the round trip, and a `RawRepresentable` enum would
    /// fail the whole page's decode instead.
    enum Kind {
        static let created = "created"
        static let updated = "updated"
        static let deleted = "deleted"
    }

    enum Entity {
        static let session = "session"
        static let message = "message"
        static let device = "device"
        static let attachment = "attachment"
    }

    enum CodingKeys: String, CodingKey {
        case seq, kind, entity
        case entityID = "entity_id"
        case createdAt = "created_at"
        case sessionID = "session_id"
        case deviceID = "device_id"
    }
}

/// One bounded page of changes.
struct SyncPage: Decodable, Sendable {
    let changes: [SyncChange]
    let cursor: Int
    let more: Bool
    /// The cursor fell off the back of the log: tombstones it needed
    /// have expired, so replaying what remains would leave this device
    /// holding a session the server has forgotten. Refetch state and
    /// restart the feed at `head`.
    let resyncRequired: Bool
    /// Where the log currently ends. A device with no local state starts
    /// here instead of at 0, which would replay every change ever made
    /// to rebuild a state it is about to fetch anyway.
    let head: Int

    enum CodingKeys: String, CodingKey {
        case changes, cursor, more, head
        case resyncRequired = "resync_required"
    }
}

struct SyncClaimRequest: Encodable, Sendable {
    let clientMsgID: String

    enum CodingKeys: String, CodingKey {
        case clientMsgID = "client_msg_id"
    }
}

/// The answer to "has this already been sent?".
struct SyncClaim: Decodable, Sendable {
    /// The only field to branch on. `false` means an earlier attempt
    /// already did this work.
    let fresh: Bool
    let clientMsgID: String
    /// `false` with `fresh == false` means an earlier attempt is still
    /// running — wait and ask again rather than sending a duplicate.
    let settled: Bool
    let result: [String: String]

    enum CodingKeys: String, CodingKey {
        case fresh, settled, result
        case clientMsgID = "client_msg_id"
    }

    /// Decoded leniently: `result` is server-shaped JSON whose values
    /// are not all strings, and a turn must not fail to send because a
    /// field the client does not read would not decode.
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        fresh = try container.decode(Bool.self, forKey: .fresh)
        settled = try container.decode(Bool.self, forKey: .settled)
        clientMsgID = try container.decode(String.self, forKey: .clientMsgID)
        result = (try? container.decode([String: String].self, forKey: .result)) ?? [:]
    }
}

struct PushRegisterRequest: Encodable, Sendable {
    let token: String
    let platform: String
    let bundleID: String
    let environment: String
    let events: [String]?

    enum CodingKeys: String, CodingKey {
        case token, platform, environment, events
        case bundleID = "bundle_id"
    }
}

/// A push registration as the server describes it.
///
/// There is deliberately no `token` field. The server never returns one
/// — a device token lets its holder push to that device — and adding a
/// property for it here would invite code that expects one.
struct PushRegistration: Decodable, Sendable, Identifiable {
    let registrationID: String
    let deviceID: String
    /// Eight hex characters. Enough to tell two of your own phones
    /// apart, useless for sending anything.
    let fingerprint: String
    let platform: String
    let bundleID: String
    let environment: String
    let events: [String]
    let enabled: Bool
    let createdAt: Double
    let updatedAt: Double
    let lastDeliveryAt: Double
    let failureCount: Int

    var id: String { registrationID }

    /// Zero means nothing has been delivered yet, not "delivered at the
    /// epoch" — worth distinguishing before showing a date.
    var hasEverDelivered: Bool { lastDeliveryAt > 0 }

    enum CodingKeys: String, CodingKey {
        case fingerprint, platform, environment, events, enabled
        case registrationID = "registration_id"
        case deviceID = "device_id"
        case bundleID = "bundle_id"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case lastDeliveryAt = "last_delivery_at"
        case failureCount = "failure_count"
    }
}

struct PushRegistrationResponse: Decodable, Sendable {
    let registration: PushRegistration
}

struct PushRegistrationList: Decodable, Sendable {
    let registrations: [PushRegistration]
    let count: Int
    /// How many notifications are queued but not yet delivered.
    let pending: Int
}

struct PushEventsRequest: Encodable, Sendable {
    let events: [String]
}

/// The event kinds a server offers.
///
/// Fetched rather than hard-coded, so a server that gains a kind can
/// offer it without an App Store release.
struct PushEventCatalogue: Decodable, Sendable {
    let events: [String]
}

/// A window of matching text, with the match located.
///
/// `ranges` are character offsets into `text`, not markup: the server
/// returning HTML would have decided how a SwiftUI view highlights a
/// match, which it cannot use.
struct SearchSnippet: Decodable, Sendable {
    let text: String
    let ranges: [[Int]]
    let truncatedStart: Bool
    let truncatedEnd: Bool

    enum CodingKeys: String, CodingKey {
        case text, ranges
        case truncatedStart = "truncated_start"
        case truncatedEnd = "truncated_end"
    }

    /// The offsets as `Range<String.Index>`, clamped to the string.
    ///
    /// Clamped because the offsets are computed server-side over
    /// case-folded text, and folding can change length — "ß" folds to
    /// "ss". An unclamped `index(_:offsetBy:)` would trap rather than
    /// merely highlight the wrong characters, and a crash is a much
    /// worse outcome than a slightly-off underline.
    var highlightRanges: [Range<String.Index>] {
        ranges.compactMap { pair in
            guard pair.count == 2, pair[0] >= 0, pair[1] > pair[0] else { return nil }
            let count = text.count
            guard pair[0] < count else { return nil }
            let lower = text.index(text.startIndex, offsetBy: pair[0])
            let upper = text.index(text.startIndex, offsetBy: min(pair[1], count))
            return lower..<upper
        }
    }
}

struct SearchHit: Decodable, Sendable, Identifiable {
    let sessionID: String
    let title: String
    let score: Double
    let updatedAt: Double
    let matchedTerms: [String]
    let messageID: String?
    let role: String?
    let createdAt: Double?
    let snippet: SearchSnippet?

    /// A session may appear both for its title and for a message in it,
    /// so the session id alone is not unique within one result list.
    var id: String { "\(sessionID)#\(messageID ?? "title")" }

    /// True when the match was on the session's title rather than in a
    /// message — worth showing differently, since tapping it should
    /// open the conversation at the top rather than at a message.
    var isTitleMatch: Bool { messageID == nil }

    enum CodingKeys: String, CodingKey {
        case title, score, role, snippet
        case sessionID = "session_id"
        case updatedAt = "updated_at"
        case matchedTerms = "matched_terms"
        case messageID = "message_id"
        case createdAt = "created_at"
    }
}

struct SearchResults: Decodable, Sendable {
    let hits: [SearchHit]
    let scanned: Int
    /// The server stopped before reading everything. Say so in the UI:
    /// a silent partial answer is what makes someone conclude a
    /// conversation is gone.
    let capped: Bool
    let terms: [String]

    /// A Decodable struct gets no memberwise initialiser once it has a
    /// custom one anywhere, and the client needs to build an empty
    /// result for a blank search box without a round trip.
    init(hits: [SearchHit], scanned: Int, capped: Bool, terms: [String]) {
        self.hits = hits
        self.scanned = scanned
        self.capped = capped
        self.terms = terms
    }
}

// MARK: - The model catalogue (T1 v1.0.26.9.2.3)
//
// `bridgeModels()` asked LM Studio what it had loaded, and that was the
// whole picture: a machine with forty GGUFs in ~/.hypernix/models showed
// an empty list and a message about opening LM Studio. `/hyperlink/models`
// merges every source the server has — the registry, the LM Studio
// bridge, and the files on disk — and says which one each model came
// from.

/// One model the server can offer, wherever it came from.
struct CatalogueModel: Decodable, Identifiable, Equatable, Sendable {
    let modelID: String
    let name: String
    /// "registry", "lmstudio" or "local".
    let source: String
    /// Non-empty only for a model with a file on this machine — which
    /// is also the only kind the runner can load.
    let path: String
    let sizeBytes: Int
    let architecture: String
    let parametersB: Double
    let contextLimit: Int
    let quant: String
    let bitsPerWeight: Double
    /// Currently answering.
    let loaded: Bool
    /// Could be loaded, as opposed to merely known about.
    let runnable: Bool
    /// Why it is not runnable, when it is not.
    let detail: String
    /// The other sources that also know this model. A GGUF on disk that
    /// LM Studio also has loaded is one model, not two.
    let alsoIn: [String]

    var id: String { modelID }

    enum CodingKeys: String, CodingKey {
        case modelID = "model_id"
        case name, source, path, quant, loaded, runnable, detail
        case sizeBytes = "size_bytes"
        case architecture
        case parametersB = "parameters_b"
        case contextLimit = "context_limit"
        case bitsPerWeight = "bits_per_weight"
        case alsoIn = "also_in"
    }

    /// Every field defaulted: this list is merged from three sources of
    /// differing richness, and a `local` entry that the indexer has not
    /// seen carries little more than a path. Throwing on a missing key
    /// would drop exactly the models this endpoint exists to surface.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelID = try c.decodeIfPresent(String.self, forKey: .modelID) ?? ""
        name = try c.decodeIfPresent(String.self, forKey: .name) ?? ""
        source = try c.decodeIfPresent(String.self, forKey: .source) ?? ""
        path = try c.decodeIfPresent(String.self, forKey: .path) ?? ""
        sizeBytes = try c.decodeIfPresent(Int.self, forKey: .sizeBytes) ?? 0
        architecture = try c.decodeIfPresent(String.self, forKey: .architecture) ?? ""
        parametersB = try c.decodeIfPresent(Double.self, forKey: .parametersB) ?? 0
        contextLimit = try c.decodeIfPresent(Int.self, forKey: .contextLimit) ?? 0
        quant = try c.decodeIfPresent(String.self, forKey: .quant) ?? ""
        bitsPerWeight = try c.decodeIfPresent(Double.self, forKey: .bitsPerWeight) ?? 0
        loaded = try c.decodeIfPresent(Bool.self, forKey: .loaded) ?? false
        runnable = try c.decodeIfPresent(Bool.self, forKey: .runnable) ?? false
        detail = try c.decodeIfPresent(String.self, forKey: .detail) ?? ""
        alsoIn = try c.decodeIfPresent([String].self, forKey: .alsoIn) ?? []
    }

    /// Where this came from, in words for a label.
    var sourceLabel: String {
        switch source {
        case "lmstudio": return "LM Studio"
        case "registry": return "Registered"
        case "local": return "On disk"
        default: return source
        }
    }

    /// "8.2B · Q4_K_M · 32k" — whichever of those the server knew.
    var summary: String {
        var parts: [String] = []
        if parametersB > 0 { parts.append(String(format: "%.1fB", parametersB)) }
        if !quant.isEmpty { parts.append(quant) }
        if contextLimit > 0 { parts.append("\(contextLimit / 1024)k") }
        if sizeBytes > 0 {
            parts.append(ByteCountFormatter.string(
                fromByteCount: Int64(sizeBytes), countStyle: .file
            ))
        }
        return parts.joined(separator: " · ")
    }
}

/// What one source contributed, and why it contributed nothing.
///
/// The reason this is on screen at all: an empty model list meant either
/// "this server has no models" or "LM Studio is not running", and the
/// app showed the same blank picker for both.
struct CatalogueSource: Decodable, Identifiable, Equatable, Sendable {
    let name: String
    let available: Bool
    let count: Int
    let detail: String

    var id: String { name }

    enum CodingKeys: String, CodingKey {
        case name, available, count, detail
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        name = try c.decodeIfPresent(String.self, forKey: .name) ?? ""
        available = try c.decodeIfPresent(Bool.self, forKey: .available) ?? false
        count = try c.decodeIfPresent(Int.self, forKey: .count) ?? 0
        detail = try c.decodeIfPresent(String.self, forKey: .detail) ?? ""
    }

    var label: String {
        switch name {
        case "lmstudio": return "LM Studio"
        case "registry": return "Registry"
        case "local": return "Models folder"
        default: return name
        }
    }
}

struct ModelCatalogue: Decodable, Equatable, Sendable {
    let models: [CatalogueModel]
    let count: Int
    let sources: [CatalogueSource]

    enum CodingKeys: String, CodingKey {
        case models, count, sources
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        models = try c.decodeIfPresent([CatalogueModel].self, forKey: .models) ?? []
        count = try c.decodeIfPresent(Int.self, forKey: .count) ?? 0
        sources = try c.decodeIfPresent([CatalogueSource].self, forKey: .sources) ?? []
    }

    static let empty = ModelCatalogue(models: [], count: 0, sources: [])

    private init(models: [CatalogueModel], count: Int, sources: [CatalogueSource]) {
        self.models = models
        self.count = count
        self.sources = sources
    }

    /// Sources that reported a problem, for the line under an empty list.
    var unavailable: [CatalogueSource] { sources.filter { !$0.available } }
}

// MARK: - Uptime

/// How long the server and the machine have been up.
///
/// Worth having on screen because it answers a question people actually
/// ask: a conversation that lost its context, or a pairing that stopped
/// working, is usually a PC that rebooted, and nothing in the app said so.
struct ServerUptime: Decodable, Equatable, Sendable {
    let processUptimeSeconds: Double
    let machineUptimeSeconds: Double?
    let startedAt: Double
    let serverName: String
    let t1Version: String

    enum CodingKeys: String, CodingKey {
        case processUptimeSeconds = "process_uptime_seconds"
        case machineUptimeSeconds = "machine_uptime_seconds"
        case startedAt = "started_at"
        case serverName = "server_name"
        case t1Version = "t1_version"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        processUptimeSeconds = try c.decodeIfPresent(
            Double.self, forKey: .processUptimeSeconds
        ) ?? 0
        machineUptimeSeconds = try c.decodeIfPresent(
            Double.self, forKey: .machineUptimeSeconds
        )
        startedAt = try c.decodeIfPresent(Double.self, forKey: .startedAt) ?? 0
        serverName = try c.decodeIfPresent(String.self, forKey: .serverName) ?? ""
        t1Version = try c.decodeIfPresent(String.self, forKey: .t1Version) ?? ""
    }

    /// "3d 4h", "4h 12m", "12m", "just now" — never "0 seconds", and
    /// never more precision than the question deserves.
    static func describe(_ seconds: Double?) -> String {
        guard let seconds, seconds > 0 else { return "unknown" }
        let total = Int(seconds)
        let days = total / 86_400
        let hours = (total % 86_400) / 3_600
        let minutes = (total % 3_600) / 60
        if days > 0 { return hours > 0 ? "\(days)d \(hours)h" : "\(days)d" }
        if hours > 0 { return minutes > 0 ? "\(hours)h \(minutes)m" : "\(hours)h" }
        if minutes > 0 { return "\(minutes)m" }
        return "just now"
    }

    var serverDescription: String { Self.describe(processUptimeSeconds) }
    var machineDescription: String { Self.describe(machineUptimeSeconds) }
}

/// What Stop actually stopped. An empty list is a success: the model
/// finishing a quarter-second before the button arrives is the common
/// race, and an error for a button that worked is the wrong answer.
struct GenerationStopResult: Decodable, Equatable, Sendable {
    let stopped: [String]
    let count: Int

    enum CodingKeys: String, CodingKey {
        case stopped, count
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        stopped = try c.decodeIfPresent([String].self, forKey: .stopped) ?? []
        count = try c.decodeIfPresent(Int.self, forKey: .count) ?? 0
    }
}

// MARK: - The runner (T1 v1.0.26.9.2.3)
//
// Switching the model used to mean walking over to the PC and using LM
// Studio. `/runner/*` owns a llama.cpp process instead, so load, unload
// and switch are operations rather than instructions — and the app can
// do them from six hundred miles away.

/// Where a model's layers would go, or did.
struct RunnerPlacement: Decodable, Equatable, Sendable {
    let gpuLayers: Int
    let totalLayers: Int
    let backend: String
    let vramBytes: Int
    let ramBytes: Int
    let swapUsedBytes: Int
    let fullyOffloaded: Bool
    /// True when the operator gave a layer count rather than letting the
    /// server work one out. Somebody who has tuned their own machine
    /// should be able to see that their number survived.
    let explicit: Bool
    /// The plan in words — shown rather than summarised, because the
    /// server knows things about the machine that the phone does not.
    let reason: String

    enum CodingKeys: String, CodingKey {
        case backend, explicit, reason
        case gpuLayers = "gpu_layers"
        case totalLayers = "total_layers"
        case vramBytes = "vram_bytes"
        case ramBytes = "ram_bytes"
        case swapUsedBytes = "swap_used_bytes"
        case fullyOffloaded = "fully_offloaded"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        gpuLayers = try c.decodeIfPresent(Int.self, forKey: .gpuLayers) ?? 0
        totalLayers = try c.decodeIfPresent(Int.self, forKey: .totalLayers) ?? 0
        backend = try c.decodeIfPresent(String.self, forKey: .backend) ?? ""
        vramBytes = try c.decodeIfPresent(Int.self, forKey: .vramBytes) ?? 0
        ramBytes = try c.decodeIfPresent(Int.self, forKey: .ramBytes) ?? 0
        swapUsedBytes = try c.decodeIfPresent(Int.self, forKey: .swapUsedBytes) ?? 0
        fullyOffloaded = try c.decodeIfPresent(Bool.self, forKey: .fullyOffloaded) ?? false
        explicit = try c.decodeIfPresent(Bool.self, forKey: .explicit) ?? false
        reason = try c.decodeIfPresent(String.self, forKey: .reason) ?? ""
    }

    /// "33 of 33 on the GPU" / "41 of 81 on the GPU, 40 on the CPU".
    var layerSummary: String {
        guard totalLayers > 0 else { return backend.isEmpty ? "" : backend }
        if fullyOffloaded { return "all \(totalLayers) layers on the GPU" }
        if gpuLayers <= 0 { return "all \(totalLayers) layers on the CPU" }
        return "\(gpuLayers) of \(totalLayers) on the GPU, "
            + "\(totalLayers - gpuLayers) on the CPU"
    }
}

/// The model the server is running, if it is running one.
struct RunningModel: Decodable, Equatable, Sendable {
    let modelID: String
    let path: String
    let port: Int
    let pid: Int
    let baseURL: String
    let contextLength: Int
    let uptimeSeconds: Double
    let placement: RunnerPlacement?

    enum CodingKeys: String, CodingKey {
        case path, port, pid, placement
        case modelID = "model_id"
        case baseURL = "base_url"
        case contextLength = "context_length"
        case uptimeSeconds = "uptime_seconds"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelID = try c.decodeIfPresent(String.self, forKey: .modelID) ?? ""
        path = try c.decodeIfPresent(String.self, forKey: .path) ?? ""
        port = try c.decodeIfPresent(Int.self, forKey: .port) ?? 0
        pid = try c.decodeIfPresent(Int.self, forKey: .pid) ?? 0
        baseURL = try c.decodeIfPresent(String.self, forKey: .baseURL) ?? ""
        contextLength = try c.decodeIfPresent(Int.self, forKey: .contextLength) ?? 0
        uptimeSeconds = try c.decodeIfPresent(Double.self, forKey: .uptimeSeconds) ?? 0
        placement = try c.decodeIfPresent(RunnerPlacement.self, forKey: .placement)
    }
}

struct RunnerStatus: Decodable, Equatable, Sendable {
    let loaded: Bool
    let model: RunningModel?
    let baseURL: String
    /// What this server can actually run on, as it reported them. Not a
    /// list the app carries: a machine without CUDA must not be offered
    /// CUDA, and only the server knows which build it has.
    let backends: [String]
    let wasRunning: Bool

    enum CodingKeys: String, CodingKey {
        case loaded, model, backends
        case baseURL = "base_url"
        case wasRunning = "was_running"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        loaded = try c.decodeIfPresent(Bool.self, forKey: .loaded) ?? false
        baseURL = try c.decodeIfPresent(String.self, forKey: .baseURL) ?? ""
        backends = try c.decodeIfPresent([String].self, forKey: .backends) ?? []
        wasRunning = try c.decodeIfPresent(Bool.self, forKey: .wasRunning) ?? false
        // An unloaded server sends `{}` here, which decodes to a
        // RunningModel of empty strings rather than to nil — so the
        // emptiness is checked rather than the presence of the key.
        let decoded = try c.decodeIfPresent(RunningModel.self, forKey: .model)
        model = (decoded?.modelID.isEmpty ?? true) ? nil : decoded
    }

    static let unknown = RunnerStatus()

    private init() {
        loaded = false
        model = nil
        baseURL = ""
        backends = []
        wasRunning = false
    }
}

/// What loading a model *would* do. Changes nothing.
///
/// Loading evicts whatever people are currently talking to, so being
/// able to see the consequence first is not a nicety.
struct RunnerPlan: Decodable, Equatable, Sendable {
    let modelID: String
    let path: String
    let placement: RunnerPlacement?

    enum CodingKeys: String, CodingKey {
        case path, placement
        case modelID = "model_id"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelID = try c.decodeIfPresent(String.self, forKey: .modelID) ?? ""
        path = try c.decodeIfPresent(String.self, forKey: .path) ?? ""
        placement = try c.decodeIfPresent(RunnerPlacement.self, forKey: .placement)
    }
}

/// What the app sends to load a model.
///
/// Every tuning field is optional and nil means "work it out": a person
/// who has not tuned anything should not have to, and a person who has
/// should not have their number second-guessed.
struct RunnerLoadRequest: Encodable, Sendable {
    let model_id: String
    let gpu_layers: Int?
    let backend: String
    let context_length: Int?
    let total_layers: Int?
}

// MARK: - Server hardware
//
// Field names taken from `hypernix.system.hardware` rather than guessed:
// `cores_logical` not `count`, `mount` not `path`, and
// `utilization_percent` with the American spelling the sampler uses. A
// CodingKey that does not match is a silent nil, which renders as "—"
// and looks exactly like a sensor that could not be read.
//
// Every field is optional on purpose. The server sends an `unavailable`
// list naming what it could not sample, and a panel that renders a
// missing GPU temperature as 0°C is a confident wrong answer about
// hardware nobody can see.

struct CPUReading: Decodable, Equatable, Sendable {
    let percent: Double?
    let coresPhysical: Int?
    let coresLogical: Int?
    let loadAverage: [Double]?
    let temperatureC: Double?
    let model: String?

    enum CodingKeys: String, CodingKey {
        case percent, model
        case coresPhysical = "cores_physical"
        case coresLogical = "cores_logical"
        case loadAverage = "load_average"
        case temperatureC = "temperature_c"
    }
}

struct MemoryReading: Decodable, Equatable, Sendable {
    let totalBytes: Int?
    let usedBytes: Int?
    let availableBytes: Int?
    let percent: Double?

    enum CodingKeys: String, CodingKey {
        case percent
        case totalBytes = "total_bytes"
        case usedBytes = "used_bytes"
        case availableBytes = "available_bytes"
    }
}

struct DiskReading: Decodable, Equatable, Sendable {
    let mount: String?
    let totalBytes: Int?
    let freeBytes: Int?
    let percent: Double?

    enum CodingKeys: String, CodingKey {
        case mount, percent
        case totalBytes = "total_bytes"
        case freeBytes = "free_bytes"
    }
}

struct GPUReading: Decodable, Equatable, Sendable {
    let index: Int?
    let vendor: String?
    let name: String?
    let memoryTotalBytes: Int?
    let memoryUsedBytes: Int?
    let utilizationPercent: Double?
    let temperatureC: Double?

    enum CodingKeys: String, CodingKey {
        case index, vendor, name
        case memoryTotalBytes = "memory_total_bytes"
        case memoryUsedBytes = "memory_used_bytes"
        case utilizationPercent = "utilization_percent"
        case temperatureC = "temperature_c"
    }
}

struct ServerHardware: Decodable, Equatable, Sendable {
    let hostname: String
    let platform: String
    let uptimeSeconds: Double?
    let cpu: CPUReading?
    let memory: MemoryReading?
    let swap: MemoryReading?
    let disks: [DiskReading]
    let gpus: [GPUReading]
    /// What could not be sampled, named. The reason this screen can be
    /// honest about a machine it is not running on.
    let unavailable: [String]

    enum CodingKeys: String, CodingKey {
        case hostname, platform, cpu, memory, swap, disks, gpus, unavailable
        case uptimeSeconds = "uptime_seconds"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hostname = try c.decodeIfPresent(String.self, forKey: .hostname) ?? ""
        platform = try c.decodeIfPresent(String.self, forKey: .platform) ?? ""
        uptimeSeconds = try c.decodeIfPresent(Double.self, forKey: .uptimeSeconds)
        // An empty dictionary is what the server sends for a subsystem
        // it could not read at all, and it decodes to a reading of all
        // nils rather than to nil — which renders identically, so it is
        // left as-is rather than special-cased.
        cpu = try c.decodeIfPresent(CPUReading.self, forKey: .cpu)
        memory = try c.decodeIfPresent(MemoryReading.self, forKey: .memory)
        swap = try c.decodeIfPresent(MemoryReading.self, forKey: .swap)
        disks = try c.decodeIfPresent([DiskReading].self, forKey: .disks) ?? []
        gpus = try c.decodeIfPresent([GPUReading].self, forKey: .gpus) ?? []
        unavailable = try c.decodeIfPresent([String].self, forKey: .unavailable) ?? []
    }
}

// MARK: - Updating the server
//
// "The T1 installed thinks it is running an older version." Half of that
// was the installer printing a stale constant; the other half is that a
// server genuinely does fall behind, and the only way to find out was to
// walk over to the machine.
//
// The commands are text to copy rather than a button that runs them.
// Updating the package under a running server is a decision with a
// restart attached, and a phone button that did it silently would be a
// phone button that takes a machine down mid-conversation.

/// Where the server's code lives, and how it got there.
struct ServerInstallation: Decodable, Equatable, Sendable {
    /// The interpreter the server is running under. The one fact that
    /// makes the commands correct rather than plausible.
    let executable: String
    let prefix: String
    let inVenv: Bool
    /// A development install, which pip will not replace.
    let editable: Bool
    let location: String
    let pythonVersion: String
    let packageVersion: String
    let t1Version: String

    enum CodingKeys: String, CodingKey {
        case executable, prefix, editable, location
        case inVenv = "in_venv"
        case pythonVersion = "python_version"
        case packageVersion = "package_version"
        case t1Version = "t1_version"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        executable = try c.decodeIfPresent(String.self, forKey: .executable) ?? ""
        prefix = try c.decodeIfPresent(String.self, forKey: .prefix) ?? ""
        inVenv = try c.decodeIfPresent(Bool.self, forKey: .inVenv) ?? false
        editable = try c.decodeIfPresent(Bool.self, forKey: .editable) ?? false
        location = try c.decodeIfPresent(String.self, forKey: .location) ?? ""
        pythonVersion = try c.decodeIfPresent(String.self, forKey: .pythonVersion) ?? ""
        packageVersion = try c.decodeIfPresent(String.self, forKey: .packageVersion) ?? ""
        t1Version = try c.decodeIfPresent(String.self, forKey: .t1Version) ?? ""
    }
}

/// One copyable line, and what it is for.
struct ServerCommand: Decodable, Identifiable, Equatable, Sendable {
    let label: String
    let command: String
    let primary: Bool
    let note: String

    var id: String { command }

    enum CodingKeys: String, CodingKey {
        case label, command, primary, note
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        label = try c.decodeIfPresent(String.self, forKey: .label) ?? ""
        command = try c.decodeIfPresent(String.self, forKey: .command) ?? ""
        primary = try c.decodeIfPresent(Bool.self, forKey: .primary) ?? false
        note = try c.decodeIfPresent(String.self, forKey: .note) ?? ""
    }
}

struct UpgradeAdvice: Decodable, Equatable, Sendable {
    let installation: ServerInstallation?
    let commands: [ServerCommand]
    /// Things that make the commands not enough on their own — chiefly
    /// that a pip upgrade does not restart a running server, which is
    /// the most confusing possible outcome of following them.
    let warnings: [String]

    enum CodingKeys: String, CodingKey {
        case installation, commands, warnings
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        installation = try c.decodeIfPresent(
            ServerInstallation.self, forKey: .installation
        )
        commands = try c.decodeIfPresent([ServerCommand].self, forKey: .commands) ?? []
        warnings = try c.decodeIfPresent([String].self, forKey: .warnings) ?? []
    }
}

// MARK: - Editing what was said
//
// The truncation is the feature rather than a side effect: everything
// below an edited message was written in reply to the *old* text, and
// that same transcript is what gets sent as context on the next turn.
// Leaving it means telling the model it said things it never said.

struct MessageEdit: Decodable, Equatable, Sendable {
    let message: ChatMessage?
    /// What the edit cost. Reported so the app can say "this removes 11
    /// messages" *before* removing them.
    let removed: [ChatMessage]
    let removedCount: Int

    enum CodingKeys: String, CodingKey {
        case message, removed
        case removedCount = "removed_count"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        message = try c.decodeIfPresent(ChatMessage.self, forKey: .message)
        removed = try c.decodeIfPresent([ChatMessage].self, forKey: .removed) ?? []
        removedCount = try c.decodeIfPresent(Int.self, forKey: .removedCount) ?? 0
    }
}

// MARK: - Settings
//
// On the server rather than the phone, for two reasons and the second
// decides it: a person with a phone and a tablet is one person, and
// these are *inputs to generation* — the system prompt, the effort
// level and the context bounds all have to be in the process that
// builds the request.

struct UserPreferences: Decodable, Equatable, Sendable {
    let displayName: String
    let bio: String
    let systemPrompt: String
    let effort: String
    /// 0 for both means "no opinion", which is the right default: a
    /// number here overrides what the model itself says it can do.
    let contextMinimum: Int
    let contextMaximum: Int
    /// Tried when the main model fails. Empty means "fail honestly",
    /// which beats silently answering as somebody else.
    let backupModel: String
    let backend: String
    let toolsEnabled: Bool
    let autoMemory: Bool

    enum CodingKeys: String, CodingKey {
        case bio, effort, backend
        case displayName = "display_name"
        case systemPrompt = "system_prompt"
        case contextMinimum = "context_minimum"
        case contextMaximum = "context_maximum"
        case backupModel = "backup_model"
        case toolsEnabled = "tools_enabled"
        case autoMemory = "auto_memory"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        displayName = try c.decodeIfPresent(String.self, forKey: .displayName) ?? ""
        bio = try c.decodeIfPresent(String.self, forKey: .bio) ?? ""
        systemPrompt = try c.decodeIfPresent(String.self, forKey: .systemPrompt) ?? ""
        effort = try c.decodeIfPresent(String.self, forKey: .effort) ?? "medium"
        contextMinimum = try c.decodeIfPresent(Int.self, forKey: .contextMinimum) ?? 0
        contextMaximum = try c.decodeIfPresent(Int.self, forKey: .contextMaximum) ?? 0
        backupModel = try c.decodeIfPresent(String.self, forKey: .backupModel) ?? ""
        backend = try c.decodeIfPresent(String.self, forKey: .backend) ?? ""
        toolsEnabled = try c.decodeIfPresent(Bool.self, forKey: .toolsEnabled) ?? false
        autoMemory = try c.decodeIfPresent(Bool.self, forKey: .autoMemory) ?? true
    }

    static let defaults = UserPreferences()

    private init() {
        displayName = ""; bio = ""; systemPrompt = ""; effort = "medium"
        contextMinimum = 0; contextMaximum = 0; backupModel = ""; backend = ""
        toolsEnabled = false; autoMemory = true
    }
}

/// The settings plus the bounds the *server* will accept.
///
/// The limits come with the values so the app does not carry its own
/// copy of a list the server owns: an effort level the phone offers and
/// the server rejects is a settings screen that cannot save, and the
/// phone has no way to know which levels a given build has.
struct PreferencesEnvelope: Decodable, Equatable, Sendable {
    let preferences: UserPreferences
    let effortLevels: [String]
    let contextFloor: Int
    let contextCeiling: Int
    let maxSystemPrompt: Int
    /// Every clamp the server applied, in words. Shown rather than
    /// swallowed — silently storing something other than what somebody
    /// typed is how a settings screen becomes untrustworthy.
    let notes: [String]

    enum CodingKeys: String, CodingKey {
        case preferences, notes
        case effortLevels = "effort_levels"
        case contextFloor = "context_floor"
        case contextCeiling = "context_ceiling"
        case maxSystemPrompt = "max_system_prompt"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        preferences = try c.decodeIfPresent(
            UserPreferences.self, forKey: .preferences
        ) ?? .defaults
        effortLevels = try c.decodeIfPresent([String].self, forKey: .effortLevels)
            ?? ["minimal", "low", "medium", "high", "maximum"]
        contextFloor = try c.decodeIfPresent(Int.self, forKey: .contextFloor) ?? 1024
        contextCeiling = try c.decodeIfPresent(Int.self, forKey: .contextCeiling)
            ?? 1_048_576
        maxSystemPrompt = try c.decodeIfPresent(Int.self, forKey: .maxSystemPrompt)
            ?? 32_000
        notes = try c.decodeIfPresent([String].self, forKey: .notes) ?? []
    }

    static let empty = PreferencesEnvelope()

    private init() {
        preferences = .defaults
        effortLevels = ["minimal", "low", "medium", "high", "maximum"]
        contextFloor = 1024
        contextCeiling = 1_048_576
        maxSystemPrompt = 32_000
        notes = []
    }
}

/// What the app sends. Every field optional: nil means "leave it alone",
/// so a build that knows about six settings cannot blank the four it has
/// never heard of.
struct PreferencesPatch: Encodable, Sendable {
    // `= nil` on every one, and it is load-bearing rather than
    // decorative: the synthesised memberwise initialiser only gives a
    // parameter a default when the property has an initial value, so
    // without these `PreferencesPatch(effort: "high")` would not
    // compile — every call site would have to name all ten fields,
    // which is exactly the whole-object PUT this type exists to avoid.
    var display_name: String? = nil
    var bio: String? = nil
    var system_prompt: String? = nil
    var effort: String? = nil
    var context_minimum: Int? = nil
    var context_maximum: Int? = nil
    var backup_model: String? = nil
    var backend: String? = nil
    var tools_enabled: Bool? = nil
    var auto_memory: Bool? = nil
}

// MARK: - What can answer

/// One thing that could serve a message, and whether it can right now.
///
/// Worth showing because the two failures look identical from the app
/// and need opposite fixes: "this server has no models" is solved by
/// downloading one, "a model is loaded but nothing is serving it" by
/// turning something on.
struct InferenceBackend: Decodable, Identifiable, Equatable, Sendable {
    let name: String
    let label: String
    let available: Bool
    let modelID: String
    let detail: String

    var id: String { name }

    enum CodingKeys: String, CodingKey {
        case name, label, available, detail
        case modelID = "model_id"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        name = try c.decodeIfPresent(String.self, forKey: .name) ?? ""
        label = try c.decodeIfPresent(String.self, forKey: .label) ?? ""
        available = try c.decodeIfPresent(Bool.self, forKey: .available) ?? false
        modelID = try c.decodeIfPresent(String.self, forKey: .modelID) ?? ""
        detail = try c.decodeIfPresent(String.self, forKey: .detail) ?? ""
    }
}

struct BackendList: Decodable, Equatable, Sendable {
    let backends: [InferenceBackend]
    /// The one a message would go to right now, or "" when none would.
    let active: String

    enum CodingKeys: String, CodingKey {
        case backends, active
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        backends = try c.decodeIfPresent([InferenceBackend].self, forKey: .backends) ?? []
        active = try c.decodeIfPresent(String.self, forKey: .active) ?? ""
    }

    static let empty = BackendList()
    private init() { backends = []; active = "" }
}

// MARK: - Memory

struct MemoryItem: Decodable, Identifiable, Equatable, Sendable {
    let memoryID: String
    let content: String
    let category: String
    /// "manual" or "auto" — what the person wrote versus what the model
    /// noticed. Shown, because "why does it think that about me" needs
    /// an answer.
    let source: String
    let pinned: Bool
    let updatedAt: Double

    var id: String { memoryID }
    var isAutomatic: Bool { source == "auto" }

    enum CodingKeys: String, CodingKey {
        case content, category, source, pinned
        case memoryID = "memory_id"
        case updatedAt = "updated_at"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        memoryID = try c.decodeIfPresent(String.self, forKey: .memoryID) ?? ""
        content = try c.decodeIfPresent(String.self, forKey: .content) ?? ""
        category = try c.decodeIfPresent(String.self, forKey: .category) ?? ""
        source = try c.decodeIfPresent(String.self, forKey: .source) ?? "manual"
        pinned = try c.decodeIfPresent(Bool.self, forKey: .pinned) ?? false
        updatedAt = try c.decodeIfPresent(Double.self, forKey: .updatedAt) ?? 0
    }
}

struct MemoryList: Decodable, Equatable, Sendable {
    let memories: [MemoryItem]
    let count: Int

    enum CodingKeys: String, CodingKey {
        case memories, count
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        memories = try c.decodeIfPresent([MemoryItem].self, forKey: .memories) ?? []
        count = try c.decodeIfPresent(Int.self, forKey: .count) ?? 0
    }
}
