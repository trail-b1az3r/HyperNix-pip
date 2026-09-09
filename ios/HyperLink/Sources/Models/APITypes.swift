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
    let title: String
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
