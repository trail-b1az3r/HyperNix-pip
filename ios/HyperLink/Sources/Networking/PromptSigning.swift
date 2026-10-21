//  PromptSigning.swift
//  Signed, timestamped prompts (0.72.8) — the app side of
//  hypernix/hyperlink/signing.py.
//
//  When Send is pressed the app pings the server (answered without a
//  model) and gives it four seconds; then every send, resend and edit
//  carries a signature over who sent it (account:<id> or key:<id>), when
//  Send was pressed (Unix seconds), a fresh 23–26 character nonce, the
//  chat, the server session from the ping, the action and the SHA-256 of
//  the text. The key is SHA-256("hyperlink-sign/v1:" + credential), the
//  same credential the request already carries.

import CryptoKit
import Foundation

struct PromptSignature: Codable, Equatable, Sendable {
    let signer: String
    let signed_at: Int
    let nonce: String
    let chat_session_id: String
    let server_session_id: String
    let action: String
    let content_sha256: String
    let signature: String
}

enum PromptSigning {
    static let version = "hyperlink-sign/v1"
    /// How long a ping may take before the server is called unreachable.
    static let pingTimeout: TimeInterval = 4
    private static let nonceCharacters = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")

    static func nonce(length: Int = Int.random(in: 23...26)) -> String {
        var generator = SystemRandomNumberGenerator()
        return String((0..<length).map { _ in nonceCharacters.randomElement(using: &generator)! })
    }

    static func sha256Hex(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    static func canonical(_ s: PromptSignature) -> String {
        [version, s.signer, String(s.signed_at), s.nonce, s.chat_session_id,
         s.server_session_id, s.action, s.content_sha256].joined(separator: "\n")
    }

    static func sign(
        credential: String, signer: String, chatSessionID: String = "", serverSessionID: String = "",
        action: String, content: String = "", signedAt: Date, nonce: String = PromptSigning.nonce()
    ) -> PromptSignature {
        let unsigned = PromptSignature(
            signer: signer, signed_at: Int(signedAt.timeIntervalSince1970), nonce: nonce,
            chat_session_id: chatSessionID, server_session_id: serverSessionID, action: action,
            content_sha256: content.isEmpty ? "" : sha256Hex(content), signature: ""
        )
        let key = SymmetricKey(data: SHA256.hash(data: Data("\(version):\(credential)".utf8)))
        let mac = HMAC<SHA256>.authenticationCode(for: Data(canonical(unsigned).utf8), using: key)
        return PromptSignature(
            signer: unsigned.signer, signed_at: unsigned.signed_at, nonce: unsigned.nonce,
            chat_session_id: unsigned.chat_session_id, server_session_id: unsigned.server_session_id,
            action: unsigned.action, content_sha256: unsigned.content_sha256,
            signature: mac.map { String(format: "%02x", $0) }.joined()
        )
    }
}

struct SigningIdentity: Decodable, Sendable {
    let signer: String
    let owner_kind: String
    let server_session_id: String
}

struct PingResponse: Decodable, Sendable {
    let server_session_id: String
    let signer: String
    let transit_seconds: Double
    let ping_timeout_seconds: Double?
}
