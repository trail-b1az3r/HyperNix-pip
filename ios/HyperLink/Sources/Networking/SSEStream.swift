//  SSEStream.swift
//  Reading the server's streaming chat frames.
//
//  The server sends HyperLink's own small frame shape rather than raw
//  OpenAI chunks (see t1api/routers/hyperlink.py) because the app needs
//  three things the OpenAI stream does not carry: the persisted id of
//  the user's message, a terminal frame with the assistant's id and
//  token counts, and errors delivered in-band once the HTTP status is
//  long gone.
//
//  `URLSession.bytes(for:)` gives an `AsyncSequence` of lines, which is
//  almost exactly SSE — the remaining work is skipping comments and
//  blank separators, and stopping at the sentinel.

import Foundation

/// One decoded frame from a `/chat/stream` response.
enum ChatStreamEvent: Sendable {
    /// The turn was accepted and the user's message is persisted.
    ///
    /// `generationID` is what Stop names. Empty against a server too
    /// old to send one, which is why the stop call treats it as
    /// optional rather than required: "stop whatever is running in this
    /// session" is still the right answer, it is just less precise.
    case start(userMessageID: String, seq: Int, generationID: String)
    /// A piece of the assistant's reply.
    case delta(String)
    /// The reply is complete and persisted.
    case done(messageID: String, seq: Int, modelID: String, finishReason: String, outputTokens: Int)
    /// The model backend failed part-way. The text already delivered is
    /// still valid and is still saved server-side.
    case failed(code: String, message: String)
}

enum SSEStream {
    /// Stream the frames of one chat turn.
    ///
    /// Cancellation is cooperative and matters here: when the user
    /// leaves the screen mid-answer, the task is cancelled, the byte
    /// stream is torn down, and the server sees the disconnect and
    /// persists the partial reply. Dropping the task without letting the
    /// sequence finish would leave the connection open until the
    /// resource timeout.
    static func events(for request: URLRequest, session: URLSession = .shared) -> AsyncThrowingStream<ChatStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let (bytes, response) = try await session.bytes(for: request)
                    if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
                        // An error before the stream opened still has a
                        // status and a JSON body, so report it properly
                        // rather than as an empty stream.
                        var collected = Data()
                        for try await byte in bytes { collected.append(byte) }
                        if let envelope = try? JSONDecoder().decode(APIErrorEnvelope.self, from: collected) {
                            throw HyperLinkError.serverError(
                                code: envelope.error.code,
                                message: envelope.error.message,
                                status: http.statusCode
                            )
                        }
                        throw HyperLinkError.serverError(
                            code: "HTTP_\(http.statusCode)",
                            message: "The server returned HTTP \(http.statusCode).",
                            status: http.statusCode
                        )
                    }

                    for try await line in bytes.lines {
                        try Task.checkCancellation()
                        // Comments (": hypernix bridge open") keep the
                        // connection warm before the first token; blank
                        // lines separate frames. Neither carries data.
                        guard line.hasPrefix("data:") else { continue }
                        let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
                        if payload == "[DONE]" { break }
                        guard let data = payload.data(using: .utf8),
                              let event = decode(data) else { continue }
                        continuation.yield(event)
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Decode one frame, tolerating shapes this build does not know.
    ///
    /// An unrecognised `type` returns nil and is skipped rather than
    /// failing the stream: a newer server adding a frame kind must not
    /// break an older app mid-answer.
    private static func decode(_ data: Data) -> ChatStreamEvent? {
        struct Frame: Decodable {
            let type: String?
            let text: String?
            let user_message_id: String?
            let generation_id: String?
            let message_id: String?
            let seq: Int?
            let model_id: String?
            let finish_reason: String?
            let output_tokens: Int?
            let error: ErrorBody?

            struct ErrorBody: Decodable {
                let code: String?
                let message: String?
            }
        }
        guard let frame = try? JSONDecoder().decode(Frame.self, from: data) else { return nil }

        // A bare {"error": ...} with no type is what the bridge's own
        // relay emits; treat it as a failure rather than dropping it.
        if let error = frame.error, frame.type == nil || frame.type == "error" {
            return .failed(
                code: error.code ?? "error",
                message: error.message ?? "The model backend failed."
            )
        }
        switch frame.type {
        case "start":
            return .start(
                userMessageID: frame.user_message_id ?? "",
                seq: frame.seq ?? 0,
                generationID: frame.generation_id ?? ""
            )
        case "delta":
            guard let text = frame.text, !text.isEmpty else { return nil }
            return .delta(text)
        case "done":
            return .done(
                messageID: frame.message_id ?? "",
                seq: frame.seq ?? 0,
                modelID: frame.model_id ?? "",
                finishReason: frame.finish_reason ?? "",
                outputTokens: frame.output_tokens ?? 0
            )
        default:
            return nil
        }
    }
}
