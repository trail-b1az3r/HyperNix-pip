//  ParsingTests.swift
//  The pure logic that has no business being wrong: address
//  normalisation, fenced-code splitting, and SSE frame decoding.
//
//  Nothing here touches the network. These are the three places where a
//  quiet bug produces a symptom that looks like a server problem — an
//  address that silently loses its port, a code block rendered as
//  prose, a stream frame dropped — so they are the three worth pinning
//  down in CI.

import XCTest
@testable import HyperLink

final class AddressNormalizationTests: XCTestCase {
    func testAddsSchemeAndDefaultPort() {
        XCTAssertEqual(HyperLinkClient.normalize("desktop"), "http://desktop:8000")
        XCTAssertEqual(HyperLinkClient.normalize("192.168.1.10"), "http://192.168.1.10:8000")
    }

    func testKeepsAnExplicitPort() {
        XCTAssertEqual(HyperLinkClient.normalize("desktop:1234"), "http://desktop:1234")
        XCTAssertEqual(HyperLinkClient.normalize("http://desktop:9000"), "http://desktop:9000")
    }

    func testKeepsHTTPSAndDoesNotInventAPort() {
        // 443 is implied; appending :8000 to an https URL would break a
        // perfectly good reverse-proxy address.
        XCTAssertEqual(HyperLinkClient.normalize("https://t1.example.com"), "https://t1.example.com")
    }

    func testStripsTrailingSlashes() {
        XCTAssertEqual(HyperLinkClient.normalize("http://desktop:8000///"), "http://desktop:8000")
    }

    func testTailscaleNameSurvivesIntact() {
        XCTAssertEqual(
            HyperLinkClient.normalize("desktop.tailnet-abc.ts.net:8000"),
            "http://desktop.tailnet-abc.ts.net:8000"
        )
    }

    func testEmptyStaysEmpty() {
        XCTAssertEqual(HyperLinkClient.normalize("   "), "")
    }
}

final class MessageSegmentTests: XCTestCase {
    func testPlainTextIsOneSegment() {
        let segments = MessageSegment.parse("just prose")
        XCTAssertEqual(segments.count, 1)
        guard case let .text(body) = segments[0] else { return XCTFail("expected text") }
        XCTAssertEqual(body, "just prose")
    }

    func testFencedCodeIsSplitOutWithItsLanguage() {
        let content = """
        Here you go:

        ```swift
        let x = 1
        ```

        That is all.
        """
        let segments = MessageSegment.parse(content)
        XCTAssertEqual(segments.count, 3)
        guard case let .code(language, body) = segments[1] else { return XCTFail("expected code") }
        XCTAssertEqual(language, "swift")
        XCTAssertEqual(body, "let x = 1")
    }

    func testUnterminatedFenceIsStillCode() {
        // This is every code block mid-stream. Rendering it as prose
        // until the closing fence arrives makes the answer visibly
        // reflow when it completes.
        let segments = MessageSegment.parse("Try:\n\n```python\nprint(1)")
        XCTAssertEqual(segments.count, 2)
        guard case let .code(language, body) = segments[1] else { return XCTFail("expected code") }
        XCTAssertEqual(language, "python")
        XCTAssertEqual(body, "print(1)")
    }

    func testBlankProseBetweenBlocksIsDropped() {
        let segments = MessageSegment.parse("```\na\n```\n\n```\nb\n```")
        XCTAssertEqual(segments.count, 2)
        for segment in segments {
            guard case .code = segment else { return XCTFail("expected only code segments") }
        }
    }

    func testEmptyContentProducesNothing() {
        XCTAssertTrue(MessageSegment.parse("").isEmpty)
    }
}

final class APIDecodingTests: XCTestCase {
    func testChatMessageDecodesSnakeCase() throws {
        let json = """
        {"message_id":"msg_1","session_id":"chat_1","seq":3,"role":"assistant",
         "content":"hi","model_id":"qwen","attachment_ids":["file_1"],
         "created_at":1.0,"input_tokens":10,"output_tokens":4,"metadata":{}}
        """
        let message = try JSONDecoder().decode(ChatMessage.self, from: Data(json.utf8))
        XCTAssertEqual(message.messageID, "msg_1")
        XCTAssertEqual(message.attachmentIDs, ["file_1"])
        XCTAssertTrue(message.isAssistant)
        XCTAssertFalse(message.isUser)
    }

    func testLocalPlaceholderSortsAndIdentifiesAsTemporary() {
        let placeholder = ChatMessage.local(role: "user", content: "x", sessionID: "chat_1")
        // The negative sequence is what lets the optimistic bubble be
        // removed without touching real messages.
        XCTAssertEqual(placeholder.seq, -1)
        XCTAssertTrue(placeholder.isUser)
    }

    func testErrorEnvelopeDecodes() throws {
        let json = """
        {"error":{"code":"MODEL_UNAVAILABLE","message":"nothing loaded","details":{}},
         "request_id":"abc"}
        """
        let envelope = try JSONDecoder().decode(APIErrorEnvelope.self, from: Data(json.utf8))
        XCTAssertEqual(envelope.error.code, "MODEL_UNAVAILABLE")
        XCTAssertEqual(envelope.requestID, "abc")
    }

    func testRevokedTokenIsRecognisedAsNeedingRepairing() {
        let error = HyperLinkError.serverError(
            code: "AUTH_REVOKED_KEY", message: "unpaired", status: 401
        )
        XCTAssertTrue(error.requiresRepairing)

        let unavailable = HyperLinkError.serverError(
            code: "MODEL_UNAVAILABLE", message: "nothing loaded", status: 503
        )
        XCTAssertFalse(unavailable.requiresRepairing)
    }

    func testEndpointKnowsWhichAddressesWorkAwayFromHome() throws {
        let json = """
        {"server_name":"desktop","t1_version":"1.0.26.8.0.1","tailscale":true,
         "reachable_off_lan":true,
         "endpoints":[
           {"url":"http://desktop.tailnet.ts.net:8000","kind":"tailscale-dns","priority":1,"note":""},
           {"url":"http://192.168.1.10:8000","kind":"lan","priority":3,"note":""}]}
        """
        let response = try JSONDecoder().decode(EndpointsResponse.self, from: Data(json.utf8))
        XCTAssertTrue(response.endpoints[0].worksOffLAN)
        XCTAssertFalse(response.endpoints[1].worksOffLAN)
    }
}


/// `HyperLinkClient.looksLikeKey` — the shape check on the T2S field.
///
/// A pairing code is normalised on the way out (uppercased, punctuation
/// stripped) because it is six unambiguous characters read off a screen.
/// A key must never get that treatment: it is case-sensitive and full of
/// punctuation, so "helpfully" cleaning it up turns a correct key into a
/// wrong one. These pin the two apart.
final class KeyShapeTests: XCTestCase {
    func testAcceptsAT2SKey() {
        XCTAssertTrue(HyperLinkClient.looksLikeKey("T2S_kZsLoOvKacRu0X2nR6JkrIOq87cs$}')|/1-1"))
    }

    func testAcceptsT2AndT1() {
        XCTAssertTrue(HyperLinkClient.looksLikeKey("T2_bsCHVzqVrvM5rALT2kX3Wt9IA8qn:_;,?\\9-1"))
        XCTAssertTrue(HyperLinkClient.looksLikeKey("T1_x1Hauce3p6TE6W1ZH9BFVmojkx_`)>*/5"))
    }

    func testRejectsAPairingCode() {
        // The mistake this exists to catch: a six-character code typed
        // into the key field, or the reverse.
        XCTAssertFalse(HyperLinkClient.looksLikeKey("ABC123"))
    }

    func testRejectsEmptyAndJunk() {
        XCTAssertFalse(HyperLinkClient.looksLikeKey(""))
        XCTAssertFalse(HyperLinkClient.looksLikeKey("   "))
        XCTAssertFalse(HyperLinkClient.looksLikeKey("hello world"))
    }

    func testRejectsATruncatedPaste() {
        XCTAssertFalse(HyperLinkClient.looksLikeKey("T2S_short"))
    }

    func testSurroundingWhitespaceIsTolerated() {
        // Pasting from a terminal frequently brings a newline along.
        XCTAssertTrue(
            HyperLinkClient.looksLikeKey("  T2S_kZsLoOvKacRu0X2nR6JkrIOq87cs$}')|/1-1\n")
        )
    }
}

final class LoadedModelsParsingTests: XCTestCase {
    func testEveryLoadedModelAndTheDefault() throws {
        let json = """
        {"models": [
          {"model_id": "qwen3-8b", "backend": "hypernix", "label": "HyperNix runner", "default": true, "instances": 2},
          {"model_id": "gemma-3-12b", "backend": "lmstudio", "label": "LM Studio", "default": false}
        ], "default": "qwen3-8b", "request_id": "r"}
        """.data(using: .utf8)!
        let list = try JSONDecoder().decode(LoadedModelList.self, from: json)
        XCTAssertEqual(list.models.map(\.modelID), ["qwen3-8b", "gemma-3-12b"])
        XCTAssertEqual(list.defaultModel, "qwen3-8b")
        XCTAssertTrue(list.models[0].isDefault)
        XCTAssertEqual(list.models[0].instances, 2)
        XCTAssertEqual(list.models[1].instances, 1)
        XCTAssertNotEqual(list.models[0].id, list.models[1].id)
    }

    func testAnEmptyAnswerIsAnEmptyList() throws {
        let list = try JSONDecoder().decode(LoadedModelList.self, from: Data("{}".utf8))
        XCTAssertTrue(list.models.isEmpty)
        XCTAssertEqual(list.defaultModel, "")
    }
}
