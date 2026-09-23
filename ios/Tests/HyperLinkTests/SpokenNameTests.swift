//  SpokenNameTests.swift
//  The model names Siri is offered are the names people say.
//
//  Siri matches "Load Gemma 4 E2B in HyperLink" against the titles the
//  app publishes for its models. Those were file names, which nobody
//  says, and Siri answered that HyperLink had not added support.

import XCTest
@testable import HyperLink

final class SpokenNameTests: XCTestCase {
    func testTheQuantisationAndFormatAreDropped() {
        XCTAssertEqual(SpokenName.model("blazeindustries/gemma-4-e2b-it-Q4_K_M.gguf"), "Gemma 4 E2B")
        XCTAssertEqual(SpokenName.model("Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"), "Meta Llama 3.1 8B")
        XCTAssertEqual(SpokenName.model("phi-4-mini-instruct-IQ2_XS"), "Phi 4 Mini")
        XCTAssertEqual(SpokenName.model("gpt-oss-20b-mxfp4"), "Gpt Oss 20B")
        XCTAssertEqual(SpokenName.model("Qwen3.5-9B-Instruct-GGUF"), "Qwen3.5 9B")
    }

    func testANameWithNothingToDropIsOnlyRespaced() {
        XCTAssertEqual(SpokenName.model("gemma-4-e2b"), "Gemma 4 E2B")
        XCTAssertEqual(SpokenName.model("nanonix-nano"), "Nanonix Nano")
        XCTAssertEqual(SpokenName.model("qwen3-30b-a3b"), "Qwen3 30B A3B")
    }

    func testATuningMarkerAloneIsKept() {
        // Dropping the only word would leave nothing to say.
        XCTAssertEqual(SpokenName.model("chat"), "Chat")
    }

    func testTheFullerFormsAreSynonyms() {
        let synonyms = SpokenName.synonyms("gemma-4-e2b-it-Q4_K_M.gguf")
        XCTAssertTrue(synonyms.contains("Gemma 4 E2B It"))
        XCTAssertTrue(synonyms.contains("gemma-4-e2b-it-Q4_K_M.gguf"))
        XCTAssertFalse(synonyms.contains("Gemma 4 E2B"), "the title is not its own synonym")
    }

    func testTheEntityPublishesTheSpokenTitle() {
        let entity = ModelEntity(id: "blazeindustries/gemma-4-e2b-it-Q4_K_M.gguf")
        XCTAssertEqual(String(localized: entity.displayRepresentation.title), "Gemma 4 E2B")
    }
}
