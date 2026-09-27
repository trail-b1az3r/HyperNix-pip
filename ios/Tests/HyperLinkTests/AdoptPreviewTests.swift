//  AdoptPreviewTests.swift
//  "Move to the HyperNix runner": what the app reads to decide whether to
//  show the button (0.72.6.post1).

import XCTest
@testable import HyperLink

final class AdoptPreviewTests: XCTestCase {
    func testAnAdminWithAModelInLMStudioSeesTheButton() throws {
        let json = """
        {"allowed": true, "why": "admin", "available": true,
         "lmstudio_loaded": ["google/gemma-3-4b"], "runner_model": ""}
        """
        let preview = try JSONDecoder().decode(AdoptPreview.self, from: Data(json.utf8))
        XCTAssertTrue(preview.available)
        XCTAssertEqual(preview.lmstudioLoaded, ["google/gemma-3-4b"])
    }

    func testARefusalDecodesAndHidesIt() throws {
        let json = #"{"allowed": false, "why": "Only an admin, or a device on this server's tailnet"}"#
        let preview = try JSONDecoder().decode(AdoptPreview.self, from: Data(json.utf8))
        XCTAssertFalse(preview.available)
        XCTAssertTrue(preview.lmstudioLoaded.isEmpty)
    }
}
