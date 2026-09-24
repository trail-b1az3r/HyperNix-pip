//  MemoryMirrorTests.swift
//  The phone's copy of its memories, kept current from a cursor.
//
//  The app used to refetch /memory/list whole, capped at 200, whenever a
//  screen appeared, so a fact the model wrote mid-chat reached the phone
//  only if somebody happened to be looking, and offline there was
//  nothing. These hold the rules the copy lives by: a delta is applied
//  by id, deletions included; a full answer replaces the copy; the copy
//  survives a round trip to disk; and a chat's `memory` frame is read.

import XCTest
@testable import HyperLink

final class MemoryMirrorTests: XCTestCase {
    private func item(_ id: String, _ content: String, pinned: Bool = false,
                      at updatedAt: Double = 1) -> MemoryItem {
        MemoryItem(memoryID: id, content: content, pinned: pinned, updatedAt: updatedAt)
    }

    func testAFirstSyncFillsTheCopy() {
        var mirror = MemoryMirror()
        XCTAssertTrue(mirror.isBehind(5), "a copy never synced is always behind")
        mirror.apply(MemorySyncPage(full: true, memories: [item("a", "metric")], cursor: 7))
        XCTAssertEqual(mirror.items.map(\.content), ["metric"])
        XCTAssertEqual(mirror.cursor, 7)
        XCTAssertNotNil(mirror.syncedAt)
    }

    func testADeltaUpsertsAndDeletesByID() {
        var mirror = MemoryMirror()
        mirror.apply(MemorySyncPage(full: true, memories: [
            item("keep", "keep"), item("edit", "old"), item("drop", "drop"),
        ], cursor: 3))
        let changed = mirror.apply(MemorySyncPage(
            full: false, memories: [item("edit", "new", at: 5), item("add", "added", at: 4)],
            deleted: ["drop"], cursor: 9
        ))
        XCTAssertTrue(changed)
        XCTAssertEqual(Set(mirror.items.map(\.memoryID)), ["keep", "edit", "add"])
        XCTAssertEqual(mirror.items.first { $0.memoryID == "edit" }?.content, "new")
        XCTAssertEqual(mirror.cursor, 9)
    }

    func testAnEmptyDeltaChangesNothingButTheCursor() {
        var mirror = MemoryMirror()
        mirror.apply(MemorySyncPage(full: true, memories: [item("a", "x")], cursor: 3))
        XCTAssertFalse(mirror.apply(MemorySyncPage(full: false, memories: [], cursor: 3)))
        XCTAssertFalse(mirror.isBehind(3))
        XCTAssertTrue(mirror.isBehind(4))
    }

    func testAFullAnswerReplacesRatherThanMerges() {
        // A server restored behind the phone's back: what it no longer has
        // must go, and a delta would never say so.
        var mirror = MemoryMirror()
        mirror.apply(MemorySyncPage(full: true, memories: [item("old", "gone")], cursor: 50))
        mirror.apply(MemorySyncPage(full: true, memories: [item("new", "here")], cursor: 2,
                                    reason: "unknown_cursor"))
        XCTAssertEqual(mirror.items.map(\.memoryID), ["new"])
        XCTAssertEqual(mirror.cursor, 2)
    }

    func testPinnedFirstThenNewest() {
        var mirror = MemoryMirror()
        mirror.apply(MemorySyncPage(full: true, memories: [
            item("old", "old", at: 1), item("pin", "pin", pinned: true, at: 0), item("new", "new", at: 9),
        ], cursor: 1))
        XCTAssertEqual(mirror.items.map(\.memoryID), ["pin", "new", "old"])
    }

    func testTheSyncAnswerDecodes() throws {
        let json = """
        {"full": false, "memories": [{"memory_id": "m1", "content": "Helix", "category": "Tech",
          "source": "auto", "pinned": false, "updated_at": 12.5, "metadata": {"key": "editor"}}],
         "deleted": ["m0"], "cursor": 41, "more": true, "reason": "", "count": 3}
        """
        let page = try JSONDecoder().decode(MemorySyncPage.self, from: Data(json.utf8))
        XCTAssertFalse(page.full)
        XCTAssertEqual(page.memories.first?.key, "editor")
        XCTAssertEqual(page.deleted, ["m0"])
        XCTAssertEqual(page.cursor, 41)
        XCTAssertTrue(page.more)
    }

    func testTheCopySurvivesTheDisk() throws {
        let folder = FileManager.default.temporaryDirectory
            .appendingPathComponent("memory-cache-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        var mirror = MemoryMirror()
        mirror.apply(MemorySyncPage(full: true, memories: [
            MemoryItem(memoryID: "m1", content: "editor: Helix", category: "Tech",
                       source: "auto", pinned: true, updatedAt: 3, key: "editor"),
        ], cursor: 12), at: Date(timeIntervalSinceReferenceDate: 1_000))
        MemoryCache.save(mirror, serverID: "srv-1", in: folder)
        XCTAssertEqual(MemoryCache.load(serverID: "srv-1", in: folder), mirror)
        XCTAssertEqual(MemoryCache.load(serverID: "srv-2", in: folder), MemoryMirror(),
                       "another server's memories are not this one's")
        MemoryCache.remove(serverID: "srv-1", in: folder)
        XCTAssertEqual(MemoryCache.load(serverID: "srv-1", in: folder), MemoryMirror())
    }

    func testAServerIDCannotNameAPathOutsideTheFolder() {
        let folder = URL(fileURLWithPath: "/tmp/memories", isDirectory: true)
        let url = MemoryCache.file(serverID: "../../etc/passwd", in: folder)
        XCTAssertEqual(url?.deletingLastPathComponent().path, folder.path)
        XCTAssertNil(MemoryCache.file(serverID: "", in: folder))
    }

    func testTheChatStreamsMemoryFrameIsRead() {
        guard case let .memory(cursor)? = SSEStream.decode(Data(#"{"type":"memory","cursor":42}"#.utf8)) else {
            return XCTFail("the memory frame was not decoded")
        }
        XCTAssertEqual(cursor, 42)
        XCTAssertNil(SSEStream.decode(Data(#"{"type":"memory"}"#.utf8)),
                     "a memory frame with no cursor is skipped, not guessed")
    }
}
