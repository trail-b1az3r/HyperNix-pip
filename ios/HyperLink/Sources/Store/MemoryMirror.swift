//  MemoryMirror.swift
//  The phone's own copy of what a server remembers about this person.
//
//  The app used to fetch `/memory/list` whole: the first 200, when the
//  Memories screen appeared or a reply ended, with any error swallowed.
//  So a fact the model wrote mid-chat reached the phone only if somebody
//  happened to be looking, the 201st never did, and with no network the
//  screen was empty.
//
//  Now the phone keeps a copy per server, on disk, and asks
//  `/memory/sync` what changed since its cursor. A delta is applied by id,
//  deletions included; a `full` answer (first sync, a cursor too old, a
//  server restored behind the phone's back) replaces the copy. The copy
//  is shown straight away at launch and whenever the network is gone,
//  and a chat turn that changes a memory says so in a `memory` frame, so
//  the copy follows within the same reply.

import Foundation

/// One answer from `/memory/sync`.
struct MemorySyncPage: Decodable, Sendable {
    /// The whole set, to replace the copy with. False means a delta.
    let full: Bool
    let memories: [MemoryItem]
    /// Memories that are gone. Always empty when `full`.
    let deleted: [String]
    let cursor: Int
    /// Ask again straight away: this page was capped.
    let more: Bool
    /// Why a full set came back: "first", "expired" or "unknown_cursor".
    let reason: String

    enum CodingKeys: String, CodingKey {
        case full, memories, deleted, cursor, more, reason
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        full = try c.decodeIfPresent(Bool.self, forKey: .full) ?? false
        memories = try c.decodeIfPresent([MemoryItem].self, forKey: .memories) ?? []
        deleted = try c.decodeIfPresent([String].self, forKey: .deleted) ?? []
        cursor = try c.decodeIfPresent(Int.self, forKey: .cursor) ?? 0
        more = try c.decodeIfPresent(Bool.self, forKey: .more) ?? false
        reason = try c.decodeIfPresent(String.self, forKey: .reason) ?? ""
    }

    init(full: Bool, memories: [MemoryItem], deleted: [String] = [], cursor: Int,
         more: Bool = false, reason: String = "") {
        self.full = full
        self.memories = memories
        self.deleted = deleted
        self.cursor = cursor
        self.more = more
        self.reason = reason
    }
}

/// What the phone knows of one server's memories, and from when.
struct MemoryMirror: Codable, Equatable, Sendable {
    /// What to send next time. 0 until the first sync.
    private(set) var cursor: Int = 0
    /// Pinned first, then most recently changed: the server's order.
    private(set) var items: [MemoryItem] = []
    private(set) var syncedAt: Date?

    /// Apply one answer. Returns whether what is shown changed.
    @discardableResult
    mutating func apply(_ page: MemorySyncPage, at now: Date = Date()) -> Bool {
        var byID: [String: MemoryItem] = [:]
        if !page.full {
            for item in items { byID[item.memoryID] = item }
        }
        for item in page.memories { byID[item.memoryID] = item }
        for memoryID in page.deleted { byID.removeValue(forKey: memoryID) }
        let sorted = byID.values.sorted(by: MemoryMirror.order)
        let changed = sorted != items
        items = sorted
        cursor = page.cursor
        syncedAt = now
        return changed
    }

    /// Would a sync bring anything new, given the server's cursor from a
    /// chat stream's `memory` frame?
    func isBehind(_ serverCursor: Int) -> Bool {
        cursor == 0 || serverCursor > cursor
    }

    static func order(_ a: MemoryItem, _ b: MemoryItem) -> Bool {
        if a.pinned != b.pinned { return a.pinned }
        if a.updatedAt != b.updatedAt { return a.updatedAt > b.updatedAt }
        return a.memoryID < b.memoryID
    }
}

/// Where each server's mirror is kept: one file per server under
/// Application Support, excluded from backups because the server holds
/// the real copy.
enum MemoryCache {
    static func directory() -> URL? {
        guard let base = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first else { return nil }
        return base.appendingPathComponent("Memories", isDirectory: true)
    }

    /// The file for *serverID*. Anything but letters, digits, `-` and `_`
    /// is replaced, so a server id can never name a path outside the
    /// folder.
    static func file(serverID: String, in directory: URL? = nil) -> URL? {
        guard !serverID.isEmpty, let folder = directory ?? self.directory() else { return nil }
        let safe = String(serverID.map { $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" ? $0 : "_" })
        return folder.appendingPathComponent("\(safe).json")
    }

    static func load(serverID: String, in directory: URL? = nil) -> MemoryMirror {
        guard let url = file(serverID: serverID, in: directory),
              let data = try? Data(contentsOf: url),
              let mirror = try? JSONDecoder().decode(MemoryMirror.self, from: data)
        else { return MemoryMirror() }
        return mirror
    }

    static func save(_ mirror: MemoryMirror, serverID: String, in directory: URL? = nil) {
        guard var url = file(serverID: serverID, in: directory),
              let data = try? JSONEncoder().encode(mirror)
        else { return }
        try? FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        #if os(iOS)
        try? data.write(to: url, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        #else
        try? data.write(to: url, options: .atomic)
        #endif
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? url.setResourceValues(values)
    }

    static func remove(serverID: String, in directory: URL? = nil) {
        guard let url = file(serverID: serverID, in: directory) else { return }
        try? FileManager.default.removeItem(at: url)
    }
}
