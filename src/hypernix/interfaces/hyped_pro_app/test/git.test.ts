import { describe, expect, test } from "bun:test"

import { branchBadge, diffStats, logLines, parseGit, splitDiff, statusLines, type RepoStatus } from "../src/git.ts"

const clean: RepoStatus = {
  root: "/r", branch: "main", upstream: "origin/main", ahead: 0, behind: 0, detached: false, clean: true, files: [],
}

const dirty: RepoStatus = {
  ...clean,
  clean: false,
  ahead: 2,
  behind: 1,
  files: [
    { path: "a.py", index: ".", worktree: "M", original: null, staged: false, unstaged: true, label: "modified" },
    { path: "b.py", index: "A", worktree: ".", original: null, staged: true, unstaged: false, label: "added" },
    { path: "c.py", index: "R", worktree: ".", original: "old.py", staged: true, unstaged: false, label: "renamed" },
    { path: "new.txt", index: "?", worktree: "?", original: null, staged: false, unstaged: true, label: "untracked" },
  ],
}

describe("badge", () => {
  test("nothing outside a repository", () => expect(branchBadge(null)).toBe(""))
  test("clean", () => expect(branchBadge(clean)).toBe("main"))
  test("ahead, behind and changed", () => expect(branchBadge(dirty)).toBe("main ↑2 ↓1 ●4"))
  test("detached", () => expect(branchBadge({ ...clean, branch: null, detached: true })).toBe("detached"))
})

describe("status", () => {
  test("clean says so", () => {
    expect(statusLines(clean).join("\n")).toContain("working tree clean")
  })

  test("groups staged, unstaged and untracked", () => {
    const text = statusLines(dirty).join("\n")
    expect(text).toContain("2 ahead, 1 behind")
    const staged = text.indexOf("Staged")
    const notStaged = text.indexOf("Not staged")
    const untracked = text.indexOf("Untracked")
    expect(staged).toBeGreaterThan(-1)
    expect(notStaged).toBeGreaterThan(staged)
    expect(untracked).toBeGreaterThan(notStaged)
    expect(text).toContain("c.py  (from old.py)")
    // An untracked file is listed once, not also as "not staged".
    expect(text.split("new.txt").length - 1).toBe(1)
  })
})

describe("diffs", () => {
  const diff = [
    "diff --git a/a.py b/a.py",
    "--- a/a.py",
    "+++ b/a.py",
    "@@ -1 +1,2 @@",
    "-one",
    "+two",
    "+three",
    "diff --git a/dir/b b.txt b/dir/b b.txt",
    "--- a/dir/b b.txt",
    "+++ b/dir/b b.txt",
    "@@ -1 +1 @@",
    "-x",
    "+y",
    "",
  ].join("\n")

  test("stats ignore the file headers", () => {
    expect(diffStats(diff)).toEqual({ files: 2, added: 3, removed: 2 })
  })

  test("split per file, keeping paths with spaces", () => {
    const parts = splitDiff(diff)
    expect(parts.map((p) => p.path)).toEqual(["a.py", "dir/b b.txt"])
    expect(parts[0].diff.startsWith("diff --git a/a.py")).toBe(true)
    expect(parts[0].diff).not.toContain("dir/b")
    expect(parts[1].diff.endsWith("+y\n")).toBe(true)
  })

  test("nothing in, nothing out", () => {
    expect(splitDiff("")).toEqual([])
  })
})

describe("log", () => {
  test("empty", () => expect(logLines([])).toEqual(["No commits yet."]))
  test("aligned", () => {
    const lines = logLines([
      { hash: "1", short: "abc1234", author: "A", when: "2 hours ago", subject: "one" },
      { hash: "2", short: "def5678", author: "B", when: "3 days ago", subject: "two" },
    ])
    expect(lines[0]).toBe("abc1234  2 hours ago  one  — A")
    expect(lines[1].indexOf("two")).toBe(lines[0].indexOf("one"))
  })
})

describe("parseGit", () => {
  test("bare /git is status", () => expect(parseGit("")).toEqual({ kind: "status" }))
  test("help", () => expect(parseGit("help")).toEqual({ kind: "help" }))
  test("diff", () => {
    expect(parseGit("diff")).toEqual({ kind: "diff", staged: false, path: undefined })
    expect(parseGit("diff --staged src/a.py")).toEqual({ kind: "diff", staged: true, path: "src/a.py" })
  })
  test("log with a count and a path", () => {
    expect(parseGit("log 5 src")).toEqual({ kind: "log", count: 5, path: "src" })
    expect(parseGit("log 99999")).toMatchObject({ count: 200 })
  })
  test("commit keeps the whole message", () => {
    expect(parseGit("commit Fix the  thing")).toEqual({ kind: "commit", message: "Fix the  thing", all: false })
    expect(parseGit("commit -a -m tidy up")).toEqual({ kind: "commit", message: "tidy up", all: true })
    expect(parseGit('commit -m "quoted message"')).toEqual({ kind: "commit", message: "quoted message", all: false })
  })
  test("only leading flags are flags", () => {
    expect(parseGit("commit remove the -a flag")).toEqual({ kind: "commit", message: "remove the -a flag", all: false })
  })
  test("commit needs a message", () => {
    expect(parseGit("commit").kind).toBe("error")
    expect(parseGit("commit -a").kind).toBe("error")
  })
  test("switch", () => {
    expect(parseGit("switch -c feature")).toEqual({ kind: "switch", branch: "feature", create: true })
    expect(parseGit("checkout main")).toEqual({ kind: "switch", branch: "main", create: false })
    expect(parseGit("switch").kind).toBe("error")
  })
  test("add, unstage and restore need paths", () => {
    expect(parseGit("add .")).toEqual({ kind: "add", paths: ["."] })
    expect(parseGit("add").kind).toBe("error")
    expect(parseGit("unstage a b")).toEqual({ kind: "unstage", paths: ["a", "b"] })
    expect(parseGit("restore").kind).toBe("error")
  })
  test("push and pull", () => {
    expect(parseGit("push -u")).toEqual({ kind: "push", setUpstream: true })
    expect(parseGit("pull --rebase")).toEqual({ kind: "pull", rebase: true })
  })
  test("unknown", () => {
    const got = parseGit("rebase -i")
    expect(got.kind).toBe("error")
  })
})
