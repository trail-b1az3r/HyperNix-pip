// The extension's background page: settings, the agent loop, and the
// browser tools themselves. The sidebar talks to it over a port.

/* global HNX, browser */
"use strict"

const DEFAULTS = {
  provider: "t1",
  base: "",
  model: "",
  keys: {},              // provider -> API key, in this browser's local storage only
  textTools: false,
  policy: "every",       // ask before: "every" action, "typing" only, or "never"
  maxSteps: 25,
  maxTokens: 0,
  allowPasswords: false,
  search: "duckduckgo",  // or "t1": the T1 API's own /web/v1/search
}

async function settings() {
  const stored = await browser.storage.local.get("settings")
  return { ...DEFAULTS, ...(stored.settings || {}) }
}

function providerConfig(s) {
  return {
    provider: s.provider,
    base: s.base || HNX.PROVIDERS[s.provider].base,
    apiKey: (s.keys || {})[s.provider] || "",
    model: s.model,
    textTools: s.textTools,
    maxTokens: Number(s.maxTokens) || 0,
  }
}

async function fetchJson(url, init, timeoutMs = 120000) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const response = await fetch(url, { ...init, signal: controller.signal })
    const text = await response.text()
    let json
    try {
      json = JSON.parse(text)
    } catch (_) {
      throw new Error(`${response.status} from ${new URL(url).host}: ${text.slice(0, 300)}`)
    }
    if (!response.ok) {
      const e = json.error
      throw new Error(`${response.status} from ${new URL(url).host}: ${typeof e === "string" ? e : e?.message || JSON.stringify(json).slice(0, 300)}`)
    }
    return json
  } finally {
    clearTimeout(timer)
  }
}

// -- the tab the agent works in ------------------------------------------------

let agentTabId = null

async function currentTab() {
  if (agentTabId !== null) {
    try {
      return await browser.tabs.get(agentTabId)
    } catch (_) {
      agentTabId = null
    }
  }
  const [tab] = await browser.tabs.query({ active: true, lastFocusedWindow: true })
  if (!tab) throw new Error("there is no tab to work in")
  agentTabId = tab.id
  return tab
}

function waitForLoad(tabId, timeoutMs = 20000) {
  return new Promise((resolve) => {
    const done = () => {
      browser.tabs.onUpdated.removeListener(listener)
      clearTimeout(timer)
      resolve()
    }
    const listener = (id, info) => {
      if (id === tabId && info.status === "complete") done()
    }
    const timer = setTimeout(done, timeoutMs)
    browser.tabs.onUpdated.addListener(listener)
  })
}

async function inPage(cmd, args = {}) {
  const tab = await currentTab()
  if (!/^https?:/.test(tab.url || "")) throw new Error(`the current tab (${tab.url || "a blank tab"}) is not a web page; open one first`)
  await browser.scripting.executeScript({ target: { tabId: tab.id }, files: ["content.js"] })
  const reply = await browser.tabs.sendMessage(tab.id, { hnx: true, cmd, args })
  if (!reply || !reply.ok) throw new Error(reply ? reply.error : "the page did not answer")
  return reply.data
}

// -- the tools ---------------------------------------------------------------------

async function searchWeb({ query, max_results: max = 8 }, s) {
  const limit = Math.min(20, Math.max(1, Number(max) || 8))
  if (s.search === "t1") {
    const cfg = providerConfig({ ...s, provider: "t1" })
    const url = `${cfg.base.replace(/\/+$/, "")}/web/v1/search?q=${encodeURIComponent(query)}&depth=1`
    const json = await fetchJson(url, { headers: cfg.apiKey ? { authorization: `Bearer ${cfg.apiKey}` } : {} })
    return HNX.formatResults(query, (json.hits || []).slice(0, limit), "T1 web search")
  }
  const response = await fetch(`https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`, { credentials: "omit" })
  return HNX.formatResults(query, HNX.parseDuckDuckGo(await response.text(), limit), "DuckDuckGo")
}

const EXECUTORS = {
  search_web: async (args, s) => ({ result: await searchWeb(args, s) }),
  open_url: async ({ url, new_tab: newTab }) => {
    const href = HNX.safeUrl(url)
    let tab
    if (newTab) {
      tab = await browser.tabs.create({ url: href, active: true })
      agentTabId = tab.id
    } else {
      tab = await currentTab()
      await browser.tabs.update(tab.id, { url: href })
    }
    await waitForLoad(tab.id)
    const now = await browser.tabs.get(tab.id)
    return { result: `Opened ${now.url} — "${now.title}". Use read_page to see it.` }
  },
  list_tabs: async () => {
    const tabs = await browser.tabs.query({ lastFocusedWindow: true })
    return { result: tabs.map((t) => `${t.id}${t.active ? " (current)" : ""}: ${t.title} — ${t.url}`).join("\n") }
  },
  switch_tab: async ({ tab_id: id }) => {
    const tab = await browser.tabs.update(Number(id), { active: true })
    agentTabId = tab.id
    return { result: `Now on tab ${tab.id}: ${tab.title}` }
  },
  read_page: async ({ max_chars: maxChars }) => ({ result: HNX.formatSnapshot(await inPage("snapshot", { maxChars: Number(maxChars) || 12000 })) }),
  find_in_page: async ({ text }) => {
    const found = await inPage("find", { text })
    return { result: [`Lines containing "${text}":`, ...found.lines.map((l) => `  ${l}`),
      "Matching elements:", ...found.elements.map((e) => `  ${HNX.formatElement(e)}`)].join("\n") }
  },
  click: async ({ element }) => {
    const tab = await currentTab()
    const result = await inPage("click", { element })
    await new Promise((r) => setTimeout(r, 600))
    const now = await browser.tabs.get(tab.id)
    if (now.status === "loading") await waitForLoad(tab.id)
    return { result: `${result}. The page is now ${now.url}.` }
  },
  type_text: async (args, s) => ({ result: await inPage("type", { ...args, allowPasswords: s.allowPasswords }) }),
  press_key: async (args) => ({ result: await inPage("key", args) }),
  select_option: async (args) => ({ result: await inPage("select", args) }),
  scroll: async (args) => ({ result: await inPage("scroll", args) }),
  go_back: async () => {
    const tab = await currentTab()
    await browser.tabs.goBack(tab.id)
    await waitForLoad(tab.id, 10000)
    return { result: `Went back; now on ${(await browser.tabs.get(tab.id)).url}` }
  },
  wait: async ({ seconds }) => {
    const ms = Math.min(10, Math.max(0, Number(seconds) || 1)) * 1000
    await new Promise((r) => setTimeout(r, ms))
    return { result: `Waited ${ms / 1000}s.` }
  },
  screenshot: async () => {
    const tab = await currentTab()
    const image = await browser.tabs.captureVisibleTab(tab.windowId, { format: "png" })
    return { result: `A screenshot of ${tab.url} is attached.`, image }
  },
}

// -- the sidebar connection -------------------------------------------------------

let running = null
const pending = new Map()

function describeCall(call) {
  const a = call.args || {}
  switch (call.name) {
    case "open_url": return `open ${a.url}${a.new_tab ? " in a new tab" : ""}`
    case "click": return `click element ${a.element}`
    case "type_text": return `type "${a.text}" into element ${a.element}${a.submit ? " and submit" : ""}`
    case "press_key": return `press ${a.key}${a.element ? ` on element ${a.element}` : ""}`
    case "select_option": return `choose "${a.value}" in element ${a.element}`
    case "go_back": return "go back a page"
    default: return `${call.name} ${JSON.stringify(a)}`
  }
}

browser.runtime.onConnect.addListener((port) => {
  if (port.name !== "hnx-sidebar") return
  const send = (message) => {
    try {
      port.postMessage(message)
    } catch (_) {
      // The sidebar closed; the task carries on and its result is dropped.
    }
  }

  port.onMessage.addListener(async (message) => {
    if (message.type === "confirm-reply") {
      const resolve = pending.get(message.id)
      pending.delete(message.id)
      if (resolve) resolve(message)
      return
    }
    if (message.type === "stop") {
      if (running) running.abort()
      for (const resolve of pending.values()) resolve({ allow: false })
      pending.clear()
      return
    }
    if (message.type === "reset") {
      transcript.length = 0
      agentTabId = null
      send({ type: "reset-done" })
      return
    }
    if (message.type !== "run" || running) return

    const s = await settings()
    const config = providerConfig(s)
    if (!config.model) {
      send({ type: "event", kind: "error", text: "Choose a model in the extension's settings first." })
      return
    }
    running = new AbortController()
    let allowRest = false
    let confirmations = 0
    try {
      await HNX.runAgent({
        task: message.task,
        transcript,
        policy: s.policy,
        maxSteps: Number(s.maxSteps) || 25,
        signal: running.signal,
        call: async (items) => {
          const request = HNX.buildRequest(config, items, HNX.SYSTEM_PROMPT)
          return HNX.parseResponse(config, await fetchJson(request.url, request.init))
        },
        execute: (call) => EXECUTORS[call.name](call.args || {}, s),
        confirm: async (call) => {
          if (allowRest) return true
          const id = ++confirmations
          const tab = await currentTab().catch(() => null)
          send({ type: "confirm", id, call, text: describeCall(call), site: tab ? tab.url : "" })
          const reply = await new Promise((resolve) => pending.set(id, resolve))
          if (reply.always) allowRest = true
          return Boolean(reply.allow)
        },
        onEvent: (event) => send({ type: "event", ...event }),
      })
    } catch (error) {
      send({ type: "event", kind: "error", text: error.message || String(error) })
    } finally {
      running = null
    }
  })
})

// One conversation per browser session, kept until the sidebar resets it.
const transcript = []

browser.action.onClicked.addListener(() => browser.sidebarAction.toggle())
