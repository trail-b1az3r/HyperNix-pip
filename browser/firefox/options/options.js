/* global browser, HNX */
"use strict"

const $ = (id) => document.getElementById(id)
const DEFAULTS = { provider: "t1", base: "", model: "", keys: {}, textTools: false, policy: "every", maxSteps: 25, search: "duckduckgo", allowPasswords: false }
let current = { ...DEFAULTS }

for (const [id, p] of Object.entries(HNX.PROVIDERS)) {
  $("provider").appendChild(Object.assign(document.createElement("option"), { value: id, textContent: p.label }))
}

function showProvider() {
  const p = HNX.PROVIDERS[$("provider").value]
  $("provider-note").textContent = p.note
  $("base").placeholder = p.base
  $("key").value = (current.keys || {})[$("provider").value] || ""
  $("key").placeholder = p.needsKey ? "required" : "optional"
  $("textTools").disabled = !p.nativeTools
}

function read() {
  const keys = { ...(current.keys || {}), [$("provider").value]: $("key").value.trim() }
  return {
    provider: $("provider").value,
    base: $("base").value.trim(),
    model: $("model").value.trim(),
    keys,
    textTools: $("textTools").checked,
    policy: $("policy").value,
    maxSteps: Math.min(200, Math.max(1, Number($("maxSteps").value) || 25)),
    search: $("search").value,
    allowPasswords: $("allowPasswords").checked,
  }
}

async function load() {
  const { settings = {} } = await browser.storage.local.get("settings")
  current = { ...DEFAULTS, ...settings }
  $("provider").value = current.provider
  $("base").value = current.base
  $("model").value = current.model
  $("textTools").checked = current.textTools
  $("policy").value = current.policy
  $("maxSteps").value = current.maxSteps
  $("search").value = current.search
  $("allowPasswords").checked = current.allowPasswords
  showProvider()
}

$("provider").addEventListener("change", () => {
  current.keys = read().keys
  $("base").value = ""
  showProvider()
})

$("load").addEventListener("click", async () => {
  const s = read()
  const config = { provider: s.provider, base: s.base || HNX.PROVIDERS[s.provider].base, apiKey: s.keys[s.provider] }
  const request = HNX.modelsRequest(config)
  $("load-status").textContent = "asking…"
  try {
    // The server's origin must be allowed before the extension can call it.
    await browser.permissions.request({ origins: [`${new URL(request.url).origin}/*`] })
    const response = await fetch(request.url, request.init)
    const ids = HNX.parseModels(config, await response.json())
    $("models").textContent = ""
    for (const id of ids) $("models").appendChild(Object.assign(document.createElement("option"), { value: id }))
    $("load-status").textContent = ids.length ? `${ids.length} models — pick one in the box above` : "the server listed no models"
  } catch (error) {
    $("load-status").textContent = `could not list models: ${error.message}`
  }
})

$("save").addEventListener("click", async () => {
  current = read()
  await browser.storage.local.set({ settings: current })
  // Pages and the model's server need host access; asked for on a click,
  // as Firefox requires.
  await browser.permissions.request({ origins: ["<all_urls>"] }).catch(() => false)
  $("status").textContent = "Saved"
  setTimeout(() => { $("status").textContent = "" }, 2000)
})

load()
