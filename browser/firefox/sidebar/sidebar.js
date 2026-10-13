/* global browser */
"use strict"

const $ = (id) => document.getElementById(id)
const log = $("log")
const port = browser.runtime.connect({ name: "hnx-sidebar" })

function entry(kind, text) {
  const div = document.createElement("div")
  div.className = `entry ${kind}`
  div.textContent = text
  log.appendChild(div)
  log.scrollTop = log.scrollHeight
  return div
}

function busy(on) {
  $("run").disabled = on
  $("stop").disabled = !on
  $("task").disabled = on
}

function confirmCard(message) {
  const card = entry("confirm", "")
  const what = document.createElement("p")
  what.textContent = `The model wants to ${message.text}`
  const where = document.createElement("p")
  where.className = "dim"
  where.textContent = message.site ? `on ${message.site}` : ""
  const row = document.createElement("div")
  row.className = "row"
  const answer = (allow, always = false) => {
    port.postMessage({ type: "confirm-reply", id: message.id, allow, always })
    row.remove()
    card.appendChild(Object.assign(document.createElement("p"), {
      className: "dim", textContent: allow ? (always ? "Allowed, and the rest of this task" : "Allowed") : "Declined",
    }))
  }
  for (const [label, allow, always] of [["Allow", true, false], ["Allow for this task", true, true], ["Decline", false, false]]) {
    const b = document.createElement("button")
    b.textContent = label
    if (!allow) b.className = "quiet"
    b.addEventListener("click", () => answer(allow, always))
    row.appendChild(b)
  }
  card.append(what, where, row)
}

port.onMessage.addListener((message) => {
  if (message.type === "confirm") return confirmCard(message)
  if (message.type === "reset-done") {
    log.textContent = ""
    return entry("note", "New task: the conversation was cleared.")
  }
  if (message.type !== "event") return undefined
  switch (message.kind) {
    case "assistant":
      if (message.text) entry("assistant", message.text)
      for (const call of message.calls || []) entry("call", `→ ${call.name} ${JSON.stringify(call.args || {})}`)
      break
    case "tool":
      entry("tool", message.result.length > 400 ? `${message.result.slice(0, 400)}…` : message.result)
      break
    case "declined":
      entry("note", `Declined: ${message.call.name}`)
      break
    case "done":
    case "limit":
    case "stopped": {
      const t = message.totals || {}
      entry("note", `${message.kind === "done" ? "Done" : message.text} · ${t.steps || 0} steps · ${t.input_tokens || 0} in / ${t.output_tokens || 0} out tokens`)
      busy(false)
      break
    }
    case "error":
      entry("error", message.text)
      busy(false)
      break
    default:
      break
  }
  return undefined
})

$("ask").addEventListener("submit", (event) => {
  event.preventDefault()
  const task = $("task").value.trim()
  if (!task) return
  entry("user", task)
  $("task").value = ""
  busy(true)
  port.postMessage({ type: "run", task })
})
$("task").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault()
    $("ask").requestSubmit()
  }
})
$("stop").addEventListener("click", () => port.postMessage({ type: "stop" }))
$("reset").addEventListener("click", () => port.postMessage({ type: "reset" }))
$("settings").addEventListener("click", (event) => {
  event.preventDefault()
  browser.runtime.openOptionsPage()
})

const ORIGINS = { origins: ["<all_urls>"] }
async function checkAccess() {
  $("access").hidden = await browser.permissions.contains(ORIGINS)
}
$("grant").addEventListener("click", async () => {
  await browser.permissions.request(ORIGINS)
  checkAccess()
})

async function showModel() {
  const { settings = {} } = await browser.storage.local.get("settings")
  $("model").textContent = settings.model ? `${settings.model} · ${settings.provider || "t1"}` : "no model chosen"
}

browser.storage.onChanged.addListener(showModel)
checkAccess()
showModel()
