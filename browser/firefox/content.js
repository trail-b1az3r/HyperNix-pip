// Runs inside a web page, injected only when the model asks to read or use
// that page. Reads the page and its controls, and clicks, types, scrolls and
// presses keys on the person's behalf.
//
// The numbers it hands out live on the elements themselves
// (data-hnx-id) and are renumbered on every read, so a number always means
// what the last read showed.

(function () {
  "use strict"
  if (window.__hnxAgent) return

  const INTERACTIVE = [
    "a[href]", "button", "input:not([type=hidden])", "textarea", "select", "summary",
    "[role=button]", "[role=link]", "[role=checkbox]", "[role=tab]", "[role=menuitem]",
    "[role=option]", "[role=switch]", "[role=textbox]", "[role=combobox]", "[contenteditable=''], [contenteditable=true]",
  ].join(",")

  const squash = (s) => String(s || "").replace(/\s+/g, " ").trim()

  function visible(el) {
    const rect = el.getBoundingClientRect()
    if (rect.width < 1 || rect.height < 1) return false
    const style = getComputedStyle(el)
    return style.visibility !== "hidden" && style.display !== "none" && Number(style.opacity) > 0.05
  }

  function label(el) {
    const id = el.id && document.querySelector(`label[for="${CSS.escape(el.id)}"]`)
    return squash(
      el.getAttribute("aria-label") || (id && id.innerText) || el.innerText || el.getAttribute("placeholder") ||
      el.getAttribute("title") || el.getAttribute("alt") || el.getAttribute("name") ||
      (el.tagName === "INPUT" && /^(submit|button|reset)$/i.test(el.type) ? el.value : ""),
    ).slice(0, 100)
  }

  function describe(el, index) {
    const tag = el.tagName.toLowerCase()
    const out = { id: index, tag, label: label(el) }
    const role = el.getAttribute("role")
    if (role) out.role = role
    if (tag === "input") out.type = (el.type || "text").toLowerCase()
    if (tag === "a") out.href = el.href
    if (el.disabled) out.disabled = true
    if (tag === "input" && /^(checkbox|radio)$/.test(out.type)) out.checked = el.checked
    // Never what is in a password box, and nothing from hidden fields.
    if ((tag === "input" || tag === "textarea") && out.type !== "password" && el.value) out.value = String(el.value).slice(0, 200)
    if (out.type === "password") out.password = true
    if (tag === "select") out.value = el.options[el.selectedIndex] ? squash(el.options[el.selectedIndex].text) : ""
    return out
  }

  function snapshot({ maxChars = 12000, maxElements = 250 } = {}) {
    document.querySelectorAll("[data-hnx-id]").forEach((el) => el.removeAttribute("data-hnx-id"))
    const elements = []
    for (const el of document.querySelectorAll(INTERACTIVE)) {
      if (elements.length >= maxElements) break
      if (!visible(el)) continue
      const index = elements.length + 1
      el.setAttribute("data-hnx-id", String(index))
      elements.push(describe(el, index))
    }
    const text = squash(document.body ? document.body.innerText : "")
    return {
      title: document.title,
      url: location.href,
      text: text.slice(0, maxChars),
      truncated: text.length > maxChars,
      elements,
      scroll: { y: Math.round(scrollY), height: document.documentElement.scrollHeight, view: innerHeight },
    }
  }

  function find(index) {
    const el = document.querySelector(`[data-hnx-id="${Number(index)}"]`)
    if (!el) throw new Error(`there is no element ${index} -- read the page again, the numbers change`)
    return el
  }

  function click({ element }) {
    const el = find(element)
    el.scrollIntoView({ block: "center", inline: "center" })
    for (const type of ["pointerdown", "mousedown", "pointerup", "mouseup"]) {
      el.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, view: window }))
    }
    el.click()
    return `clicked ${element} (${label(el) || el.tagName.toLowerCase()})`
  }

  function setNativeValue(el, value) {
    const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype
      : el instanceof HTMLSelectElement ? HTMLSelectElement.prototype : HTMLInputElement.prototype
    const setter = Object.getOwnPropertyDescriptor(proto, "value").set
    // The prototype's setter, so frameworks that track the value (React) see the change.
    setter.call(el, value)
    el.dispatchEvent(new Event("input", { bubbles: true }))
    el.dispatchEvent(new Event("change", { bubbles: true }))
  }

  function keyEvents(el, key) {
    const init = { key, code: key.length === 1 ? `Key${key.toUpperCase()}` : key, bubbles: true, cancelable: true }
    if (key === "Enter") init.keyCode = 13
    const down = el.dispatchEvent(new KeyboardEvent("keydown", init))
    el.dispatchEvent(new KeyboardEvent("keypress", init))
    el.dispatchEvent(new KeyboardEvent("keyup", init))
    return down
  }

  function type({ element, text, submit = false, clear = true, allowPasswords = false }) {
    const el = find(element)
    if (el.type === "password" && !allowPasswords) {
      throw new Error("typing into password fields is turned off in the extension's settings")
    }
    el.scrollIntoView({ block: "center" })
    el.focus()
    if (el.isContentEditable) {
      if (clear) el.textContent = ""
      if (!document.execCommand("insertText", false, text)) el.textContent += text
      el.dispatchEvent(new InputEvent("input", { bubbles: true, data: text }))
    } else {
      setNativeValue(el, clear ? text : `${el.value || ""}${text}`)
    }
    if (submit) {
      const notCancelled = keyEvents(el, "Enter")
      if (notCancelled && el.form) {
        if (el.form.requestSubmit) el.form.requestSubmit()
        else el.form.submit()
      }
    }
    return `typed ${text.length} characters into ${element}${submit ? " and submitted" : ""}`
  }

  function pressKey({ key, element }) {
    const el = element ? find(element) : document.activeElement || document.body
    keyEvents(el, key)
    if (key === "Enter" && el.form && el.tagName === "INPUT") {
      if (el.form.requestSubmit) el.form.requestSubmit()
      else el.form.submit()
    }
    return `pressed ${key}`
  }

  function select({ element, value }) {
    const el = find(element)
    if (!(el instanceof HTMLSelectElement)) throw new Error(`element ${element} is not a drop-down`)
    const option = [...el.options].find((o) => o.value === value || squash(o.text).toLowerCase() === squash(value).toLowerCase())
    if (!option) throw new Error(`no option "${value}"; there are: ${[...el.options].map((o) => squash(o.text)).join(", ")}`)
    setNativeValue(el, option.value)
    return `chose "${squash(option.text)}"`
  }

  function scroll({ direction, pages = 1 }) {
    const step = innerHeight * 0.85 * (Number(pages) || 1)
    if (direction === "top") scrollTo(0, 0)
    else if (direction === "bottom") scrollTo(0, document.documentElement.scrollHeight)
    else scrollBy(0, direction === "up" ? -step : step)
    return `scrolled ${direction}; now at ${Math.round(scrollY)} of ${document.documentElement.scrollHeight}`
  }

  function findText({ text }) {
    const needle = squash(text).toLowerCase()
    if (!needle) throw new Error("nothing to find")
    const lines = (document.body ? document.body.innerText : "").split(/\n+/).map(squash).filter((l) => l.toLowerCase().includes(needle))
    const elements = [...document.querySelectorAll("[data-hnx-id]")]
      .filter((el) => label(el).toLowerCase().includes(needle))
      .map((el) => describe(el, Number(el.getAttribute("data-hnx-id"))))
    return { lines: lines.slice(0, 40), elements: elements.slice(0, 40) }
  }

  const COMMANDS = { snapshot, click, type, key: pressKey, select, scroll, find: findText }

  window.__hnxAgent = COMMANDS
  if (typeof browser !== "undefined" && browser.runtime && browser.runtime.onMessage) {
    browser.runtime.onMessage.addListener((message) => {
      if (!message || message.hnx !== true || !COMMANDS[message.cmd]) return undefined
      try {
        return Promise.resolve({ ok: true, data: COMMANDS[message.cmd](message.args || {}) })
      } catch (error) {
        return Promise.resolve({ ok: false, error: error.message || String(error) })
      }
    })
  }
})()
