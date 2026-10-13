// The tools a model is given to use the browser, and the plain-text
// protocol for models that cannot call tools natively.
//
// Loaded as a classic script in the extension (it sets globalThis.HNX)
// and with require() by the tests.

(function (root) {
  "use strict"

  const element = { type: "integer", description: "The element's number, from the last read_page" }

  // `acts`: the tool changes something -- a page, a form, which page is
  // open -- so the extension may ask the person first.
  // `typing`: one of the actions the "ask before typing" setting covers.
  const TOOLS = [
    {
      name: "search_web",
      description: "Search the web. Returns numbered results with title, address and snippet.",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "What to search for" },
          max_results: { type: "integer", description: "How many results (default 8, at most 20)" },
        },
        required: ["query"],
      },
    },
    {
      name: "open_url",
      acts: true,
      description: "Open an http(s) address in the current tab, or a new one. Waits for it to load.",
      parameters: {
        type: "object",
        properties: {
          url: { type: "string" },
          new_tab: { type: "boolean", description: "Open in a new tab (default false)" },
        },
        required: ["url"],
      },
    },
    {
      name: "list_tabs",
      description: "The tabs in the current window: id, title, address, which is active.",
      parameters: { type: "object", properties: {} },
    },
    {
      name: "switch_tab",
      description: "Make another tab the current one.",
      parameters: { type: "object", properties: { tab_id: { type: "integer" } }, required: ["tab_id"] },
    },
    {
      name: "read_page",
      description:
        "Read the current page: its title, address, visible text and a numbered list of the things " +
        "on it that can be clicked or typed into. Read it again after the page changes: numbers are " +
        "only valid until the next read.",
      parameters: {
        type: "object",
        properties: { max_chars: { type: "integer", description: "How much text (default 12000)" } },
      },
    },
    {
      name: "find_in_page",
      description: "Find text on the current page: the lines that contain it, and matching elements.",
      parameters: { type: "object", properties: { text: { type: "string" } }, required: ["text"] },
    },
    {
      name: "click",
      acts: true,
      description: "Click an element on the current page.",
      parameters: { type: "object", properties: { element }, required: ["element"] },
    },
    {
      name: "type_text",
      acts: true,
      typing: true,
      description: "Type into a text box on the current page, optionally pressing Enter after.",
      parameters: {
        type: "object",
        properties: {
          element,
          text: { type: "string" },
          submit: { type: "boolean", description: "Press Enter / submit the form afterwards" },
          clear: { type: "boolean", description: "Replace what is there (default true)" },
        },
        required: ["element", "text"],
      },
    },
    {
      name: "press_key",
      acts: true,
      typing: true,
      description: "Press a key (Enter, Escape, Tab, ArrowDown, …) on an element or the page.",
      parameters: {
        type: "object",
        properties: { key: { type: "string" }, element },
        required: ["key"],
      },
    },
    {
      name: "select_option",
      acts: true,
      typing: true,
      description: "Choose an option in a drop-down, by its value or visible text.",
      parameters: {
        type: "object",
        properties: { element, value: { type: "string" } },
        required: ["element", "value"],
      },
    },
    {
      name: "scroll",
      description: "Scroll the current page.",
      parameters: {
        type: "object",
        properties: {
          direction: { type: "string", enum: ["up", "down", "top", "bottom"] },
          pages: { type: "number", description: "How many screens (default 1)" },
        },
        required: ["direction"],
      },
    },
    {
      name: "go_back",
      acts: true,
      description: "Go back to the previous page in the current tab.",
      parameters: { type: "object", properties: {} },
    },
    {
      name: "wait",
      description: "Wait for a page to finish something, up to 10 seconds.",
      parameters: { type: "object", properties: { seconds: { type: "number" } }, required: ["seconds"] },
    },
    {
      name: "screenshot",
      description: "A picture of the visible part of the current tab (only for models that can see images).",
      parameters: { type: "object", properties: {} },
    },
  ]

  const byName = (name) => TOOLS.find((t) => t.name === name)

  // Whether `call` needs the person's go-ahead under `policy`:
  // "every" (every acting tool), "typing" (typing, keys and choices) or "never".
  function needsConfirmation(call, policy) {
    const tool = byName(call.name)
    if (!tool || !tool.acts || policy === "never") return false
    if (policy === "typing") return Boolean(tool.typing)
    return true
  }

  // -- the plain-text protocol -------------------------------------------------

  function textToolPrompt(tools = TOOLS) {
    const lines = tools.map((t) => {
      const props = Object.entries(t.parameters.properties || {})
        .map(([k, v]) => `${k}${(t.parameters.required || []).includes(k) ? "" : "?"}: ${v.type}`)
        .join(", ")
      return `- ${t.name}(${props}): ${t.description}`
    })
    return [
      "You can use these tools:",
      ...lines,
      "",
      "To use one, reply with exactly one block like this and nothing after it:",
      '<tool_call>{"name": "read_page", "arguments": {}}</tool_call>',
      "The result comes back in the next message. When the task is done, reply normally, with no tool_call block.",
    ].join("\n")
  }

  // Tool calls in a model's text: <tool_call>{...}</tool_call> blocks, or a
  // ```tool / ```json fence holding {"name", "arguments"}. Returns the text
  // with the calls taken out, and the calls.
  function parseTextToolCalls(text) {
    const calls = []
    let rest = String(text || "")
    const take = (raw) => {
      try {
        const obj = JSON.parse(raw.trim())
        if (obj && typeof obj.name === "string") {
          const args = obj.arguments ?? obj.parameters ?? obj.input ?? {}
          calls.push({ id: `call_${calls.length + 1}`, name: obj.name, args: typeof args === "string" ? JSON.parse(args) : args })
          return true
        }
      } catch (_) {
        // Not a call: left in the text, where the person can see it.
      }
      return false
    }
    rest = rest.replace(/<tool_call>([\s\S]*?)<\/tool_call>/g, (whole, inner) => (take(inner) ? "" : whole))
    rest = rest.replace(/```(?:tool|json|tool_call)?\s*\n([\s\S]*?)```/g, (whole, inner) =>
      /"name"\s*:/.test(inner) && take(inner) ? "" : whole,
    )
    return { text: rest.trim(), calls }
  }

  function formatTextToolCall(call) {
    return `<tool_call>${JSON.stringify({ name: call.name, arguments: call.args || {} })}</tool_call>`
  }

  const api = { TOOLS, byName, needsConfirmation, textToolPrompt, parseTextToolCalls, formatTextToolCall }
  if (typeof module === "object" && module.exports) module.exports = api
  else root.HNX = Object.assign(root.HNX || {}, api)
})(typeof globalThis !== "undefined" ? globalThis : this)
