// Where the model runs: the HyperNix T1 API, a local OpenAI-compatible
// server (llama-server, LM Studio, Ollama), OpenAI, or Anthropic.
//
// Each provider turns the extension's transcript into its own request and
// its reply back into { text, calls, usage }. Pure functions: the
// background script does the fetch, the tests check the shapes.
//
// The transcript is a list of:
//   { role: "user", text }
//   { role: "assistant", text, calls: [{ id, name, args }] }
//   { role: "tool", id, name, result, image? }   (image: a data: URL)

(function (root) {
  "use strict"

  const tools = typeof module === "object" && module.exports ? require("./tools.js") : root.HNX

  const PROVIDERS = {
    t1: {
      label: "HyperNix T1 API",
      base: "http://127.0.0.1:8000",
      nativeTools: false,
      needsKey: true,
      note: "Uses /inference/chat, so the server's registry, quota and prices apply.",
    },
    local: {
      label: "Local model (llama-server, LM Studio, Ollama)",
      base: "http://127.0.0.1:1234/v1",
      nativeTools: true,
      needsKey: false,
      note: "Any OpenAI-compatible server. Turn on text tools for a model without tool calling.",
    },
    openai: {
      label: "OpenAI",
      base: "https://api.openai.com/v1",
      nativeTools: true,
      needsKey: true,
      note: "Your own API key; billed by OpenAI.",
    },
    anthropic: {
      label: "Anthropic (Claude)",
      base: "https://api.anthropic.com/v1",
      nativeTools: true,
      needsKey: true,
      note: "Your own API key; billed by Anthropic.",
    },
  }

  const trimBase = (base) => String(base || "").replace(/\/+$/, "")

  function usesTextTools(config) {
    const p = PROVIDERS[config.provider]
    return !p || !p.nativeTools || Boolean(config.textTools)
  }

  function systemPrompt(config, base) {
    return usesTextTools(config) ? `${base}\n\n${tools.textToolPrompt()}` : base
  }

  // -- the plain-text form, for T1 and models without tool calling -------------

  function textMessages(transcript) {
    const out = []
    for (const item of transcript) {
      if (item.role === "user") out.push({ role: "user", content: item.text })
      else if (item.role === "assistant") {
        const calls = (item.calls || []).map(tools.formatTextToolCall).join("\n")
        out.push({ role: "assistant", content: [item.text, calls].filter(Boolean).join("\n") })
      } else if (item.role === "tool") {
        out.push({ role: "user", content: `<tool_result name="${item.name}">\n${item.result}\n</tool_result>` })
      }
    }
    return out
  }

  // -- OpenAI and OpenAI-compatible ---------------------------------------------

  function openaiMessages(transcript, system) {
    const out = [{ role: "system", content: system }]
    for (const item of transcript) {
      if (item.role === "user") out.push({ role: "user", content: item.text })
      else if (item.role === "assistant") {
        const msg = { role: "assistant", content: item.text || null }
        if (item.calls && item.calls.length) {
          msg.tool_calls = item.calls.map((c) => ({
            id: c.id, type: "function", function: { name: c.name, arguments: JSON.stringify(c.args || {}) },
          }))
        }
        out.push(msg)
      } else if (item.role === "tool") {
        out.push({ role: "tool", tool_call_id: item.id, content: item.result })
        // A tool message cannot carry a picture; the next user turn does.
        if (item.image) {
          out.push({ role: "user", content: [
            { type: "text", text: `The screenshot from ${item.name}:` },
            { type: "image_url", image_url: { url: item.image } },
          ] })
        }
      }
    }
    return out
  }

  // -- Anthropic ------------------------------------------------------------------

  function anthropicMessages(transcript) {
    const out = []
    for (const item of transcript) {
      if (item.role === "user") out.push({ role: "user", content: [{ type: "text", text: item.text }] })
      else if (item.role === "assistant") {
        const content = []
        if (item.text) content.push({ type: "text", text: item.text })
        for (const c of item.calls || []) content.push({ type: "tool_use", id: c.id, name: c.name, input: c.args || {} })
        out.push({ role: "assistant", content })
      } else if (item.role === "tool") {
        const block = { type: "tool_result", tool_use_id: item.id, content: [{ type: "text", text: item.result }] }
        if (item.image) {
          const m = /^data:(image\/[a-z]+);base64,(.*)$/.exec(item.image)
          if (m) block.content.push({ type: "image", source: { type: "base64", media_type: m[1], data: m[2] } })
        }
        // Every result for one assistant turn goes in one user message.
        const last = out[out.length - 1]
        if (last && last.role === "user" && last.content.every((b) => b.type === "tool_result")) last.content.push(block)
        else out.push({ role: "user", content: [block] })
      }
    }
    return out
  }

  // -- requests -------------------------------------------------------------------

  function buildRequest(config, transcript, basePrompt) {
    const base = trimBase(config.base || (PROVIDERS[config.provider] || {}).base)
    const system = systemPrompt(config, basePrompt)
    const headers = { "content-type": "application/json" }
    const text = usesTextTools(config)

    if (config.provider === "anthropic") {
      headers["x-api-key"] = config.apiKey || ""
      headers["anthropic-version"] = "2023-06-01"
      // Anthropic refuses calls from a browser without this; the key is the
      // person's own, kept in this browser's extension storage.
      headers["anthropic-dangerous-direct-browser-access"] = "true"
      const body = { model: config.model, max_tokens: config.maxTokens || 4096, system }
      if (text) {
        body.messages = textMessages(transcript)
      } else {
        body.messages = anthropicMessages(transcript)
        body.tools = tools.TOOLS.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }))
      }
      return { url: `${base}/messages`, init: { method: "POST", headers, body: JSON.stringify(body) } }
    }

    if (config.apiKey) headers.authorization = `Bearer ${config.apiKey}`

    if (config.provider === "t1") {
      const body = { model: config.model, messages: [{ role: "system", content: system }, ...textMessages(transcript)] }
      if (config.maxTokens) body.max_tokens = config.maxTokens
      return { url: `${base}/inference/chat`, init: { method: "POST", headers, body: JSON.stringify(body) } }
    }

    const body = { model: config.model }
    if (text) {
      body.messages = [{ role: "system", content: system }, ...textMessages(transcript)]
    } else {
      body.messages = openaiMessages(transcript, system)
      body.tools = tools.TOOLS.map((t) => ({ type: "function", function: { name: t.name, description: t.description, parameters: t.parameters } }))
      body.tool_choice = "auto"
    }
    if (config.maxTokens && config.provider === "local") body.max_tokens = config.maxTokens
    return { url: `${base}/chat/completions`, init: { method: "POST", headers, body: JSON.stringify(body) } }
  }

  function parseResponse(config, json) {
    if (!json || typeof json !== "object") throw new Error("the model's server sent something that is not JSON")
    if (json.error) {
      const e = json.error
      throw new Error(typeof e === "string" ? e : e.message || JSON.stringify(e))
    }
    let text = ""
    let calls = []
    let usage = {}
    if (config.provider === "anthropic") {
      for (const block of json.content || []) {
        if (block.type === "text") text += block.text
        else if (block.type === "tool_use") calls.push({ id: block.id, name: block.name, args: block.input || {} })
      }
      usage = { input_tokens: json.usage?.input_tokens, output_tokens: json.usage?.output_tokens }
    } else if (config.provider === "t1") {
      text = String(json.content || "")
      usage = { input_tokens: json.input_tokens, output_tokens: json.output_tokens, cost: json.cost, currency: json.currency }
    } else {
      const message = (json.choices && json.choices[0] && json.choices[0].message) || {}
      text = String(message.content || "")
      for (const c of message.tool_calls || []) {
        let args = {}
        try {
          args = typeof c.function.arguments === "string" ? JSON.parse(c.function.arguments || "{}") : c.function.arguments || {}
        } catch (_) {
          args = { _unparsed: c.function.arguments }
        }
        calls.push({ id: c.id || `call_${calls.length + 1}`, name: c.function.name, args })
      }
      usage = { input_tokens: json.usage?.prompt_tokens, output_tokens: json.usage?.completion_tokens }
    }
    if (usesTextTools(config) && !calls.length) {
      const parsed = tools.parseTextToolCalls(text)
      text = parsed.text
      calls = parsed.calls
    }
    return { text, calls, usage }
  }

  // -- listing models ---------------------------------------------------------------

  function modelsRequest(config) {
    const base = trimBase(config.base || (PROVIDERS[config.provider] || {}).base)
    const headers = {}
    if (config.provider === "anthropic") {
      headers["x-api-key"] = config.apiKey || ""
      headers["anthropic-version"] = "2023-06-01"
      headers["anthropic-dangerous-direct-browser-access"] = "true"
    } else if (config.apiKey) {
      headers.authorization = `Bearer ${config.apiKey}`
    }
    return { url: `${base}/models`, init: { method: "GET", headers } }
  }

  function parseModels(config, json) {
    if (config.provider === "t1") return (json.models || []).map((m) => m.model_id).filter(Boolean)
    return (json.data || json.models || []).map((m) => m.id || m.name || m.model).filter(Boolean)
  }

  const api = { PROVIDERS, usesTextTools, buildRequest, parseResponse, modelsRequest, parseModels, openaiMessages, anthropicMessages, textMessages }
  if (typeof module === "object" && module.exports) module.exports = api
  else root.HNX = Object.assign(root.HNX || {}, api)
})(typeof globalThis !== "undefined" ? globalThis : this)
