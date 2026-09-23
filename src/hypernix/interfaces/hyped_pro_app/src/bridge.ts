// The Python side of hyped-pro.
//
// Everything real — models, keys, downloads, the T1 API, Noodle — lives in
// hypernix.interfaces.hyped_pro_bridge, which hyped-plus talks to as well.
// The protocol is one JSON object per line each way:
//
//   -> {"id": 7, "cmd": "chat", ...}
//   <- {"id": 7, "ok": true, "data": {...}}
//   <- {"id": 7, "ok": false, "code": "HPC-...", "error": "..."}
//
// Replies arrive in any order (long commands run on their own thread in
// the bridge), so each request waits on its own id.

export interface BridgeFailure {
  code: string
  error: string
}

export class BridgeError extends Error {
  readonly code: string
  constructor(code: string, message: string) {
    super(message)
    this.name = "BridgeError"
    this.code = code
  }
}

// What the bridge needs from a process. Split out so the tests can drive
// the protocol without spawning Python.
export interface Transport {
  write(line: string): void
  close(): void
}

type Pending = {
  resolve: (data: any) => void
  reject: (error: BridgeError) => void
}

export class Bridge {
  private nextId = 1
  private pending = new Map<number, Pending>()
  private buffer = ""
  private closed = false
  // Last lines the bridge wrote to stderr: download progress, warnings.
  // Shown in the status bar instead of being printed over the screen.
  readonly log: string[] = []
  onLog: ((line: string) => void) | null = null

  constructor(private transport: Transport) {}

  request<T = any>(cmd: string, fields: Record<string, unknown> = {}): { id: number; done: Promise<T> } {
    const id = this.nextId++
    const done = new Promise<T>((resolve, reject) => {
      if (this.closed) {
        reject(new BridgeError("HPO-BRIDGE-001", "the Python bridge is not running"))
        return
      }
      this.pending.set(id, { resolve, reject })
    })
    if (!this.closed) this.transport.write(JSON.stringify({ ...fields, id, cmd }) + "\n")
    return { id, done }
  }

  call<T = any>(cmd: string, fields: Record<string, unknown> = {}): Promise<T> {
    return this.request<T>(cmd, fields).done
  }

  // Ask the bridge to stop a running request. The request still settles:
  // a cancelled chat resolves with whatever it had, marked cancelled.
  cancel(id: number): Promise<unknown> {
    return this.call("cancel", { target: id })
  }

  get inFlight(): number {
    return this.pending.size
  }

  // Feed raw stdout text. Lines can be split across chunks.
  feed(chunk: string): void {
    this.buffer += chunk
    let newline = this.buffer.indexOf("\n")
    while (newline >= 0) {
      const line = this.buffer.slice(0, newline).trim()
      this.buffer = this.buffer.slice(newline + 1)
      if (line) this.receive(line)
      newline = this.buffer.indexOf("\n")
    }
  }

  feedLog(chunk: string): void {
    for (const raw of chunk.split("\n")) {
      const line = raw.trimEnd()
      if (!line) continue
      this.log.push(line)
      if (this.log.length > 200) this.log.shift()
      this.onLog?.(line)
    }
  }

  private receive(line: string): void {
    let message: any
    try {
      message = JSON.parse(line)
    } catch {
      // Not ours: something in the bridge printed to stdout. Keep it
      // where it can be read rather than dropping it.
      this.feedLog(line)
      return
    }
    const waiting = this.pending.get(message?.id)
    if (!waiting) return
    this.pending.delete(message.id)
    if (message.ok) waiting.resolve(message.data)
    else waiting.reject(new BridgeError(String(message.code ?? "HPO-BRIDGE-002"), String(message.error ?? "unknown error")))
  }

  // The process went away. Everything still waiting fails now rather than
  // hanging the screen forever.
  closedBy(reason: string): void {
    if (this.closed) return
    this.closed = true
    for (const [, waiting] of this.pending) waiting.reject(new BridgeError("HPO-BRIDGE-001", reason))
    this.pending.clear()
  }

  close(): void {
    this.transport.close()
    this.closedBy("hyped-pro closed the bridge")
  }
}

async function pump(stream: ReadableStream<Uint8Array>, sink: (text: string) => void): Promise<void> {
  const decoder = new TextDecoder()
  const reader = stream.getReader()
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    sink(decoder.decode(value, { stream: true }))
  }
  const rest = decoder.decode()
  if (rest) sink(rest)
}

// HYPED_PRO_PYTHON is set by the `hyped-pro` launcher to the interpreter
// that has hypernix installed, which is not necessarily `python3`.
export function pythonCommand(env: Record<string, string | undefined> = process.env): string[] {
  const python = env.HYPED_PRO_PYTHON || "python3"
  return [python, "-m", "hypernix.interfaces.hyped_pro_bridge", "serve"]
}

export function spawnBridge(command: string[] = pythonCommand()): Bridge {
  const child = Bun.spawn(command, {
    stdin: "pipe",
    stdout: "pipe",
    // Piped, not inherited: anything written straight to the terminal
    // would land on top of the rendered screen.
    stderr: "pipe",
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  })
  const bridge = new Bridge({
    write(line) {
      child.stdin.write(line)
      child.stdin.flush()
    },
    close() {
      try {
        child.stdin.end()
      } catch {
        // already gone
      }
    },
  })
  void pump(child.stdout, (text) => bridge.feed(text))
  void pump(child.stderr, (text) => bridge.feedLog(text))
  void child.exited.then((code) => bridge.closedBy(`the Python bridge exited (code ${code})`))
  return bridge
}
