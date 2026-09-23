// The Python side of tvtop-max.
//
// Every number and every reading of the script and log comes from
// hypernix.monitoring.tvtop_max_bridge. The protocol is hyped-pro's (this
// file started as a copy of hyped-pro's bridge.ts): one JSON object per
// line each way:
//
//   -> {"id": 7, "cmd": "frame"}
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
  // Lines the bridge sends on its own during a request: a tool the model
  // ran, or a question only the person can answer.
  onEvent: ((event: Record<string, any>) => void) | null = null

  constructor(private transport: Transport) {}

  request<T = any>(cmd: string, fields: Record<string, unknown> = {}): { id: number; done: Promise<T> } {
    const id = this.nextId++
    const done = new Promise<T>((resolve, reject) => {
      if (this.closed) {
        reject(new BridgeError("TVM-BRIDGE-001", "the Python bridge is not running"))
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
    if (message && typeof message === "object" && typeof message.event === "string") {
      this.onEvent?.(message)
      return
    }
    const waiting = this.pending.get(message?.id)
    if (!waiting) return
    this.pending.delete(message.id)
    if (message.ok) waiting.resolve(message.data)
    else waiting.reject(new BridgeError(String(message.code ?? "TVM-BRIDGE-002"), String(message.error ?? "unknown error")))
  }

  // The process went away. Everything still waiting fails now rather than
  // hanging the screen forever.
  closedBy(reason: string): void {
    if (this.closed) return
    this.closed = true
    for (const [, waiting] of this.pending) waiting.reject(new BridgeError("TVM-BRIDGE-001", reason))
    this.pending.clear()
  }

  close(): void {
    this.transport.close()
    this.closedBy("tvtop-max closed the bridge")
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

// TVTOP_MAX_PYTHON is set by the `tvtop-max` launcher to the interpreter
// that has hypernix installed, which is not necessarily `python3`.
export interface Target {
  log?: string
  script?: string
  pid?: number
}

export function pythonCommand(target: Target = {}, env: Record<string, string | undefined> = process.env): string[] {
  const python = env.TVTOP_MAX_PYTHON || "python3"
  const command = [python, "-m", "hypernix.monitoring.tvtop_max_bridge", "serve"]
  if (target.log) command.push("--log", target.log)
  if (target.script) command.push("--script", target.script)
  if (target.pid !== undefined) command.push("--pid", String(target.pid))
  return command
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
