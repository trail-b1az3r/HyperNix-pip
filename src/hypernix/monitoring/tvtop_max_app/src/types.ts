// What the Python bridge (hypernix.monitoring.tvtop_max_bridge) sends.
// Every field is optional on this side: an older bridge, a machine with
// no GPU or a run with no script leaves parts out, and a panel says so
// rather than crashing.

export interface Finding {
  level: "error" | "warn" | "info"
  source: "script" | "log"
  message: string
  line?: number
  count?: number
}

export interface ImportInfo {
  module: string
  kind: "hypernix" | "library" | "local" | "stdlib" | "missing"
  names?: string[]
  lines?: number[]
  version?: string
  summary?: string
  deprecated?: string
}

export interface ArchInfo {
  source?: string
  arch?: string
  repo?: string
  fields?: Record<string, unknown>
  preset?: Record<string, unknown>
  params?: number | null
  line?: number
  note?: string
}

export interface CookerInfo {
  name: string
  generation?: string
  deprecated?: string
  summary?: string
  kwargs?: Record<string, unknown>
  line?: number
}

export interface ScriptReport {
  path?: string
  error?: string
  imports?: ImportInfo[]
  arch?: ArchInfo
  cookers?: CookerInfo[]
  findings?: Finding[]
}

export interface ProcessRow {
  pid: number
  name?: string
  command?: string
  cpu_percent?: number
  memory_percent?: number
  rss_bytes?: number
  age_seconds?: number
  looks_like_training?: boolean
}

export interface Info {
  script?: string
  log?: string
  pid?: number | null
  process?: ProcessRow | null
  notes?: string[]
  hypernix_version?: string
  report?: ScriptReport | null
  arch?: ArchInfo
}

export interface Frame {
  step?: number
  total_steps?: number | null
  loss?: number | null
  lr?: number | null
  throughput?: number | null
  elapsed_seconds?: number
  eta_seconds?: number | null
  cpu_percent?: number | null
  ram_percent?: number | null
  gpu_util_percent?: number | null
  gpu_mem_used_mib?: number | null
  gpu_mem_total_mib?: number | null
  cpu_per_core?: number[]
  cpu_history?: number[]
  ram_history?: number[]
  gpu_util_history?: number[]
  memory?: Record<string, number>
  gpu_temp_c?: number | null
  gpu_power_w?: number | null
  gpu_power_limit_w?: number | null
  gpu_name?: string | null
  recent_losses?: number[]
  has_training_data?: boolean
  log_lines?: string[]
  log_findings?: Finding[]
  log_age_seconds?: number | null
  processes?: ProcessRow[]
}
