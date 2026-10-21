// What the Python bridge (hypernix.monitoring.tvtop_max_bridge) sends.
// Every field is optional on this side: an older bridge, a machine with
// no GPU or a run with no script leaves parts out, and a panel says so
// rather than crashing.

export interface Finding {
  level: "error" | "warn" | "info"
  source: "script" | "log" | "audio"
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

// 0.72.8: the machine, from hypernix.monitoring.hostprobe.

export interface CpuTemps {
  // One reading per logical CPU, in °C; null where the driver has none.
  per_core?: (number | null)[]
  package_c?: number | null
  dies?: Record<string, number>
  source?: string
  note?: string
}

export interface AudioState {
  available?: boolean
  backend?: string
  device?: string
  port?: string
  headphones?: boolean
  // What the sound server applies: 0 dB is 100%, above it is amplified.
  volume_db?: number | null
  volume_db_silent?: boolean
  volume_percent?: number | null
  muted?: boolean
  playing?: boolean
  // The level of what is playing, read from the output's monitor.
  signal_peak_dbfs?: number | null
  signal_rms_dbfs?: number | null
  level?: "ok" | "loud" | "unsafe"
  warning?: string
  note?: string
}

export interface DiskRow {
  mount: string
  device?: string
  fstype?: string
  total_bytes?: number
  used_bytes?: number
  percent?: number
  read_bps?: number | null
  write_bps?: number | null
}

export interface NetRow {
  name: string
  recv_bytes?: number
  sent_bytes?: number
  down_bps?: number | null
  up_bps?: number | null
  link_mbps?: number | null
}

export interface SystemInfo {
  hostname?: string
  os?: string
  load?: number[] | null
  uptime_seconds?: number | null
  battery?: { percent?: number; plugged?: boolean | null; seconds_left?: number | null } | null
  processes?: number
  cores?: number | null
}

export interface MemUser {
  pid: number
  name?: string
  command?: string
  rss_bytes?: number
  // Bytes per second since the previous sample; negative when shrinking.
  growth_bps?: number | null
}

export interface Info {
  script?: string
  log?: string
  pid?: number | null
  process?: ProcessRow | null
  notes?: string[]
  discovering?: boolean
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
  // True while the bridge is still looking for the run's script and log.
  discovering?: boolean
  cpu_temps?: CpuTemps | null
  audio?: AudioState | null
  disks?: DiskRow[] | null
  network?: NetRow[] | null
  system?: SystemInfo | null
  memory_users?: MemUser[] | null
}
