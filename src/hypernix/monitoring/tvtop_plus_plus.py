"""tvtop++ — The ultimate live training dashboard (btop++ style).

A premium, highly-styled console training monitor with a box layout, custom
spinners, process list, CPU/RAM/GPU block histories, and asymptotic loss curve extrapolation.

Run with:
    tvtop++
    tvtop++ --log train.log

This module uses Rich v15+ for a locked-window, flicker-free TUI experience.
v0.70.5: Fixed layout tree rebuild bug, added small_mode, re-exported _block_history_bar.
"""
from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.box import DOUBLE, ROUNDED
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from hypernix.monitoring.tv import (
    Frame,
    LogTail,
    _autodetect_log,
    _bar_str,
    _block_history_bar,
    _fmt_duration,
    _gauge_line,
    _query_nvidia_smi_full,
    _read_memory_breakdown,
    _read_proc_stat_cpu,
    _read_proc_stat_per_core,
    _safe_psutil_per_core,
    _safe_psutil_percent,
    multi_row_graph,
)

# Re-export for backwards compatibility and test imports
__all__ = [
    "TVTopPlusPlus",
    "cli_main",
    "_block_history_bar",
]

# Premium spinner characters for micro-animations
SPINNERS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def _append_ansi(content: Text, s: str, *, style: str | None = None) -> None:
    """Append a string that may contain raw ANSI SGR codes to ``content``.

    ``tv._bar_str`` / ``_gauge_line`` / ``_block_history_bar`` embed
    literal ``\\x1b[...m`` escape sequences when called with
    ``color_enabled=True`` (they were written for the plain-ANSI
    ``tvtop`` renderer, which measures visible width with its own
    ``_strip_ansi`` helper). Rich's ``Text`` has no idea those bytes
    are escape codes -- appending them via a plain f-string leaves the
    raw control characters sitting in ``Text.plain``, so Rich's cell
    width for that line comes out wrong. Since panel borders are
    positioned from that same width calculation, the result is a
    right-hand border drawn in the wrong place (stray "│" characters
    floating outside the box, sometimes duplicated across several rows).

    ``Text.from_ansi`` parses the escape codes into proper zero-width
    style spans instead of literal characters, so the width Rich sees
    always matches what actually reaches the screen.

    ``Text.from_ansi`` treats its input as a sequence of lines and does
    *not* preserve a trailing ``"\\n"`` the way plain string
    concatenation does -- it silently drops it. Since every call site
    here builds a multi-row panel by appending one ``"...\\n"``-terminated
    string per row, that would glue consecutive rows onto a single
    line. Trailing newlines never contain escape codes, so they're
    split off and re-appended as plain text after conversion.
    """
    trimmed = s.rstrip("\n")
    trailing = s[len(trimmed):]
    content.append(Text.from_ansi(trimmed, style=style or ""))
    if trailing:
        content.append(trailing, style=style)


@dataclass
class TVTopPlusPlus:
    log_path: Path | str | None = None
    refresh_seconds: float = 1.0
    color: bool = True
    ascii_only: bool = False
    width: int | None = None
    small_mode: bool = False
    started_at: float = field(default_factory=time.time)
    log_tail: LogTail | None = field(default=None, init=False)
    tick: int = field(default=0, init=False)

    # Histories
    _cpu_history: deque[float] = field(default_factory=lambda: deque(maxlen=120), init=False, repr=False)
    _ram_history: deque[float] = field(default_factory=lambda: deque(maxlen=120), init=False, repr=False)
    _gpu_util_history: deque[float] = field(default_factory=lambda: deque(maxlen=120), init=False, repr=False)

    # Layout cache — built once, reused every tick (fixes border flicker)
    _layout: Layout | None = field(default=None, init=False, repr=False)
    _console: Console | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.log_path is not None:
            self.log_tail = LogTail(Path(self.log_path), history_size=8)

    def latest_frame(self) -> Frame:
        f = Frame()
        if self.log_tail is not None:
            self.log_tail.poll()
            if self.log_tail.has_training_data:
                f.has_training_data = True
                f.step = self.log_tail.step or 0
                f.total_steps = self.log_tail.total_steps
                f.loss = self.log_tail.loss
                f.lr = self.log_tail.lr
                f.throughput = self.log_tail.throughput
            elif self.log_tail.latest_match is not None:
                m = self.log_tail.latest_match
                f.has_training_data = True
                f.step = int(m.group("step"))
                if m.group("total"):
                    f.total_steps = int(m.group("total"))
                if m.group("loss"):
                    f.loss = float(m.group("loss"))
                if m.group("lr"):
                    f.lr = float(m.group("lr"))
                if m.group("tput"):
                    f.throughput = float(m.group("tput"))
            f.recent_losses = list(self.log_tail.losses)
            f.log_tail = list(self.log_tail.tail)
        f.elapsed_seconds = time.time() - self.started_at
        if f.has_training_data and f.throughput is None and f.step > 0 and f.elapsed_seconds > 0:
            f.throughput = f.step / f.elapsed_seconds
        if f.has_training_data and f.total_steps and f.throughput and f.throughput > 0:
            f.eta_seconds = max(0.0, (f.total_steps - f.step) / f.throughput)

        # CPU
        cpu, ram = _safe_psutil_percent()
        if cpu is None:
            cpu = _read_proc_stat_cpu()
        per_core = _safe_psutil_per_core() or _read_proc_stat_per_core() or []
        f.cpu_per_core = list(per_core)

        # Memory
        mem = _read_memory_breakdown() or {}
        f.memory = mem
        if ram is None and mem.get("percent") is not None:
            ram = float(mem["percent"])
        f.cpu_percent = cpu
        f.ram_percent = ram

        # GPU
        gpu = _query_nvidia_smi_full()
        f.gpu_mem_used_mib = gpu["mem_used_mib"]
        f.gpu_mem_total_mib = gpu["mem_total_mib"]
        f.gpu_util_percent = gpu["util_percent"]
        f.gpu_temp_c = gpu["temp_c"]
        f.gpu_power_w = gpu["power_w"]
        f.gpu_power_limit_w = gpu["power_limit_w"]
        f.gpu_name = gpu["name"]

        # History buffers
        if cpu is not None:
            self._cpu_history.append(cpu)
        if ram is not None:
            self._ram_history.append(ram)
        if gpu["util_percent"] is not None:
            self._gpu_util_history.append(gpu["util_percent"])

        f.cpu_history = list(self._cpu_history)
        f.ram_history = list(self._ram_history)
        f.gpu_util_history = list(self._gpu_util_history)
        return f

    def _build_layout(self, f: Frame, console: Console) -> Layout:
        """Backward compatibility for tests."""
        layout = self._init_layout()
        self._update_layout(f, console, layout)
        return layout

    def _init_layout(self) -> Layout:
        """Create the static layout tree once. The tree structure never
        changes after initialisation — only panel *contents* are swapped
        on each refresh. This eliminates the border-flicker bug caused
        by repeatedly calling ``split_row`` / ``split_column``."""
        layout = Layout()

        if self.small_mode:
            # Compact layout for small terminals
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="body"),
                Layout(name="footer", size=2),
            )
            layout["body"].split_column(
                Layout(name="training", ratio=2),
                Layout(name="hardware", ratio=1),
                Layout(name="loss", ratio=1),
            )
        else:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="body"),
                Layout(name="footer", size=3),
            )
            layout["body"].split_column(
                Layout(name="top", ratio=2),
                Layout(name="bottom", ratio=1),
            )
            layout["top"].split_row(
                Layout(name="left"),
                Layout(name="right"),
            )
            layout["left"].split_column(
                Layout(name="training", ratio=2),
                Layout(name="process", ratio=1),
            )
            layout["right"].split_column(
                Layout(name="hardware", ratio=2),
                Layout(name="gpu", ratio=1),
            )
            layout["bottom"].split_row(
                Layout(name="loss", ratio=2),
                Layout(name="log", ratio=1),
            )
        return layout

    def _update_layout(self, f: Frame, console: Console, layout: Layout) -> None:
        """Update the Rich layout for the dashboard.

        Only swaps panel *contents* — the tree structure is immutable
        after ``_init_layout`` so borders never shift or flicker."""
        self.tick += 1

        # Header with spinner
        spinner_char = SPINNERS[self.tick % len(SPINNERS)] if not self.ascii_only else "*"
        title_text = Text.assemble(
            (" ", "default"),
            (spinner_char, "bright_cyan"),
            ("  ✦ HYPERNIX TVTOP++ ✦  ", "bold bright_blue"),
            (time.strftime(" %Y-%m-%d %H:%M:%S "), "dim"),
        )

        # Build all panels
        training_panel = self._make_training_panel(f, console)
        hardware_panel = self._make_hardware_panel(f, console)
        process_panel = self._make_process_panel(f)
        gpu_panel = self._make_gpu_panel(f)
        loss_panel = self._make_loss_panel(f, console)
        log_panel = self._make_log_panel(f, console)

        # Assign panels — tree structure is already fixed
        layout["header"].update(Panel(title_text, style="bright_blue", padding=(0, 2)))
        layout["training"].update(training_panel)
        layout["hardware"].update(hardware_panel)
        layout["loss"].update(loss_panel)

        if not self.small_mode:
            layout["process"].update(process_panel)
            layout["gpu"].update(gpu_panel)
            layout["log"].update(log_panel)

        footer_text = Text.assemble(
            (" ❖ Ctrl-C to exit · ", "dim"),
            (f"refresh {self.refresh_seconds:.1f}s", "dim green"),
            (" · ", "dim"),
            (f"log={self.log_path}", "dim green"),
        )
        layout["footer"].update(Panel(footer_text, style="white", padding=(0, 2)))

    def _make_training_panel(self, f: Frame, console: Console | None = None) -> Panel:
        """Create the Training Vitals panel."""
        content = Text()

        if f.has_training_data:
            # Progress bar
            pct = f.progress * 100.0
            prog_text = f"Step {f.step}"
            if f.total_steps:
                prog_text += f"/{f.total_steps}"
            pbar = _bar_str(f.progress, 20, ascii_only=self.ascii_only, color_enabled=self.color)
            _append_ansi(content, f"{prog_text:<16} {pbar} {pct:>5.1f}%\n\n", style="cyan")

            # Metrics
            loss_val = f"{f.loss:.4f}" if f.loss is not None else "—"
            lr_val = f"{f.lr:.2e}" if f.lr is not None else "—"
            tput_val = f"{f.throughput:.2f}/s" if f.throughput is not None else "—"
            content.append(f"Loss       {loss_val}\n", style="yellow")
            content.append(f"LearnRate  {lr_val}\n", style="green")
            content.append(f"Speed      {tput_val}\n\n", style="magenta")

            # Time
            elapsed = _fmt_duration(f.elapsed_seconds)
            eta = _fmt_duration(f.eta_seconds) if f.eta_seconds is not None else "—"
            content.append(f"Time       {elapsed:<9}  ETA: {eta}", style="white")
        else:
            content.append(" ⏳ Waiting for training log data...\n\n", style="yellow")
            content.append(f" File: {self.log_path or 'not specified'}\n", style="dim")
            content.append(" Ensure your training scripts output logs containing\n", style="dim")
            content.append(" steps, loss=, and throughput values.", style="dim")

        return Panel(content, title="Training Vitals", box=DOUBLE, title_align="left", style="cyan")

    def _make_hardware_panel(self, f: Frame, console: Console | None = None) -> Panel:
        """Create the Hardware Vitals panel (colors match original tvtop)."""
        content = Text()
        hist_width = max(20, min(40, (console.width // 4) - 14 if console else 30))

        # Gauges — CPU=green, RAM=magenta, GPU=red (matching original tvtop)
        _append_ansi(content, _gauge_line("CPU", f.cpu_percent, 20, ascii_only=self.ascii_only, color_enabled=self.color) + "\n", style="green")
        _append_ansi(content, _gauge_line("RAM", f.ram_percent, 20, ascii_only=self.ascii_only, color_enabled=self.color) + "\n", style="magenta")
        _append_ansi(content, _gauge_line("GPU", f.gpu_util_percent, 20, ascii_only=self.ascii_only, color_enabled=self.color) + "\n\n", style="red")

        # Block history
        if f.cpu_history:
            cpu_hist = _block_history_bar(f.cpu_history, hist_width, self.color)
            _append_ansi(content, f"CPU History [{cpu_hist}]\n", style="green")
        if f.ram_history:
            ram_hist = _block_history_bar(f.ram_history, hist_width, self.color)
            _append_ansi(content, f"RAM History [{ram_hist}]\n", style="magenta")
        if f.gpu_util_history:
            gpu_hist = _block_history_bar(f.gpu_util_history, hist_width, self.color)
            _append_ansi(content, f"GPU History [{gpu_hist}]", style="red")

        return Panel(content, title="Hardware Vitals", box=ROUNDED, title_align="left", style="green")

    def _make_process_panel(self, f: Frame) -> Panel:
        """Create the Process Monitor panel."""
        table = Table(show_header=True, header_style="bold blue", show_lines=False, padding=(0, 1))
        table.add_column("PID", style="cyan", width=6)
        table.add_column("USER", style="green", width=8)
        table.add_column("CPU%", style="yellow", width=6)
        table.add_column("MEM%", style="magenta", width=6)
        table.add_column("COMMAND", style="white", overflow="ellipsis")

        processes = self._get_active_processes()
        if processes:
            for p in processes:
                table.add_row(
                    str(p['pid']),
                    p['user'][:8],
                    f"{p['cpu']:>5.1f}",
                    f"{p['mem']:>5.1f}",
                    p['cmd'][:40],
                )
        else:
            table.add_row("", "(no active python/training processes)", "", "", "")

        return Panel(table, title="Process Monitor", box=ROUNDED, title_align="left", style="blue")

    def _make_gpu_panel(self, f: Frame) -> Panel:
        """Create the GPU Details panel."""
        content = Text()

        if f.gpu_util_percent is None and f.gpu_mem_total_mib is None:
            content.append("(no GPU detected or nvidia-smi missing)", style="dim")
        else:
            if f.gpu_name:
                content.append(f"{f.gpu_name}\n\n", style="bright_cyan bold")

            # VRAM
            if f.gpu_mem_used_mib is not None and f.gpu_mem_total_mib:
                mem_pct = 100.0 * f.gpu_mem_used_mib / f.gpu_mem_total_mib
                bar = _bar_str(mem_pct / 100.0, 15, ascii_only=self.ascii_only, color_enabled=self.color)
                _append_ansi(content, f"VRAM  {bar} {f.gpu_mem_used_mib:>5}/{f.gpu_mem_total_mib:<5} MiB\n", style="yellow")

            # Temp
            if f.gpu_temp_c is not None:
                temp_norm = max(0.0, min(1.0, (f.gpu_temp_c - 30) / 70.0))
                bar = _bar_str(temp_norm, 15, ascii_only=self.ascii_only, color_enabled=self.color)
                _append_ansi(content, f"Temp  {bar} {f.gpu_temp_c:>5.1f}°C\n", style="red")

            # Power
            if f.gpu_power_w is not None and f.gpu_power_limit_w:
                pwr_pct = 100.0 * f.gpu_power_w / f.gpu_power_limit_w
                bar = _bar_str(pwr_pct / 100.0, 15, ascii_only=self.ascii_only, color_enabled=self.color)
                _append_ansi(content, f"Power {bar} {f.gpu_power_w:>5.1f}/{f.gpu_power_limit_w:<5.0f} W", style="magenta")

        return Panel(content, title="GPU Details", box=ROUNDED, title_align="left", style="magenta")

    def _make_loss_panel(self, f: Frame, console: Console | None = None) -> Panel:
        """Create the Loss Curve panel."""
        content = Text()
        graph_width = max(20, min(80, (console.width // 2) - 10 if console else 40))

        if f.recent_losses:
            # Exponential decay dampening projection
            if len(f.recent_losses) >= 5:
                recent_ma = sum(f.recent_losses[-3:]) / 3.0
                prev_ma = sum(f.recent_losses[-6:-3]) / 3.0 if len(f.recent_losses) >= 6 else f.recent_losses[0]
                slope = (recent_ma - prev_ma) / 3.0

                future_losses = []
                cur = f.recent_losses[-1]
                temp_slope = slope
                for _ in range(15):
                    cur += temp_slope
                    cur = max(0.0, cur)
                    future_losses.append(cur)
                    temp_slope *= 0.85

                combined = f.recent_losses + future_losses

            else:
                combined = f.recent_losses

            # Render graph as text using block characters — width scales with terminal
            graph_rows = multi_row_graph(combined, width=graph_width, height=5, ascii_only=self.ascii_only)
            for i, row in enumerate(graph_rows):
                if self.color:
                    if len(combined) > 15:
                        pred_chars = int((15 / len(combined)) * graph_width)
                        real_chars = max(0, graph_width - pred_chars)
                        real_str = row[:real_chars]
                        pred_str = row[real_chars:]
                        content.append(real_str + "\n" if i < len(graph_rows) - 1 else real_str, style="cyan")
                        if pred_str:
                            content.append(pred_str + "\n" if i < len(graph_rows) - 1 else pred_str, style="magenta")
                    else:
                        content.append(row + ("\n" if i < len(graph_rows) - 1 else ""), style="cyan")
                else:
                    content.append(row + ("\n" if i < len(graph_rows) - 1 else ""))
        else:
            content.append("(no loss values yet — tailing train.log...)", style="dim")

        return Panel(content, title="Loss Curve", box=ROUNDED, title_align="left", style="yellow")

    def _make_log_panel(self, f: Frame, console: Console | None = None) -> Panel:
        """Create the Recent Log Tail panel (line width scales with terminal)."""
        content = Text()
        # The log panel gets roughly 1/3 of the terminal width — use that budget
        line_width = max(40, (console.width // 3) - 6 if console else 60)
        log_lines = (f.log_tail or [])[-8:]
        for line in log_lines:
            content.append(line[:line_width] + "\n", style="white")

        return Panel(content, title="Recent Log Tail", box=ROUNDED, title_align="left", style="white")

    #: Seconds a process table is reused. Walking every process with its
    #: owner and command line costs about a second on Windows, and the
    #: layout is rebuilt every refresh — so without this the dashboard
    #: spent most of its time listing processes that had not changed.
    PROCESS_REFRESH_SECONDS = 5.0

    def _get_active_processes(self) -> list[dict[str, Any]]:
        """Active hypernix or python training processes, refreshed every few seconds."""
        now = time.monotonic()
        cached = getattr(self, "_process_cache", None)
        if cached is not None and now - cached[0] < self.PROCESS_REFRESH_SECONDS:
            return cached[1]
        processes = self._scan_processes()
        self._process_cache = (now, processes)
        return processes

    def _scan_processes(self) -> list[dict[str, Any]]:
        processes = []
        try:
            import psutil
            for proc in psutil.process_iter(['pid', 'username', 'cpu_percent', 'memory_percent', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or []) if proc.info['cmdline'] else ''
                    cmd_lower = cmdline.lower()
                    # Look for training-related processes
                    if 'hypernix' in cmd_lower or 'python' in cmd_lower or 'train' in cmd_lower:
                        # Skip tvtop/this process and system monitoring tools
                        skip_keywords = ('tvtop', 'psutil', 'htop', 'btop', 'nvidia-smi', 'watch')
                        if any(kw in cmd_lower for kw in skip_keywords):
                            continue
                        processes.append({
                            'pid': proc.info['pid'],
                            'user': proc.info['username'] or 'unknown',
                            'cpu': round(proc.info['cpu_percent'] or 0.0, 1),
                            'mem': round(proc.info['memory_percent'] or 0.0, 1),
                            'cmd': cmdline[:50]
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass
        return sorted(processes, key=lambda p: p['cpu'], reverse=True)[:5]

    def run(self) -> None:
        """Run the tvtop++ dashboard using Rich Live."""
        # Create the console once with no fixed width
        self._console = Console(force_terminal=True)
        console = self._console

        # Build layout tree ONCE — never rebuild it (fixes border flicker)
        self._layout = self._init_layout()
        layout = self._layout

        try:
            self._update_layout(self.latest_frame(), console, layout)
            with Live(
                layout,
                console=console,
                refresh_per_second=1 / self.refresh_seconds,
                screen=True,
            ):
                while True:
                    frame = self.latest_frame()
                    self._update_layout(frame, console, layout)
                    time.sleep(self.refresh_seconds)
        except KeyboardInterrupt:
            pass


def cli_main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    log: Path | None = None
    color = True
    ascii_only = False
    refresh = 1.0
    small_mode = False

    if "--no-color" in args:
        color = False
        args.remove("--no-color")
    if "--ascii" in args:
        ascii_only = True
        args.remove("--ascii")
    if "-s" in args or "--small" in args:
        small_mode = True
        args = [a for a in args if a not in ("-s", "--small")]
    if "--refresh" in args:
        i = args.index("--refresh")
        if i + 1 < len(args):
            refresh = float(args[i + 1])
            del args[i : i + 2]
    if "--log" in args:
        i = args.index("--log")
        if i + 1 < len(args):
            log = Path(args[i + 1])
            del args[i : i + 2]

    if "--help" in args or "-h" in args:
        print(
            "usage: tvtop++ [--log path] [--no-color] [--ascii] "
            "[--refresh SECONDS] [-s|--small]\n"
            "An advanced btop++ styled training dashboard with moving spinner,\n"
            "process list, block history and improved decay curve estimations.\n"
            "  -s, --small   compact mode for smaller terminals\n"
            "\n"
            "If no --log is specified, tvtop++ will search for *.log files in the\n"
            "current directory. If none are found, it will display a blank dashboard\n"
            "waiting for training data.",
        )
        return 0

    if log is None:
        try:
            from hypernix.timing.spinner import Spinner
            with Spinner("Detecting training log...", style="dots"):
                log = _autodetect_log()
        except Exception:
            log = _autodetect_log()

    if log is not None:
        print(f"[tvtop++] Tailing {log}...", file=sys.stderr)
    else:
        print("[tvtop++] No log file specified - displaying live system metrics only", file=sys.stderr)

    TVTopPlusPlus(
        log_path=log,
        color=color,
        ascii_only=ascii_only,
        refresh_seconds=refresh,
        small_mode=small_mode,
    ).run()
    return 0
