"""hypernix.hyper_log — Premium Training TUI Logger.

Produces consistent, highly user-friendly colored logs that remain
compatible with tvtop/tvtop++ parsing (step N/M loss=X lr=Y).
Includes advanced metrics (grad norm, epoch, GPU, storage) and an
emergency stop / pause button.
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

try:
    from rich.box import ROUNDED
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
except ImportError:
    pass

from hypernix.monitoring.tv import _fmt_duration, _query_nvidia_smi_full


@dataclass
class TrainState:
    step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    lr: float = 0.0
    grad_norm: float = 0.0
    epoch: float = 0.0
    throughput: float = 0.0
    eta_seconds: float = 0.0


class HyperLogger:
    """A live-updating rich console logger for training loops."""
    
    def __init__(
        self, 
        total_steps: int = 1000, 
        checkpoint_dir: str | Path = ".", 
        on_emergency_stop: Callable[[], None] | None = None
    ):
        self.state = TrainState(total_steps=total_steps)
        self.console = Console()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.on_emergency_stop = on_emergency_stop
        
        self.start_time = time.time()
        self._stop_event = threading.Event()
        self.paused = False
        self._listener_thread = None
        self._live = None

    def _listen_for_input(self):
        """Background thread listening for emergency stop/pause."""
        import select
        import termios
        import tty
        
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            return
            
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not self._stop_event.is_set():
                dr, dw, de = select.select([sys.stdin], [], [], 0.5)
                if dr:
                    ch = sys.stdin.read(1)
                    if ch.lower() == 'p':
                        self.paused = not self.paused
                    elif ch.lower() == 's':
                        if self.on_emergency_stop:
                            self.on_emergency_stop()
                        self.paused = True
        except Exception:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _build_layout(self) -> Layout:
        # Create a single compatibility log line at the very top for tvtop to parse if writing to file
        # But this is a UI, so the UI itself is the log. We will still print the raw line to a file.
        
        gpu = _query_nvidia_smi_full()
        usage = shutil.disk_usage(self.checkpoint_dir)
        storage_gb = usage.free / (1024**3)
        
        layout = Layout()
        layout.split_column(
            Layout(name="metrics"),
            Layout(name="hardware"),
            Layout(name="progress", size=4)
        )
        
        # Metrics Panel
        m_text = Text()
        m_text.append(f"Step      {self.state.step}/{self.state.total_steps}\n", style="cyan bold")
        m_text.append(f"Loss      {self.state.loss:.4f}\n", style="yellow")
        m_text.append(f"Grad Norm {self.state.grad_norm:.5f}\n", style="magenta")
        m_text.append(f"LearnRate {self.state.lr:.2e}\n", style="green")
        m_text.append(f"Epoch     {self.state.epoch:.2f}", style="blue")
        
        if self.paused:
            m_text.append("\n\n[PAUSED] Press 'P' to resume, 'S' to emergency checkpoint.", style="red bold blink")
        else:
            m_text.append("\n\n[ACTIVE] Press 'P' to pause, 'S' to emergency checkpoint.", style="dim")
            
        layout["metrics"].update(Panel(m_text, title="Training Metrics", box=ROUNDED, border_style="cyan"))
        
        # Hardware / System Panel
        h_text = Text()
        h_text.append(f"GPU Temp   {gpu['temp_c'] or 'N/A'} °C\n", style="red")
        if gpu['power_w']:
            h_text.append(f"GPU Power  {gpu['power_w']:.1f} W\n", style="yellow")
        h_text.append(f"Storage    {storage_gb:.1f} GB free\n", style="green")
        h_text.append("Next auto-delete in ~2 checkpoints", style="dim")
        
        layout["hardware"].update(Panel(h_text, title="Hardware & System", box=ROUNDED, border_style="green"))
        
        # Progress Panel
        p_text = Text()
        pct = (self.state.step / max(1, self.state.total_steps)) * 100
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        eta = _fmt_duration(self.state.eta_seconds)
        
        p_text.append(f"[{bar}] {pct:.1f}%\n", style="blue")
        p_text.append(f"ETA: {eta}  |  Speed: {self.state.throughput:.2f} it/s", style="white")
        
        layout["progress"].update(Panel(p_text, title="Progress", box=ROUNDED, border_style="blue"))
        
        return layout

    def start(self):
        self._listener_thread = threading.Thread(target=self._listen_for_input, daemon=True)
        self._listener_thread.start()
        
        self._live = Live(self._build_layout(), console=self.console, refresh_per_second=2)
        self._live.start()

    def update(self, step: int, loss: float, grad_norm: float, lr: float, epoch: float):
        if self.paused:
            while self.paused and not self._stop_event.is_set():
                time.sleep(0.5)
                
        now = time.time()
        elapsed = now - self.start_time
        tput = step / elapsed if elapsed > 0 else 0.0
        eta = (self.state.total_steps - step) / tput if tput > 0 else 0.0
        
        self.state.step = step
        self.state.loss = loss
        self.state.grad_norm = grad_norm
        self.state.lr = lr
        self.state.epoch = epoch
        self.state.throughput = tput
        self.state.eta_seconds = eta
        
        if self._live:
            self._live.update(self._build_layout())
            
        # Log to file in tvtop format silently
        try:
            with open("train.log", "a", encoding="utf-8") as f:
                f.write(f"step {step}/{self.state.total_steps} loss={loss:.4f} lr={lr:.2e} tput={tput:.2f}\n")
        except Exception:
            pass

    def stop(self):
        self._stop_event.set()
        if self._live:
            self._live.stop()
        if self._listener_thread:
            self._listener_thread.join(timeout=1.0)
