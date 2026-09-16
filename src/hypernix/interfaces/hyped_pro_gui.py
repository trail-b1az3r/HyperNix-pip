"""hyped_pro_gui — hyped-pro's desktop GUI.

A real, working GUI for hyped-pro, backed by :mod:`hypernix.hyped_pro_core`
— the same provider/model/dispatch layer the Node TUI's bridge
(:mod:`hypernix.hyped_pro_bridge`) uses, so a model switch, a key, or a
chat reply matches exactly between the TUI and the GUI.

Launched as its own skill/command from the TUI (``/gui``) or directly:

    python3 -m hypernix.hyped_pro_gui

Backend selection, in order:

  1. **Qt6 via PySide6** — covers both X11 and Wayland from one codebase.
     ``QT_QPA_PLATFORM`` is chosen from the real session ($XDG_SESSION_TYPE,
     $WAYLAND_DISPLAY, $DISPLAY) before Qt is imported (Qt reads it at
     import/init time), with an automatic retry against ``xcb`` if the
     Wayland platform plugin fails to load.
  2. **GTK4 via PyGObject (gi)** — used only if PySide6 isn't installed at
     all, so there's always a working option on a GNOME/GTK-only system.

Every branch — platform detection, backend selection, import failures,
worker-thread errors — prints a coded line to stderr, on every path, per
spec. Set HYPED_GUI_DEBUG=1 for verbose per-step logging.

Error/debug codes
------------------
  HPG-PLATFORM-001  could not detect a usable display session at all
  HPG-QT-001        PySide6 is not installed
  HPG-QT-002        Qt6 failed to initialize under the chosen platform plugin
  HPG-GTK-001       PyGObject (gi) / GTK4 is not installed
  HPG-GTK-002       GTK4 failed to initialize
  HPG-BACKEND-001   neither Qt6 nor GTK4 is available — nothing to launch
  HPG-CHAT-001      a chat request failed (wraps the underlying HPC-* code)
  HPG-DOWNLOAD-001  a model download failed (wraps the underlying HPC-* code)
  HPG-KEY-001       saving/reading a provider key failed
"""
from __future__ import annotations

import logging
import os
import sys

log = logging.getLogger("hypernix.hyped_pro_gui")

DEBUG = bool(os.environ.get("HYPED_GUI_DEBUG"))


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.DEBUG if DEBUG else logging.INFO,
        format="[hyped-pro-gui] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )


def _err(code: str, msg: str) -> None:
    log.error("%s: %s", code, msg)


def _dbg(msg: str) -> None:
    log.debug(msg)


# ---------------------------------------------------------------------------
# Session/platform detection — pure function, no Qt/GTK imports, so it's
# testable and reusable regardless of which backend ends up running.
# ---------------------------------------------------------------------------


def detect_session() -> dict:
    session_type = os.environ.get("XDG_SESSION_TYPE", "")
    wayland_display = os.environ.get("WAYLAND_DISPLAY")
    x11_display = os.environ.get("DISPLAY")
    is_wayland = bool(wayland_display) or session_type.lower() == "wayland"
    is_x11 = bool(x11_display) or session_type.lower() == "x11"
    return {
        "session_type": session_type or "unknown",
        "wayland_display": wayland_display,
        "x11_display": x11_display,
        "is_wayland": is_wayland,
        "is_x11": is_x11,
    }


def choose_qt_platform(session: dict) -> str | None:
    """Return a QT_QPA_PLATFORM value to try first, or None to let Qt decide.

    Preferring Wayland when the session is genuinely Wayland avoids Qt's
    XWayland compatibility path (blurrier scaling, extra latency); falling
    through to xcb covers plain X11 and any Wayland compositor without a
    working wayland Qt plugin.
    """
    if session["is_wayland"]:
        return "wayland"
    if session["is_x11"]:
        return "xcb"
    return None  # unknown session — let Qt probe for itself


# ---------------------------------------------------------------------------
# Qt6 (PySide6) backend
# ---------------------------------------------------------------------------


def try_launch_qt() -> bool:
    """Attempt to launch the Qt6 GUI. Returns False if Qt6 isn't usable at
    all (missing package or platform init failure) so main() can fall back
    to GTK4; True once the Qt event loop has run and returned (i.e. the
    user closed the window) — a normal exit, not a failure.

    Caveat: if QApplication() can't initialize *any* platform plugin at
    all (e.g. genuinely no display/compositor present), Qt's C++ layer
    calls qFatal() -> abort() and terminates the process before this
    try/except ever gets a chance to run — no Python exception is
    raised. That failure mode can't be caught here; it's why cli_main()
    checks detect_session() and refuses to call this function at all when
    no display is present, rather than relying on this catching it.
    """
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        _err("HPG-QT-001", f"PySide6 not installed ({exc}). Install with: pip install hypernix[gui-qt]")
        return False

    session = detect_session()
    preferred = choose_qt_platform(session)
    if preferred and "QT_QPA_PLATFORM" not in os.environ:
        os.environ["QT_QPA_PLATFORM"] = preferred
        _dbg(f"session={session} -> QT_QPA_PLATFORM={preferred}")
    else:
        _dbg(f"session={session}, QT_QPA_PLATFORM left as-is "
             f"({os.environ.get('QT_QPA_PLATFORM', 'unset, Qt will probe')})")

    def _build_and_run() -> int:
        app = QApplication.instance() or QApplication(sys.argv)
        window = _qt_build_window()
        window.show()
        log.info(f"hyped-pro GUI running (Qt6, platform={os.environ.get('QT_QPA_PLATFORM', 'auto')})")
        return app.exec()

    try:
        _build_and_run()
        return True
    except Exception as exc:  # noqa: BLE001
        if preferred == "wayland":
            _err("HPG-QT-002", f"Qt6 failed under platform='wayland' ({exc}); retrying with xcb (X11)")
            os.environ["QT_QPA_PLATFORM"] = "xcb"
            try:
                _build_and_run()
                return True
            except Exception as exc2:  # noqa: BLE001
                _err("HPG-QT-002", f"Qt6 also failed under platform='xcb': {exc2}")
                return False
        _err("HPG-QT-002", f"Qt6 failed to initialize (platform={os.environ.get('QT_QPA_PLATFORM')}): {exc}")
        return False


# PySide6 types (QMainWindow etc.) don't exist until try_launch_qt()'s import
# above succeeds, so the window class that subclasses them is built lazily
# by _make_main_window_class() the first time it's actually needed — that's
# what lets `import hypernix.hyped_pro_gui` succeed even without PySide6
# installed at all (HPG-QT-001 is a runtime message, not an import crash).


def _make_main_window_class():
    from PySide6.QtCore import QObject, QThread, Signal
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    from hypernix.interfaces import hyped_pro_core as core

    class ChatSignals(QObject):
        finished = Signal(str)
        failed = Signal(str, str)  # code, message

    class ChatThread(QThread):
        def __init__(self, model_short: str, messages: list, system: str | None):
            super().__init__()
            self.model_short = model_short
            self.messages = messages
            self.system = system
            self.signals = ChatSignals()

        def run(self) -> None:
            try:
                result = core.send_chat_message(self.model_short, self.messages, system=self.system)
                self.signals.finished.emit(result["content"] or "")
            except core.HypedProError as exc:
                self.signals.failed.emit(exc.code, exc.message)
            except Exception as exc:  # noqa: BLE001
                self.signals.failed.emit("HPG-CHAT-001", str(exc))

    class DownloadSignals(QObject):
        finished = Signal(str)
        failed = Signal(str, str)

    class DownloadThread(QThread):
        def __init__(self, model_short: str):
            super().__init__()
            self.model_short = model_short
            self.signals = DownloadSignals()

        def run(self) -> None:
            try:
                model = core.get_model(self.model_short)
                path = core.ensure_downloaded(model, quiet=False)
                self.signals.finished.emit(str(path))
            except core.HypedProError as exc:
                self.signals.failed.emit(exc.code, exc.message)
            except Exception as exc:  # noqa: BLE001
                self.signals.failed.emit("HPG-DOWNLOAD-001", str(exc))

    class HypedProMainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("hyped-pro")
            self.resize(880, 620)
            self.history: list[dict] = []
            self._chat_thread: QThread | None = None
            self._dl_thread: QThread | None = None

            self._noodle_session: str | None = None
            self._noodle_timer = None

            tabs = QTabWidget()
            self.setCentralWidget(tabs)
            tabs.addTab(self._build_chat_tab(), "Chat")
            tabs.addTab(self._build_noodle_tab(), "Noodle")
            tabs.addTab(self._build_keys_tab(), "API Keys")

        # -- Chat tab ---------------------------------------------------
        def _build_chat_tab(self) -> QWidget:
            w = QWidget()
            layout = QVBoxLayout(w)

            top = QHBoxLayout()
            self.model_box = QComboBox()
            for m in core.MODELS:
                self.model_box.addItem(f"{m.badge} {m.short}  ({m.kind}/{m.vendor})", m.short)
            top.addWidget(QLabel("Model:"))
            top.addWidget(self.model_box, stretch=1)
            self.download_btn = QPushButton("Download / ensure ready")
            self.download_btn.clicked.connect(self._on_download_clicked)
            top.addWidget(self.download_btn)
            layout.addLayout(top)

            self.transcript = QTextEdit()
            self.transcript.setReadOnly(True)
            layout.addWidget(self.transcript, stretch=1)

            bottom = QHBoxLayout()
            self.input_line = QLineEdit()
            self.input_line.returnPressed.connect(self._on_send_clicked)
            self.send_btn = QPushButton("Send")
            self.send_btn.clicked.connect(self._on_send_clicked)
            bottom.addWidget(self.input_line, stretch=1)
            bottom.addWidget(self.send_btn)
            layout.addLayout(bottom)

            return w

        # -- Noodle tab -------------------------------------------------
        #
        # The same session API the TUI drives over the bridge, called
        # directly because the GUI is already a Python process. A QTimer
        # polls for events rather than a thread: poll() is a lock and a
        # deque drain, far too cheap to be worth a thread, and doing it
        # on the UI thread keeps every widget update where Qt wants it.
        def _build_noodle_tab(self) -> QWidget:
            w = QWidget()
            layout = QVBoxLayout(w)

            layout.addWidget(QLabel(
                "Noodle runs agents against a task. They get file tools rooted "
                "at ./.noodle and nothing outside it."
            ))

            top = QHBoxLayout()
            self.noodle_task = QLineEdit()
            self.noodle_task.setPlaceholderText("What should the swarm do?")
            self.noodle_task.returnPressed.connect(self._on_noodle_start)
            self.noodle_start_btn = QPushButton("Run")
            self.noodle_start_btn.clicked.connect(self._on_noodle_start)
            self.noodle_stop_btn = QPushButton("Stop")
            self.noodle_stop_btn.setEnabled(False)
            self.noodle_stop_btn.clicked.connect(self._on_noodle_stop)
            top.addWidget(self.noodle_task, stretch=1)
            top.addWidget(self.noodle_start_btn)
            top.addWidget(self.noodle_stop_btn)
            layout.addLayout(top)

            self.noodle_exec = QCheckBox(
                "Allow shell execution (agents may run commands)"
            )
            layout.addWidget(self.noodle_exec)

            self.noodle_log = QTextEdit()
            self.noodle_log.setReadOnly(True)
            layout.addWidget(self.noodle_log, stretch=1)

            self.noodle_status = QLabel("")
            layout.addWidget(self.noodle_status)

            self._refresh_noodle_providers()
            return w

        def _refresh_noodle_providers(self) -> None:
            from hypernix.interfaces.noodle import hyped as noodle

            try:
                info = noodle.providers()
            except Exception as exc:  # noqa: BLE001 - a tab must still build
                self.noodle_status.setText(f"Could not list providers: {exc}")
                return
            if info["any_ready"]:
                self.noodle_status.setText(
                    "Ready: " + ", ".join(info["ready"])
                )
            else:
                self.noodle_status.setText(
                    "No provider is usable yet — set a key in the API Keys tab, "
                    "or start Ollama."
                )

        def _noodle_append(self, text: str) -> None:
            self.noodle_log.append(text)

        def _on_noodle_start(self) -> None:
            from PySide6.QtCore import QTimer

            from hypernix.interfaces.noodle import hyped as noodle

            task = self.noodle_task.text().strip()
            if not task:
                return
            self.noodle_log.clear()
            try:
                session = noodle.start(
                    task, allow_execute=self.noodle_exec.isChecked()
                )
            except noodle.NoodleSessionError as exc:
                self._noodle_append(
                    f'<span style="color:#e05555">{exc}</span>'
                )
                return
            self._noodle_session = session["session_id"]
            self.noodle_start_btn.setEnabled(False)
            self.noodle_stop_btn.setEnabled(True)
            self.noodle_task.setEnabled(False)
            self._noodle_append(
                f'<i>session {session["session_id"]} · '
                f'{", ".join(session["roster"])} · {session["root"]}</i>'
            )
            self._noodle_timer = QTimer(self)
            self._noodle_timer.timeout.connect(self._on_noodle_tick)
            self._noodle_timer.start(400)

        def _on_noodle_tick(self) -> None:
            from hypernix.interfaces.noodle import hyped as noodle

            if not self._noodle_session:
                return
            try:
                tick = noodle.poll(self._noodle_session)
            except noodle.NoodleSessionError as exc:
                self._noodle_append(f'<span style="color:#e05555">{exc}</span>')
                self._noodle_finish()
                return
            for event in tick["events"]:
                kind = str(event.get("kind") or event.get("type") or "event")
                detail = str(
                    event.get("message") or event.get("text")
                    or event.get("tool") or event.get("task_id") or ""
                )
                self._noodle_append(f"<b>{kind}</b> {detail}")
            if not tick["running"]:
                if tick["error"]:
                    self._noodle_append(
                        f'<span style="color:#e05555">{tick["error"]}</span>'
                    )
                elif tick["report"]:
                    ok = tick["report"].get("ok")
                    self._noodle_append(
                        f'<b>{"Done" if ok else "Finished with failures"}</b> '
                        f'in {tick["elapsed"]}s'
                    )
                self._noodle_finish()

        def _on_noodle_stop(self) -> None:
            from hypernix.interfaces.noodle import hyped as noodle

            if not self._noodle_session:
                return
            try:
                result = noodle.stop(self._noodle_session)
            except noodle.NoodleSessionError as exc:
                self._noodle_append(f'<span style="color:#e05555">{exc}</span>')
                return
            self._noodle_append("<i>Stopping…</i>")
            if not result["swarm_notified"]:
                self._noodle_append(
                    "<i>Turns already in flight will finish — they are paid "
                    "for either way.</i>"
                )

        def _noodle_finish(self) -> None:
            if self._noodle_timer is not None:
                self._noodle_timer.stop()
                self._noodle_timer = None
            self._noodle_session = None
            self.noodle_start_btn.setEnabled(True)
            self.noodle_stop_btn.setEnabled(False)
            self.noodle_task.setEnabled(True)

        def _current_model_short(self) -> str:
            return self.model_box.currentData()

        def _append(self, text: str) -> None:
            self.transcript.append(text)

        def _on_download_clicked(self) -> None:
            model_short = self._current_model_short()
            model = core.get_model(model_short)
            if model.kind != "local":
                self._append(f"<i>{model.short} is a cloud model ({model.vendor}) — set an API key instead of downloading.</i>")
                return
            self.download_btn.setEnabled(False)
            self._append(f"<i>Downloading {model.repo} ...</i>")
            self._dl_thread = DownloadThread(model_short)
            self._dl_thread.signals.finished.connect(self._on_download_finished)
            self._dl_thread.signals.failed.connect(self._on_download_failed)
            self._dl_thread.start()

        def _on_download_finished(self, path: str) -> None:
            self.download_btn.setEnabled(True)
            self._append(f"<b>Ready</b> -> {path}")

        def _on_download_failed(self, code: str, msg: str) -> None:
            self.download_btn.setEnabled(True)
            _err(code, msg)
            self._append(f'<span style="color:#e05555">ERROR {code}: {msg}</span>')

        def _on_send_clicked(self) -> None:
            text = self.input_line.text().strip()
            if not text:
                return
            self.input_line.clear()
            self._append(f"<b>you&gt;</b> {text}")
            self.history.append({"role": "user", "content": text})
            self.send_btn.setEnabled(False)

            model_short = self._current_model_short()
            self._chat_thread = ChatThread(model_short, list(self.history), system=None)
            self._chat_thread.signals.finished.connect(self._on_chat_finished)
            self._chat_thread.signals.failed.connect(self._on_chat_failed)
            self._chat_thread.start()

        def _on_chat_finished(self, reply: str) -> None:
            self.send_btn.setEnabled(True)
            self.history.append({"role": "assistant", "content": reply})
            self._append(f"<b>agent&gt;</b> {reply}")

        def _on_chat_failed(self, code: str, msg: str) -> None:
            self.send_btn.setEnabled(True)
            self.history.pop()  # don't leave a dangling turn with no reply
            _err(code, msg)
            self._append(f'<span style="color:#e05555">ERROR {code}: {msg}</span>')

        # -- API Keys tab -------------------------------------------------
        def _build_keys_tab(self) -> QWidget:
            from hypernix.interfaces.hyped_pro_core import PROVIDERS
            from hypernix.system.config import get_provider_key, set_provider_key

            w = QWidget()
            form = QFormLayout(w)
            self._key_fields: dict[str, QLineEdit] = {}
            for vendor, info in PROVIDERS.items():
                if info.kind != "cloud" and vendor != "t1":
                    continue
                field = QLineEdit()
                field.setEchoMode(QLineEdit.EchoMode.Password)
                existing = get_provider_key(vendor)
                if existing:
                    field.setPlaceholderText("(key set — leave blank to keep it)")
                save_btn = QPushButton("Save")

                def make_saver(v: str, f: QLineEdit) -> callable:
                    def _save() -> None:
                        key = f.text().strip()
                        if not key:
                            return
                        try:
                            set_provider_key(v, key)
                            os.environ[PROVIDERS[v].auth_env_var or ""] = key
                            f.clear()
                            f.setPlaceholderText("(key set — leave blank to keep it)")
                            QMessageBox.information(w, "hyped-pro", f"Saved key for {v}.")
                        except Exception as exc:  # noqa: BLE001
                            _err("HPG-KEY-001", f"could not save key for {v}: {exc}")
                            QMessageBox.critical(w, "hyped-pro", f"HPG-KEY-001: {exc}")
                    return _save

                save_btn.clicked.connect(make_saver(vendor, field))
                row = QHBoxLayout()
                row.addWidget(field)
                row.addWidget(save_btn)
                row_widget = QWidget()
                row_widget.setLayout(row)
                form.addRow(f"{info.label} ({vendor}):", row_widget)
                self._key_fields[vendor] = field
            return w

    return HypedProMainWindow


# Cache the lazily-built window class so repeated launches (e.g. a second
# /gui call in the same process) don't rebuild it from scratch.
def _get_main_window_cls():
    global HypedProMainWindow
    if "HypedProMainWindow" not in globals():
        globals()["HypedProMainWindow"] = _make_main_window_class()
    return globals()["HypedProMainWindow"]


def _qt_build_window():
    cls = _get_main_window_cls()
    return cls()


# ---------------------------------------------------------------------------
# GTK4 (PyGObject) backend — fallback when PySide6 isn't installed at all.
# ---------------------------------------------------------------------------


def try_launch_gtk() -> bool:
    try:
        import gi
        gi.require_version("Gtk", "4.0")
        from gi.repository import GLib, Gtk
    except (ImportError, ValueError) as exc:
        _err("HPG-GTK-001", f"GTK4 (PyGObject) not installed ({exc}). Install with: pip install hypernix[gui-gtk]")
        return False

    import threading

    from hypernix.interfaces import hyped_pro_core as core

    session = detect_session()
    _dbg(f"session={session} -> using GTK4 (backend does not need QT_QPA_PLATFORM)")

    class HypedProGtkWindow(Gtk.ApplicationWindow):
        def __init__(self, app: Gtk.Application) -> None:
            super().__init__(application=app, title="hyped-pro")
            self.set_default_size(880, 620)
            self.history: list[dict] = []

            root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            root.set_margin_top(8)
            root.set_margin_bottom(8)
            root.set_margin_start(8)
            root.set_margin_end(8)
            self.set_child(root)

            top = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            self.model_dropdown = Gtk.DropDown.new_from_strings(
                [f"{m.badge} {m.short} ({m.kind}/{m.vendor})" for m in core.MODELS]
            )
            top.append(Gtk.Label(label="Model:"))
            top.append(self.model_dropdown)
            download_btn = Gtk.Button(label="Download / ensure ready")
            download_btn.connect("clicked", self._on_download_clicked)
            top.append(download_btn)
            root.append(top)

            scroller = Gtk.ScrolledWindow()
            scroller.set_vexpand(True)
            self.transcript = Gtk.TextView()
            self.transcript.set_editable(False)
            self.transcript.set_wrap_mode(Gtk.WrapMode.WORD)
            scroller.set_child(self.transcript)
            root.append(scroller)

            bottom = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            self.entry = Gtk.Entry(hexpand=True)
            self.entry.connect("activate", self._on_send_clicked)
            send_btn = Gtk.Button(label="Send")
            send_btn.connect("clicked", self._on_send_clicked)
            bottom.append(self.entry)
            bottom.append(send_btn)
            root.append(bottom)

        def _current_model_short(self) -> str:
            idx = self.model_dropdown.get_selected()
            return core.MODELS[idx].short

        def _append(self, text: str) -> None:
            buf = self.transcript.get_buffer()
            end = buf.get_end_iter()
            buf.insert(end, text + "\n")

        def _on_download_clicked(self, _btn: Gtk.Button) -> None:
            model_short = self._current_model_short()
            model = core.get_model(model_short)
            if model.kind != "local":
                self._append(f"{model.short} is a cloud model ({model.vendor}) — set an API key instead.")
                return
            self._append(f"Downloading {model.repo} ...")

            def worker() -> None:
                try:
                    path = core.ensure_downloaded(model, quiet=False)
                    GLib.idle_add(self._append, f"Ready -> {path}")
                except core.HypedProError as exc:
                    _err(exc.code, exc.message)
                    GLib.idle_add(self._append, f"ERROR {exc.code}: {exc.message}")
                except Exception as exc:  # noqa: BLE001
                    _err("HPG-DOWNLOAD-001", str(exc))
                    GLib.idle_add(self._append, f"ERROR HPG-DOWNLOAD-001: {exc}")

            threading.Thread(target=worker, daemon=True).start()

        def _on_send_clicked(self, _widget: object) -> None:
            text = self.entry.get_text().strip()
            if not text:
                return
            self.entry.set_text("")
            self._append(f"you> {text}")
            self.history.append({"role": "user", "content": text})
            model_short = self._current_model_short()
            hist_copy = list(self.history)

            def worker() -> None:
                try:
                    result = core.send_chat_message(model_short, hist_copy)
                    reply = result["content"] or ""
                    self.history.append({"role": "assistant", "content": reply})
                    GLib.idle_add(self._append, f"agent> {reply}")
                except core.HypedProError as exc:
                    self.history.pop()
                    _err(exc.code, exc.message)
                    GLib.idle_add(self._append, f"ERROR {exc.code}: {exc.message}")
                except Exception as exc:  # noqa: BLE001
                    self.history.pop()
                    _err("HPG-CHAT-001", str(exc))
                    GLib.idle_add(self._append, f"ERROR HPG-CHAT-001: {exc}")

            threading.Thread(target=worker, daemon=True).start()

    def on_activate(app: Gtk.Application) -> None:
        win = HypedProGtkWindow(app)
        win.present()

    try:
        app = Gtk.Application(application_id="industries.blaze.hypedpro")
        app.connect("activate", on_activate)
        log.info("hyped-pro GUI running (GTK4)")
        app.run(None)
        return True
    except Exception as exc:  # noqa: BLE001
        _err("HPG-GTK-002", f"GTK4 failed to initialize: {exc}")
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def cli_main(argv: list[str] | None = None) -> int:
    _setup_logging()
    session = detect_session()
    log.info(f"session detection: type={session['session_type']!r} "
              f"wayland_display={session['wayland_display']!r} x11_display={session['x11_display']!r}")
    if not session["is_wayland"] and not session["is_x11"]:
        # Deliberately stop here rather than attempting Qt/GTK anyway: a Qt
        # platform-plugin init failure calls the C++ qFatal() -> abort(),
        # which terminates the process immediately and cannot be caught by
        # a Python try/except (see try_launch_qt()). On a genuinely
        # headless session (no compositor at all) that abort is guaranteed,
        # so the only correct fix is to not attempt it and fail cleanly
        # with a coded, actionable message instead.
        _err("HPG-PLATFORM-001",
             "no Wayland or X11 session detected (no $WAYLAND_DISPLAY, no $DISPLAY, "
             "$XDG_SESSION_TYPE is not 'wayland'/'x11'). The GUI needs a real display/compositor "
             "to attach to — run this on the desktop session itself, not over a plain SSH shell "
             "without X forwarding.")
        return 1

    if try_launch_qt():
        return 0
    if try_launch_gtk():
        return 0

    _err("HPG-BACKEND-001",
         "neither PySide6 (Qt6, X11+Wayland) nor PyGObject/GTK4 is installed. "
         "Install one with: pip install hypernix[gui-qt]  (or hypernix[gui-gtk])")
    return 1


if __name__ == "__main__":
    sys.exit(cli_main())
