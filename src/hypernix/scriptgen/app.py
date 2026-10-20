"""scriptgen.app — the GUI, and the form model underneath it.

Split in two on purpose:

:class:`FormModel` holds the values, the validation, the dirty tracking
and the script preview, and imports no GUI toolkit at all. :class:`ScriptGenApp`
draws it. That split is not architectural tidiness — it is the only way
the interesting behaviour can be tested, because a headless machine
cannot open a Tk window and this package has to import cleanly on the
server that runs the training.

Layout
------
Six tabs, one per :class:`~hypernix.scriptgen.params.ParamGroup`, with
the live script preview always visible on the right. Dense on purpose:
43 parameters across six tabs is seven or eight controls a tab, which
fits without scrolling on a laptop, and seeing the script change as you
drag a slider is the feature — a form that only shows its output after
you press Generate is a form you press Generate on twenty times.

Advanced parameters are hidden behind one checkbox rather than a
separate tab, because the ones that matter (betas, epsilon) matter
*next to* the ones above them.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from .params import ALL_PARAMS, GROUPS, ParamKind, defaults, validate_all
from .templates import as_config, generate, inject
from .theme import FONTS, PALETTE
from .widgets import tk_available

logger = logging.getLogger(__name__)

__all__ = ["FormModel", "ScriptGenApp", "launch"]


class FormModel:
    """Values, validation and preview. No GUI toolkit involved."""

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self.values: dict[str, Any] = defaults()
        if values:
            self.values.update(values)
        self._baseline = dict(self.values)
        self.mode: str = "generate"          # generate | inject | config
        self.config_format: str = "python"
        self.show_advanced = False
        self.target_path: Path | None = None

    # -- values -------------------------------------------------------

    def set(self, name: str, value: Any) -> tuple[bool, str]:
        """Set one parameter, returning ``(ok, message)``.

        A rejected value is *not* stored: leaving an invalid value in the
        model means the preview shows a script that would not run, which
        is worse than the field snapping back.
        """
        param = ALL_PARAMS.get(name)
        if param is None:
            return False, f"Unknown parameter {name!r}"
        ok, message = param.validate(value)
        if ok:
            self.values[name] = param.coerce(value)
        return ok, message

    def get(self, name: str) -> Any:
        return self.values.get(name)

    @property
    def dirty(self) -> bool:
        return self.values != self._baseline

    def changed_from_defaults(self) -> dict[str, tuple[Any, Any]]:
        """What differs from the defaults, for the summary strip.

        Being able to see at a glance that you changed three things and
        not thirty is what makes a 43-field form usable.
        """
        base = defaults()
        return {
            name: (base[name], value)
            for name, value in self.values.items()
            if name in base and value != base[name]
        }

    def reset(self) -> None:
        self.values = defaults()
        self._baseline = dict(self.values)

    def mark_saved(self) -> None:
        self._baseline = dict(self.values)

    # -- visibility ---------------------------------------------------

    def visible_params(self, group_key: str) -> list[Any]:
        """Parameters to draw for a tab, honouring advanced and depends_on."""
        group = next((g for g in GROUPS if g.key == group_key), None)
        if group is None:
            return []
        out = []
        for param in group.params:
            if param.advanced and not self.show_advanced:
                continue
            if param.depends_on:
                gate = self.values.get(param.depends_on)
                # A dependency on a choice field means "shown when that
                # field is set to something", not "shown when it is
                # truthy" — every non-empty string is truthy and that
                # would show everything.
                if isinstance(gate, bool) and not gate:
                    continue
            out.append(param)
        return out

    # -- output -------------------------------------------------------

    def validate(self) -> tuple[list[str], list[str]]:
        return validate_all(self.values)

    def preview(self, existing: str = "") -> str:
        """The script (or config) as it currently stands."""
        if self.mode == "config":
            return as_config(self.values, fmt=self.config_format)
        if self.mode == "inject":
            if not existing and self.target_path and self.target_path.exists():
                existing = self.target_path.read_text(encoding="utf-8", errors="replace")
            if not existing:
                return (
                    "# Inject mode: choose a script to inject into.\n"
                    "# The config block replaces a previous one when the scriptgen markers\n"
                    "# are present, so regenerating does not stack copies.\n"
                    + as_config(self.values)
                )
            return inject(existing, self.values)
        return generate(self.values)

    def save(self, path: str | Path, *, existing: str = "") -> Path:
        """Write the output. Refuses while there are validation errors.

        A generated script that does not run is worse than no script: it
        looks finished, and the error surfaces an hour later on a GPU
        someone is paying for.
        """
        errors, _ = self.validate()
        if errors:
            raise ValueError(
                "Fix these before saving:\n" + "\n".join(f"  - {e}" for e in errors)
            )
        target = Path(path)
        if self.mode == "inject" and not existing and target.exists():
            existing = target.read_text(encoding="utf-8", errors="replace")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.preview(existing), encoding="utf-8")
        self.mark_saved()
        return target

    def to_json(self) -> str:
        return json.dumps(
            {"mode": self.mode, "values": self.values}, indent=2, default=str
        )

    def load_json(self, text: str) -> list[str]:
        """Load a saved preset, returning any values that were rejected."""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Not a valid preset file: {exc}") from exc
        values = payload.get("values") if isinstance(payload, dict) else None
        if not isinstance(values, dict):
            raise ValueError("Preset file has no 'values' object")
        rejected: list[str] = []
        for name, value in values.items():
            ok, message = self.set(name, value)
            if not ok:
                # A preset from a newer version may carry a parameter
                # this build does not have. Skipping it with a note beats
                # refusing the whole file.
                rejected.append(message or f"{name}: rejected")
        self.mode = str(payload.get("mode") or self.mode)
        return rejected


# ---------------------------------------------------------------------------
# The GUI
# ---------------------------------------------------------------------------


class ScriptGenApp:
    """The Tk application. Constructed only when Tk is actually present."""

    def __init__(self, model: FormModel | None = None) -> None:
        ok, reason = tk_available()
        if not ok:
            raise RuntimeError(reason)
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.model = model or FormModel()
        self.widgets: dict[str, Any] = {}

        self.root = tk.Tk()
        self.root.title("hnx scriptgen — HyperNix training script builder")
        self.root.geometry("1440x900")
        self.root.minsize(1140, 720)
        self.root.configure(bg=PALETTE.obsidian)
        # Let the desktop's own window manager draw its native title bar and
        # outer corners. The app's content uses rounded canvas-backed cards,
        # which work consistently even on Tk builds without native rounding.
        self._style()
        self._build()
        self._refresh_preview()

    # -- chrome -------------------------------------------------------

    def _style(self) -> None:
        style = self.ttk.Style(self.root)
        # "clam" is the only built-in theme that honours background
        # colours on all three platforms; the native themes ignore most
        # of what you set, which is how a "dark" app ends up with white
        # notebook tabs on Windows.
        try:
            style.theme_use("clam")
        except Exception:  # noqa: BLE001
            logger.debug("scriptgen: clam theme unavailable")
        style.configure(
            "TNotebook", background=PALETTE.charcoal, borderwidth=0,
            tabmargins=(6, 6, 6, 0),
        )
        style.configure(
            "TNotebook.Tab",
            background=PALETTE.slate, foreground=PALETTE.text_dim,
            padding=(16, 10), font=FONTS["body_bold"], borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", PALETTE.red), ("active", PALETTE.slate_high)],
            foreground=[("selected", PALETTE.text), ("active", PALETTE.text)],
        )
        style.configure("TFrame", background=PALETTE.charcoal)
        style.configure(
            "TCombobox",
            fieldbackground=PALETTE.slate, background=PALETTE.slate,
            foreground=PALETTE.text, arrowcolor=PALETTE.red_bright,
            bordercolor=PALETTE.edge, lightcolor=PALETTE.slate,
            darkcolor=PALETTE.slate, padding=(8, 6), arrowsize=12,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", PALETTE.slate)],
            foreground=[("readonly", PALETTE.text)],
            bordercolor=[("focus", PALETTE.red_bright)],
        )
        style.configure(
            "Dark.Vertical.TScrollbar",
            background=PALETTE.slate_high, troughcolor=PALETTE.charcoal,
            bordercolor=PALETTE.charcoal, arrowcolor=PALETTE.text_dim,
            relief="flat", gripcount=0,
        )
        self.root.option_add("*TCombobox*Listbox.background", PALETTE.slate)
        self.root.option_add("*TCombobox*Listbox.foreground", PALETTE.text)
        self.root.option_add("*TCombobox*Listbox.selectBackground", PALETTE.red)

    class _RoundedPanel:
        """Canvas-backed rounded panel with an inset Tk content frame.

        Tk widgets are rectangular by design. Drawing the surface on a canvas
        and keeping children inset gives cards genuinely rounded visible
        corners without requiring Pillow, a theme package, or a custom window
        manager integration.
        """

        def __init__(
            self, parent: Any, tk: Any, *, background: str, fill: str,
            outline: str, radius: int = 18, inset: int = 12,
        ) -> None:
            self.tk = tk
            self.fill = fill
            self.outline = outline
            self.radius = radius
            self.inset = inset
            self.frame = tk.Frame(parent, bg=background, bd=0, highlightthickness=0)
            self.canvas = tk.Canvas(
                self.frame, bg=background, bd=0, highlightthickness=0,
                relief="flat",
            )
            self.canvas.pack(fill="both", expand=True)
            self.content = tk.Frame(self.canvas, bg=fill, bd=0, highlightthickness=0)
            self.window = self.canvas.create_window(
                inset, inset, anchor="nw", window=self.content,
            )
            self.surface = self.canvas.create_polygon(
                0, 0, 1, 0, 1, 1, 0, 1,
                fill=fill, outline=outline, width=1, smooth=True,
                splinesteps=16,
            )
            # Keep the surface behind the embedded content frame.
            self.canvas.tag_lower(self.surface)
            self.canvas.bind("<Configure>", self._resize)

        def pack(self, *args: Any, **kwargs: Any) -> None:
            self.frame.pack(*args, **kwargs)

        def _resize(self, event: Any) -> None:
            width = max(2, event.width - 1)
            height = max(2, event.height - 1)
            radius = max(4, min(self.radius, width // 2, height // 2))
            points: list[float] = []
            corners = (
                (width - radius, radius, -90, 0),
                (width - radius, height - radius, 0, 90),
                (radius, height - radius, 90, 180),
                (radius, radius, 180, 270),
            )
            for cx, cy, start, end in corners:
                for step in range(7):
                    angle = math.radians(start + (end - start) * step / 6)
                    points.extend((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
            self.canvas.coords(self.surface, *points)
            self.canvas.coords(self.window, self.inset, self.inset)
            self.canvas.itemconfigure(
                self.window,
                width=max(1, width - self.inset * 2),
                height=max(1, height - self.inset * 2),
            )

    def _build(self) -> None:
        tk = self.tk
        self._build_header()

        body = tk.Frame(self.root, bg=PALETTE.obsidian)
        body.pack(fill="both", expand=True, padx=18, pady=(4, 12))

        left_card = self._RoundedPanel(
            body, tk, background=PALETTE.obsidian, fill=PALETTE.charcoal,
            outline=PALETTE.edge, radius=20, inset=10,
        )
        left_card.pack(side="left", fill="both", expand=True, padx=(0, 9))
        left = left_card.content
        self.notebook = self.ttk.Notebook(left)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        for group in GROUPS:
            self.notebook.add(self._build_group(group), text=group.title)

        right_card = self._RoundedPanel(
            body, tk, background=PALETTE.obsidian, fill=PALETTE.charcoal,
            outline=PALETTE.edge, radius=20, inset=10,
        )
        right_card.frame.configure(width=570)
        right_card.pack(side="right", fill="both", padx=(9, 0))
        right_card.frame.pack_propagate(False)
        right = right_card.content
        self._build_preview(right)
        self._build_status()

    def _build_header(self) -> None:
        tk = self.tk
        header = tk.Frame(self.root, bg=PALETTE.obsidian)
        header.pack(fill="x", padx=22, pady=(16, 10))
        brand = tk.Frame(header, bg=PALETTE.obsidian)
        brand.pack(side="left", anchor="w")
        tk.Label(
            brand, text="HN", font=("DejaVu Sans", 10, "bold"),
            bg=PALETTE.red, fg=PALETTE.text, padx=8, pady=5,
        ).pack(side="left", padx=(0, 12), pady=(1, 0))
        titles = tk.Frame(brand, bg=PALETTE.obsidian)
        titles.pack(side="left", anchor="w")
        tk.Label(
            titles, text="scriptgen", font=FONTS["title"],
            bg=PALETTE.obsidian, fg=PALETTE.text,
        ).pack(anchor="w")
        tk.Label(
            titles, text="HyperNix training script builder  ·  live preview",
            font=FONTS["small"], bg=PALETTE.obsidian, fg=PALETTE.text_faint,
        ).pack(anchor="w", pady=(1, 0))

        for label, command in (
            ("Reset", self._on_reset),
            ("Save preset", self._on_save_preset),
            ("Load preset", self._on_load),
            ("Save script", self._on_save),
        ):
            self._button(header, label, command, primary=(label == "Save script")).pack(
                side="right", padx=(7, 0), pady=3,
            )

        from .widgets import ToggleSwitch

        advanced = tk.Frame(header, bg=PALETTE.obsidian)
        advanced.pack(side="right", padx=(0, 18))
        tk.Label(advanced, text="Advanced", font=FONTS["small"],
                 bg=PALETTE.obsidian, fg=PALETTE.text_dim).pack(side="left", padx=(0, 6))
        self.advanced_toggle = ToggleSwitch(
            advanced, value=False, command=self._on_advanced
        )
        self.advanced_toggle.configure(bg=PALETTE.obsidian)
        self.advanced_toggle.canvas.configure(bg=PALETTE.obsidian)
        self.advanced_toggle.pack(side="left")

    def _button(self, parent: Any, label: str, command: Any, *, primary: bool = False) -> Any:
        tk = self.tk
        normal = PALETTE.red if primary else PALETTE.slate
        hover = PALETTE.red_bright if primary else PALETTE.slate_high
        button = tk.Button(
            parent, text=label, command=command, font=FONTS["body_bold"],
            bg=normal, fg=PALETTE.text,
            activebackground=hover, activeforeground=PALETTE.text,
            relief="flat", bd=0, padx=13, pady=7, cursor="hand2",
            highlightthickness=0,
        )
        button.bind("<Enter>", lambda _event: button.configure(bg=hover))
        button.bind("<Leave>", lambda _event: button.configure(bg=normal))
        return button

    def _build_group(self, group: Any) -> Any:
        tk = self.tk
        frame = tk.Frame(self.notebook, bg=PALETTE.charcoal)
        tk.Label(frame, text=group.description, font=FONTS["small"],
                 bg=PALETTE.charcoal, fg=PALETTE.text_faint,
                 anchor="w", wraplength=600, justify="left").pack(
            fill="x", padx=18, pady=(17, 10),
        )
        rows = tk.Frame(frame, bg=PALETTE.charcoal)
        rows.pack(fill="both", expand=True, padx=18, pady=(0, 16))
        rows.grid_columnconfigure(1, weight=1)
        for index, param in enumerate(group.params):
            self._build_row(rows, param, index)
        return frame

    def _build_row(self, parent: Any, param: Any, row: int) -> None:
        tk = self.tk
        from .widgets import DualSlider, LogSlider, ToggleSwitch

        label = tk.Label(parent, text=param.label, font=FONTS["body_bold"],
                         bg=PALETTE.charcoal, fg=PALETTE.text_dim, anchor="w")
        label.grid(row=row * 2, column=0, sticky="w", pady=(8, 0), padx=(0, 16))

        value = self.model.get(param.name)
        holder = tk.Frame(parent, bg=PALETTE.charcoal)
        holder.grid(row=row * 2, column=1, sticky="ew", pady=(8, 0))

        if param.kind is ParamKind.TOGGLE:
            widget = ToggleSwitch(
                holder, value=bool(value),
                command=lambda v, n=param.name: self._on_change(n, v),
            )
            widget.pack(anchor="w")
        elif param.kind is ParamKind.RANGE:
            low, high = value if isinstance(value, (list, tuple)) else (0.0, 1.0)
            widget = DualSlider(
                holder, minimum=param.minimum or 0.0, maximum=param.maximum or 1.0,
                low=low, high=high,
                logarithmic=(param.minimum or 0) > 0 and (param.maximum or 0) > 0,
                command=lambda lo, hi, n=param.name: self._on_change(n, (lo, hi)),
            )
            widget.pack(anchor="w", fill="x")
        elif param.kind is ParamKind.LOG_FLOAT:
            widget = LogSlider(
                holder, minimum=param.minimum or 1e-8, maximum=param.maximum or 1.0,
                value=float(value),
                command=lambda v, n=param.name: self._on_change(n, v),
            )
            widget.pack(anchor="w", fill="x")
        elif param.kind is ParamKind.CHOICE:
            var = tk.StringVar(value=str(value))
            widget = self.ttk.Combobox(
                holder, textvariable=var, state="readonly",
                values=[c[0] for c in param.choices], font=FONTS["body"],
            )
            widget.pack(anchor="w", fill="x", ipady=2)
            widget.bind(
                "<<ComboboxSelected>>",
                lambda _e, n=param.name, v=var: self._on_change(n, v.get()),
            )
            widget.var = var  # type: ignore[attr-defined]
        elif param.kind is ParamKind.MULTILINE:
            widget = tk.Text(
                holder, height=3, font=FONTS["body"], bg=PALETTE.slate, fg=PALETTE.text,
                insertbackground=PALETTE.red_bright, relief="flat", bd=0,
                highlightthickness=1, highlightbackground=PALETTE.edge,
                highlightcolor=PALETTE.red,
            )
            widget.insert("1.0", str(value or ""))
            widget.pack(anchor="w", fill="x")
            widget.bind(
                "<KeyRelease>",
                lambda _e, n=param.name, w=widget: self._on_change(
                    n, w.get("1.0", "end-1c")
                ),
            )
        else:
            var = tk.StringVar(value=str(value))
            widget = tk.Entry(
                holder, textvariable=var, font=FONTS["body"],
                bg=PALETTE.slate, fg=PALETTE.text, insertbackground=PALETTE.red_bright,
                relief="flat", bd=0, highlightthickness=1,
                highlightbackground=PALETTE.edge, highlightcolor=PALETTE.red,
            )
            widget.pack(anchor="w", fill="x", ipady=4)
            var.trace_add(
                "write", lambda *_a, n=param.name, v=var: self._on_change(n, v.get())
            )
            widget.var = var  # type: ignore[attr-defined]

        self.widgets[param.name] = widget
        if param.hint:
            tk.Label(parent, text=param.hint, font=FONTS["tiny"],
                     bg=PALETTE.charcoal, fg=PALETTE.text_faint,
                     anchor="w", wraplength=520, justify="left").grid(
                row=row * 2 + 1, column=1, sticky="w", pady=(1, 4))

    def _build_preview(self, parent: Any) -> None:
        tk = self.tk
        bar = tk.Frame(parent, bg=PALETTE.charcoal)
        bar.pack(fill="x", padx=10, pady=(8, 10))
        title_group = tk.Frame(bar, bg=PALETTE.charcoal)
        title_group.pack(side="left")
        tk.Label(title_group, text="Live output", font=FONTS["heading"],
                 bg=PALETTE.charcoal, fg=PALETTE.text).pack(anchor="w")
        tk.Label(title_group, text="Updates as you change parameters", font=FONTS["tiny"],
                 bg=PALETTE.charcoal, fg=PALETTE.text_faint).pack(anchor="w", pady=(2, 0))
        self.mode_var = tk.StringVar(value="generate")
        for mode, caption in (
            ("generate", "full script"), ("inject", "inject"), ("config", "config only")
        ):
            tk.Radiobutton(
                bar, text=caption, value=mode, variable=self.mode_var,
                command=self._on_mode, font=FONTS["small"],
                bg=PALETTE.charcoal, fg=PALETTE.text_dim,
                selectcolor=PALETTE.slate_high, activebackground=PALETTE.charcoal,
                activeforeground=PALETTE.text, indicatoron=True, bd=0,
                highlightthickness=0,
            ).pack(side="right", padx=4, pady=(2, 0))

        wrapper_card = self._RoundedPanel(
            parent, tk, background=PALETTE.charcoal, fill=PALETTE.obsidian,
            outline=PALETTE.edge, radius=14, inset=8,
        )
        wrapper_card.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        wrapper = wrapper_card.content
        scroll = self.ttk.Scrollbar(wrapper, orient="vertical", style="Dark.Vertical.TScrollbar")
        scroll.pack(side="right", fill="y", padx=(2, 0))
        self.preview = tk.Text(
            wrapper, font=FONTS["code"], bg=PALETTE.obsidian, fg=PALETTE.text,
            insertbackground=PALETTE.red_bright, relief="flat", bd=0,
            highlightthickness=0, wrap="none", yscrollcommand=scroll.set,
            padx=12, pady=12, selectbackground=PALETTE.red_dim,
            selectforeground=PALETTE.text, spacing1=1,
        )
        self.preview.pack(side="left", fill="both", expand=True)
        scroll.configure(command=self.preview.yview)
        # Enough highlighting to make the block boundaries findable
        # without pulling in a syntax highlighter.
        self.preview.tag_configure("marker", foreground=PALETTE.red_text)
        self.preview.tag_configure("comment", foreground=PALETTE.text_faint)

    def _build_status(self) -> None:
        tk = self.tk
        status_card = self._RoundedPanel(
            self.root, tk, background=PALETTE.obsidian, fill=PALETTE.charcoal,
            outline=PALETTE.edge, radius=13, inset=5,
        )
        status_card.pack(fill="x", side="bottom", padx=18, pady=(0, 12))
        self.status = status_card.content
        tk.Label(
            self.status, text="●", font=FONTS["small"],
            bg=PALETTE.charcoal, fg=PALETTE.ok,
        ).pack(side="left", padx=(10, 7), pady=5)
        self.status_label = tk.Label(
            self.status, text="", font=FONTS["small"], anchor="w",
            bg=PALETTE.charcoal, fg=PALETTE.text_dim,
        )
        self.status_label.pack(side="left", padx=(0, 10), pady=7)

    # -- events -------------------------------------------------------

    def _on_change(self, name: str, value: Any) -> None:
        ok, message = self.model.set(name, value)
        self._set_status(message, level="error" if not ok else ("warn" if message else "ok"))
        self._refresh_preview()

    def _on_advanced(self, value: bool) -> None:
        self.model.show_advanced = value
        self._set_status(
            "Advanced parameters shown" if value else "Advanced parameters hidden"
        )

    def _on_mode(self) -> None:
        self.model.mode = self.mode_var.get()
        self._refresh_preview()

    def _on_reset(self) -> None:
        self.model.reset()
        for name, widget in self.widgets.items():
            value = self.model.get(name)
            if hasattr(widget, "var"):
                widget.var.set(str(value))
            elif hasattr(widget, "set"):
                widget.set(*value) if isinstance(value, tuple) else widget.set(value)
        self._refresh_preview()
        self._set_status("Reset to defaults")

    def _on_save(self) -> None:
        from tkinter import filedialog, messagebox

        errors, _ = self.model.validate()
        if errors:
            messagebox.showerror("Cannot save", "\n".join(errors))
            return
        default = "train.py" if self.model.mode != "config" else "config.py"
        path = filedialog.asksaveasfilename(defaultextension=".py", initialfile=default)
        if not path:
            return
        try:
            saved = self.model.save(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not save", str(exc))
            return
        self._set_status(f"Saved {saved}")

    def _on_load(self) -> None:
        from tkinter import filedialog, messagebox

        path = filedialog.askopenfilename(filetypes=[("Preset", "*.json"), ("All", "*")])
        if not path:
            return
        try:
            rejected = self.model.load_json(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not load", str(exc))
            return
        self._on_reset_widgets()
        self._set_status(
            f"Loaded {Path(path).name}"
            + (f" ({len(rejected)} value(s) skipped)" if rejected else "")
        )

    def _on_reset_widgets(self) -> None:
        for name, widget in self.widgets.items():
            value = self.model.get(name)
            if hasattr(widget, "var"):
                widget.var.set(str(value))
            elif hasattr(widget, "set"):
                widget.set(*value) if isinstance(value, tuple) else widget.set(value)
        self._refresh_preview()

    def _on_save_preset(self) -> None:
        from tkinter import filedialog, messagebox

        path = filedialog.asksaveasfilename(defaultextension=".json", initialfile="preset.json")
        if not path:
            return
        try:
            Path(path).write_text(self.model.to_json(), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Could not save", str(exc))
            return
        self._set_status(f"Preset saved to {Path(path).name}")

    # -- rendering ----------------------------------------------------

    def _refresh_preview(self) -> None:
        text = self.model.preview()
        self.preview.configure(state="normal")
        self.preview.delete("1.0", "end")
        self.preview.insert("1.0", text)
        for index, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("# --- hypernix scriptgen"):
                self.preview.tag_add("marker", f"{index}.0", f"{index}.end")
            elif stripped.startswith("#"):
                self.preview.tag_add("comment", f"{index}.0", f"{index}.end")
        self.preview.configure(state="disabled")

        errors, warnings = self.model.validate()
        changed = len(self.model.changed_from_defaults())
        if errors:
            self._set_status(f"{len(errors)} problem(s): {errors[0]}", level="error")
        elif warnings:
            self._set_status(f"{changed} changed · {warnings[0]}", level="warn")
        else:
            self._set_status(f"{changed} parameter(s) changed from defaults")

    def _set_status(self, message: str, *, level: str = "ok") -> None:
        colour = {
            "ok": PALETTE.text_dim, "warn": PALETTE.warn, "error": PALETTE.error,
        }.get(level, PALETTE.text_dim)
        self.status_label.configure(text=message, fg=colour)

    def run(self) -> None:
        self.root.mainloop()


def launch(values: dict[str, Any] | None = None) -> int:
    """Open the GUI, or explain why it cannot open."""
    try:
        app = ScriptGenApp(FormModel(values))
    except RuntimeError as exc:
        print(f"scriptgen: {exc}")
        return 1
    app.run()
    return 0
