"""scriptgen.app — the GUI, and the form model underneath it.

Split in two on purpose:

:class:`FormModel` holds the values, the validation, the dirty tracking
and the script preview, and imports no GUI toolkit at all. :class:`ScriptGenApp`
draws it. That split is not architectural tidiness — it is the only way
the interesting behaviour can be tested, because a headless machine
cannot open a Tk window and this package has to import cleanly on the
server that runs the training.

Layout (0.72.8rc1)
------------------
Three columns. On the left, the seven sections as a list, each with a
count of what was changed in it and a mark when something there is
wrong, plus a search box over every setting and the Advanced switch. In
the middle, the section's settings as cards: the label, the control (a
slider beside an entry for numbers, the options side by side when there
are four or fewer), what the setting does, and a Reset on anything
changed. On the right, the output with line numbers and syntax colour,
and under it the checks: each problem or warning, which jumps to its
setting when clicked.

"Start from" in the header lays a recipe (:mod:`.recipes`) over the
defaults. Ctrl+S saves the script, Ctrl+F searches, Ctrl+1..7 opens a
section.

The 0.72.1 window used notebook tabs, which cut their titles to "Model
an" and "Precision anc"; a status strip that took a third of the window;
an Advanced switch that changed nothing on screen until the tab was
rebuilt; and a preview in one colour.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .params import ALL_PARAMS, GROUPS, ParamKind, defaults, search, validate_all
from .templates import CONFIG_FORMATS, as_config, generate, inject
from .theme import PALETTE, pick_fonts
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
        #: The recipe the values started from, if any.
        self.recipe: str = ""

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

    def starting_point(self) -> dict[str, Any]:
        """The defaults, or the recipe's values when one was applied."""
        if self.recipe:
            from .recipes import apply_recipe

            return apply_recipe(self.recipe)
        return defaults()

    def changed_from_defaults(self) -> dict[str, tuple[Any, Any]]:
        """What differs from the starting point, for the summary strip.

        Being able to see at a glance that you changed three things and
        not thirty is what makes a 62-field form usable.
        """
        base = self.starting_point()
        return {
            name: (base[name], value)
            for name, value in self.values.items()
            if name in base and value != base[name]
        }

    def is_changed(self, name: str) -> bool:
        return name in self.changed_from_defaults()

    def reset(self) -> None:
        self.values = defaults()
        self.recipe = ""
        self._baseline = dict(self.values)

    def reset_param(self, name: str) -> None:
        """Back to the starting point's value for one setting."""
        if name in ALL_PARAMS:
            self.values[name] = self.starting_point()[name]

    def apply_recipe(self, key: str) -> None:
        """Start again from a recipe: the defaults with its values laid over."""
        from .recipes import apply_recipe

        self.values = apply_recipe(key)
        self.recipe = key

    def mark_saved(self) -> None:
        self._baseline = dict(self.values)

    # -- visibility ---------------------------------------------------

    def visible_params(self, group_key: str) -> list[Any]:
        """Parameters to draw for a section, honouring advanced and depends_on."""
        group = next((g for g in GROUPS if g.key == group_key), None)
        if group is None:
            return []
        return [p for p in group.params if self.is_visible(p)]

    def is_visible(self, param: Any) -> bool:
        if param.advanced and not self.show_advanced:
            return False
        # A dependency on a choice field names the values it applies to
        # ("loss_function=label_smoothed_ce"), so label smoothing is not
        # shown beside a loss that ignores it.
        return param.shown_with(self.values)

    def search(self, query: str) -> list[Any]:
        """Visible settings matching *query*, across every section."""
        return [p for p in search(query) if self.is_visible(p)]

    def section_status(self, group_key: str) -> tuple[int, int]:
        """``(changed, problems)`` in one section, for the sidebar."""
        group = next((g for g in GROUPS if g.key == group_key), None)
        if group is None:
            return 0, 0
        changed = self.changed_from_defaults()
        names = {p.name for p in group.params}
        problems = sum(1 for name, _message in self.issues()[0] if name in names)
        return sum(1 for n in changed if n in names), problems

    # -- output -------------------------------------------------------

    def validate(self) -> tuple[list[str], list[str]]:
        return validate_all(self.values)

    def issues(self) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """``(errors, warnings)``, each message paired with the setting it
        is about ("" when it is about several), so the GUI can jump there."""
        errors, warnings = self.validate()
        return [(_owner(m), m) for m in errors], [(_owner(m), m) for m in warnings]

    def summary(self) -> dict[str, Any]:
        """The few facts the status line shows."""
        v = self.values
        micro, accum = int(v.get("micro_batch") or 1), int(v.get("gradient_accumulation") or 1)
        return {
            "effective_batch": micro * accum,
            "tokens_per_step": micro * accum * int(v.get("sequence_length") or 0),
            "optimizer": v.get("optimizer"),
            "precision": v.get("precision"),
            "device": v.get("device"),
            "lora": bool(v.get("use_lora")),
            "changed": len(self.changed_from_defaults()),
        }

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
            {"mode": self.mode, "recipe": self.recipe, "values": self.values}, indent=2, default=str
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
        recipe = str(payload.get("recipe") or "")
        from .recipes import RECIPES

        self.recipe = recipe if any(r.key == recipe for r in RECIPES) else ""
        return rejected


def _owner(message: str) -> str:
    """The setting a validation message is about: messages from
    :meth:`Param.validate` start with its label."""
    for param in ALL_PARAMS.values():
        if message.startswith(f"{param.label}:"):
            return param.name
    lowered = message.lower()
    for name, words in (("dataset", "no dataset"), ("six_bit_mode", "6-bit"), ("hub_repo", "hub repo"),
                        ("lora_targets", "target layers"), ("micro_batch", "effective batch"),
                        ("val_split", "held-out"), ("max_steps", "max_steps"),
                        ("precision", "bf16"), ("save_total_limit", "checkpoint")):
        if words in lowered:
            return name
    return ""


# ---------------------------------------------------------------------------
# The GUI
# ---------------------------------------------------------------------------

_SIDEBAR_WIDTH = 272
_PREVIEW_WIDTH = 600


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
        self.section = GROUPS[0].key
        self._query = ""
        self._pending: str | None = None
        self._nav: dict[str, dict[str, Any]] = {}

        self.root = tk.Tk()
        self.root.title("scriptgen · HyperNix training script builder")
        self.root.geometry("1440x900")
        self.root.minsize(1180, 720)
        self.root.configure(bg=PALETTE.obsidian)
        self.fonts = pick_fonts(self.root)
        self._style()
        self._build()
        self._show_section(self.section)
        self._refresh_preview()
        self._bind_keys()

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
            "TCombobox",
            fieldbackground=PALETTE.slate, background=PALETTE.slate,
            foreground=PALETTE.text, arrowcolor=PALETTE.text_dim,
            bordercolor=PALETTE.edge, lightcolor=PALETTE.slate,
            darkcolor=PALETTE.slate, padding=(8, 5), arrowsize=13,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", PALETTE.slate)],
            foreground=[("readonly", PALETTE.text)],
            bordercolor=[("focus", PALETTE.red)],
            arrowcolor=[("active", PALETTE.text)],
        )
        style.configure(
            "Dark.Vertical.TScrollbar",
            background=PALETTE.slate_high, troughcolor=PALETTE.obsidian,
            bordercolor=PALETTE.obsidian, arrowcolor=PALETTE.text_faint,
            lightcolor=PALETTE.slate_high, darkcolor=PALETTE.slate_high,
            relief="flat", gripcount=0, arrowsize=0, width=10,
        )
        style.map("Dark.Vertical.TScrollbar", background=[("active", PALETTE.edge)])
        style.configure(
            "Dark.Horizontal.TScrollbar",
            background=PALETTE.slate_high, troughcolor=PALETTE.obsidian,
            bordercolor=PALETTE.obsidian, lightcolor=PALETTE.slate_high,
            darkcolor=PALETTE.slate_high, relief="flat", gripcount=0, arrowsize=0, width=10,
        )
        self.root.option_add("*TCombobox*Listbox.background", PALETTE.slate)
        self.root.option_add("*TCombobox*Listbox.foreground", PALETTE.text)
        self.root.option_add("*TCombobox*Listbox.selectBackground", PALETTE.red)
        self.root.option_add("*TCombobox*Listbox.font", self.fonts["body"])

    def _build(self) -> None:
        tk = self.tk
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self._build_header()

        body = tk.Frame(self.root, bg=PALETTE.edge)
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_columnconfigure(2, minsize=_PREVIEW_WIDTH)

        self._build_sidebar(body)
        self._build_form(body)
        self._build_preview(body)
        self._build_status()

    # -- header -------------------------------------------------------

    def _build_header(self) -> None:
        tk = self.tk
        header = tk.Frame(self.root, bg=PALETTE.obsidian, height=70)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(2, weight=1)

        mark = tk.Canvas(header, width=34, height=34, bg=PALETTE.obsidian, highlightthickness=0)
        mark.grid(row=0, column=0, padx=(20, 12), pady=14)
        _rounded(mark, 1, 1, 33, 33, 8, fill=PALETTE.red)
        mark.create_text(17, 17, text="HN", fill=PALETTE.text, font=self.fonts["body_bold"])
        titles = tk.Frame(header, bg=PALETTE.obsidian)
        titles.grid(row=0, column=1, sticky="w")
        tk.Label(titles, text="scriptgen", font=self.fonts["title"], bg=PALETTE.obsidian,
                 fg=PALETTE.text).pack(anchor="w")
        tk.Label(titles, text="HyperNix training scripts", font=self.fonts["small"],
                 bg=PALETTE.obsidian, fg=PALETTE.text_faint).pack(anchor="w")

        from .recipes import RECIPES

        picker = tk.Frame(header, bg=PALETTE.obsidian)
        picker.grid(row=0, column=2, sticky="w", padx=(36, 0))
        tk.Label(picker, text="Start from", font=self.fonts["small"], bg=PALETTE.obsidian,
                 fg=PALETTE.text_dim).grid(row=0, column=0, padx=(0, 8))
        self._recipe_titles = ["Defaults", *(r.title for r in RECIPES)]
        self.recipe_var = tk.StringVar(value=self._recipe_title(self.model.recipe))
        combo = self.ttk.Combobox(picker, textvariable=self.recipe_var, state="readonly",
                                  values=self._recipe_titles, width=30, font=self.fonts["body"])
        combo.grid(row=0, column=1, sticky="w")
        combo.bind("<<ComboboxSelected>>", lambda _e: self._on_recipe())
        self.recipe_note = tk.Label(picker, text="", font=self.fonts["tiny"], bg=PALETTE.obsidian,
                                    fg=PALETTE.text_faint, anchor="w", justify="left", wraplength=520)
        self.recipe_note.grid(row=1, column=1, sticky="w", pady=(3, 0))
        self._describe_recipe()

        actions = tk.Frame(header, bg=PALETTE.obsidian)
        actions.grid(row=0, column=3, sticky="e", padx=(0, 18))
        for label, command, primary, tip in (
            ("Copy", self._on_copy, False, "Copy the output (Ctrl+Shift+C)"),
            ("Load preset", self._on_load, False, "Ctrl+O"),
            ("Save preset", self._on_save_preset, False, ""),
            ("Save script", self._on_save, True, "Ctrl+S"),
        ):
            self._button(actions, label, command, primary=primary, tip=tip).pack(side="left", padx=(8, 0))

    def _recipe_title(self, key: str) -> str:
        from .recipes import RECIPES

        return next((r.title for r in RECIPES if r.key == key), "Defaults")

    def _describe_recipe(self) -> None:
        from .recipes import RECIPES

        recipe = next((r for r in RECIPES if r.key == self.model.recipe), None)
        self.recipe_note.configure(
            text=recipe.description if recipe else "Every setting at its default. Pick a recipe "
                                                    "for a ready-made starting point.")

    def _button(self, parent: Any, label: str, command: Any, *, primary: bool = False,
                tip: str = "", small: bool = False) -> Any:
        tk = self.tk
        normal = PALETTE.red if primary else PALETTE.slate
        hover = PALETTE.red_bright if primary else PALETTE.slate_high
        button = tk.Label(
            parent, text=label, font=self.fonts["small" if small else "body_bold"],
            bg=normal, fg=PALETTE.text, padx=10 if small else 14, pady=3 if small else 7,
            cursor="hand2", bd=0, highlightthickness=1,
            highlightbackground=PALETTE.red if primary else PALETTE.edge,
        )
        button.bind("<Button-1>", lambda _e: command())
        button.bind("<Enter>", lambda _e: (button.configure(bg=hover),
                                           tip and self._set_status(tip, level="hint")))
        button.bind("<Leave>", lambda _e: button.configure(bg=normal))
        return button

    # -- sidebar ------------------------------------------------------

    def _build_sidebar(self, parent: Any) -> None:
        tk = self.tk
        side = tk.Frame(parent, bg=PALETTE.obsidian, width=_SIDEBAR_WIDTH)
        side.grid(row=0, column=0, sticky="ns", padx=(0, 1))
        side.grid_propagate(False)
        side.pack_propagate(False)

        box = tk.Frame(side, bg=PALETTE.slate, highlightthickness=1,
                       highlightbackground=PALETTE.edge, highlightcolor=PALETTE.red)
        box.pack(fill="x", padx=14, pady=(14, 12))
        tk.Label(box, text="⌕", font=self.fonts["body"], bg=PALETTE.slate,
                 fg=PALETTE.text_faint).pack(side="left", padx=(8, 2))
        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(box, textvariable=self.search_var, font=self.fonts["body"],
                                     bg=PALETTE.slate, fg=PALETTE.text, bd=0, relief="flat",
                                     insertbackground=PALETTE.red_bright, highlightthickness=0)
        self.search_entry.pack(side="left", fill="x", expand=True, ipady=6, padx=(2, 8))
        self._placeholder(self.search_entry, self.search_var, "Search settings   Ctrl+F")
        self.search_var.trace_add("write", lambda *_a: self._on_search())

        tk.Label(side, text="SECTIONS", font=self.fonts["tiny"], bg=PALETTE.obsidian,
                 fg=PALETTE.text_faint).pack(anchor="w", padx=18, pady=(4, 4))
        for index, group in enumerate(GROUPS, start=1):
            self._nav_item(side, group, index)

        tk.Frame(side, bg=PALETTE.edge, height=1).pack(fill="x", padx=14, pady=14)
        from .widgets import ToggleSwitch

        row = tk.Frame(side, bg=PALETTE.obsidian)
        row.pack(fill="x", padx=18)
        texts = tk.Frame(row, bg=PALETTE.obsidian)
        texts.pack(side="left", fill="x", expand=True)
        tk.Label(texts, text="Advanced settings", font=self.fonts["body_bold"], bg=PALETTE.obsidian,
                 fg=PALETTE.text_dim).pack(anchor="w")
        tk.Label(texts, text="Betas, epsilon, splits, ports", font=self.fonts["tiny"],
                 bg=PALETTE.obsidian, fg=PALETTE.text_faint).pack(anchor="w")
        self.advanced_toggle = ToggleSwitch(row, value=self.model.show_advanced,
                                            command=self._on_advanced, width=40, height=22)
        self.advanced_toggle.configure(bg=PALETTE.obsidian)
        self.advanced_toggle.canvas.configure(bg=PALETTE.obsidian)
        self.advanced_toggle.pack(side="right")

        keys = tk.Label(side, text="Ctrl+S save   Ctrl+F search\nCtrl+1…7 sections   Esc clear",
                        font=self.fonts["tiny"], bg=PALETTE.obsidian, fg=PALETTE.text_faint,
                        justify="left")
        keys.pack(side="bottom", anchor="w", padx=18, pady=14)

    def _nav_item(self, parent: Any, group: Any, index: int) -> None:
        tk = self.tk
        item = tk.Frame(parent, bg=PALETTE.obsidian, cursor="hand2")
        item.pack(fill="x", padx=8, pady=1)
        bar = tk.Frame(item, bg=PALETTE.obsidian, width=3)
        bar.pack(side="left", fill="y")
        number = tk.Label(item, text=str(index), font=self.fonts["small"], width=2,
                          bg=PALETTE.obsidian, fg=PALETTE.text_faint)
        number.pack(side="left", padx=(8, 4), pady=8)
        # The badge is packed first so a long title gives way to it, not
        # the other way round.
        badge = tk.Label(item, text="", font=self.fonts["tiny"], bg=PALETTE.obsidian,
                         fg=PALETTE.text_faint, padx=4)
        badge.pack(side="right", padx=(0, 8))
        title = tk.Label(item, text=group.title, font=self.fonts["nav"], bg=PALETTE.obsidian,
                         fg=PALETTE.text_dim, anchor="w")
        title.pack(side="left", fill="x", expand=True)
        parts = {"item": item, "bar": bar, "number": number, "title": title, "badge": badge}
        self._nav[group.key] = parts
        for widget in parts.values():
            widget.bind("<Button-1>", lambda _e, k=group.key: self._show_section(k))
            widget.bind("<Enter>", lambda _e, k=group.key: self._paint_nav(k, hover=True))
            widget.bind("<Leave>", lambda _e, k=group.key: self._paint_nav(k))

    def _paint_nav(self, key: str, *, hover: bool = False) -> None:
        parts = self._nav[key]
        chosen = key == self.section and not self._query
        bg = PALETTE.charcoal if chosen else (PALETTE.slate if hover else PALETTE.obsidian)
        for name in ("item", "number", "title", "badge"):
            parts[name].configure(bg=bg)
        parts["bar"].configure(bg=PALETTE.red if chosen else bg)
        parts["title"].configure(fg=PALETTE.text if chosen else PALETTE.text_dim,
                                 font=self.fonts["nav_bold" if chosen else "nav"])
        changed, problems = self.model.section_status(key)
        if problems:
            parts["badge"].configure(text=f"● {problems}", fg=PALETTE.red_text)
        elif changed:
            parts["badge"].configure(text=f"{changed} ✎", fg=PALETTE.text_faint)
        else:
            parts["badge"].configure(text="")

    def _paint_all_nav(self) -> None:
        for key in self._nav:
            self._paint_nav(key)

    # -- form ---------------------------------------------------------

    def _build_form(self, parent: Any) -> None:
        tk = self.tk
        panel = tk.Frame(parent, bg=PALETTE.charcoal)
        panel.grid(row=0, column=1, sticky="nsew", padx=(0, 1))
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=1)

        head = tk.Frame(panel, bg=PALETTE.charcoal)
        head.grid(row=0, column=0, sticky="ew", padx=28, pady=(22, 10))
        self.form_title = tk.Label(head, text="", font=self.fonts["heading"], bg=PALETTE.charcoal,
                                   fg=PALETTE.text, anchor="w")
        self.form_title.pack(anchor="w")
        self.form_lede = tk.Label(head, text="", font=self.fonts["body"], bg=PALETTE.charcoal,
                                  fg=PALETTE.text_faint, anchor="w", justify="left", wraplength=520)
        self.form_lede.pack(anchor="w", fill="x", pady=(3, 0))
        head.bind("<Configure>", lambda e: self.form_lede.configure(wraplength=max(240, e.width - 8)))

        holder = tk.Frame(panel, bg=PALETTE.charcoal)
        holder.grid(row=1, column=0, sticky="nsew")
        holder.grid_columnconfigure(0, weight=1)
        holder.grid_rowconfigure(0, weight=1)
        self.form_canvas = tk.Canvas(holder, bg=PALETTE.charcoal, highlightthickness=0, bd=0)
        self.form_canvas.grid(row=0, column=0, sticky="nsew")
        scroll = self.ttk.Scrollbar(holder, orient="vertical", style="Dark.Vertical.TScrollbar",
                                    command=self.form_canvas.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.form_canvas.configure(yscrollcommand=scroll.set)
        self.cards = tk.Frame(self.form_canvas, bg=PALETTE.charcoal)
        self._cards_window = self.form_canvas.create_window(0, 0, anchor="nw", window=self.cards)
        self.cards.bind("<Configure>", lambda _e: self.form_canvas.configure(
            scrollregion=self.form_canvas.bbox("all")))
        self.form_canvas.bind("<Configure>", lambda e: self.form_canvas.itemconfigure(
            self._cards_window, width=e.width))
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.form_canvas.bind_all(sequence, self._on_wheel, add="+")

    def _on_wheel(self, event: Any) -> None:
        widget = event.widget
        # Only scroll the form when the pointer is over it.
        while widget is not None and widget is not self.form_canvas:
            widget = getattr(widget, "master", None)
        if widget is None:
            return
        delta = -1 if getattr(event, "num", 0) == 4 or getattr(event, "delta", 0) > 0 else 1
        self.form_canvas.yview_scroll(delta * 3, "units")

    def _show_section(self, key: str) -> None:
        self.section = key
        if self._query:
            self.search_var.set("")
            self._query = ""
        self._rebuild_form(scroll_top=True)

    def _rebuild_form(self, *, scroll_top: bool = False) -> None:
        top = 0.0 if scroll_top else self.form_canvas.yview()[0]
        for child in self.cards.winfo_children():
            child.destroy()
        self.widgets.clear()
        if self._query:
            matches = search(self._query)
            params = [p for p in matches if self.model.is_visible(p)]
            advanced = sum(1 for p in matches if p.advanced and not self.model.show_advanced)
            self.form_title.configure(text=f"Settings matching “{self._query}”")
            self.form_lede.configure(text=f"{len(params)} found across every section" + (
                f", {advanced} more among the advanced settings" if advanced else "")
                + ". Esc clears the search.")
            groups = {p.name: g.title for g in GROUPS for p in g.params}
        else:
            group = next(g for g in GROUPS if g.key == self.section)
            params = self.model.visible_params(group.key)
            hidden = sum(1 for p in group.params if p.advanced and not self.model.show_advanced)
            self.form_title.configure(text=group.title)
            self.form_lede.configure(text=group.description + (
                f"  ·  {hidden} advanced setting{'s' if hidden != 1 else ''} hidden" if hidden else ""))
            groups = {}
        if not params:
            self.tk.Label(self.cards, text="Nothing here matches.", font=self.fonts["body"],
                          bg=PALETTE.charcoal, fg=PALETTE.text_faint).pack(anchor="w", padx=28, pady=20)
        for param in params:
            self._card(param, caption=groups.get(param.name, ""))
        self.root.update_idletasks()
        self.form_canvas.configure(scrollregion=self.form_canvas.bbox("all"))
        self.form_canvas.yview_moveto(top)
        self._paint_all_nav()

    def _card(self, param: Any, *, caption: str = "") -> None:
        tk = self.tk
        card = tk.Frame(self.cards, bg=PALETTE.slate, highlightthickness=1,
                        highlightbackground=PALETTE.edge)
        card.pack(fill="x", padx=(28, 22), pady=(0, 10))
        inner = tk.Frame(card, bg=PALETTE.slate)
        inner.pack(fill="x", padx=16, pady=12)

        top = tk.Frame(inner, bg=PALETTE.slate)
        top.pack(fill="x")
        tk.Label(top, text=param.label, font=self.fonts["label"], bg=PALETTE.slate,
                 fg=PALETTE.text).pack(side="left")
        changed = self.model.is_changed(param.name)
        if changed:
            tk.Label(top, text="●", font=self.fonts["tiny"], bg=PALETTE.slate,
                     fg=PALETTE.red_text).pack(side="left", padx=(6, 0))
        if caption:
            tk.Label(top, text=caption, font=self.fonts["tiny"], bg=PALETTE.slate,
                     fg=PALETTE.text_faint).pack(side="left", padx=(10, 0))
        tk.Label(top, text=param.name, font=self.fonts["code_small"], bg=PALETTE.slate,
                 fg=PALETTE.text_faint).pack(side="right")
        if changed:
            reset = tk.Label(top, text="Reset", font=self.fonts["tiny"], bg=PALETTE.slate,
                             fg=PALETTE.red_text, cursor="hand2")
            reset.pack(side="right", padx=(0, 12))
            reset.bind("<Button-1>", lambda _e, n=param.name: self._on_reset_param(n))

        control = tk.Frame(inner, bg=PALETTE.slate)
        control.pack(fill="x", pady=(8, 0))
        self._control(control, param)

        error = tk.Label(inner, text="", font=self.fonts["small"], bg=PALETTE.slate,
                         fg=PALETTE.red_text, anchor="w", justify="left")
        self.widgets[param.name + ":error"] = error
        if param.hint:
            hint = tk.Label(inner, text=param.hint, font=self.fonts["small"], bg=PALETTE.slate,
                            fg=PALETTE.text_faint, anchor="w", justify="left", wraplength=560)
            hint.pack(fill="x", pady=(7, 0))
            card.bind("<Configure>", lambda e, h=hint: h.configure(wraplength=max(200, e.width - 40)))

    def _entry(self, parent: Any, value: Any, *, width: int | None = None, mono: bool = False) -> Any:
        tk = self.tk
        var = tk.StringVar(value="" if value is None else _show(value))
        entry = tk.Entry(parent, textvariable=var, font=self.fonts["code" if mono else "body"],
                         bg=PALETTE.charcoal, fg=PALETTE.text, insertbackground=PALETTE.red_bright,
                         relief="flat", bd=0, highlightthickness=1, highlightbackground=PALETTE.edge,
                         highlightcolor=PALETTE.red, width=width or 20)
        entry.var = var  # type: ignore[attr-defined]
        return entry

    def _control(self, holder: Any, param: Any) -> None:
        tk = self.tk
        from .widgets import DualSlider, LinearSlider, LogSlider, Segmented, ToggleSwitch

        name = param.name
        value = self.model.get(name)
        if param.kind is ParamKind.TOGGLE:
            switch = ToggleSwitch(holder, value=bool(value), width=44, height=24,
                                  command=lambda v, n=name: self._on_change(n, v, rebuild=True))
            switch.configure(bg=PALETTE.slate)
            switch.canvas.configure(bg=PALETTE.slate)
            switch.pack(side="left")
            state = tk.Label(holder, text="On" if value else "Off", font=self.fonts["small"],
                             bg=PALETTE.slate, fg=PALETTE.text_dim)
            state.pack(side="left", padx=(10, 0))
            self.widgets[name] = switch
        elif param.kind is ParamKind.CHOICE and len(param.choices) <= 4:
            seg = Segmented(holder, options=[(c[0], c[0]) for c in param.choices], value=str(value),
                            font=self.fonts["small"],
                            command=lambda v, n=name: self._on_change(n, v, rebuild=True))
            seg.pack(anchor="w")
            note = tk.Label(holder, text=_choice_note(param, value), font=self.fonts["small"],
                            bg=PALETTE.slate, fg=PALETTE.text_dim, anchor="w", justify="left")
            note.pack(anchor="w", pady=(6, 0))
            self.widgets[name] = seg
        elif param.kind is ParamKind.CHOICE:
            var = tk.StringVar(value=str(value))
            combo = self.ttk.Combobox(holder, textvariable=var, state="readonly",
                                      values=[c[0] for c in param.choices], font=self.fonts["body"],
                                      width=28)
            combo.pack(anchor="w")
            note = tk.Label(holder, text=_choice_note(param, value), font=self.fonts["small"],
                            bg=PALETTE.slate, fg=PALETTE.text_dim, anchor="w", justify="left")
            note.pack(anchor="w", pady=(6, 0))
            combo.bind("<<ComboboxSelected>>", lambda _e, n=name, v=var, lab=note, p=param: (
                lab.configure(text=_choice_note(p, v.get())), self._on_change(n, v.get(), rebuild=True)))
            combo.var = var  # type: ignore[attr-defined]
            self.widgets[name] = combo
        elif param.kind in (ParamKind.INT, ParamKind.FLOAT, ParamKind.LOG_FLOAT):
            entry = self._entry(holder, value, width=14, mono=True)
            entry.pack(side="left", ipady=5)
            slider = None
            bounded = param.minimum is not None and param.maximum is not None
            if param.kind is ParamKind.LOG_FLOAT and bounded and param.minimum > 0:
                slider = LogSlider(holder, minimum=param.minimum, maximum=param.maximum,
                                   value=float(value), height=30, show_value=False,
                                   command=lambda v, e=entry: e.var.set(_show(float(f"{v:.3g}"))))
                slider.configure(bg=PALETTE.slate)
                slider.canvas.configure(bg=PALETTE.slate)
            elif bounded and (param.maximum - param.minimum) <= 2000:
                slider = LinearSlider(holder, minimum=param.minimum, maximum=param.maximum,
                                      value=float(value), step=param.step or (1 if param.kind is ParamKind.INT else None),
                                      bg=PALETTE.slate, height=30,
                                      command=lambda v, e=entry, p=param: e.var.set(
                                          _show(int(round(v)) if p.kind is ParamKind.INT else v)))
            if slider is not None:
                slider.pack(side="left", fill="x", expand=True, padx=(14, 0))
            else:
                stepper = tk.Frame(holder, bg=PALETTE.slate)
                stepper.pack(side="left", padx=(8, 0))
                for text, sign in (("−", -1), ("+", 1)):
                    self._button(stepper, text, lambda s=sign, e=entry, p=param: self._step(p, e, s),
                                 small=True).pack(side="left", padx=(0, 4))
                if bounded:
                    tk.Label(holder, text=f"{_show(param.minimum)} – {_show(param.maximum)}",
                             font=self.fonts["tiny"], bg=PALETTE.slate,
                             fg=PALETTE.text_faint).pack(side="left", padx=(8, 0))
            entry.var.trace_add("write", lambda *_a, n=name, e=entry, s=slider: self._on_typed(n, e, s))
            self.widgets[name] = entry
        elif param.kind is ParamKind.RANGE:
            low, high = value if isinstance(value, (list, tuple)) else (0.0, 1.0)
            dual = DualSlider(holder, minimum=param.minimum or 0.0, maximum=param.maximum or 1.0,
                              low=low, high=high, height=36,
                              logarithmic=(param.minimum or 0) > 0 and (param.maximum or 0) > 0,
                              command=lambda lo, hi, n=name: self._on_change(n, (lo, hi)))
            dual.configure(bg=PALETTE.slate)
            dual.canvas.configure(bg=PALETTE.slate)
            dual.pack(fill="x")
            self.widgets[name] = dual
        elif param.kind is ParamKind.MULTILINE:
            text = tk.Text(holder, height=3, font=self.fonts["body"], bg=PALETTE.charcoal,
                           fg=PALETTE.text, insertbackground=PALETTE.red_bright, relief="flat", bd=0,
                           highlightthickness=1, highlightbackground=PALETTE.edge,
                           highlightcolor=PALETTE.red, padx=8, pady=6, wrap="word")
            text.insert("1.0", str(value or ""))
            text.pack(fill="x")
            text.bind("<KeyRelease>", lambda _e, n=name, w=text: self._on_change(n, w.get("1.0", "end-1c")))
            self.widgets[name] = text
        else:
            entry = self._entry(holder, value, mono=param.kind is ParamKind.PATH or name == "model_id")
            entry.pack(side="left", fill="x", expand=True, ipady=5)
            entry.var.trace_add("write", lambda *_a, n=name, e=entry: self._on_typed(n, e, None))
            if param.kind is ParamKind.PATH:
                self._button(holder, "Browse…", lambda n=name, e=entry: self._browse(n, e),
                             small=True).pack(side="left", padx=(8, 0))
            self.widgets[name] = entry

    def _step(self, param: Any, entry: Any, sign: int) -> None:
        try:
            current = float(entry.var.get())
        except ValueError:
            current = float(param.default)
        step = param.step or (1 if param.kind is ParamKind.INT else max(abs(current) * 0.1, 1e-9))
        new = current + sign * step
        if param.minimum is not None:
            new = max(param.minimum, new)
        if param.maximum is not None:
            new = min(param.maximum, new)
        entry.var.set(_show(int(round(new)) if param.kind is ParamKind.INT else round(new, 10)))

    def _browse(self, name: str, entry: Any) -> None:
        from tkinter import filedialog

        if name == "output_dir":
            path = filedialog.askdirectory(title="Output directory")
        else:
            path = filedialog.askopenfilename(
                title="Dataset", filetypes=[("Text data", "*.jsonl *.json *.txt"), ("All", "*")])
        if path:
            entry.var.set(path)

    # -- preview ------------------------------------------------------

    def _build_preview(self, parent: Any) -> None:
        tk = self.tk
        from .widgets import Segmented

        panel = tk.Frame(parent, bg=PALETTE.obsidian, width=_PREVIEW_WIDTH)
        panel.grid(row=0, column=2, sticky="nsew")
        panel.grid_propagate(False)
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=1)

        bar = tk.Frame(panel, bg=PALETTE.obsidian)
        bar.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 10))
        tk.Label(bar, text="Output", font=self.fonts["heading"], bg=PALETTE.obsidian,
                 fg=PALETTE.text).pack(side="left")
        self.lines_label = tk.Label(bar, text="", font=self.fonts["tiny"], bg=PALETTE.obsidian,
                                    fg=PALETTE.text_faint)
        self.lines_label.pack(side="left", padx=(10, 0), pady=(5, 0))
        self.format_var = tk.StringVar(value=self.model.config_format)
        self.format_combo = self.ttk.Combobox(bar, textvariable=self.format_var, state="readonly",
                                              values=list(CONFIG_FORMATS), width=8,
                                              font=self.fonts["small"])
        self.format_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_format())
        self.mode_seg = Segmented(bar, options=[("generate", "Script"), ("inject", "Inject"),
                                                ("config", "Config")],
                                  value=self.model.mode, command=self._on_mode, font=self.fonts["small"])
        self.mode_seg.pack(side="right")
        self.inject_button = self._button(bar, "Choose script…", self._on_choose_target, small=True)

        code = tk.Frame(panel, bg=PALETTE.edge)
        code.grid(row=1, column=0, sticky="nsew", padx=16)
        code.grid_rowconfigure(0, weight=1)
        code.grid_columnconfigure(1, weight=1)
        self.gutter = tk.Text(code, width=4, font=self.fonts["code"], bg=PALETTE.obsidian,
                              fg=PALETTE.edge, bd=0, highlightthickness=0, padx=8, pady=12,
                              takefocus=0, cursor="arrow", spacing1=1, state="disabled")
        self.gutter.grid(row=0, column=0, sticky="ns", padx=(1, 0), pady=1)
        self.preview = tk.Text(code, font=self.fonts["code"], bg=PALETTE.obsidian, fg=PALETTE.text,
                               insertbackground=PALETTE.red_bright, relief="flat", bd=0,
                               highlightthickness=0, wrap="none", padx=12, pady=12,
                               selectbackground=PALETTE.red_dim, selectforeground=PALETTE.text,
                               spacing1=1)
        self.preview.grid(row=0, column=1, sticky="nsew", pady=1)
        yscroll = self.ttk.Scrollbar(code, orient="vertical", style="Dark.Vertical.TScrollbar",
                                     command=self._scroll_both)
        yscroll.grid(row=0, column=2, sticky="ns", padx=(0, 1), pady=1)
        xscroll = self.ttk.Scrollbar(code, orient="horizontal", style="Dark.Horizontal.TScrollbar",
                                     command=self.preview.xview)
        xscroll.grid(row=1, column=1, sticky="ew")
        self.preview.configure(xscrollcommand=xscroll.set,
                               yscrollcommand=lambda a, b: (yscroll.set(a, b), self.gutter.yview_moveto(a)))
        colours = {"keyword": PALETTE.syn_keyword, "string": PALETTE.syn_string,
                   "number": PALETTE.syn_number, "comment": PALETTE.text_faint,
                   "name": PALETTE.syn_name, "builtin": PALETTE.syn_builtin,
                   "marker": PALETTE.red_text}
        for tag, colour in colours.items():
            self.preview.tag_configure(tag, foreground=colour)
        self.preview.tag_configure("marker", background=PALETTE.charcoal)

        checks = tk.Frame(panel, bg=PALETTE.obsidian)
        checks.grid(row=2, column=0, sticky="ew", padx=16, pady=(12, 14))
        tk.Label(checks, text="CHECKS", font=self.fonts["tiny"], bg=PALETTE.obsidian,
                 fg=PALETTE.text_faint).pack(anchor="w")
        self.checks = tk.Frame(checks, bg=PALETTE.obsidian)
        self.checks.pack(fill="x", pady=(4, 0))

    def _scroll_both(self, *args: Any) -> None:
        self.preview.yview(*args)
        self.gutter.yview(*args)

    def _build_status(self) -> None:
        tk = self.tk
        bar = tk.Frame(self.root, bg=PALETTE.obsidian, height=30, highlightthickness=0)
        bar.grid(row=2, column=0, sticky="ew")
        tk.Frame(bar, bg=PALETTE.edge, height=1).pack(fill="x", side="top")
        self.status_dot = tk.Label(bar, text="●", font=self.fonts["tiny"], bg=PALETTE.obsidian,
                                   fg=PALETTE.ok)
        self.status_dot.pack(side="left", padx=(18, 6), pady=5)
        self.status_label = tk.Label(bar, text="", font=self.fonts["small"], anchor="w",
                                     bg=PALETTE.obsidian, fg=PALETTE.text_dim)
        self.status_label.pack(side="left", fill="x", expand=True)
        self.summary_label = tk.Label(bar, text="", font=self.fonts["small"], bg=PALETTE.obsidian,
                                      fg=PALETTE.text_faint)
        self.summary_label.pack(side="right", padx=(0, 18))

    # -- events -------------------------------------------------------

    def _bind_keys(self) -> None:
        root = self.root
        root.bind("<Control-s>", lambda _e: self._on_save())
        root.bind("<Control-o>", lambda _e: self._on_load())
        root.bind("<Control-f>", lambda _e: (self.search_entry.focus_set(), "break")[1])
        root.bind("<Control-C>", lambda _e: self._on_copy())
        root.bind("<Escape>", lambda _e: self.search_var.set("") if self._query else None)
        for index, group in enumerate(GROUPS, start=1):
            root.bind(f"<Control-Key-{index}>", lambda _e, k=group.key: self._show_section(k))

    def _placeholder(self, entry: Any, var: Any, text: str) -> None:
        def show(_e: Any = None) -> None:
            if not var.get():
                entry.configure(fg=PALETTE.text_faint)
                entry.insert(0, text)
                entry.placeholder = True

        def hide(_e: Any = None) -> None:
            if getattr(entry, "placeholder", False):
                entry.placeholder = False
                entry.delete(0, "end")
                entry.configure(fg=PALETTE.text)

        entry.placeholder = False
        entry.bind("<FocusIn>", hide)
        entry.bind("<FocusOut>", show)
        show()

    def _on_search(self) -> None:
        if getattr(self.search_entry, "placeholder", False):
            return
        query = self.search_var.get().strip()
        if query == self._query:
            return
        self._query = query
        self._rebuild_form(scroll_top=True)

    def _on_typed(self, name: str, entry: Any, slider: Any) -> None:
        text = entry.var.get()
        ok, message = self.model.set(name, text)
        entry.configure(highlightbackground=PALETTE.edge if ok else PALETTE.red,
                        highlightcolor=PALETTE.red)
        error = self.widgets.get(name + ":error")
        if error is not None:
            if ok:
                error.pack_forget()
            else:
                error.configure(text=message)
                error.pack(fill="x", pady=(6, 0))
        if ok and slider is not None:
            try:
                slider.set(float(text))
            except (TypeError, ValueError):
                pass
        self._after_change(message if not ok else "", level="error" if not ok else "ok")

    def _on_change(self, name: str, value: Any, *, rebuild: bool = False) -> None:
        ok, message = self.model.set(name, value)
        if rebuild:
            # Toggles and choices can show or hide others (LoRA's settings
            # under "Use LoRA"); and the changed dot and Reset follow.
            self.root.after_idle(self._rebuild_form)
        self._after_change(message, level="error" if not ok else ("warn" if message else "ok"))

    def _after_change(self, message: str, *, level: str) -> None:
        if message:
            self._set_status(message, level=level)
        self._schedule_preview()
        self._paint_all_nav()

    def _on_reset_param(self, name: str) -> None:
        self.model.reset_param(name)
        self._rebuild_form()
        self._schedule_preview()

    def _on_advanced(self, value: bool) -> None:
        self.model.show_advanced = value
        self._rebuild_form()
        self._set_status("Advanced settings shown" if value else "Advanced settings hidden")

    def _on_mode(self, mode: str) -> None:
        self.model.mode = mode
        self.format_combo.pack_forget()
        self.inject_button.pack_forget()
        if mode == "config":
            self.format_combo.pack(side="right", padx=(0, 10))
        elif mode == "inject":
            self.inject_button.pack(side="right", padx=(0, 10))
        self._refresh_preview()

    def _on_format(self) -> None:
        self.model.config_format = self.format_var.get()
        self._refresh_preview()

    def _on_choose_target(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(title="Script to inject into",
                                          filetypes=[("Python", "*.py"), ("All", "*")])
        if path:
            self.model.target_path = Path(path)
            self._refresh_preview()
            self._set_status(f"Injecting into {Path(path).name}")

    def _on_recipe(self) -> None:
        from .recipes import RECIPES

        title = self.recipe_var.get()
        recipe = next((r for r in RECIPES if r.title == title), None)
        if recipe is None:
            self.model.reset()
        else:
            self.model.apply_recipe(recipe.key)
        self._describe_recipe()
        self._rebuild_form()
        self._refresh_preview()
        self._set_status(f"Started from {title}" if recipe else "Reset to defaults")

    def _on_save(self) -> None:
        from tkinter import filedialog, messagebox

        errors, _ = self.model.validate()
        if errors:
            messagebox.showerror("Cannot save yet", "\n".join(errors))
            return
        if self.model.mode == "inject" and self.model.target_path:
            path = str(self.model.target_path)
        else:
            default = "train.py" if self.model.mode != "config" else f"config.{_extension(self.model.config_format)}"
            path = filedialog.asksaveasfilename(initialfile=default)
        if not path:
            return
        try:
            saved = self.model.save(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not save", str(exc))
            return
        self._set_status(f"Saved {saved}" + (
            f"  ·  run it with: python {saved.name}" if self.model.mode == "generate" else ""))
        self._refresh_title()

    def _on_copy(self) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(self.model.preview())
        self._set_status("Output copied to the clipboard")

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
        self.recipe_var.set(self._recipe_title(self.model.recipe))
        self._describe_recipe()
        self.mode_seg.set(self.model.mode)
        self._on_mode(self.model.mode)
        self._rebuild_form()
        self._set_status(
            f"Loaded {Path(path).name}"
            + (f" ({len(rejected)} value(s) skipped)" if rejected else "")
        )

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

    def _schedule_preview(self) -> None:
        # Typing a path re-renders 900 lines on every keystroke otherwise.
        if self._pending is not None:
            self.root.after_cancel(self._pending)
        self._pending = self.root.after(120, self._refresh_preview)

    def _refresh_preview(self) -> None:
        from .highlight import TAGS, spans

        self._pending = None
        text = self.model.preview()
        top = self.preview.yview()[0]
        self.preview.configure(state="normal")
        self.preview.delete("1.0", "end")
        self.preview.insert("1.0", text)
        for tag in TAGS:
            self.preview.tag_remove(tag, "1.0", "end")
        fmt = "python" if self.model.mode != "config" or self.model.config_format == "python" else "config"
        for tag, (l1, c1), (l2, c2) in spans(text, fmt):
            self.preview.tag_add(tag, f"{l1}.{c1}", f"{l2}.{c2}")
        self.preview.configure(state="disabled")
        count = len(text.splitlines())
        self.gutter.configure(state="normal", width=max(3, len(str(count))))
        self.gutter.delete("1.0", "end")
        self.gutter.insert("1.0", "\n".join(str(i) for i in range(1, count + 1)))
        self.gutter.tag_configure("right", justify="right")
        self.gutter.tag_add("right", "1.0", "end")
        self.gutter.configure(state="disabled")
        self.preview.yview_moveto(top)
        self.gutter.yview_moveto(top)
        self.lines_label.configure(text=f"{count} lines")
        self._refresh_checks()
        self._refresh_title()

    def _refresh_checks(self) -> None:
        tk = self.tk
        for child in self.checks.winfo_children():
            child.destroy()
        errors, warnings = self.model.issues()
        summary = self.model.summary()
        if not errors and not warnings:
            tk.Label(self.checks, text="✓  Ready to save", font=self.fonts["body_bold"],
                     bg=PALETTE.obsidian, fg=PALETTE.ok, anchor="w").pack(anchor="w")
        for (owner, message), colour, mark in (
            *((e, PALETTE.red_text, "●") for e in errors[:4]),
            *((w, PALETTE.warn, "▲") for w in warnings[:max(0, 4 - len(errors))]),
        ):
            row = tk.Label(self.checks, text=f"{mark}  {message}", font=self.fonts["small"],
                           bg=PALETTE.obsidian, fg=colour, anchor="w", justify="left",
                           wraplength=_PREVIEW_WIDTH - 50, cursor="hand2" if owner else "")
            row.pack(anchor="w", fill="x", pady=1)
            if owner:
                row.bind("<Button-1>", lambda _e, n=owner: self._go_to(n))
        hidden = len(errors) + len(warnings) - 4
        if hidden > 0:
            tk.Label(self.checks, text=f"…and {hidden} more", font=self.fonts["tiny"],
                     bg=PALETTE.obsidian, fg=PALETTE.text_faint).pack(anchor="w")
        lora = "  ·  LoRA" if summary["lora"] else ""
        self.summary_label.configure(
            text=f"effective batch {summary['effective_batch']}  ·  "
                 f"{summary['tokens_per_step']:,} tokens/step  ·  {summary['optimizer']}  ·  "
                 f"{summary['device']}/{summary['precision']}{lora}  ·  "
                 f"{summary['changed']} changed")
        if errors:
            self._set_status(f"{len(errors)} problem(s) to fix before saving", level="error")
        elif not self.status_label.cget("text"):
            self._set_status("Ready")

    def _go_to(self, name: str) -> None:
        group = next(g for g in GROUPS if any(p.name == name for p in g.params))
        param = ALL_PARAMS[name]
        if param.advanced and not self.model.show_advanced:
            self.model.show_advanced = True
            self.advanced_toggle.set(True)
        self._show_section(group.key)
        widget = self.widgets.get(name)
        if widget is not None:
            widget.focus_set()  # every control here is a Tk widget

    def _refresh_title(self) -> None:
        mark = "● " if self.model.dirty else ""
        self.root.title(f"{mark}scriptgen · HyperNix training script builder")

    def _set_status(self, message: str, *, level: str = "ok") -> None:
        colour = {
            "ok": PALETTE.ok, "warn": PALETTE.warn, "error": PALETTE.error, "hint": PALETTE.info,
        }.get(level, PALETTE.ok)
        self.status_dot.configure(fg=colour)
        self.status_label.configure(text=message,
                                    fg=PALETTE.red_text if level == "error" else PALETTE.text_dim)

    def run(self) -> None:
        self.root.mainloop()


def _rounded(canvas: Any, x1: float, y1: float, x2: float, y2: float, r: float, **kw: Any) -> Any:
    """A rounded rectangle; Tk has no such primitive."""
    points = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
              x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return canvas.create_polygon(points, smooth=True, **kw)


def _show(value: Any) -> str:
    """A value as an entry shows it: 3e-05 not 3.0000000000000004e-05, 2 not 2.0."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() and abs(value) < 1e15 else f"{value:.6g}"
    return str(value)


def _choice_note(param: Any, value: Any) -> str:
    return next((c[1] for c in param.choices if c[0] == str(value)), "")


def _extension(fmt: str) -> str:
    return {"python": "py", "json": "json", "yaml": "yaml", "toml": "toml", "env": "env",
            "args": "sh"}.get(fmt, "txt")


def launch(values: dict[str, Any] | None = None, *, recipe: str = "") -> int:
    """Open the GUI, or explain why it cannot open."""
    model = FormModel(values)
    if recipe:
        model.recipe = recipe
    try:
        app = ScriptGenApp(model)
    except RuntimeError as exc:
        print(f"scriptgen: {exc}")
        return 1
    app.run()
    return 0

