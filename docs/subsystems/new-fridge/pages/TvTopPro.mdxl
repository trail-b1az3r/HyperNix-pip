# tvtoppro — tvtop++'s numbers, btop++'s presentation

> **tvtop-max** (0.72.6.rc2) is this on OpenTUI, with panels for the run
> itself: its script's modules and libraries, model architecture,
> Pressure Cooker, logs and warnings, and a phone layout. See
> [TvTopMax](TvTopMax.md). tvtoppro stays, and needs no Bun.

```bash
tvtoppro
tvtoppro --theme gruvbox-dark
tvtoppro --theme ~/.config/btop/themes/nord.theme
tvtoppro --list-themes
tvtoppro --once --width 100        # one frame, for a script or a screenshot
```

## Why not cctvtop

`cctvtop` wraps a compiled C++ dashboard and shells out to it. When the
extension is not built for the running Python, it falls back to importing
`tvtop_plus_plus` and rendering that instead — so "run cctvtop" means one
of two different programs depending on how the wheel was built, and the
fallback is the one most people see.

`tvtoppro` takes the other half of that split deliberately.
`TVTopPlusPlus` is used purely as a **stat source**, held as an attribute
rather than inherited from, so the two can diverge without either
dragging the other. Its `latest_frame` already collects per-core CPU, the
`/proc/meminfo` breakdown, the whole `nvidia-smi` row and the
training-log tail. Everything drawn is new.

Anything with a `latest_frame()` and a `_get_active_processes()` can be
substituted, which is what makes the drawing testable at all.

## What "presented exactly like btop++" means

Four specific things. Skip any one and the result is merely a boxed TUI.

### 1. Boxes titled in the border

```
┌─┤cpu├──────────────────────────────────────────────────┤1├─┐
```

The title sits *in* the rule with a bracket either side, and the box
number goes in the opposite corner. A heading inside the box is the
ordinary way to do this and it is the wrong look.

### 2. Braille graphs

Two samples per character cell across, four levels per cell down. A
90×3 graph plots **180 points over 12 levels** where block characters
would give 90 over 3.

```
⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣴⣶⣶⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣴⣶⣶⣤⣄⡀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣶⣄⡀⠀⠀⠀⠀⣀⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣶⣄
⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣤⣄⣀⣤⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
```

The dot map's fourth row is `0x40`/`0x80` rather than a continuation of
the pattern. Getting that wrong leaves a gap along the bottom of every
graph — which reads as a font quirk rather than a bug, and so never gets
fixed. There is a test for it.

Each row is coloured by its own height, not by the value: the top of a
busy graph is red because it is the top.

### 3. Gradient meters

```
CPU ████████████████████████████──────────────────────  56.3%
```

Each *cell* takes its colour from its own position along the
start/mid/end ramp — so a full bar shows the whole ramp and a quarter bar
shows only its cool end. A bar that is one colour chosen by the value is
a progress bar, and it is the single thing that most makes a btop-alike
look like something else.

The ramp itself is two linear segments rather than one, because btop's
are not monotonic in any channel: the cyan→yellow→red CPU ramp passes
*through* yellow, and interpolating cyan straight to red gives a muddy
purple that looks nothing like it.

### 4. Themes as data

btop's own `.theme` files — `theme[key]="#hex"` lines — load directly, so
anything already in `~/.config/btop/themes` works without conversion.

## Themes

Seven built in: `hypernix`, `gruvbox-dark`, `nord`, `dracula`,
`tokyo-night`, `monokai`, and `mono` for a terminal without truecolor or
a screenshot that has to survive being printed.

```bash
tvtoppro --theme nord --dump-theme > mine.theme
$EDITOR mine.theme
tvtoppro --theme ./mine.theme
```

`--dump-theme` writes btop's format and it round-trips, so it is the
quickest way to start editing one. Every theme file is also in
[`examples/tvtoppro/`](../examples/tvtoppro/).

A **partial theme is a valid theme**: unset keys inherit from the
default, so a three-line file changes three colours. btop behaves the
same way, and refusing a partial file makes hand-editing one a guessing
game about which keys are mandatory.

Colours are `#RRGGBB` or btop's two-digit greyscale shorthand `#XX`.
`#40` is a mid-dark grey, not `#400000` — read the wrong way it turns
every neutral in a real theme dark red.

| Ramp | Drives |
|---|---|
| `cpu_*` | CPU meter and graph, GPU meter |
| `used_*` | RAM used, VRAM used |
| `available_*` | RAM available |
| `free_*` | cache, training progress, loss curve |
| `temp_*` | GPU temperature |

## What it draws

Five panels: **cpu** (meter, braille history, per-core grid in as many
columns as fit), **mem** (used, available, cached), **gpu** (utilisation,
VRAM, temperature and power, history), **training** (step progress, loss,
learning rate, throughput, ETA, and a loss curve), and **proc**.

A machine with no `nvidia-smi` gets a sentence saying so rather than a
row of zeroes, because zeroes read as an idle GPU and that is a different
fact from not having one. Same for a missing training log.

## Alignment, which is the whole thing

Every row of a frame is exactly the requested width. It is the property
that separates a btop-alike from a mess, it breaks the moment a label
grows a character, and it **cannot be checked by counting string
length** — the rows carry colour tags that print as nothing and braille
that prints as one cell each, so `len()` is wrong in both directions.

Widths are measured with Rich, and the tests assert every row at four
widths under all seven themes. That caught the missing half of the
padding helper: at 60 columns the "no nvidia-smi here" line is longer
than the box, and an over-long row does not wrap tidily — it pushes the
right border onto the next line and every box below it looks broken. Rows
are truncated as well as padded, by Rich, because a naive slice of markup
can cut a colour tag in two.

## Flags

| Flag | |
|---|---|
| `--log PATH` | training log to tail (autodetected otherwise) |
| `--theme NAME\|PATH` | built-in name, btop `.theme`, or JSON |
| `--list-themes` | names and their CPU ramps |
| `--dump-theme` | print the active theme in btop's format |
| `--refresh N` | seconds between frames |
| `--width N` | fix the width instead of using the terminal's |
| `--no-processes` | drop the proc panel |
| `--once` | draw one frame and exit |
| `--once --json` | emit the frame's numbers instead of drawing |

## See also

- [Dashboards](Dashboards.md) — `tvtop`, `cctvtop` and `tvtop++`, and
  how the three of them relate
