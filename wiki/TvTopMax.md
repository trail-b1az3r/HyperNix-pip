# tvtop-max — the machine, and the run on it

```bash
tvtop-max                          # find the busiest Python run, its script and its log
tvtop-max -s                       # phone layout: one column, 40-56 columns wide
tvtop-max -l train.log -S train.py # name the log and script yourself
tvtop-max -P 4242                  # watch that process
tvtop-max --hide procs,modules     # start with panels hidden
```

[tvtop-pro](TvTopPro.md) shows how the machine is doing. tvtop-max shows
that too, and also what the run on the machine *is*: which script, which
HyperNix modules and libraries it pulls in, which model it builds, which
Pressure Cooker it steps with, what its log says, and what is wrong with
it. It is an [OpenTUI](https://opentui.com/docs/) app on Bun, like
`hyped-pro`, in the HyperNix site's colours. tvtop-pro stays installed
and needs no Bun.

## The panels

Each panel's number toggles it, the way btop's do.

| Key | Panel | Shows |
|---|---|---|
| 1 | cpu | Total and per-core meters, and a history graph |
| 2 | mem | RAM, swap, cached and free, and a history graph |
| 3 | gpu | Utilisation, VRAM, power and temperature, from `nvidia-smi` |
| 4 | training | Step out of total, loss, learning rate, speed, ETA and a loss graph. It says so when the log has not moved for ten minutes. |
| 5 | pressure cooker | Every Pressure Cooker the script uses, called or passed as `optimizer_class=`: its generation, the arguments it was given, whether it is deprecated, and the learning rate the log reports now |
| 6 | model | The architecture: from a `new_oven(arch=…)` call, a `preheat("repo")` snapshot, or a `config.json` beside the run. Shows the estimated parameter count, the sizes the script set and the preset's defaults. |
| 7 | logs | The log's last lines, errors in red and warnings in amber. ↑↓ scroll back, End follows again. |
| 8 | warnings | What is wrong, from the script and from the log, errors first (see below) |
| 9 | modules & libraries | The HyperNix modules the script imports, each with its one-line summary and any deprecation; the libraries with their installed versions; anything not installed; local and standard-library imports |
| 0 | processes | The busiest Python processes. The one being watched is marked. |

## Warnings

From the script, read with `ast` and never run:

- a module that is not installed here;
- a deprecated HyperNix module, with what to use instead;
- a deprecated Pressure Cooker generation (V1, V3);
- `torch.load` without `weights_only=True`, which can run code hidden in
  the file;
- no seed set, so two runs will not match;
- a training script that saves no checkpoint, so a crash loses the run;
- a script that does not parse.

From the log's last 2,000 lines, each message counted once however often
it repeats:

- tracebacks, exceptions, out-of-memory errors, a NaN or infinite loss,
  and a process killed by the OOM killer;
- `…Warning:` lines and `WARNING` lines.

## Finding the run

With no options, tvtop-max watches the busiest Python process that is
not itself. It reads the script from that process's command line: the
first argument that is not an option. For `torchrun`, `accelerate
launch`, `deepspeed` and `python -m torch.distributed.run`, it is the
first `.py` file after the launcher's own options. The log is found the
way tvtop-pro finds one, in the run's working directory, or else among
the files the process has open.

`-S` names the script. Then tvtop-max watches only a process that is
running that script. If none is, it watches no process, rather than
guessing. The header always says which script, log and process are in
use.

The script is only ever read: tvtop-max never imports or runs it, so
watching a run can never start a second one. Reading the model presets
imports torch once, in the background, and the panels fill in when it
finishes. On a machine with no torch the model panel still shows what
the script passed.

## Phone layout (`-s`)

`-s` stacks the panels in one scrolling column, sized for a phone's SSH
app (Termius, Blink, a-Shell: 40-56 columns in portrait), with the
training and warnings panels first. It is also chosen automatically on a
terminal narrower than 72 columns, and `s` switches layout at any time.
On a desktop, `-s` stays 56 columns wide, so it previews what the phone
will show. j/k, the arrows, Page Up/Down and Space scroll it.

## Keys

| Key | Does |
|---|---|
| `0`-`9` | Toggle a panel |
| `s` | Switch between the wide and phone layouts |
| `p` | Pause sampling |
| `r` | Look for the run, script and log again |
| ↑ ↓ / `j` `k` | Scroll the log (wide) or the column (phone) |
| End / `f` | Follow the log again |
| `q` / Esc / Ctrl-C | Quit |

## How it is built

- `hypernix.monitoring.tvtop_max` is the launcher. It finds Bun, installs
  `@opentui/core` on first run the way `hyped-pro` does (into
  `~/.hypernix/tvtop-max/<version>` when the install is read-only), and
  starts the app.
- `hypernix.monitoring.tvtop_max_bridge` is the Python side. It speaks
  hyped-pro's JSON-lines protocol and answers `info`, `frame` and
  `rescan`. Its numbers come from the same source as tvtop-pro
  (`TVTopPlusPlus`), so the two dashboards agree.
- `hypernix.monitoring.run_inspect` reads the script and the log.
- `src/hypernix/monitoring/tvtop_max_app/` holds the TypeScript. Panels
  and layout are pure functions with their own `bun test` suite, and CI
  runs it with the type check.

`TVTOP_MAX_DEBUG=1` prints what the launcher runs. `HYPED_PRO_BUN` points
it at a particular `bun`, as it does for hyped-pro.
