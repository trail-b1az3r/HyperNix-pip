# `fusebox` — the breaker panel for a training run

`hnx fusebox` holds a GPU at a temperature you chose, and stops the run
if it goes past a hard limit anyway.

```bash
hnx fusebox status                    # what the cards are doing now
hnx fusebox watch --target 78         # hold 78 °C by pacing the run
hnx fusebox watch --target 78 --underclock --yes
hnx fusebox plan                      # what underclocking would do
hnx fusebox restore                   # undo what a crashed run left
```

## Read this part first

This is **not a speedup**, and the module says so in its own docstring.

The pitch a module like this usually makes is "ease off before the
driver throttles and win back more than you gave up". That was the
design brief. It does not survive being simulated, and the simulation is
in the test suite (`tests/test_fusebox.py::TestTheThroughputClaim`) so
you can check rather than take our word:

| strategy | throughput | temperature |
| --- | --- | --- |
| run flat out, take the driver's throttling | **fastest** | hottest |
| hold a target with a lower power limit | −2 % to −7 % | much cooler |
| hold a target by pausing between steps | −12 % to −24 % | much cooler |

The reason is not subtle once you see it. A driver's thermal throttle
still does *most* of the work — clocks at 55 % are 55 % of a card, not
zero — while a pause does none. And performance scales sublinearly with
power, roughly `power ** 0.35` over the useful range, so 80 % of the
power buys about 92 % of the throughput while pausing 20 % of the time
buys 80 %. Nothing you do above the throttle point recovers more than
the throttle costs.

The ordering holds across the range a real machine occupies — marginal
cooling and a savage throttle included, which are the conditions under
which the story would be true if it were ever true.

## So what is it for

**A temperature you chose, at a cost you can see.** "Hold this card at
78 °C" is a legitimate thing to want: a shared machine, a laptop on a
desk, a room someone sleeps in, a card you would like to still own in
three years, an electricity bill. This delivers exactly that, and prints
what it cost in seconds and percent when the run ends:

```
fusebox eased for 412s, peak 79°C. That cost 11.4% of the run.
```

**A fuse.** A card at 95 °C with a failing fan should stop, and "the
driver will handle it" is not a plan when the driver's next move is a
shutdown in the middle of a checkpoint write. The breaker pauses the
run, waits for the card to cool to `--reset`, and resumes — eased, since
whatever got it there is still true. If it is still hot an hour later it
raises `ThermalStall` and says to check the cooling, rather than
blocking forever without saying why.

## The two levers

| | pausing (`ease`) | power limit (`--underclock`) |
| --- | --- | --- |
| Cost for the same temperature | −12 % to −24 % | −2 % to −7 % |
| Privileges | none | root |
| Outlives the process | no | **yes** |
| On by default | yes | no |

Because the power limit is cheaper by three to four times, it is the one
fusebox reaches for first *when you allow it* — and once a limit is
actually applied, the pause stands down to 40 % of its ceiling, because
both levers at full pays twice for the same degrees.

Pausing is the fallback that always works. It needs no permissions, it
changes nothing outside the process, and it is instantly reversible.

## What it will not do

**It will not raise a power limit above the card's factory default.**
Not with a flag, not on request, not through the restore path. Lowering
a limit and putting it back is thermal management; going past the
default is overclocking, and a training run that quietly overvolts
someone's card is not a feature. The check lives at the single function
that could break it, and is asserted from four directions in the tests.

**It will not touch a card unless you ask twice.** `--underclock` turns
the feature on and `--yes` confirms it. Without both, every subcommand
is read-only and reports what it *would* set. A power limit outlives the
process that changed it, so a flag in your shell history should not be
enough to change one.

**It will not sudo.** Setting a power limit needs root on every current
driver. If the vendor tool refuses, that is reported once and the
governor falls back to pausing, which needs none. A tool that escalates
on its own is a tool nobody can audit.

**It will not run anything it was handed.** Vendor commands are built
from a fixed table in the module; the card index goes through `int()`
and the wattage through `float()`, every invocation is a list, and there
is no shell anywhere.

**It will not intervene when there is nothing to do.** On a card that
never approaches the target, this is one cheap sensor read per interval
and nothing else, and the summary says so:

```
fusebox did not intervene: peaked at 63°C, target 80°C. It cost the run nothing.
```

## Putting things back

Underclocking is recorded to `~/.hypernix/fusebox-state.json` **as each
change is made**, not at exit — because the case it exists for is the
process not reaching its exit. A clean exit restores; an exception
restores; a `SIGKILL` leaves the file, and:

```bash
hnx fusebox status     # says what is still applied, exits 3
hnx fusebox restore    # reports what it would do
hnx fusebox restore --yes
```

## From a training run

```bash
hnx train run --model-dir ./m --dataset ./d.txt --out-dir ./out \
    --steps 10000 --thermal-target 78
```

Or from the environment, which is what `hypernix-t1 launch-script` sets:

```bash
HNX_THERMAL_TARGET=78 hnx train run ...
```

Either way the run is *paced* and no card setting is touched — the
underclocking half needs a person to have typed `--underclock --yes` at
`hnx fusebox`, and is not reachable from the trainer.

A target at or above the default trip point moves the fuse up with it
(`+10 °C`), since a breaker that is open from the first reading is not
what anyone meant.

## From Python

```python
from hypernix.system import fusebox

box = fusebox.FuseBox(fusebox.Policy(target_c=78))
with box.session():
    for step in range(steps):
        train_one_step()
        box.pace()          # returns the seconds it paused, if any

print(box.summary())
print(box.governor.report())   # steps, eased_seconds, trips, peak_c, cost_pct
```

`box.pace()` reads the sensors no more often than `poll_seconds`
(default 2 s) however often it is called, so a model whose steps take
20 ms does not shell out to `nvidia-smi` fifty times a second.

## Which cards, and which tools

Everything goes through [`hypernix.system.gpus`](Devices.md), so NVIDIA
and AMD are read the same way — `nvidia-smi`, `amd-smi`, `rocm-smi`,
whichever answers. CPU temperature comes from
`hypernix.monitoring.thermometer` (psutil, then sysfs).

The CPU is **reported** and can trip the breaker with `--cpu-trip`, but
it never drives the pacing. A CPU at 85 °C during data loading is
normal, and a governor that eased a cold GPU over it would be doing harm
on no evidence.

There is no CPU-side actuator: changing a machine's frequency scaling
because a training run is warm would slow down everything else the
person is doing, and they did not ask for that. The one exception is
`limit_cpu_threads`, which halves *this process's* torch thread count
while easing — job-scoped, reversible, and needs no privileges.

## Exit codes

| Exit | Meaning |
| --- | --- |
| `0` | Fine. |
| `1` | Could not start — bad flags, no sensors. |
| `2` | A card is at or above the trip temperature right now. |
| `3` | Power limits from an earlier run are still applied. |

## See also

- [Devices](Devices.md) — the GPU abstraction underneath.
- [Alarms](Alarms.md) — the smoke alarm, for a run that has gone wrong
  in ways temperature will not show.
- [CLI](CLI.md) — every subcommand.
