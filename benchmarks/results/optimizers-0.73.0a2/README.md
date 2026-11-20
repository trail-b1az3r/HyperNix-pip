# Optimizer benchmark results, 0.73.0a2

The raw records behind the measured tables on the wiki's
[Pressure Cooker V7](../../../wiki/Pressure-Cooker-V7.md) page. Each folder is
one run of `hnx optimizer benchmark --out-dir`, with its JSON (every per-step
time and every per-step loss), CSV and Markdown report; the `.log` beside it
is what the run printed. The page's tables are rendered from these files by
`python scripts/render_optimizer_results.py`.

One machine: Linux, 4 CPU threads, no GPU, Python 3.12, torch 2.15.0 (CPU
build), fp32, seed 0. These numbers are this machine's. Compare runs from one
machine only, and run the benchmark on your own hardware before choosing an
optimizer on speed.

| Folder | Command (after `hnx optimizer benchmark`) | What it is |
|---|---|---|
| `band/` | `--optimizers v7 --workloads mlp --lrs 1e-2 --steps 8 --optimizer-steps 2 --band` | `step()` alone at 30M, 39M, 39.1M and 150M parameters: below, at and inside V7Mini's band. The tiny `mlp` run is only there because a run needs one |
| `in-band/` | `--optimizers adamw,v6,v7,v7mini,v7q --workloads band-transformer --lrs 1e-3,3e-3,1e-2 --steps 30 --optimizer-steps 6 --warmup 2` | Training a 41.6M-parameter transformer, inside the band. Thirty steps: a cost and early-trend measurement, not convergence |
| `main/` | `--workloads all --steps 200` | All eleven optimizers on six workloads, each swept over 3e-4, 1e-3, 3e-3, 1e-2 and 3e-2 |
| `ablations/` | `--optimizers v7,v7s,v7m,v5,v5s --ablations --workloads small-transformer,mlp,mixed-1d-2d --steps 200` | V7, V7S and V7M with one mechanism changed at a time |
| `high-lr/` | `--optimizers v5,v5plus,v5s,v6,v7q --lrs 0.03,0.1,0.3,1.0 --workloads all --steps 200` | The optimizers whose best rate in `main/` was the grid's largest, at higher rates |
| `band-edge/` | `--optimizers v7 --workloads mlp --lrs 1e-2 --steps 8 --optimizer-steps 2 --band 749000000,750000000` | `step()` alone just inside and exactly at the band's upper edge |

`band-edge/` was run last, after two fixes its first attempt exposed: the
band run now frees each optimizer before building the next (the first
attempt skipped every optimizer after AdamW at 749M as not fitting), and on
a CPU `state_dtype="auto"` now switches to int8 only when fp32 momentum
would take more than half the free memory (at a quarter it had picked int8
for V7 Balanced at 749M). Neither change affects the other folders: their
models are small enough that every V7 ran fp32 either way.

Not measured anywhere here: anything on a GPU (V6V is recorded as
unavailable, and V7V ran its CPU fallback), CPU and GPU utilisation, kernel
launches and host-device synchronisation counts.
