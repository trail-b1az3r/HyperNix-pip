# Optimizer benchmark

- When: 2026-11-20T08:21:39Z; HyperNix 0.73.0a2, torch 2.15.0+cpu
- Machine: Linux 6.18.44-fc-v726, x86_64, 4 threads, device cpu
- Each optimizer: 30 training steps at each of 0.001, 0.003, 0.01; 6 optimizer-only steps after 2 warm-up; seed 0; fp32
- Not measured: cpu utilization, gpu utilization, kernel launches, host device synchronization count

## band-transformer (41,636,864 parameters)

| Optimizer | Step only, ms | State B/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share | lr sensitivity |
|---|---|---|---|---|---|---|---|---|
| adamw | 91.777 | 8.00 | 0.001 | 7.4722 | 13.6778 | 251.038 | 36% | 1.723 |
| v6 | 79.975 | 4.00 | 0.01 | 9.8313 | 9.8316 | 223.548 | 34% | 1.002 |
| v7 | 52.598 | 4.02 | 0.003 | 2.4426 | 10.0484 | 215.962 | 27% | 1.825 |
| v7mini | 45.131 | 4.02 | 0.003 | 0.4164 | 9.9079 | 223.066 | 26% | 5.425 |
| v7q | 19.450 | 4.00 | 0.01 | 9.8074 | 9.8309 | 158.573 | 11% | 1.005 |

**Best rate at the edge of the grid** on at least one workload: adamw, v6, v7q. Their real best may lie outside the grid, so those rows may understate them.

`Step only` is the optimizer's `step()` on fixed gradients. `Full step` is forward, backward and step. `lr sensitivity` is the largest final loss over the smallest across the learning-rate grid: closer to 1 is less sensitive.

Every figure was measured in this run, on this machine.
