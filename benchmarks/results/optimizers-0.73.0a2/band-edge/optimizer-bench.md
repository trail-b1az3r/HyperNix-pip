# Optimizer benchmark

- When: 2026-11-20T09:05:06Z; HyperNix 0.73.0a2, torch 2.15.0+cpu
- Machine: Linux 6.18.44-fc-v726, x86_64, 4 threads, device cpu
- Each optimizer: 8 training steps at each of 0.01; 2 optimizer-only steps after 5 warm-up; seed 0; fp32
- Not measured: cpu utilization, gpu utilization, kernel launches, host device synchronization count

## mlp (86,544 parameters)

| Optimizer | Step only, ms | State B/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share | lr sensitivity |
|---|---|---|---|---|---|---|---|---|
| v7 | 0.581 | 4.10 | 0.01 | 2.7510 | 2.7548 | 1.402 | 44% |  |

## V7Mini's band (39,000,000 < n < 750,000,000)

`step()` alone, median ms over 6 steps, on tensors with a transformer's shapes and random gradients. This is the optimizer's cost at that size; it says nothing about training quality.

| Parameters | In band | Auto-selects | adamw, ms | v6, ms | v7, ms | v7mini, ms | v7q, ms |
|---|---|---|---|---|---|---|---|
| 749,000,000 | yes | v7mini | 2075.2 | 1500.9 | 1633.7 | 1420.6 | 372.9 |
| 750,000,000 | no | v7 | 1942.9 | 1436.3 | 1489.1 | 1414.5 | 386.2 |

`Step only` is the optimizer's `step()` on fixed gradients. `Full step` is forward, backward and step. `lr sensitivity` is the largest final loss over the smallest across the learning-rate grid: closer to 1 is less sensitive.

Every figure was measured in this run, on this machine.
