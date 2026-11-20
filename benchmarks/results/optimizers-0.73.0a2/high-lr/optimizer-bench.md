# Optimizer benchmark

- When: 2026-11-20T08:45:02Z; HyperNix 0.73.0a2, torch 2.15.0+cpu
- Machine: Linux 6.18.44-fc-v726, x86_64, 4 threads, device cpu
- Each optimizer: 200 training steps at each of 0.03, 0.1, 0.3, 1; 30 optimizer-only steps after 5 warm-up; seed 0; fp32
- Not measured: cpu utilization, gpu utilization, kernel launches, host device synchronization count

## small-transformer (70,080 parameters)

| Optimizer | Step only, ms | State B/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share | lr sensitivity |
|---|---|---|---|---|---|---|---|---|
| v5 | 7.652 | 1.23 | 0.03 | 1.5166 | 1.7172 | 13.848 | 68% | 3.272 |
| v5plus | 13.118 | 6.13 | 0.03 | 1.5471 | 1.7386 | 16.034 | 74% | 3.17 |
| v5s | 10.254 | 1.23 | 0.1 | 1.4965 | 1.7090 | 15.266 | 74% | 3.248 |
| v6 | 0.608 | 4.00 | 1 | 1.8122 | 1.7749 | 4.429 | 17% | 2.162 |
| v7q | 0.233 | 4.01 | 1 | 1.9599 | 2.0113 | 3.903 | 8% | 1.997 |

## medium-transformer (387,936 parameters)

| Optimizer | Step only, ms | State B/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share | lr sensitivity |
|---|---|---|---|---|---|---|---|---|
| v5 | 14.957 | 1.12 | 0.03 | 1.0368 | 1.4706 | 24.468 | 63% | 5.646 |
| v5plus | 22.422 | 6.07 | 0.03 | 1.0909 | 1.4624 | 31.000 | 70% | 5.157 |
| v5s | 24.173 | 1.12 | 0.03 | 1.0231 | 1.4596 | 32.855 | 74% | 5.459 |
| v6 | 1.448 | 4.00 | 1 | 1.6687 | 1.8948 | 9.476 | 17% | 2.499 |
| v7q | 0.640 | 4.00 | 1 | 1.7314 | 2.0790 | 9.622 | 7% | 2.406 |

## large-transformer (1,985,664 parameters)

| Optimizer | Step only, ms | State B/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share | lr sensitivity |
|---|---|---|---|---|---|---|---|---|
| v5 | 30.407 | 1.06 | 0.03 | 0.6796 | 1.1643 | 62.390 | 46% | 10.355 |
| v5plus | 36.855 | 6.03 | 0.03 | 0.7160 | 1.1353 | 75.248 | 53% | 9.369 |
| v5s | 65.514 | 1.06 | 0.03 | 0.6641 | 1.1585 | 100.436 | 62% | 9.728 |
| v6 | 3.232 | 4.00 | 1 | 1.5963 | 1.6170 | 37.412 | 12% | 2.791 |
| v7q | 0.850 | 4.00 | 1 | 1.6132 | 1.6462 | 32.023 | 5% | 2.759 |

## mlp (86,544 parameters)

| Optimizer | Step only, ms | State B/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share | lr sensitivity |
|---|---|---|---|---|---|---|---|---|
| v5 | 2.673 | 1.09 | 0.03 | 0.6208 | 0.7713 | 3.444 | 81% | 4177.34 |
| v5plus | 3.635 | 6.05 | 0.03 | 0.6196 | 0.7620 | 4.530 | 85% | 35.512 |
| v5s | 4.059 | 1.09 | 0.03 | 0.6140 | 0.7506 | 4.947 | 86% | 288.264 |
| v6 | 0.344 | 4.00 | 0.3 | 0.6830 | 0.9441 | 1.059 | 43% | 3.613 |
| v7q | 0.128 | 4.00 | 0.3 | 0.8315 | 1.2530 | 0.865 | 27% | 2.965 |

## embedding-heavy (1,285,200 parameters)

| Optimizer | Step only, ms | State B/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share | lr sensitivity |
|---|---|---|---|---|---|---|---|---|
| v5 | 17.982 | 1.08 | 0.1 | 0.0929 | 9.8285 | 10.510 | 93% | 1383.388 |
| v5plus | 22.537 | 6.05 | 0.1 | 0.1071 | 6.7804 | 21.717 | 96% | 225.87 |
| v5s | 66.346 | 1.08 | 0.1 | 0.0697 | 7.5505 | 58.839 | 98% | 1551.012 |
| v6 | 1.603 | 4.00 | 0.1 | 2.6855 | 2.7567 | 2.192 | 71% | 1.01 |
| v7q | 0.398 | 4.00 | 0.1 | 2.6854 | 2.7567 | 1.028 | 42% | 1.009 |

## mixed-1d-2d (38,280 parameters)

| Optimizer | Step only, ms | State B/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share | lr sensitivity |
|---|---|---|---|---|---|---|---|---|
| v5 | 3.358 | 1.13 | 0.03 | 0.6805 | 1.0402 | 4.866 | 77% | 1417.403 |
| v5plus | 4.975 | 6.07 | 0.03 | 0.6921 | 1.0507 | 6.486 | 83% | 9.238 |
| v5s | 5.106 | 1.13 | 0.03 | 0.6563 | 1.0297 | 8.802 | 85% | 4.23 |
| v6 | 0.394 | 4.00 | 0.1 | 0.8687 | 1.0993 | 1.739 | 30% | 2.426 |
| v7q | 0.129 | 4.01 | 0.1 | 1.2099 | 1.3305 | 1.313 | 19% | 2.095 |

**Best rate at the edge of the grid** on at least one workload: v5, v5plus, v5s, v6, v7q. Their real best may lie outside the grid, so those rows may understate them.

`Step only` is the optimizer's `step()` on fixed gradients. `Full step` is forward, backward and step. `lr sensitivity` is the largest final loss over the smallest across the learning-rate grid: closer to 1 is less sensitive.

Every figure was measured in this run, on this machine.
