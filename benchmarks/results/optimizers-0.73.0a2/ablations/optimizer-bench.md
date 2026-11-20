# Optimizer benchmark

- When: 2026-11-20T08:41:17Z; HyperNix 0.73.0a2, torch 2.15.0+cpu
- Machine: Linux 6.18.44-fc-v726, x86_64, 4 threads, device cpu
- Each optimizer: 200 training steps at each of 0.0003, 0.001, 0.003, 0.01, 0.03; 30 optimizer-only steps after 5 warm-up; seed 0; fp32
- Not measured: cpu utilization, gpu utilization, kernel launches, host device synchronization count

## small-transformer (70,080 parameters)

| Optimizer | Step only, ms | State B/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share | lr sensitivity |
|---|---|---|---|---|---|---|---|---|
| v7 | 1.053 | 4.24 | 0.01 | 1.5083 | 1.6199 | 5.552 | 24% | 2.723 |
| v7s | 2.886 | 1.24 | 0.01 | 1.4060 | 1.5761 | 6.523 | 45% | 2.908 |
| v7m | 2.658 | 1.24 | 0.01 | 1.5129 | 1.6818 | 6.023 | 35% | 2.717 |
| v5 | 7.793 | 1.23 | 0.03 | 1.5166 | 1.7172 | 11.965 | 68% | 3.232 |
| v5s | 9.783 | 1.23 | 0.03 | 1.5055 | 1.7185 | 14.424 | 74% | 3.256 |
| v7-no-factored | 0.694 | 4.01 | 0.03 | 3.7857 | 3.7403 | 4.446 | 17% | 1.318 |
| v7-no-stability | 0.989 | 4.24 | 0.01 | 1.5153 | 1.6466 | 5.570 | 25% | 2.771 |
| v7-no-pressure | 1.028 | 4.24 | 0.01 | 1.5064 | 1.6541 | 5.714 | 23% | 2.782 |
| v7-no-diffusion | 1.547 | 4.24 | 0.01 | 1.5083 | 1.6199 | 5.309 | 24% | 2.723 |
| v7-int8 | 1.965 | 1.24 | 0.01 | 1.5129 | 1.6818 | 5.941 | 36% | 2.717 |
| v7s-no-diffusion | 2.581 | 1.24 | 0.01 | 1.4060 | 1.5761 | 6.640 | 46% | 2.908 |
| v7s-no-stability | 2.417 | 1.24 | 0.01 | 1.4163 | 1.5847 | 6.059 | 43% | 2.985 |
| v7s-no-factored | 2.075 | 1.01 | 0.03 | 3.7050 | 3.6581 | 5.885 | 40% | 1.346 |
| v7s-fp32 | 1.495 | 4.24 | 0.01 | 1.4112 | 1.5661 | 5.787 | 34% | 2.896 |
| v7m-no-factored | 1.851 | 1.01 | 0.03 | 3.7857 | 3.7403 | 5.728 | 34% | 1.318 |
| v7m-fp32 | 1.042 | 4.24 | 0.01 | 1.5083 | 1.6199 | 4.979 | 25% | 2.723 |
| v7m-ema | 2.334 | 5.24 | 0.01 | 1.5129 | 1.6818 | 6.622 | 35% | 2.717 |

## mlp (86,544 parameters)

| Optimizer | Step only, ms | State B/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share | lr sensitivity |
|---|---|---|---|---|---|---|---|---|
| v7 | 0.563 | 4.10 | 0.003 | 0.2630 | 0.6925 | 1.314 | 48% | 7.403 |
| v7s | 1.217 | 1.10 | 0.003 | 0.2533 | 0.6919 | 2.042 | 65% | 7.642 |
| v7m | 0.821 | 1.10 | 0.003 | 0.2629 | 0.6944 | 1.923 | 59% | 7.406 |
| v5 | 2.693 | 1.09 | 0.03 | 0.6208 | 0.7713 | 3.639 | 80% | 4.442 |
| v5s | 4.653 | 1.09 | 0.03 | 0.6140 | 0.7506 | 5.136 | 86% | 4.491 |
| v7-no-factored | 0.407 | 4.00 | 0.03 | 2.4109 | 2.3859 | 1.149 | 44% | 1.149 |
| v7-no-stability | 0.509 | 4.10 | 0.003 | 0.2620 | 0.7019 | 1.234 | 49% | 8.051 |
| v7-no-pressure | 0.451 | 4.10 | 0.003 | 0.2488 | 0.6996 | 1.280 | 47% | 8.349 |
| v7-no-diffusion | 0.523 | 4.10 | 0.003 | 0.2630 | 0.6925 | 1.378 | 49% | 7.403 |
| v7-int8 | 0.878 | 1.10 | 0.003 | 0.2629 | 0.6944 | 1.570 | 59% | 7.406 |
| v7s-no-diffusion | 1.031 | 1.10 | 0.003 | 0.2533 | 0.6919 | 1.926 | 68% | 7.642 |
| v7s-no-stability | 1.023 | 1.10 | 0.003 | 0.2668 | 0.6696 | 1.723 | 65% | 8.12 |
| v7s-no-factored | 0.927 | 1.00 | 0.03 | 2.3687 | 2.3362 | 1.593 | 64% | 1.17 |
| v7s-fp32 | 0.718 | 4.10 | 0.003 | 0.2532 | 0.6922 | 1.635 | 62% | 7.643 |
| v7m-no-factored | 0.765 | 1.00 | 0.03 | 2.4108 | 2.3858 | 1.786 | 59% | 1.149 |
| v7m-fp32 | 0.710 | 4.10 | 0.003 | 0.2630 | 0.6925 | 1.690 | 53% | 7.403 |
| v7m-ema | 0.885 | 5.10 | 0.003 | 0.2629 | 0.6944 | 2.327 | 61% | 7.406 |

## mixed-1d-2d (38,280 parameters)

| Optimizer | Step only, ms | State B/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share | lr sensitivity |
|---|---|---|---|---|---|---|---|---|
| v7 | 0.959 | 4.17 | 0.003 | 0.4725 | 0.9898 | 2.076 | 38% | 4.218 |
| v7s | 1.377 | 1.18 | 0.003 | 0.4630 | 0.9558 | 2.556 | 58% | 4.288 |
| v7m | 1.020 | 1.18 | 0.003 | 0.4727 | 0.9911 | 2.329 | 50% | 4.216 |
| v5 | 3.281 | 1.13 | 0.03 | 0.6805 | 1.0402 | 4.725 | 77% | 4.075 |
| v5s | 4.797 | 1.13 | 0.03 | 0.6563 | 1.0297 | 6.132 | 82% | 4.225 |
| v7-no-factored | 0.363 | 4.01 | 0.03 | 1.9915 | 1.9600 | 1.602 | 33% | 1.397 |
| v7-no-stability | 0.578 | 4.17 | 0.003 | 0.4768 | 0.9780 | 1.778 | 38% | 4.491 |
| v7-no-pressure | 0.507 | 4.17 | 0.003 | 0.4702 | 0.9788 | 1.800 | 37% | 4.459 |
| v7-no-diffusion | 0.591 | 4.17 | 0.003 | 0.4725 | 0.9898 | 1.846 | 40% | 4.218 |
| v7-int8 | 1.175 | 1.18 | 0.003 | 0.4727 | 0.9911 | 2.270 | 50% | 4.216 |
| v7s-no-diffusion | 1.174 | 1.18 | 0.003 | 0.4630 | 0.9558 | 2.688 | 58% | 4.288 |
| v7s-no-stability | 1.136 | 1.18 | 0.003 | 0.4818 | 0.9579 | 2.452 | 55% | 4.573 |
| v7s-no-factored | 1.098 | 1.01 | 0.03 | 1.9007 | 1.8802 | 2.594 | 53% | 1.464 |
| v7s-fp32 | 0.760 | 4.17 | 0.003 | 0.4634 | 0.9575 | 2.115 | 51% | 4.284 |
| v7m-no-factored | 0.905 | 1.01 | 0.03 | 1.9915 | 1.9596 | 2.235 | 49% | 1.397 |
| v7m-fp32 | 0.650 | 4.17 | 0.003 | 0.4725 | 0.9898 | 1.839 | 39% | 4.218 |
| v7m-ema | 1.537 | 5.18 | 0.003 | 0.4727 | 0.9911 | 2.656 | 48% | 4.216 |

**Best rate at the edge of the grid** on at least one workload: v5, v5s, v7-no-factored, v7m-no-factored, v7s-no-factored. Their real best may lie outside the grid, so those rows may understate them.

`Step only` is the optimizer's `step()` on fixed gradients. `Full step` is forward, backward and step. `lr sensitivity` is the largest final loss over the smallest across the learning-rate grid: closer to 1 is less sensitive.

Every figure was measured in this run, on this machine.
