# Optimizer benchmark

- When: 2026-11-20T08:23:43Z; HyperNix 0.73.0a2, torch 2.15.0+cpu
- Machine: Linux 6.18.44-fc-v726, x86_64, 4 threads, device cpu
- Each optimizer: 200 training steps at each of 0.0003, 0.001, 0.003, 0.01, 0.03; 30 optimizer-only steps after 5 warm-up; seed 0; fp32
- Not measured: cpu utilization, gpu utilization, kernel launches, host device synchronization count

## small-transformer (70,080 parameters)

| Optimizer | Step only, ms | State B/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share | lr sensitivity |
|---|---|---|---|---|---|---|---|---|
| adamw | 0.955 | 8.00 | 0.01 | 0.8723 | 1.2203 | 4.668 | 24% | 4.253 |
| v5 | 7.280 | 1.23 | 0.03 | 1.5166 | 1.7172 | 11.674 | 68% | 3.232 |
| v5plus | 10.870 | 6.13 | 0.03 | 1.5471 | 1.7386 | 16.090 | 74% | 3.171 |
| v5s | 10.818 | 1.23 | 0.03 | 1.5055 | 1.7185 | 14.650 | 75% | 3.256 |
| v6 | 0.641 | 4.00 | 0.03 | 3.9171 | 3.8755 | 4.520 | 17% | 1.274 |
| v6v | N/A: Pressure Cooker V6V needs CUDA; this run is on cpu | | | | | | | |
| v7 | 1.097 | 4.24 | 0.01 | 1.5083 | 1.6199 | 5.136 | 24% | 2.723 |
| v7mini | 1.376 | 4.24 | 0.01 | 1.6386 | 1.7214 | 4.988 | 24% | 2.492 |
| v7q | 0.240 | 4.01 | 0.03 | 3.9147 | 3.8729 | 4.068 | 8% | 1.275 |
| v7s | 2.812 | 1.24 | 0.01 | 1.4060 | 1.5761 | 6.469 | 46% | 2.908 |
| v7m | 2.084 | 1.24 | 0.01 | 1.5129 | 1.6818 | 6.511 | 36% | 2.717 |

## medium-transformer (387,936 parameters)

| Optimizer | Step only, ms | State B/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share | lr sensitivity |
|---|---|---|---|---|---|---|---|---|
| adamw | 1.966 | 8.00 | 0.003 | 0.6171 | 1.2598 | 11.073 | 19% | 4.922 |
| v5 | 15.783 | 1.12 | 0.03 | 1.0368 | 1.4706 | 23.896 | 64% | 5.329 |
| v5plus | 19.790 | 6.07 | 0.03 | 1.0909 | 1.4624 | 31.516 | 71% | 5.07 |
| v5s | 24.122 | 1.12 | 0.03 | 1.0231 | 1.4596 | 33.412 | 74% | 5.4 |
| v6 | 1.443 | 4.00 | 0.03 | 4.1698 | 4.0423 | 10.658 | 16% | 1.362 |
| v6v | N/A: Pressure Cooker V6V needs CUDA; this run is on cpu | | | | | | | |
| v7 | 2.038 | 4.12 | 0.003 | 1.3572 | 1.6161 | 10.819 | 19% | 2.733 |
| v7mini | 1.726 | 4.12 | 0.003 | 1.4090 | 1.6591 | 11.283 | 17% | 2.586 |
| v7q | 0.506 | 4.00 | 0.03 | 4.1659 | 4.0383 | 9.164 | 7% | 1.363 |
| v7s | 4.986 | 1.12 | 0.003 | 1.2713 | 1.5432 | 13.043 | 41% | 2.916 |
| v7m | 3.589 | 1.12 | 0.003 | 1.3586 | 1.6225 | 13.052 | 30% | 2.733 |

## large-transformer (1,985,664 parameters)

| Optimizer | Step only, ms | State B/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share | lr sensitivity |
|---|---|---|---|---|---|---|---|---|
| adamw | 2.905 | 8.00 | 0.003 | 0.2452 | 0.9230 | 39.061 | 11% | 19.673 |
| v5 | 29.615 | 1.06 | 0.03 | 0.6796 | 1.1643 | 61.020 | 46% | 8.939 |
| v5plus | 36.249 | 6.03 | 0.03 | 0.7160 | 1.1353 | 69.688 | 53% | 8.488 |
| v5s | 47.985 | 1.06 | 0.03 | 0.6641 | 1.1585 | 86.486 | 60% | 9.149 |
| v6 | 3.053 | 4.00 | 0.03 | 4.4558 | 4.1950 | 40.513 | 11% | 1.426 |
| v6v | N/A: Pressure Cooker V6V needs CUDA; this run is on cpu | | | | | | | |
| v7 | 4.253 | 4.06 | 0.003 | 0.7971 | 1.0663 | 38.521 | 11% | 3.567 |
| v7mini | 2.813 | 4.06 | 0.003 | 0.8787 | 1.1018 | 36.621 | 10% | 3.75 |
| v7q | 1.033 | 4.00 | 0.03 | 4.4503 | 4.1887 | 34.500 | 4% | 1.428 |
| v7s | 7.699 | 1.06 | 0.003 | 0.6994 | 0.9947 | 47.849 | 22% | 4.069 |
| v7m | 7.943 | 1.06 | 0.003 | 0.7970 | 1.0704 | 43.600 | 17% | 3.565 |

## mlp (86,544 parameters)

| Optimizer | Step only, ms | State B/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share | lr sensitivity |
|---|---|---|---|---|---|---|---|---|
| adamw | 0.422 | 8.00 | 0.003 | 0.1701 | 0.8938 | 1.422 | 43% | 7.848 |
| v5 | 3.727 | 1.09 | 0.03 | 0.6208 | 0.7713 | 3.746 | 81% | 4.442 |
| v5plus | 3.644 | 6.05 | 0.03 | 0.6196 | 0.7620 | 5.952 | 86% | 4.45 |
| v5s | 5.665 | 1.09 | 0.03 | 0.6140 | 0.7506 | 4.946 | 85% | 4.491 |
| v6 | 0.360 | 4.00 | 0.03 | 2.4681 | 2.4522 | 1.090 | 41% | 1.123 |
| v6v | N/A: Pressure Cooker V6V needs CUDA; this run is on cpu | | | | | | | |
| v7 | 0.599 | 4.10 | 0.003 | 0.2630 | 0.6925 | 1.677 | 51% | 7.403 |
| v7mini | 0.482 | 4.10 | 0.003 | 0.3192 | 0.7018 | 1.265 | 48% | 5.943 |
| v7q | 0.136 | 4.00 | 0.03 | 2.4656 | 2.4493 | 0.849 | 29% | 1.124 |
| v7s | 1.191 | 1.10 | 0.003 | 0.2533 | 0.6919 | 2.009 | 66% | 7.642 |
| v7m | 0.904 | 1.10 | 0.003 | 0.2629 | 0.6944 | 2.093 | 62% | 7.406 |

## embedding-heavy (1,285,200 parameters)

| Optimizer | Step only, ms | State B/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share | lr sensitivity |
|---|---|---|---|---|---|---|---|---|
| adamw | 2.809 | 8.00 | 0.01 | 0.0086 | 5.0736 | 2.047 | 67% | 316.614 |
| v5 | 17.653 | 1.08 | 0.03 | 1.3066 | 2.8899 | 18.169 | 95% | 2.124 |
| v5plus | 23.225 | 6.05 | 0.03 | 2.1868 | 2.7693 | 23.247 | 96% | 1.269 |
| v5s | 61.694 | 1.08 | 0.03 | 1.3515 | 2.8772 | 60.059 | 98% | 2.053 |
| v6 | 1.588 | 4.00 | 0.03 | 2.7004 | 2.7431 | 2.115 | 71% | 1.028 |
| v6v | N/A: Pressure Cooker V6V needs CUDA; this run is on cpu | | | | | | | |
| v7 | 1.374 | 4.13 | 0.003 | 0.5973 | 4.2796 | 1.891 | 67% | 2.439 |
| v7mini | 1.184 | 4.13 | 0.003 | 0.3567 | 5.4862 | 1.792 | 65% | 5.939 |
| v7q | 0.389 | 4.00 | 0.03 | 2.7003 | 2.7432 | 1.109 | 42% | 1.028 |
| v7s | 2.877 | 1.13 | 0.001 | 1.3598 | 3.4100 | 4.007 | 83% | 1.608 |
| v7m | 2.247 | 1.13 | 0.003 | 0.6446 | 3.8487 | 2.743 | 75% | 1.918 |

## mixed-1d-2d (38,280 parameters)

| Optimizer | Step only, ms | State B/param | Best lr | Final loss | Held-out loss | Full step, ms | Optimizer share | lr sensitivity |
|---|---|---|---|---|---|---|---|---|
| adamw | 0.439 | 8.00 | 0.003 | 0.3087 | 1.0343 | 2.136 | 32% | 4.589 |
| v5 | 4.253 | 1.13 | 0.03 | 0.6805 | 1.0402 | 4.837 | 77% | 4.075 |
| v5plus | 4.794 | 6.07 | 0.03 | 0.6921 | 1.0507 | 6.345 | 82% | 4.006 |
| v5s | 5.386 | 1.13 | 0.03 | 0.6563 | 1.0297 | 6.454 | 82% | 4.225 |
| v6 | 0.441 | 4.00 | 0.03 | 2.1073 | 2.0649 | 1.614 | 30% | 1.321 |
| v6v | N/A: Pressure Cooker V6V needs CUDA; this run is on cpu | | | | | | | |
| v7 | 0.572 | 4.17 | 0.003 | 0.4725 | 0.9898 | 2.074 | 42% | 4.218 |
| v7mini | 0.534 | 4.17 | 0.003 | 0.5088 | 1.0426 | 1.781 | 39% | 3.834 |
| v7q | 0.124 | 4.01 | 0.03 | 2.0949 | 2.0473 | 1.325 | 18% | 1.328 |
| v7s | 1.342 | 1.18 | 0.003 | 0.4630 | 0.9558 | 3.062 | 54% | 4.288 |
| v7m | 1.044 | 1.18 | 0.003 | 0.4727 | 0.9911 | 2.318 | 50% | 4.216 |

**Best rate at the edge of the grid** on at least one workload: v5, v5plus, v5s, v6, v7q. Their real best may lie outside the grid, so those rows may understate them.

`Step only` is the optimizer's `step()` on fixed gradients. `Full step` is forward, backward and step. `lr sensitivity` is the largest final loss over the smallest across the learning-rate grid: closer to 1 is less sensitive.

Every figure was measured in this run, on this machine.
