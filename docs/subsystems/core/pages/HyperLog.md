# Hyper-Log

Hyper-Log is a premium Training TUI Logger for HyperNix. 

## Overview
When running training loops, Hyper-Log produces consistent, highly user-friendly colored logs that remain fully compatible with tvtop, tvtop++, and cctvtop parsers. 

## Features
- **Deep Metrics**: Visualizes Step, Loss, Grad Norm (up to 5 decimals), Learning Rate, and Epochs.
- **Hardware Telemetry**: Real-time readouts of GPU temps, Power usage, and Storage remaining.
- **Emergency Controls**: Press `P` to pause training and `S` for emergency checkpoints and stop.
- **ETA & Throughput**: See estimated completion time and iterations per second dynamically.

## Usage
Hyper-Log can be initialized in your custom training scripts:

```python
from hypernix.hyper_log import HyperLogger

logger = HyperLogger(total_steps=1000)
logger.start()

for step in range(1000):
    # ... training code ...
    logger.update(step, loss, grad_norm, lr, epoch)

logger.stop()
```
