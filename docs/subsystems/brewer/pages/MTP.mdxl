# Multi-Token Prediction (MTP)

## Overview

Multi-Token Prediction (MTP) trains language models to predict multiple future tokens simultaneously, improving sample efficiency by 1.5-3x and enabling speculative decoding at inference time.

## How It Works

Instead of predicting only the next token, MTP adds auxiliary prediction heads that predict tokens at positions t+1, t+2, ..., t+D. The main loss is combined with weighted MTP losses:

```
L_total = L_main + lambda * sum(w_i * L_mtp_i)
```

## Features

- **Sequential mode**: Feed predictions back as input for next position
- **Independent mode**: Parallel predictions from same hidden state
- **Shared heads**: Single projection layer for all positions
- **Loss weighting**: Linearly decaying or custom per-token weights
- **Native workshop integration**: Built into WorkshopFramework

## Usage

### With PressureCookerV5

```python
from hypernix.pressure_cooker_v5 import PressureCookerV5, MTPConfig

mtp_cfg = MTPConfig(num_tokens=4, lambda_weight=0.3, sequential=True)
cooker = PressureCookerV5(
    model.parameters(),
    enable_mtp=True,
    mtp_config=mtp_cfg,
)

# Get MTP head
mtp_head = cooker.get_mtp_head(hidden_dim=768, vocab_size=32000)
```

### With Workshop

```python
from hypernix.workshop import WorkshopFramework, FrameworkConfig

config = FrameworkConfig(enable_mtp=True)
framework = WorkshopFramework(config)
framework.build()

# Attach MTP head
mtp_head = framework.attach_mtp_head(hidden_dim=768, vocab_size=32000)

# Compute combined loss
losses = framework.compute_mtp_loss(logits, labels, hidden_states)
total_loss = losses["total"]  # main + weighted MTP losses
```

### Standalone

```python
from hypernix.mtp import MTPTrainer, MTPConfig, MTPHead

config = MTPConfig(num_tokens=4, lambda_weight=0.3)
mtp = MTPTrainer(model, config)
mtp_head = mtp.attach_head(hidden_dim=768, vocab_size=1000)

# In training loop
losses = mtp.compute_loss(logits, labels, hidden_states)
losses["total"].backward()
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| num_tokens | 4 | Number of future tokens (D) |
| lambda_weight | 0.3 | MTP loss weight |
| sequential | True | Chain predictions |
| shared_head | True | Share projection layer |
| temperature | 1.0 | Softmax temperature |
| label_smoothing | 0.0 | Label smoothing |

## CLI

```bash
hnx mtp --tokens 4 --lambda 0.3 --sequential --shared-head
```
