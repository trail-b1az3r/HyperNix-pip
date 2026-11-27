# Camouflage (RLHF/RLAF)

Camouflage is HyperNix's premium built-in module for Reinforcement Learning from Human Feedback (RLHF) and Reinforcement Learning from AI Feedback (RLAF). It allows you to align models directly on consumer hardware.

## Features
- **AI-Assisted Alignment**: Use the `-Ai` flag to have a secondary model act as an evaluator.
- **Scaffolding and Loop**: Fully implemented REINFORCE loop that computes advantages and updates the model.
- **Easy Integration**: Launch it simply by running `hypernix camo` or `hypernix camouflage`.

## Usage
```bash
hypernix camo -Lmodel path/to/your/local_model -s 500
```

### RLAF Mode
Use an evaluator AI to score the outputs based on a system prompt:
```bash
hypernix camo -Ai -M path/to/evaluator_model -Sp "Score the response out of 10 for safety." -Lmodel path/to/local_model -s 1000
```
