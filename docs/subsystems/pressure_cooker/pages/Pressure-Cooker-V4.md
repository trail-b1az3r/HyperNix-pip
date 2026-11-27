# Pressure Cooker V4

The Pressure Cooker V4 is HyperNix's optimized quantization-aware training mechanism and optimizer wrapper.

## Enhancements in 0.70.5a2
- **Pascal Compatibility**: Includes hardware-specific optimizations and architectural warnings for CUDA 6.1/6.2 (`Agedcookerv4`).
- **Sophia Clipping Approximation**: Enabled through `sophia_clipping` flags, approximating Hutchinson curvature via a gradient exponential moving average (EMA) to scale gradients before clipping.
- **IQ-Quant Scaling**: In `Ultracookerv4`, we added a heuristic scaling mechanism for low-bit quantization modes (like `iq`) to maintain gradient stability across 16-bit tensors.

## Getting Started
Instead of using vanilla AdamW, simply swap your optimizer for `PressureCookerV4` or its variants to unlock seamless mixed-precision and quantization-aware dynamics.
