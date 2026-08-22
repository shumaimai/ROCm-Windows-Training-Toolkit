# v0.1.0

First source and container release of the unofficial ROCm Windows Training
Toolkit.

- Windows-native HIP extension build and synthetic tests
- WSL2 Docker validation through `/dev/dxg`
- Linux-native Docker definition through `/dev/kfd` and `/dev/dri`
- ROCm 7.14 / PyTorch 2.12 / ROCDXG 1.2.2 pinned WSL2 image
- RX 7800 XT (`gfx1101`) validated
- Apache-2.0 source-only repository; no model or training data included

The causal convolution kernel is validated. The SSD kernel remains
experimental because multi-step model integration produced NaN gradients.
