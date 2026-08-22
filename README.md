# Unofficial ROCm Windows Training Toolkit

Independent HIP kernels and PyTorch extension tooling for native Windows AMD
GPU training experiments. This is an unofficial source-only ROCm compatibility
project; it is not an AMD, PyTorch, PFN, or Mamba distribution.

## Scope

- SSD-style recurrent forward/backward kernel
- width-limited causal convolution forward/backward kernel
- source/Torch/HIP/GPU-architecture hash builds
- FP32 parity and independent FP16/BF16 closed-form tests
- no model weights, model code, tokenizer, checkpoint, adapter, training data,
  ROCm runtime DLLs, or prebuilt extension binaries

## Status

Validated locally on:

- Windows 11 build 26200
- Radeon RX 7800 XT (`gfx1101`, 16 GiB)
- PyTorch `2.12.0+rocm7.14.0`
- HIP `7.14.60850`
- Python 3.12

The causal convolution kernel passes multi-step integration in the source
project. The SSD kernel passes synthetic parity checks but produced NaN
gradients in multi-step PLaMo QLoRA, so SSD is experimental and must not be
treated as a stable training backend.

## Build and test

Install a matching AMD ROCm PyTorch environment and the ROCm development SDK.
Set `ROCM_PYTHON` if it is not the active Python.

```powershell
$env:ROCM_PYTHON = "C:\path\to\rocm-venv\Scripts\python.exe"
.\scripts\build.ps1
.\scripts\test.ps1
```

The build identity hashes the C++/HIP sources, PyTorch version, HIP version,
and detected GPU architecture. Generated `.pyd`, `.obj`, `.lib`, and build
directories are ignored and are not distributed.

## Python API

```python
from rocm_windows_training.hip_extension import (
    extension_info,
    make_autograd_function,
    make_conv_autograd_function,
)

ssd = make_autograd_function()          # experimental
causal_conv = make_conv_autograd_function()
print(extension_info())
```

## Reporting another Radeon GPU

Open the Radeon validation Issue form and include GPU model, VRAM,
`gcnArchName`, driver, OS, Python, PyTorch/HIP versions, source commit, build
identity, and first failing command. Do not attach credentials, model files,
checkpoints, adapters, or private data.

See [GPU_MATRIX.md](GPU_MATRIX.md) for requested architectures and
[CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## License and trademarks

ROCm Windows Training Toolkit is licensed under Apache-2.0. Runtime dependencies remain
under their respective licenses; see [THIRD_PARTY.md](THIRD_PARTY.md) and
[NOTICE](NOTICE).

This project is independently developed and is not affiliated with, sponsored
by, or endorsed by Advanced Micro Devices, Inc., the PyTorch Foundation, The
Linux Foundation, Preferred Networks, Inc., Preferred Elements, Inc., or the
authors of Mamba.
