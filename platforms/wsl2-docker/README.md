# WSL2 Docker

This path runs a Linux PyTorch/HIP extension inside a container while the AMD
Windows driver exposes the GPU through WSL's `/dev/dxg` interface.

Requirements:

- Windows 11 and a WSL-compatible AMD driver
- WSL2 with Ubuntu 24.04
- Docker Desktop WSL2 engine or a compatible Docker Engine
- `/dev/dxg`
- `/usr/lib/wsl/lib/libdxcore.so`

One-command validation from PowerShell:

```powershell
.\scripts\wsl2-docker.ps1 -Action validate -Distro Ubuntu-24.04 `
  -User shuhei -GpuArch gfx1101
```

The image pins Ubuntu 24.04, ROCm 7.14, PyTorch 2.12, and ROCDXG 1.2.2. It
installs ROCm/PyTorch through AMD's multi-architecture wheel repository and
builds the Linux `.so` extension only after the GPU is available at runtime.

Validated on RX 7800 XT (`gfx1101`) with Docker Engine reachable from the
Ubuntu-24.04 WSL distribution. The test confirmed `/dev/dxg` forwarding,
PyTorch GPU discovery, Linux `.so` compilation, packed-sequence reset, FP32
forward/backward, and FP16/BF16 checks.

Docker Desktop documents NVIDIA GPU support, not AMD GPU support. AMD documents
the explicit `/dev/dxg` plus DXCore/ROCDXG container path. Treat Docker Desktop
with manual AMD forwarding as an experimental integration even when the
synthetic tests pass.

Interactive shell:

```powershell
.\scripts\wsl2-docker.ps1 -Action shell
```
