# Windows native

This is the primary validated path. It uses AMD's native Windows ROCm PyTorch
wheel, Visual Studio 2022 Build Tools, `amdclang-cl`, and a `.pyd` extension.

```powershell
$env:ROCM_PYTHON = "C:\path\to\rocm-venv\Scripts\python.exe"
.\scripts\build.ps1
.\scripts\test.ps1
```

Validated on RX 7800 XT (`gfx1101`) with PyTorch 2.12 and ROCm 7.14. This path
does not use WSL, Docker, `/dev/dxg`, or Linux ABI compatibility.
