# Contributing

Thank you for helping validate and improve WaveTrain Windows.

## Ground rules

- Be precise about what was tested. A successful source build does not prove
  runtime correctness.
- Do not attach API keys, access tokens, private data, model weights,
  tokenizers, checkpoints, adapters, generated extension binaries, or caches.
- Keep generated files under ignored directories such as `artifacts/`.
- Do not claim AMD, PyTorch, PFN, or Mamba endorsement.
- The SSD kernel is experimental. Do not describe it as a stable training
  backend while issue #3 remains open.

## License for contributions

By submitting a contribution, you agree that your contribution is licensed
under the repository's Apache License 2.0. Do not submit code that you do not
have permission to contribute.

If a change copies or adapts third-party source, identify the exact source,
version, license, and modifications in the pull request. Independent
implementations should cite the relevant paper or public specification.

## Development setup

Use a matching native Windows ROCm PyTorch environment and ROCm development
SDK. Do not replace the AMD PyTorch wheel with PyPI `torch`.

```powershell
$env:ROCM_PYTHON = "C:\path\to\rocm-venv\Scripts\python.exe"
.\scripts\build.ps1
.\scripts\test.ps1
```

## Pull requests

Before opening a pull request:

1. Run Python and PowerShell syntax checks.
2. Build from a clean ignored `artifacts/` directory.
3. Run `scripts/test.ps1` on an actual Radeon GPU.
4. Record GPU model, `gcnArchName`, driver, PyTorch, HIP/ROCm, and build
   identity.
5. Compare forward and backward values against the Torch references.
6. Confirm no generated binaries, model materials, artifacts, or credentials
   are tracked.
7. Update README, NOTICE, or THIRD_PARTY when licensing or packaging
   boundaries change.

Keep pull requests focused. Separate kernel correctness, performance tuning,
packaging, and CI changes when possible.

## Performance reports

Report warm-up count, measured iterations, distribution statistics, tensor
shape, dtype, forward-only versus forward/backward, peak VRAM, GPU power mode,
thermal state, and concurrent load. One timing sample is not sufficient.

## Security and private reports

Do not post credentials or private data in Issues. Revoke exposed keys
immediately. This project currently has no private security-reporting channel,
so redact reproductions to synthetic data before filing.
