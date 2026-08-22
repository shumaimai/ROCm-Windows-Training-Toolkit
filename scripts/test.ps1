$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = if ($env:ROCM_PYTHON) { $env:ROCM_PYTHON } else { "python" }
$env:PYTHONPATH = $Root
& $Python (Join-Path $Root "tests\test_kernels.py")
if ($LASTEXITCODE -ne 0) { throw "WaveTrain kernel tests failed." }
