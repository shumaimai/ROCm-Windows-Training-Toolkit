$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = if ($env:ROCM_PYTHON) { $env:ROCM_PYTHON } else { "python" }
$env:PYTHONPATH = $Root

$VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path -LiteralPath $VsWhere)) { throw "Visual Studio vswhere.exe was not found." }
$VsRoot = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $VsRoot) { throw "Visual Studio 2022 C++ Build Tools were not found." }
$VcVars = Join-Path $VsRoot "VC\Auxiliary\Build\vcvars64.bat"
$Command = "call `"$VcVars`" >nul && `"$Python`" -c `"from rocm_windows_training.hip_extension import load_extension, extension_info; load_extension(); print(extension_info())`""
cmd.exe /d /s /c $Command
if ($LASTEXITCODE -ne 0) { throw "ROCm Windows Training HIP extension build failed." }
