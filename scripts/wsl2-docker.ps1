param(
    [ValidateSet("validate", "build", "shell", "check")]
    [string]$Action = "validate",
    [string]$Distro = "Ubuntu-24.04",
    [string]$User = "shuhei",
    [string]$GpuArch = "gfx1101"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

$Distros = (wsl --list --quiet) -replace "`0", "" | ForEach-Object { $_.Trim() }
if ($Distro -notin $Distros) { throw "WSL distribution not found: $Distro" }

$ResolvedRoot = (Resolve-Path -LiteralPath $Root).Path
if ($ResolvedRoot -notmatch '^([A-Za-z]):\\(.*)$') {
    throw "Only local Windows drive paths are supported: $ResolvedRoot"
}
$Drive = $Matches[1].ToLowerInvariant()
$Tail = $Matches[2].Replace('\', '/')
$LinuxRoot = "/mnt/$Drive/$Tail"

$Preflight = @"
set -e
test -c /dev/dxg
test -f /usr/lib/wsl/lib/libdxcore.so
test -S /var/run/docker.sock
docker info >/dev/null
"@
wsl -d $Distro -u $User -- bash -lc $Preflight
if ($LASTEXITCODE -ne 0) { throw "WSL2/Docker/DXG preflight failed." }

$DockerArguments = switch ($Action) {
    "build" { @("compose", "-f", "docker/compose.wsl2.yml", "build") }
    "shell" { @("compose", "-f", "docker/compose.wsl2.yml", "run", "--rm", "validate", "shell") }
    "check" { @("compose", "-f", "docker/compose.wsl2.yml", "run", "--rm", "validate", "check") }
    default { @("compose", "-f", "docker/compose.wsl2.yml", "run", "--rm", "--build", "validate") }
}
wsl -d $Distro -u $User --cd $LinuxRoot -- env "GPU_ARCH=$GpuArch" docker @DockerArguments
if ($LASTEXITCODE -ne 0) { throw "WSL2 Docker action failed: $Action" }
