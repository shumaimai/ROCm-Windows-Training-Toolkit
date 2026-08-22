param(
    [ValidateSet("validate", "build", "shell", "check", "release")]
    [string]$Action = "validate",
    [string]$Distro = "Ubuntu-24.04",
    [string]$User = "",
    [string]$GpuArch = "gfx1101",
    [string]$ImageTag = "latest-gfx1101"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

$Distros = (wsl --list --quiet) -replace "`0", "" | ForEach-Object { $_.Trim() }
if ($Distro -notin $Distros) { throw "WSL distribution not found: $Distro" }
if (-not $User) {
    $User = (wsl -d $Distro -- id -un).Trim()
    if (-not $User) { throw "Failed to detect the default user for $Distro." }
}

$ResolvedRoot = (Resolve-Path -LiteralPath $Root).Path
if ($ResolvedRoot -notmatch '^([A-Za-z]):\\(.*)$') {
    throw "Only local Windows drive paths are supported: $ResolvedRoot"
}
$Drive = $Matches[1].ToLowerInvariant()
$Tail = $Matches[2].Replace('\', '/')
$LinuxRoot = "/mnt/$Drive/$Tail"

$PreflightCommands = @(
    @("test", "-c", "/dev/dxg"),
    @("test", "-f", "/usr/lib/wsl/lib/libdxcore.so"),
    @("test", "-S", "/var/run/docker.sock"),
    @("docker", "info")
)
foreach ($Command in $PreflightCommands) {
    wsl -d $Distro -u $User -- @Command 1>$null
    if ($LASTEXITCODE -ne 0) {
        throw "WSL2/Docker/DXG preflight failed: $($Command -join ' ')"
    }
}

$DockerArguments = switch ($Action) {
    "build" { @("compose", "-f", "docker/compose.wsl2.yml", "build") }
    "shell" { @("compose", "-f", "docker/compose.wsl2.yml", "run", "--rm", "validate", "shell") }
    "check" { @("compose", "-f", "docker/compose.wsl2.yml", "run", "--rm", "validate", "check") }
    "release" { @("compose", "-f", "docker/compose.wsl2.release.yml", "run", "--rm", "--pull", "always", "validate") }
    default { @("compose", "-f", "docker/compose.wsl2.yml", "run", "--rm", "--build", "validate") }
}
wsl -d $Distro -u $User --cd $LinuxRoot -- env "GPU_ARCH=$GpuArch" "IMAGE_TAG=$ImageTag" docker @DockerArguments
if ($LASTEXITCODE -ne 0) { throw "WSL2 Docker action failed: $Action" }
