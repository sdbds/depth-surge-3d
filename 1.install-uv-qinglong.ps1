[CmdletBinding()]
param(
    [string]$EnvironmentPath,
    [switch]$WithMoGe2
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'uv.lock') -PathType Leaf)) {
    throw 'Missing uv.lock. Run this installer from a complete Depth Surge 3D checkout.'
}

$uv = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $uv) {
    throw 'Install uv from https://docs.astral.sh/uv/getting-started/installation/ and reopen PowerShell.'
}
if (-not (Get-Command git -CommandType Application -ErrorAction SilentlyContinue)) {
    throw 'Install Git for Windows and reopen PowerShell. The lock includes Git dependencies.'
}
foreach ($tool in 'ffmpeg', 'ffprobe') {
    if (-not (Get-Command $tool -CommandType Application -ErrorAction SilentlyContinue)) {
        throw "Missing $tool. Install FFmpeg from https://ffmpeg.org/download.html and add its bin directory to PATH."
    }
}

$previousTarget = $env:UV_PROJECT_ENVIRONMENT
if (-not $EnvironmentPath) {
    $EnvironmentPath = if (-not $previousTarget) {
        Join-Path $PSScriptRoot '.venv'
    } elseif ([IO.Path]::IsPathRooted($previousTarget)) {
        $previousTarget
    } else {
        Join-Path $PSScriptRoot $previousTarget
    }
}
$target = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($EnvironmentPath)
$python = Join-Path $target 'Scripts\python.exe'
$cli = Join-Path $target 'Scripts\depth-surge-3d.exe'
$request = '3.11'

# uv sync can replace an environment when its interpreter does not match the request.
$existing = Get-Item -LiteralPath $target -Force -ErrorAction SilentlyContinue
if ($existing) {
    if (($existing.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
        -not (Test-Path -LiteralPath (Join-Path $target 'pyvenv.cfg') -PathType Leaf) -or
        -not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "Refusing to replace existing path '$target'. Choose a new -EnvironmentPath."
    }
    $check = "import os, struct, sys; sys.exit(not (sys.implementation.name == 'cpython' and sys.version_info[:2] == (3, 11) and struct.calcsize('P') == 8 and sys.prefix != sys.base_prefix and os.path.samefile(sys.prefix, sys.argv[1])))"
    & $python -I -c $check $target
    if ($LASTEXITCODE -ne 0) {
        throw "Existing environment '$target' must be 64-bit CPython 3.11. Choose a new -EnvironmentPath; nothing was removed."
    }
    $request = $python
}

$syncArgs = @('sync', '--locked', '--no-editable', '--no-dev', '--inexact', '--python', $request, '--project', $PSScriptRoot)
if ($WithMoGe2) { $syncArgs += '--extra', 'moge2' }

Write-Host "Installing Depth Surge 3D into $target"
try {
    $env:UV_PROJECT_ENVIRONMENT = $target
    & $uv @syncArgs
    if ($LASTEXITCODE -ne 0) {
        throw "uv sync failed (exit $LASTEXITCODE). See the error above. For native build errors install MSVC C++17 and the Windows SDK; see docs/INSTALLATION.md."
    }
} finally {
    $env:UV_PROJECT_ENVIRONMENT = $previousTarget
}

if (-not (Test-Path -LiteralPath $cli -PathType Leaf)) {
    throw "Missing installed CLI: $cli. Check that uv installed the project, not just its dependencies."
}
if (-not $existing) {
    & $uv pip check --python $python
    if ($LASTEXITCODE -ne 0) {
        throw "New environment failed dependency checks (exit $LASTEXITCODE). See the conflicting or missing packages above."
    }
}
# -I excludes the working directory; check wheel ownership too, since .pth files still load.
@'
import json
import sys
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.metadata import distribution
from pathlib import Path

dist = distribution("depth-surge-3d")
if json.loads(dist.read_text("direct_url.json") or "{}").get("dir_info", {}).get("editable"):
    sys.exit("Found an editable installation instead of the project wheel.")
import depth_surge_3d
import _quality_kd_native

native = Path(_quality_kd_native.__file__)
if not any(native.name.endswith(suffix) for suffix in EXTENSION_SUFFIXES):
    sys.exit(f"Not a compiled extension: {native}")
files = {p.as_posix() for p in dist.files or ()}
for name, module in (("depth_surge_3d/__init__.py", depth_surge_3d), (native.name, _quality_kd_native)):
    actual = Path(module.__file__).resolve()
    if (name not in files or actual != Path(dist.locate_file(name)).resolve()
            or not actual.is_relative_to(Path(sys.prefix).resolve())):
        sys.exit(f"Module does not belong to this environment's project wheel: {actual}")
print(f"Installed Quality extension: {native}")
'@ | & $python -I -
if ($LASTEXITCODE -ne 0) {
    throw 'Could not verify the installed project and Quality extension. See docs/INSTALLATION.md.'
}
Write-Host "Installed CLI: $cli"
Write-Host 'Model weights are separate; see docs/INSTALLATION.md before the first conversion.'
