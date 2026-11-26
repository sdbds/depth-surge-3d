# Quick smoke-test script to verify Depth Surge 3D environment setup
Set-Location $PSScriptRoot

function Write-Lime {
    param([string]$Text)
    Write-Host $Text -ForegroundColor ([System.ConsoleColor]::Green)
}

function Write-Cyan {
    param([string]$Text)
    Write-Host $Text -ForegroundColor ([System.ConsoleColor]::Cyan)
}

Write-Cyan "Testing Depth Surge 3D setup..."

if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Lime "Activating virtual environment (.venv)..."
    & .\.venv\Scripts\Activate.ps1
} elseif (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Lime "Activating virtual environment (venv)..."
    & .\venv\Scripts\Activate.ps1
} else {
    Write-Host "Error: Virtual environment not found. Run .\1.install-uv-qinglong.ps1 first." -ForegroundColor Red
    exit 1
}

$failed = 0

Write-Host ""
Write-Cyan "Testing Python imports and native Quality extension..."
python -c @"
import sys
import torch
import cv2
import numpy as np
import flask
import flask_socketio
import depth_surge_3d
import _quality_kd_native

print('OK PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('OK CUDA device:', torch.cuda.get_device_name(0))
print('OK OpenCV:', cv2.__version__, 'NumPy:', np.__version__)
print('OK depth_surge_3d:', depth_surge_3d.__version__)
print('OK _quality_kd_native:', _quality_kd_native.__file__)
"@
if ($LASTEXITCODE -ne 0) { $failed = 1 }

Write-Host ""
foreach ($tool in @("ffmpeg", "ffprobe")) {
    if (Get-Command $tool -ErrorAction SilentlyContinue) {
        Write-Lime "OK $tool is available"
    } else {
        Write-Host "FAIL $tool not found on PATH" -ForegroundColor Red
        $failed = 1
    }
}

Write-Host ""
if ($failed -eq 0) {
    Write-Lime "Setup test passed!"
    Write-Host "Next steps:"
    Write-Host "  Web UI:  .\run_ui.ps1"
    Write-Host "  CLI:     .\.venv\Scripts\depth-surge-3d.exe input_video.mp4"
} else {
    Write-Host "Setup test failed. Check errors above." -ForegroundColor Red
}
exit $failed
