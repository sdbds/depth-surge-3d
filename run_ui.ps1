# Depth Surge 3D - Web UI Launcher for Windows
# Starts the Flask web interface at http://localhost:5000

Set-Location $PSScriptRoot
Write-Host "Starting Depth Surge 3D Web UI..." -ForegroundColor Cyan

# Check if virtual environment exists
$venvPath = $null
if (Test-Path ".venv\Scripts\Activate.ps1") {
    $venvPath = ".venv\Scripts\Activate.ps1"
} elseif (Test-Path "venv\Scripts\Activate.ps1") {
    $venvPath = "venv\Scripts\Activate.ps1"
} else {
    Write-Host "Error: Virtual environment not found. Run .\1.install-uv-qinglong.ps1 first." -ForegroundColor Red
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..."
& $venvPath

# Create output directory
New-Item -ItemType Directory -Force -Path "output" | Out-Null

# Start the Flask application
Write-Host ""
Write-Host "Starting web server at http://localhost:5000" -ForegroundColor Green
Write-Host "Navigate to http://localhost:5000"
Write-Host "Press Ctrl+C to stop the server and all background processes"
Write-Host ""

# Open browser asynchronously after the server starts listening
Start-Job -ScriptBlock {
    Start-Sleep -Seconds 2
    Start-Process "http://localhost:5000"
} | Out-Null

# Start the Flask app
$env:DEPTH_SURGE_UI_SCRIPT = "1"
python app.py @args
exit $LASTEXITCODE
