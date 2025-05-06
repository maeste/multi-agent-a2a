# PowerShell script to set up the A2A Multi-Agent System using uv package manager

# Check if uv is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is not installed. Installing uv..."
    iwr -useb https://astral.sh/uv/install.ps1 | iex
    # Add uv to PATH for this session
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","User") + ";" + [System.Environment]::GetEnvironmentVariable("Path","Machine")
    Write-Host "uv installed successfully."
}

# Create and activate a virtual environment
Write-Host "Creating virtual environment with uv..."
uv venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies with uv
Write-Host "Installing dependencies with uv..."
uv pip install -e .

# Create .env file if it doesn't exist
if (-not (Test-Path .env)) {
    Write-Host "Creating .env file from .env-ollama template..."
    Copy-Item .env-ollama .env
    Write-Host ".env file created."
} else {
    Write-Host ".env file already exists, skipping creation."
}

Write-Host ""
Write-Host "Setup complete! You can now run the A2A Multi-Agent System:"
Write-Host "  • Start all agents: python -m main --agent all"
Write-Host "  • Start web UI: python run_ui.py"
Write-Host ""
Write-Host "Be sure to configure your .env file with the appropriate model settings."
Write-Host "See README.md for more details on running individual agents." 