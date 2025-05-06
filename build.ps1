# PowerShell script to build the A2A Multi-Agent System

# Stop on first error
$ErrorActionPreference = "Stop"

# Ensure we have uv installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    iwr -useb https://astral.sh/uv/install.ps1 | iex
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","User") + ";" + [System.Environment]::GetEnvironmentVariable("Path","Machine")
}

# Parse command line arguments
$BuildType = "package"
if ($args.Count -ge 1) {
    $BuildType = $args[0]
}

# Activate virtual environment if it exists, otherwise create it
if (Test-Path ".venv") {
    Write-Host "Activating existing virtual environment..."
    .\.venv\Scripts\Activate.ps1
} else {
    Write-Host "Creating new virtual environment..."
    uv venv .venv
    .\.venv\Scripts\Activate.ps1
    uv pip install -e ".[dev]"
}

# Generate lock file
Write-Host "Generating lock file..."
uv pip compile pyproject.toml -o requirements.lock

switch ($BuildType) {
    "package" {
        Write-Host "Building Python package..."
        uv pip install build
        python -m build
        Write-Host "Build completed. Packages available in dist/"
    }
    default {
        Write-Host "Unknown build type: $BuildType"
        Write-Host "Usage: .\build.ps1 [package]"
        exit 1
    }
} 