# PowerShell script to generate requirements.lock file for Docker builds

# Check if uv is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is not installed. Installing uv..."
    iwr -useb https://astral.sh/uv/install.ps1 | iex
    # Add uv to PATH for this session
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","User") + ";" + [System.Environment]::GetEnvironmentVariable("Path","Machine")
    Write-Host "uv installed successfully."
}

Write-Host "Generating requirements.lock file..."
uv pip compile pyproject.toml -o requirements.lock

Write-Host "Lock file generated successfully."
Write-Host "This lock file will be used by Docker to ensure consistent builds." 