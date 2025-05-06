#!/bin/bash
# Script to generate requirements.lock file for Docker builds

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "uv installed successfully."
fi

echo "Generating requirements.lock file..."
uv pip compile pyproject.toml -o requirements.lock

echo "Lock file generated successfully."
echo "This lock file will be used by Docker to ensure consistent builds." 