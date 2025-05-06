#!/bin/bash
# Script to set up the A2A Multi-Agent System using uv package manager

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "uv installed successfully."
fi

# Create and activate a virtual environment
echo "Creating virtual environment with uv..."
uv venv .venv
source .venv/bin/activate

# Install dependencies with uv
echo "Installing dependencies with uv..."
uv pip install -e .

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from .env-ollama template..."
    cp .env-ollama .env
    echo ".env file created."
else
    echo ".env file already exists, skipping creation."
fi

echo ""
echo "Setup complete! You can now run the A2A Multi-Agent System:"
echo "  • Start all agents: python -m main --agent all"
echo "  • Start web UI: python run_ui.py"
echo ""
echo "Be sure to configure your .env file with the appropriate model settings."
echo "See README.md for more details on running individual agents." 