#!/bin/bash
# Convenience script to build the A2A Multi-Agent System

set -e  # Exit on error

# Ensure we have uv installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Parse command line arguments
BUILD_TYPE="package"
if [ $# -ge 1 ]; then
    BUILD_TYPE=$1
fi

# Activate virtual environment if it exists, otherwise create it
if [ -d ".venv" ]; then
    echo "Activating existing virtual environment..."
    source .venv/bin/activate
else
    echo "Creating new virtual environment..."
    uv venv .venv
    source .venv/bin/activate
    uv pip install -e ".[dev]"
fi

# Generate lock file
echo "Generating lock file..."
uv pip compile pyproject.toml -o requirements.lock

case $BUILD_TYPE in
    "package")
        echo "Building Python package..."
        uv pip install build
        python -m build
        echo "Build completed. Packages available in dist/"
        ;;
    *)
        echo "Unknown build type: $BUILD_TYPE"
        echo "Usage: ./build.sh [package]"
        exit 1
        ;;
esac 