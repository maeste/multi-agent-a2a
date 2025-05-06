# A2A Multi-Agent System

This project implements a multi-agent system using the Agent2Agent (A2A) protocol, showcasing its capabilities compared to the Message Context Protocol (MCP).

## Features

- Dynamic agent discovery via AgentCards
- Direct agent-to-agent communication
- Rich content types through the A2A Part system
- Task management with full lifecycle
- Streaming and push notifications
- Artifact system for complex outputs

## System Architecture

```
[User Interface (Web App)]
         |
         v
[Host Agent (Orchestrator)]
         |
         v
[A2A Protocol]
     /       \       \
    v         v       v
[Agent 1]  [Agent 2]  [Agent 3]
   MCP        |        |
    |         |        |
[Tools]    [Tools]   [Tools]
```

## Components

1. **Web UI**: Built with Gradio for user interaction
2. **Host Agent**: Built with Google ADK, orchestrates other agents
3. **Specialized Agents**:
   - **Data Analysis Agent**: Built with LangGraph, uses MCP for tool calling
   - **Planning Agent**: Built with CrewAI
   - **Creative Agent**: Custom framework

## Implementation Summary

This implementation demonstrates how A2A and MCP can complement each other in a multi-agent system:

- **A2A Protocol**: Handles agent-to-agent communication, including agent discovery, task management, and artifact sharing.
- **MCP**: Used by the Data Analysis Agent for tool calling, showing how both protocols can work together.

Key aspects demonstrated:

1. **Agent Discovery**: All agents expose AgentCards through a `.well-known/agent.json` endpoint, enabling dynamic discovery. The system demonstrates:
   - Automatic agent discovery at startup
   - Real-time agent status monitoring
   - Dynamic registration of new agents
   - Web UI with agent discovery interface
   - Agent capability and skill discovery

2. **Task Lifecycle**: Tasks flow through states (submitted → working → completed) with proper status updates.
   - Task lifecycle with states (submitted → working → completed) supporting long-running operations.
   - Simulation of processing time in some agents to demonstrate long-running tasks.
3. **Streaming Updates**: Agents support real-time updates via WebSockets.
4. **Artifact System**: Agents produce rich, structured artifacts beyond simple text responses.
5. **Multi-framework Integration**: The system integrates agents built on different frameworks (ADK, LangGraph, CrewAI).

## Setup

### Prerequisites

- Python 3.12+
- API keys for language models (set in `.env` file)
- Ollama (optional, for local LLM support)

### Installation

#### Option 1: Standard Installation with pip

1. Clone the repository
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file from the sample:
   ```
   cp env.sample .env
   ```

#### Option 2: Fast Installation with uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a faster, more reliable Python package installer and resolver. To use uv:

1. Clone the repository
2. Run the setup script:
   ```bash
   # On Linux/macOS:
   chmod +x setup_uv.sh
   ./setup_uv.sh
   
   # On Windows (PowerShell):
   .\setup_uv.ps1
   ```

This will:
- Install uv if not already installed
- Create and activate a virtual environment
- Install all dependencies using uv (much faster than pip)
- Create a `.env` file from the `.env-ollama` template if one doesn't exist

#### Configuring the Environment

Configure your `.env` file:

**For OpenAI models:**
```
MODEL_API_KEY=your_openai_api_key
MODEL_NAME=gpt-4-turbo
# Leave MODEL_BASE_URL commented out
```

**For Ollama:**
```
# MODEL_API_KEY can be blank for Ollama
MODEL_API_KEY=
MODEL_BASE_URL=http://localhost:11434/v1
MODEL_NAME=llama3  # or any other model you have pulled in Ollama

# Optional: Agent-specific model configuration
DATA_AGENT_MODEL=codellama
PLANNING_AGENT_MODEL=llama3:8b
CREATIVE_AGENT_MODEL=mistral
```

You can copy the provided example configuration file for Ollama:
```
cp .env-ollama .env
```

Make sure Ollama is running and the model is available:
```
ollama pull llama3  # Download the model
ollama serve        # Start the Ollama server
```

## Running the System

### Option 1: Running Individual Components

Start each agent separately:

1. **Data Analysis Agent**:
   ```
   python -m main --agent data --host localhost --port 8001
   ```

2. **Planning Agent**:
   ```
   python -m main --agent planning --host localhost --port 8002
   ```

3. **Creative Agent**:
   ```
   python -m main --agent creative --host localhost --port 8003
   ```

4. **Host Agent**:
   ```
   python -m main --agent host --host localhost --port 8000
   ```

5. **Web UI**:
   ```
   python run_ui.py --port 7860
   ```

You can override the model settings from the command line:
```
python -m main --agent data --model llama3 --base-url http://localhost:11434/v1
```

Then visit `http://localhost:7860` to access the web UI.

### Option 2: Running All Agents

To start all agents in development mode (not recommended for production):

```
python -m main --agent all --host localhost
```

This will start all specialized agents on different ports, but not the web UI. You'll need to run the web UI separately:

```
python run_ui.py
```

## Sample Workflows

1. **Data Analysis**: Upload a CSV file and ask "Analyze this sales data for trends"
2. **Planning**: Ask "Create a project plan for launching a new mobile app"
3. **Creative Content**: Request "Write a blog post about renewable energy"
4. **Combined Task**: Try "Analyze this customer data and create a marketing plan based on the findings"

## Project Structure

```
multi-agent-a2a/
├── common/                # Common libraries and utilities
│   ├── a2a/               # A2A client and server libraries
│   └── types.py           # Shared type definitions
├── agents/
│   ├── host_agent/        # Host orchestrator agent (ADK)
│   ├── data_agent/        # Data analysis agent (LangGraph)
│   ├── planning_agent/    # Planning agent (CrewAI)
│   └── creative_agent/    # Creative agent
├── web_ui/                # Gradio-based web UI
├── docker/                # Docker configuration
├── tests/                 # Test suite
├── docker-compose.yml     # Docker Compose configuration
├── main.py                # Entry point for running agents
└── run_ui.py              # Entry point for the web UI
```

## Using Ollama Models

This project fully integrates with Ollama to provide efficient, locally-run language models for all agents. Each agent can use a different model, configured through environment variables.

### Ollama Setup

1. **Install Ollama**: Follow instructions at [Ollama.ai](https://ollama.ai) for your platform
2. **Pull your preferred models**: 
   ```
   ollama pull llama3       # General purpose model
   ollama pull codellama    # Code-focused model
   ollama pull mistral      # Alternative model
   ```
3. **Start the Ollama server**: `ollama serve`

### Agent-Specific Model Configuration

Each agent can use a different model, configured through environment variables:

```
# General model configuration
MODEL_API_KEY=                            # Can be blank for Ollama
MODEL_BASE_URL=http://localhost:11434/v1  # Ollama API endpoint
MODEL_NAME=llama3                         # Default model for all agents

# Agent-specific model overrides
DATA_AGENT_MODEL=codellama                # Data agent uses codellama
PLANNING_AGENT_MODEL=llama3:8b            # Planning agent uses smaller llama3 variant
CREATIVE_AGENT_MODEL=mistral              # Creative agent uses mistral
HOST_AGENT_MODEL=llama3                   # Host agent uses llama3
```

The system prioritizes agent-specific model settings if available, otherwise falls back to the general `MODEL_NAME` setting.

To use a specific Ollama model for all agents:

```
python -m main --agent all --model llama3
```

To use a specific model for a single agent:

```
python -m main --agent data --model codellama
```

Note that Ollama offers various model sizes and specializations. Choose models appropriate for each agent's tasks:
- Data analysis benefits from code-oriented models like CodeLlama
- Planning works well with reasoning-focused models 
- Creative tasks benefit from models with strong writing capabilities

## Development

To contribute to this project, please follow these steps:

1. Create a feature branch
2. Implement your changes
3. Add tests for your feature
4. Run the test suite: `pytest`
5. Submit a pull request 

### Development with uv

For a faster development workflow, we recommend using uv:

1. **Setup your development environment**:
   ```bash
   uv venv .venv
   source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
   uv pip install -e ".[dev]"
   ```

2. **Add dependencies**:
   ```bash
   # Add a runtime dependency
   uv pip add package-name
   
   # Add a development dependency
   uv pip add --dev pytest
   
   # Update pyproject.toml after adding dependencies
   uv pip freeze > requirements.txt
   ```

3. **Run the test suite**:
   ```bash
   uv pip run pytest
   ```

4. **Generate lock file** (optional but recommended for CI):
   ```bash
   uv pip compile pyproject.toml -o requirements.lock
   ```
   
uv provides significantly faster package installation compared to pip, especially with complex dependency trees.

## Building the Project

The A2A Multi-Agent System can be built in several ways depending on your needs.

### Quick Build (Recommended)

Use the provided build scripts for a one-step build process:

```bash
# Linux/macOS - Build Python package (default)
./build.sh

# Windows PowerShell - Build Python package (default)
.\build.ps1
```

These scripts:
- Ensure uv is installed
- Create or activate a virtual environment
- Generate a lock file for reproducible builds
- Build the requested components (Python package)

### Building with uv (Manual Process)

1. **Generate a lock file** for reproducible builds:
   ```bash
   # Linux/macOS
   ./generate_lock.sh
   
   # Windows (PowerShell)
   .\generate_lock.ps1
   ```

2. **Install in development mode**:
   ```bash
   uv pip install -e .
   ```

3. **Build a distributable package**:
   ```bash
   uv pip install build
   python -m build
   ```
   
   This creates:
   - A source distribution (`.tar.gz`) in `dist/`
   - A wheel package (`.whl`) in `dist/`

### Build Artifacts

After building:
- Python package: `dist/a2a-multi-agent-0.1.0.tar.gz` and `dist/a2a_multi_agent-0.1.0-py3-none-any.whl`

For CI/CD pipelines, you can use:
```bash
# Create lock file first
./generate_lock.sh
# Build the packages
uv pip install build
python -m build
```

### Continuous Integration

This project includes a GitHub Actions workflow that:

1. Sets up Python 3.12
2. Installs dependencies using uv
3. Runs linting, type checking, and tests
4. Generates a lock file for reproducible builds
5. Builds Python packages

The workflow is triggered on pushes to main and on pull requests. You can find the workflow configuration in `.github/workflows/build.yml`.

## Development

To contribute to this project, please follow these steps:

1. Create a feature branch
2. Implement your changes
3. Add tests for your feature
4. Run the test suite: `pytest`
5. Submit a pull request 

### Development with uv

For a faster development workflow, we recommend using uv:

1. **Setup your development environment**:
   ```bash
   uv venv .venv
   source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
   uv pip install -e ".[dev]"
   ```

2. **Add dependencies**:
   ```bash
   # Add a runtime dependency
   uv pip add package-name
   
   # Add a development dependency
   uv pip add --dev pytest
   
   # Update pyproject.toml after adding dependencies
   uv pip freeze > requirements.txt
   ```

3. **Run the test suite**:
   ```bash
   uv pip run pytest
   ```

4. **Generate lock file** (optional but recommended for CI):
   ```bash
   uv pip compile pyproject.toml -o requirements.lock
   ```
   
uv provides significantly faster package installation compared to pip, especially with complex dependency trees. 