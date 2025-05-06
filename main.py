"""
Main entry point for the A2A multi-agent system.
"""
import argparse
import asyncio
import os
from dotenv import load_dotenv

from agents.data_agent import DataAnalysisAgent
from agents.planning_agent import PlanningAgent
from agents.creative_agent import CreativeAgent
from agents.host_agent import HostAgent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="A2A Multi-Agent System")
    parser.add_argument(
        "--agent",
        choices=["data", "planning", "creative", "host", "all"],
        default="data",
        help="Agent to run (default: data)"
    )
    parser.add_argument(
        "--host", 
        default="localhost", 
        help="Host to bind to (default: localhost)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--model", 
        default=None, 
        help="Model to use (defaults to MODEL_NAME from .env)"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for OpenAI-compatible API (defaults to MODEL_BASE_URL from .env)"
    )
    return parser.parse_args()


def main():
    """Run the A2A multi-agent system."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    args = parse_args()
    
    # Get API keys and model configuration from environment
    model_api_key = os.getenv("MODEL_API_KEY")
    model_base_url = args.base_url or os.getenv("MODEL_BASE_URL")
    model_name = args.model or os.getenv("MODEL_NAME", "gpt-4-turbo")
    
    # Get agent-specific model names from environment
    data_agent_model = args.model or os.getenv("DATA_AGENT_MODEL", model_name)
    planning_agent_model = args.model or os.getenv("PLANNING_AGENT_MODEL", model_name)
    creative_agent_model = args.model or os.getenv("CREATIVE_AGENT_MODEL", model_name)
    host_agent_model = args.model or os.getenv("HOST_AGENT_MODEL", model_name)
    
    # Get agent URLs from environment or use defaults
    data_agent_url = os.getenv("DATA_AGENT_URL", "http://localhost:8001")
    planning_agent_url = os.getenv("PLANNING_AGENT_URL", "http://localhost:8002")
    creative_agent_url = os.getenv("CREATIVE_AGENT_URL", "http://localhost:8003")
    host_agent_url = os.getenv("HOST_AGENT_URL", "http://localhost:8000")
    
    # Run the requested agent
    if args.agent == "data" or args.agent == "all":
        # Run the Data Analysis Agent
        port = 8001 if args.agent == "all" else args.port
        print(f"Starting Data Analysis Agent on {args.host}:{port}")
        print(f"Using model {data_agent_model} with base URL {model_base_url}")
        data_agent = DataAnalysisAgent(
            model_name=data_agent_model,
            api_key=model_api_key,
            host=args.host,
            port=port
        )
        data_agent.run()
        
    elif args.agent == "planning":
        # Run the Planning Agent
        port = 8002 if args.agent == "planning" else args.port
        print(f"Starting Planning Agent on {args.host}:{port}")
        print(f"Using model {planning_agent_model} with base URL {model_base_url}")
        planning_agent = PlanningAgent(
            model_name=planning_agent_model,
            api_key=model_api_key,
            host=args.host,
            port=port
        )
        planning_agent.run()
        
    elif args.agent == "creative":
        # Run the Creative Agent
        port = 8003 if args.agent == "creative" else args.port
        print(f"Starting Creative Agent on {args.host}:{port}")
        print(f"Using model {creative_agent_model} with base URL {model_base_url}")
        creative_agent = CreativeAgent(
            model_name=creative_agent_model,
            api_key=model_api_key,
            host=args.host,
            port=port
        )
        creative_agent.run()
        
    elif args.agent == "host":
        # Run the Host Agent
        print(f"Starting Host Agent on {args.host}:{args.port}")
        print(f"Using model {host_agent_model} with base URL {model_base_url}")
        host_agent = HostAgent(
            api_key=model_api_key,
            host=args.host,
            port=args.port,
            data_agent_url=data_agent_url,
            planning_agent_url=planning_agent_url,
            creative_agent_url=creative_agent_url
        )
        host_agent.run()
        
    else:
        print(f"Agent type '{args.agent}' not recognized")


if __name__ == "__main__":
    main() 