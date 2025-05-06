"""
Data Analysis Agent using LangGraph and MCP for tool calling.
"""
import asyncio
import json
import os
import uuid
from typing import Any, Dict, List, Optional, TypedDict, Annotated
import operator

import pandas as pd
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from common.a2a import A2ABaseServer
from common.types import (
    AgentCard, Artifact, ArtifactType, Capabilities, Message, 
    Skill, Task, TaskState, TaskStatus, TextPart
)

class AgentState(TypedDict):
    """State for the agent."""
    messages: Annotated[list, operator.add]

class DataAnalysisAgent(A2ABaseServer):
    """Data Analysis Agent that uses MCP for tool calling.
    
    This agent demonstrates how A2A can be used alongside MCP, with
    A2A handling agent-to-agent communication and MCP handling tool calls.
    """
    
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        host: str = "localhost",
        port: int = 8001
    ):
        """Initialize the Data Analysis Agent.
        
        Args:
            model_name: Name of the LLM to use
            api_key: API key for the LLM
            host: Host to bind to
            port: Port to bind to
        """
        # Create AgentCard
        agent_card = AgentCard(
            name="Data Analysis Agent",
            description="Processes and analyzes data files",
            url=f"http://{host}:{port}",
            version="1.0.0",
            capabilities=Capabilities(
                streaming=True,
                pushNotifications=True
            ),
            defaultInputModes=["text", "file"],
            defaultOutputModes=["text", "data"],
            skills=[
                Skill(
                    id="data_analysis",
                    name="Data Analysis",
                    description="Analyzes structured data files"
                ),
                Skill(
                    id="visualization",
                    name="Data Visualization",
                    description="Creates visual representations of data"
                )
            ]
        )
        
        # Initialize A2A server
        super().__init__(agent_card=agent_card)
        
        # Save configuration
        self.model_name = model_name
        self.api_key = api_key
        self.host = host
        self.port = port
        
        # Set up LangGraph for MCP tool calling
        self.tools = self._create_tools()
        self.graph = self._create_graph()
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for data analysis.
        
        Returns:
            List of tools for the agent to use
        """
        tools = [
            Tool.from_function(
                func=self._load_csv,
                name="load_csv",
                description="Load a CSV file for analysis"
            ),
            Tool.from_function(
                func=self._load_json,
                name="load_json",
                description="Load a JSON file for analysis"
            ),
            Tool.from_function(
                func=self._analyze_data,
                name="analyze_data",
                description="Analyze loaded data"
            ),
            Tool.from_function(
                func=self._visualize_data,
                name="visualize_data",
                description="Create visualization from data"
            )
        ]
        return tools
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph for tool calling.
        
        Returns:
            LangGraph for handling tool calls
        """
        # Define the state schema
        # The state will have a 'messages' key, which is a list.
        # When the state is updated, new items for 'messages' will be appended (concatenated).
        workflow = StateGraph(AgentState)

        # Create tool node
        tool_node = ToolNode(tools=self.tools)
        
        # Define nodes
        workflow.add_node("tools", tool_node)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "tools",
            tools_condition,
            {
                "continue": "tools",  # Continue using tools if needed
                "complete": END  # Exit when done
            }
        )
        
        # Set entry point
        workflow.set_entry_point("tools")
        
        return workflow.compile()
    
    async def handle_task(self, task: Task) -> Task:
        """Handle a data analysis task.
        
        Args:
            task: Task to handle
            
        Returns:
            Updated task with results
        """
        # Extract message
        message_text = "No input provided"
        if task.status.message and task.status.message.parts:
            for part in task.status.message.parts:
                if hasattr(part, 'text'):
                    message_text = part.text
                    break
        
        # Set up task context
        task_data = {
            "task_id": task.id,
            "input": message_text,
            "files": [],  # Would extract file parts here
            "results": []
        }
        
        # Process with LangGraph (simulating tool calls via MCP)
        # In a real implementation, we would use the model's generate_content with tools
        result = await self._process_with_mcp(task_data)
        
        # Create response message
        response_text = result.get("output", "Analysis complete")
        response_message = Message(parts=[TextPart(text=response_text)])
        
        # Create artifacts
        artifacts = []
        for r in result.get("results", []):
            artifact_id = str(uuid.uuid4())
            artifact = Artifact(
                id=artifact_id,
                type=ArtifactType.DATA,
                name=r.get("name", "Analysis Result"),
                description=r.get("description", ""),
                content=r.get("data", {})
            )
            artifacts.append(artifact)
        
        # Update task with results
        task.status.state = TaskState.COMPLETED
        task.status.message = response_message
        task.artifacts = artifacts
        
        return task
    
    async def _process_with_mcp(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the task using MCP for tool calling.
        
        In a real implementation, this would use the model's generate_content with tools.
        This simulates the process for demonstration purposes.
        
        Args:
            task_data: Task data including input and files
            
        Returns:
            Processing results
        """
        # For demonstration, simulate a tool calling session
        # In real implementation, this would use the model and MCP
        
        # Simulate invoking tools based on the input
        state = {"messages": [{"role": "user", "content": task_data["input"]}]}
        
        # Run the graph
        for chunk in self.graph.stream(state):
            # This would normally process streaming updates
            await asyncio.sleep(0.1)  # Simulate processing time
        
        # For this example, just return a simulated result
        # In real implementation, this would come from model responses and tool results
        return {
            "output": f"Analysis complete. Found insights in the data.",
            "results": [
                {
                    "name": "Summary Statistics",
                    "description": "Basic statistical measures of the dataset",
                    "data": {
                        "count": 100,
                        "mean": 42.5,
                        "median": 40.2,
                        "std": 15.7
                    }
                },
                {
                    "name": "Trend Analysis",
                    "description": "Identified trends in the data",
                    "data": {
                        "trend": "upward",
                        "growth_rate": 0.15,
                        "seasonality": "quarterly"
                    }
                }
            ]
        }
    
    def _load_csv(self, file_path: str) -> str:
        """Load a CSV file for analysis.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Confirmation message
        """
        # In a real implementation, this would load the file
        return f"Loaded CSV file {file_path}"
    
    def _load_json(self, file_path: str) -> str:
        """Load a JSON file for analysis.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Confirmation message
        """
        # In a real implementation, this would load the file
        return f"Loaded JSON file {file_path}"
    
    def _analyze_data(self, analysis_type: str) -> Dict[str, Any]:
        """Analyze loaded data.
        
        Args:
            analysis_type: Type of analysis to perform
            
        Returns:
            Analysis results
        """
        # In a real implementation, this would perform actual analysis
        return {
            "analysis_type": analysis_type,
            "result": {
                "count": 100,
                "mean": 42.5,
                "median": 40.2,
                "std": 15.7
            }
        }
    
    def _visualize_data(self, visualization_type: str) -> str:
        """Create a visualization of the data.
        
        Args:
            visualization_type: Type of visualization to create
            
        Returns:
            Path to the created visualization
        """
        # In a real implementation, this would create a visualization
        return f"Created {visualization_type} visualization"
    
    def run(self):
        """Run the agent server."""
        import uvicorn
        uvicorn.run(self.app, host=self.host, port=self.port) 