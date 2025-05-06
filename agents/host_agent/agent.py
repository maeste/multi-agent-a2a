"""
Host Agent (Orchestrator) using Google ADK.
"""
import asyncio
import os
import uuid
from typing import Dict, List, Optional, Any

from common.a2a import A2ABaseServer, A2AClient
from common.types import (
    AgentCard, Artifact, ArtifactType, Capabilities, Message, 
    Skill, Task, TaskState, TaskStatus, TextPart
)


class HostAgent(A2ABaseServer):
    """Host Agent that orchestrates other specialized agents.
    
    This agent acts as the central coordinator, receiving user requests
    and delegating to specialized agents based on the request type.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        host: str = "localhost",
        port: int = 8000,
        data_agent_url: str = "http://localhost:8001",
        planning_agent_url: str = "http://localhost:8002",
        creative_agent_url: str = "http://localhost:8003"
    ):
        """Initialize the Host Agent.
        
        Args:
            api_key: API key for the ADK model
            host: Host to bind to
            port: Port to bind to
            data_agent_url: URL of the Data Analysis Agent
            planning_agent_url: URL of the Planning Agent
            creative_agent_url: URL of the Creative Agent
        """
        # Create AgentCard
        agent_card = AgentCard(
            name="Host Agent",
            description="Orchestrates specialized agents and manages conversation flow",
            url=f"http://{host}:{port}",
            version="1.0.0",
            capabilities=Capabilities(
                streaming=True,
                pushNotifications=True
            ),
            defaultInputModes=["text"],
            defaultOutputModes=["text"],
            skills=[
                Skill(
                    id="orchestration",
                    name="Agent Orchestration",
                    description="Delegates tasks to specialized agents"
                ),
                Skill(
                    id="conversation",
                    name="Conversation Management",
                    description="Manages multi-turn conversations"
                )
            ]
        )
        
        # Initialize A2A server
        super().__init__(agent_card=agent_card)
        
        # Save configuration
        self.api_key = api_key
        self.host = host
        self.port = port
        
        # Store agent URLs
        self.agent_urls = {
            "data": data_agent_url,
            "planning": planning_agent_url,
            "creative": creative_agent_url
        }
        
        # Initialize A2A client for communicating with other agents
        self.client = A2AClient()
        
        # Store discovered agent capabilities
        self.agent_capabilities = {}
        
    async def startup(self):
        """Discover available agents on startup."""
        for agent_type, url in self.agent_urls.items():
            try:
                agent_card = await self.client.discover_agent(url)
                self.agent_capabilities[agent_type] = agent_card
                print(f"Discovered {agent_type} agent at {url}")
            except Exception as e:
                print(f"Error discovering {agent_type} agent: {e}")
    
    async def handle_task(self, task: Task) -> Task:
        """Handle an orchestration task.
        
        This method processes user requests, determines which specialized 
        agents to call, and consolidates their responses.
        
        Args:
            task: Task to handle
            
        Returns:
            Updated task with results
        """
        # Extract message text
        message_text = "No input provided"
        print(f"Task: {task}")
        if task.status.message and task.status.message.parts:
            for part in task.status.message.parts:
                if hasattr(part, 'text'):
                    message_text = part.text
                    break
        
        # Analyze request to determine which agents to call
        agents_to_call = await self._analyze_request(message_text)
        
        # Call each agent and collect results
        results = []
        for agent_type in agents_to_call:
            if agent_type in self.agent_urls:
                agent_result = await self._call_agent(
                    agent_type, 
                    message_text
                )
                results.append(agent_result)
        
        # Consolidate results
        consolidated_result = await self._consolidate_results(
            message_text, 
            results
        )
        
        # Update task with consolidated results
        task.status.state = TaskState.COMPLETED
        task.status.message = Message(parts=[
            TextPart(text=consolidated_result["response"])
        ])
        
        # Add artifacts from all agents
        for result in results:
            task.artifacts.extend(result.get("artifacts", []))
        
        return task
    
    async def _analyze_request(self, message: str) -> List[str]:
        """Analyze a user request to determine which agents to call.
        
        Args:
            message: User message
            
        Returns:
            List of agent types to call
        """
        # In a real implementation, this would use ADK to analyze the request
        # For now, use a simple keyword-based approach
        
        agents = []
        
        # Check for data analysis keywords
        if any(kw in message.lower() for kw in ["data", "analyze", "statistics", "csv", "excel", "json"]):
            agents.append("data")
            
        # Check for planning keywords
        if any(kw in message.lower() for kw in ["plan", "schedule", "task", "timeline", "project"]):
            agents.append("planning")
            
        # Check for creative keywords
        if any(kw in message.lower() for kw in ["create", "generate", "write", "story", "content"]):
            agents.append("creative")
            
        # If no specific agent is identified, use all of them
        if not agents:
            agents = ["data", "planning", "creative"]
            
        return agents
    
    async def _call_agent(self, agent_type: str, message: str) -> Dict[str, Any]:
        """Call a specialized agent.
        
        Args:
            agent_type: Type of agent to call
            message: Message to send to the agent
            
        Returns:
            Agent response
        """
        agent_url = self.agent_urls.get(agent_type)
        if not agent_url:
            return {
                "agent_type": agent_type,
                "success": False,
                "response": f"Agent {agent_type} not available",
                "artifacts": []
            }
        
        try:
            # Send task to agent
            task = await self.client.send_task(agent_url, message)
            
            # Subscribe to task updates (for streaming)
            responses = []
            async for update in self.client.subscribe_to_task(agent_url, task.id):
                # In real implementation, would process streaming updates
                responses.append(update)
            
            # Get final result
            final_response = responses[-1] if responses else task
            
            # Extract response text
            response_text = "No response"
            if (final_response.status.message and 
                final_response.status.message.parts):
                for part in final_response.status.message.parts:
                    if hasattr(part, 'text'):
                        response_text = part.text
                        break
            
            return {
                "agent_type": agent_type,
                "success": final_response.status.state == TaskState.COMPLETED,
                "response": response_text,
                "artifacts": final_response.artifacts
            }
            
        except Exception as e:
            return {
                "agent_type": agent_type,
                "success": False,
                "response": f"Error calling {agent_type} agent: {str(e)}",
                "artifacts": []
            }
    
    async def _consolidate_results(
        self, 
        original_message: str, 
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Consolidate results from multiple agents.
        
        Args:
            original_message: Original user message
            results: Results from specialized agents
            
        Returns:
            Consolidated response
        """
        # In a real implementation, this would use ADK to create a coherent response
        # For now, use a simple template-based approach
        
        consolidated_text = "Here's what I found:\n\n"
        
        for result in results:
            agent_type = result.get("agent_type", "unknown")
            success = result.get("success", False)
            response = result.get("response", "No response")
            artifacts = result.get("artifacts", [])
            
            if success:
                consolidated_text += f"**{agent_type.capitalize()} Agent**:\n{response}\n\n"
                
                if artifacts:
                    consolidated_text += f"*{len(artifacts)} artifacts produced*\n\n"
            else:
                consolidated_text += f"**{agent_type.capitalize()} Agent**: Unable to complete task\n\n"
        
        return {
            "response": consolidated_text.strip()
        }
    
    def run(self):
        """Run the agent server."""
        import uvicorn
        
        # Start agent discovery
        event_loop = asyncio.get_event_loop()
        event_loop.run_until_complete(self.startup())
        
        # Run server
        uvicorn.run(self.app, host=self.host, port=self.port) 