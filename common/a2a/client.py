"""
A2A client for agent-to-agent communication.
"""
import asyncio
import json
import uuid
from typing import AsyncIterator, Dict, List, Optional

import httpx
from websockets.client import connect

from ..types import AgentCard, Artifact, Message, Part, Task, TaskState, TaskStatus, TextPart


class A2AClient:
    """Client for interacting with A2A agents."""

    def __init__(self, timeout: int = 60):
        """Initialize the A2A client.
        
        Args:
            timeout: HTTP request timeout in seconds
        """
        self.timeout = timeout
        self.http_client = httpx.AsyncClient(timeout=timeout)
    
    async def close(self):
        """Close the client and release resources."""
        await self.http_client.aclose()
    
    async def discover_agent(self, agent_url: str) -> AgentCard:
        """Discover an agent's capabilities by fetching its AgentCard.
        
        Args:
            agent_url: Base URL of the agent
            
        Returns:
            AgentCard describing the agent's capabilities
        """
        well_known_url = f"{agent_url.rstrip('/')}/.well-known/agent.json"
        response = await self.http_client.get(well_known_url)
        response.raise_for_status()
        return AgentCard.model_validate(response.json())
    
    async def send_task(self, agent_url: str, text: str) -> Task:
        """Send a new task to an agent.
        
        Args:
            agent_url: Base URL of the agent
            text: Text content of the task
            
        Returns:
            The created task
        """
        task_id = str(uuid.uuid4())
        print(f"Sending task to {agent_url} with text: {text}")
        message = Message(parts=[TextPart(text=text)])
        print(f"Message: {message}")
        print(f"Message JSON: {message.model_dump_json()}")
        status = TaskStatus(state=TaskState.SUBMITTED, message=message)
        print(f"Status: {status}")
        task = Task(id=task_id, status=status)
        print(f"Task: {task}")
        task_url = f"{agent_url.rstrip('/')}/tasks"
        print(f"Task URL: {task_url}")
        print(f"Task JSON: {task.model_dump_json()}")
        response = await self.http_client.post(
            task_url,
            json=task.model_dump(mode='json')
        )
        print(f"Response: {response}")
        response.raise_for_status()
        return Task.model_validate(response.json())
    
    async def get_task(self, agent_url: str, task_id: str) -> Task:
        """Get the current status of a task.
        
        Args:
            agent_url: Base URL of the agent
            task_id: ID of the task to get
            
        Returns:
            Current task status
        """
        task_url = f"{agent_url.rstrip('/')}/tasks/{task_id}"
        response = await self.http_client.get(task_url)
        response.raise_for_status()
        return Task.model_validate(response.json())
    
    async def cancel_task(self, agent_url: str, task_id: str) -> Task:
        """Cancel a task.
        
        Args:
            agent_url: Base URL of the agent
            task_id: ID of the task to cancel
            
        Returns:
            Updated task with cancelled status
        """
        task_url = f"{agent_url.rstrip('/')}/tasks/{task_id}"
        response = await self.http_client.delete(task_url)
        response.raise_for_status()
        return Task.model_validate(response.json())
    
    async def update_task(self, agent_url: str, task: Task) -> Task:
        """Update a task (e.g., provide input when in input-required state).
        
        Args:
            agent_url: Base URL of the agent
            task: Updated task data
            
        Returns:
            Updated task status
        """
        task_url = f"{agent_url.rstrip('/')}/tasks/{task.id}"
        response = await self.http_client.put(
            task_url, 
            json=task.model_dump(mode='json')
        )
        response.raise_for_status()
        return Task.model_validate(response.json())
    
    async def subscribe_to_task(self, agent_url: str, task_id: str) -> AsyncIterator[Task]:
        """Subscribe to real-time updates for a task.
        
        Args:
            agent_url: Base URL of the agent
            task_id: ID of the task to subscribe to
            
        Yields:
            Task updates as they occur
        """
        ws_url = f"{agent_url.rstrip('/')}/tasks/{task_id}/subscribe".replace("http", "ws")
        
        async with connect(ws_url) as websocket:
            while True:
                try:
                    message = await websocket.recv()
                    task_update = Task.model_validate(json.loads(message))
                    yield task_update
                    
                    # If task reached terminal state, exit
                    if task_update.status.state in [
                        TaskState.COMPLETED,
                        TaskState.FAILED,
                        TaskState.CANCELLED
                    ]:
                        break
                        
                except asyncio.CancelledError:
                    # Handle client-side cancellation
                    break 