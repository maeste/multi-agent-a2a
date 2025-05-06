"""
A2A server base class for implementing A2A agents.
"""
import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..types import AgentCard, Artifact, Message, Task, TaskState, TaskStatus, TextPart


class TaskUpdate(BaseModel):
    """An update to a task."""
    task_id: str
    task: Task


class A2ABaseServer(ABC):
    """Base class for A2A server implementations."""

    def __init__(
        self,
        agent_card: AgentCard,
        app: Optional[FastAPI] = None,
    ):
        """Initialize the A2A server.
        
        Args:
            agent_card: AgentCard describing this agent
            app: Optional FastAPI application to use
        """
        self.agent_card = agent_card
        self.app = app or FastAPI(title=agent_card.name)
        self.tasks: Dict[str, Task] = {}
        self.task_subscribers: Dict[str, Set[WebSocket]] = {}
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register routes for the A2A server."""
        
        # AgentCard discovery
        @self.app.get("/.well-known/agent.json")
        async def get_agent_card():
            return self.agent_card.model_dump()
        
        # Task management
        @self.app.post("/tasks")
        async def create_task(task: Dict[str, Any]):
            print(f"Task: {task}")
            task_obj = Task.model_validate(task)
            print(f"Task object: {task_obj}")
            # Validate task state
            if task_obj.status.state != TaskState.SUBMITTED:
                raise HTTPException(
                    status_code=400, 
                    detail="New tasks must have 'submitted' state"
                )
            
            # Store task
            self.tasks[task_obj.id] = task_obj
            
            # Start processing task in background
            task_obj = await self._update_task_state(
                task_obj, 
                TaskState.WORKING, 
                task_obj.status.message
            )
            asyncio.create_task(self._process_task(task_obj))
            
            return task_obj.model_dump()
        
        @self.app.get("/tasks/{task_id}")
        async def get_task(task_id: str):
            if task_id not in self.tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            return self.tasks[task_id].model_dump()
        
        @self.app.put("/tasks/{task_id}")
        async def update_task(task_id: str, task: Dict[str, Any]):
            if task_id not in self.tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            
            updated_task = Task.model_validate(task)
            
            # Validate task state transition
            current_task = self.tasks[task_id]
            if current_task.status.state != TaskState.INPUT_REQUIRED:
                raise HTTPException(
                    status_code=400,
                    detail="Only tasks in 'input-required' state can be updated"
                )
            
            # Update task
            self.tasks[task_id] = updated_task
            updated_task = await self._update_task_state(
                updated_task, 
                TaskState.WORKING, 
                updated_task.status.message
            )
            
            # Continue processing
            asyncio.create_task(self._process_task(updated_task))
            
            return updated_task.model_dump()
        
        @self.app.delete("/tasks/{task_id}")
        async def cancel_task(task_id: str):
            if task_id not in self.tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            
            task = self.tasks[task_id]
            task = await self._update_task_state(
                task, 
                TaskState.CANCELLED, 
                task.status.message,
                "Task cancelled by client"
            )
            return task.model_dump()
        
        @self.app.websocket("/tasks/{task_id}/subscribe")
        async def subscribe_to_task(websocket: WebSocket, task_id: str):
            if task_id not in self.tasks:
                await websocket.close(code=1000, reason="Task not found")
                return
            
            await websocket.accept()
            
            # Add to subscribers
            if task_id not in self.task_subscribers:
                self.task_subscribers[task_id] = set()
            self.task_subscribers[task_id].add(websocket)
            
            try:
                # Send current state immediately
                await websocket.send_text(self.tasks[task_id].model_dump_json())
                
                # Keep connection open until client disconnects
                while True:
                    # This will block until client sends something or disconnects
                    # Typically clients don't send anything, they just listen
                    await websocket.receive_text()
            except WebSocketDisconnect:
                # Remove from subscribers
                if task_id in self.task_subscribers:
                    self.task_subscribers[task_id].discard(websocket)
                    if not self.task_subscribers[task_id]:
                        del self.task_subscribers[task_id]
    
    async def _notify_subscribers(self, task_id: str, task: Task):
        """Notify all subscribers about a task update.
        
        Args:
            task_id: ID of the task that was updated
            task: Updated task data
        """
        if task_id not in self.task_subscribers:
            return
        
        dead_subscribers = set()
        
        # Create task update JSON
        task_json = task.model_dump_json()
        
        # Send update to all subscribers
        for websocket in self.task_subscribers[task_id]:
            try:
                await websocket.send_text(task_json)
            except RuntimeError:
                # Websocket might be closed
                dead_subscribers.add(websocket)
        
        # Remove dead subscribers
        self.task_subscribers[task_id] -= dead_subscribers
        if not self.task_subscribers[task_id]:
            del self.task_subscribers[task_id]
    
    async def _update_task_state(
        self, 
        task: Task, 
        state: TaskState, 
        message: Optional[Message] = None,
        reason: Optional[str] = None
    ) -> Task:
        """Update a task's state and notify subscribers.
        
        Args:
            task: Task to update
            state: New state
            message: Optional new message
            reason: Optional reason for state change
            
        Returns:
            Updated task
        """
        task.status.state = state
        if message:
            task.status.message = message
        if reason:
            task.status.reason = reason
        
        # Update stored task
        self.tasks[task.id] = task
        
        # Notify subscribers
        await self._notify_subscribers(task.id, task)
        
        return task
    
    async def _process_task(self, task: Task):
        """Process a task in the background.
        
        This method calls the abstract handle_task method and handles
        errors and state transitions.
        
        Args:
            task: Task to process
        """
        try:
            result_task = await self.handle_task(task)
            if result_task.status.state not in [
                TaskState.COMPLETED, 
                TaskState.FAILED,
                TaskState.CANCELLED,
                TaskState.INPUT_REQUIRED
            ]:
                # If handler didn't set a terminal state, set completed
                await self._update_task_state(
                    result_task,
                    TaskState.COMPLETED,
                    result_task.status.message
                )
            else:
                # Make sure any state changes are saved and subscribers notified
                await self._update_task_state(
                    result_task,
                    result_task.status.state,
                    result_task.status.message,
                    result_task.status.reason
                )
        except Exception as e:
            # Handle any unhandled exceptions
            error_message = Message(
                parts=[TextPart(text=f"Error processing task: {str(e)}")]
            )
            await self._update_task_state(
                task,
                TaskState.FAILED,
                error_message,
                str(e)
            )
    
    @abstractmethod
    async def handle_task(self, task: Task) -> Task:
        """Handle a task.
        
        This method should be implemented by subclasses to process tasks.
        It should return the updated task with appropriate state and artifacts.
        
        Args:
            task: Task to handle
            
        Returns:
            Updated task with results
        """
        pass 