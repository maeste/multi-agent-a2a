"""
Gradio web interface for the A2A multi-agent system.
"""
import asyncio
import json
import os
from typing import Dict, List, Any, Tuple

import gradio as gr
import httpx
from dotenv import load_dotenv

from common.a2a import A2AClient
from common.types import (
    AgentCard, Artifact, ArtifactType, Task, TaskState
)


class A2AWebUI:
    """Gradio web interface for the A2A multi-agent system."""
    
    def __init__(self):
        """Initialize the web UI."""
        # Load environment variables
        load_dotenv()
        
        # Get agent URLs from environment or use defaults
        self.host_agent_url = os.getenv("HOST_AGENT_URL", "http://localhost:8000")
        self.data_agent_url = os.getenv("DATA_AGENT_URL", "http://localhost:8001")
        self.planning_agent_url = os.getenv("PLANNING_AGENT_URL", "http://localhost:8002")
        self.creative_agent_url = os.getenv("CREATIVE_AGENT_URL", "http://localhost:8003")
        
        # Initialize A2A client
        self.client = A2AClient()
        
        # Store agent capabilities
        self.agent_capabilities = {}
        
        # Initialize Gradio interface
        self.interface = self._create_interface()
        
    def _create_interface(self) -> gr.Blocks:
        """Create the Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title="A2A Multi-Agent System") as interface:
            gr.Markdown("# A2A Multi-Agent System")
            gr.Markdown("This interface demonstrates the capabilities of the A2A protocol.")
            
            with gr.Row():
                with gr.Column():
                    # Input components
                    prompt = gr.Textbox(
                        label="Enter your request", 
                        placeholder="e.g., Analyze this sales data and create a marketing plan",
                        lines=3
                    )
                    file_input = gr.File(
                        label="Upload file (optional)",
                        file_types=["csv", "json", "txt", "xlsx"]
                    )
                    direct_agent = gr.Dropdown(
                        label="Send directly to agent (optional)",
                        choices=["Host (Auto-route)", "Data Analysis", "Planning", "Creative"],
                        value="Host (Auto-route)"
                    )
                    submit_btn = gr.Button("Submit Request")
                    
                with gr.Column():
                    # Status components
                    status = gr.Textbox(label="Status", value="Ready", interactive=False)
                    agents_status = gr.JSON(label="Agents Status", value={})
                    
            # Output components
            with gr.Tab("Response"):
                response = gr.Markdown(label="Response")
                
            with gr.Tab("Artifacts"):
                artifacts_list = gr.JSON(label="Artifacts", value=[])
                selected_artifact = gr.Textbox(label="Selected Artifact ID")
                artifact_content = gr.JSON(label="Artifact Content")
                
            with gr.Tab("Raw Task"):
                task_json = gr.JSON(label="Raw Task Data")
            
            # Agent discovery
            with gr.Tab("Agent Discovery"):
                refresh_btn = gr.Button("Refresh Agent Information")
                agent_info = gr.JSON(label="Agent Information")
            
            # Register event handlers
            submit_btn.click(
                fn=self._handle_submit,
                inputs=[prompt, file_input, direct_agent],
                outputs=[status, response, artifacts_list, task_json]
            )
            
            artifacts_list.change(
                fn=self._select_artifact,
                inputs=[artifacts_list, selected_artifact],
                outputs=[artifact_content]
            )
            
            refresh_btn.click(
                fn=self._discover_agents,
                outputs=[agent_info, agents_status]
            )
            
            # Run agent discovery on startup
            interface.load(
                fn=self._discover_agents,
                outputs=[agent_info, agents_status]
            )
        
        return interface
    
    async def _discover_agents_async(self) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Discover available agents asynchronously.
        
        Returns:
            Tuple of (agent info, agent status)
        """
        agent_urls = {
            "host": self.host_agent_url,
            "data": self.data_agent_url,
            "planning": self.planning_agent_url,
            "creative": self.creative_agent_url
        }
        
        agent_info = {}
        agent_status = {}
        
        for agent_type, url in agent_urls.items():
            try:
                agent_card = await self.client.discover_agent(url)
                agent_info[agent_type] = agent_card.model_dump()
                agent_status[agent_type] = "Available"
                self.agent_capabilities[agent_type] = agent_card
            except Exception as e:
                agent_status[agent_type] = f"Unavailable: {str(e)}"
        
        return agent_info, agent_status
    
    def _discover_agents(self) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Discover available agents.
        
        Returns:
            Tuple of (agent info, agent status)
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(self._discover_agents_async())
        loop.close()
        return result
    
    async def _submit_task_async(
        self, 
        prompt: str, 
        file_data: Any, 
        direct_agent: str
    ) -> Tuple[str, str, List[Dict[str, Any]], Dict[str, Any]]:
        """Submit a task to an agent asynchronously.
        
        Args:
            prompt: User prompt
            file_data: Uploaded file data
            direct_agent: Agent to send the task to
            
        Returns:
            Tuple of (status, response, artifacts, task data)
        """
        # Determine the agent URL to use
        if direct_agent == "Data Analysis":
            agent_url = self.data_agent_url
        elif direct_agent == "Planning":
            agent_url = self.planning_agent_url
        elif direct_agent == "Creative":
            agent_url = self.creative_agent_url
        else:
            # Default to host agent
            agent_url = self.host_agent_url
        
        try:
            # Send task to agent
            task = await self.client.send_task(agent_url, prompt)
            
            # Subscribe to task updates (for streaming)
            updates = []
            async for update in self.client.subscribe_to_task(agent_url, task.id):
                updates.append(update)
            
            # Get final result
            final_task = updates[-1] if updates else task
            
            # Extract response text
            response_text = "No response"
            if (final_task.status.message and 
                final_task.status.message.parts):
                for part in final_task.status.message.parts:
                    if hasattr(part, 'text'):
                        response_text = part.text
                        break
            
            # Extract artifacts
            artifacts = [
                {
                    "id": artifact.id,
                    "type": artifact.type,
                    "name": artifact.name,
                    "description": artifact.description
                }
                for artifact in final_task.artifacts
            ]
            
            # Return results
            status = f"Task {final_task.status.state}"
            return (status, response_text, artifacts, final_task.model_dump())
            
        except Exception as e:
            return (f"Error: {str(e)}", "Failed to process request", [], {})
    
    def _handle_submit(
        self, 
        prompt: str, 
        file_data: Any, 
        direct_agent: str
    ) -> Tuple[str, str, List[Dict[str, Any]], Dict[str, Any]]:
        """Handle form submission.
        
        Args:
            prompt: User prompt
            file_data: Uploaded file data
            direct_agent: Agent to send the task to
            
        Returns:
            Tuple of (status, response, artifacts, task data)
        """
        if not prompt:
            return "Error: No prompt provided", "Please enter a request", [], {}
            
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            self._submit_task_async(prompt, file_data, direct_agent)
        )
        loop.close()
        return result
    
    def _select_artifact(
        self, 
        artifacts: List[Dict[str, Any]], 
        selected_id: str
    ) -> Dict[str, Any]:
        """Handle artifact selection.
        
        Args:
            artifacts: List of artifacts
            selected_id: Selected artifact ID
            
        Returns:
            Artifact content
        """
        for artifact in artifacts:
            if artifact.get("id") == selected_id:
                return artifact
        return {}
    
    def run(self, port: int = 7860):
        """Run the web interface.
        
        Args:
            port: Port to run the server on
        """
        self.interface.launch(server_port=port)


def main():
    """Run the web UI."""
    ui = A2AWebUI()
    ui.run()


if __name__ == "__main__":
    main() 