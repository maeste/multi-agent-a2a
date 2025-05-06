"""
Creative Agent for content generation.
"""
import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional

from common.a2a import A2ABaseServer
from common.types import (
    AgentCard, Artifact, ArtifactType, Capabilities, Message, 
    Skill, Task, TaskState, TaskStatus, TextPart
)


class CreativeAgent(A2ABaseServer):
    """Creative Agent that generates creative content.
    
    This agent specializes in generating creative text content,
    stories, and formatted output.
    """
    
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        host: str = "localhost",
        port: int = 8003
    ):
        """Initialize the Creative Agent.
        
        Args:
            model_name: Name of the LLM to use
            api_key: API key for the LLM
            host: Host to bind to
            port: Port to bind to
        """
        # Create AgentCard
        agent_card = AgentCard(
            name="Creative Agent",
            description="Generates creative content based on prompts",
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
                    id="text_generation",
                    name="Text Generation",
                    description="Generates creative text content"
                ),
                Skill(
                    id="story_creation",
                    name="Story Creation",
                    description="Creates compelling stories"
                ),
                Skill(
                    id="content_formatting",
                    name="Content Formatting",
                    description="Formats content for various purposes"
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
        
        # Configure content types and templates
        self.content_types = {
            "blog": {
                "sections": ["introduction", "body", "conclusion"],
                "word_count": 500
            },
            "story": {
                "sections": ["setting", "characters", "plot", "resolution"],
                "word_count": 1000
            },
            "social": {
                "sections": ["headline", "body", "call_to_action"],
                "word_count": 200
            },
            "email": {
                "sections": ["greeting", "body", "closing"],
                "word_count": 300
            }
        }
    
    async def handle_task(self, task: Task) -> Task:
        """Handle a creative content generation task.
        
        Args:
            task: Task to handle
            
        Returns:
            Updated task with results
        """
        # Extract message
        message_text = "No input provided"
        print(f"Task: {task}")
        if task.status.message and task.status.message.parts:
            for part in task.status.message.parts:
                if hasattr(part, 'text'):
                    message_text = part.text
                    break
        
        # Determine content type based on message
        content_type = self._determine_content_type(message_text)
        
        # Generate creative content
        content = await self._generate_content(message_text, content_type)
        
        # Create response message
        response_text = content.get("summary", "Content generation complete")
        response_message = Message(parts=[TextPart(text=response_text)])
        
        # Create artifact for the main content
        artifact_id = str(uuid.uuid4())
        artifact = Artifact(
            id=artifact_id,
            type=ArtifactType.DOCUMENT,
            name=content.get("title", "Generated Content"),
            description=content.get("description", "Creative content based on your prompt"),
            content={
                "content_type": content_type,
                "full_text": content.get("full_text", ""),
                "sections": content.get("sections", {})
            }
        )
        
        # Update task with results
        task.status.state = TaskState.COMPLETED
        task.status.message = response_message
        task.artifacts = [artifact]
        
        return task
    
    def _determine_content_type(self, message: str) -> str:
        """Determine the type of content to generate based on the message.
        
        Args:
            message: User message
            
        Returns:
            Content type
        """
        message_lower = message.lower()
        print(f"Message: {message_lower}")
        
        if any(kw in message_lower for kw in ["blog", "article", "post"]):
            return "blog"
        elif any(kw in message_lower for kw in ["story", "narrative", "tale"]):
            return "story"
        elif any(kw in message_lower for kw in ["social", "twitter", "facebook", "instagram"]):
            return "social"
        elif any(kw in message_lower for kw in ["email", "message", "communication"]):
            return "email"
        else:
            # Default to blog
            return "blog"
    
    async def _generate_content(
        self, 
        prompt: str, 
        content_type: str
    ) -> Dict[str, Any]:
        """Generate creative content based on the prompt and content type.
        
        Args:
            prompt: User prompt
            content_type: Type of content to generate
            
        Returns:
            Generated content
        """
        # In a real implementation, would use an LLM to generate content
        # For now, return a placeholder
        
        # Get content structure based on type
        template = self.content_types.get(content_type, self.content_types["blog"])
        sections = template["sections"]
        
        # Simulate different content for different types
        if content_type == "blog":
            title = "The Future of AI: Opportunities and Challenges"
            description = "A blog post exploring the future landscape of artificial intelligence"
            full_text = (
                "Artificial Intelligence has evolved rapidly in recent years, transforming "
                "industries and reshaping how we interact with technology. This blog post "
                "explores the opportunities and challenges that lie ahead in this exciting field.\n\n"
                
                "Introduction\n"
                "The pace of AI development has accelerated dramatically, with breakthroughs "
                "in machine learning, natural language processing, and computer vision. These "
                "advances have enabled applications that were once the realm of science fiction.\n\n"
                
                "Opportunities\n"
                "AI offers tremendous potential for solving complex problems, from healthcare "
                "diagnostics to climate modeling. Personalized education, efficient resource "
                "allocation, and enhanced scientific discovery are just a few areas where AI "
                "can make significant contributions.\n\n"
                
                "Challenges\n"
                "With these opportunities come important challenges. Ethical concerns, job "
                "displacement, algorithmic bias, and privacy issues must be addressed thoughtfully. "
                "Ensuring AI benefits humanity broadly requires careful governance and inclusive design.\n\n"
                
                "Conclusion\n"
                "The future of AI will be shaped by how we navigate these opportunities and "
                "challenges. By approaching AI development with both enthusiasm and responsibility, "
                "we can harness its potential while mitigating risks."
            )
            
            sections_content = {
                "introduction": "The pace of AI development has accelerated dramatically...",
                "body": "AI offers tremendous potential for solving complex problems...",
                "conclusion": "The future of AI will be shaped by how we navigate these opportunities and challenges..."
            }
            
        elif content_type == "story":
            title = "The Quantum Messenger"
            description = "A short science fiction story about communication across time"
            full_text = (
                "The Quantum Messenger\n\n"
                
                "In the shadow of Mount Rainier, the quantum research facility hummed with activity. "
                "Dr. Elena Reyes adjusted her glasses as she reviewed the readouts from the latest experiment. "
                "\"We're getting something,\" she whispered, almost afraid to believe it.\n\n"
                
                "The team had been working on quantum entanglement for years, but this was different. "
                "The patterns weren't random—they carried information. Information that couldn't possibly "
                "originate from their lab.\n\n"
                
                "\"It's... it's a message,\" her colleague Michael said, staring at the screen. \"But from where?\"\n\n"
                
                "As they decoded the quantum signature, a chill ran down Elena's spine. The timestamp "
                "embedded in the message was from 2157—thirty years in the future.\n\n"
                
                "\"To past researchers: Success achieved. Causality loop stable. Your next step works. "
                "—Dr. E. Reyes\"\n\n"
                
                "Elena looked at her hands. Had she just received a message from herself? More importantly, "
                "had she just proven that information could traverse time?\n\n"
                
                "\"What do we do now?\" Michael asked.\n\n"
                
                "Elena smiled. \"We take the next step.\""
            )
            
            sections_content = {
                "setting": "In the shadow of Mount Rainier, the quantum research facility hummed with activity...",
                "characters": "Dr. Elena Reyes adjusted her glasses... her colleague Michael said, staring at the screen.",
                "plot": "The team had been working on quantum entanglement for years, but this was different...",
                "resolution": "Elena smiled. \"We take the next step.\""
            }
            
        elif content_type == "social":
            title = "Announcing Our New Product Line"
            description = "Social media post for product announcement"
            full_text = (
                "🚀 EXCITING NEWS! 🚀\n\n"
                "We're thrilled to announce our new eco-friendly product line, launching next month!\n\n"
                "♻️ Sustainable materials\n"
                "🌿 Carbon-neutral production\n"
                "💪 Same great quality you trust\n\n"
                "Sign up for early access and get 15% off your first purchase: link.example.com/early-access\n\n"
                "#SustainableLiving #NewProduct #EcoFriendly"
            )
            
            sections_content = {
                "headline": "🚀 EXCITING NEWS! 🚀",
                "body": "We're thrilled to announce our new eco-friendly product line, launching next month!...",
                "call_to_action": "Sign up for early access and get 15% off your first purchase: link.example.com/early-access"
            }
            
        else:  # email
            title = "Follow-up: Project Proposal Discussion"
            description = "Business email follow-up after meeting"
            full_text = (
                "Dear Marcus,\n\n"
                
                "I hope this email finds you well. I wanted to thank you for taking the time to meet "
                "with our team yesterday to discuss the proposed collaboration.\n\n"
                
                "As promised, I've attached the revised proposal that incorporates the feedback you provided "
                "during our discussion. We've particularly focused on addressing your concerns about the "
                "timeline and resource allocation.\n\n"
                
                "Key changes include:\n"
                "- Extended the research phase by two weeks\n"
                "- Added two additional quality assurance checkpoints\n"
                "- Included detailed cost breakdown as requested\n\n"
                
                "Please let me know if you'd like to schedule a follow-up call to review these changes. "
                "We're available any time next week that works for your schedule.\n\n"
                
                "Thank you again for your valuable input. We're excited about the potential of working together.\n\n"
                
                "Best regards,\n"
                "Sarah Johnson\n"
                "Project Director"
            )
            
            sections_content = {
                "greeting": "Dear Marcus,",
                "body": "I hope this email finds you well. I wanted to thank you for taking the time to meet...",
                "closing": "Best regards,\nSarah Johnson\nProject Director"
            }
        
        # Return structured content
        return {
            "title": title,
            "description": description,
            "content_type": content_type,
            "summary": f"I've created a {content_type} titled \"{title}\" based on your request.",
            "full_text": full_text,
            "sections": sections_content,
            "word_count": len(full_text.split())
        }
    
    def run(self):
        """Run the agent server."""
        import uvicorn
        uvicorn.run(self.app, host=self.host, port=self.port) 