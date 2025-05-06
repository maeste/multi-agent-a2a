"""
Common type definitions for the A2A multi-agent system.
"""
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, field_serializer, field_validator, ValidationInfo


class TaskState(str, Enum):
    """States for A2A tasks."""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PartType(str, Enum):
    """Types of content parts in A2A messages."""
    TEXT = "text"
    FILE = "file"
    DATA = "data"


class Part(BaseModel):
    """Base class for all message parts."""
    model_config = {'discriminator': 'type'}
    type: PartType


class TextPart(Part):
    """Text content part."""
    type: PartType = PartType.TEXT
    text: str


class FilePart(Part):
    """File content part."""
    type: PartType = PartType.FILE
    file_name: str
    mime_type: str
    content: bytes


class DataPart(Part):
    """Structured data content part."""
    type: PartType = PartType.DATA
    data: Dict[str, Any]


class Message(BaseModel):
    """A message in the A2A protocol."""
    parts: List[Union[TextPart, FilePart, DataPart]]

    @field_serializer('parts', mode='plain')
    def serialize_parts(self, parts: List[Union[TextPart, FilePart, DataPart]]):
        return [part.model_dump(mode='json') for part in parts]



class TaskStatus(BaseModel):
    """Status of an A2A task."""
    state: TaskState
    message: Optional[Message] = None
    reason: Optional[str] = None


class ArtifactType(str, Enum):
    """Types of artifacts produced by agents."""
    DOCUMENT = "document"
    VISUALIZATION = "visualization"
    DATA = "data"
    PLAN = "plan"
    OTHER = "other"


class Artifact(BaseModel):
    """An artifact produced by an agent."""
    id: str
    type: ArtifactType
    name: str
    description: Optional[str] = None
    content: Any


class Task(BaseModel):
    """A task in the A2A protocol."""
    id: str
    status: TaskStatus
    artifacts: List[Artifact] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Skill(BaseModel):
    """A skill that an agent can perform."""
    id: str
    name: str
    description: str


class Capabilities(BaseModel):
    """Capabilities of an agent."""
    streaming: bool = False
    pushNotifications: bool = False


class AgentCard(BaseModel):
    """Agent description card for discovery."""
    name: str
    description: str
    url: str
    version: str
    capabilities: Capabilities
    defaultInputModes: List[str]
    defaultOutputModes: List[str]
    skills: List[Skill] 