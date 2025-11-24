"""
Core Types and Data Structures

This module contains the basic data structures for message handling and conversations.
Separated from model.py for better modularity and reusability.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class MessageRole(Enum):
    """Enumeration of message roles in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ContentType(Enum):
    """Enumeration of content types in messages."""
    TEXT = "text"
    IMAGE_URL = "image_url"
    IMAGE_BASE64 = "image_base64"


@dataclass
class MessageContent:
    """Represents a piece of content within a message (text, image, etc.)."""
    type: ContentType
    text: Optional[str] = None
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format compatible with API calls."""
        if self.type == ContentType.TEXT:
            return {"type": "text", "text": self.text}
        elif self.type == ContentType.IMAGE_URL:
            return {"type": "image_url", "image_url": {"url": self.image_url}}
        elif self.type == ContentType.IMAGE_BASE64:
            return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.image_base64}"}}


@dataclass
class Message:
    """Represents a single message in a conversation."""
    role: MessageRole
    content: Union[str, List[MessageContent]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format compatible with API calls."""
        if isinstance(self.content, str):
            return {"role": self.role.value, "content": self.content}
        else:
            return {
                "role": self.role.value,
                "content": [content.to_dict() for content in self.content]
            }


@dataclass
class Conversation:
    """Represents a conversation with multiple messages."""
    messages: List[Message] = field(default_factory=list)
    system_prompt: Optional[str] = None
    
    def __post_init__(self):
        if self.system_prompt and not any(msg.role == MessageRole.SYSTEM for msg in self.messages):
            self.add_system_message(self.system_prompt)
    
    def add_message(self, role: MessageRole, content: Union[str, List[MessageContent]]) -> None:
        """Add a message to the conversation."""
        self.messages.append(Message(role=role, content=content))
    
    def add_system_message(self, content: str) -> None:
        """Add or update a system message at the beginning of the conversation."""
        # Insert at beginning if no system message exists, otherwise replace
        if self.messages and self.messages[0].role == MessageRole.SYSTEM:
            self.messages[0] = Message(role=MessageRole.SYSTEM, content=content)
        else:
            self.messages.insert(0, Message(role=MessageRole.SYSTEM, content=content))
    
    def add_user_message(self, content: Union[str, List[MessageContent]]) -> None:
        """Add a user message to the conversation."""
        self.add_message(MessageRole.USER, content)
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation."""
        self.add_message(MessageRole.ASSISTANT, content)
    
    def clear(self) -> None:
        """Clear all messages, keeping only system prompt if it exists."""
        self.messages.clear()
        if self.system_prompt:
            self.add_system_message(self.system_prompt)
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert conversation to list of dictionaries compatible with API calls."""
        return [msg.to_dict() for msg in self.messages]