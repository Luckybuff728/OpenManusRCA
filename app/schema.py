from __future__ import annotations

from enum import Enum
from typing import Any, List, Literal, Optional, Union, Dict
import json
import re

from pydantic import BaseModel, Field, ConfigDict, field_validator

from app.logger import logger


class Role(str, Enum):
    """Message role options"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


ROLE_VALUES = tuple(role.value for role in Role)
ROLE_TYPE = Literal[ROLE_VALUES]  # type: ignore


class ToolChoice(str, Enum):
    """Tool choice options"""

    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


TOOL_CHOICE_VALUES = tuple(choice.value for choice in ToolChoice)
TOOL_CHOICE_TYPE = Literal[TOOL_CHOICE_VALUES]  # type: ignore


class AgentState(str, Enum):
    """Agent execution states"""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


class Function(BaseModel):
    name: str
    arguments: str
    model_config = ConfigDict(from_attributes=True)

    def arguments_as_dict(self) -> Dict[str, Any]:
        """将参数字符串解析为Python字典。"""
        try:
            return json.loads(self.arguments)
        except json.JSONDecodeError:
            # 如果解析失败，返回一个空字典或者进行其他错误处理
            return {}


class ToolCall(BaseModel):
    """Represents a tool/function call in a message"""

    id: str
    type: str = "function"
    function: Function
    model_config = ConfigDict(from_attributes=True)


class Message(BaseModel):
    """Represents a chat message in the conversation"""

    role: ROLE_TYPE = Field(...)  # type: ignore
    content: Optional[Any] = Field(default=None)
    tool_calls: Optional[List[ToolCall]] = Field(default=None)
    name: Optional[str] = Field(default=None)
    tool_call_id: Optional[str] = Field(default=None)
    base64_image: Optional[str] = Field(default=None)

    # Allow creating from attributes (e.g., from another ORM model or class)
    model_config = ConfigDict(from_attributes=True)

    @field_validator("content", mode="before")
    @classmethod
    def validate_and_sanitize_content(cls, v: Any) -> Any:
        """
        验证并净化content字段，防止LLM返回的乱码或无效内容污染系统。
        """
        if not isinstance(v, str) or not v:
            return v

        is_garbled = False
        # 1. 检查是否存在非预期的字符集（例如阿拉伯文、特殊符号等）
        # 这个正则表达式允许中文、英文、数字、基本标点和一些特殊格式字符。
        # 它可以有效地过滤掉大多数意外的脚本，如日志中出现的阿拉伯文。
        if not re.fullmatch(
            r"[\s\S\w\u4e00-\u9fa5\u3000-\u303f\uff00-\uffef.,!?'\"`#:/\\(){}\[\]_=+~-]*",
            v,
        ):
            is_garbled = True

        # 2. 检查是否有无效的unicode序列
        try:
            json.dumps(v)
        except UnicodeEncodeError:
            is_garbled = True

        if is_garbled:
            logger.warning(f"检测到无效或乱码的思考内容，已进行净化。原始内容: '{v}'")
            return "[Agent is formulating a plan...]"

        return v

    def __add__(self, other) -> List["Message"]:
        """支持 Message + list 或 Message + Message 的操作"""
        if isinstance(other, list):
            return [self] + other
        elif isinstance(other, Message):
            return [self, other]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __radd__(self, other) -> List["Message"]:
        """支持 list + Message 的操作"""
        if isinstance(other, list):
            return other + [self]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(other).__name__}' and '{type(self).__name__}'"
            )

    def to_dict(self) -> dict:
        """Convert message to dictionary format, ensuring content is serialized for APIs."""
        message = {"role": self.role}

        if self.content is not None:
            if isinstance(self.content, str):
                message["content"] = self.content
            elif isinstance(self.content, (dict, list)):
                message["content"] = json.dumps(self.content, ensure_ascii=False)
            else:
                # For other types (like objects), rely on their string representation.
                message["content"] = str(self.content)

        if self.tool_calls is not None:
            message["tool_calls"] = [
                tool_call.model_dump() for tool_call in self.tool_calls
            ]
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.base64_image is not None:
            message["base64_image"] = self.base64_image
        return message

    @classmethod
    def user_message(
        cls, content: str, base64_image: Optional[str] = None
    ) -> "Message":
        """Create a user message"""
        return cls(role=Role.USER, content=content, base64_image=base64_image)

    @classmethod
    def system_message(cls, content: str) -> "Message":
        """Create a system message"""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def assistant_message(
        cls, content: Optional[str] = None, base64_image: Optional[str] = None
    ) -> "Message":
        """Create an assistant message"""
        return cls(role=Role.ASSISTANT, content=content, base64_image=base64_image)

    @classmethod
    def tool_message(
        cls, content: Any, name, tool_call_id: str, base64_image: Optional[str] = None
    ) -> "Message":
        """Create a tool message"""
        return cls(
            role=Role.TOOL,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
            base64_image=base64_image,
        )

    @classmethod
    def from_openai_response(cls, response_message) -> "Message":
        """
        Creates a Message instance from an OpenAI ChatCompletionMessage object.
        """
        # The 'content' can be None, especially for tool calls.
        content = response_message.content or None

        tool_calls = None
        if response_message.tool_calls:
            tool_calls = [
                ToolCall.model_validate(tc) for tc in response_message.tool_calls
            ]

        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
        )

    @classmethod
    def from_tool_calls(
        cls,
        tool_calls: List[Any],
        content: Union[str, List[str]] = "",
        base64_image: Optional[str] = None,
        **kwargs,
    ):
        """Create ToolCallsMessage from raw tool calls.

        Args:
            tool_calls: Raw tool calls from LLM
            content: Optional message content
            base64_image: Optional base64 encoded image
        """
        formatted_calls = [
            {"id": call.id, "function": call.function.model_dump(), "type": "function"}
            for call in tool_calls
        ]
        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=formatted_calls,
            base64_image=base64_image,
            **kwargs,
        )


class Memory(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    max_messages: int = Field(default=100)

    def add_message(self, message: Message) -> None:
        """Add a message to memory"""
        self.messages.append(message)
        # Optional: Implement message limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def add_user_message(
        self, content: str, base64_image: Optional[str] = None
    ) -> None:
        """Add a user message to memory

        Args:
            content: The message content
            base64_image: Optional base64 encoded image
        """
        message = Message.user_message(content=content, base64_image=base64_image)
        self.add_message(message)

    def add_system_message(self, content: str) -> None:
        """Add a system message to memory."""
        message = Message.system_message(content=content)
        self.add_message(message)

    def add_messages(self, messages: List[Message]) -> None:
        """Add multiple messages to memory"""
        self.messages.extend(messages)
        # Optional: Implement message limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def clear(self) -> None:
        """Clear all messages"""
        self.messages.clear()

    def get_recent_messages(self, n: int) -> List[Message]:
        """Get n most recent messages"""
        return self.messages[-n:]

    def to_dict_list(self) -> List[dict]:
        """Convert messages to list of dicts"""
        return [msg.to_dict() for msg in self.messages]
