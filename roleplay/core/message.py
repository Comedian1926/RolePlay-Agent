from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List

class MessageType(Enum):
    """消息类型"""
    CHAT = "chat"      # 对话消息
    SYSTEM = "system"  # 系统消息
    ACTION = "action"  # 动作消息

@dataclass
class MessageMetadata:
    """简化的消息元数据"""
    type: MessageType = MessageType.CHAT

@dataclass
class Message:
    """简化的消息实现，只保留agent中用到的功能"""
    sender: str
    content: str
    receiver: Optional[str] = None
    metadata: MessageMetadata = field(default_factory=MessageMetadata)
    timestamp: datetime = field(default_factory=datetime.now)

    def is_system_message(self) -> bool:
        """检查是否为系统消息"""
        return self.metadata.type == MessageType.SYSTEM

    def is_action(self) -> bool:
        """检查是否为动作消息"""
        return self.metadata.type == MessageType.ACTION

    @classmethod
    def create_system_message(cls, content: str) -> 'Message':
        """创建系统消息"""
        return cls(
            sender="SYSTEM",
            content=content,
            metadata=MessageMetadata(type=MessageType.SYSTEM)
        )

    @classmethod
    def create_action_message(cls, sender: str, action: str) -> 'Message':
        """创建动作消息"""
        return cls(
            sender=sender,
            content=action,
            metadata=MessageMetadata(type=MessageType.ACTION)
        )

# 使用示例
if __name__ == "__main__":
    # 创建普通对话消息
    chat_msg = Message(
        sender="user",
        content="你好！",
        receiver="bot"
    )

    # 创建系统消息
    sys_msg = Message.create_system_message("系统初始化完成")

    # 创建动作消息
    action_msg = Message.create_action_message("bot", "思考中...")

    # 测试消息类型判断
    print(f"是否系统消息: {sys_msg.is_system_message()}")
    print(f"是否动作消息: {action_msg.is_action()}")