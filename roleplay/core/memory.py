from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import deque
from message import Message, MessageType, MessageMetadata

@dataclass
class MemoryItem:
    """记忆项
    保持简单但保留消息的完整特性
    """
    content: Message
    timestamp: datetime
    importance: float = 0.0

class Memory:
    """最小化但功能完整的记忆系统"""
    def __init__(self,
                 working_memory_limit: int = 5,
                 short_term_limit: int = 20,
                 long_term_limit: int = 100,
                 importance_threshold: float = 0.7):
        """初始化记忆系统
        保持与原始接口一致的参数，但简化实现
        """
        self.messages = deque(maxlen=working_memory_limit + short_term_limit)
        self.limit = working_memory_limit + short_term_limit  # 合并工作和短期记忆限制
        self.importance_threshold = importance_threshold

    async def add_message(self, message: Message, importance: float = 0.0):
        """添加消息到记忆系统"""
        memory_item = MemoryItem(
            content=message,
            timestamp=datetime.now(),
            importance=importance
        )
        self.messages.append(memory_item)

    def get_recent_context(self, limit: int = 5) -> List[Message]:
        """获取最近的上下文"""
        # 按时间逆序获取消息
        recent_items = sorted(
            self.messages,
            key=lambda x: x.timestamp,
            reverse=True
        )[:limit]
        
        # 保持消息的时间顺序
        recent_items.reverse()
        return [item.content for item in recent_items]

    def get_memories_by_type(self, msg_type: MessageType, limit: int = 5) -> List[Message]:
        """按消息类型获取记忆"""
        type_items = [
            item for item in self.messages
            if item.content.metadata.type == msg_type
        ]
        sorted_items = sorted(
            type_items,
            key=lambda x: (x.importance, x.timestamp),
            reverse=True
        )[:limit]
        return [item.content for item in sorted_items]

    def get_messages_with(self, participant: str, limit: int = 5) -> List[Message]:
        """获取与特定参与者的对话历史"""
        relevant_items = [
            item for item in self.messages
            if (item.content.sender == participant or 
                item.content.receiver == participant)
        ]
        sorted_items = sorted(
            relevant_items,
            key=lambda x: x.timestamp,
            reverse=True
        )[:limit]
        return [item.content for item in sorted_items]

    def get_stats(self) -> Dict[str, Any]:
        """获取基本统计信息"""
        type_counts = {}
        for item in self.messages:
            msg_type = item.content.metadata.type.value
            type_counts[msg_type] = type_counts.get(msg_type, 0) + 1

        return {
            "total_messages": len(self.messages),
            "memory_capacity": self.limit,
            "message_types": type_counts
        }

    def clear(self):
        """清空记忆"""
        self.messages.clear()

# 使用示例
async def example():
    # 创建记忆系统
    memory = Memory()
    
    # 创建并添加一条消息
    message = Message(
        sender="user",
        content="Hello!",
        receiver="bot",
        metadata=MessageMetadata(type=MessageType.CHAT)
    )
    
    await memory.add_message(message, importance=0.5)
    
    # 获取上下文
    contexts = memory.get_recent_context(5)
    
    # 获取特定类型的消息
    chat_messages = memory.get_memories_by_type(MessageType.CHAT)
    
    # 获取与特定用户的对话
    user_messages = memory.get_messages_with("user")
    
    # 获取统计信息
    stats = memory.get_stats()