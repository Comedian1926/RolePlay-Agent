from typing import List, Dict, Any, Optional, TypeVar, Generic, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import json
import heapq
from message import Message, MessageMetadata, MessageType, MessagePriority
from uuid import uuid4

T = TypeVar('T')


@dataclass
class MemoryItem(Generic[T]):
    """记忆项"""
    content: T
    timestamp: datetime
    importance: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    reference_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    memory_id: str = field(default_factory=lambda: str(uuid4()))

    def update_access(self):
        """更新访问时间和引用计数"""
        self.last_accessed = datetime.now()
        self.reference_count += 1


class Memory:
    """增强的记忆系统"""

    def __init__(self,
                 working_memory_limit: int = 5,
                 short_term_limit: int = 20,
                 long_term_limit: int = 100,
                 importance_threshold: float = 0.7):
        """初始化记忆系统"""
        # 记忆存储
        self.working_memory: List[MemoryItem[Message]] = []
        self.short_term: List[MemoryItem[Message]] = []
        self.long_term: List[MemoryItem[Message]] = []

        # 配置参数
        self.working_memory_limit = working_memory_limit
        self.short_term_limit = short_term_limit
        self.long_term_limit = long_term_limit
        self.importance_threshold = importance_threshold

        # 索引结构
        self.keyword_index: Dict[str, List[str]] = defaultdict(list)  # 存储memory_id
        self.temporal_index: Dict[str, List[str]] = defaultdict(list)
        self.type_index: Dict[str, List[str]] = defaultdict(list)  # 修改为使用字符串类型
        self.memory_map: Dict[str, MemoryItem] = {}  # 通过ID快速访问记忆

        # 记忆统计
        self.stats = {
            "total_memories": 0,
            "forgotten_memories": 0,
            "consolidation_count": 0,
            "memory_types": defaultdict(int)  # 不同类型消息的统计
        }

    def _extract_tags(self, content: str) -> List[str]:
        """提取内容标签的简单实现"""
        common_topics = [
            "测试", "问题", "想法", "计划", "决定",
            "任务", "建议", "情感", "帮助", "总结"
        ]
        return [topic for topic in common_topics if topic in content]

    async def add_message(self, message: Message, importance: float = 0.0):
        """添加新消息到记忆系统"""
        memory_item = MemoryItem(
            content=message,
            timestamp=datetime.now(),
            importance=importance,
            metadata={
                "sender": message.sender,
                "receiver": message.receiver,
                "type": message.metadata.type.value,  # 存储字符串值
                "priority": message.metadata.priority.value
            },
            tags=self._extract_tags(message.content)
        )

        # 更新记忆映射
        self.memory_map[memory_item.memory_id] = memory_item

        # 更新类型统计
        msg_type = message.metadata.type.value  # 使用字符串值
        self.stats["memory_types"][msg_type] += 1

        # 根据重要性决定存储位置
        if importance >= self.importance_threshold:
            await self._add_to_long_term(memory_item)
        elif len(self.working_memory) < self.working_memory_limit:
            self.working_memory.append(memory_item)
        else:
            await self._consolidate_working_memory()
            self.working_memory.append(memory_item)

        # 更新索引
        await self._update_indices(memory_item)
        self.stats["total_memories"] += 1

    async def _update_indices(self, memory_item: MemoryItem):
        """更新索引结构"""
        # 更新关键词索引
        for tag in memory_item.tags:
            self.keyword_index[tag].append(memory_item.memory_id)

        # 更新时间索引
        time_key = memory_item.timestamp.strftime("%Y-%m-%d")
        self.temporal_index[time_key].append(memory_item.memory_id)

        # 更新类型索引
        msg_type = memory_item.metadata['type']  # 已经是字符串
        self.type_index[msg_type].append(memory_item.memory_id)

    async def _add_to_long_term(self, memory_item: MemoryItem):
        """添加到长期记忆"""
        self.long_term.append(memory_item)
        if len(self.long_term) > self.long_term_limit:
            await self._forget_least_important()

    async def _consolidate_working_memory(self):
        """整合工作记忆到短期记忆"""
        self.short_term.extend(self.working_memory)
        self.working_memory.clear()

        if len(self.short_term) > self.short_term_limit:
            await self._consolidate_memories()

    async def _consolidate_memories(self):
        """记忆整合策略"""
        # 按重要性和访问频率排序
        self.short_term.sort(
            key=lambda x: (x.importance * 0.7 + x.reference_count * 0.3),
            reverse=True
        )

        # 保留重要的记忆，移动到长期记忆
        important_memories = self.short_term[:self.short_term_limit // 2]
        self.long_term.extend(important_memories)

        # 其余的记忆可能被遗忘
        forgotten_memories = self.short_term[self.short_term_limit // 2:]
        self.stats["forgotten_memories"] += len(forgotten_memories)

        # 清理短期记忆
        self.short_term = self.short_term[:self.short_term_limit // 2]
        self.stats["consolidation_count"] += 1

    def get_memory_summary(self) -> Dict[str, Any]:
        """获取增强的记忆系统统计信息"""
        return {
            "working_memory_count": len(self.working_memory),
            "short_term_count": len(self.short_term),
            "long_term_count": len(self.long_term),
            "total_memories": self.stats["total_memories"],
            "forgotten_memories": self.stats["forgotten_memories"],
            "consolidation_count": self.stats["consolidation_count"],
            "memory_types_distribution": dict(self.stats["memory_types"]),
            "index_stats": {
                "keywords": len(self.keyword_index),
                "temporal": len(self.temporal_index),
                "types": {type_: len(msgs) for type_, msgs in self.type_index.items()}  # 直接使用字符串
            }
        }
if __name__ == "__main__":
    import asyncio
    from message import Message, MessageMetadata, MessageType, MessagePriority
    from memory import Memory


    async def main():
        # 创建记忆系统实例
        memory = Memory(working_memory_limit=3, short_term_limit=5)

        # 创建几条测试消息
        test_messages = [
            Message(
                sender="Alice",
                content="你好，这是第一条测试消息",
                metadata=MessageMetadata(type=MessageType.CHAT)
            ),
            Message(
                sender="Bob",
                content="我很开心！",
                metadata=MessageMetadata(type=MessageType.EMOTION)
            ),
            Message(
                sender="System",
                content="系统通知：测试消息",
                metadata=MessageMetadata(type=MessageType.SYSTEM)
            )
        ]

        # 添加消息到记忆系统
        for msg in test_messages:
            await memory.add_message(msg, importance=0.6)
            print(f"添加消息: {msg.content}")

        # 打印记忆统计
        stats = memory.get_memory_summary()
        print("\n记忆统计:")
        print(f"工作记忆: {stats['working_memory_count']}")
        print(f"短期记忆: {stats['short_term_count']}")
        print(f"长期记忆: {stats['long_term_count']}")


    # 运行测试
    asyncio.run(main())

