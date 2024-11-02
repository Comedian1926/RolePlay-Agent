from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from uuid import uuid4
from enum import Enum
import time


class MessageType(Enum):
    """消息类型枚举"""
    CHAT = "chat"  # 普通对话
    EMOTION = "emotion"  # 情感表达
    ACTION = "action"  # 动作描述
    THOUGHT = "thought"  # 内部思考
    SYSTEM = "system"  # 系统消息
    NARRATIVE = "narrative"  # 叙事描述


class MessagePriority(Enum):
    """消息优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class MessageMetadata:
    """消息元数据"""
    type: MessageType = MessageType.CHAT
    priority: MessagePriority = MessagePriority.NORMAL
    emotion_tags: List[str] = field(default_factory=list)
    topic_tags: List[str] = field(default_factory=list)
    custom_data: Dict[str, Any] = field(default_factory=dict)
    create_time: datetime = field(default_factory=datetime.now)
    update_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "type": self.type.value,
            "priority": self.priority.value,
            "emotion_tags": self.emotion_tags,
            "topic_tags": self.topic_tags,
            "custom_data": self.custom_data,
            "create_time": self.create_time.isoformat(),
            "update_time": self.update_time.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MessageMetadata':
        """从字典创建元数据"""
        data = data.copy()
        data['type'] = MessageType(data['type'])
        data['priority'] = MessagePriority(data['priority'])
        data['create_time'] = datetime.fromisoformat(data['create_time'])
        data['update_time'] = datetime.fromisoformat(data['update_time'])
        return cls(**data)


@dataclass
class Message:
    """增强的消息类

    Attributes:
        sender: 消息发送者
        content: 消息内容
        timestamp: 消息发送时间
        receiver: 消息接收者（None表示广播）
        message_id: 消息唯一标识符
        metadata: 消息元数据
        thread_id: 对话线程ID
        reference_ids: 引用的其他消息ID
    """
    sender: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    receiver: Optional[str] = None
    message_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: MessageMetadata = field(default_factory=MessageMetadata)
    thread_id: Optional[str] = None
    reference_ids: List[str] = field(default_factory=list)

    def __post_init__(self):
        """初始化后的处理"""
        if self.thread_id is None:
            self.thread_id = self.message_id

    def update_content(self, new_content: str):
        """更新消息内容"""
        self.content = new_content
        self.metadata.update_time = datetime.now()

    def add_reference(self, message_id: str):
        """添加消息引用"""
        if message_id not in self.reference_ids:
            self.reference_ids.append(message_id)

    def add_emotion_tag(self, emotion: str):
        """添加情感标签"""
        if emotion not in self.metadata.emotion_tags:
            self.metadata.emotion_tags.append(emotion)

    def add_topic_tag(self, topic: str):
        """添加主题标签"""
        if topic not in self.metadata.topic_tags:
            self.metadata.topic_tags.append(topic)

    def set_priority(self, priority: MessagePriority):
        """设置消息优先级"""
        self.metadata.priority = priority

    def is_private(self) -> bool:
        """检查是否为私密消息"""
        return self.receiver is not None

    def is_system_message(self) -> bool:
        """检查是否为系统消息"""
        return self.metadata.type == MessageType.SYSTEM

    def is_action(self) -> bool:
        """检查是否为动作消息"""
        return self.metadata.type == MessageType.ACTION

    def is_emotion(self) -> bool:
        """检查是否为情感消息"""
        return self.metadata.type == MessageType.EMOTION

    def get_elapsed_time(self) -> timedelta:
        """获取消息发送后经过的时间

        Returns:
            timedelta: 从消息发送到现在经过的时间
        """
        return datetime.now() - self.timestamp

    def format_elapsed_time(self) -> str:
        """获取格式化的经过时间字符串

        Returns:
            str: 格式化的时间字符串，如"5分钟前"、"1小时前"等
        """
        elapsed = self.get_elapsed_time()

        if elapsed.days > 0:
            return f"{elapsed.days}天前"
        elif elapsed.seconds >= 3600:
            hours = elapsed.seconds // 3600
            return f"{hours}小时前"
        elif elapsed.seconds >= 60:
            minutes = elapsed.seconds // 60
            return f"{minutes}分钟前"
        else:
            return "刚刚"

    def to_dict(self) -> Dict[str, Any]:
        """将消息转换为字典格式"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['metadata'] = self.metadata.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建消息对象"""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['metadata'] = MessageMetadata.from_dict(data['metadata'])
        return cls(**data)

    def __str__(self) -> str:
        """返回消息的字符串表示"""
        msg_type = f"[{self.metadata.type.value}]"
        priority = f"({self.metadata.priority.name})" if self.metadata.priority != MessagePriority.NORMAL else ""
        receiver_str = f" -> {self.receiver}" if self.receiver else " (广播)"
        tags = ""
        if self.metadata.emotion_tags or self.metadata.topic_tags:
            all_tags = self.metadata.emotion_tags + self.metadata.topic_tags
            tags = f" #{' #'.join(all_tags)}"

        return (f"[{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"{msg_type}{priority} {self.sender}{receiver_str}: "
                f"{self.content}{tags}")

    @classmethod
    def create_system_message(cls, content: str) -> 'Message':
        """创建系统消息"""
        return cls(
            sender="SYSTEM",
            content=content,
            metadata=MessageMetadata(
                type=MessageType.SYSTEM,
                priority=MessagePriority.HIGH
            )
        )

    @classmethod
    def create_action_message(cls, sender: str, action: str) -> 'Message':
        """创建动作消息"""
        return cls(
            sender=sender,
            content=action,
            metadata=MessageMetadata(
                type=MessageType.ACTION
            )
        )

    @classmethod
    def create_emotion_message(cls,
                               sender: str,
                               content: str,
                               emotion: str) -> 'Message':
        """创建情感消息"""
        msg = cls(
            sender=sender,
            content=content,
            metadata=MessageMetadata(
                type=MessageType.EMOTION
            )
        )
        msg.add_emotion_tag(emotion)
        return msg


class MessageBuilder:
    """消息构建器"""

    def __init__(self, sender: str):
        self.sender = sender
        self.content: Optional[str] = None
        self.receiver: Optional[str] = None
        self.metadata = MessageMetadata()
        self.thread_id: Optional[str] = None
        self.reference_ids: List[str] = []

    def set_content(self, content: str) -> 'MessageBuilder':
        self.content = content
        return self

    def set_receiver(self, receiver: str) -> 'MessageBuilder':
        self.receiver = receiver
        return self

    def set_type(self, msg_type: MessageType) -> 'MessageBuilder':
        self.metadata.type = msg_type
        return self

    def set_priority(self, priority: MessagePriority) -> 'MessageBuilder':
        self.metadata.priority = priority
        return self

    def set_thread(self, thread_id: str) -> 'MessageBuilder':
        self.thread_id = thread_id
        return self

    def add_reference(self, message_id: str) -> 'MessageBuilder':
        if message_id not in self.reference_ids:
            self.reference_ids.append(message_id)
        return self

    def add_emotion_tag(self, emotion: str) -> 'MessageBuilder':
        if emotion not in self.metadata.emotion_tags:
            self.metadata.emotion_tags.append(emotion)
        return self

    def add_topic_tag(self, topic: str) -> 'MessageBuilder':
        if topic not in self.metadata.topic_tags:
            self.metadata.topic_tags.append(topic)
        return self

    def build(self) -> Message:
        """构建消息对象"""
        if self.content is None:
            raise ValueError("Content must be set before building message")

        return Message(
            sender=self.sender,
            content=self.content,
            receiver=self.receiver,
            metadata=self.metadata,
            thread_id=self.thread_id,
            reference_ids=self.reference_ids
        )


if __name__ == "__main__":
    import time

    # 基础消息测试
    basic_message = Message(
        sender="Alice",
        content="Hello, Bob!",
        timestamp=datetime.now(),
        receiver="Bob"
    )
    print("\n=== 基础消息测试 ===")
    print(basic_message)

    # 使用消息构建器
    builder_message = (MessageBuilder("Alice")
                       .set_content("How are you?")
                       .set_type(MessageType.CHAT)
                       .set_priority(MessagePriority.HIGH)
                       .set_receiver("Bob")
                       .add_emotion_tag("关心")
                       .add_topic_tag("问候")
                       .build())
    print("\n=== 构建器消息测试 ===")
    print(builder_message)

    # 特殊类型消息测试
    system_message = Message.create_system_message("系统维护中...")
    emotion_message = Message.create_emotion_message("Alice", "真开心见到你！", "喜悦")
    action_message = Message.create_action_message("Alice", "微笑着挥手")

    print("\n=== 特殊消息测试 ===")
    print("系统消息:", system_message)
    print("情感消息:", emotion_message)
    print("动作消息:", action_message)

    # 时间显示测试
    print("\n=== 时间显示测试 ===")
    # 创建一个稍早的消息
    earlier_message = Message(
        sender="Alice",
        content="这是一条稍早的消息",
        timestamp=datetime.now() - timedelta(minutes=0),
        receiver="Bob"
    )
    print(f"消息发送时间: {earlier_message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"消息发送经过: {earlier_message.format_elapsed_time()}")

    # 序列化测试
    print("\n=== 序列化测试 ===")
    msg_dict = builder_message.to_dict()
    restored_message = Message.from_dict(msg_dict)
    print("序列化后恢复的消息:", restored_message)

    # 消息属性测试
    print("\n=== 消息属性测试 ===")
    print(f"是否私密消息: {builder_message.is_private()}")
    print(f"是否系统消息: {builder_message.is_system_message()}")
    print(f"是否情感消息: {builder_message.is_emotion()}")
    print(f"消息ID: {builder_message.message_id}")
    print(f"线程ID: {builder_message.thread_id}")
    print(f"情感标签: {builder_message.metadata.emotion_tags}")
    print(f"主题标签: {builder_message.metadata.topic_tags}")