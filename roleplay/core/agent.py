from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
import asyncio
from uuid import uuid4
import json

from message import Message, MessageMetadata, MessageType, MessagePriority
from memory import Memory
from role import Role
from roleplay.llm.base import BaseLLM


class AgentState(Enum):
    """Agent状态枚举"""
    IDLE = "idle"  # 空闲状态
    ACTIVE = "active"  # 活跃状态
    THINKING = "thinking"  # 思考状态
    SPEAKING = "speaking"  # 说话状态
    LISTENING = "listening"  # 倾听状态
    PAUSED = "paused"  # 暂停状态
    STOPPED = "stopped"  # 停止状态
    ERROR = "error"  # 错误状态


@dataclass
class AgentContext:
    """Agent上下文信息"""
    current_topic: Optional[str] = None
    conversation_depth: int = 0
    last_speaker: Optional[str] = None
    interaction_count: Dict[str, int] = field(default_factory=dict)
    last_emotions: Dict[str, float] = field(default_factory=dict)
    conversation_history: List[str] = field(default_factory=list)

    def update_interaction(self, agent_name: str, emotion: Optional[str] = None):
        """更新互动信息"""
        self.interaction_count[agent_name] = self.interaction_count.get(agent_name, 0) + 1
        self.last_speaker = agent_name
        if emotion:
            self.last_emotions[agent_name] = emotion


class Agent:
    """优化的Agent实现"""

    def __init__(self,
                 role: Role,
                 memory_config: Optional[Dict[str, Any]] = None,
                 llm: Optional[BaseLLM] = None):
        """初始化Agent

        Args:
            role: Agent的角色定义
            memory_config: 记忆系统配置
            llm: LLM实例
        """
        self.id = str(uuid4())
        self.role = role
        self.state = AgentState.IDLE
        self.context = AgentContext()
        self.llm = llm

        # 设置记忆系统
        memory_config = memory_config or {
            "working_memory_limit": 5,
            "short_term_limit": 20,
            "long_term_limit": 100,
            "importance_threshold": 0.7
        }
        self.memory = Memory(**memory_config)

        # 设置日志
        self.logger = logging.getLogger(f"Agent_{role.name}")
        self.logger.setLevel(logging.INFO)

        # 状态锁
        self._state_lock = asyncio.Lock()

        # 设置提示词模板
        self._setup_prompt_templates()

    def _setup_prompt_templates(self):
        """设置提示词模板"""
        self._base_prompt = """
你现在扮演一个名为{name}的角色。

角色背景：
{background}

性格特征：
{traits}

当前情绪状态：
{emotions}

你需要根据以上角色设定和以下信息生成回复：

当前话题：{topic}
对话深度：{depth}
历史消息：
{history}

刚收到的消息：
{message}

要求：
1. 保持角色性格特征的一致性
2. 根据当前情绪状态调整回应语气
3. 考虑对话历史和上下文
4. 回复要自然、流畅，符合角色身份

请以这个角色的身份回复："""

        self._emotion_prompt = """
分析以下内容的情感特征：

{content}

请从以下维度进行分析并以JSON格式返回：
{
    "emotion": "主要情绪类型",
    "intensity": "情绪强度(0.0-1.0)",
    "valence": "情感倾向(positive/negative/neutral)",
    "keywords": ["情感关键词"]
}
"""

    async def receive_message(self, message: Message) -> Optional[str]:
        """接收并处理消息

        Args:
            message: 接收到的消息

        Returns:
            Optional[str]: 生成的响应
        """
        async with self._state_lock:
            try:
                # 状态检查
                if self.state == AgentState.STOPPED:
                    return None

                self.state = AgentState.LISTENING
                self.logger.info(f"接收消息: {message.content[:50]}...")

                # 更新上下文
                self._update_context(message)

                # 评估重要性并存储消息
                importance = await self._evaluate_message_importance(message)
                await self.memory.add_message(message, importance)

                # 准备响应
                self.state = AgentState.THINKING
                response = await self._generate_response(message)

                # 如果生成了响应，创建响应消息并存储
                if response:
                    response_msg = await self._create_response_message(message, response)
                    await self.memory.add_message(response_msg, importance * 0.9)

                return response

            except Exception as e:
                self.logger.error(f"处理消息时出错: {str(e)}")
                self.state = AgentState.ERROR
                return None

            finally:
                if self.state != AgentState.ERROR:
                    self.state = AgentState.IDLE

    def _update_context(self, message: Message):
        """更新上下文信息"""
        if message.sender != self.role.name:
            self.context.update_interaction(
                message.sender,
                message.metadata.emotion_tags[0] if message.metadata.emotion_tags else None
            )
            self.context.conversation_depth += 1

        # 更新当前主题
        if message.metadata.topic_tags:
            self.context.current_topic = message.metadata.topic_tags[0]
            self.context.conversation_history.append(self.context.current_topic)

    async def _evaluate_message_importance(self, message: Message) -> float:
        """评估消息重要性"""
        importance = 0.5  # 基础重要性

        # 使用LLM进行情感分析来调整重要性
        if self.llm:
            try:
                emotion_analysis = await self._analyze_emotion(message.content)
                if emotion_analysis:
                    # 根据情感强度调整重要性
                    importance += emotion_analysis.get("intensity", 0) * 0.3

                    # 更新消息的情感标签
                    if "emotion" in emotion_analysis:
                        message.metadata.emotion_tags.append(emotion_analysis["emotion"])
            except Exception as e:
                self.logger.warning(f"情感分析失败: {str(e)}")

        # 优先级影响
        priority_weights = {
            MessagePriority.LOW: -0.1,
            MessagePriority.NORMAL: 0,
            MessagePriority.HIGH: 0.2,
            MessagePriority.URGENT: 0.3
        }
        importance += priority_weights[message.metadata.priority]

        # 消息类型影响
        type_weights = {
            MessageType.EMOTION: 0.15,
            MessageType.ACTION: 0.1,
            MessageType.SYSTEM: 0.2,
            MessageType.THOUGHT: 0.1
        }
        importance += type_weights.get(message.metadata.type, 0)

        # 是否直接对话
        if message.receiver == self.role.name:
            importance += 0.2

        return max(0.1, min(1.0, importance))

    async def _generate_response(self, message: Message) -> Optional[str]:
        """生成响应"""
        if not self.llm:
            self.logger.warning("未设置LLM，无法生成回复")
            return None

        try:
            # 获取近期对话历史
            recent_messages = self.memory.get_recent_context(5)
            history = "\n".join([
                f"{msg.sender}: {msg.content}"
                for msg in recent_messages
            ])

            # 构建提示词
            prompt = self._base_prompt.format(
                name=self.role.name,
                background=self.role.background,
                traits=self._format_traits(),
                emotions=self._format_emotions(),
                topic=self.context.current_topic or "未指定",
                depth=self.context.conversation_depth,
                history=history,
                message=message.content
            )

            # 根据消息类型调整temperature
            temperature = self._get_temperature(message)

            # 生成回复
            response = await self.llm.generate(prompt, temperature=temperature)
            return response.strip()

        except Exception as e:
            self.logger.error(f"生成回复时出错: {str(e)}")
            return None

    def _get_temperature(self, message: Message) -> float:
        """根据消息类型动态调整temperature"""
        base_temperature = 0.7  # 默认temperature

        if message.metadata.type == MessageType.EMOTION:
            return min(base_temperature + 0.2, 1.0)  # 情感消息增加随机性
        elif message.metadata.type == MessageType.SYSTEM:
            return max(base_temperature - 0.3, 0.1)  # 系统消息降低随机性

        return base_temperature

    async def _analyze_emotion(self, content: str) -> Optional[Dict[str, Any]]:
        """分析文本情感"""
        if not self.llm:
            return None

        try:
            prompt = self._emotion_prompt.format(content=content)
            result = await self.llm.generate(prompt, temperature=0.3)
            return json.loads(result)
        except Exception as e:
            self.logger.warning(f"情感分析失败: {str(e)}")
            return None

    def _format_traits(self) -> str:
        """格式化性格特征"""
        return "\n".join([
            f"- {trait}: {value:.1f}"
            for trait, value in self.role.traits.items()
        ])

    def _format_emotions(self) -> str:
        """格式化当前情绪状态"""
        return "\n".join([
            f"- {emotion}: {value:.1f}"
            for emotion, value in self.role.emotions.items()
        ])

    async def _create_response_message(self,
                                       original_msg: Message,
                                       content: str) -> Message:
        """创建响应消息"""
        # 分析回复的情感
        emotion_analysis = await self._analyze_emotion(content)

        metadata = MessageMetadata(
            type=original_msg.metadata.type,
            priority=original_msg.metadata.priority
        )

        # 添加情感标签
        if emotion_analysis and "emotion" in emotion_analysis:
            metadata.emotion_tags.append(emotion_analysis["emotion"])

        return Message(
            sender=self.role.name,
            content=content,
            receiver=original_msg.sender,
            metadata=metadata,
            thread_id=original_msg.thread_id
        )

    def set_llm(self, llm: BaseLLM):
        """设置LLM实例"""
        self.llm = llm

    def get_state(self) -> Dict[str, Any]:
        """获取Agent状态信息"""
        return {
            "id": self.id,
            "name": self.role.name,
            "state": self.state.value,
            "context": {
                "current_topic": self.context.current_topic,
                "conversation_depth": self.context.conversation_depth,
                "last_speaker": self.context.last_speaker,
                "interaction_count": self.context.interaction_count,
                "last_emotions": self.context.last_emotions
            },
            "memory_stats": self.memory.get_memory_summary(),
            "has_llm": self.llm is not None
        }

if __name__ == "__main__":
    import asyncio
    from role import Role


    async def agent_demo():
        print("开始测试Agent...")

        # 创建测试角色
        test_role = Role(
            name="TestAgent",
            traits={"友好": 0.8, "幽默": 0.6},
            background="这是一个测试用的AI助手"
        )

        # 创建Agent实例
        agent = Agent(test_role)

        # 创建测试消息
        test_messages = [
            Message(  # 普通对话消息
                sender="User1",
                content="你好，这是一条测试消息",
                metadata=MessageMetadata(
                    type=MessageType.CHAT,
                    priority=MessagePriority.NORMAL,
                    topic_tags=["打招呼"]
                )
            ),
            Message(  # 情感消息
                sender="User2",
                content="今天我很开心！",
                metadata=MessageMetadata(
                    type=MessageType.EMOTION,
                    priority=MessagePriority.HIGH,
                    emotion_tags=["快乐"]
                )
            ),
            Message(  # 系统消息
                sender="System",
                content="系统测试消息",
                metadata=MessageMetadata(
                    type=MessageType.SYSTEM,
                    priority=MessagePriority.URGENT
                )
            )
        ]

        # 测试消息处理
        for msg in test_messages:
            print(f"\n处理消息: {msg.content}")
            response = await agent.receive_message(msg)
            print(f"Agent响应: {response}")

            # 打印当前状态
            state = agent.get_state()
            print(f"\n当前状态:")
            print(f"会话深度: {state['context']['conversation_depth']}")
            print(f"最后发言者: {state['context']['last_speaker']}")
            print(f"互动次数: {state['context']['interaction_count']}")

        # 打印记忆统计
        memory_stats = agent.memory.get_memory_summary()
        print("\n记忆统计:")
        print(f"工作记忆数量: {memory_stats['working_memory_count']}")
        print(f"短期记忆数量: {memory_stats['short_term_count']}")
        print(f"长期记忆数量: {memory_stats['long_term_count']}")


    # 运行测试
    asyncio.run(agent_demo())