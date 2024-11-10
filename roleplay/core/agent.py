from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import logging
from enum import Enum
import asyncio
from uuid import uuid4
from message import Message
from role import Role
from message import Message, MessageType, MessageMetadata
from memory import Memory

class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    SPEAKING = "speaking"
    LISTENING = "listening"
    EXECUTING = "executing"
    ERROR = "error"

@dataclass
class AgentContext:
    """Agent上下文信息"""
    current_topic: Optional[str] = None
    conversation_depth: int = 0
    last_speaker: Optional[str] = None
    interaction_count: Dict[str, int] = field(default_factory=dict)
    last_emotions: Dict[str, float] = field(default_factory=dict)
    conversation_history: List[str] = field(default_factory=list)

class BaseAgent(ABC):
    """基础Agent类"""
    def __init__(self, llm):
        self.id = str(uuid4())
        self.llm = llm
        self.state = AgentState.IDLE
        self._state_lock = asyncio.Lock()
        self.logger = logging.getLogger(f"Agent_{self.__class__.__name__}")
        
    @abstractmethod
    async def process_message(self, message: Message) -> Optional[str]:
        pass

class RolePlayAgent(BaseAgent):
    def __init__(self, 
                 role: Role,
                 llm,
                 scene_description: str,
                 background_story: str,
                 memory_config: Optional[Dict[str, Any]] = None,
                 tools: Optional[List['Tool']] = None,
                 prompt_templates: Optional[Dict[str, str]] = None):
        """初始化RolePlayAgent"""
        super().__init__(llm)
        self.role = role
        self.context = AgentContext()
        self.tools = tools or []
        self.scene_description = scene_description
        self.background_story = background_story
        
        # 初始化记忆系统
        memory_config = memory_config or {
            "working_memory_limit": 5,
            "short_term_limit": 20,
            "long_term_limit": 100,
            "importance_threshold": 0.7
        }
        self.memory = Memory(**memory_config)
        
        # 初始化提示词模板
        self.prompt_templates = prompt_templates or {}
        if not self.prompt_templates:
            self._setup_prompt_templates()

    def _update_context(self, message: Message):
        """更新对话上下文"""
        # 更新发言者
        self.context.last_speaker = message.sender
        # 更新与该发言者的互动次数
        if message.sender not in self.context.interaction_count:
            self.context.interaction_count[message.sender] = 0
        self.context.interaction_count[message.sender] += 1
        # 更新对话深度
        if message.sender == self.context.last_speaker:
            self.context.conversation_depth += 1
        else:
            self.context.conversation_depth = 1
        # 分析情感并更新状态
        self._update_emotions(message)

    def _update_emotions(self, message: Message):
        """更新情感状态"""
        # 基于角色特征和消息内容更新情感
        base_emotions = {
            "happy": 0.0,
            "sad": 0.0,
            "excited": 0.0,
            "nervous": 0.0,
        }
        
        # 简单的情感分析逻辑
        if "开心" in message.content or "喜欢" in message.content:
            base_emotions["happy"] += 0.3
        if "难过" in message.content or "抱歉" in message.content:
            base_emotions["sad"] += 0.3
        if "期待" in message.content or "热情" in message.content:
            base_emotions["excited"] += 0.3
        if "担心" in message.content or "紧张" in message.content:
            base_emotions["nervous"] += 0.3
            
        # 考虑角色特征对情感的影响
        for trait, value in self.role.traits.items():
            if trait in ["浪漫", "温柔"]:
                base_emotions["happy"] *= value
            elif trait in ["害羞", "感性"]:
                base_emotions["nervous"] *= value
        
        self.context.last_emotions = base_emotions

    def _format_emotions(self) -> str:
        """格式化当前情感状态"""
        if not self.context.last_emotions:
            return "平静"
        
        emotions = []
        for emotion, value in self.context.last_emotions.items():
            if value > 0.2:
                emotions.append(f"{emotion}: {value:.1f}")
        
        return ", ".join(emotions) if emotions else "平静"

    async def process_message(self, message: Message) -> Optional[str]:
        """处理输入消息"""
        async with self._state_lock:
            try:
                self.state = AgentState.LISTENING
                self.logger.info(f"接收消息: {message.content[:50]}...")
                
                # 更新上下文
                self._update_context(message)
                
                # 评估消息重要性并存储
                importance = await self._evaluate_message_importance(message)
                await self.memory.add_message(message, importance)
                
                # 如果是任务类型的消息,则执行任务
                if self._is_task_message(message):
                    self.state = AgentState.EXECUTING
                    response = await self._execute_task(message)
                else:
                    # 生成对话回复
                    self.state = AgentState.THINKING
                    response = await self._generate_response(message)
                
                # 存储响应
                if response:
                    response_msg = await self._create_response_message(message, response)
                    await self.memory.add_message(response_msg, importance * 0.9)
                
                return response
                
            except Exception as e:
                self.logger.error(f"处理消息出错: {str(e)}")
                self.state = AgentState.ERROR
                return None
            finally:
                if self.state != AgentState.ERROR:
                    self.state = AgentState.IDLE

    async def _execute_task(self, message: Message) -> Optional[str]:
        """执行任务"""
        try:
            # 解析任务
            task = message.content
            
            # 如果有相关工具,则使用工具执行
            if self.tools:
                tool = self._select_tool(task)
                if tool:
                    result = await tool.execute(task)
                    return self._format_tool_result(result)
            
            # 如果没有工具或无法使用工具,则生成回复
            prompt = self._task_prompt.format(
                name=self.role.name,
                task=task,
                tools=self._format_tools()
            )
            return await self.llm.generate(prompt, temperature=0.3)
            
        except Exception as e:
            self.logger.error(f"执行任务失败: {str(e)}")
            return None

    async def _generate_response(self, message: Message) -> Optional[str]:
        """生成对话回复"""
        prompt = self.prompt_templates["chat"].format(
            name=self.role.name,
            background=self.role.background,
            traits=self._format_traits(),
            message=message.content,
            scene_description=self.scene_description
        )
        return await self.llm.generate(
            prompt, 
            temperature=self._get_temperature(message)
        )

    async def _evaluate_message_importance(self, message: Message) -> float:
        """评估消息的重要性"""
        importance = 0.5  # 基础重要性
        
        # 根据消息类型调整
        if message.is_system_message():
            importance += 0.3
        elif message.is_action():
            importance += 0.2
            
        # 根据发送者调整
        if message.sender == self.role.name:
            importance += 0.1
        elif message.receiver == self.role.name:
            importance += 0.2
            
        # 根据内容关键词调整
        important_keywords = ["爱", "抱歉", "生气", "难过", "开心", "记得"]
        for keyword in important_keywords:
            if keyword in message.content:
                importance += 0.1
                
        # 根据角色特征调整
        if "感性" in self.role.traits:
            importance *= (1 + self.role.traits["感性"] * 0.2)
            
        # 限制在 0-1 范围内
        return min(max(importance, 0.0), 1.0)

    async def _create_response_message(self, original_msg: Message, content: str) -> Message:
        """创建响应消息"""
        return Message(
            sender=self.role.name,
            content=content,
            receiver=original_msg.sender,
            metadata=MessageMetadata(type=MessageType.CHAT)
        )

    def _get_temperature(self, message: Message) -> float:
        """根据消息类型和角色特征确定生成温度"""
        base_temp = 0.7
        
        # 根据消息类型调整
        if message.is_system_message():
            base_temp -= 0.2
        elif message.is_action():
            base_temp += 0.1
            
        # 根据角色特征调整
        if "幽默" in self.role.traits:
            base_temp += self.role.traits["幽默"] * 0.2
        if "严谨" in self.role.traits:
            base_temp -= self.role.traits["严谨"] * 0.2
            
        return min(max(base_temp, 0.1), 1.0)

    def _format_traits(self) -> str:
        """格式化角色特征描述"""
        return ", ".join(f"{trait}: {value:.1f}" 
                        for trait, value in self.role.traits.items())

    def _setup_prompt_templates(self):
        """设置默认提示词模板"""
        self.prompt_templates = {
            "chat": (
                f"你扮演的角色是:\n"
                f"姓名：{{name}}\n"
                f"背景：{{background}}\n"
                f"性格特征：{{traits}}\n\n"
                f"请以这个角色的身份回复对话。保持对话自然流畅，符合角色性格。\n"
                f"当前场景：{self.scene_description}\n"
                f"对话内容：{{message}}\n"
                f"请用一句话回复:"
            )
        }


    def _is_task_message(self, message: Message) -> bool:
        """判断是否是任务类型的消息"""
        return message.metadata.type in [
            MessageType.SYSTEM,
            MessageType.ACTION
        ]

    def _select_tool(self, task: str) -> Optional['Tool']:
        """根据任务选择合适的工具"""
        # 可以使用LLM来智能选择工具
        for tool in self.tools:
            if tool.can_handle(task):
                return tool
        return None
        
    def _format_tools(self) -> str:
        """格式化工具列表"""
        if not self.tools:
            return "无可用工具"
        return "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools
        ])

    def _format_tool_result(self, result: Any) -> str:
        """格式化工具执行结果"""
        return f"执行结果: {result}"