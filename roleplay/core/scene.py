from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import logging
from enum import Enum
import json
from dataclasses import dataclass
import asyncio

from memory import Memory
from message import Message, MessageMetadata, MessageType, MessagePriority
from agent import Agent, AgentState
from role import Role


class SceneEvent(Enum):
    """场景事件类型"""
    CHARACTER_JOINED = "character_joined"  # 角色加入场景
    CHARACTER_LEFT = "character_left"  # 角色离开场景
    DIALOGUE_RECEIVED = "dialogue_received"  # 收到新对话
    DIALOGUE_BROADCAST = "dialogue_broadcast"  # 对话广播
    SCENE_CHANGED = "scene_changed"  # 场景状态改变
    ERROR_OCCURRED = "error_occurred"  # 发生错误


@dataclass
class SceneConfig:
    """场景配置"""
    max_characters: int = 10  # 最大角色数量
    dialogue_history_limit: int = 1000  # 对话历史限制
    broadcast_timeout: float = 5.0  # 广播超时时间（秒）
    enable_async: bool = True  # 是否启用异步处理
    enable_logging: bool = True  # 是否启用日志
    save_history: bool = True  # 是否保存历史记录
    scene_description: str = ""  # 场景描述
    background_story: str = ""  # 背景故事


class Scene:
    """场景管理器

    管理角色之间的互动、对话路由和场景状态。
    """

    def __init__(self, config: Optional[SceneConfig] = None):
        """初始化场景"""
        self.config = config or SceneConfig()
        self.characters: Dict[str, Agent] = {}
        self.dialogue_history: List[Message] = []
        self.event_handlers: Dict[SceneEvent, List[Callable]] = {
            event: [] for event in SceneEvent
        }

        if self.config.enable_logging:
            self.logger = logging.getLogger("Scene")
            self.logger.setLevel(logging.INFO)

    def add_event_handler(self, event: SceneEvent, handler: Callable):
        """添加事件处理器"""
        self.event_handlers[event].append(handler)

    def _trigger_event(self, event: SceneEvent, data: Dict[str, Any] = None):
        """触发事件"""
        if not data:
            data = {}
        for handler in self.event_handlers[event]:
            try:
                handler(data)
            except Exception as e:
                if self.config.enable_logging:
                    self.logger.error(f"事件处理错误 {event}: {str(e)}")

    async def add_character(self, agent: Agent) -> bool:
        """添加新角色到场景"""
        if len(self.characters) >= self.config.max_characters:
            self.logger.warning(f"场景已达到最大角色数量: {self.config.max_characters}")
            return False

        character_name = agent.role.name
        if character_name in self.characters:
            self.logger.warning(f"角色 '{character_name}' 已存在于场景中")
            return False

        self.characters[character_name] = agent
        self._trigger_event(SceneEvent.CHARACTER_JOINED, {"character": agent})
        self.logger.info(f"角色 '{character_name}' 已加入场景")
        return True

    async def remove_character(self, character_name: str) -> bool:
        """从场景中移除角色"""
        if character_name not in self.characters:
            return False

        character = self.characters.pop(character_name)
        self._trigger_event(SceneEvent.CHARACTER_LEFT, {"character": character})
        self.logger.info(f"角色 '{character_name}' 已离开场景")
        return True

    async def broadcast_dialogue(self, message: Message) -> Dict[str, str]:
        """广播对话给所有相关角色"""
        try:
            self._add_to_history(message)
            self._trigger_event(SceneEvent.DIALOGUE_BROADCAST, {"message": message})

            if self.config.enable_async:
                return await self._broadcast_async(message)
            else:
                return self._broadcast_sync(message)
        except Exception as e:
            self.logger.error(f"广播对话时出错: {str(e)}")
            return {}

    async def _broadcast_async(self, message: Message) -> Dict[str, str]:
        """异步广播对话"""
        tasks = []
        for name, character in self.characters.items():
            if name != message.sender:
                if message.receiver is None or message.receiver == name:
                    task = asyncio.create_task(
                        self._get_character_response(character, message)
                    )
                    tasks.append(task)

        responses = {}
        if tasks:
            try:
                done, pending = await asyncio.wait(
                    tasks,
                    timeout=self.config.broadcast_timeout
                )

                # 处理完成的任务
                for task in done:
                    try:
                        character_name, response = await task
                        if response:
                            responses[character_name] = response
                    except Exception as e:
                        self.logger.error(f"处理角色响应时出错: {str(e)}")

                # 取消未完成的任务
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            except Exception as e:
                self.logger.error(f"异步广播时出错: {str(e)}")

        return responses

    def _broadcast_sync(self, message: Message) -> Dict[str, str]:
        """同步广播对话"""
        responses = {}
        for name, character in self.characters.items():
            if name != message.sender:
                if message.receiver is None or message.receiver == name:
                    try:
                        character.receive_message(message)
                        response = character.generate_response(message)
                        if response:
                            responses[name] = response
                    except Exception as e:
                        self.logger.error(f"处理角色 {name} 响应时出错: {str(e)}")
        return responses

    async def _get_character_response(self,
                                      character: Agent,
                                      message: Message) -> tuple[str, Optional[str]]:
        """获取单个角色的回应"""
        try:
            response = await character.receive_message(message)
            return character.role.name, response
        except Exception as e:
            self.logger.error(f"获取角色 {character.role.name} 的响应时出错: {str(e)}")
            return character.role.name, None

    def _add_to_history(self, message: Message):
        """添加对话到历史记录"""
        self.dialogue_history.append(message)
        if len(self.dialogue_history) > self.config.dialogue_history_limit:
            self.dialogue_history.pop(0)

    def get_dialogue_history(self,
                             limit: Optional[int] = None,
                             character_name: Optional[str] = None) -> List[Message]:
        """获取对话历史"""
        history = self.dialogue_history
        if character_name:
            history = [
                msg for msg in history
                if msg.sender == character_name or msg.receiver == character_name
            ]
        if limit:
            history = history[-limit:]
        return history

    def get_character(self, character_name: str) -> Optional[Agent]:
        """获取特定的角色"""
        return self.characters.get(character_name)

    def get_character_states(self) -> Dict[str, str]:
        """获取所有角色的状态"""
        return {
            name: character.state.value
            for name, character in self.characters.items()
        }

    def describe_scene(self) -> str:
        """获取场景描述"""
        desc = f"""
场景：{self.config.scene_description}

背景故事：{self.config.background_story}

当前角色：
"""
        for name, character in self.characters.items():
            desc += f"- {name}: {character.role.background}\n"

        return desc.strip()

    def get_scene_state(self) -> Dict[str, Any]:
        """获取场景状态信息"""
        return {
            "character_count": len(self.characters),
            "dialogue_count": len(self.dialogue_history),
            "active_characters": sum(
                1 for character in self.characters.values()
                if character.state != AgentState.STOPPED
            ),
            "scene_description": self.config.scene_description,
            "last_dialogue_time": self.dialogue_history[-1].timestamp if self.dialogue_history else None
        }

    def save_state(self, file_path: str):
        """保存场景状态到文件"""
        state = {
            "config": vars(self.config),
            "dialogue_history": [msg.to_dict() for msg in self.dialogue_history],
            "characters": {
                name: character.get_state()
                for name, character in self.characters.items()
            }
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_state(cls, file_path: str) -> 'Scene':
        """从文件加载场景状态"""
        with open(file_path, 'r', encoding='utf-8') as f:
            state = json.load(f)

        config = SceneConfig(**state["config"])
        scene = cls(config)

        # 恢复历史记录
        scene.dialogue_history = [
            Message.from_dict(msg_dict)
            for msg_dict in state["dialogue_history"]
        ]

        return scene


if __name__ == "__main__":
    async def run_demo():
        print("=== 开始场景演示 ===")

        # 创建场景
        config = SceneConfig(
            max_characters=5,
            scene_description="一家温馨的咖啡厅",
            background_story="这是一家位于城市角落的小咖啡厅，以其独特的咖啡香和温暖的氛围闻名..."
        )
        scene = Scene(config)

        # 创建角色
        cafe_owner = Role(
            name="店长小王",
            traits={"专业": 0.9, "友好": 0.8, "耐心": 0.7},
            background="拥有十年咖啡经验的咖啡师，热爱与顾客分享咖啡知识。"
        )
        owner_agent = Agent(cafe_owner)

        customer = Role(
            name="顾客小李",
            traits={"好奇": 0.8, "健谈": 0.7},
            background="一位咖啡爱好者，总是喜欢尝试新品。"
        )
        customer_agent = Agent(customer)

        # 添加角色到场景
        print("\n=== 角色加入场景 ===")
        await scene.add_character(owner_agent)
        await scene.add_character(customer_agent)
        print(scene.describe_scene())

        # 模拟对话
        print("\n=== 对话开始 ===")
        messages = [
            Message(
                sender="顾客小李",
                content="您好！今天有什么特别推荐的咖啡吗？",
                metadata=MessageMetadata(type=MessageType.CHAT)
            ),
            Message(
                sender="顾客小李",
                content="我比较喜欢醇厚的口感。",
                metadata=MessageMetadata(type=MessageType.CHAT)
            )
        ]

        for msg in messages:
            print(f"\n{msg.sender}: {msg.content}")
            responses = await scene.broadcast_dialogue(msg)
            for name, response in responses.items():
                print(f"{name}: {response}")
            await asyncio.sleep(1)

        # 显示场景状态
        print("\n=== 场景状态 ===")
        state = scene.get_scene_state()
        print(f"角色数量: {state['character_count']}")
        print(f"对话数量: {state['dialogue_count']}")
        print(f"活跃角色: {state['active_characters']}")


    # 运行演示
    asyncio.run(run_demo())