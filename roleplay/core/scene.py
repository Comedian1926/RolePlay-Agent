from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio
import logging
from message import Message, MessageType
from role import Role 
from agent import RolePlayAgent

@dataclass
class SceneConfig:
    """场景配置"""
    max_characters: int = 10  # 支持多人
    scene_description: str = "" 
    background_story: str = ""
    broadcast_timeout: float = 5.0  # 广播超时时间

class Scene:
    """多人对话场景管理器"""
    def __init__(self, config: SceneConfig):
        self.config = config
        self.characters: Dict[str, RolePlayAgent] = {}
        self.dialogue_history: List[Message] = []
        self.logger = logging.getLogger("Scene")

    async def add_character(self, agent: RolePlayAgent) -> bool:
        """添加角色到场景"""
        if len(self.characters) >= self.config.max_characters:
            self.logger.warning(f"场景已达到最大角色数量: {self.config.max_characters}")
            return False

        name = agent.role.name
        if name in self.characters:
            self.logger.warning(f"角色 '{name}' 已存在")
            return False

        self.characters[name] = agent
        self.logger.info(f"角色 '{name}' 已加入场景")
        return True

    async def broadcast_dialogue(self, message: Message) -> Dict[str, str]:
        """异步广播对话消息"""
        self.dialogue_history.append(message)
        
        tasks = []
        # 为每个其他角色创建响应任务
        for name, character in self.characters.items():
            if name != message.sender:
                task = asyncio.create_task(
                    self._get_response(character, message)
                )
                tasks.append((name, task))

        responses = {}
        if tasks:
            try:
                # 等待所有响应，带超时
                done, pending = await asyncio.wait(
                    [task for _, task in tasks],
                    timeout=self.config.broadcast_timeout
                )

                # 处理完成的任务
                for (name, task) in tasks:
                    if task in done:
                        try:
                            response = await task
                            if response:
                                responses[name] = response
                        except Exception as e:
                            self.logger.error(f"处理角色 {name} 响应出错: {str(e)}")
                    else:
                        self.logger.warning(f"角色 {name} 响应超时")

            except Exception as e:
                self.logger.error(f"广播对话出错: {str(e)}")

        return responses

    async def _get_response(self, agent: RolePlayAgent, message: Message) -> Optional[str]:
        """获取单个角色的响应"""
        try:
            return await agent.process_message(message)
        except Exception as e:
            self.logger.error(f"获取响应出错: {str(e)}")
            return None

    def describe_scene(self) -> str:
        """获取场景描述"""
        desc = [
            f"场景：{self.config.scene_description}",
            f"背景：{self.config.background_story}",
            f"\n当前角色({len(self.characters)}):"
        ]
        for name, char in self.characters.items():
            desc.append(f"- {name}: {char.role.background}")
        return "\n".join(desc)

    def get_dialogue_history(self, 
                           limit: Optional[int] = None,
                           character: Optional[str] = None) -> List[Message]:
        """获取对话历史,支持按角色筛选"""
        history = self.dialogue_history
        if character:
            history = [
                msg for msg in history 
                if msg.sender == character or msg.receiver == character
            ]
        if limit:
            history = history[-limit:]
        return history