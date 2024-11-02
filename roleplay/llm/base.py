from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """LLM基础配置"""
    model_name: str  # 模型名称
    temperature: float = 0.7  # 温度参数（越高创造性越强）
    max_tokens: int = 1000  # 最大token数
    top_p: float = 0.9  # 采样概率（影响输出多样性）
    api_key: str = None  # API密钥


class BaseLLM(ABC):
    """LLM基础抽象类

    提供两种主要的调用方式：
    1. generate: 适用于单次、独立的生成任务，如创作、分析等
    2. chat: 适用于多轮对话，保持上下文的交互场景
    """

    def __init__(self, config: LLMConfig):
        """初始化LLM

        Args:
            config (LLMConfig): LLM配置
        """
        self.config = config

    @abstractmethod
    async def generate(self,
                       prompt: str,
                       temperature: Optional[float] = None) -> str:
        """生成回复的抽象方法

        适用场景：
        - 单次生成任务（写作、分析、总结等）
        - 不需要上下文的独立询问
        - 工具性调用（如生成分析报告）

        Args:
            prompt (str): 输入提示词
            temperature (Optional[float]): 可选的温度参数，覆盖默认配置

        Returns:
            str: 生成的回复
        """
        pass

    @abstractmethod
    async def chat(self,
                   messages: List[Dict[str, str]],
                   system_prompt: Optional[str] = None,
                   temperature: Optional[float] = None) -> str:
        """对话方法

        适用场景：
        - 多轮对话交互
        - 需要角色设定的对话
        - 需要维持上下文的交互

        Args:
            messages (List[Dict[str, str]]): 对话历史
                格式: [{"role": "user"/"assistant", "content": str}]
            system_prompt (Optional[str]): 系统提示词，用于设定角色人设
            temperature (Optional[float]): 可选的温度参数，覆盖默认配置

        Returns:
            str: 生成的回复
        """
        pass

    def _get_temperature(self, temperature: Optional[float] = None) -> float:
        """获取实际使用的temperature值

        Args:
            temperature (Optional[float]): 可选的temperature值

        Returns:
            float: 实际使用的temperature值
        """
        if temperature is not None:
            # 确保temperature在有效范围内
            return max(0.0, min(2.0, temperature))
        return self.config.temperature