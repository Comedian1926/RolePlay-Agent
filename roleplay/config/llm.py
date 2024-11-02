
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum


class LLMProvider(Enum):
    """LLM提供商"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"


@dataclass
class LLMConfig:
    """LLM配置"""
    provider: LLMProvider
    model_name: str
    api_key: str
    api_base: Optional[str] = None

    # 生成参数
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop_sequences: Optional[list] = None

    # 重试配置
    max_retries: int = 3
    retry_delay: float = 1.0

    # 其他设置
    timeout: float = 30.0
    extra_params: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "provider": self.provider.value,
            "model_name": self.model_name,
            "api_key": "***",  # 隐藏API key
            "api_base": self.api_base,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "stop_sequences": self.stop_sequences,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "timeout": self.timeout,
            "extra_params": self.extra_params
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMConfig':
        """从字典创建配置"""
        data = data.copy()
        if "provider" in data:
            data["provider"] = LLMProvider(data["provider"])
        return cls(**data)