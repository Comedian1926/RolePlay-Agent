from typing import Dict, Any
from dataclasses import dataclass, field

@dataclass
class Role:
    """最小化的角色定义，仅保留agent实际使用的功能"""
    name: str
    background: str
    traits: Dict[str, float] = field(default_factory=dict)

    def _format_traits(self) -> str:
        """格式化特征描述，用于提示词生成"""
        return ", ".join(f"{k}: {v:.1f}" for k, v in self.traits.items())

    def to_prompt(self) -> str:
        """生成角色提示词，用于LLM对话"""
        return (
            f"角色：{self.name}\n"
            f"背景：{self.background}\n"
            f"特征：{self._format_traits()}"
        )

# 使用示例
if __name__ == "__main__":
    # 创建角色示例
    assistant = Role(
        name="AI助手",
        background="一个专业的AI助手，擅长解决问题和交流",
        traits={
            "专业性": 0.8,
            "友好": 0.9
        }
    )
    
    # 打印角色提示词
    print(assistant.to_prompt())