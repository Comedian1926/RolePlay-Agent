from anthropic import Anthropic
from typing import List, Dict, Optional
import asyncio

from ..base import BaseLLM, LLMConfig


class AnthropicLLM(BaseLLM):
    """Anthropic Claude LLM实现"""

    def __init__(self, config: LLMConfig):
        """初始化Anthropic LLM

        Args:
            config (LLMConfig): LLM配置
        """
        super().__init__(config)
        self.client = Anthropic(api_key=config.api_key)

        # Claude模型的temperature范围是0-1
        if self.config.temperature > 1:
            self.config.temperature = 1.0

    async def generate(self,
                       prompt: str,
                       temperature: Optional[float] = None) -> str:
        """单次生成

        Example:
            ```python
            # 生成分析内容
            analysis = await llm.generate(
                "分析这篇文章的主要观点：" + article,
                temperature=0.2
            )

            # 创意写作
            story = await llm.generate(
                "创作一个科幻短篇故事，主题是时间旅行",
                temperature=0.9
            )
            ```
        """
        try:
            # 构建消息
            messages = [{
                "role": "user",
                "content": prompt
            }]

            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self._get_temperature(temperature),
                messages=messages
            )

            return response.content[0].text

        except Exception as e:
            print(f"Anthropic生成错误: {str(e)}")
            return ""

    async def chat(self,
                   messages: List[Dict[str, str]],
                   system_prompt: Optional[str] = None,
                   temperature: Optional[float] = None) -> str:
        """多轮对话

        Example:
            ```python
            messages = [
                {"role": "user", "content": "你能扮演一个历史学家吗？"},
                {"role": "assistant", "content": "当然可以..."},
                {"role": "user", "content": "告诉我关于古罗马的有趣事实"}
            ]
            response = await llm.chat(
                messages=messages,
                system_prompt="你是一位专精于古罗马历史的学者"
            )
            ```
        """
        try:
            # 构建Claude格式的消息列表
            claude_messages = []

            # 处理系统提示词
            # Claude没有专门的system role，我们需要在第一条消息中包含角色设定
            if system_prompt:
                claude_messages.append({
                    "role": "user",
                    "content": f"System: {system_prompt}\n\nHuman: 让我们开始对话。"
                })
                claude_messages.append({
                    "role": "assistant",
                    "content": "我明白我的角色设定。让我们开始对话。"
                })

            # 转换消息格式
            for msg in messages:
                role = msg["role"]
                content = msg["content"]

                # Claude使用"user"和"assistant"
                claude_role = "user" if role == "user" else "assistant"

                claude_messages.append({
                    "role": claude_role,
                    "content": content
                })

            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self._get_temperature(temperature),
                messages=claude_messages
            )

            return response.content[0].text

        except Exception as e:
            print(f"Anthropic对话错误: {str(e)}")
            return ""

    def _get_temperature(self, temperature: Optional[float] = None) -> float:
        """获取适用于Claude的temperature值

        Claude的temperature范围是0-1，需要做相应调整

        Args:
            temperature (Optional[float]): 输入的temperature值

        Returns:
            float: 调整后的temperature值
        """
        temp = super()._get_temperature(temperature)
        # 确保temperature在0-1范围内
        return min(1.0, max(0.0, temp))

    def _format_message_for_claude(self, role: str, content: str) -> Dict[str, str]:
        """格式化消息为Claude格式

        Args:
            role (str): 消息角色
            content (str): 消息内容

        Returns:
            Dict[str, str]: Claude格式的消息
        """
        # Claude中"user"对应Human，"assistant"对应Assistant
        claude_role = "user" if role == "user" else "assistant"
        return {
            "role": claude_role,
            "content": content
        }