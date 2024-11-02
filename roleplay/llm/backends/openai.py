import openai
from typing import List, Dict, Optional
import asyncio

from ..base import BaseLLM, LLMConfig


class OpenAILLM(BaseLLM):
    """OpenAI LLM实现"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = openai.OpenAI(api_key=config.api_key)

    async def generate(self,
                       prompt: str,
                       temperature: Optional[float] = None) -> str:
        """单次生成

        Example:
            ```python
            # 生成创意内容
            story = await llm.generate(
                "写一个关于机器人和人类友谊的短故事",
                temperature=0.9
            )

            # 分析内容
            analysis = await llm.generate(
                "分析这段文字的关键论点：" + text,
                temperature=0.2  # 降低随机性，使输出更确定性
            )
            ```
        """
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.config.model_name,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=self._get_temperature(temperature),
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"OpenAI生成错误: {str(e)}")
            return ""

    async def chat(self,
                   messages: List[Dict[str, str]],
                   system_prompt: Optional[str] = None,
                   temperature: Optional[float] = None) -> str:
        """多轮对话

        Example:
            ```python
            messages = [
                {"role": "system", "content": "你是一位富有同理心的心理咨询师"},
                {"role": "user", "content": "我最近感到很焦虑"},
                {"role": "assistant", "content": "我理解你的感受..."},
                {"role": "user", "content": "是的，主要是工作压力大"}
            ]
            response = await llm.chat(messages)
            ```
        """
        try:
            # 构建完整的消息列表
            full_messages = []

            # 添加系统提示词
            if system_prompt:
                full_messages.append({
                    "role": "system",
                    "content": system_prompt
                })

            # 添加历史消息
            full_messages.extend(messages)

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.config.model_name,
                messages=full_messages,
                temperature=self._get_temperature(temperature),
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"OpenAI对话错误: {str(e)}")
            return ""