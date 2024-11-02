from typing import List, Dict, Optional, Any, Callable
import asyncio
import json
import httpx
from datetime import datetime

from ..base import BaseLLM, LLMConfig


class CustomLLMConfig(LLMConfig):
    """自定义LLM配置

    扩展基础配置，添加自定义LLM所需的特殊配置项
    """

    def __init__(self,
                 model_name: str,
                 api_url: str,
                 request_timeout: float = 30.0,
                 custom_headers: Optional[Dict[str, str]] = None,
                 **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        self.api_url = api_url
        self.request_timeout = request_timeout
        self.custom_headers = custom_headers or {}


class CustomLLM(BaseLLM):
    """自定义LLM基础实现

    提供基础的HTTP请求处理和响应解析框架，
    用户可以通过继承此类来实现自己的LLM集成。
    """

    def __init__(self, config: CustomLLMConfig):
        super().__init__(config)
        self.config: CustomLLMConfig = config
        self.client = httpx.AsyncClient(
            timeout=config.request_timeout,
            headers=config.custom_headers
        )

        # 自定义处理函数
        self.pre_process_func: Optional[Callable] = None
        self.post_process_func: Optional[Callable] = None

    async def generate(self,
                       prompt: str,
                       temperature: Optional[float] = None) -> str:
        """生成回复"""
        try:
            # 构建请求数据
            data = self._build_generate_request(prompt, temperature)

            # 发送请求
            response = await self._make_request(data)

            # 处理响应
            return self._process_response(response)

        except Exception as e:
            print(f"CustomLLM生成错误: {str(e)}")
            return ""

    async def chat(self,
                   messages: List[Dict[str, str]],
                   system_prompt: Optional[str] = None,
                   temperature: Optional[float] = None) -> str:
        """对话方法"""
        try:
            # 构建请求数据
            data = self._build_chat_request(messages, system_prompt, temperature)

            # 发送请求
            response = await self._make_request(data)

            # 处理响应
            return self._process_response(response)

        except Exception as e:
            print(f"CustomLLM对话错误: {str(e)}")
            return ""

    def _build_generate_request(self,
                                prompt: str,
                                temperature: Optional[float]) -> Dict[str, Any]:
        """构建生成请求的数据

        子类应该重写此方法以适配特定的API格式
        """
        return {
            "prompt": prompt,
            "temperature": self._get_temperature(temperature),
            "max_tokens": self.config.max_tokens,
            "model": self.config.model_name
        }

    def _build_chat_request(self,
                            messages: List[Dict[str, str]],
                            system_prompt: Optional[str],
                            temperature: Optional[float]) -> Dict[str, Any]:
        """构建对话请求的数据

        子类应该重写此方法以适配特定的API格式
        """
        return {
            "messages": messages,
            "system_prompt": system_prompt,
            "temperature": self._get_temperature(temperature),
            "max_tokens": self.config.max_tokens,
            "model": self.config.model_name
        }

    async def _make_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """发送HTTP请求

        子类可以重写此方法以实现自定义的请求逻辑
        """
        try:
            response = await self.client.post(
                self.config.api_url,
                json=data
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"API请求失败: {str(e)}")

    def _process_response(self, response: Dict[str, Any]) -> str:
        """处理API响应

        子类应该重写此方法以适配特定的响应格式
        """
        # 默认实现假设响应中有'content'字段
        return response.get('content', '')

    def set_pre_process(self, func: Callable):
        """设置请求预处理函数"""
        self.pre_process_func = func

    def set_post_process(self, func: Callable):
        """设置响应后处理函数"""
        self.post_process_func = func


# 示例：实现一个具体的自定义LLM
class ExampleCustomLLM(CustomLLM):
    """示例自定义LLM实现"""

    def _build_generate_request(self,
                                prompt: str,
                                temperature: Optional[float]) -> Dict[str, Any]:
        """适配特定API的请求格式"""
        return {
            "input": {
                "prompt": prompt,
                "parameters": {
                    "temperature": self._get_temperature(temperature),
                    "max_length": self.config.max_tokens,
                    "model_name": self.config.model_name
                }
            }
        }

    def _build_chat_request(self,
                            messages: List[Dict[str, str]],
                            system_prompt: Optional[str],
                            temperature: Optional[float]) -> Dict[str, Any]:
        """适配特定API的对话请求格式"""
        formatted_messages = []

        if system_prompt:
            formatted_messages.append({
                "role": "system",
                "content": system_prompt
            })

        formatted_messages.extend(messages)

        return {
            "input": {
                "messages": formatted_messages,
                "parameters": {
                    "temperature": self._get_temperature(temperature),
                    "max_length": self.config.max_tokens,
                    "model_name": self.config.model_name
                }
            }
        }

    def _process_response(self, response: Dict[str, Any]) -> str:
        """处理特定API的响应格式"""
        try:
            return response["output"]["text"]
        except KeyError:
            raise Exception("响应格式错误")


# 使用示例
async def custom_llm_example():
    # 配置
    config = CustomLLMConfig(
        model_name="custom-model",
        api_url="http://api.example.com/v1/generate",
        request_timeout=30.0,
        custom_headers={
            "Authorization": "Bearer your-api-key",
            "Content-Type": "application/json"
        }
    )

    # 创建LLM实例
    llm = ExampleCustomLLM(config)

    # 添加自定义的预处理函数
    def pre_process(data: Dict[str, Any]) -> Dict[str, Any]:
        data["timestamp"] = datetime.now().isoformat()
        return data

    llm.set_pre_process(pre_process)

    # 添加自定义的后处理函数
    def post_process(response: str) -> str:
        return response.strip()

    llm.set_post_process(post_process)

    # 使用generate方法
    response = await llm.generate(
        "讲一个故事",
        temperature=0.8
    )
    print(f"生成结果: {response}")

    # 使用chat方法
    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么我可以帮你的吗？"},
        {"role": "user", "content": "讲个故事吧"}
    ]

    response = await llm.chat(
        messages=messages,
        system_prompt="你是一个擅长讲故事的人"
    )
    print(f"对话结果: {response}")