# RolePlay-Agent

RolePlay-Agent 是一个灵活的多角色对话框架，支持创建具有个性化特征的AI角色，并实现多角色之间的自然对话交互。

## 特性

- 🎭 支持多角色交互
- 🧠 基于大语言模型的对话生成
- 💭 简单的记忆系统
- 🎯 可自定义的角色特征
- 🔌 支持多种LLM提供商
- 📝 内置日志记录系统

## 快速开始

### 安装

```bash
pip install roleplay-roleplay
```

### 基础使用

```python
from roleplay.core import Agent, Personality, ChatEnvironment
from roleplay.llm.backends import OpenAILLM

# 创建角色并开始对话...
```

## 开发

### 设置开发环境

```bash
git clone https://github.com/Comedian1926/RolePlay-Agent.git
cd RolePlay-Agent
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -e ".[dev]"
```

## 许可证

MIT License