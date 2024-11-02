from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class Role:
    """角色定义"""
    name: str
    traits: Dict[str, float]
    background: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    emotions: Dict[str, float] = field(default_factory=dict)

    # 添加常用情绪类型
    COMMON_EMOTIONS = {
        "快乐", "悲伤", "愤怒", "恐惧", "惊讶",
        "期待", "信任", "厌恶", "中性"
    }

    # 添加特征值范围常量
    MIN_VALUE = 0.0
    MAX_VALUE = 1.0

    def __post_init__(self):
        """初始化后的数据验证"""
        self.validate_traits()
        self.validate_emotions()

    def validate_traits(self):
        """验证特征值是否在有效范围内"""
        invalid_traits = [
            (trait, value) for trait, value in self.traits.items()
            if not self.MIN_VALUE <= value <= self.MAX_VALUE
        ]
        if invalid_traits:
            traits_str = ", ".join(f"{t}:{v}" for t, v in invalid_traits)
            raise ValueError(f"特征值必须在{self.MIN_VALUE}到{self.MAX_VALUE}之间，"
                             f"无效的特征: {traits_str}")

    def validate_emotions(self):
        """验证情绪值是否在有效范围内"""
        invalid_emotions = [
            (emotion, value) for emotion, value in self.emotions.items()
            if not self.MIN_VALUE <= value <= self.MAX_VALUE
        ]
        if invalid_emotions:
            emotions_str = ", ".join(f"{e}:{v}" for e, v in invalid_emotions)
            raise ValueError(f"情绪强度必须在{self.MIN_VALUE}到{self.MAX_VALUE}之间，"
                             f"无效的情绪: {emotions_str}")

    def add_trait(self, trait: str, value: float):
        """添加新的性格特征"""
        if not self.MIN_VALUE <= value <= self.MAX_VALUE:
            raise ValueError(f"特征值 {value} 无效，必须在"
                             f"{self.MIN_VALUE}到{self.MAX_VALUE}之间")
        self.traits[trait] = value

    def update_emotion(self, emotion: str, intensity: float):
        """更新情绪状态"""
        if not self.MIN_VALUE <= intensity <= self.MAX_VALUE:
            raise ValueError(f"情绪强度 {intensity} 无效，必须在"
                             f"{self.MIN_VALUE}到{self.MAX_VALUE}之间")
        self.emotions[emotion] = intensity

    def get_dominant_emotion(self) -> Optional[str]:
        """获取当前主导情绪"""
        if not self.emotions:
            return None
        return max(self.emotions.items(), key=lambda x: x[1])[0]

    def clear_emotions(self):
        """清空所有情绪状态"""
        self.emotions.clear()

    def add_preference(self, category: str, preference: Any):
        """添加个人偏好"""
        self.preferences[category] = preference

    def get_preferences_by_category(self, category: str) -> Optional[Any]:
        """获取特定类别的偏好"""
        return self.preferences.get(category)

    def to_dict(self) -> Dict[str, Any]:
        """将角色信息转换为字典格式"""
        return {
            "name": self.name,
            "traits": self.traits.copy(),
            "background": self.background,
            "preferences": self.preferences.copy(),
            "emotions": self.emotions.copy()
        }

    def to_prompt(self) -> str:
        """生成用于AI模型的角色提示词"""
        # 性格特征描述
        traits_str = ", ".join([f"{k}程度为{v:.1f}" for k, v in self.traits.items()])

        # 情绪状态描述
        emotions_str = ""
        if self.emotions:
            emotions_str = "当前情绪: " + ", ".join(
                [f"{k}({v:.1f})" for k, v in self.emotions.items()]
            )

        # 偏好描述
        prefs_str = ""
        if self.preferences:
            prefs_str = "偏好: " + ", ".join(
                [f"{k}:{v}" for k, v in self.preferences.items()]
            )

        # 使用join拼接非空字段
        fields = [
            f"角色：{self.name}",
            f"特征：{traits_str}",
            f"背景：{self.background}"
        ]
        if emotions_str:
            fields.append(emotions_str)
        if prefs_str:
            fields.append(prefs_str)

        return "\n".join(fields)


if __name__ == "__main__":
    def create_story_character():
        """创建一个故事角色示例"""
        print("\n=== 创建故事角色 ===")

        # 创建一个AI助手角色
        assistant = Role(
            name="小艾",
            traits={
                "友好": 0.9,
                "幽默": 0.7,
                "耐心": 0.8,
                "好奇": 0.6
            },
            background="小艾是一个充满活力的AI助手，总是以积极乐观的态度帮助他人。"
                       "她热爱学习新知识，善于倾听，并能够用温和幽默的方式与人交流。"
        )

        print(f"角色'{assistant.name}' 创建成功!")
        print(f"性格特征: {assistant.traits}")
        print(f"背景故事: {assistant.background}")
        return assistant


    def demonstrate_emotional_changes(role: Role):
        """演示角色情绪变化"""
        print("\n=== 情绪变化演示 ===")

        print("1. 开心的日常状态:")
        role.update_emotion("快乐", 0.8)
        role.update_emotion("期待", 0.6)
        print(role.to_prompt())

        print("\n2. 遇到困难问题时:")
        role.clear_emotions()
        role.update_emotion("专注", 0.9)
        role.update_emotion("好奇", 0.7)
        print(role.to_prompt())

        print("\n3. 成功解决问题后:")
        role.clear_emotions()
        role.update_emotion("快乐", 0.9)
        role.update_emotion("满足", 0.8)
        print(role.to_prompt())

        print(f"\n主导情绪: {role.get_dominant_emotion()}")


    def demonstrate_preferences(role: Role):
        """演示角色偏好设置"""
        print("\n=== 个性化偏好演示 ===")

        # 添加各种类型的偏好
        role.add_preference("交流方式", ["耐心解释", "循序渐进", "类比说明"])
        role.add_preference("回答风格", "简洁友好")
        role.add_preference("专业领域", ["编程", "数学", "科普"])
        role.add_preference("语言风格", {
            "正式程度": 0.7,
            "幽默程度": 0.6,
            "专业程度": 0.8
        })

        print("已设置的偏好:")
        for category, pref in role.preferences.items():
            print(f"{category}: {pref}")

        print("\n生成的角色描述:")
        print(role.to_prompt())


    def demonstrate_role_adaptation(role: Role):
        """演示角色适应不同场景"""
        print("\n=== 场景适应演示 ===")

        print("1. 教学场景:")
        role.clear_emotions()
        role.update_emotion("耐心", 0.9)
        role.update_emotion("专注", 0.8)
        role.add_trait("教学倾向", 0.9)
        print(role.to_prompt())

        print("\n2. 闲聊场景:")
        role.clear_emotions()
        role.update_emotion("轻松", 0.8)
        role.update_emotion("愉快", 0.7)
        role.add_trait("幽默感", 0.8)
        print(role.to_prompt())


    # 运行完整演示
    print("=== AI助手角色演示 ===")

    # 创建角色
    ai_assistant = create_story_character()

    # 演示情绪变化
    demonstrate_emotional_changes(ai_assistant)

    # 演示偏好设置
    demonstrate_preferences(ai_assistant)

    # 演示场景适应
    demonstrate_role_adaptation(ai_assistant)

    print("\n=== 演示结束 ===")