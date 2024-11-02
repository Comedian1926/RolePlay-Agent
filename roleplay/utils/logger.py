import logging
from datetime import datetime

class ChatLogger:
    """聊天日志记录器"""
    def __init__(self, log_file: str = None):
        self.logger = logging.getLogger("ChatLogger")
        self.logger.setLevel(logging.INFO)