from typing import List, Optional

from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel


class ChatState(BaseModel):
    message: str
    response: str = ""
    message_history: List[str] = []  # 이전 메시지들 (채팅 히스토리)
    vector_result: str = ""
    # memory: Optional[ConversationBufferMemory] = None  # 기본값 None


class ChatRequest(BaseModel):
    message: str
