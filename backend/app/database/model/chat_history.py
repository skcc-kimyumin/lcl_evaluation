from database.database import Base
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func


class ChatHistory(Base):
    __tablename__ = 'chat_history'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Text)  # 사용자 ID (ForeignKey로 설정 가능)
    message = Column(Text)  # 사용자 메시지
    response = Column(Text)  # AI 응답
    timestamp = Column(DateTime, server_default=func.now())  # 메시지 타임스탬프
    chat_session_id = Column(Integer, index=True)  # 세션 ID (하나의 대화 흐름을 구분하는 데 사용)

    # ChatSession 테이블과의 관계 (ForeignKey)
    # chat_session_id는 실제로 연결된 다른 테이블과 연결될 수 있음
    # chat_session = relationship("ChatSession", back_populates="messages")

    def __repr__(self):
        return f"<ChatHistory(id={self.id}, user_id={self.user_id}, message={self.message}, response={self.response})>"
