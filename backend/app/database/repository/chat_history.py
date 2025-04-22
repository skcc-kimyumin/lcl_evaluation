from database.model.chat_history import ChatHistory
from sqlalchemy.orm import Session

# def get_user_by_user_id(db: Session, user_id: str):
#     user = db.query(User).filter(User.user_id == user_id).first()
#     return user


# def get_user_all(db: Session):
#     return db.query(User).all()


def create_history(
    db: Session, user_id: str,  message: str, response: str
):
    db_chat_history = ChatHistory(
        user_id=user_id,
        message=message,
        response=response,  # 실제 응답을 받는 코드 추가
    )
    db.add(db_chat_history)
    db.flush()
    db.refresh(db_chat_history)
    return db_chat_history


def get_histories_by_user(db: Session, user_id: str):
    chat_state_list = []
    histories = db.query(ChatHistory).filter(ChatHistory.user_id == user_id).all()

    for history in histories:
        chat_state_list.append(f"User: {history.message}")
        chat_state_list.append(f"Assistant: {history.response}")

    return chat_state_list
