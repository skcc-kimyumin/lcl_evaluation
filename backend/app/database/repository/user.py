from database.model.user import User
from sqlalchemy.orm import Session


def get_user_by_user_id(db: Session, user_id: str):
    user = db.query(User).filter(User.user_id == user_id).first()
    return user


def get_user_all(db: Session):
    return db.query(User).all()


def create_user(
    db: Session, user_id: str, email: str, username: str, hashed_password: str
):
    db_user = User(
        user_id=user_id,
        email=email,
        username=username,
        hashed_password=hashed_password,
        is_active=True,
    )
    db.add(db_user)
    db.flush()
    db.refresh(db_user)
    return db_user
