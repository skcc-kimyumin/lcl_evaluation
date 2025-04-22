from typing import List, Optional

from core.security import get_password_hash
from database.repository.role import create_role, get_role_by_user_id
from database.repository.user import create_user, get_user_by_user_id
from sqlalchemy.orm import Session


def get_user_info(db: Session, user_id: str):
    user = get_user_by_user_id(db, user_id)
    role = get_role_by_user_id(db, user.id)
    user.role = role

    return {
        "id": user.id,
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "role": role,
        "is_active": user.is_active,
    }


def create_user_with_role(
    db: Session,
    user_id: str,
    email: str,
    username: str,
    company: str,
    password: str,
    role: List[str],
    is_active: bool = True,
):
    hashed_password = get_password_hash(password)
    db_user = create_user(
        db, user_id, email, username, company, hashed_password, is_active
    )
    db_role = create_role(db, db_user.id, role)
    user_info = {
        "id": db_user.id,
        "user_id": db_user.user_id,
        "username": db_user.username,
        "company": db_user.company,
        "email": db_user.email,
        "role": db_role,
        "is_active": db_user.is_active,
    }
    db.commit()
    return user_info
