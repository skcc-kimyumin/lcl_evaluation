from typing import List

from database.model.role import Role, UserRole
from sqlalchemy.orm import Session


def get_role_all(db: Session):
    return db.query(Role).all()


def get_role_by_user_id(db: Session, user_id: str):
    all_roles = get_role_all(db)
    role_id_key = {role.id: role.key for role in all_roles}
    roles = db.query(UserRole).filter(UserRole.user_id == user_id).all()
    role = [role_id_key[role.role_id] for role in roles]

    return role


def create_role(db: Session, user_id: str, role_list: List[str]):
    all_roles = get_role_all(db)
    role_id_key = {role.key: role.id for role in all_roles}
    for role_key in role_list:
        db_role = UserRole(user_id=user_id, role_id=role_id_key[role_key])
        db.add(db_role)
    db.flush()
    return role_list
