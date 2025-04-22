from typing import List

from pydantic import BaseModel, EmailStr


class UserRoles:
    common: str = "common"
    admin: str = "admin"
    superadmin: str = "superadmin"


class UserBase(BaseModel):
    user_id: str
    email: EmailStr
    username: str


class UserCreate(UserBase):
    password: str
    role: List[str]


class UserMutate(UserBase):
    password: str
    role: List[str]
    is_active: bool = True
