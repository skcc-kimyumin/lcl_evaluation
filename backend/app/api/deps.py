from typing import Annotated, Any

# import redis
from core.config import get_setting
from core.security import get_current_user
from database.database import get_db

# from database.redis import get_redis
from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader
from service.model.user import UserBase, UserRoles
# from vectordb.initialize import get_milvus

settings = get_setting()

api_key_header = APIKeyHeader(name="Authorization")


# --- Redis ---
# RedisDep = Annotated[redis.StrictRedis, Depends(get_redis)]

# --- Milvus ---
# MilvusDep = Annotated[Any, Depends(get_milvus)]

# --- Database ---
DatabaseDep = Annotated[Any, Depends(get_db)]


# --- User Data ---
CurrentUserDep = Annotated[UserBase, Depends(get_current_user)]


# --- Access control ---
default_admin_roles = [
    UserRoles.admin,
    UserRoles.superadmin,
]
superadmin_roles = [UserRoles.superadmin]