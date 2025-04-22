from urllib.parse import unquote

from core.config import get_setting
from fastapi import Depends
from fastapi.security import APIKeyHeader
from passlib.context import CryptContext

settings = get_setting()
pwd_context = CryptContext(schemes=["bcrypt"])

username_header = APIKeyHeader(
    name="username", scheme_name="username", auto_error=False
)
user_id_header = APIKeyHeader(name="user_id", scheme_name="user_id", auto_error=False)
email_header = APIKeyHeader(name="email", scheme_name="email", auto_error=False)
role_header = APIKeyHeader(name="role", scheme_name="role", auto_error=False)


# apigateway 거쳐서 들어온 요청의 header에 있는 정보를 가져옴
def get_current_user(
    username=Depends(username_header),
    user_id=Depends(user_id_header),
    email=Depends(email_header),
    role=Depends(role_header),
):
    try:
        return {
            "user_id": user_id,
            "email": email,
            "username": unquote(username),
            "role": role.split(","),
        }
    except Exception as e:
        raise e


def get_password_hash(password):
    return pwd_context.hash(password)
