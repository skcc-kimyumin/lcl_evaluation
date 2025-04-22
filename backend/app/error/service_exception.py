from enum import Enum

from core.config import get_setting
from fastapi import HTTPException

settings = get_setting()


class ErrorCode(Enum):
    # 400 Bad Request
    NOT_DEFINED = 1000
    DUPLICATE_ENTITY = 1001
    UNSUPPORTED_FILE_TYPE = 1002
    INPUT_TOKEN_EXCEEDED = 1101
    FILE_LIMIT_EXCEEDED = 1003

    # 401 Unauthorized
    NO_TOKEN = 2001
    TOKEN_EXPIRED = 2002
    INVALID_TOKEN = 2003

    # 403 Forbidden
    ACCESS_DENIED = 3000
    OFFICIAL_UPDATE_DENIED = 3001

    # 404 Not Found
    RESOURCE_NOT_FOUND = 4001

    # 409 Conflict
    DUPLICATE_VALUE = 5001

    # 429 Too Many Requests
    TOO_MANY_REQUEST = 6101

    # 500 Internal Server Error
    UNEXPECTED_ERROR = 9000
    CONNECTION_ERROR = 9001
    MODEL_TIMEOUT = 9101
    TOKEN_LENGTH_LIMIT = 9102
    TOKEN_REQUEST_LIMIT_PER_MINUTE = 9103


class ServiceException(HTTPException):
    def __init__(
        self,
        status_code: int,
        error_code: ErrorCode,
        detail: str,
        src: str = settings.APP_NAME,
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.code = error_code
        self.src = src
