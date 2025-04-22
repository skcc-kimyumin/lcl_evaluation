from core.config import get_setting
from error.service_exception import ErrorCode, ServiceException
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from log.logging import get_logging

settings = get_setting()
logger = get_logging()


def set_error_handlers(app: FastAPI):
    @app.exception_handler(ServiceException)
    async def predefined_exception_handler(request: Request, exc: ServiceException):
        logger.error(str(exc))
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": exc.code.value,
                "code_name": exc.code.name,
                "detail": exc.detail,
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.error(str(exc))
        if exc.status_code < 200 or exc.status_code == 204 or exc.status_code == 304:
            return Response(status_code=exc.status_code)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": ErrorCode.UNEXPECTED_ERROR.value,
                "code_name": ErrorCode.UNEXPECTED_ERROR.name,
                "detail": exc.detail,
            },
        )

    @app.exception_handler(Exception)
    async def base_exception_handler(request: Request, exc: Exception):
        logger.error(str(exc))
        return JSONResponse(
            status_code=500,
            content={
                "code": ErrorCode.UNEXPECTED_ERROR.value,
                "code_name": ErrorCode.UNEXPECTED_ERROR.name,
                "detail": "Unexpected error",
            },
        )
