from contextlib import asynccontextmanager

import uvicorn
from api.router import api_router
from core.config import get_setting
from database.database import Base, db_engine, get_db

# from database.initialize import init_db
from error.error_handler import set_error_handlers
from fastapi import FastAPI
from fastapi.routing import APIRoute
from log.logging import get_logging

settings = get_setting()
logger = get_logging()


def create_app():
    def custom_generate_unique_id(route: APIRoute) -> str:
        return f"{route.tags[0]}-{route.name}"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("initializing ...")
        Base.metadata.create_all(bind=db_engine)
        db = next(get_db())
        # milvus = get_milvus()
        # init_db(db)
        logger.info(
            f"{settings.APP_NAME}[{settings.APP_PORT}] service is ready and now running!!"
        )
        yield

    app = FastAPI(
        title=settings.APP_NAME,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        lifespan=lifespan,
        generate_unique_id_function=custom_generate_unique_id,
    )
    app.include_router(api_router, prefix=settings.API_V1_STR)
    set_error_handlers(app)

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "__main__:app",
        host="0.0.0.0",
        port=int(settings.APP_PORT),
        workers=int(settings.WORKER),
    )
