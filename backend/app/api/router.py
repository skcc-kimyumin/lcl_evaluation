# from api.routes import agent, faiss, milvus, rss
from api.routes import agent, faiss, rss
from fastapi import APIRouter

api_router = APIRouter()
# api_router.include_router(milvus.router, prefix="/milvus", tags=["milvus"])  # Milvus 라우터 주석 처리
api_router.include_router(faiss.router, prefix="/faiss", tags=["faiss"])
api_router.include_router(agent.router, prefix="/agent", tags=["agent"])
api_router.include_router(rss.router, prefix="/rss", tags=["rss"])
