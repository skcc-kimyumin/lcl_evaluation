from api.deps import DatabaseDep, MilvusDep
from fastapi import APIRouter, Depends, HTTPException
from langchain.memory import ConversationBufferMemory
from service.agent.agent_node import (
    workflow_builder1,
    workflow_builder2,
    workflow_builder3,
)
from service.model.agent import ChatRequest, ChatState

router = APIRouter()

# 임시
global_memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")


def get_memory():
    return global_memory


@router.post("/chat1")
async def chat(request: ChatRequest, db: DatabaseDep):
    try:
        # LangGraph 실행
        result = workflow_builder1(request, db)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat2")
async def chat2(request: ChatRequest, collection_name: str, db: DatabaseDep, milvus: MilvusDep):
    try:
        # LangGraph 실행
        result = workflow_builder2(request, collection_name, db, milvus)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat3")
async def chat3(request: ChatRequest, collection_name: str, db: DatabaseDep, milvus: MilvusDep, memory: ConversationBufferMemory = Depends(get_memory)):
    try:
        # LangGraph 실행
        result = workflow_builder3(request, collection_name, db, milvus, memory)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
