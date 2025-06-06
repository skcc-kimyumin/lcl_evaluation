# from api.deps import DatabaseDep, MilvusDep
from api.deps import DatabaseDep
from fastapi import APIRouter, Depends, HTTPException
from langchain.memory import ConversationBufferMemory
from service.agent.agent_node import (
    best_pratice,
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
async def chat2(request: ChatRequest, db: DatabaseDep):
    try:
        # LangGraph 실행
        result = workflow_builder2(request, db)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat3")
async def chat3(request: ChatRequest, db: DatabaseDep, memory: ConversationBufferMemory = Depends(get_memory)):
    try:
        # LangGraph 실행
        result = workflow_builder3(request, db, memory)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat4")
async def chat4(request: ChatRequest, collection_name: str, db: DatabaseDep):
    try:
        # LangGraph 실행
        result = best_pratice(request, collection_name, db)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))