# 상태 클래스 정의
from functools import partial
from operator import itemgetter
from typing import Dict

import openai
from core.config import get_setting
from database.repository.chat_history import ChatHistory
from fastapi import HTTPException
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from pymilvus import MilvusClient
from service.agent.prompts import basic_system_prompt_with_vector_search
from service.model.agent import ChatRequest, ChatState
from service.vectordb.milvus import search_vectors_info
from sqlalchemy.orm import Session

settings = get_setting()

OPENAI_API_KEY = settings.OPENAI_API_KEY
USER_ID = settings.USER_ID

def workflow_builder1(request: ChatRequest, db: Session):
    graph = StateGraph(ChatState)

    # 노드 추가
    graph.add_node("input", input_node)
    graph.add_node("llm", llm_node)
    graph.add_node("output", partial(output_node, user_id=USER_ID, db=db))

    # 노드 간 연결 설정 (Edges)
    graph.set_entry_point("input")
    graph.add_edge("input", "llm")
    graph.add_edge("llm", "output")

    # 그래프 컴파일
    app_graph = graph.compile()

    # LangGraph 실행
    result = app_graph.invoke(request)

    return result


def workflow_builder2(request: ChatRequest, collection_name: str, db: Session, milvus: MilvusClient):
    graph = StateGraph(ChatState)

    # 노드 추가
    graph.add_node("input", input_node)
    graph.add_node("search_vector", partial(vector_node, collection_name=collection_name, milvus=milvus))
    graph.add_node("llm", partial(llm_node_lcel, user_id=USER_ID, db=db))
    graph.add_node("output", partial(output_node, user_id=USER_ID, db=db))

    # 노드 간 연결 설정 (Edges)
    graph.set_entry_point("input")
    graph.add_edge("input", "search_vector")

    graph.add_edge("search_vector", "llm")
    graph.add_edge("llm", "output")

    # 그래프 컴파일
    app_graph = graph.compile()

    # LangGraph 실행
    result = app_graph.invoke(request)

    return result


def workflow_builder3(request: ChatRequest, collection_name: str, db: Session, milvus: MilvusClient, memory: ConversationBufferMemory):
    graph = StateGraph(ChatState)
    # 노드 추가
    graph.add_node("input", input_node)
    graph.add_node("search_vector", partial(vector_node, collection_name=collection_name, milvus=milvus))
    graph.add_node("llm", partial(llm_node_lcel_with_memory, user_id=USER_ID, db=db, memory=memory))
    graph.add_node("output", partial(output_node, user_id=USER_ID, db=db))

    # 노드 간 연결 설정 (Edges)
    graph.set_entry_point("input")
    graph.add_edge("input", "search_vector")
    graph.add_edge("search_vector", "llm")
    graph.add_edge("llm", "output")

    # 그래프 컴파일
    app_graph = graph.compile()

    # LangGraph 실행
    result = app_graph.invoke(request)

    return result


def input_node(state: "ChatState") -> "ChatState":
    print(f"Input Node - Message received: {state.message}")
    return state


def vector_node(state: "ChatState", collection_name: str, milvus: MilvusClient):
    try:
        search_result = search_vectors_info(milvus=milvus, query=state.message, collection_name=collection_name)
        state.vector_result = search_result[0][0]["entity"]["text"]  # 벡터 검색 결과를 state에 저장
        return state
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def llm_node(state: "ChatState"):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": state.message}],
        )
        state.response = response.choices[0].message.content
        return state
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def llm_node_lcel(state: "ChatState", user_id: str, db: Session):
    try:
        # OpenAI LLM 사용
        llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)

        # 프롬프트 템플릿 정의
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", basic_system_prompt_with_vector_search),
                MessagesPlaceholder(variable_name="user_message"),
                ("assistant", f"Vector search result: {state.vector_result}"), # 벡터 검색 결과 활용
                # ("user", state.message),
            ]
        )

        # chain 생성
        chain = prompt | llm

        # 호출
        # result = chain.invoke() # placeholder 안쓸때
        result = chain.invoke({"user_message": [{"role": "user", "content": state.message}]})

        # 결과를 state에 반영
        state.response = result.content

        return state

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def llm_node_lcel_with_memory(state: "ChatState", user_id: str, db: Session, memory: ConversationBufferMemory):
    try:
        # OpenAI LLM 사용
        llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "친절히 답변해줘"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                ("assistant", f"Vector search result: {state.vector_result}"), # 벡터 검색 결과 활용
            ]
        )

        # 이전 memory 유지
        # if memory is None:
        #     memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

        # 메모리에 저장된 Dict를 "chat_history"라는 key로 값 전달, 위에 선언된 prompt에서 variable_name="chat_hisotry"에서 값 받아서 prompt에 대화이력을 넣어줌 
        runnable = RunnablePassthrough.assign(
            chat_history=RunnableLambda(memory.load_memory_variables)
            | itemgetter("chat_history")  # memory_key 와 동일하게 입력합니다.
        )

        # chain 생성
        chain = runnable | prompt | llm

        # 호출
        result = chain.invoke({"input": state.message})

        # 결과를 state에 반영
        state.response = result.content

        memory.save_context(
            {"human": state.message}, {"ai": state.response}
        )

        return state

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def output_node(state: "ChatState", user_id: str, db: Session) -> Dict:
    print(f"Output Node - Response: {state.response}")

    # 데이터 이력 저장
    chat_history = ChatHistory(
        user_id=user_id,
        message=state.message,
        response=state.response,
    )

    db.add(chat_history)
    db.commit()

    return {"response": state.response}
