# 상태 클래스 정의
from functools import partial
from operator import itemgetter
from typing import Dict

import openai
from core.config import get_setting
from database.repository.chat_history import ChatHistory
from fastapi import HTTPException
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from log.logging import get_logging
from pymilvus import MilvusClient
from service.agent.prompts import basic_system_prompt_with_vector_search
from service.model.agent import ChatRequest, ChatState
from service.vectordb.milvus import search_vectors_info
from sqlalchemy.orm import Session

settings = get_setting()
logger = get_logging()

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


def best_pratice(request, collection_name: str, db, milvus) -> Dict:
    # LLM 설정
    llm = ChatOpenAI(model="gpt-4")

    def get_next_step_from_plan(state):
        return state.get("next_step", "generate")    

    # 0. Planner Agent (가장 먼저 실행)
    planner_prompt = PromptTemplate.from_template(
        """사용자의 질문: {query}

다음 작업을 계획하고, 다음 단계 하나를 결정해줘.
가능한 다음 단계는 retrieve, rewrite, retrieve, generate, reflect, end 중 하나야.
- 인사나 일상 대화 같은 간단한 질문이면 generate를 선택해.
- 쿼리 정제가 필요하면 rewrite를 선택해.
- 정보 검색이 필요하면 retrieve를 선택해.

형식:
계획: <계획 내용>
다음 단계: <retrieve|rewrite|generate>
""")
    planner_chain = LLMChain(llm=llm, prompt=planner_prompt)

    def planner_node(state):
        logger.info("=================Planner node 시작 =================")
        query_for_planner = state.get("rewritten_query") or state["query"]
        output = planner_chain.run(query=query_for_planner)
        lines = output.strip().splitlines()
        plan = "\n".join([line for line in lines if not line.lower().startswith("다음 단계:")])
        next_step_line = next((line for line in lines if line.lower().startswith("다음 단계:")), None)
        next_step = next_step_line.split(":", 1)[1].strip().lower() if next_step_line else "generate"
        logger.info(f"Plan:\n{plan}")
        logger.info(f"➡️ Next Step: {next_step}")
        return {**state, "plan": plan, "next_step": next_step}

    # 1. Rewriter Agent, 일상질문일 경우 해당 단계 없이 바로 generator
    rewriter_prompt = PromptTemplate.from_template(
        "사용자의 원래 질문: {query}\n이 질문을 더 명확하고 이해하기 쉽게 다시 표현해줘."
    )
    rewriter_chain = LLMChain(llm=llm, prompt=rewriter_prompt)

    def rewriter_node(state):
        logger.info("=================Rewriter node 시작 =================")
        rewritten = rewriter_chain.run(query=state["query"])
        logger.info(f"🔁 Rewritten Query: {rewritten}")
        return {**state, "rewritten_query": rewritten, "next_step": "retrieve"}    

    # 2. Retriever Agent
    def retriever_node(state):
        logger.info("=================Retriever node 시작 =================")
        try:
            search_result = search_vectors_info(milvus=milvus, query=state["query"], collection_name=collection_name)
            parsed = search_result[0][0]["entity"]["text"]
            # logger.info(f"Retrieved Document:\n{parsed}")
            return {**state, "documents": parsed, "next_step": "generate"}
        except Exception as e:
            logger.exception("Retriever Error")
            raise HTTPException(status_code=500, detail=str(e))

    # 3. Generator Agent
    generator_prompt = PromptTemplate.from_template(
        "사용자 질문: {query}\n관련 문서: {documents}\n이 정보를 바탕으로 응답을 생성해줘."
    )
    generator_chain = LLMChain(llm=llm, prompt=generator_prompt)

    def generator_node(state):
        logger.info("=================Generator node 시작 =================")
        query_for_generator = state.get("rewritten_query") or state["query"]
        response = generator_chain.run(
            query=query_for_generator,
            documents="\n".join(state.get("documents", []))
        )
        logger.info(f"📝 Response:\n{response}")
        return {**state, "response": response, "next_step": "reflect"}

    # 4. Reflector Agent
    reflector_prompt = PromptTemplate.from_template(
        """질문: {query}\n 응답: {response}\n
        질문이 인사나 일상 대화 같은 간단한 질문이면 무조건 OK라고 답변해줘.
        그 외의 질문이라면 질문에 대한 답변을 평가하는게 너의 역할이야.
        부족하거나 개선할 점이 있다면 설명하고, 괜찮은 답변이라면 OK라고 답변해줘."""
    )
    reflector_chain = LLMChain(llm=llm, prompt=reflector_prompt)

    def reflector_node(state):
        logger.info("=================Reflector node 시작 =================")
        query_for_reflector = state.get("rewritten_query") or state["query"]
        feedback = reflector_chain.run(response=state["response"], query=query_for_reflector)
        next_step = "end" if "OK" in feedback else "generate"
        logger.info(f"Feedback:\n{feedback}")
        logger.info(f"➡️ Feedback judged next_step = {next_step}")
        return {**state, "feedback": feedback, "next_step": next_step}

    # 그래프 구성
    graph_builder = StateGraph(dict)
    graph_builder.add_node("plan", RunnableLambda(planner_node))
    graph_builder.add_node("rewrite", RunnableLambda(rewriter_node))
    graph_builder.add_node("retrieve", RunnableLambda(retriever_node))
    graph_builder.add_node("generate", RunnableLambda(generator_node))
    graph_builder.add_node("reflect", RunnableLambda(reflector_node))

    graph_builder.set_entry_point("plan")
    graph_builder.add_conditional_edges("plan", get_next_step_from_plan, {
        "rewrite": "rewrite",
        "retrieve": "retrieve",
        "generate": "generate",
        "reflect": "reflect",
        "end": END,
    })
    graph_builder.add_edge("rewrite", "plan")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", "reflect")
    graph_builder.add_conditional_edges("reflect", get_next_step_from_plan, {
        "generate": "generate",
        "end": END,
    })

    # 실행
    try:
        app = graph_builder.compile()
        initial_state = {"query": request.message}
        result = app.invoke(initial_state)
    except Exception as e:
        logger.exception("Invoke error")
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(f"Final Response: {result['response']}")
    logger.info(f"Final Feedback: {result.get('feedback', 'N/A')}")
    return result
