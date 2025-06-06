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
# from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph
from log.logging import get_logging
# from pymilvus import MilvusClient
from service.agent.prompts import (
    basic_system_prompt_with_vector_search,
    generator_prompt,
    hyde_prompt,
    planner_prompt,
    reflector_prompt,
    rewriter_prompt,
)
from service.model.agent import ChatRequest, ChatState
from service.vectordb.aisearch import search_vectors_info
from sqlalchemy.orm import Session

settings = get_setting()
logger = get_logging()

OPENAI_API_KEY = settings.OPENAI_API_KEY
USER_ID = settings.USER_ID
DEPLOYMENT = settings.DEPLOYMENT
MODEL_NAME = settings.MODEL_NAME
AZURE_OPENAI_ENDPOINT = settings.AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY = settings.AZURE_OPENAI_API_KEY
AZURE_OPENAI_API_VERSION = settings.AZURE_OPENAI_API_VERSION
INDEX_NAME = 'tax'


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


def workflow_builder2(request: ChatRequest, db: Session):
    graph = StateGraph(ChatState)

    # 노드 추가
    graph.add_node("input", input_node)
    graph.add_node("search_vector", partial(vector_node))
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


def workflow_builder3(request: ChatRequest, db: Session, memory: ConversationBufferMemory):
    graph = StateGraph(ChatState)
    # 노드 추가
    graph.add_node("input", input_node)
    graph.add_node("search_vector", vector_node)
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


def vector_node(state: "ChatState"):
    try:
        search_result = search_vectors_info(query=state.message, index_name=INDEX_NAME)
        state.vector_result = search_result[0]['text']  # 벡터 검색 결과를 state에 저장
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
        # llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
        # AzureOpenAI LLM 사용
        llm = AzureChatOpenAI(
            deployment_name=DEPLOYMENT,
            model_name=MODEL_NAME,  
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION
        )  

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
        # llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
        # AzureOpenAI LLM 사용  
        llm = AzureChatOpenAI(
            deployment_name=DEPLOYMENT,
            model_name=MODEL_NAME,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION
        )

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


def best_pratice(request, db) -> Dict:
    # LLM 설정
    llm = AzureChatOpenAI(
        deployment_name=DEPLOYMENT,
        model_name=MODEL_NAME,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION
    )

    def get_next_step_from_plan(state):
        return state.get("next_step", "generate")    

    # 1. Planner Agent
    planner_prompt_template = PromptTemplate.from_template(planner_prompt)
    planner_chain = planner_prompt_template | llm

    def planner_node(state):
        logger.info("=================Planner node 시작 =================")
        query_for_planner = state.get("rewritten_query") or state["query"]
        output = planner_chain.invoke({"query": query_for_planner}).content
        lines = output.strip().splitlines()
        plan = "\n".join([line for line in lines if not line.lower().startswith("다음 단계:")])
        next_step_line = next((line for line in lines if line.lower().startswith("다음 단계:")), None)
        next_step = next_step_line.split(":", 1)[1].strip().lower() if next_step_line else "generate"
        logger.info(f"Plan:\n{plan}")
        logger.info(f"Next Step: {next_step}")
        return {**state, "plan": plan, "next_step": next_step}


    # 2. Rewriter Agent
    rewriter_prompt_template = PromptTemplate.from_template(rewriter_prompt)
    rewriter_chain = rewriter_prompt_template | llm

    def rewriter_node(state):
        logger.info("=================Rewriter node 시작 =================")
        rewritten = rewriter_chain.invoke({"query": state["query"]}).content
        logger.info(f"Rewritten Query: {rewritten}")
        return {**state, "rewritten_query": rewritten, "next_step": "retrieve"}    



   # 3. HyDE Agent
    hyde_prompt_template = PromptTemplate.from_template(hyde_prompt)
    hyde_chain = hyde_prompt_template | llm

    def hyde_node(state):
        logger.info("=================HyDE node 시작 =================")
        query_for_hyde = state.get("rewritten_query") or state["query"]
        hypothetical_doc = hyde_chain.invoke({"query": query_for_hyde}).content
        logger.info(f"Hypothetical Document: {hypothetical_doc}")
        return {**state, "hypothetical_doc": hypothetical_doc, "next_step": "retrieve"}



    # 4. Retriever Agent
    def retriever_node(state):
        logger.info("=================Retriever node 시작 =================")
        try:
            query_for_retrieve = state.get("hypothetical_doc")
            search_result = search_vectors_info(
                query=query_for_retrieve,
                index_name=INDEX_NAME
            )
            parsed = search_result[0]['text']
            return {**state, "documents": parsed, "next_step": "generate"}
        except Exception as e:
            logger.exception("Retriever Error")
            raise HTTPException(status_code=500, detail=str(e))



    # 5. Generator Agent
    generator_prompt_template = PromptTemplate.from_template(generator_prompt)
    generator_chain = generator_prompt_template | llm

    def generator_node(state):
        logger.info("=================Generator node 시작 =================")
        query_for_generator = state.get("rewritten_query") or state["query"]
        response = generator_chain.invoke({
            "query": query_for_generator,
            "documents": "\n".join(state.get("documents", []))
        }).content
        logger.info(f"Response:\n{response}")
        return {**state, "response": response, "next_step": "reflection"}



    # 6. Reflector Agent
    reflector_prompt_template = PromptTemplate.from_template(reflector_prompt)
    reflector_chain = reflector_prompt_template | llm

    def reflector_node(state):
        logger.info("=================Reflector node 시작 =================")
        query_for_reflector = state.get("rewritten_query") or state["query"]
        feedback = reflector_chain.invoke({
            "query": query_for_reflector,
            "response": state["response"]
        }).content
        next_step = "end" if "OK" in feedback else "generate"
        logger.info(f"Feedback:\n{feedback}")
        logger.info(f"Feedback judged next_step = {next_step}")
        return {**state, "feedback": feedback, "next_step": next_step}



    # 7. 그래프 구성
    graph_builder = StateGraph(dict)
    graph_builder.add_node("plan", RunnableLambda(planner_node))
    graph_builder.add_node("rewrite", RunnableLambda(rewriter_node))
    graph_builder.add_node("hyde", RunnableLambda(hyde_node))
    graph_builder.add_node("retrieve", RunnableLambda(retriever_node))
    graph_builder.add_node("generate", RunnableLambda(generator_node))
    graph_builder.add_node("reflection", RunnableLambda(reflector_node))

    graph_builder.set_entry_point("plan")
    graph_builder.add_conditional_edges("plan", get_next_step_from_plan, {
        "rewrite": "rewrite",
        "hyde": "hyde",
        "generate": "generate",
        "end": END,
    })
    graph_builder.add_edge("rewrite", "hyde")
    graph_builder.add_edge("hyde", "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", "reflection")
    graph_builder.add_conditional_edges("reflection", get_next_step_from_plan, {
        "generate": "generate",
        "end": END,
    })



    # 8. 실행
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
