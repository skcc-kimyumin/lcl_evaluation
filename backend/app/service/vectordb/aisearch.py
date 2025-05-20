from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from core.config import get_setting
from log.logging import get_logging
from fastapi import HTTPException
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    SearchField, SearchFieldDataType,
    SimpleField, SearchableField
)

settings = get_setting()
logger = get_logging()

AZURE_SEARCH_ENDPOINT = settings.AZURE_SEARCH_ENDPOINT
AZURE_SEARCH_KEY = settings.AZURE_SEARCH_KEY
AZURE_OPENAI_ENDPOINT = settings.AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY = settings.AZURE_OPENAI_API_KEY
AZURE_OPENAI_API_VERSION = settings.AZURE_OPENAI_API_VERSION
DEPLOYMENT = settings.DEPLOYMENT


# Embedding 모델 초기화
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=DEPLOYMENT,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY
)

    
def get_azure_search_client(index_name: str):

    # Azure AI Search 클라이언트 생성
    return AzureSearch(
        azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
        azure_search_key=AZURE_SEARCH_KEY,
        index_name=index_name,
        embedding_function=embeddings.embed_query,
        fields=get_field_schema(),
        search_type="hybrid"
    )

def get_field_schema():
    return [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="myHnswProfile"
        ),
        SearchableField(
            name="metadata",
            type=SearchFieldDataType.String,
            searchable=True
        )
    ]
    
def search_vectors_info(query: str, index_name: str, k: int = 5):
    try:
        # 1. 질문 벡터화 (Azure OpenAI 임베딩 사용)
        query_vector = embeddings.embed_query(query)
        
        # 2. 벡터 쿼리 생성
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=k,
            fields="content_vector"  # 인덱스에 정의된 벡터 필드명
        )
        
        # 3. Azure Search 클라이언트 초기화
        search_client = get_azure_search_client(index_name).client  # 내부 SearchClient 추출
        
        # 4. 벡터 검색 실행
        results = search_client.search(
            vector_queries=[vector_query],
            select=["content", "metadata"],  # 반환 필드 지정
        )
        
        # 5. 결과 포맷팅
        return [{
            "text": result["content"],
            "metadata": result.get("metadata"),
            "score": result["@search.score"]
        } for result in results]
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))