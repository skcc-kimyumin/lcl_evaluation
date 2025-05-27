import io, csv
from core.config import get_setting
from log.logging import get_logging
from fastapi import APIRouter, File, HTTPException, UploadFile
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex
from azure.core.credentials import AzureKeyCredential
from langchain_core.documents import Document
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents.indexes.models import (
    SearchField, SearchFieldDataType,
    SimpleField, SearchableField
)
from azure.search.documents.indexes.models import (
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    HnswParameters
)

settings = get_setting()
logger = get_logging()

AZURE_SEARCH_ENDPOINT = settings.AZURE_SEARCH_ENDPOINT
AZURE_SEARCH_KEY = settings.AZURE_SEARCH_KEY
AZURE_OPENAI_ENDPOINT = settings.AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY = settings.AZURE_OPENAI_API_KEY
AZURE_EMBEDDING_API_VERSION = settings.AZURE_EMBEDDING_API_VERSION
AZURE_EMBEDDING_DEPLOYMENT = settings.AZURE_EMBEDDING_DEPLOYMENT

# Embedding 모델 초기화
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
    openai_api_version=AZURE_EMBEDDING_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    chunk_size=10
)

router = APIRouter()


@router.post("/create_index")
async def create_index_from_csv(index_name: str):
    try:
        # 0. 인덱스 생성 or 업데이트
        create_or_update_search_index(index_name)

        return {"message": f"Index '{index_name}' created/updated."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add_documents_from_csv")
async def add_documents_from_csv(index_name: str, file: UploadFile = File(...)):
    try:
        
        # 1. CSV 파일 읽기
        file_content = await file.read()
        file_str = file_content.decode("utf-8")
        reader = csv.DictReader(io.StringIO(file_str))

        # 2. 텍스트 데이터 추출 (vector 필드 제외)
        documents = []
        for row in reader:
            doc = Document(
                page_content=row.get("content", ""),
                metadata={
                    "id": row.get("id", ""),
                    "metadata": row.get("metadata", "")
                }
            )
            documents.append(doc)

        # 3. AzureSearch 클라이언트 생성
        client = get_azure_search_client(index_name)

        # 4. 문서 추가 (임베딩 자동 생성 포함)
        # add_documents: 내부에서 embed_query를 통해 임베딩 생성 후 업로드함
        client.add_documents(documents)

        return {"message": f"Index '{index_name}': Documents added successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
def get_azure_search_client(index_name: str):

    # Azure AI Search 클라이언트 생성
    return AzureSearch(
        azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
        azure_search_key=AZURE_SEARCH_KEY,
        index_name=index_name,
        embedding_function=embeddings.embed_query,
        fields=get_field_schema(index_name),
        search_type="hybrid"
    )

def get_field_schema(index_name):
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
            vector_search_dimensions=3072,
            vector_search_profile_name=f"myHnswProfile_{index_name}"
        ),
        SearchableField(
            name="metadata",
            type=SearchFieldDataType.String,
            searchable=True
        )
    ]

def create_or_update_search_index(index_name: str):
    # Azure Search 관리 클라이언트 초기화
    credential = AzureKeyCredential(AZURE_SEARCH_KEY)
    index_client = SearchIndexClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        credential=credential
    )

    # 현재 서비스에 존재하는 모든 인덱스명 출력
    index_names = [idx.name for idx in index_client.list_indexes()]
    print("***INDEXES:", index_names)
    
    # 기존에 동일한 이름의 인덱스 존재할 경우
    try:
        # 기존 인덱스 삭제
        index_client.delete_index(index_name)
        print('**********DELETED**********')
    except ResourceNotFoundError:
        pass  # 없는 경우 무시
    
    # 인덱스 스키마 정의
    fields = get_field_schema(index_name)
    
    # 벡터 검색 프로파일 설정: hnsw알고리즘
    vector_search_config = VectorSearch(
        profiles=[VectorSearchProfile(name=f"myHnswProfile_{index_name}", algorithm_configuration_name=f"myHnswConfig_{index_name}")],
        algorithms=[HnswAlgorithmConfiguration(name=f"myHnswConfig_{index_name}", 
                    parameters=HnswParameters(metric="cosine"))]
    )
    
    # SearchIndex 객체 생성
    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search_config
    )
    
    # 인덱스 생성/업데이트
    index_client.create_or_update_index(index)    
