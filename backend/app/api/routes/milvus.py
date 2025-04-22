import csv
import io

import numpy as np
import openai
from api.deps import MilvusDep
from core.config import get_setting
from fastapi import APIRouter, File, HTTPException, UploadFile
from pymilvus import DataType
from service.vectordb.milvus import search_vectors_info

settings = get_setting()

# OpenAI API 키 설정
OPENAI_API_KEY = settings.OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

router = APIRouter()

# milvus lite docs -> https://milvus.io/docs/milvus_lite.md


# 컬렉션 및 인덱스 생성
@router.post("/create_collection_from_csv")
async def create_collection_from_csv(collection_name: str, milvus: MilvusDep, file: UploadFile = File(...)):
    try:
        # CSV 파일을 읽고 벡터화하기
        file_content = await file.read()
        file_str = file_content.decode("utf-8")
        reader = csv.DictReader(io.StringIO(file_str))

        # 텍스트 데이터를 추출하고 벡터화
        text_data = []
        for row in reader:
            text = " ".join([value for key, value in row.items() if key != "vector"])  # 벡터 필드 제외하고 텍스트 합침
            text_data.append(text)

        # OpenAI를 사용하여 텍스트 벡터화
        vectors = []
        for text in text_data:
            response = openai.embeddings.create(
                model="text-embedding-ada-002",
                input=text,
                encoding_format="float"
            )
            vectors.append(np.array(response.model_dump()['data'][0]['embedding'], dtype=np.float32))

        # Milvus 컬렉션과 스키마 생성
        dim = len(vectors[0])  # 벡터 차원

        # 3.1. Create schema
        schema = milvus.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )

        # 3.2. Add fields to schema
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=3000)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1536)

        index_params = milvus.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_name="vector_index",
            index_type="IVF_FLAT",
            metric_type="L2",
            # HNSW는 milvus_lite에서 지원 x
            # index_type="HNSW",  # Type of the index to create
            # params={
            #     "M": 64,  # Maximum number of neighbors each node can connect to in the graph
            #     "efConstruction": 100  # Number of candidate neighbors considered for connection during index construction
            # }  # Index building params
        )

        collection = milvus.create_collection(
            collection_name=collection_name,
            dimension=1536,
            schema=schema,
            index_params=index_params,
        )

        # 데이터 삽입
        ids = [i for i in range(len(text_data))]
        data = [{"id": ids[i], "vector": vectors[i], "text": text_data[i]} for i in range(len(vectors))]
        milvus.insert(collection_name=collection_name, data=data)

        return {"message": f"Collection '{collection_name}' created and vectors added successfully."}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/search_vectors")
async def search_vectors(milvus: MilvusDep, query: str, collection_name: str, k: int = 5):
    try:
        result = search_vectors_info(milvus=milvus, query=query, collection_name=collection_name, k=5)
        return {"results": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
