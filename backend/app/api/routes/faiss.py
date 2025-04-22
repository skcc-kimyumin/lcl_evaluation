import csv
from io import StringIO

import faiss
import numpy as np
import openai
from core.config import get_setting
from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter()
settings = get_setting()

# OpenAI API 키 설정
OPENAI_API_KEY = settings.OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

# FAISS 인덱스를 전역 변수로 설정
index = None
dim = 1536  # OpenAI의 ada-002 모델은 1536 차원 임베딩을 생성
texts = []  # 텍스트 데이터를 저장할 리스트


# OpenAI 임베딩을 이용한 벡터화 함수
def vectorize_text(text: str) -> np.ndarray:
    try:
        # OpenAI Embedding API를 사용하여 텍스트를 벡터로 변환
        response = openai.embeddings.create(
                model="text-embedding-ada-002",
                input=text,
                encoding_format="float"
            )
        # 벡터 추출
        embedding = response.model_dump()['data'][0]['embedding']
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while embedding text: {str(e)}")


# CSV 파일을 읽어 벡터화하고 FAISS 인덱스를 생성하는 함수
def read_csv_and_create_vectors(file: UploadFile) -> np.ndarray:
    try:
        # CSV 파일을 읽기
        contents = file.file.read().decode("utf-8")
        csv_reader = csv.reader(StringIO(contents))

        # CSV에서 데이터를 읽고 각 텍스트를 벡터화
        vectors = []
        for row in csv_reader:
            if row:  # 빈 줄을 제외하고
                text = row[4]  # 첫 번째 컬럼의 텍스트를 사용 (필요에 맞게 수정 가능)
                vector = vectorize_text(text)
                vectors.append(vector)
                texts.append(text)  # 텍스트도 함께 저장

        # 벡터들을 NumPy 배열로 변환
        return np.vstack(vectors)  # 2D 배열로 변환하여 반환
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error while reading CSV: {str(e)}")


# 컬렉션 및 인덱스 생성
@router.post("/create_index_from_csv")
async def create_index_from_csv(file: UploadFile = File(...)):
    global index
    try:
        # CSV 파일을 읽고 벡터화
        vectors = read_csv_and_create_vectors(file)

        # FAISS 인덱스 생성 (예: L2 거리 기반)
        if index is None:
            index = faiss.IndexFlatL2(vectors.shape[1])  # L2 거리 기반 인덱스 생성

        # 벡터를 인덱스에 추가
        index.add(vectors)

        return {"message": "Index created successfully and vectors added."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/search_vectors")
async def search_vectors(question: str, k: int = 5):
    try:
        # 질문을 벡터화
        query_vector = vectorize_text(question)  # OpenAI 임베딩 모델을 사용하여 벡터화

        # 벡터 검색
        distances, indices = index.search(query_vector.reshape(1, -1), k)  # 상위 k개의 벡터 검색

        # 검색된 인덱스에 해당하는 텍스트 추출
        search_results = [{"id": int(idx), "text": texts[int(idx)], "distance": float(dist)} 
                          for idx, dist in zip(indices[0], distances[0])]

        return {"results": search_results}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))