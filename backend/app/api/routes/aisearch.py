from fastapi import APIRouter, File, HTTPException, UploadFile
import csv
import io
from service.vectordb.aisearch import embeddings, get_azure_search_client  # 기존에 정의한 임베딩과 클라이언트 함수

router = APIRouter()

@router.post("/create_or_update_index_from_csv")
async def create_or_update_index_from_csv(index_name: str, file: UploadFile = File(...)):
    try:
        # 1. CSV 파일 읽기
        file_content = await file.read()
        file_str = file_content.decode("utf-8")
        reader = csv.DictReader(io.StringIO(file_str))

        # 2. 텍스트 데이터 추출 (vector 필드 제외)
        texts = []
        for row in reader:
            text = " ".join([v for k, v in row.items() if k != "vector"])
            texts.append(text)

        # 3. AzureSearch 클라이언트 생성
        client = get_azure_search_client(index_name)

        # 4. 인덱스 생성 또는 업데이트
        client.create_or_update_index()

        # 5. 문서 추가 (임베딩 자동 생성 포함)
        # documents는 texts 리스트를 그대로 넘기면 내부에서 embed_query를 통해 임베딩 생성 후 업로드함
        client.add_documents(texts)

        return {"message": f"Index '{index_name}' created/updated and documents added successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
