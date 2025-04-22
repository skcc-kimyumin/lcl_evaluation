import csv
import io
from typing import List

import numpy as np
import openai
from core.config import get_setting
from fastapi import APIRouter, File, HTTPException, UploadFile
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    utility,
)
from sklearn.feature_extraction.text import TfidfVectorizer

settings = get_setting()

# OpenAI API 키 설정
OPENAI_API_KEY = settings.OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

def search_vectors_info(milvus: MilvusClient, query: str, collection_name: str, k: int = 5):
    try:
        # 쿼리 벡터화 (OpenAI API 사용)
        query = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=query,
            encoding_format="float"
        )

        query_vector = np.array(query.model_dump()['data'][0]['embedding'], dtype=np.float32).reshape(1, -1)

        search_params = {
            "metric_type": "L2",
            "params": {
                # "M": 64,  # Maximum number of neighbors each node can connect to in the graph
                # "efConstruction": 100 
            }
        }

        res = milvus.search(
            collection_name=collection_name,
            data=query_vector.tolist(),
            limit=3,
            search_params=search_params,
            output_fields=["text", "vector"],
        )

        return res

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))