from pymilvus import MilvusClient

client = MilvusClient("./milvus_demo.db")


# Milvus 클라이언트 가져오는 함수 (의존성 주입용)
def get_milvus():
    yield client
