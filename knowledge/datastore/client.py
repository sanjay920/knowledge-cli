from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, MilvusClient
import click

# Singleton pattern for Milvus connection
_client = None


def get_client(uri="http://localhost:19530", token="root:Milvus", db_name="default"):
    global _client
    if _client is None:
        _client = MilvusClient(uri=uri, token=token, db_name=db_name)
    return _client
