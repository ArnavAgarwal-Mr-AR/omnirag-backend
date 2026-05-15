import os
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv, find_dotenv

env_path = find_dotenv(".env.local") or find_dotenv(".env")
load_dotenv(env_path)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

qdrant_client = None

if QDRANT_URL:
    qdrant_client = AsyncQdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

async def init_collections():
    """Initializes collections if they don't exist in Qdrant Cloud."""
    if not qdrant_client:
        print("Warning: Qdrant client not initialized.")
        return

    collections = [
        ("omnirag_text", 1024),
        ("omnirag_image", 768),
        ("omnirag_multimodal", 1024)
    ]
    
    for name, vector_size in collections:
        exists = await qdrant_client.collection_exists(name)
        if not exists:
            print(f"Creating collection {name} with size {vector_size}...")
            await qdrant_client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            try:
                await qdrant_client.create_payload_index(
                    collection_name=name,
                    field_name="source_id",
                    field_schema="keyword"
                )
            except Exception as e:
                print(f"Failed to create payload index for {name}: {e}")

async def upsert_chunks(collection_name: str, points: list):
    """
    Upsert points into Qdrant.
    `points` should be a list of qdrant_client.models.PointStruct
    """
    if not qdrant_client:
        raise Exception("Qdrant client not initialized")
        
    await qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )
