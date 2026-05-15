import os
import asyncio
import boto3
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from database.neon import get_db, engine
from sqlalchemy import text
from dotenv import load_dotenv, find_dotenv

async def reset_system():
    # Load env
    env_path = find_dotenv(".env.local") or find_dotenv(".env")
    load_dotenv(env_path)

    print("--- 1. Resetting PostgreSQL (Neon) ---")
    try:
        async with engine.begin() as conn:
            await conn.execute(text("TRUNCATE TABLE documents RESTART IDENTITY CASCADE;"))
            print("[SUCCESS] Postgres: 'documents' table truncated.")
    except Exception as e:
        print(f"[ERROR] Postgres error: {e}")

    print("\n--- 2. Resetting Qdrant Cloud ---")
    try:
        q_client = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ.get("QDRANT_API_KEY")
        )
        # Delete collections
        for coll in ["omnirag_text", "omnirag_image", "omnirag_audio"]:
            try:
                q_client.delete_collection(collection_name=coll)
                print(f"Deleted collection: {coll}")
            except Exception:
                pass
        
        # Recreate collections
        q_client.create_collection(
            collection_name="omnirag_text",
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        print("[SUCCESS] Qdrant: 'omnirag_text' recreated.")
        
        q_client.create_collection(
            collection_name="omnirag_image",
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        print("[SUCCESS] Qdrant: 'omnirag_image' recreated.")
    except Exception as e:
        print(f"[ERROR] Qdrant error: {e}")

    print("\n--- 3. Resetting Backblaze B2 ---")
    try:
        s3 = boto3.client(
            's3',
            endpoint_url=os.environ["B2_ENDPOINT_URL"],
            aws_access_key_id=os.environ["B2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["B2_SECRET_ACCESS_KEY"]
        )
        bucket = os.environ["B2_BUCKET_NAME"]
        
        objects = s3.list_objects_v2(Bucket=bucket)
        if 'Contents' in objects:
            for obj in objects['Contents']:
                s3.delete_object(Bucket=bucket, Key=obj['Key'])
                print(f"Deleted {obj['Key']} from B2.")
            print(f"[SUCCESS] B2: All files deleted from bucket '{bucket}'.")
        else:
            print("[SUCCESS] B2: Bucket is already empty.")
    except Exception as e:
        print(f"[ERROR] B2 error: {e}")

    print("\n[SUCCESS] FULL SYSTEM RESET COMPLETE!")

if __name__ == "__main__":
    asyncio.run(reset_system())
