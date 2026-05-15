import modal
from pydantic import BaseModel
import os

# --- Modal App Setup ---
# This defines the cloud environment for our ML tasks.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers==3.3.0",
        "pymupdf==1.24.0",
        "qdrant-client==1.12.0",
        "torch==2.5.0",
        "boto3==1.35.39",
        "pydantic"
    )
)

app = modal.App("omnirag-backend", image=image)

# Secrets should be configured in your Modal dashboard and loaded here
secrets = [
    modal.Secret.from_name("qdrant-secrets", required_keys=["QDRANT_URL", "QDRANT_API_KEY"], environment_name="main"),
    modal.Secret.from_name("b2-secrets", required_keys=["B2_ENDPOINT_URL", "B2_ACCESS_KEY_ID", "B2_SECRET_ACCESS_KEY", "B2_BUCKET_NAME"], environment_name="main")
]

class ChunkResult(BaseModel):
    id: str
    content: str
    page: int
    embedding: list[float]

@app.cls(gpu="T4", secrets=secrets)
class DocumentProcessor:
    @modal.enter()
    def load_models(self):
        from sentence_transformers import SentenceTransformer
        # Load the embedding model into GPU VRAM once when the container starts
        print("Loading BGE model...")
        self.embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
        print("Model loaded.")

    @modal.method()
    def wakeup(self) -> bool:
        return True

    @modal.method()
    def process_pdf(self, file_key: str) -> list[ChunkResult]:
        """
        1. Downloads PDF from B2
        2. Extracts text
        3. Embeds chunks
        4. Upserts to Qdrant Cloud
        """
        import boto3
        import fitz  # PyMuPDF
        import uuid
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct
        
        # 1. Download from B2
        s3 = boto3.client(
            's3',
            endpoint_url=os.environ["B2_ENDPOINT_URL"],
            aws_access_key_id=os.environ["B2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["B2_SECRET_ACCESS_KEY"]
        )
        
        local_path = f"/tmp/{file_key}"
        bucket = os.environ["B2_BUCKET_NAME"]
        
        print(f"Downloading {file_key} from B2...")
        s3.download_file(bucket, file_key, local_path)
        
        # 2. Extract Text using PyMuPDF
        print("Extracting text from PDF...")
        doc = fitz.open(local_path)
        extracted_pages = []
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                extracted_pages.append((i + 1, text.strip()))
        
        # 3. Simple Chunking & Embedding
        prefix = "Represent this sentence for searching relevant passages: "
        
        results = []
        points = []
        
        print(f"Embedding {len(extracted_pages)} pages...")
        for page_num, text in extracted_pages:
            # Note: For production, we would use LangChain's RecursiveCharacterTextSplitter here.
            # For simplicity, we embed page-by-page.
            emb = self.embed_model.encode(prefix + text).tolist()
            chunk_id = str(uuid.uuid4())
            
            results.append(
                ChunkResult(
                    id=chunk_id,
                    content=text,
                    page=page_num,
                    embedding=emb
                )
            )
            points.append(
                PointStruct(
                    id=chunk_id,
                    vector=emb,
                    payload={
                        "source_id": file_key,
                        "modality": "pdf",
                        "page": page_num,
                        "content": text
                    }
                )
            )
            
        print(f"Processed {len(results)} chunks. Uploading to Qdrant...")
        
        # 4. Upsert to Qdrant Cloud
        if os.environ.get("QDRANT_URL"):
            q_client = QdrantClient(
                url=os.environ["QDRANT_URL"],
                api_key=os.environ.get("QDRANT_API_KEY")
            )
            q_client.upsert(
                collection_name="omnirag_text",
                points=points
            )
            print("Successfully upserted to Qdrant.")
            
        return results

    @modal.method()
    def embed_query(self, query: str) -> list[float]:
        """Embeds a search query using the BGE model."""
        prefix = "Represent this sentence for searching relevant passages: "
        return self.embed_model.encode(prefix + query).tolist()

# To test this function directly on Modal:
@app.local_entrypoint()
def main():
    processor = DocumentProcessor()
    # Mock file key that would exist in your B2 bucket
    results = processor.process_pdf.remote("test_document.pdf")
    print(f"Returned {len(results)} chunks. First chunk ID: {results[0].id}")
    
    q_emb = processor.embed_query.remote("Test search query")
    print(f"Query embedding size: {len(q_emb)}")
