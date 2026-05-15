import modal
from pydantic import BaseModel
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "sentence-transformers==3.3.0",
        "torch==2.5.0",
        "pydub",
        "pydantic"
    )
)

app = modal.App("omnirag-retrieval-processor", image=image)

class RerankRequestItem(BaseModel):
    id: str
    content: str
    metadata: dict

class RerankResultItem(BaseModel):
    id: str
    content: str
    score: float
    metadata: dict

@app.cls(gpu="T4")
class Reranker:
    @modal.enter()
    def load_model(self):
        from sentence_transformers import CrossEncoder
        print("Loading BGE-Reranker-v2-m3...")
        self.reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', device='cuda')
        print("Reranker loaded successfully.")

    @modal.method()
    def wakeup(self) -> bool:
        return True

    @modal.method()
    def rerank(self, query: str, documents: list, top_k: int = 5) -> list[RerankResultItem]:
        if not documents:
            return []

        # Extract content regardless of whether it's a Pydantic model or a dictionary
        pairs = []
        for doc in documents:
            content = doc.content if hasattr(doc, 'content') else doc.get('content', '')
            pairs.append([query, content])
        
        # predict returns a list of float scores
        scores = self.reranker.predict(pairs, batch_size=16)

        # Pair up scores with documents
        scored_docs = []
        for score, doc in zip(scores, documents):
            doc_id = doc.id if hasattr(doc, 'id') else doc.get('id', '')
            doc_content = doc.content if hasattr(doc, 'content') else doc.get('content', '')
            doc_metadata = doc.metadata if hasattr(doc, 'metadata') else doc.get('metadata', {})
            
            scored_docs.append(
                RerankResultItem(
                    id=doc_id,
                    content=doc_content,
                    metadata=doc_metadata,
                    score=float(score)
                )
            )

        # Sort descending by score
        scored_docs.sort(key=lambda x: x.score, reverse=True)

        # Apply top_k without a strict positive threshold since logits can be negative
        return scored_docs[:top_k]
