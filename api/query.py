from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import modal
import os
from retrieval.vector_db import qdrant_client
from llm.generation import generate_response_stream
try:
    from modal_app import DocumentProcessor
    from modal_image_app import ImageProcessor
    from modal_retrieval_app import Reranker
    modal_available = True
except Exception as e:
    print(f"Modal imports failed: {e}")
    modal_available = False

router = APIRouter()

from qdrant_client.models import Filter, FieldCondition, MatchAny

class QueryRequest(BaseModel):
    query: str
    collection_id: str = "default"
    top_k: int = 5
    selected_source_ids: list[str] | None = None

@router.post("/api/v1/query")
async def query_endpoint(req: QueryRequest):
    if not qdrant_client:
        raise HTTPException(status_code=500, detail="Qdrant client not initialized")
    if not modal_available:
        raise HTTPException(status_code=500, detail="Modal backend not available for embeddings")

    # If selected_source_ids is explicitly an empty list, user deselected all sources.
    if req.selected_source_ids is not None and len(req.selected_source_ids) == 0:
        return StreamingResponse(
            (f"data: {chunk}\n\n" for chunk in ["[DONE]"]), 
            media_type="text/event-stream"
        )

    # Build Qdrant Filter
    query_filter = None
    if req.selected_source_ids:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="source_id",
                    match=MatchAny(any=req.selected_source_ids)
                )
            ]
        )

    # 1. Embed query via Modal for both Text (BGE) and Image (CLIP)
    if not os.getenv("MODAL_TOKEN_ID") or not os.getenv("MODAL_TOKEN_SECRET"):
         raise HTTPException(status_code=500, detail="Modal credentials missing. Please set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in Vercel.")

    try:
        print(f"Embedding query: {req.query}")
        # Use Cls.from_name for modern Modal SDK compatibility
        cls_text = modal.Cls.from_name("omnirag-backend", "DocumentProcessor")
        cls_image = modal.Cls.from_name("omnirag-image-processor", "ImageProcessor")
        
        query_vector_text = await cls_text().embed_query.remote.aio(req.query)
        query_vector_image = await cls_image().embed_query_for_image.remote.aio(req.query)
        
        if query_vector_text is None or query_vector_image is None:
            raise Exception("Modal returned empty embedding. Check if the Modal app is deployed.")
            
        print("Embedding successful.")
    except Exception as e:
        error_msg = str(e)
        if "'NoneType' object has no attribute '__dict__'" in error_msg:
            error_msg = "Modal Client Initialization Error. Verify tokens."
        print(f"Embedding failed: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Failed to embed query: {error_msg}")

    # 2. Search Qdrant Collections (Broad Search)
    broad_results = []
    
    try:
        search_result_text = await qdrant_client.search(
            collection_name="omnirag_text",
            query_vector=query_vector_text,
            query_filter=query_filter,
            limit=20
        )
        for res in search_result_text:
            broad_results.append({
                "id": str(res.id),
                "content": res.payload.get("content", ""),
                "metadata": res.payload
            })
            
        search_result_image = await qdrant_client.search(
            collection_name="omnirag_image",
            query_vector=query_vector_image,
            query_filter=query_filter,
            limit=10
        )
        for res in search_result_image:
            broad_results.append({
                "id": str(res.id),
                "content": res.payload.get("content", "No caption available"),
                "metadata": res.payload
            })
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant search failed: {str(e)}")

    if not broad_results:
        # No context found
        context_str = "No relevant documents found."
    else:
        # 3. Rerank the Broad Results via Cross-Encoder
        try:
            reranker_cls = modal.Cls.from_name("omnirag-retrieval-processor", "Reranker")
            reranked_results = await reranker_cls().rerank.remote.aio(req.query, broad_results, top_k=req.top_k)
        except Exception as e:
            print(f"Reranking failed, falling back to raw Qdrant scores: {e}")
            reranked_results = broad_results[:req.top_k]

        # 4. Format Context for LLM
        chunks = []
        for doc in reranked_results:
            # Check if it's a dict (fallback) or RerankResultItem object (from modal)
            payload = doc.metadata if hasattr(doc, 'metadata') else doc.get("metadata", {})
            content = doc.content if hasattr(doc, 'content') else doc.get("content", "")
            
            source = payload.get("source_id", "unknown")
            modality = payload.get("modality", "unknown")
            
            if modality == "audio":
                speaker = payload.get("speaker", "Unknown")
                ts_start = payload.get("timestamp_s", 0.0)
                ts_end = payload.get("timestamp_e", 0.0)
                chunks.append(f"[AUDIO:{source} | {speaker} | {ts_start:.1f}s - {ts_end:.1f}s] {content}")
            elif modality == "image":
                url = payload.get("image_url", "")
                chunks.append(f"[IMAGE_SOURCE:{source}] Image URL: {url} | Image Description: {content}")
            else:
                page = payload.get("page", "?")
                chunks.append(f"[SOURCE:{source} Page:{page}] {content}")
                
        context_str = "\n\n".join(chunks)
        print("======== CONTEXT SENT TO LLM ========")
        print(context_str)
        print("=====================================")

    # 5. Stream LLM Response
    async def event_generator():
        try:
            async for chunk in generate_response_stream(req.query, context_str):
                # Yield SSE formatted data
                if chunk:
                    yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
