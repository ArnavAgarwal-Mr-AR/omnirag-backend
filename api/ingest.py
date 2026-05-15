from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from database.neon import get_db
from database.models import Document, Job
from storage.b2 import b2_storage
import uuid
import os

# We import the Modal app logic. 
# In a real deployed environment, we can also use modal.Function.lookup to call it.
try:
    from modal_app import DocumentProcessor
    from modal_image_app import ImageProcessor
    from modal_audio_app import AudioProcessor
    modal_available = True
except ImportError:
    modal_available = False

router = APIRouter()

@router.get("/api/v1/wakeup")
async def wakeup_modal():
    """Lightweight endpoint to ping all Modal containers to spin them up before use."""
    if modal_available:
        try:
            import modal
            processor_cls = modal.Cls.from_name("omnirag-backend", "DocumentProcessor")
            img_processor_cls = modal.Cls.from_name("omnirag-image-processor", "ImageProcessor")
            audio_processor_cls = modal.Cls.from_name("omnirag-audio-processor", "AudioProcessor")
            reranker_cls = modal.Cls.from_name("omnirag-retrieval-processor", "Reranker")
            
            import asyncio
            
            # Wait for all models to fully load into VRAM before returning
            await asyncio.gather(
                processor_cls().wakeup.remote.aio(),
                img_processor_cls().wakeup.remote.aio(),
                audio_processor_cls().wakeup.remote.aio(),
                reranker_cls().wakeup.remote.aio()
            )
            
            return {"status": "wakeup_signals_sent"}
        except Exception as e:
            return {"status": "error", "detail": str(e)}
    return {"status": "modal_unavailable"}

@router.post("/api/v1/ingest")
async def ingest_file(
    file: UploadFile = File(...),
    collection_id: str = "default",
    db: AsyncSession = Depends(get_db)
):
    """
    Ingest endpoint:
    1. Uploads to Backblaze B2
    2. Creates Document & Job entries in Neon Postgres
    3. Spawns a Modal job to process the file in the background
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # 1. Upload to B2
    try:
        b2_file_key = await b2_storage.upload_file(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"B2 Upload failed: {str(e)}")

    # Determine modality
    modality = "pdf"
    if file.content_type.startswith("image/"):
        modality = "image"
    elif file.content_type.startswith("audio/") or file.content_type.startswith("video/"):
        modality = "audio"

    # 2. Database Entries
    doc_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    new_doc = Document(
        id=doc_id,
        filename=file.filename,
        modality=modality,
        status="processing",
        collection_id=collection_id,
        b2_file_key=b2_file_key
    )
    
    new_job = Job(
        id=job_id,
        document_id=doc_id,
        status="queued",
        progress=0
    )

    db.add(new_doc)
    db.add(new_job)
    await db.commit()

    # 3. Spawn Modal Task (Background execution)
    # 3. Spawn Modal Task (Background execution)
    if not os.getenv("MODAL_TOKEN_ID") or not os.getenv("MODAL_TOKEN_SECRET"):
         print("Modal credentials missing. Processing will not start.")
         return {
            "job_id": job_id,
            "document_id": doc_id,
            "b2_file_key": b2_file_key,
            "status": "warning",
            "message": "File uploaded, but Modal credentials are missing in Vercel. Processing skipped."
        }

    if modal_available:
        try:
            import modal
            # Corrected: Use Cls.from_name for class methods
            if modality == "pdf":
                cls = modal.Cls.from_name("omnirag-backend", "DocumentProcessor")
                await cls().process_pdf.remote.aio(b2_file_key)
            elif modality == "image":
                cls = modal.Cls.from_name("omnirag-image-processor", "ImageProcessor")
                await cls().process_image.remote.aio(b2_file_key)
            elif modality == "audio":
                cls = modal.Cls.from_name("omnirag-audio-processor", "AudioProcessor")
                await cls().process_audio.remote.aio(b2_file_key)
        except Exception as e:
            error_msg = str(e)
            if "'NoneType' object has no attribute '__dict__'" in error_msg:
                error_msg = "Modal Client Initialization Error. Verify tokens."
            print(f"Failed to spawn modal job: {error_msg}")
            
    return {
        "job_id": job_id,
        "document_id": doc_id,
        "b2_file_key": b2_file_key,
        "status": "queued",
        "message": "File uploaded and processing started."
    }

@router.get("/api/v1/ingest/{job_id}/status")
async def get_job_status(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress,
        "error": job.error
    }
