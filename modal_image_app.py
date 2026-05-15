import modal
from pydantic import BaseModel
import os

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")  # Required for OpenCV/PaddleOCR
    .pip_install(
        "transformers==4.46.0",
        "timm",      # Required by Florence-2
        "einops",    # Required by Florence-2
        "paddleocr==2.9.0",
        "paddlepaddle-gpu==2.6.1",
        "opencv-python-headless",
        "albumentations",  # Required by paddleocr
        "shapely",         # Required by paddleocr
        "pyclipper",       # Required by paddleocr
        "qdrant-client==1.12.0",
        "torch==2.5.0",
        "nvidia-cudnn-cu12",
        "boto3==1.35.39",
        "Pillow",
        "sentence-transformers==3.3.0"  # For text embedding of captions
    )
)

app = modal.App("omnirag-image-processor", image=image)

secrets = [
    modal.Secret.from_name("qdrant-secrets", required_keys=["QDRANT_URL", "QDRANT_API_KEY"], environment_name="main"),
    modal.Secret.from_name("b2-secrets", required_keys=["B2_ENDPOINT_URL", "B2_ACCESS_KEY_ID", "B2_SECRET_ACCESS_KEY", "B2_BUCKET_NAME"], environment_name="main")
]

class ImageChunkResult(BaseModel):
    id: str
    caption: str
    image_url: str
    is_diagram: bool

@app.cls(gpu="T4", secrets=secrets)
class ImageProcessor:
    @modal.enter()
    def load_models(self):
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM, CLIPProcessor, CLIPModel
        from sentence_transformers import SentenceTransformer
        from paddleocr import PaddleOCR

        print("Loading Florence-2...")
        self.florence_proc = AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True)
        self.florence_model = AutoModelForCausalLM.from_pretrained(
            'microsoft/Florence-2-large',
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to('cuda')

        print("Loading CLIP...")
        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to('cuda')
        self.clip_proc = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')

        print("Loading Text Embedder...")
        self.text_embedder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")

        print("Loading PaddleOCR...")
        # use_angle_cls=True detects rotated text
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, show_log=False)
        print("Models loaded successfully.")

    @modal.method()
    def wakeup(self) -> bool:
        return True

    def caption_image(self, image_obj, task='<DETAILED_CAPTION>') -> str:
        import torch
        inputs = self.florence_proc(text=task, images=image_obj, return_tensors='pt').to('cuda', torch.float16)
        generated = self.florence_model.generate(**inputs, max_new_tokens=1024)
        return self.florence_proc.decode(generated[0], skip_special_tokens=True)

    @modal.method()
    def process_image(self, file_key: str) -> ImageChunkResult:
        import boto3
        from PIL import Image
        import uuid
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct
        
        # 1. Download image from B2
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
        
        img = Image.open(local_path).convert('RGB')
        
        # 2. Florence-2 Detection: Diagram vs Photo
        print("Captioning with Florence-2...")
        od_result = self.caption_image(img, '<OD>')  # Object detection
        
        technical_keywords = ['arrow','box','node','diagram','chart','flow','axis']
        is_diagram = any(k in od_result.lower() for k in technical_keywords)
        
        if is_diagram:
            caption = self.caption_image(img, '<MORE_DETAILED_CAPTION>')
        else:
            caption = self.caption_image(img, '<CAPTION>')
            
        print(f"Caption generated: {caption}")

        # 3. OCR (if text is detected or for all images)
        print("Running PaddleOCR...")
        ocr_result = self.ocr.ocr(local_path, cls=True)
        ocr_text = ""
        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                bbox, (text, confidence) = line
                if confidence > 0.6:
                    ocr_text += text + "\n"

        if ocr_text:
            caption += f"\n\nExtracted Text:\n{ocr_text}"

        # 4. Embeddings
        print("Embedding image (CLIP) and caption (BGE)...")
        # Image Embedding (CLIP)
        inputs = self.clip_proc(images=img, return_tensors='pt').to('cuda')
        feats = self.clip_model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        image_embedding = feats.squeeze().cpu().tolist()

        # Text Embedding (BGE)
        prefix = "Represent this sentence for searching relevant passages: "
        text_embedding = self.text_embedder.encode(prefix + caption).tolist()

        # 5. Upsert to Qdrant
        print("Upserting to Qdrant...")
        chunk_id = str(uuid.uuid4())
        image_url = f"{os.environ['B2_ENDPOINT_URL']}/{bucket}/{file_key}"

        if os.environ.get("QDRANT_URL"):
            q_client = QdrantClient(
                url=os.environ["QDRANT_URL"],
                api_key=os.environ.get("QDRANT_API_KEY")
            )
            # Upsert into text collection for text-based semantic search
            q_client.upsert(
                collection_name="omnirag_text",
                points=[
                    PointStruct(
                        id=chunk_id,
                        vector=text_embedding,
                        payload={
                            "source_id": file_key,
                            "modality": "image",
                            "image_url": image_url,
                            "content": caption,
                            "is_diagram": is_diagram
                        }
                    )
                ]
            )
            # Upsert into image collection for pure image-to-image or CLIP text-to-image search
            q_client.upsert(
                collection_name="omnirag_image",
                points=[
                    PointStruct(
                        id=chunk_id,
                        vector=image_embedding,
                        payload={
                            "source_id": file_key,
                            "modality": "image",
                            "image_url": image_url,
                            "content": caption
                        }
                    )
                ]
            )
            print("Successfully upserted.")

        return ImageChunkResult(
            id=chunk_id,
            caption=caption,
            image_url=image_url,
            is_diagram=is_diagram
        )

    @modal.method()
    def embed_query_for_image(self, query: str) -> list[float]:
        """Embeds text using CLIP for searching the image collection."""
        inputs = self.clip_proc(text=query, return_tensors='pt', padding=True).to('cuda')
        feats = self.clip_model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze().cpu().tolist()
