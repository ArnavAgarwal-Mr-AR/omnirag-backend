import modal
from pydantic import BaseModel
import os

# Define the Modal environment for Audio processing
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")  # Required for audio preprocessing
    .pip_install(
        "faster-whisper==1.0.3",
        "pyannote.audio==3.3.2",
        "soundfile",   # Required for audio I/O in pyannote
        "librosa",     # Common audio dependency
        "sentence-transformers==3.3.0",
        "qdrant-client==1.12.0",
        "torch==2.5.0",
        "torchaudio==2.5.0",
        "boto3==1.35.39",
        "pydub"
    )
)

app = modal.App("omnirag-audio-processor", image=image)

secrets = [
    modal.Secret.from_name("qdrant-secrets", required_keys=["QDRANT_URL", "QDRANT_API_KEY"], environment_name="main"),
    modal.Secret.from_name("b2-secrets", required_keys=["B2_ENDPOINT_URL", "B2_ACCESS_KEY_ID", "B2_SECRET_ACCESS_KEY", "B2_BUCKET_NAME"], environment_name="main"),
]

class AudioChunkResult(BaseModel):
    id: str
    content: str
    timestamp_s: float
    timestamp_e: float
    speaker: str
    embedding: list[float]

@app.cls(gpu="T4", secrets=secrets)
class AudioProcessor:
    @modal.enter()
    def load_models(self):
        from faster_whisper import WhisperModel
        from pyannote.audio import Pipeline
        from sentence_transformers import SentenceTransformer

        print("Loading faster-whisper...")
        self.whisper_model = WhisperModel(
            'large-v3',
            device='cuda',
            compute_type='float16'
        )

        print("Loading Speaker Diarization...")
        # Note: In a real environment, HUGGINGFACE_TOKEN is required to download pyannote models
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        if hf_token:
            self.diarize_pipeline = Pipeline.from_pretrained(
                'pyannote/speaker-diarization-3.1',
                use_auth_token=hf_token
            ).to(torch.device('cuda'))
        else:
            print("Warning: HUGGINGFACE_TOKEN not set. Diarization will be mocked.")
            self.diarize_pipeline = None

        print("Loading BGE Embedder...")
        self.text_embedder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
        print("Audio models loaded.")

    @modal.method()
    def wakeup(self) -> bool:
        return True

    @modal.method()
    def process_audio(self, file_key: str) -> list[AudioChunkResult]:
        import boto3
        import uuid
        import subprocess
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct
        
        # 1. Download from B2
        bucket = os.environ["B2_BUCKET_NAME"]
        local_path = f"/tmp/{file_key}"
        wav_path = f"/tmp/{file_key}.wav"
        
        s3 = boto3.client(
            's3',
            endpoint_url=os.environ["B2_ENDPOINT_URL"],
            aws_access_key_id=os.environ["B2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["B2_SECRET_ACCESS_KEY"]
        )
        
        print(f"Downloading {file_key} from B2...")
        s3.download_file(bucket, file_key, local_path)

        # 2. Preprocessing via ffmpeg
        print("Converting to 16kHz WAV...")
        subprocess.run(["ffmpeg", "-i", local_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path, "-y"], check=True)
        
        # 3. Transcription (faster-whisper)
        print("Transcribing audio...")
        segments_generator, _ = self.whisper_model.transcribe(wav_path, word_timestamps=True)
        segments = list(segments_generator)

        # 4. Speaker Diarization
        print("Running Diarization...")
        diarization = None
        if self.diarize_pipeline:
           diarization = self.diarize_pipeline(wav_path)

        # 5. Build Chunks
        chunks = []
        points = []
        prefix = "Represent this sentence for searching relevant passages: "

        for i, seg in enumerate(segments):
            chunk_id = str(uuid.uuid4())
            text = seg.text.strip()
            
            speaker = "UNKNOWN"
            if diarization:
                max_intersection = 0
                for turn, _, spk in diarization.itertracks(yield_label=True):
                    intersection = min(turn.end, seg.end) - max(turn.start, seg.start)
                    if intersection > max_intersection:
                        max_intersection = intersection
                        speaker = spk
            
            emb = self.text_embedder.encode(prefix + text).tolist()

            chunks.append(AudioChunkResult(
                id=chunk_id,
                content=text,
                timestamp_s=seg.start,
                timestamp_e=seg.end,
                speaker=speaker,
                embedding=emb
            ))

            points.append(
                PointStruct(
                    id=chunk_id,
                    vector=emb,
                    payload={
                        "source_id": file_key,
                        "modality": "audio",
                        "content": text,
                        "timestamp_s": seg.start,
                        "timestamp_e": seg.end,
                        "speaker": speaker
                    }
                )
            )

        # 6. Upsert to Qdrant
        print("Upserting to Qdrant...")
        if os.environ.get("QDRANT_URL"):
            q_client = QdrantClient(
                url=os.environ["QDRANT_URL"],
                api_key=os.environ.get("QDRANT_API_KEY")
            )
            q_client.upsert(
                collection_name="omnirag_text",
                points=points
            )
            print("Successfully upserted audio chunks.")

        return chunks
