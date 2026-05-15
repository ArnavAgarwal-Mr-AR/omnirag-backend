import modal
import sys
import asyncio

# Create an ephemeral Modal script to test DocumentProcessor synchronously
app = modal.App("test-app")

@app.local_entrypoint()
def run():
    processor_cls = modal.Cls.from_name("omnirag-backend", "DocumentProcessor")
    processor = processor_cls()
    
    # We don't have a valid B2 file key right now, so let's just test Qdrant count
    print("Test finished.")
