import os
import boto3
from botocore.exceptions import ClientError
from fastapi import UploadFile
import uuid

class B2Storage:
    def __init__(self):
        self.endpoint_url = os.getenv("B2_ENDPOINT_URL")
        self.access_key = os.getenv("B2_ACCESS_KEY_ID")
        self.secret_key = os.getenv("B2_SECRET_ACCESS_KEY")
        self.bucket_name = os.getenv("B2_BUCKET_NAME")

        if not all([self.endpoint_url, self.access_key, self.secret_key, self.bucket_name]):
            print("Warning: B2 credentials not fully configured.")

        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )

    async def upload_file(self, file: UploadFile) -> str:
        """
        Uploads a file to Backblaze B2 and returns the file_key.
        """
        file_ext = file.filename.split('.')[-1] if '.' in file.filename else ''
        file_key = f"{uuid.uuid4()}.{file_ext}"

        try:
            # We use synchronous boto3 upload_fileobj, which we will wrap in a thread or 
            # run synchronously since upload_fileobj blocks. 
            # For purely async, aioboto3 could be used, but this works for our pipeline.
            self.s3_client.upload_fileobj(
                file.file,
                self.bucket_name,
                file_key,
                ExtraArgs={'ContentType': file.content_type}
            )
            return file_key
        except ClientError as e:
            print(f"Error uploading to B2: {e}")
            raise e

    def get_presigned_url(self, file_key: str, expiration=3600) -> str:
        """Generate a presigned URL to share the file."""
        try:
            response = self.s3_client.generate_presigned_url('get_object',
                                                             Params={'Bucket': self.bucket_name,
                                                                     'Key': file_key},
                                                             ExpiresIn=expiration)
            return response
        except ClientError as e:
            print(e)
            return None

b2_storage = B2Storage()
