import os
import boto3
import uuid
import zipfile
import shutil
from datetime import datetime

s3_client = None

def get_s3_client():
    global s3_client
    if s3_client is not None:
        return s3_client

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    )
    return s3_client

def download_file_from_s3(s3_url: str) -> bytes:
    # s3://bucket/key 형식 처리
    if not s3_url.startswith("s3://"):
        raise ValueError("Invalid S3 URL format")

    _, path = s3_url.split("s3://", 1)
    bucket, key = path.split("/", 1)

    s3 = get_s3_client()
    response = s3.get_object(Bucket=bucket, Key=key)
    return response['Body'].read()
