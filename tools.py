import os
import aioboto3
from botocore.exceptions import ClientError

s3_client = None

async def get_s3_client():
    global s3_client
    if s3_client is not None:
        return s3_client

    session = aioboto3.Session()
    s3_client = await session.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    ).__aenter__()
    return s3_client

async def download_file_from_s3(s3_url: str) -> bytes:
    if not s3_url.startswith("s3://"):
        raise ValueError("Invalid S3 URL format")

    _, path = s3_url.split("s3://", 1)
    bucket, key = path.split("/", 1)

    s3 = await get_s3_client()
    try:
        response = await s3.get_object(Bucket=bucket, Key=key)
        async with response['Body'] as stream:
            return await stream.read()
    except ClientError as e:
        raise RuntimeError(f"Failed to get object from S3: {e}")