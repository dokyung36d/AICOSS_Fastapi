from timm.models import create_model
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from key import SECRET_KEY, ACCESS_KEY, REGION, WEIGHT_BUCKET, WEIGHT_KEY
import boto3

model = create_model(
    'DaViT_base',
    pretrained=False,
    num_classes=60,
    drop_rate=0.1)

model = model.to("cpu")

s3_client = boto3.client(
    's3',
    region_name=REGION,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)


checkpoint_path = '2023_AICOSS_model_weight_EggTheProtein.pth'
s3_client.download_file(WEIGHT_BUCKET, WEIGHT_KEY, checkpoint_path)
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
