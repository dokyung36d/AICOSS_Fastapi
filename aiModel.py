from timm.models import create_model
import torch
import torch.nn as nn

model = create_model(
    'DaViT_base',
    pretrained=False,
    num_classes=60,
    drop_rate=0.1)

model = model.to("cpu")

checkpoint_path = '2023_AICOSS_model_weight_EggTheProtein.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

