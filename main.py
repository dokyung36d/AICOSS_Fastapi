from fastapi import FastAPI, File, UploadFile, HTTPException
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO

from aiModel import model

app = FastAPI()

@app.get("/AICOSS/image/prediction")
async def handleUploadedImage(file: UploadFile = File(...)):
    if file:
        image = Image.open(BytesIO(await file.read()))

        savePath = f"uploads/{file.filename.replace(' ', '_').lower()}.jpg"
        image.save(savePath, "JPEG")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform(image)
        

