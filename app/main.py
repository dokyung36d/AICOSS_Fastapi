from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Request
import requests
from fastapi.responses import JSONResponse
import uvicorn
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
import os
import torch
import csv
import json
import redis

from aiModel import model

app = FastAPI()

imagePath = "./uploads"

@app.post("/AICOSS/image/prediction")
async def handleUploadedImage(file: UploadFile = File(...)):
    if file:
        image = Image.open(BytesIO(await file.read()))

        numImage = getNumberOfImages(imagePath) #integer

        savePath = imagePath + f"/{str(numImage)}.jpg"
        image.save(savePath, "JPEG")

        modelPrediction = getModelPrediction(image)
        labelList = getLabelList()
        
        jsonData = makeJsonObject(keyList = labelList, valueList = modelPrediction)
        print(jsonData)

        return JSONResponse(content=jsonData)
    
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

@app.middleware("http")
async def session_checker(request: Request, call_next):
    jsessionid = request.cookies.get("JSESSIONID")
    if not jsessionid:
        return JSONResponse(status_code=401, content={"error": "Unauthorized: No session"})

    redis_key = f"spring:session:sessions:{jsessionid}"
    session_data = r.hgetall(redis_key)

    if not session_data:
        return JSONResponse(status_code=401, content={"error": "Unauthorized: Invalid session"})

    # 세션이 유효하면 다음 요청 처리
    response = await call_next(request)
    return response
    
@app.post("/AICOSS/image/prediction/URL")
async def handleImageURL(image_url: str = Body(..., embed=True)):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    
    modelPrediction = getModelPrediction(image)
    labelList = getLabelList()
    
    jsonData = makeJsonObject(keyList = labelList, valueList = modelPrediction)


    return JSONResponse(content=jsonData)

        

def getNumberOfImages(directoryPath):
    files = os.listdir(directoryPath)

    files = [file for file in files if os.path.isfile(os.path.join(directoryPath, file))]

    return len(files)

def getLabelList() -> list: #returns list of labels
    csvFilePath = "sample_submission.csv"

    with open(csvFilePath ,"r", newline="") as csvFile:
        csvReader = csv.reader(csvFile)

        labelList = next(csvReader)[1:] #Remove First because of Index

    return labelList

def getModelPrediction(image) -> list:
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # image = Image.open(imagePath)
    image = transform(image).reshape(-1, 3, 224, 224)

    output = F.sigmoid(model(image))
    roundedOuput = torch.round(output).squeeze(dim=0).to(dtype=torch.int) #if probability is larger than 0.5, assumes present in image.
    modelPrediction = roundedOuput.tolist()
    modelPrediction = [str(element) for element in modelPrediction] #Change to Str type

    return modelPrediction

def makeJsonObject(keyList : list, valueList : list):
    data = dict(zip(keyList, valueList))

    jsonData = json.dumps(data)

    return jsonData


# if __name__ == "__main__":
#     portNumber = 8081

#     uvicorn.run("main:app", host="0.0.0.0", port= portNumber, reload=True)