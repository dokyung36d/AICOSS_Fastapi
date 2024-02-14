from fastapi import FastAPI, File, UploadFile, HTTPException
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


if __name__ == "__main__":
    portNumber = 8081

    uvicorn.run("main:app", host="0.0.0.0", port= portNumber, reload=True)