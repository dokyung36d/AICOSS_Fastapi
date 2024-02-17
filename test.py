from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def testUploadFile():
    files = {"file": ("testImage.jpg", open("test_image.jpg", "rb"))}
    response = client.post("/AICOSS/image/prediction", files=files)

    assert response.status_code == 200
    assert response.json() != None