from fastapi import FastAPI, File, UploadFile
from typing import List

from schemas import Detection
from cv_manager import CVManager


app = FastAPI(title="TIL-26 CV Server")

manager = CVManager()


@app.get("/")
def healthcheck():
    return {"status": "ok"}


@app.post("/cv", response_model=List[Detection])
async def cv_endpoint(file: UploadFile = File(...)):

    image_bytes = await file.read()

    detections = manager.cv(image_bytes)

    return detections
