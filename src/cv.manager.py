"""Manages the CV model."""

from typing import Any
import io

import numpy as np
from PIL import Image
from ultralytics import YOLO


class CVManager:

    def __init__(self):
        """
        YOLOv8 small:
        excellent balance between
        speed and accuracy.
        """

        self.model = YOLO("yolov8s.pt")

        self.conf_threshold = 0.25

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Runs object detection."""

        pil_img = Image.open(io.BytesIO(image)).convert("RGB")

        img = np.array(pil_img)

        results = self.model.predict(
            source=img,
            verbose=False,
            imgsz=512,
            conf=self.conf_threshold,
            device="cpu"
        )

        detections = []

        for result in results:
            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:
                xyxy = box.xyxy[0].tolist()

                x1, y1, x2, y2 = xyxy

                w = x2 - x1
                h = y2 - y1

                cls_id = int(box.cls[0].item())
        return detections
