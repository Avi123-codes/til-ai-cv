from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
import io

from src.cv_server import app


client = TestClient(app)


def create_test_image():

    img = np.random.randint(
        0,
        255,
        (640, 640, 3),
        dtype=np.uint8
    )

    pil_img = Image.fromarray(img)

    buffer = io.BytesIO()

    pil_img.save(buffer, format="JPEG")

    return buffer.getvalue()


def test_healthcheck():

    response = client.get("/")

    assert response.status_code == 200


def test_cv_endpoint():

    image_bytes = create_test_image()

    response = client.post(
        "/cv",
        files={
            "file": ("test.jpg", image_bytes, "image/jpeg")
        }
    )

    assert response.status_code == 200

    assert isinstance(response.json(), list)
