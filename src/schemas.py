from pydantic import BaseModel
from typing import List


class Detection(BaseModel):
    bbox: List[float]
    category_id: int
