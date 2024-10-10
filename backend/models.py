from typing import Any, Optional, Union
from pydantic import BaseModel


class TrainApiData(BaseModel):
    model_name: str
    epochs: int
    config_file: Optional[str] = None


class PredictApiData(BaseModel):
    input_image: Any
    model_name: str


class DeleteApiData(BaseModel):
    model_name: str
    model_version: Optional[Union[list[int], int]]  # list | int in python 10
