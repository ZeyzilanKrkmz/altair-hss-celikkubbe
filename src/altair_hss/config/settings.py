from pathlib import Path
from pydantic import BaseModel,Field

class YoloSettings(BaseModel):
    model_path:Path=Field(default=Path("models/best.pt"))
    conf_threshold:float=Field(default=0.40)
    image_size:int=Field(default=640)
    device:str=Field(default="cpu")

class AppSettings(BaseModel):
    input_source:str=Field(default="data/samples/videos/demo.mp4")
    yolo:YoloSettings=Field(default_factory=YoloSettings)