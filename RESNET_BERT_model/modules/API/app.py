import uvicorn
from fastapi import FastAPI,UploadFile,File,Form
from .request_data import RequestData   
from pathlib import Path
import os
from ..inference import MemeDetectorAPI
from .config import config_app
from PIL import Image


app=FastAPI()

@app.get("/")
def welcome():
    return "Welcome to the Hateful Meme detector"


@app.post("/detection")
async def meme_detection(
    meme_text:str=Form(...),
    meme_image:UploadFile=File(...)
    ):
    meme_image=Image.open(meme_image.file)
    config_dict=config_app()
    tokenizer=config_dict["tokenizer"]
    model=config_dict["model"]
    with_clip_image=config_dict["with_clip_image"]
    with_clip_text=config_dict["with_clip_text"]

    meme_detector=MemeDetectorAPI(
        meme_image=meme_image,
        meme_text=meme_text,
        tokenizer=tokenizer,
        model=model,
        with_clip_image=with_clip_image,
        with_clip_text=with_clip_text
    )
    classification_dict=meme_detector.detection()

    return classification_dict

if __name__=="__main__":
    uvicorn.run(app="modules.API.app:app",host="0.0.0.0",port=8000,reload=False)



