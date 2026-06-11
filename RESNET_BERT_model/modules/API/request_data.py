from pydantic import BaseModel
from typing import List
from fastapi import UploadFile


class RequestData(BaseModel):
    meme_text:str
