import json
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import v2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

""""
Custom Dataset class respecting Pytorch standards (torch.utils.data.Dataset) 
that reaches to images.
This dataset is used for the forward in the pretrained CLIP model.
"""
class CreationClipDataset(Dataset):
    def __init__(self, df:pd.DataFrame):
        self.df=df
        self.to_tensor=transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx:int) -> dict:
        img_path = f"../data/{self.df['img'].iloc[idx]}"
        img=Image.open(img_path) #as img:
            #img=img.resize((224,224))
            #img = img.convert("RGB")
            #img = self.to_tensor(img)
        text=self.df["text"].iloc[idx]
        label=self.df["label"].iloc[idx]
        return {"images": img, "texts": text,"labels":label}


""""
Custom Dataset class respecting Pytorch standards (torch.utils.data.Dataset) 
that reaches to images embeddings, texts embeddings, and similarity scores, outputed
from the CLIP model.
This dataset is used for the training of the custom model.
"""
class CreationProcessedDataset(Dataset):
    def __init__(self,data: dict):
        self.data=data
    def __len__(self) -> int:
        return len(self.data["labels"])
    def __getitem__(self, idx:int)->dict:
        return {"images_embeddings":self.data["images_embeddings"][idx],"texts_embeddings":self.data["texts_embeddings"][idx],"sim_scores":self.data["sim_scores"][idx],"labels":self.data["labels"][idx]}