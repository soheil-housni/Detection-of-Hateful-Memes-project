import json
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import v2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def creation_dataframe(path):
    data=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    df=pd.DataFrame(data).drop(columns="id")
    return df


class CreationFlavaDataset(Dataset):
    def __init__(self, df):
        self.df=df
        self.to_tensor=transforms.ToTensor(),

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        img_path = f"../data/{self.df['img'].iloc[idx]}"
        img=Image.open(img_path).convert("RGB")
        text=self.df["text"].iloc[idx]
        label=self.df["label"].iloc[idx]
        return {"images": img, "texts": text,"labels":label}
    
class CreationProcessedDataset(Dataset):
    def __init__(self,data):
        self.data=data
    def __len__(self):
        return len(self.data["labels"])
    def __getitem__(self,idx):
        return {'multimodal_embeddings':self.data["multimodal_embeddings"][idx],"pooler_embeddings":self.data["pooler_embeddings"][idx],"labels":self.data["labels"][idx]}