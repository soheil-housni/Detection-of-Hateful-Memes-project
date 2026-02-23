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

class CreationDataset(Dataset):
    def __init__(self, df):
        self.df=df
        self.transformation=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        img_path = f"../data/{self.df['img'].iloc[idx]}"
        img=Image.open(img_path).convert("RGB")
        img=self.transformation(img)
        text=self.df["text"].iloc[idx]
        label=self.df["label"].iloc[idx]
        return {"images": img, "texts": text,"labels":label}