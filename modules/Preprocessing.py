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

class MemeDataset(Dataset):
    def __init__(self, df):
        self.df=df
        self.to_tensor=transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        img_path = f"data/{self.df['img'].iloc[idx]}"
        with Image.open(img_path) as img:
            img=img.resize((224,224))
            img = img.convert("RGB")
            img = self.to_tensor(img)
        text=self.df["text"].iloc[idx]
        label=self.df["label"].iloc[idx]
        return {"images": img, "texts": text,"labels":label}


def preprocessing(dataloader,processor):
    all=[]
    for batch in dataloader:
        inputs=processor(list(batch["images"]),batch["texts"],return_tensors="pt", padding=True, max_length=77, truncation=True)
        all.append(inputs)
    return all