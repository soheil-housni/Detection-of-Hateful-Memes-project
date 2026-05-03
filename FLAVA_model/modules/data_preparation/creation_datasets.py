from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd

"""
Custom Dataset class respecting Pytorch standards (torch.utils.data.Dataset) 
that reaches to images.
This dataset is used for the forward in the pretrained FLAVA model.
"""
class CreationFLAVADataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df=df
        self.to_tensor=transforms.ToTensor(),

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx:int) -> dict:
        img_path = f"../data/{self.df['img'].iloc[idx]}"
        img=Image.open(img_path).convert("RGB")
        text=self.df["text"].iloc[idx]
        label=self.df["label"].iloc[idx]
        return {"images": img, "texts": text,"labels":label}

""""
Custom Dataset class respecting Pytorch standards (torch.utils.data.Dataset) 
that reaches to multimodal embeddings (images and texts) and pooler embeddings.
This dataset is used for the training of the custom model.
"""
class CreationProcessedDataset(Dataset):
    def __init__(self,data:dict):
        self.data=data
    def __len__(self)->int:
        return len(self.data["labels"])
    def __getitem__(self,idx:int)-> dict:
        return {'multimodal_embeddings':self.data["multimodal_embeddings"][idx],"pooler_embeddings":self.data["pooler_embeddings"][idx],\
                "images_embeddings":self.data["images_embeddings"][idx],"texts_embeddings":self.data["texts_embeddings"][idx],\
                    "sim_scores":self.data["sim_scores"][idx],"labels":self.data["labels"][idx]}