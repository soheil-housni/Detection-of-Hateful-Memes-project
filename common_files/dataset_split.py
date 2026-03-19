from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple
from torch.utils.data import random_split
from torch.utils.data import Dataset
import torch


"""
This function split the train dataframe to obtain a new train dataframe and a test dataframe,
the validation dataframe being extracted from a second file.
"""


def split(train_df:pd.DataFrame)->Tuple[pd.DataFrame]:
    train_df,test_df=train_test_split(train_df,test_size=0.1,random_state=5,shuffle=True,stratify=train_df["label"])
    return train_df,test_df




"""
def split(train_dataset:Dataset):
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset=random_split(train_dataset,[0.9,0.1],generator=generator)
    return train_dataset,test_dataset
"""
