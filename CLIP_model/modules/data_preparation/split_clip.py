from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple
from torch.utils.data import random_split
from torch.utils.data import Dataset
import torch


def split_clip_embeddings(original_train_data:dict,
                          seed:int=42):
    
    copy_data=original_train_data.copy()
    for key in list(copy_data.keys()):
        copy_data[key]=copy_data[key].numpy()
    train_text_embeddings,test_text_embeddings,train_image_embeddings,test_image_embeddings,\
    train_sim_scores,test_sim_scores,train_labels,test_labels=train_test_split(copy_data["texts_embeddings"],
                                                                               copy_data["images_embeddings"],
                                                                               copy_data["sim_scores"],
                                                                               copy_data["labels"],
                                                                               test_size=0.1,
                                                                               stratify=original_train_data["labels"],
                                                                               random_state=seed)
    train_data={
        "texts_embeddings":torch.tensor(train_text_embeddings),
        "images_embeddings":torch.tensor(train_image_embeddings),
        "sim_scores":torch.tensor(train_sim_scores),
        "labels":torch.tensor(train_labels)
    }

    test_data={
        "texts_embeddings":torch.tensor(test_text_embeddings),
        "images_embeddings":torch.tensor(test_image_embeddings),
        "sim_scores":torch.tensor(test_sim_scores),
        "labels":torch.tensor(test_labels)
    }


    return train_data,test_data
    