from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple
from torch.utils.data import random_split
from torch.utils.data import Dataset
import torch


def split_flava_embeddings(original_train_all_embeddings:dict,
                          seed:int=42):
    
    copy_data=original_train_all_embeddings.copy()
    for key in list(copy_data.keys()):
        copy_data[key]=copy_data[key].numpy()
    train_multimodal_embeddings,test_multimodal_embeddings,train_pooler_embeddings,test_pooler_embeddings,\
        train_images_embeddings,test_images_embeddings,train_texts_embeddings,test_texts_embeddings,\
            train_sim_scores,test_sim_scores,train_labels,test_labels=train_test_split(copy_data["multimodal_embeddings"],
                                                                                       copy_data["pooler_embeddings"],
                                                                                       copy_data["images_embeddings"],
                                                                                       copy_data["texts_embeddings"],
                                                                                       copy_data["sim_scores"],
                                                                                       copy_data["labels"],
                                                                                       test_size=0.1,
                                                                                       stratify=original_train_all_embeddings["labels"],
                                                                                       random_state=seed)
    train_data={
        "multimodal_embeddings":torch.tensor(train_multimodal_embeddings),
        "pooler_embeddings":torch.tensor(train_pooler_embeddings),
        "images_embeddings":torch.tensor(train_images_embeddings),
        "texts_embeddings":torch.tensor(train_texts_embeddings),
        "sim_scores":torch.tensor(train_sim_scores),
        "labels":torch.tensor(train_labels)
    }

    test_data={
        "multimodal_embeddings":torch.tensor(test_multimodal_embeddings),
        "pooler_embeddings":torch.tensor(test_pooler_embeddings),
        "images_embeddings":torch.tensor(test_images_embeddings),
        "texts_embeddings":torch.tensor(test_texts_embeddings),
        "sim_scores":torch.tensor(test_sim_scores),
        "labels":torch.tensor(test_labels)
    }


    return train_data,test_data
    