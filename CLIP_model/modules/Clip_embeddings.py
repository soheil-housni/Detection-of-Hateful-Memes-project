import torch
import torch.nn as nn
from transformers import CLIPModel
from torch.utils.data import DataLoader

"""
This class enables to extract the outputs of the pretrained CLIP model that will then be used
in the custom head classifier.
More specifically, it extracts the texts embeddings, the images embeddings, and the similarity scores
of the CLIP model
"""
class CLIPExtractor():
    def __init__(self,
                 clip_model:CLIPModel,
                 device:torch.device):
        super().__init__()
        self.clip_model=clip_model
        self.device=device
        self.clip_model.to(device)

    def get_embeddings(self,
                       dataloader:DataLoader,
                       set_type:str,
                       path:str) -> dict:

        """
        Args:
        FLAVA_dataloader: dataloader used for the forward of the pretrained CLIP model
        set_type: type of the dataset being processed (train,validation,test)
        path: path where to save the CLIP embeddings
        

        Return:
        Dictionnary containing all the outputs of the forward of the pretrained FLAVA model,
        including texts embeddings, images embeddings, and similarity scores
        """
        
        all_data={}
        #dictionnary containing the images embeddings, text embeddings, similarity scores,
        #and labels

        all_texts_embeddings=[]
        #list of all the texts embeddings
        all_images_embeddings=[]
        #list of all the images embeddings
        all_sim_scores=[]
        #list of all similarity scores
        all_labels=[]
        #list of all labels

        self.clip_model.eval()
        with torch.inference_mode():
            for batch in dataloader:
                for key in batch.keys():
                    batch[key]=batch[key].to(self.device)
                outputs=self.clip_model(input_ids=batch["input_ids"],pixel_values=batch["pixel_values"],attention_mask=batch["attention_mask"])
                all_texts_embeddings.append(outputs["text_embeds"])
                all_images_embeddings.append(outputs["image_embeds"])
                for i in range(len(outputs.logits_per_image)):
                    all_sim_scores.append(outputs.logits_per_image[i,i].item())
                    #here, we extract only the similarity score of the image with its 
                    #true corresponding text (diagonal value of the matrix)
                all_labels.append(batch["labels"])

        all_texts_embeddings=torch.cat(all_texts_embeddings,dim=0)
        all_images_embeddings=torch.cat(all_images_embeddings,dim=0)
        all_sim_scores=torch.tensor(all_sim_scores)
        all_labels=torch.cat(all_labels)

        all_data={"texts_embeddings":all_texts_embeddings,"images_embeddings":all_images_embeddings,"sim_scores":all_sim_scores,"labels":all_labels}
        all_data={"texts_embeddings":all_texts_embeddings,"images_embeddings":all_images_embeddings,"sim_scores":all_sim_scores,"labels":all_labels}

        torch.save(all_data,f"{path}/{set_type}_clip_embeddings.pt")
        print(f"{set_type} CLIP embeddings saved in {path}")

        return all_data

    """
    def get_embeddings(self,
                       train_clip_dataloader,
                       val_clip_dataloader,
                       path):
        all_train_data={}
        all_val_data={}

        train_all_texts_embeddings=[]
        train_all_images_embeddings=[]
        train_all_sim_scores=[]
        train_all_labels=[]

        val_all_texts_embeddings=[]
        val_all_images_embeddings=[]
        val_all_sim_scores=[]
        val_all_labels=[]

        self.clip_model.eval()
        with torch.inference_mode():
            for batch in train_clip_dataloader:
                for key in batch.keys():
                    batch[key]=batch[key].to(self.device)
                train_outputs=self.clip_model(input_ids=batch["input_ids"],pixel_values=batch["pixel_values"],attention_mask=batch["attention_mask"])
                train_all_texts_embeddings.append(train_outputs["text_embeds"])
                train_all_images_embeddings.append(train_outputs["image_embeds"])
                for i in range(len(train_outputs.logits_per_image)):
                    train_all_sim_scores.append(train_outputs.logits_per_image[i,i].item())
                train_all_labels.append(batch["labels"])

            for batch in val_clip_dataloader:
                for key in batch.keys():
                    batch[key]=batch[key].to(self.device)
                val_outputs=self.clip_model(input_ids=batch["input_ids"],pixel_values=batch["pixel_values"],attention_mask=batch["attention_mask"])
                val_all_texts_embeddings.append(val_outputs["text_embeds"])
                val_all_images_embeddings.append(val_outputs["image_embeds"])
                for i in range(len(val_outputs.logits_per_image)):
                    val_all_sim_scores.append(val_outputs.logits_per_image[i,i].item())
                val_all_labels.append(batch["labels"])

        train_all_texts_embeddings=torch.cat(train_all_texts_embeddings,dim=0)
        train_all_images_embeddings=torch.cat(train_all_images_embeddings,dim=0)
        train_all_sim_scores=torch.tensor(train_all_sim_scores)
        train_all_labels=torch.cat(train_all_labels)

        val_all_texts_embeddings=torch.cat(val_all_texts_embeddings,dim=0)
        val_all_images_embeddings=torch.cat(val_all_images_embeddings,dim=0)
        val_all_sim_scores=torch.tensor(val_all_sim_scores)
        val_all_labels=torch.cat(val_all_labels)

        all_train_data={"texts_embeddings":train_all_texts_embeddings,"images_embeddings":train_all_images_embeddings,"sim_scores":train_all_sim_scores,"labels":train_all_labels}
        all_val_data={"texts_embeddings":val_all_texts_embeddings,"images_embeddings":val_all_images_embeddings,"sim_scores":val_all_sim_scores,"labels":val_all_labels}

        torch.save(all_train_data,f"{path}/train_clip_embeddings.pt")
        torch.save(all_val_data,f"{path}/val_clip_embeddings.pt")
        print(f"Clip embeddings saved in {path}")

        return all_train_data,all_val_data
        """