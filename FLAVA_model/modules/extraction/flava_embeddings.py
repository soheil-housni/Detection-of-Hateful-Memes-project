from torch import nn
import torch
from transformers import FlavaModel
import torch
from torch.utils.data import DataLoader

"""
This class enables to extract the outputs of the pretrained FLAVA model that will then be used
in the custom head classifier.
More specifically, it extracts the mulimodal (texts and images) and pooler embeddings
of the pretrained FLAVA model
"""

class FLAVAExtractor():
    def __init__(self,
                 flava_model:FlavaModel,
                 device: torch.device):
        """
        Args:
        flava_model: pretrained FLAVA model used to extract the embeddings
        device: device used for the forward for the FLAVA model

        """
        super().__init__()
        self.flava_model=flava_model
        self.device=device
        self.flava_model.to(self.device)

    def get_embeddings(self,
                       FLAVA_dataloader:DataLoader,
                       save_path:str,
                       set_type:str) -> dict:
        """
        Args:
        FLAVA_dataloader: dataloader used for the forward of the pretrained FLAVA model
        save_path: path where to save the FLAVA embeddings
        set_type: type of the dataset being processed (train,validation,test)

        Return:
        Dictionnary containing all the outputs of the forward of the pretrained FLAVA model,
        including multimodal embeddings and pooler embeddings
        """
        all_embeddings_outputs=[]
        #list containing all the multimodal embeddings
        all_pooler_outputs=[]
        #list containing all the pooler embeddings
        all_labels=[]
        #list containing all the labels

        self.flava_model.eval()
        with torch.inference_mode():
            for batch in FLAVA_dataloader:
                batch["input_ids"]=batch["input_ids"].to(self.device)
                batch["pixel_values"]=batch["pixel_values"].to(self.device)
                batch["attention_mask"]=batch["attention_mask"].to(self.device)
                output=self.flava_model(input_ids=batch["input_ids"],pixel_values=batch["pixel_values"],attention_mask=batch["attention_mask"])
                all_pooler_outputs.append(output.multimodal_output.pooler_output)
                all_embeddings_outputs.append(output.multimodal_embeddings)
                all_labels.append(batch["labels"])

            all_embeddings_outputs=torch.cat(all_embeddings_outputs,dim=0)
            all_pooler_outputs=torch.cat(all_pooler_outputs,dim=0)
            all_labels=torch.cat(all_labels,dim=0)

            all_data={"multimodal_embeddings":all_embeddings_outputs,"pooler_embeddings":all_pooler_outputs,"labels":all_labels}
            #dictionnary containing all the FLAVA outputs

            torch.save(all_data, f"{save_path}/{set_type}_flava_embeddings.pt")
            print(f"Embeddings saved to {save_path}")

        return all_data
            





"""
class FLAVAExtractor():
    def __init__(self,
                 flava_model:FlavaModel,
                 device: torch.device):
        super().__init__()
        self.flava_model=flava_model
        self.device=device
        self.flava_model.to(self.device)
    def get_embeddings(self,train_FLAVA_dataloader,val_FLAVA_dataloader,save_path):
        all_train_embeddings_outputs=[]
        all_train_pooler_outputs=[]
        all_train_labels=[]

        all_val_embeddings_outputs=[]
        all_val_pooler_outputs=[]
        all_val_labels=[]

        self.flava_model.eval()
        with torch.inference_mode():
            for batch in train_FLAVA_dataloader:
                batch["input_ids"]=batch["input_ids"].to(self.device)
                batch["pixel_values"]=batch["pixel_values"].to(self.device)
                batch["attention_mask"]=batch["attention_mask"].to(self.device)
                output=self.flava_model(input_ids=batch["input_ids"],pixel_values=batch["pixel_values"],attention_mask=batch["attention_mask"])
                all_train_pooler_outputs.append(output.multimodal_output.pooler_output)
                all_train_embeddings_outputs.append(output.multimodal_embeddings)
                all_train_labels.append(batch["labels"])

            for batch in val_FLAVA_dataloader:
                batch["input_ids"]=batch["input_ids"].to(self.device)
                batch["pixel_values"]=batch["pixel_values"].to(self.device)
                batch["attention_mask"]=batch["attention_mask"].to(self.device)
                output=self.flava_model(input_ids=batch["input_ids"],pixel_values=batch["pixel_values"],attention_mask=batch["attention_mask"])
                all_val_pooler_outputs.append(output.multimodal_output.pooler_output)
                all_val_embeddings_outputs.append(output.multimodal_embeddings)
                all_val_labels.append(batch["labels"])

            all_train_embeddings_outputs=torch.cat(all_train_embeddings_outputs,dim=0)
            all_train_pooler_outputs=torch.cat(all_train_pooler_outputs,dim=0)
            all_train_labels=torch.cat(all_train_labels,dim=0)

            all_val_embeddings_outputs=torch.cat(all_val_embeddings_outputs,dim=0)
            all_val_pooler_outputs=torch.cat(all_val_pooler_outputs,dim=0)
            all_val_labels=torch.cat(all_val_labels,dim=0)

            all_train_data={"multimodal_embeddings":all_train_embeddings_outputs,"pooler_embeddings":all_train_pooler_outputs,"labels":all_train_labels}
            all_val_data={"multimodal_embeddings":all_val_embeddings_outputs,"pooler_embeddings":all_val_pooler_outputs,"labels":all_val_labels}

            torch.save(all_train_data, f"{save_path}/train_flava_embeddings.pt")
            torch.save(all_val_data, f"{save_path}/val_flava_embeddings.pt")
            print(f"Embeddings saved to {save_path}")

        return all_train_data,all_val_data
"""       