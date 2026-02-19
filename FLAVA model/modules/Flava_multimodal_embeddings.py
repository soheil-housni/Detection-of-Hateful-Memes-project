from torch import nn
import torch

class FlavaExtractor():
    def __init__(self,flava_model,device):
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
            