import torch
import torch.nn as nn

class ClipEmbeddings(nn.Module):
    def __init__(self, train_clip_dataloader,val_clip_dataloader,clip_model,device):
        super().__init__()
        self.clip_model=clip_model
        self.device=device
        self.clip_model.to(device)
        self.train_clip_dataloader=train_clip_dataloader
        self.val_clip_dataloader=val_clip_dataloader

    def forward(self):
        all_train_data={}
        all_val_data={}


        train_all_texts_embeddings=[]
        train_all_images_embeddings=[]
        train_all_labels=[]

        val_all_texts_embeddings=[]
        val_all_images_embeddings=[]
        val_all_labels=[]
        for p in self.clip_model.parameters():
            p.requires_grad=False
        self.clip_model.eval()
        with torch.no_grad():
            for batch in self.train_clip_dataloader:
                batch["input_ids"]=batch["input_ids"].to(self.device)
                batch["pixel_values"]=batch["pixel_values"].to(self.device)
                batch["attention_mask"]=batch["attention_mask"].to(self.device)
                train_outputs=self.clip_model(input_ids=batch["input_ids"],pixel_values=batch["pixel_values"],attention_mask=batch["attention_mask"])
                train_all_texts_embeddings.append(train_outputs["text_embeds"])
                train_all_images_embeddings.append(train_outputs["image_embeds"])
                train_all_labels.append(batch["labels"])
            for batch in self.val_clip_dataloader:
                batch["input_ids"]=batch["input_ids"].to(self.device)
                batch["pixel_values"]=batch["pixel_values"].to(self.device)
                batch["attention_mask"]=batch["attention_mask"].to(self.device)
                val_outputs=self.clip_model(input_ids=batch["input_ids"],pixel_values=batch["pixel_values"],attention_mask=batch["attention_mask"])
                val_all_texts_embeddings.append(val_outputs["text_embeds"])
                val_all_images_embeddings.append(val_outputs["image_embeds"])
                val_all_labels.append(batch["labels"])

        train_all_texts_embeddings=torch.cat(train_all_texts_embeddings,dim=0)
        train_all_images_embeddings=torch.cat(train_all_images_embeddings,dim=0)
        train_all_labels=torch.cat(train_all_labels)

        val_all_texts_embeddings=torch.cat(val_all_texts_embeddings,dim=0)
        val_all_images_embeddings=torch.cat(val_all_images_embeddings,dim=0)
        val_all_labels=torch.cat(val_all_labels)

        all_train_data={"texts_embeddings":train_all_texts_embeddings,"images_embeddings":train_all_images_embeddings,"labels":train_all_labels}
        all_val_data={"texts_embeddings":val_all_texts_embeddings,"images_embeddings":val_all_images_embeddings,"labels":val_all_labels}

        return all_train_data,all_val_data