import numpy as np
import torch
from loguru import logger
from torchvision import transforms
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from common_files import save_performances
from transformers import AutoTokenizer
from transformers import CLIPProcessor,CLIPImageProcessor,CLIPTokenizerFast,CLIPModel



class MemeDetectorAPI():
    def __init__(self,
                 meme_image:str,
                 meme_text:str,
                 tokenizer:AutoTokenizer,
                 model,
                 with_clip_image:bool=False,
                 with_clip_text:bool=False,
                 clip_processor=None,
                 clip_model=None):
        
        self.meme_image=meme_image
        self.meme_text=meme_text
        self.tokenizer=tokenizer
        self.model=model
        self.with_clip_image=with_clip_image
        self.with_clip_text=with_clip_text

        if clip_processor:
            self.clip_processor=clip_processor
        
        if clip_model:
            self.clip_model=clip_model

        if (with_clip_image or with_clip_text) and (clip_processor is None or clip_model is None):
            raise ValueError("To use CLIP embeddings, a CLIP processor and a CLIP model must be provided")

    
    def detection(self):
        meme_image_processed=self.image_preprocessing()
        meme_text_processed=self.text_preprocessing()
        if self.with_clip_image or self.with_clip_text:
            clip_input=self.clip_preprocessing()
            clip_text_embedding,clip_image_embedding=self.get_clip_embeddings(clip_input)
        
        self.model.eval()
        with torch.inference_mode():
            if self.with_clip_image and self.with_clip_text:
                logit=self.model(images=meme_image_processed,input_ids=meme_text_processed["input_ids"],attention_mask=meme_text_processed["attention_mask"],clip_text_embeddings=clip_text_embedding,clip_image_embeddings=clip_image_embedding)
            elif self.with_clip_image and not self.with_clip_text:
                logit=self.model(images=meme_image_processed,input_ids=meme_text_processed["input_ids"],attention_mask=meme_text_processed["attention_mask"],clip_image_embeddings=clip_image_embedding)
            elif not self.with_clip_image and self.with_clip_text:
                logit=self.model(images=meme_image_processed,input_ids=meme_text_processed["input_ids"],attention_mask=meme_text_processed["attention_mask"],clip_text_embeddings=clip_text_embedding)
            else:
                logit=self.model(images=meme_image_processed,input_ids=meme_text_processed["input_ids"],attention_mask=meme_text_processed["attention_mask"])
        
        prediction=torch.argmax(logit,dim=1).long().item()

        if prediction==1:
            print("Classification: The meme is hateful")
            return {"label":1,"meaning":"Classification: The meme is hateful"}
        else:
            print("Classification: The meme is lovely")
            return {"label":0,"meaning":"Classification: The meme is lovely"} 
        

    def image_preprocessing(self):

        transformation=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        meme_image_processed=transformation(self.meme_image)
        meme_image_processed=torch.unsqueeze(meme_image_processed,dim=0)

        return meme_image_processed
    
    def text_preprocessing(self):
        meme_text_processed=self.tokenizer(self.meme_text,return_tensors="pt",max_length=128,padding="max_length",truncation=True)
        return meme_text_processed


    def clip_preprocessing(self):
        clip_input=self.clip_processor(self.meme_image,self.meme_text,return_tensors="pt", padding=True, max_length=77, truncation=True)
        return clip_input
    
    def get_clip_embeddings(self,clip_input):
        clip_embeddings=self.clip_model(input_ids=clip_input["input_ids"],pixel_values=clip_input["pixel_values"],attention_mask=clip_input["attention_mask"])
        return clip_embeddings["text_embeds"],clip_embeddings["image_embeds"]



        