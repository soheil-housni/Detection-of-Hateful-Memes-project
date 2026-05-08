import torch
from PIL import Image
import matplotlib.pyplot as plt


class MemeDetector():
    def __init__(self,
                 model,
                 device,
                 flava_processor,
                 flava_model,
                 clip_processor=None,
                 clip_model=None):
        self.model=model
        self.device=device
        self.flava_model=flava_model
        self.clip_model=clip_model
        self.flava_processor=flava_processor
        self.clip_processor=clip_processor

        self.model=self.model.to(self.device)


    def detection(self,
                  img_path,
                  text,
                  with_clip:bool=False,
                  multimodal:bool=False):
        
        img=Image.open(img_path).convert("RGB")
        flava_input=self.flava_preprocessing(img,text)
        clip_input=self.clip_preprocessing(img,text)

        flava_output=self.flava_model(input_ids=flava_input["input_ids"],pixel_values=flava_input["pixel_values"],attention_mask=flava_input["attention_mask"])

        if multimodal:
            flava_multimodal_embeddings=flava_output["multimodal_embeddings"].float().to(self.device)
        else:
            flava_pooler_embeddings=flava_output.multimodal_output.pooler_output.float().to(self.device)
        
        if with_clip:
            clip_output=self.clip_model(input_ids=clip_input["input_ids"],pixel_values=clip_input["pixel_values"],attention_mask=clip_input["attention_mask"])
            clip_texts_embeddings=clip_output["text_embeds"].float().to(self.device)
            clip_images_embeddings=clip_output["image_embeds"].float().to(self.device)
            if not multimodal:
                logit=self.model(pooler_embedding=flava_pooler_embeddings,clip_text_embeddings=clip_texts_embeddings,clip_image_embeddings=clip_images_embeddings).float()
                #Forward of the model that returns the logits, using CLIP arguments
            else:
                logit=self.model(multimodal_embedding=flava_multimodal_embeddings,clip_text_embeddings=clip_texts_embeddings,clip_image_embeddings=clip_images_embeddings).float()
        else:
            if not multimodal:
                logit=self.model(pooler_embedding=flava_pooler_embeddings).float()
            else:
                logit=self.model(multimodal_embedding=flava_multimodal_embeddings).float()
        
        
        prediction=torch.argmax(logit,dim=1)


        self.printing_meme(img,text)

        if prediction.item()==1:
            print("Classification: The meme is hateful")
        else:
            print("Classification: The meme is lovely")
        
    
    def printing_meme(self,img,text):
        plt.imshow(img)
        plt.show()
        print("---------------------------")
        print("The text of the meme is :")
        print(f"'{text}'")
        print("---------------------------")


    
    def flava_preprocessing(self,img,text):
        flava_input=self.flava_processor(img,text,return_tensors="pt",padding="max_length",truncation=True, max_length=128)
        return flava_input

    def clip_preprocessing(self,img,text):
        clip_input=self.clip_processor(img,text,return_tensors="pt", padding=True, max_length=77, truncation=True)
        return clip_input