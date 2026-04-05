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

        if multimodal:
            flava_input["multimodal_embeddings"]=flava_input["multimodal_embeddings"].float().to(self.device)
        else:
            flava_input["pooler_embeddings"]=flava_input["pooler_embeddings"].float().to(self.device)
        
        if with_clip:
            clip_input["texts_embeddings"]=clip_input["texts_embeddings"].float().to(self.device)
            clip_input["images_embeddings"]=clip_input["images_embeddings"].float().to(self.device)
            if not multimodal:
                logit=self.model(pooler_embedding=flava_input['pooler_embeddings'],clip_text_embedding=clip_input['texts_embeddings'],clip_image_embedding=clip_input["images_embeddings"]).float()
                #Forward of the model that returns the logits, using CLIP arguments
            else:
                logit=self.model(multimodal_embedding=flava_input['multimodal_embeddings'],clip_text_embedding=clip_input['texts_embeddings'],clip_image_embedding=clip_input["images_embeddings"]).float()
        else:
            if not multimodal:
                logit=self.model(pooler_embedding=flava_input['pooler_embeddings']).float()
            else:
                logit=self.model(multimodal_embedding=flava_input['multimodal_embeddings']).float()
        
        prediction=torch.argmax(logit,dim=1)

        self.printing_meme(img,text)

        if prediction==1:
            print("Classification: The meme is hateful")
        else:
            print("Classification: The meme is lovely")
        
    
    def printing_meme(self,img,text):
        print("The meme is :")
        plt.imshow(img)
        plt.show()
        print("---------------------------")
        print("The text of the meme is :")
        print(f"'{text}'")
        print("---------------------------")


    
    def flava_preprocessing(self,img,text):
        clip_inputs=self.flava_processor(img,text,return_tensors="pt",padding="max_length",truncation=True, max_length=128)
        return clip_inputs

    def clip_preprocessing(self,img,text):
        clip_inputs=self.clip_processor(img,text,return_tensors="pt", padding=True, max_length=77, truncation=True)
        return clip_inputs