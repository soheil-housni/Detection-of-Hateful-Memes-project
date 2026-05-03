import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class MemeDetector():
    def __init__(self,
                 clip_processor,
                 clip_model,
                 model):
        
        self.clip_processor=clip_processor
        self.clip_model=clip_model
        self.model=model
    
    def detection(self,img_path,text,with_sim_score:bool=False):
        img=Image.open(img_path).convert("RGB")
        self.printing_meme(img,text)
        self.model.eval()
        input=self.clip_preprocessing(img_path,text)
        with torch.inference_mode():
            self.clip_model.eval()
            clip_embeddings=self.clip_model(input_ids=input["input_ids"],pixel_values=input["pixel_values"],attention_mask=input["attention_mask"])
            if with_sim_score:
                logit=self.model(texts_embeddings=clip_embeddings["text_embeds"],images_embeddings=clip_embeddings["image_embeds"],sim_scores=clip_embeddings["logits_per_image"])
            else:
                logit=self.model(texts_embeddings=clip_embeddings["text_embeds"],images_embeddings=clip_embeddings["image_embeds"])
        
        prediction=torch.argmax(logit,dim=1).long().item()

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

    def clip_preprocessing(self,img,text):
        inputs=self.clip_processor(img,text,return_tensors="pt", padding=True, max_length=77, truncation=True)
        return inputs
        