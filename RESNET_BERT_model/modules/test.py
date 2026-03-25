import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import torch
from loguru import logger
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .models_architectures import DistilbertResnetModel
from torch.nn.modules import loss
from typing import Tuple, List
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from common_files import save_performances


class Test():
    def __init__(self,
        model: DistilbertResnetModel,
        device :torch.device,
        loss_fn : loss,
        with_clip_text:bool=False,
        with_clip_images:bool=False,
        concat:bool=False
    ):
        self.model=model
        self.device=device
        self.with_clip_text=with_clip_text
        self.with_clip_images=with_clip_images
        self.concat=concat
        self.model.to(self.device)
        self.criterion=loss_fn

    def run_testing(self,
                    test_dataloader:DataLoader):
        
        logger.info("Testing run:")
        self.model.eval()
        with torch.inference_mode():
            all_predictions=[]
            all_targets=[]
            batch_losses=[]

            for batch in test_dataloader:
                batch["images"] = batch["images"].float().to(self.device)
                batch["input_ids"] = batch["input_ids"].long().to(self.device)
                batch["attention_mask"] = batch["attention_mask"].long().to(self.device)
                if self.with_clip_images:
                    batch["images_embeddings"] = batch["images_embeddings"].float().to(self.device)
                if self.with_clip_text:
                    batch["texts_embeddings"] = batch["texts_embeddings"].float().to(self.device)
                
                if self.with_clip_images and self.with_clip_text:
                    logits=self.model(images=batch["images"],input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],clip_text_embeddings=batch["texts_embeddings"],clip_image_embeddings=batch["images_embeddings"]).float()
                elif self.with_clip_images and not self.with_clip_text:
                    logits=self.model(images=batch["images"],input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],clip_image_embeddings=batch["images_embeddings"]).float()
                elif self.with_clip_text and not self.with_clip_images:
                    logits=self.model(images=batch["images"],input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],clip_text_embeddings=batch["texts_embeddings"]).float()
                else:
                    logits=self.model(images=batch["images"],input_ids=batch["input_ids"],attention_mask=batch["attention_mask"]).float()
                
                targets=batch["labels"].long().to(self.device)

                loss=self.criterion(logits,targets)
                batch_losses.append(loss.detach().item())

                predictions=torch.argmax(logits.detach(),dim=1).long()

                all_predictions.append(predictions)
                all_targets.append(targets)
            
            all_predictions=torch.cat(all_predictions)
            all_targets=torch.cat(all_targets)

            f1=f1_score(all_targets.cpu().numpy(),all_predictions.cpu().numpy(),average="weighted")
            accuracy=accuracy_score(all_targets.cpu().numpy(),all_predictions.cpu().numpy())
            final_loss=np.mean(batch_losses)

            logger.info(f"Test Loss = {final_loss}")
            logger.info(f"Test F1 Score = {f1}")
            logger.info(f"Test Accuracy: = {accuracy}")

            return f1,accuracy,final_loss,all_predictions.numpy(),all_targets.numpy()



