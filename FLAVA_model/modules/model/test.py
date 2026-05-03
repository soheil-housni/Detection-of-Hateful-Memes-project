import torch
from torch.utils.data import DataLoader
from loguru import logger
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

class Test():
    def __init__(self,
                 model,
                 device,
                 loss_fn):
        
        self.model=model
        self.device=device
        self.model=self.model.to(self.device)
        self.criterion=loss_fn

    def run_testing(self,
                    test_dataloader:DataLoader,
                    path:str,
                    multimodal:bool=False,
                    with_clip:bool=False):
        
        batch_losses=[]
        all_targets=[]
        all_predictions=[]
        all_logits=[]
        logger.info("Testing run:")
        self.model.eval()
        with torch.inference_mode():
            for batch in test_dataloader:
                if multimodal:
                    batch["multimodal_embeddings"]=batch["multimodal_embeddings"].float().to(self.device)
                else:
                    batch["pooler_embeddings"]=batch["pooler_embeddings"].float().to(self.device)

                if with_clip:
                    batch["texts_embeddings"]=batch["texts_embeddings"].float().to(self.device)
                    batch["images_embeddings"]=batch["images_embeddings"].float().to(self.device)
                    if not multimodal:
                        logits=self.model(pooler_embedding=batch['pooler_embeddings'],clip_text_embeddings=batch['texts_embeddings'],clip_image_embeddings=batch["images_embeddings"]).float()
                    else:
                        logits=self.model(multimodal_embedding=batch['multimodal_embeddings'],clip_text_embeddings=batch['texts_embeddings'],clip_image_embeddings=batch["images_embeddings"]).float()
                else:
                    if not multimodal:
                        logits=self.model(pooler_embedding=batch['pooler_embeddings']).float()
                    else:
                        logits=self.model(multimodal_embedding=batch['multimodal_embeddings']).float()
                
                targets=batch["labels"].long().to(self.device)
                all_targets.append(targets)
                loss=self.criterion(logits,targets)
                batch_losses.append(loss.item())

                predictions=torch.argmax(logits,dim=1).long()
                all_predictions.append(predictions)

                all_logits.append(logits)

            all_logits=torch.cat(all_logits)
            all_targets=torch.cat(all_targets)
            all_predictions=torch.cat(all_predictions)
            f1=f1_score(all_targets.cpu().numpy(),all_predictions.cpu().numpy(),average="weighted")
            accuracy=accuracy_score(all_targets.cpu().numpy(),all_predictions.cpu().numpy())
            final_loss=np.mean(batch_losses)
            log_probs=torch.nn.functional.softmax(all_logits,dim=1)

            log_probs=log_probs.numpy()
            all_targets=all_targets.numpy()
            all_predictions=all_predictions.numpy()

            test_performances={
                "test_f1":f1,
                "test_accuracy":accuracy,
                "test_loss":final_loss,
                "targets":all_targets,
                "predictions":all_predictions,
                "log_probs":log_probs
            }

            torch.save(test_performances,f"{path}/test_performances.pt")

            logger.info(f"Test Loss = {final_loss}")
            logger.info(f"Test F1 Score = {f1}")
            logger.info(f"Test Accuracy: = {accuracy}")

        return f1,accuracy,final_loss,all_targets,all_predictions,log_probs

