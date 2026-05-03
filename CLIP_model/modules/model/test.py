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
        self.criterion=loss_fn
        
        self.model=self.model.to(self.device)
        
    
    def run_testing(self,
                    test_dataloader:DataLoader,
                    path:str,
                    with_scores:bool=False
                    ):
        
        logger.info("Testing run:")
        self.model.eval()
        all_predictions=[]
        all_targets=[]
        all_logits=[]
        batch_losses=[]
        with torch.inference_mode():
            for batch in test_dataloader:
                batch["texts_embeddings"]=batch["texts_embeddings"].float().to(self.device)
                batch["images_embeddings"]=batch["images_embeddings"].float().to(self.device)
                if with_scores:
                    batch["sim_scores"]=batch["sim_scores"].float().to(self.device)
                    logits=self.model(batch["texts_embeddings"],batch["images_embeddings"],batch["sim_scores"]).float()
                else:
                    logits=self.model(batch["texts_embeddings"],batch["images_embeddings"]).float()

                targets=batch["labels"].long().to(self.device)
                all_targets.append(targets)

                loss=self.criterion(logits,targets)
                batch_losses.append(loss.item())

                predictions=torch.argmax(logits,dim=1).long()
                all_predictions.append(predictions)
                all_logits.append(logits)

        all_predictions=torch.cat(all_predictions)
        all_targets=torch.cat(all_targets)
        all_logits=torch.cat(all_logits)
        log_probs=torch.nn.functional.softmax(all_logits,dim=1)

        f1=f1_score(all_targets.cpu().numpy(),all_predictions.cpu().numpy(),average="weighted")
        accuracy=accuracy_score(all_targets.cpu().numpy(),all_predictions.cpu().numpy())
        final_loss=np.mean(batch_losses)

        logger.info(f"Test Loss = {final_loss}")
        logger.info(f"Test F1 Score = {f1}")
        logger.info(f"Test Accuracy: = {accuracy}")

        all_predictions=all_predictions.numpy()
        all_targets=all_targets.numpy()
        log_probs=log_probs.numpy()

        performances={
            "test_f1":f1,
            "test_accuracy":accuracy,
            "test_loss":final_loss,
            "predictions":all_predictions,
            "targets":all_targets,
            "log_probs":log_probs
            
        }

        torch.save(performances,f"{path}/test_performances.pt")

        return f1,accuracy,final_loss,all_predictions,all_targets,log_probs

