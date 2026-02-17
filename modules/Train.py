import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import torch
from loguru import logger

class Train():
    def __init__(self,processor,model,loss_fn,optmizer,n_epochs, train_dataloader,val_dataloader,scheduler,device,batch_size,patience,min_improvement):
        self.processor=processor
        self.model=model
        self.device=device
        self.criterion=loss_fn
        self.optimizer=optmizer
        self.n_epochs=n_epochs
        self.train_dataloader=train_dataloader
        self.val_dataloader=val_dataloader
        self.scheduler=scheduler
        self.batch_size=batch_size
        self.model.to(self.device)
        self.patience=patience
        self.min_improvement=min_improvement

    def run_training(self):
        epoch_train_losses=[]
        epoch_train_f1=[]
        epoch_train_accuracies=[]

        epoch_val_losses=[]
        epoch_val_f1=[]
        epoch_val_accuracies=[]

        epoch_counter=0
        for epoch in range(self.n_epochs):
            self.model.train()
            batch_train_losses=[]

            all_train_targets=[]
            all_train_predictions=[]

            all_val_targets=[]
            all_val_predictions=[]

            logger.info(f"Epoch {epoch} :")

            if not epoch_val_f1:
                previous_f1_score=0
            else:
                previous_f1_score=epoch_val_f1[-1]

            for batch in self.train_dataloader:
                batch["texts_embeddings"]=batch["texts_embeddings"].to(self.device)
                batch["images_embeddings"]=batch["images_embeddings"].to(self.device)
                logits=self.model(batch["texts_embeddings"],batch["images_embeddings"]).view(-1)
                targets=batch["labels"].to(self.device)
                loss=self.criterion(logits,targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                batch_train_losses.append(loss.detach().item())
                predictions=(logits.detach()>=0).long()

                all_train_predictions.append(predictions)
                all_train_targets.append(targets)
                

            self.model.eval()
            batch_val_losses=[]
            for batch in self.val_dataloader:
                with torch.no_grad():
                    batch["texts_embeddings"]=batch["texts_embeddings"].to(self.device)
                    batch["images_embeddings"]=batch["images_embeddings"].to(self.device)
                    logits=self.model(batch["texts_embeddings"],batch["images_embeddings"]).view(-1)
                    targets=batch["labels"].to(self.device)
                    loss=self.criterion(logits,targets)
                    batch_val_losses.append(loss.detach().item())
                    predictions=(logits.detach()>=0).long()

                    all_val_targets.append(targets)
                    all_val_predictions.append(predictions)

                
            all_train_targets=torch.cat(all_train_targets)
            all_train_predictions=torch.cat(all_train_predictions)
            all_val_targets=torch.cat(all_val_targets)
            all_val_predictions=torch.cat(all_val_predictions)


            epoch_train_losses.append(np.mean(batch_train_losses))
            epoch_train_f1.append(f1_score(all_train_targets.cpu().numpy(),all_train_predictions.cpu().numpy()))
            epoch_train_accuracies.append(accuracy_score(all_train_targets.cpu().numpy(),all_train_predictions.cpu().numpy()))

            val_f1_score=f1_score(all_val_targets.cpu().numpy(),all_val_predictions.cpu().numpy())
            epoch_val_losses.append(np.mean(batch_val_losses))
            epoch_val_f1.append(val_f1_score)
            epoch_val_accuracies.append(accuracy_score(all_val_targets.cpu().numpy(),all_val_predictions.cpu().numpy()))

            logger.info(f"Epoch {epoch}: Train Loss = {epoch_train_losses[epoch]}")
            logger.info(f"Epoch {epoch}: Train Accuracy = {epoch_train_accuracies[epoch]}")
            logger.info(f"Epoch {epoch}: Train F1 = {epoch_train_f1[epoch]}")

            logger.info(f"Epoch {epoch}: Validation Loss = {epoch_val_losses[epoch]}")
            logger.info(f"Epoch {epoch}: Validation Accuracy = {epoch_val_accuracies[epoch]}")
            logger.info(f"Epoch {epoch}: Validation F1 = {epoch_val_f1[epoch]}")

            if val_f1_score<previous_f1_score+self.min_improvement:
                epoch_counter+=1
            else:
                epoch_counter=0
            
            if epoch_counter>=self.patience:
                break
                logger.info(f"Training stops after {epoch} epochs")

                
