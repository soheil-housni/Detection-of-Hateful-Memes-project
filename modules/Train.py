import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import torch

class Train():
    def __init__(self,processor,model,loss_fn,optmizer,n_epochs, train_dataloader,eval_dataloader,scheduler):
        self.processor=processor
        self.model=model
        self.criterion=loss_fn
        self.optimizer=optmizer
        self.n_epochs=n_epochs
        self.train_dataloader=train_dataloader
        self.eval_dataloader=eval_dataloader
        self.scheduler=scheduler

    def freeze_model(self):
        for p in self.model.parameters():
            p.requires_grad=False

    def run_training(self):
        self.freeze_model(self)
        epoch_train_losses=[]
        epoch_train_f1=[]
        epoch_train_accuracies=[]

        epoch_val_losses=[]
        epoch_val_f1=[]
        epoch_val_accuracies=[]
        for i in range(self.n_epochs):
            self.model.train()
            batch_train_losses=[]
            batch_train_f1=[]
            batch_train_accuracies=[]

            for batch in self.train_dataloader:
                inputs=self.processor(list(batch["images"]),batch["texts"],return_tensors="pt", padding=True, max_length=77, truncation=True)
                logits=self.model(inputs).view(-1)
                targets=batch["labels"].view(-1)
                loss=self.criterion(logits,targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                batch_train_losses.append(loss)
                predictions=(logits>=0.5).long()
                f1=f1_score(np.array(targets),np.array(predictions))
                accuracy=f1_score(np.array(targets),np.array(predictions))
                batch_train_f1.append(f1)
                batch_train_accuracies.append(accuracy)

            self.model.eval()
            batch_val_losses=[]
            batch_val_f1=[]
            batch_val_accuracies=[]
            for batch in self.eval_dataloader:
                with torch.no_grad:
                    inputs=self.processor(list(batch["images"]),batch["texts"],return_tensors="pt", padding=True, max_length=77, truncation=True)
                    logits=self.model(inputs).view(-1)
                    targets=batch["labels"].view(-1)
                    loss=self.criterion(logits,targets)
                    batch_val_losses.append(loss)
                    predictions=(logits>=0.5).long()
                    f1=f1_score(np.array(targets),np.array(predictions))
                    accuracy=f1_score(np.array(targets),np.array(predictions))
                    batch_val_f1.append(f1)
                    batch_val_accuracies.append(accuracy)

                

            epoch_train_losses.append(np.mean(batch_train_losses))
            epoch_train_f1.append(np.mean(batch_train_f1))
            epoch_train_accuracies.append(np.mean(batch_train_accuracies))

            epoch_val_losses.append(np.mean(batch_val_losses))
            epoch_val_f1.append(np.mean(batch_val_f1))
            epoch_val_accuracies.append(np.mean(batch_val_accuracies))

                
