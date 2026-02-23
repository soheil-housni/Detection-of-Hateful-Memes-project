import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import torch
from loguru import logger
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Train():
    def __init__(self,model,loss_fn,n_epochs,device,n_steps,n_warmup_steps,lr=0.01,weight_decay=1e-4,patience=2,min_improvement=1e-3,n_frozen_distilbert_layers=5,n_frozen_resnet_layers=3):
        self.model=model
        self.device=device
        self.criterion=loss_fn
        self.n_epochs=n_epochs
        self.model.to(self.device)
        self.patience=patience
        self.min_improvement=min_improvement
        self.n_frozen_distilbert_layers=n_frozen_distilbert_layers
        self.n_frozen_resnet_layers=n_frozen_resnet_layers
        self.freeze_layers(n_frozen_distilbert_layers,n_frozen_resnet_layers)

        head_params,backbone_params=self.get_params()
        self.optimizer=AdamW([{"params": head_params, "lr": lr},{"params": backbone_params, "lr": 1e-5}],weight_decay=weight_decay)
        #self.optimizer=AdamW(backbone_params+head_params,lr=lr,weight_decay=weight_decay)
        #self.scheduler=get_linear_schedule_with_warmup(self.optimizer,n_warmup_steps,n_steps)
        self.scheduler=ReduceLROnPlateau(self.optimizer,mode="max",factor=0.5,patience=1,threshold=1e-3)

    def get_params(self):
        head_params=[]
        backbone_params=[]
        for module in [self.model.projection_image,self.model.mha,self.model.fc_layers,self.model.fc_norm_layers,self.model.layer_norm_mha]:
            head_params+=[p for p in module.parameters() if p.requires_grad==True]
        for module in [self.model.distilbert_model,self.model.resnet_model]:
            backbone_params+=[p for p in module.parameters() if p.requires_grad==True]
        return head_params,backbone_params
    
    def freeze_layers(self,n_frozen_distilbert_layers,n_frozen_resnet_layers):
        for p in self.model.distilbert_model.embeddings.parameters():
            p.requires_grad=False

        for i in range(n_frozen_distilbert_layers):
            for p in self.model.distilbert_model.transformer.layer[i].parameters():
                p.requires_grad=False
        
        resnet_modules=["conv1","bn1","relu","maxpool","avgpool","fc"]
        for module in resnet_modules:
            for p in getattr(self.model.resnet_model,module).parameters():
                p.requires_grad=False

        resnet_layers=["layer1","layer2","layer3","layer4"]
        for i in range(n_frozen_resnet_layers):
            for p in getattr(self.model.resnet_model,resnet_layers[i]).parameters():
                p.requires_grad=False

    def save_performances(self,path,epoch_train_losses,epoch_train_f1,epoch_train_accuracies,epoch_val_losses,epoch_val_f1,epoch_val_accuracies):
        train_performances={
            "epoch_train_losses":torch.tensor(epoch_train_losses),
            "epoch_train_f1":torch.tensor(epoch_train_f1),
            "epoch_train_accuracies":torch.tensor(epoch_train_accuracies),

            "epoch_val_losses":torch.tensor(epoch_val_losses),
            "epoch_val_f1":torch.tensor(epoch_val_f1),
            "epoch_val_accuracies":torch.tensor(epoch_val_accuracies)
        }   
        torch.save(train_performances,f"{path}/epoch_performances.pt")

    def run_training(self,train_dataloader,val_dataloader,path):
        epoch_train_losses=[]
        epoch_train_f1=[]
        epoch_train_accuracies=[]

        epoch_val_losses=[]
        epoch_val_f1=[]
        epoch_val_accuracies=[]

        epoch_counter=0

        best_val_f1=0
        for epoch in range(self.n_epochs):
            self.model.train()
            batch_train_losses=[]

            all_train_targets=[]
            all_train_predictions=[]

            all_val_targets=[]
            all_val_predictions=[]

            logger.info(f"Epoch {epoch} :")

            for batch in train_dataloader:
                batch["images"] = batch["images"].float().to(self.device)
                batch["input_ids"] = batch["input_ids"].long().to(self.device)
                batch["attention_mask"] = batch["attention_mask"].long().to(self.device)
                logits=self.model(batch["images"],batch["input_ids"],batch["attention_mask"]).float()
                targets=batch["labels"].long().to(self.device)
                loss=self.criterion(logits,targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_train_losses.append(loss.detach().item())
                predictions=torch.argmax(logits.detach(),dim=1).long()

                all_train_predictions.append(predictions)
                all_train_targets.append(targets)
            
            #self.scheduler.step()

            self.model.eval()
            batch_val_losses=[]
            for batch in val_dataloader:
                with torch.no_grad():
                    batch["images"] = batch["images"].float().to(self.device)
                    batch["input_ids"] = batch["input_ids"].long().to(self.device)
                    batch["attention_mask"] = batch["attention_mask"].long().to(self.device)
                    logits=self.model(batch["images"],batch["input_ids"],batch["attention_mask"]).float()
                    targets=batch["labels"].long().to(self.device)
                    loss=self.criterion(logits,targets)
                    batch_val_losses.append(loss.detach().item())
                    predictions=torch.argmax(logits.detach(),dim=1).long()

                    all_val_targets.append(targets)
                    all_val_predictions.append(predictions)

                
            all_train_targets=torch.cat(all_train_targets)
            all_train_predictions=torch.cat(all_train_predictions)
            all_val_targets=torch.cat(all_val_targets)
            all_val_predictions=torch.cat(all_val_predictions)


            epoch_train_losses.append(np.mean(batch_train_losses))
            epoch_train_f1.append(f1_score(all_train_targets.cpu().numpy(),all_train_predictions.cpu().numpy(),average="weighted"))
            epoch_train_accuracies.append(accuracy_score(all_train_targets.cpu().numpy(),all_train_predictions.cpu().numpy()))

            val_f1_score=f1_score(all_val_targets.cpu().numpy(),all_val_predictions.cpu().numpy(),average="weighted")
            epoch_val_losses.append(np.mean(batch_val_losses))
            epoch_val_f1.append(val_f1_score)
            epoch_val_accuracies.append(accuracy_score(all_val_targets.cpu().numpy(),all_val_predictions.cpu().numpy()))

            self.scheduler.step(val_f1_score)

            logger.info(f"Epoch {epoch}: Train Loss = {epoch_train_losses[epoch]}")
            logger.info(f"Epoch {epoch}: Train Accuracy = {epoch_train_accuracies[epoch]}")
            logger.info(f"Epoch {epoch}: Train F1 = {epoch_train_f1[epoch]}")

            logger.info(f"Epoch {epoch}: Validation Loss = {epoch_val_losses[epoch]}")
            logger.info(f"Epoch {epoch}: Validation Accuracy = {epoch_val_accuracies[epoch]}")
            logger.info(f"Epoch {epoch}: Validation F1 = {epoch_val_f1[epoch]}")

            if val_f1_score<best_val_f1+self.min_improvement:
                epoch_counter+=1
            else:
                epoch_counter=0
                best_val_f1=val_f1_score
                torch.save(self.model.state_dict(),f"{path}/model_state.pt")                
            
            if epoch_counter>=self.patience:
                logger.info(f"Training stops after {epoch} epochs")
                self.save_performances(path,epoch_train_losses,epoch_train_f1,epoch_train_accuracies,epoch_val_losses,epoch_val_f1,epoch_val_accuracies)
                break
        
        self.save_performances(path,epoch_train_losses,epoch_train_f1,epoch_train_accuracies,epoch_val_losses,epoch_val_f1,epoch_val_accuracies)
            

                
