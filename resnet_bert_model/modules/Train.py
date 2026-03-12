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

class Train():
    def __init__(self,
                 model : DistilbertResnetModel,
                 loss_fn : loss,
                 n_epochs: int,
                 device :torch.device,
                 n_steps :int,
                 n_warmup_steps: int,
                 lr : float =0.01,
                 weight_decay : float =1e-4,
                 patience : int=2,
                 min_improvement : float=1e-3,
                 n_frozen_distilbert_layers : int=5,
                 n_frozen_resnet_layers : int=3):
        """
        model: the DistilbertResnetModel we train
        loss_fn: the loss function we use for the training
        n_epochs: the number of epochs of the training
        device: the device we use for the training
        n_steps: the number of steps of the training
        n_warmup_steps: the number of steps for the warmup of the learning rate (during which the larning rate will increase)
        lr: the learning rate
        weight_decay: the weight decay for the L2 (ridge) regularization
        patience: the number of epochs to wait when the performance does not improve before stopping the training
        min_improvement: the minimum improvement the model need to accomplish to be considered as an actual improvement that reinitialize the patience counter
        n_frozen_distilbert_layers: the number of first encoder layers of the pretrained distilbert transformer we freeze during the training
        n_frozen_resnet_layers: the number of first resnet layers we freeze during the training
        """
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
        #self.optimizer=AdamW([{"params": head_params, "lr": lr},{"params": backbone_params, "lr": 1e-4}],weight_decay=weight_decay)
        self.optimizer=AdamW(backbone_params+head_params,lr=lr,weight_decay=weight_decay)
        self.scheduler=get_linear_schedule_with_warmup(self.optimizer,n_warmup_steps,n_steps)
        #self.scheduler=ReduceLROnPlateau(self.optimizer,mode="max",factor=0.5,patience=1,threshold=1e-3)

    """
    This function differenciates the parameters of the pretrained model used (backbone parameters)
    and of the head of the model (head parameters)
    """
    def get_params(self) ->Tuple[List[torch.nn.Parameter], ...]:
        head_params=[]
        backbone_params=[]
        for module in [self.model.projection_image,self.model.mha,self.model.fc_layers,self.model.fc_norm_layers,self.model.layer_norm_mha]:
            head_params+=[p for p in module.parameters() if p.requires_grad==True]
        for module in [self.model.distilbert_model,self.model.resnet_model]:
            backbone_params+=[p for p in module.parameters() if p.requires_grad==True]
        return head_params,backbone_params
    
    """
    This function freezes the desired layers of the pretrained distilbert
    and resnet model.

    In the disitilbert model, the parameters of the embeddings block are
    automatically frozen. The freezing is customisable only for the encoder layers.

    In the resnet model, the parameters of the all the blocks apart of
    the Convolutionnal layers are also automatically frozen.
    The freezing is customisable only for the convolutionnal layers.

    """
    def freeze_layers(self,
                      n_frozen_distilbert_layers:int,
                      n_frozen_resnet_layers:int):
        
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
    
    """
    This function saves the performances per epoch of the model
    """
    def save_performances(self,
                          path:str,
                          epoch_train_losses:list,
                          epoch_train_f1:list,
                          epoch_train_accuracies:list,
                          epoch_val_losses:list,
                          epoch_val_f1:list,
                          epoch_val_accuracies:list):
        
        train_performances={
            "epoch_train_losses":torch.tensor(epoch_train_losses),
            "epoch_train_f1":torch.tensor(epoch_train_f1),
            "epoch_train_accuracies":torch.tensor(epoch_train_accuracies),

            "epoch_val_losses":torch.tensor(epoch_val_losses),
            "epoch_val_f1":torch.tensor(epoch_val_f1),
            "epoch_val_accuracies":torch.tensor(epoch_val_accuracies)
        }   
        torch.save(train_performances,f"{path}/epoch_performances.pt")

    def run_training(self,
                     train_dataloader: DataLoader,
                     val_dataloader: DataLoader,
                     path: str):
        
        epoch_train_losses=[]
        #list of training loss per epoch
        epoch_train_f1=[]
        #list of f1 score per epoch
        epoch_train_accuracies=[]
        #list of accuracy per epoch

        epoch_val_losses=[]
        #list of validation loss per epoch
        epoch_val_f1=[]
        #list of validation f1 per epoch
        epoch_val_accuracies=[]
        #list of validation accuracy per epoch

        epoch_counter=0
        best_val_f1=0

        for epoch in range(self.n_epochs):
            self.model.train()
            batch_train_losses=[]
            #list of train loss per batch in the current epoch

            all_train_targets=[]
            #list of all the targets of the train dataloader for the batch
            all_train_predictions=[]
            #list of all the prediction of the train dataloader for the batch

            all_val_targets=[]
            #list of all the targets of the validation dataloader for the batch
            all_val_predictions=[]
            #list of all the prediction of the validation dataloader for the batch

            logger.info(f"Epoch {epoch} :")

            for batch in train_dataloader:
                batch["images"] = batch["images"].float().to(self.device)
                batch["input_ids"] = batch["input_ids"].long().to(self.device)
                batch["attention_mask"] = batch["attention_mask"].long().to(self.device)
                logits=self.model(batch["images"],batch["input_ids"],batch["attention_mask"]).float()
                #Forward of the model that returns the logits
                targets=batch["labels"].long().to(self.device)
                loss=self.criterion(logits,targets)
                #Computation of the train batch loss
                self.optimizer.zero_grad()
                #Reset gradients of all parameters to zero
                loss.backward()
                #Computation of the gradient of the loss for each parameter
                self.optimizer.step()
                #Update of the parameters using the gradient descent

                batch_train_losses.append(loss.detach().item())
                predictions=torch.argmax(logits.detach(),dim=1).long()

                all_train_predictions.append(predictions)
                all_train_targets.append(targets)
            
            self.scheduler.step()
            #Update of the learning rate at the end of the epoch

            self.model.eval()
            batch_val_losses=[]
            with torch.no_grad():
                for batch in val_dataloader:
                    batch["images"] = batch["images"].float().to(self.device)
                    batch["input_ids"] = batch["input_ids"].long().to(self.device)
                    batch["attention_mask"] = batch["attention_mask"].long().to(self.device)
                    logits=self.model(batch["images"],batch["input_ids"],batch["attention_mask"]).float()
                    #Forward of the model that returns the logits
                    targets=batch["labels"].long().to(self.device)
                    loss=self.criterion(logits,targets)
                    #Computation of the validation batch loss
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
            #Computation of the train f1 score for the epoch
            epoch_train_accuracies.append(accuracy_score(all_train_targets.cpu().numpy(),all_train_predictions.cpu().numpy()))
            #Computation of the train accuracy score for the epoch
            val_f1_score=f1_score(all_val_targets.cpu().numpy(),all_val_predictions.cpu().numpy(),average="weighted")
            #Computation of the validation f1 score for the epoch
            epoch_val_losses.append(np.mean(batch_val_losses))
            epoch_val_f1.append(val_f1_score)
            epoch_val_accuracies.append(accuracy_score(all_val_targets.cpu().numpy(),all_val_predictions.cpu().numpy()))
            #Computation of the validation accuracy score for the epoch
            #self.scheduler.step(val_f1_score)

            logger.info(f"Epoch {epoch}: Train Loss = {epoch_train_losses[epoch]}")
            logger.info(f"Epoch {epoch}: Train Accuracy = {epoch_train_accuracies[epoch]}")
            logger.info(f"Epoch {epoch}: Train F1 = {epoch_train_f1[epoch]}")

            logger.info(f"Epoch {epoch}: Validation Loss = {epoch_val_losses[epoch]}")
            logger.info(f"Epoch {epoch}: Validation Accuracy = {epoch_val_accuracies[epoch]}")
            logger.info(f"Epoch {epoch}: Validation F1 = {epoch_val_f1[epoch]}")

            #Early stopping
            if val_f1_score<best_val_f1+self.min_improvement:
                epoch_counter+=1
            else:
                epoch_counter=0
                best_val_f1=val_f1_score
                torch.save(self.model.state_dict(),f"{path}/model_state.pt")                
            
            if epoch_counter>=self.patience:
                logger.info(f"Training stops after {epoch} epochs")
                break
        
        self.save_performances(path,epoch_train_losses,epoch_train_f1,epoch_train_accuracies,epoch_val_losses,epoch_val_f1,epoch_val_accuracies)
            

                
