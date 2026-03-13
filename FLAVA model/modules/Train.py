import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import torch
from loguru import logger
from .models_architectures import HeadClassifierFLAVAModel
from torch.nn.modules import loss
from torch.utils.data import DataLoader


class Train():
    def __init__(self,
                 model:HeadClassifierFLAVAModel,
                 loss_fn:loss,
                 optmizer:torch.optim,
                 n_epochs:int,
                 scheduler:torch.optim.lr_scheduler,
                 device:torch.device,
                 patience:int = 2,
                 min_improvement:float=1e-3):
        
        """
        Args:
        model: Head classifier for the CLIP model we want to train
        loss_fn: loss function used for the training
        optmizer: optimizer used for the training
        n_epochs: number of epochs for the training
        scheduler: scheduler used to decrease the learning rate during the training
        device: device used for the training
        batch_size: size of the batch
        patience: the number of epochs to wait when the performance does not improve before stopping the training
        min_improvement: the minimum improvement the model need to accomplish to be considered as an actual improvement that reinitialize the patience counter
        """
        
        self.model=model
        self.device=device
        self.criterion=loss_fn
        self.optimizer=optmizer
        self.n_epochs=n_epochs
        self.scheduler=scheduler
        self.model.to(self.device)
        self.patience=patience
        self.min_improvement=min_improvement
    
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
                     train_dataloader:DataLoader,
                     val_dataloader:DataLoader,
                     path: str,
                     multimodal:bool=False,
                     with_clip:bool=False
                     ):
        """
        Args:
        train_dataloader: dataloader of the training set used for the training
        val_dataloader: dataloader of the validation set used for the training
        path: path where to save the performances of the training
        multimodal: boolean indicating whether we use the HeadClassifierMultimodalFLAVAModel or not
        with_clip: boolean determining if we use CLIP embeddings or not (for concatenation)
        """
        
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
                if multimodal:
                    batch["multimodal_embeddings"]=batch["multimodal_embeddings"].float().to(self.device)
                else:
                    batch["pooler_embeddings"]=batch["pooler_embeddings"].float().to(self.device)

                if with_clip:
                    batch["texts_embeddings"]=batch["texts_embeddings"].float().to(self.device)
                    batch["images_embeddings"]=batch["images_embeddings"].float().to(self.device)
                    if not multimodal:
                        logits=self.model(pooler_embedding=batch['pooler_embeddings'],clip_text_embedding=batch['texts_embeddings'],clip_image_embedding=batch["images_embeddings"]).float()
                        #Forward of the model that returns the logits, using CLIP arguments
                    else:
                        logits=self.model(multimodal_embedding=batch['multimodal_embeddings'],clip_text_embedding=batch['texts_embeddings'],clip_image_embedding=batch["images_embeddings"]).float()
                else:
                    if not multimodal:
                        logits=self.model(pooler_embedding=batch['pooler_embeddings']).float()
                    else:
                        logits=self.model(multimodal_embedding=batch['multimodal_embeddings']).float()
                    
                    #Forward of the model that returns the logits, without using CLIP arguments
                targets=batch["labels"].long().to(self.device)
                loss=self.criterion(logits,targets)
                #Computation of the train batch loss

                self.optimizer.zero_grad()
                #Reset gradients of all parameters to zero
                loss.backward()
                #Computation of the gradient of the loss with respect to each parameter
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
                    if multimodal:
                        batch["multimodal_embeddings"]=batch["multimodal_embeddings"].float().to(self.device)
                    else:
                        batch["pooler_embeddings"]=batch["pooler_embeddings"].float().to(self.device)

                    if with_clip:
                        batch["texts_embeddings"]=batch["texts_embeddings"].float().to(self.device)
                        batch["images_embeddings"]=batch["images_embeddings"].float().to(self.device)
                        if not multimodal:
                            logits=self.model(pooler_embedding=batch['pooler_embeddings'],clip_text_embedding=batch['texts_embeddings'],clip_image_embedding=batch["images_embeddings"]).float()
                            #Forward of the model that returns the logits, using CLIP arguments
                        else:
                            logits=self.model(multimodal_embedding=batch['multimodal_embeddings'],clip_text_embedding=batch['texts_embeddings'],clip_image_embedding=batch["images_embeddings"]).float()
                    else:
                        if not multimodal:
                            logits=self.model(pooler_embedding=batch['pooler_embeddings']).float()
                        else:
                            logits=self.model(multimodal_embedding=batch['multimodal_embeddings']).float()
                    
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
                break
        self.save_performances(path,epoch_train_losses,epoch_train_f1,epoch_train_accuracies,epoch_val_losses,epoch_val_f1,epoch_val_accuracies)
            