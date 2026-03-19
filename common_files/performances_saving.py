import torch

"""
This function saves the performances per epoch of the model, especially losses, f1 scores, and accuracy scores,
for both the training set and the validation set.
"""

def save_performances(
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
