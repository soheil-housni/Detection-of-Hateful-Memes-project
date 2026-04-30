import gc
import torch
from torch.utils.data import DataLoader
from loguru import logger
from ..data_preparation import CreationDataset,CollateFunction
from . import DistilbertResnetModel
from . import Train
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../..")))
from common_files import seed_worker
import numpy as np

class OptunaFunction():
    def __init__(self,
                 train_df,
                 val_df,
                 tokenizer,
                 clip_embeddings,
                 device,
                 resnet_model,
                 distilbert_model,
                 hyperparemeters_ranges,
                 loss_fn,
                 with_clip_image,
                 with_clip_text,
                 concat_interaction,
                 simple_concat,
                 n_epochs,
                 seed:int=42):
        
        self.train_df=train_df
        self.val_df=val_df
        self.tokenizer=tokenizer
        self.device=device
        self.hyperparemeters_ranges=hyperparemeters_ranges
        self.seed=seed
        self.resnet_model=resnet_model
        self.distilbert_model=distilbert_model
        self.loss_fn=loss_fn

        self.with_clip_image=with_clip_image
        self.with_clip_text=with_clip_text
        self.concat_interaction=concat_interaction
        self.simple_concat=simple_concat
        self.n_epochs=n_epochs

    
    def objective(self,trial):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        try:
            batch_size = trial.suggest_categorical("batch_size",self.hyperparemeters_ranges["batch_size"])
            dropout = trial.suggest_float("dropout",self.hyperparemeters_ranges["dropout"]["low"],self.hyperparemeters_ranges["dropout"]["high"],step=self.hyperparemeters_ranges["dropout"]["step"])
            dropout_ca = trial.suggest_float("dropout_ca",self.hyperparemeters_ranges["dropout_ca"]["low"],self.hyperparemeters_ranges["dropout_ca"]["high"],step=self.hyperparemeters_ranges["dropout_ca"]["step"])
            lr=trial.suggest_float("lr",self.hyperparemeters_ranges["lr"]["low"],self.hyperparemeters_ranges["lr"]["high"],log=self.hyperparemeters_ranges["lr"]["log"])
            weight_decay=trial.suggest_float("weight_decay",self.hyperparemeters_ranges["weight_decay"]["low"],self.hyperparemeters_ranges["weight_decay"]["high"],log=self.hyperparemeters_ranges["weight_decay"]["log"])
            warmup_prop=trial.suggest_float("warmup_prop",self.hyperparemeters_ranges["warmup_prop"]["low"],self.hyperparemeters_ranges["warmup_prop"]["high"],step=self.hyperparemeters_ranges["warmup_prop"]["step"])
            use_n_layers=trial.suggest_int("use_n_layers",self.hyperparemeters_ranges["use_n_layers"]["low"],self.hyperparemeters_ranges["use_n_layers"]["high"],step=self.hyperparemeters_ranges["use_n_layers"]["step"])
            fc_layer_1_size=trial.suggest_int("fc_layer_1_size",self.hyperparemeters_ranges["fc_layer_1_size"]["low"],self.hyperparemeters_ranges["fc_layer_1_size"]["high"],step=self.hyperparemeters_ranges["fc_layer_1_size"]["step"])
            fc_layer_2_size=trial.suggest_int("fc_layer_2_size",self.hyperparemeters_ranges["fc_layer_2_size"]["low"],self.hyperparemeters_ranges["fc_layer_2_size"]["high"],step=self.hyperparemeters_ranges["fc_layer_2_size"]["step"])
            fc_layer_3_size=trial.suggest_int("fc_layer_3_size",self.hyperparemeters_ranges["fc_layer_3_size"]["low"],self.hyperparemeters_ranges["fc_layer_3_size"]["high"],step=self.hyperparemeters_ranges["fc_layer_3_size"]["step"])
            fc_layers_sizes=[fc_layer_1_size,fc_layer_2_size,fc_layer_3_size]
            n_frozen_distilbert_layers=trial.suggest_int("n_frozen_distilbert_layers",self.hyperparemeters_ranges["n_frozen_distilbert_layers"]["low"],self.hyperparemeters_ranges["n_frozen_distilbert_layers"]["high"],step=self.hyperparemeters_ranges["n_frozen_distilbert_layers"]["step"])
            n_frozen_resnet_layers=trial.suggest_int("n_frozen_resnet_layers",self.hyperparemeters_ranges["n_frozen_resnet_layers"]["low"],self.hyperparemeters_ranges["n_frozen_resnet_layers"]["high"],step=self.hyperparemeters_ranges["n_frozen_resnet_layers"]["step"])


            train_dataset=CreationDataset(self.train_df,"../CLIP_model/modules/clip_embeddings/train_clip_embeddings.pt")
            val_dataset=CreationDataset(self.train_df,"../CLIP_model/modules/clip_embeddings/val_clip_embeddings.pt")

            generator=torch.Generator()
            generator.manual_seed(self.seed)
            collate_function_obj=CollateFunction(self.tokenizer)
            train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_function_obj.collate_fn,worker_init_fn=seed_worker,generator=generator)
            val_dataloader=DataLoader(val_dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_function_obj.collate_fn,worker_init_fn=seed_worker,generator=generator)

            n_steps=len(train_dataloader)*self.n_epochs
            n_warmup_steps=int(warmup_prop*n_steps)

            model=DistilbertResnetModel(self.distilbert_model,self.resnet_model,with_clip_image=self.with_clip_image,with_clip_text=self.with_clip_text,concat_interaction=self.concat_interaction,dropout=dropout,fc_layer_sizes=fc_layers_sizes,use_n_layers=use_n_layers,dropout_ca=dropout_ca,simple_concat=self.simple_concat)
            trainer=Train(model=model,loss_fn=self.loss_fn,n_epochs=self.n_epochs,device=self.device,n_steps=n_steps,n_warmup_steps=n_warmup_steps,n_frozen_distilbert_layers=n_frozen_distilbert_layers,n_frozen_resnet_layers=n_frozen_resnet_layers,weight_decay=weight_decay,lr=lr,with_clip_images=self.with_clip_image,with_clip_text=self.with_clip_text,concat=self.concat_interaction)

            if not os.path.exists(f"./train_savings/model_{trial.number}"):
                os.mkdir(f"./train_savings/model_{trial.number}")

            path=f"./train_savings/model_{trial.number}"
            logger.info(f"Trial number {trial.number}: ")
            strict_best_val_f1=trainer.Train(path,trial)
            logger.info("-------------------------------------------------------------------")

            return strict_best_val_f1

        finally:
            del model
            del trainer
            del train_dataloader
            del val_dataloader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()

            


