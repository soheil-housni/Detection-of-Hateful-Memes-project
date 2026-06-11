from transformers import AutoTokenizer,DistilBertModel
from ..model import DistilbertResnetModel
from torchvision.models import resnet18,ResNet18_Weights
import torch
from pathlib import Path

def config_app(
        with_clip_image=False,
        with_clip_text=False,
        concat_interaction=True,
        dropout=0.3,
        fc_layer_sizes=[384]
        ):
    
    config_model_dict={
        "with_clip_image":with_clip_image,
        "with_clip_text":with_clip_text,
        "concat_interaction":concat_interaction,
        "dropout":dropout,
        "fc_layer_sizes":fc_layer_sizes
        }
    
    tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased")
    distilbert_model=DistilBertModel.from_pretrained("distilbert-base-uncased")
    resnet_model=resnet18(weights=ResNet18_Weights.DEFAULT)

    model=DistilbertResnetModel(
        distilbert_model=distilbert_model,
        resnet_model=resnet_model,
        with_clip_image=config_model_dict["with_clip_image"],
        with_clip_text=config_model_dict["with_clip_text"],
        concat_interaction=config_model_dict["concat_interaction"],
        dropout=config_model_dict["dropout"],
        fc_layer_sizes=config_model_dict["fc_layer_sizes"]
    )

    BASE_DIR=str(Path(__file__).resolve().parents[2]).replace("\\","/")
    model_parameters=torch.load(f"{BASE_DIR}/single_train_savings/interaction_emb_concatenation_only/model_state.pt")
    model.load_state_dict(model_parameters)

    config_dict={
        "tokenizer":tokenizer,
        "model":model,
        "with_clip_image":config_model_dict["with_clip_image"],
        "with_clip_text":config_model_dict["with_clip_text"]
    }

    return config_dict

    