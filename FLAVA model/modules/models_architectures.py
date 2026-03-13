import torch
import torch.nn as nn

class HeadClassifierFLAVAModel(nn.Module):
    def __init__(self,
                 use_n_layers:int=1,
                 fc_layer_sizes:list[int]=[384],
                 dmodel:int=768,
                 dropout:float=0.3,
                 with_clip_image:bool=False,
                 with_clip_text:bool=False,
                 clip_dmodel:int=512):
        
        """
        Args:
        use_n_layers: number of layers to used in the FNN
        fc_layer_sizes: sizes of the layers used in the FNN
        dmodel: hidden dimension of the outputs of the pretrained FLAVA model
        dropout: dropout probability used in the model
        with_clip_image: use of the CLIP images embeddings to concatenate with FLAVA embeddings
        with_clip_text: use of the CLIP texts embeddings to concatenate with FLAVA embeddings
        clip_dmodel: hidden dimension of CLIP embeddings
        """
        
        super().__init__()
        self.dmodel=dmodel
        self.fc_layer_sizes=fc_layer_sizes[:use_n_layers]
        self.clip_dmodel=clip_dmodel

        if bool(with_clip_image)+bool(with_clip_text)==2:
            self.enter_dim=self.dmodel*3

        elif bool(with_clip_image)+bool(with_clip_text)==1:
            self.enter_dim=self.dmodel*2
        else:
            self.enter_dim=self.dmodel
        
        self.projection_clip_image=nn.Linear(self.clip_dmodel,self.dmodel)
        self.projection_clip_text=nn.Linear(self.clip_dmodel,self.dmodel)

        self.norm_proj_image=nn.LayerNorm(self.dmodel)
        self.norm_proj_text=nn.LayerNorm(self.dmodel)

        self.first_layer_norm=nn.LayerNorm(self.enter_dim)
            
        fc_layers=[nn.Linear(self.enter_dim,self.fc_layer_sizes[0])]
        fc_norm_layers=[nn.LayerNorm(self.fc_layer_sizes[0])]

        for i in range(len(self.fc_layer_sizes)):
            if i == len(self.fc_layer_sizes)-1:
                fc_layers.append(nn.Linear(self.fc_layer_sizes[i],2))
            else:
                fc_layers.append(nn.Linear(self.fc_layer_sizes[i],self.fc_layer_sizes[i+1]))
                fc_norm_layers.append(nn.LayerNorm(self.fc_layer_sizes[i+1]))
        
        self.fc_layers=nn.ModuleList(fc_layers)
        self.fc_norm_layers=nn.ModuleList(fc_norm_layers)

        self.dropout=dropout
    
    def forward(self,
                pooler_embedding=None,
                multimodal_embedding=None,
                clip_text_embedding=None,
                clip_image_embedding=None):
        
        """
        Args:
        pooler_embedding: torch tensor of the pretrained FLAVA pooler embeddings (batch_size,768)
        multimodal_embeddings: torch tensor of the pretrained FLAVA pooler embeddings (batch_size,326,768)
        clip_text_embedding: torch tensor of the pretrained CLIP texts embeddings (batch_size,512)
        clip_images_embedding: torch tensor of the pretrained CLIP images embeddings (batch_size,512)

        Return:
        logits: torch tensor (batch_size,n_classes)=(batch_size,2)
        """
        if pooler_embedding is not None and multimodal_embedding is None:
            flava_embedding=pooler_embedding
        elif pooler_embedding is None and multimodal_embedding is not None:
            flava_embedding=multimodal_embedding.mean(dim=1)
        else:
            raise ValueError("Cannot use both the pooler embedding and the multimodal embeddings as input simultaneously")
        
        if clip_text_embedding is not None:
            clip_text_embedding=self.projection_clip_text(clip_text_embedding)
            clip_text_embedding=self.norm_proj_text(clip_text_embedding)
        
        if clip_image_embedding is not None:
            clip_image_embedding=self.projection_clip_image(clip_image_embedding)
            clip_image_embedding=self.norm_proj_image(clip_image_embedding)

        if clip_text_embedding is not None and clip_image_embedding is not None:
            x=torch.cat([flava_embedding,clip_text_embedding,clip_image_embedding],dim=1)
        elif clip_text_embedding is not None:
            x=torch.cat([flava_embedding,clip_text_embedding],dim=1)
        elif clip_image_embedding is not None:
            x=torch.cat([flava_embedding,clip_image_embedding],dim=1)
        else:
            x=flava_embedding
        #concatenation of FLAVA and CLIP embeddings if possible
        x=self.first_layer_norm(x)
        for i in range(len(self.fc_layers)-1):
            x=self.fc_layers[i](x)
            x=self.fc_norm_layers[i](x)
            x=nn.functional.relu(x)
            x=nn.functional.dropout(x,p=self.dropout,training=self.training)
        logits=self.fc_layers[-1](x)
        return logits
    