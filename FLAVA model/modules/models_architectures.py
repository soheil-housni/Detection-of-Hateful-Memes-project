import torch
import torch.nn as nn

class HeadClassifierFlavaModel(nn.Module):
    def __init__(self,use_n_layers=1, fc_layer_sizes=[384], dmodel=768, dropout=0.1, with_clip_image=False,with_clip_text=False,clip_dmodel=512):
        super().__init__()
        self.dmodel=dmodel
        self.fc_layer_sizes=fc_layer_sizes[:use_n_layers]
        self.clip_dmodel=clip_dmodel

        if bool(with_clip_image)+bool(with_clip_text)==2:
            self.enter_dim=self.dmodel+self.clip_dmodel*2

        elif bool(with_clip_image)+bool(with_clip_text)==1:
            self.enter_dim=self.dmodel+self.clip_dmodel

        else:
            self.enter_dim=self.dmodel
            
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

        self.first_layer_norm=nn.LayerNorm(self.enter_dim)


    
    def forward(self,pooler_embedding,clip_text_embedding=None,clip_image_embedding=None):
        if clip_text_embedding is not None and clip_image_embedding is not None:
            x=torch.cat([pooler_embedding,clip_text_embedding,clip_image_embedding],dim=1)
        elif clip_text_embedding is not None:
            x=torch.cat([pooler_embedding,clip_text_embedding],dim=1)
        elif clip_image_embedding is not None:
            x=torch.cat([pooler_embedding,clip_image_embedding],dim=1)
        else:
            x=pooler_embedding
            
        x=self.first_layer_norm(x)
        for i in range(len(self.fc_layers)-1):
            x=self.fc_layers[i](x)
            x=self.fc_norm_layers[i](x)
            x=nn.functional.relu(x)
            x=nn.functional.dropout(x,p=self.dropout,training=self.training)
        logits=self.fc_layers[-1](x)
        return logits