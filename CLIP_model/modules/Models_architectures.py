import torch
import torch.nn as nn

class HeadClassifierClipModel(nn.Module):
    def __init__(self,use_n_layers=1, fc_layer_sizes=[512], dmodel=512, dropout=0.1):
        super().__init__()
        self.dmodel=dmodel
        self.fc_layer_sizes=fc_layer_sizes[:use_n_layers]
        fc_layers=[nn.Linear(self.dmodel*2,self.fc_layer_sizes[0])]
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


    
    def forward(self,texts_embeddings, images_embeddings):
        x=torch.cat([texts_embeddings,images_embeddings],dim=1)
        for i in range(len(self.fc_layers)-1):
            x=self.fc_layers[i](x)
            x=self.fc_norm_layers[i](x)
            x=nn.functional.relu(x)
            x=nn.functional.dropout(x,p=self.dropout,training=self.training)
        logits=self.fc_layers[-1](x)
        return logits