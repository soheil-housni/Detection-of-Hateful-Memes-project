import torch
import torch.nn as nn

class HeadClassifierCLIPModel(nn.Module):
    def __init__(self,
                 use_n_layers :int =1,
                 fc_layer_sizes :list[int] =[1024],
                 dmodel: int=512,
                 dropout: float=0.3,
                 with_scores: bool =False):
        
        """
        Args:
        use_n_layers: number of layers used in the FNN
        fc_layer_sizes: size of each layer used in the FNN
        dmodel: dimensionality of the embeddings output of the CLIP model
        dropout: dropout probability used in the model
        with_scores: define whether we use similarity scores in the input or not
        """
        super().__init__()
        self.dmodel=dmodel
        self.fc_layer_sizes=fc_layer_sizes[:use_n_layers]

        if with_scores:
            self.projection_sim_score=nn.Linear(1,self.dmodel//4)
            fc_layers=[nn.Linear(self.dmodel*4+self.dmodel//4,self.fc_layer_sizes[0])]
        else:
            fc_layers=[nn.Linear(self.dmodel*4,self.fc_layer_sizes[0])]

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
                texts_embeddings: torch.Tensor,
                images_embeddings: torch.Tensor,
                sim_scores:torch.Tensor=None) -> torch.Tensor:
        """"
        Args:
        texts_embeddings: torch tensor of the texts CLIP embeddings (batch_size,512)
        images_embeddings: torch tensor of the images CLIP embeddings (batch_size,512)
        """
        if sim_scores is not None:
            sim_scores=sim_scores.view(-1,1)
            projected_sim_score=self.projection_sim_score(sim_scores)
            x=torch.cat([texts_embeddings,images_embeddings,abs(texts_embeddings-images_embeddings),texts_embeddings*images_embeddings,projected_sim_score],dim=1)
        else:
            x=torch.cat([texts_embeddings,images_embeddings,abs(texts_embeddings-images_embeddings),texts_embeddings*images_embeddings],dim=1)
        #concatenation of the text and image embeddings (batch_size,1024)
        for i in range(len(self.fc_layers)-1):
            x=self.fc_layers[i](x)
            x=self.fc_norm_layers[i](x)
            x=nn.functional.gelu(x)
            x=nn.functional.dropout(x,p=self.dropout,training=self.training)
        logits=self.fc_layers[-1](x)
        return logits