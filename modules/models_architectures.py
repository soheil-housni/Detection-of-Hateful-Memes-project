import torch
import torch.nn as nn

class ClipModel(nn.Modules):
    def __init__(self, clipmodel,h, use_n_layers=1, fc_layer_sizes=[768], dmodel=512, dropout=0.1):
        super().__init__()

        self.clipmodel=clipmodel

        self.dmodel=dmodel

        self.fc_layer_sizes=fc_layer_sizes[:use_n_layers]
        fc_layers=[nn.Linear(dmodel,self.fc_layer_sizes[0])]
        fc_norm_layers=[nn.LayerNorm(self.fc_layer_sizes[0])]

        for i in range(len(self.fc_layer_sizes)):
            if i == len(self.fc_layer_sizes)-1:
                fc_layers.append(nn.Linear(self.fc_layer_sizes[i],1))
            else:
                fc_layers.append(nn.Linear(self.fc_layer_sizes[i],self.fc_layer_sizes[i+1]))
                fc_norm_layers.append(nn.LayerNorm(self.fc_layer_sizes[i+1]))
        
        self.fc_layers=nn.ModuleList(fc_layers)
        self.fc_norm_layers=nn.ModuleList(fc_norm_layers)

        self.attention=nn.MultiheadAttention(dmodel,8,dropout=0.1,batch_first=True)
        self.dropout=0.1


    
    def forward(self,input_ids, pixel_values, attention_mask, position_ids,batch_size):
        self.clip_output=self.clipmodel(input_ids=input_ids,pixel_values=pixel_values,attention_mask = attention_mask, position_ids = position_ids )
        self.texts_embeddings=self.clip_output["text_embeds"]
        self.images_embeddings=self.clip_output["image_embeds"]
        self.concat=torch.stack([self.texts_embeddings,self.images_embeddings],dim=1)
        self.attention_output=self.attention(self.concat,self.concat,self.concat)
        self.x=self.attention_output.view(batch_size,-1)
        for i in range(len(self.fc_layers)-1):
            self.x=self.fc_layers[i](self.x)
            self.x=self.fc_norm_layers[i](self.x)
            self.x=nn.functional.relu(self.x)
            self.x=nn.functional.dropout(self.x,p=self.dropout,training=self.training)
        logits=self.fc_layers[-1](self.x)
        return logits