import torch
from torch import nn
from torch.nn import MultiheadAttention

class DistilbertResnetModel(nn.Module):
    def __init__(self,distilbert_model,resnet_model,use_n_layers=1, fc_layer_sizes=[384], resnet_dmodel=512, distilbert_dmodel=768, dropout=0.3):
        super().__init__()
        self.distilbert_model=distilbert_model
        self.resnet_model=resnet_model
        self.distilbert_dmodel=distilbert_dmodel
        self.resnet_dmodel=resnet_dmodel
        self.dropout=dropout

        self.projection_image=nn.Linear(self.resnet_dmodel,self.distilbert_dmodel)
        self.mha=MultiheadAttention(self.distilbert_dmodel,num_heads=8,dropout=self.dropout, batch_first=True)

        self.fc_layer_sizes=fc_layer_sizes[:use_n_layers]
        fc_layers=[nn.Linear(self.distilbert_dmodel,self.fc_layer_sizes[0])]
        fc_norm_layers=[nn.LayerNorm(self.fc_layer_sizes[0])]

        for i in range(len(self.fc_layer_sizes)):
            if i == len(self.fc_layer_sizes)-1:
                fc_layers.append(nn.Linear(self.fc_layer_sizes[i],2))
            else:
                fc_layers.append(nn.Linear(self.fc_layer_sizes[i],self.fc_layer_sizes[i+1]))
                fc_norm_layers.append(nn.LayerNorm(self.fc_layer_sizes[i+1]))
        
        self.fc_layers=nn.ModuleList(fc_layers)
        self.fc_norm_layers=nn.ModuleList(fc_norm_layers)

        self.layer_norm_mha=nn.LayerNorm(self.distilbert_dmodel)
        self.layer_norm_image=nn.LayerNorm(self.distilbert_dmodel)
        self.layer_norm_text=nn.LayerNorm(self.distilbert_dmodel)

    
    def forward(self,images,input_ids,attention_mask):

        x_image=self.resnet_model.conv1(images)
        x_image=self.resnet_model.bn1(x_image)
        x_image=self.resnet_model.relu(x_image)
        x_image=self.resnet_model.maxpool(x_image)
        x_image=self.resnet_model.layer1(x_image)
        x_image=self.resnet_model.layer2(x_image)
        x_image=self.resnet_model.layer3(x_image)
        x_image=self.resnet_model.layer4(x_image)

        x_image=x_image.flatten(2).transpose(1,2)
        x_image=self.projection_image(x_image)
        x_image=self.layer_norm_image(x_image)

        x_text=self.distilbert_model(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        x_text=self.layer_norm_text(x_text)

        x=torch.concat([x_image,x_text],dim=1)


        attn_mask_image=torch.ones(attention_mask.shape[0],x_image.shape[1])
        attn_mask=torch.concat([attn_mask_image,attention_mask],dim=1)
        key_padding_mask=(attn_mask==0)
        #attn_mask=attn_mask.unsqueeze(1)*attn_mask.unsqueeze(2)
        
        mha_output=self.mha(query=x,key=x,value=x,key_padding_mask=key_padding_mask )

        x=x+mha_output[0]
        x=self.layer_norm_mha(x)
        x=x.mean(dim=1)

        for i in range(len(self.fc_layers)-1):
            x=self.fc_layers[i](x)
            x=self.fc_norm_layers[i](x)
            x=nn.functional.relu(x)
            x=nn.functional.dropout(x,p=self.dropout,training=self.training)
        logits=self.fc_layers[-1](x)
        return logits






