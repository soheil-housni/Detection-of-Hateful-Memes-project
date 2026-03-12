import torch
from torch import nn
from torch.nn import MultiheadAttention
from transformers import DistilBertModel
from torchvision.models import ResNet

class DistilbertResnetModel(nn.Module):
    def __init__(self,
                 distilbert_model:DistilBertModel,
                 resnet_model : ResNet,
                 use_n_layers : int = 1,
                 fc_layer_sizes : list[int] =[384] ,
                 resnet_dmodel : int =512,
                 distilbert_dmodel : int =768,
                 dropout : float =0.3):
        """
        Args:
        distilbert_model: the pretrained DistilBert model used as the transformer in the model
        resnet_model: the pretrained ResNet model used as the CNN in the model
        use_n_layers: number of layers used in the FNN of the model
        resnet_dmodel: the number of channels of the output of the resnet model.
                        Before the average pooling operation, we have 512 channels and each kernel is of dimension (7,7)
        distilbert_dmodel: the hidden dimension of the output of the distilber transformer. Each token has an embedding of dimension 768
        dropout: the dropout probability used in the model   
        """
        super().__init__()
        self.distilbert_model=distilbert_model
        self.resnet_model=resnet_model
        self.distilbert_dmodel=distilbert_dmodel
        self.resnet_dmodel=resnet_dmodel
        self.dropout=dropout

        #Projection of the output of the resnet model (batch_size,49,512) in the space of the ouput of the distilbertmodel (batch_size,seq_len_768)
        #For the output of the resnet model, seq_len=49 as each kernel is of dimension (7,7), so when flattened, we have 49 patches.
        self.projection_image=nn.Linear(self.resnet_dmodel,self.distilbert_dmodel)
        self.layer_norm_image=nn.LayerNorm(self.distilbert_dmodel)

        self.layer_norm_text=nn.LayerNorm(self.distilbert_dmodel)

        #Self Multi-Head-Attention applied to the the concatennated tokens and image patches
        self.mha=MultiheadAttention(self.distilbert_dmodel,num_heads=8,dropout=self.dropout, batch_first=True)

        self.layer_norm_mha=nn.LayerNorm(self.distilbert_dmodel)

        #FFN
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

    
    def forward(self,
                images : torch.Tensor,
                input_ids : torch.Tensor,
                attention_mask : torch.Tensor) -> torch.Tensor:
        """
        Args:
        images: tensor of the images (batch_size,224,224)
        input_ids: tensor of the input ids of the tokens (batch_size,seq_len)
        attention_mask: tensor of the attention masks of the tokens (batch_size, seq_len)

        Return:
        logits: tensor of the logits (batch_size,n_classes)=(batch_size,2)
        """
        x_image=self.resnet_model.conv1(images)
        x_image=self.resnet_model.bn1(x_image)
        x_image=self.resnet_model.relu(x_image)
        x_image=self.resnet_model.maxpool(x_image)
        x_image=self.resnet_model.layer1(x_image)
        x_image=self.resnet_model.layer2(x_image)
        x_image=self.resnet_model.layer3(x_image)
        x_image=self.resnet_model.layer4(x_image)
        #We take the output of the 4th layer of the resnet model (just before the average pooling operation)

        x_image=x_image.flatten(2).transpose(1,2)
        #We flatten each kernel and we transpose the last two dimensions, so the sequence length equals the number of patches within a kernel
        #Thus, we go from (batch_size,512,7,7) to (batch_size,49,512)
        x_image=self.projection_image(x_image)
        #We project our outputed image into the space of the texts
        x_image=self.layer_norm_image(x_image)

        x_text=self.distilbert_model(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        x_text=self.layer_norm_text(x_text)

        x=torch.concat([x_image,x_text],dim=1)
        #We concatenate the outputed images and the outputed texts along the seq_len dimension (dim=1):
        #(batch_size,text_seq_len+49,768)


        attn_mask_image=torch.ones(attention_mask.shape[0],x_image.shape[1])
        attn_mask=torch.concat([attn_mask_image,attention_mask],dim=1)
        #We create the attention mask for the images for the Multi-Head-Attention
        #The "tokens" of each image have a value of 1,as they can attend to every other tokens of the sequence
        key_padding_mask=(attn_mask==0)
        #attn_mask=attn_mask.unsqueeze(1)*attn_mask.unsqueeze(2)
        
        mha_output=self.mha(query=x,key=x,value=x,key_padding_mask=key_padding_mask )

        x=x+mha_output[0]
        #residual connection
        x=self.layer_norm_mha(x)
        x=x.mean(dim=1)
        #averaging over the sequence length (dim=1), so have an output of dimension (batch_size,768)

        for i in range(len(self.fc_layers)-1):
            x=self.fc_layers[i](x)
            x=self.fc_norm_layers[i](x)
            x=nn.functional.relu(x)
            x=nn.functional.dropout(x,p=self.dropout,training=self.training)
        logits=self.fc_layers[-1](x)
        return logits






