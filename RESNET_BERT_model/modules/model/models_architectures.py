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
                 fc_layer_sizes : list[int] =[768],
                 resnet_dmodel : int =512,
                 distilbert_dmodel : int =768,
                 dropout : float =0.3,
                 dropout_ca:float=0.3,
                 concat_interaction:bool=False,
                 simple_concat:bool=False,
                 with_clip_text:bool=False,
                 with_clip_image:bool=False,
                 clip_dmodel:int=512):
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
        self.clip_dmodel=clip_dmodel
        self.dropout=dropout
        self.dropout_ca=dropout_ca
        self.concat_interaction=concat_interaction
        self.simple_concat=simple_concat
        self.use_n_layers=use_n_layers

        #Projection of the output of the resnet model (batch_size,49,512) in the space of the ouput of the distilbertmodel (batch_size,seq_len_768)
        #For the output of the resnet model, seq_len=49 as each kernel is of dimension (7,7), so when flattened, we have 49 patches.
        self.projection_image=nn.Linear(self.resnet_dmodel,self.distilbert_dmodel)
        self.layer_norm_image=nn.LayerNorm(self.distilbert_dmodel)

        self.layer_norm_text=nn.LayerNorm(self.distilbert_dmodel)

        #Self Multi-Head-Attention applied to the the concatennated tokens and image patches
        self.ca_text=MultiheadAttention(self.distilbert_dmodel,num_heads=8,dropout=self.dropout_ca, batch_first=True)
        self.ca_image=MultiheadAttention(self.distilbert_dmodel,num_heads=8,dropout=self.dropout_ca, batch_first=True)

        self.fnn_ca_image=nn.Sequential(
            nn.Linear(self.distilbert_dmodel,self.distilbert_dmodel//2),
            nn.GELU(),
            nn.Dropout(p=self.dropout_ca),
            nn.Linear(self.distilbert_dmodel//2,self.distilbert_dmodel),
        )

        self.fnn_ca_text=nn.Sequential(
            nn.Linear(self.distilbert_dmodel,self.distilbert_dmodel//2),
            nn.GELU(),
            nn.Dropout(p=self.dropout_ca),
            nn.Linear(self.distilbert_dmodel//2,self.distilbert_dmodel),
        )

        self.layer_norm_ca_attn_image=nn.LayerNorm(self.distilbert_dmodel)
        self.layer_norm_ca_attn_text=nn.LayerNorm(self.distilbert_dmodel)

        self.layer_norm_ca_fnn_image=nn.LayerNorm(self.distilbert_dmodel)
        self.layer_norm_ca_fnn_text=nn.LayerNorm(self.distilbert_dmodel)


        self.norm_text_pool=nn.LayerNorm(self.distilbert_dmodel)
        self.norm_image_pool=nn.LayerNorm(self.distilbert_dmodel)

        if self.concat_interaction:
            self.projection_x=nn.Linear(4*self.distilbert_dmodel,self.distilbert_dmodel)
        
        if self.simple_concat:
            self.projection_x=nn.Linear(2*self.distilbert_dmodel,self.distilbert_dmodel)
        
        self.norm_x=nn.LayerNorm(self.distilbert_dmodel)
            

        if with_clip_image and not with_clip_text:
            self.norm_clip_image=nn.LayerNorm(self.clip_dmodel)

        if not with_clip_image and with_clip_text:
            self.norm_clip_text=nn.LayerNorm(self.clip_dmodel)

        if with_clip_image and with_clip_text:
            self.projection_clip_features=nn.Linear(self.clip_dmodel*2,self.clip_dmodel)
            self.norm_clip_features=nn.LayerNorm(self.clip_dmodel)

        #FFN
        """
        if not self.concat_interaction and not self.simple_concat:
            if bool(with_clip_image)+bool(with_clip_text)==2:
                self.enter_dim=self.distilbert_dmodel*3

            elif bool(with_clip_image)+bool(with_clip_text)==1:
                self.enter_dim=self.distilbert_dmodel*2
            else:
                self.enter_dim=distilbert_dmodel

        elif self.concat_interaction and not self.simple_concat:
            if bool(with_clip_image)+bool(with_clip_text)==2:
                self.enter_dim=self.distilbert_dmodel*6
            elif bool(with_clip_image)+bool(with_clip_text)==1:
                self.enter_dim=self.distilbert_dmodel*5
            else:
                self.enter_dim=distilbert_dmodel*4

        elif self.simple_concat and not self.concat_interaction:
            if bool(with_clip_image)+bool(with_clip_text)==2:
                self.enter_dim=self.distilbert_dmodel*4

            elif bool(with_clip_image)+bool(with_clip_text)==1:
                self.enter_dim=self.distilbert_dmodel*3
            else:
                self.enter_dim=distilbert_dmodel*2
        
        else:
            raise ValueError("Can't use both concat_interaction and simple_concat modes at the same time")

        """
        if with_clip_image or with_clip_text:
            self.enter_dim=self.distilbert_dmodel+self.clip_dmodel
        else:
            self.enter_dim=self.distilbert_dmodel


        self.first_layer_norm=nn.LayerNorm(self.enter_dim)

        self.fc_layer_sizes=fc_layer_sizes[:self.use_n_layers]
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

        self.dropout_layer=nn.Dropout(p=self.dropout_ca)

    def forward(self,
                images : torch.Tensor,
                input_ids : torch.Tensor,
                attention_mask : torch.Tensor,
                clip_text_embeddings: torch.Tensor=None,
                clip_image_embeddings: torch.Tensor=None) -> torch.Tensor:
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

        attn_mask_image=torch.ones(attention_mask.shape[0],x_image.shape[1],device=attention_mask.device)  

        ca_image_output=self.ca_image(query=x_image,key=x_text,value=x_text,key_padding_mask=(attention_mask==0))
        x_image=x_image+self.dropout_layer(ca_image_output[0])
        #residual connection adding the cross attention output
        x_image=self.layer_norm_ca_attn_image(x_image)
        #normalization after the first residual connection adding the cross attention output
        x_image=x_image+self.dropout_layer(self.fnn_ca_image(x_image))
        #residual connection adding the fnn output
        x_image=self.layer_norm_ca_fnn_image(x_image)
        #normalization after the second residual connection adding the fnn output

        ca_text_output=self.ca_text(query=x_text,key=x_image,value=x_image,key_padding_mask=(attn_mask_image==0))
        x_text=x_text+self.dropout_layer(ca_text_output[0])
        #residual connection adding the cross attention output
        x_text=self.layer_norm_ca_attn_text(x_text)
        #normalization after the first residual connection adding the cross attention output
        x_text=x_text+self.dropout_layer(self.fnn_ca_text(x_text))
        #residual connection adding the fnn output
        x_text=self.layer_norm_ca_fnn_text(x_text)
        #normalization after the second residual connection adding the fnn output

        attention_mask_sum=attention_mask.sum(dim=1,keepdim=True).clamp(1)
        x_text_pooled=(x_text*attention_mask.unsqueeze(2)).sum(dim=1)/attention_mask_sum
        x_text_pooled=self.norm_text_pool(x_text_pooled)

        x_image_pooled=x_image.mean(dim=1)
        x_image_pooled=self.norm_image_pool(x_image_pooled)


        if clip_image_embeddings is not None and clip_text_embeddings is None:
            clip_image_embeddings=self.norm_clip_image(clip_image_embeddings)
        
        if clip_text_embeddings is not None and clip_image_embeddings is None:
            clip_text_embeddings=self.norm_clip_text(clip_text_embeddings)
        
        if clip_image_embeddings is not None and clip_text_embeddings is not None:
            clip_features=torch.concat([clip_image_embeddings*clip_text_embeddings,abs(clip_image_embeddings-clip_text_embeddings)],dim=1)
            clip_features=self.projection_clip_features(clip_features)
            clip_features=self.norm_clip_features(clip_features)


        if not self.concat_interaction and not self.simple_concat:
            x=(x_text_pooled+x_image_pooled)/2
            x=self.norm_x(x)
            if clip_image_embeddings is not None and clip_text_embeddings is not None:
                x=torch.concat([x,clip_features],dim=1)
            elif clip_image_embeddings is not None and clip_text_embeddings is None:
                x=torch.concat([x,clip_image_embeddings],dim=1)
            elif clip_image_embeddings is None and clip_text_embeddings is not None:
                x=torch.concat([x,clip_text_embeddings],dim=1)
            else:
                x=x
        elif self.concat_interaction and not self.simple_concat:
            x=torch.concat([x_text_pooled,x_image_pooled,abs(x_text_pooled-x_image_pooled),x_text_pooled*x_image_pooled],dim=1)
            x=self.projection_x(x)
            x=self.norm_x(x)
            if clip_image_embeddings is not None and clip_text_embeddings is not None:
                x=torch.concat([x,clip_features],dim=1)
            elif clip_image_embeddings is not None and clip_text_embeddings is None:
                x=torch.concat([x,clip_image_embeddings],dim=1)
            elif clip_image_embeddings is None and clip_text_embeddings is not None:
                x=torch.concat([x,clip_text_embeddings],dim=1)
            else:
                x=x
        
        elif self.simple_concat and not self.concat_interaction:
            x=torch.concat([x_text_pooled,x_image_pooled],dim=1)
            x=self.projection_x(x)
            x=self.norm_x(x)
            if clip_image_embeddings is not None and clip_text_embeddings is not None:
                x=torch.concat([x,clip_features],dim=1)
            elif clip_image_embeddings is not None and clip_text_embeddings is None:
                x=torch.concat([x,clip_image_embeddings],dim=1)
            elif clip_image_embeddings is None and clip_text_embeddings is not None:
                x=torch.concat([x,clip_text_embeddings],dim=1)
            else:
                x=x
                
        else:
            raise ValueError("Can't use both concat_interaction and simple_concat modes at the same time")


        x=self.first_layer_norm(x)
        for i in range(len(self.fc_layers)-1):
            x=self.fc_layers[i](x)
            x=self.fc_norm_layers[i](x)
            x=nn.functional.gelu(x)
            x=nn.functional.dropout(x,p=self.dropout,training=self.training)
        logits=self.fc_layers[-1](x)
        return logits


        """
        x=torch.concat([x_image,x_text],dim=1)
        #We concatenate the outputed images and the outputed texts along the seq_len dimension (dim=1):
        #(batch_size,text_seq_len+49,768)
        x=x.mean(dim=1)
        #averaging over the sequence length (dim=1), so have an output of dimension (batch_size,768)

        for i in range(len(self.fc_layers)-1):
            x=self.fc_layers[i](x)
            x=self.fc_norm_layers[i](x)
            x=nn.functional.relu(x)
            x=nn.functional.dropout(x,p=self.dropout,training=self.training)
        logits=self.fc_layers[-1](x)
        return logits
        """






