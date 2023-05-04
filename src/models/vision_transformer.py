import torch
import torch.nn as nn
import numpy
import torch.nn.functional as F




class ViTInputLayer(nn.Module):
    def __init__(self, in_channels:int=3, 
                embedd_dim:int=128, 
                num_patch_row:int=2, 
                image_size_h:int=32, image_size_w:int=32):
        super(ViTInputLayer).__init__()
        self.in_channels = in_channels
        self.embed_dim = embedd_dim
        self.num_patch_row = num_patch_row
        self.image_size_h = image_size_h
        self.image_size_w = image_size_w
        
        self.num_patch = self.num_patch_row**2
        
        self.patch_size = int(self.image_size_w // self.num_patch_row)#入力画像のH,Wが等しいとした時
        
        # 入力画像のパッチへの分割 & パッチの埋め込みを一気に行う層
        self.embedd_layer = nn.Conv2d(
            in_channels = self.in_channels,
            out_channels = self.embed_dim,
            kernel_size = self.patch_size,
            stride = self.patch_size
        )
        
        self.class_token = nn.Parameter(
            torch.randn(1, 1, self.embed_dim)
        )
        
        self.positional_embed = nn.Parameter(
            torch.randn(1, self.num_patch+1, self.embed_dim)
        )
    
    def forward(self, input_image:torch.Tensor)->torch.Tensor:
        '''
        input.shape : (B, C, H, W)
        output.shape :(B, num_patch+1, embed_dim)
        '''
        z0_token = self.embedd_layer(input_image)#(B, embed_dim, num_patch_row, num_patch_column)
        z0_token = torch.flatten(z0_token, 2)#(B, embed_dim, num_patch)
        z0_token = z0_token.transpose(1,2)
        
        z0_token = torch.concat([ self.class_token.repeat(repeats=(input_image.shape[0], 1, 1)),
                                z0_token])
        z0_token = z0_token + self.positional_embed
        
        return z0_token



class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim:int,head_num:int, dropout:float):
        super(MultiHeadAttention).__init__()
        self.head_num = head_num
        self.embed_dim = embed_dim
        self.head_dim = embed_dim//head_num
        self.sqrt_head_dim = self.head_dim**0.5#token軸の正規化,softmaxの時につける
        
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)
        
        #実装上では使うらしい、headを少なくしても精度が変わらないから、
        #なんならアンサンブル学習させてもええやろってことなのか -> 論文に書いてあるかな？
        self.atten_drop = nn.Dropout(dropout)
        
        self.out_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        
    def forward(self, z0_token:torch.Tensor)->torch.Tensor:
        '''
        input.shape :(B, num_patch+1, embed_dim)
        output.shape:(B, num_patch+1, embed_dim)
        '''
        
        batch_size, token_num, _ = z0_token.size()
        
        query = self.query_layer(z0_token)#(B, token_num = num_patch+1, embed_dim)
        key = self.key_layer(z0_token)
        value = self.value_layer(z0_token)
        
        #multi-head : embed_dim -> head_dim * embed_dim//head_dim
        query = query.reshape(batch_size, token_num, self.head_num, self.head_dim)
        key_T = key.reshape(batch_size, token_num, self.head_num, self.head_dim)
        value = value.reshape(batch_size, token_num, self.head_num, self.head_dim)
        
        #上で一気にこの形に持っていくのもできるかもしれないが心配なので、
        query = query.transpose(1,2)
        value = value.transpose(1, 2)
        
        inner_q_k = query @ key_T / self.sqrt_head_dim#(B, head_num, token_num, token_num)
        
        #列方向なら-2,行方向なら-1、query, keyどっちを起点と考えるかによる
        #個人的にはqueryを起点として考えたい派
        attention_weight = F.softmax(inner_q_k, dim=-2)
        attention_weight = self.atten_drop(attention_weight)
        
        out = attention_weight @ value#(B, head_num, token_num, head_dim)
        out = out.reshape(batch_size, token_num, -1)
        
        out = self.out_layer(out)
        
        return out


class VitEncoderBlock(nn.Module):
    def __init__(self,embed_dim:int, head_num:int, hidden_dim:int, dropout:float):
        '''
        argument:
        
            embed_dim  : token_dim
            head_num   : head_num in multihead_self_attention
                        -> recommended size in paper is four times the size of embed_dim=
            hidden_dim : hidden_dim in mlp_layer
            dropout    : argument of nn.Dropout()
        '''
        
        super(VitEncoderBlock).__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.mhsa = MultiHeadAttention(
            embed_dim = embed_dim,
            head_num = head_num,
            dropout=dropout
        )
        
        self.mlp_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        
        
    def forward(self, z_token:torch.Tensor)->torch.Tensor:
        '''
        input.shape :(B, token_num = num_patch+1, embed_dim)
        output.shape:(B, token_num, embed_dim)
        '''
        out = self.layer_norm(z_token)
        out = self.mhsa(out)
        out = self.layer_norm(out + z_token)
        
        z_next_token = self.mlp_layer(self.layer_norm(out)) + out
        
        return z_next_token


class VitMlpHead():
    def __init__(self, class_num:int, embed_dim:int):
        super(VitMlpHead).__init__()
        self.mltihead = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, class_num)
        )
        
    def forward(self, class_token):
        '''
        inupt.shape : (B, embed_dim)
        outputshape : (B, class_num)
        '''
        output = self.mltihead(class_token) 
        return output



class Vit(nn.Module):
    def __init__(self, class_num:int, in_channels:int, embed_dim:int, num_patch_row, image_size_h, image_size_w,
                head_num:int, hidden_dim:int, dropout:float, encoder_block_num:int):
        
        '''
        argument:
        
            embed_dim  : token_dim
            head_num   : head_num in multihead_self_attention
                        -> recommended size is four times the size of embed_dim
            hidden_dim : hidden_dim in mlp_layer of mlp_head
            dropout    : argument of nn.Dropout()
        '''
        
        self.input_layer = ViTInputLayer(
            in_channels,
            embed_dim,
            num_patch_row,
            image_size_h,
            image_size_w)
        
        
        self.encoder = nn.Sequential(
            VitEncoderBlock(embed_dim, head_num, hidden_dim, dropout)
            for _ in range(encoder_block_num)
        )
        
        self.mlp_head = VitMlpHead(class_num, embed_dim)
        
    def forward(self, input_image:torch.Tensor)->torch.Tensor:
        '''
        input.shape : (B, C, H, W)
        out.shape   : (B, class_num)
        
            B:batch_size, C:num_image_channel, H:image_height, W:image:width
        '''
        z0_token = self.input_layer(input_image)
        z_last_token = self.encoder(z0_token)[:, 0]
        class_token = z_last_token[:, 0]
        output = self.mlp_head(class_token)
        
        return output