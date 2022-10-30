#! -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.sqrt(d_model)
        self.dropout = nn.Dropout(attn_dropout)
    def forward(self, q, k, v, mask):
        k=torch.unsqueeze(k,1) # n_head * batch_size, 1, len_q, d_k
        q=torch.unsqueeze(q,3) # n_head * batch_size, len_q, d_k, 1
        attn=torch.matmul(k,q) # n_head * batch_size, len_q, len_q, 1
        attn=torch.squeeze(attn) # n_head * batch_size, len_q, len_q
        attn=attn/self.temper
        if mask is not None:
            pass

        attn=nn.Softmax(dim=-1)(attn)
        attn = self.dropout(attn)
        output=torch.matmul(attn,v) #n_head * batch_size, len_q, d_v
        return output, attn

class TransformAttention(nn.Module):
    def __init__(self,K,in_planes):
        super(TransformAttention, self).__init__()
        self.fc1 = nn.Linear(in_planes,K)
        self.fc2 = nn.Linear(K, K)
    def forward(self,x):
        x=torch.mean(x,1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return nn.Sigmoid()(x)

class MultiHeadAttention(nn.Module):
    def __init__(self,n_head, d_model, d_k, d_v, num_experts, dropout, mode=0, use_norm=True):
        super(MultiHeadAttention, self).__init__()
        self.mode = mode
        self.d_model=d_model
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        self.num_experts = num_experts #Dynamic Transformer parameter
        if mode == 0:
            self.qs_layer = nn.Linear(d_model,n_head*d_k*self.num_experts, bias=False)
            self.num_expertss_layer = nn.Linear(d_model,n_head*d_k*self.num_experts, bias=False)
            self.vs_layer = nn.Linear(d_model,n_head*d_v*self.num_experts, bias=False)
        elif mode == 1:
            pass

        self.tranfrom_attention=TransformAttention(self.num_experts,d_model)
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = nn.LayerNorm(d_model) if use_norm else None
        self.w_o=nn.Linear(n_head*d_v,d_model)

    def forward(self, q, k, v, x_,mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        attention=self.tranfrom_attention(x_) #[batch_size,K]
        if len(attention.size())<2:  
            attention=torch.unsqueeze(attention,0)

        if self.mode == 0:

            def dynamic_transformer(weight,input):
                s=weight.size()
                weight=weight.view(s[0],self.num_experts,-1)
                weight=weight.permute(1,0,2)
                weight=weight.contiguous().view(self.num_experts,-1)
                weight=torch.mm(attention,weight)
                weight=weight.view(attention.size()[0],self.d_model,-1) #[batch_size,d_model,n_head*d_k]
                weight = torch.unsqueeze(weight, 1) #[batch_size,1,d_model,n_head*d_k]
                input = torch.unsqueeze(input, 2)  #[batch_size,seq_len,1,d_model]
                attn = torch.matmul(input, weight) #[batch_size,seq_len,1,n_head*d_k]
                attn = torch.squeeze(attn)
                if len(attn.size()) < 3: 
                    attn = torch.unsqueeze(attn, 0)
                return attn

            # qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            # ks = self.num_expertss_layer(k)
            # vs = self.vs_layer(v)
            qs=dynamic_transformer(self.qs_layer.weight.t(),q)
            ks=dynamic_transformer(self.num_expertss_layer.weight.t(),k)
            vs=dynamic_transformer(self.vs_layer.weight.t(),v)

            def reshape1(x):
                s = x.size()   # [batch_size, len_q, n_head * d_k]
                x=x.view(s[0],s[1],n_head,d_k)
                x=x.permute(2,0,1,3)
                x=x.contiguous().view(-1,s[1],d_k)  # [n_head * batch_size, len_q, d_k]
                return x
            qs = reshape1(qs)
            ks = reshape1(ks)
            vs = reshape1(vs)

            if mask is not None:
                pass
                # mask = Lambda(lambda x:K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)

            def reshape2(x):
                s = x.size()   # [n_head * batch_size, len_v, d_v]
                x = x.view(n_head, -1, s[1], s[2])
                x = x.permute(1, 2, 0, 3)
                x = x.contiguous().view(-1, s[1], n_head*d_v)  # [batch_size, len_v, n_head * d_v]
                return x
            head = reshape2(head)
        elif self.mode == 1:
            pass


        outputs = self.w_o(head)
        outputs = nn.Dropout(self.dropout)(outputs)
        if not self.layer_norm:
            return outputs, attn
        outputs=outputs+q
        return self.layer_norm(outputs), attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_hid,d_inner_hid, 1)
        self.w_2 = nn.Conv1d(d_inner_hid,d_hid, 1)
        self.layer_norm =nn.LayerNorm(d_hid)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        output = self.w_1(x.permute(0,2,1))
        output=output.permute(0,2,1)
        output=nn.ReLU()(output)
        output = output.permute(0, 2, 1)
        output = self.w_2(output)
        output = output.permute(0, 2, 1)
        output = self.dropout(output)
        output = output+x
        return self.layer_norm(output)

class EncoderLayer(nn.Module):
    def __init__(self,d_model, d_inner_hid, n_head, d_k, d_v, num_experts, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, num_experts, dropout=dropout)
        self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
    def forward(self, enc_input, mask=None):
        output, slf_attn = self.self_att_layer(enc_input[0], enc_input[0], enc_input[0],enc_input[1], mask=mask)
        output = self.pos_ffn_layer(output)
        return [output, enc_input[1]]



class Encoder(nn.Module):
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, num_experts, layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.emb_dropout = nn.Dropout(dropout)
        blocks = [EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, num_experts, dropout) for _ in range(layers)]
        self.layers = nn.Sequential(*blocks)
    def forward(self, x, return_att=False, mask=None, active_layers=999):
        x[0] = self.emb_dropout(x[0])
        ans = self.layers(x)
        return ans[0]

class Model(nn.Module):
    def __init__(self, seq_len, revw_feature,d_feature, d_model=64, d_inner_hid=128, n_head=3, d_k=64, d_v=64, num_experts=4, layers=4,dropout=0.1):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.d_feature = d_feature
        self.d_model = d_model
        self.revw_feature=revw_feature
        self.encoder = Encoder(d_model, d_inner_hid, n_head, d_k, d_v, num_experts, layers, dropout)

        MLP_modules_1 = []
        MLP_modules_1.append(nn.Linear(d_feature*2, d_model))
        self.MLP_layers_1 = nn.Sequential(*MLP_modules_1)

        MLP_modules_2 = []
        MLP_modules_2.append(nn.Linear(revw_feature*2, d_model))
        MLP_modules_2.append(nn.ReLU())
        MLP_modules_2.append(nn.Dropout(p=dropout))
        for i in range(3):
            MLP_modules_2.append(nn.Linear(d_model, d_model))
            MLP_modules_2.append(nn.ReLU())
            MLP_modules_2.append(nn.Dropout(p=dropout))
        self.MLP_layers_2 = nn.Sequential(*MLP_modules_2)

        self.linear_3 = nn.Linear(self.d_model*2, self.d_model)
        self.pos_embedding=nn.Embedding(self.seq_len, self.d_model)
        self.linear1=nn.Linear(self.d_model, self.d_model)
        self.linear2=nn.Linear(self.d_model, 1)

        self.layer_norm1 = nn.LayerNorm(self.d_model)
        self.layer_norm2 = nn.LayerNorm(self.d_model)

        self.linear_item_1 = nn.Linear(self.d_feature, self.d_model)
        self.linear_item_2 = nn.Linear(self.revw_feature, self.d_model)
        self.linear_user_1 = nn.Linear(self.d_feature, self.d_model)
        self.linear_user_2 = nn.Linear(self.revw_feature, self.d_model)
        self.layer_norm_item_1 = nn.LayerNorm(self.d_model)
        self.layer_norm_item_2 = nn.LayerNorm(self.d_model)
        self.layer_norm_item_3 = nn.LayerNorm(self.d_model)
        self.layer_norm_user_1 = nn.LayerNorm(self.d_model)
        self.layer_norm_user_2 = nn.LayerNorm(self.d_model)

        self._init_weight_()

    def _init_weight_(self):
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)


    def forward(self,user_embeddings,item_embeddings,revw_user_features,revw_item_features,pvs,pos,pos_mode=0,use_mask=False, active_layers=999):
        if pos_mode == 0:  # use fixed position embedding
            pass
            # pos_embedding = Embedding(self.seq_len, self.d_model, trainable=False,\
            #     weights=[GetPosEncodingMatrix(self.seq_len, self.d_model)])
            # p0 = pos_embedding(pos_input)
        elif pos_mode == 1: # use trainable position embedding
            p0 = self.pos_embedding(pos.long())
        else:  # no position embedding
            p0 = None

        d0=torch.cat((item_embeddings,user_embeddings),-1)
        d0=self.MLP_layers_1(d0)
        d1 = torch.cat((revw_user_features,revw_item_features),-1)
        d1=self.MLP_layers_2(d1)
        v_input=torch.cat((d0,d1),-1)
        v_input = self.linear_3(v_input)

        v_input = self.layer_norm1(v_input)
        p0 = self.layer_norm2(p0)
        inputs=v_input+p0

        user_embeddings_ = self.linear_user_1(user_embeddings)
        revw_user_features_ = self.linear_user_2(revw_user_features)

        user_embeddings_= self.layer_norm_user_1(user_embeddings_)
        revw_user_features_ = self.layer_norm_user_2(revw_user_features_)
        user_features = user_embeddings_ + revw_user_features_
		
        item_embeddings_ = self.linear_item_1(item_embeddings)
        revw_item_features_ = self.linear_item_2(revw_item_features)
        item_embeddings_ = self.layer_norm_item_2(item_embeddings_)
        revw_item_features_ = self.layer_norm_item_3(revw_item_features_)
        item_features= item_embeddings_ + revw_item_features_

        inputs_ = user_features * item_features

        sub_mask = None
        if use_mask:
            pass

        enc_output = self.encoder([inputs,inputs_], mask=sub_mask, active_layers=active_layers)
		
        # score prediction
        time_score_dense1=nn.Tanh()(self.linear1(enc_output))
        time_score_dense2=self.linear2(time_score_dense1)
        flat=time_score_dense2.view(-1,self.seq_len)
        score_output=nn.Softmax(dim=-1)(flat)
        return score_output

