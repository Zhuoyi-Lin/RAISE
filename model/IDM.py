import torch
import torch.nn as nn
class Co_attention(nn.Module):


    def __init__(self,emb_size,output_size,dropout=None,gumbel=False):
        super(Co_attention, self).__init__()
        self.dropout=dropout
        self.gumbel=gumbel
        self.linear_a=nn.Linear(emb_size,output_size)
        self.linear_b= nn.Linear(emb_size, output_size)
        self.W=nn.Linear(output_size,output_size)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    def forward(self,input_a,input_b):
        # print(input_a)
        orig_a = input_a
        orig_b = input_b
        seq_len=orig_a.size()[1]

        input_a=input_a.view(-1,input_a.size()[2],input_a.size()[3])
        input_b = input_b.view(-1, input_b.size()[2], input_b.size()[3])

        a_len = input_a.size()[1]
        b_len = input_b.size()[1]
        input_dim = input_a.size()[2]
        max_len = a_len

        # transform_layers==1
        input_a=self.linear_a(input_a)
        input_a = nn.ReLU()(input_a)
        input_b = self.linear_b(input_b)
        input_b = nn.ReLU()(input_b)

        dim = input_a.size()[2]

        # elif(att_type=='SOFT'):
        # Soft match without parameters
        # _b = tf.transpose(input_b, [0,2,1])
        _b = input_b.permute(0, 2, 1)
        zz= self.W(input_a)
        z = torch.matmul(zz, _b)  # bsz * num_vec_a *num_vec_b
        #z=torch.matmul(input_a,_b)
        y = z

        # if(pooling=='MAX'):
        att_row = torch.mean(y, 1)  # bsz * num_vec_b
        att_col = torch.mean(y, 2)  # bsz * num_vec_a

        att_row=nn.Sigmoid()(att_row)
        att_col=nn.Sigmoid()(att_col)

        # Get attention weights
        if (self.gumbel):
            pass
            # att_row = gumbel_softmax(att_row, temp, hard=hard)
            # att_col = gumbel_softmax(att_col, temp, hard=hard)
        else:
            att_row = nn.Softmax(dim=-1)(att_row)
            att_col = nn.Softmax(dim=-1)(att_col)
        _a2 = att_row
        _a1 = att_col

        att_col = torch.unsqueeze(att_col, 2)
        att_row = torch.unsqueeze(att_row, 2)

        # Weighted Representations
        final_a = att_col * input_a
        final_b = att_row * input_b

        final_a = nn.Dropout(self.dropout)(final_a)
        final_b = nn.Dropout(self.dropout)(final_b)

        final_a = torch.sum(final_a, 1)
        final_b = torch.sum(final_b, 1)

        final_a=final_a.view(-1,seq_len,dim)
        final_b=final_b.view(-1,seq_len,dim)

        return final_a, final_b, _a1, _a2
