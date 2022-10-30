import torch.utils.data as data
import torch
class Dataset(data.Dataset):
    def __init__(self,seq_len,user_embeddings,item_embeddings,pvs,labels,text_users,text_items):
        super(Dataset, self).__init__()
        self.user_embeddings = user_embeddings
        self.item_embeddings=item_embeddings
        self.text_users=text_users
        self.text_items=text_items
        self.pvs=pvs
        self.labels=labels
        self.pos=list(range(seq_len))

    def __len__(self):
        return len(self.item_embeddings) #intialList的数目

    def __getitem__(self, idx):
        return torch.Tensor(self.user_embeddings[idx]),\
               torch.Tensor(self.item_embeddings[idx]), \
               torch.Tensor(self.pvs[idx]), \
               torch.Tensor(self.pos),\
               torch.Tensor(self.labels[idx]),\
               torch.Tensor(self.text_users[idx]),\
               torch.Tensor(self.text_items[idx])
