import numpy as np
import time
import json
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import dataset.utilities
import dataset.myDataset
import model.DTE
import model.IDM
import metric

parser = argparse.ArgumentParser()
#parser.add_argument("--train_set",type=str, default="",help="the file path of train set")
#parser.add_argument("--validation_set", type=str,default="",help= "the file path of validation set")
#parser.add_argument("--test_set", type=str,default="", help="the file path of test set")
#parser.add_argument("--log_dir", type=str,default="./log/", help="the log directory")
#parser.add_argument("--saved_model_name",type=str,default= "DTE.pth", help="the saved model name")
#parser.add_argument("--model_type", type=int,default=0, help="drr model type, 0:drr_base 1:drr_personalized_v1 2:drr_personalized_v2")
parser.add_argument("--batch_size",type=int,default=256, help="batch size for training")
parser.add_argument("--seq_len", type=int,default=50, help="the length of input list")#"n" in the paper
parser.add_argument("--train_epochs", type=int,default=100, help="epoch for training")
#parser.add_argument("--train_steps_per_epoch", type=int,default=1000, help="steps per epoch for training")
#parser.add_argument("--validation_steps", type=int,default=2000, help="steps for validation")
#parser.add_argument("--early_stop_patience", type=int,default=10, help="early stop when model is not improved	 with X epochs")
#parser.add_argument("--lr_per_step", type=int,default=4000, help="update learning rate per X step")
parser.add_argument("--d_feature", type=int ,default=32, help="the feature length of each item in the input list")
parser.add_argument("--d_model", type=int,default=64, help="param used in DTE")
parser.add_argument("--d_inner_hid", type=int, default=128, help="param used in DTE")
parser.add_argument("--n_head", type=int,default=4, help="param used in DTE")
parser.add_argument("--d_k", type=int, default=64, help="param used in DTE")
parser.add_argument("--d_v", type=int, default=64, help="param used in DTE")
parser.add_argument("--num_layers", type=int,default=2, help="param used in DTE")#"b" in the paper
parser.add_argument("--drr_dropout", type=float,default=0.1, help="param used in DTE")
parser.add_argument("--pos_embedding_mode", type=int,default=1, help="param used in DTE: 0:use fix PE  1:use learnable PE  2:unuse PE")
parser.add_argument("--num_experts", type=int, default=4, help="param used in DTE")#number of experts: "t" in the paper
parser.add_argument("--gpu", type=str,default="0",  help="gpu card ID")
parser.add_argument("--test",action='store_true',default=False,help="train or not")
parser.add_argument("--dataset",type=str,default="A2_Sports_and_Outdoors_5",help="dataset")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
#parser.add_argument("--dmax", type=int, default=20, help="Max Number of documents (or reviews)")
parser.add_argument("--emb_size", type=int,default=50, help="Embeddings dimension (default=50)")
parser.add_argument("--revw_dropout",type=float,default=0.8, help="The dropout probability.")
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
cudnn.benchmark = True

def train():
    ############################## CONSTRUCT DATASET ####################################################
    user_embeddings,item_embeddings,pvs,labels,text_users,text_items,train_num,test_num,val_num = dataset.utilities.load_all(opt.dataset,opt.seq_len)

    train_user_embeddings=[user_embeddings[i] for i in train_num]
    train_item_embeddings=[item_embeddings[i] for i in train_num]
    train_pvs = [pvs[i]for i in train_num]
    train_labels=[labels[i]for i in train_num]
    train_text_users=[text_users[i]for i in train_num]
    train_text_items=[text_items[i]for i in train_num]

    val_user_embeddings = [user_embeddings[i] for i in val_num]
    val_item_embeddings = [item_embeddings[i] for i in val_num]
    val_pvs = [pvs[i] for i in val_num]
    val_labels = [labels[i] for i in val_num]
    val_text_users = [text_users[i] for i in val_num]
    val_text_items = [text_items[i] for i in val_num]

    # construct the train and test datasets
    train_dataset = dataset.myDataset.Dataset(
        opt.seq_len, train_user_embeddings, train_item_embeddings, train_pvs, train_labels, train_text_users,
        train_text_items)
    val_dataset = dataset.myDataset.Dataset(
        opt.seq_len, val_user_embeddings, val_item_embeddings, val_pvs, val_labels, val_text_users, val_text_items)
    train_loader = data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8)
    val_loader = data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    print('data loading finish,model time begins...')
	
    ########################### CREATE MODEL #################################

    revw_model=model.IDM.Co_attention(768,opt.emb_size,opt.revw_dropout,False)
    revw_model.cuda()
    revw_optimizer=optim.Adam(revw_model.parameters(),lr=opt.lr)

    DTE = model.DTE.Model(opt.seq_len, opt.emb_size,opt.d_feature, d_model=opt.d_model, d_inner_hid=opt.d_inner_hid, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v, num_experts = opt.num_experts, layers=opt.num_layers, dropout=opt.drr_dropout)
    DTE.cuda()
    drr_optimizer = optim.Adam(DTE.parameters(), lr=opt.lr)


    print(revw_model)
    print(DTE)
    print('model created finish,training time begins...')
    running_time_start= time.time()
	
    ########################### TRAINING #####################################
    best_p5 = 0
    for epoch in range(opt.train_epochs):
        revw_model.train()
        DTE.train()
        start_time = time.time()
        count=0
        loss_sum=0
        for user_embeddings,item_embeddings,pvs,pos,labels,text_users,text_items in train_loader:

            user_embeddings = user_embeddings.cuda()
            item_embeddings = item_embeddings.cuda()
            pvs=pvs.cuda()
            pos=pos.cuda()
            labels = labels.cuda()
            text_users=text_users.cuda()
            text_items=text_items.cuda()

            revw_model.zero_grad()
            DTE.zero_grad()

            revw_user_features,revw_item_features,_,_=revw_model(text_users,text_items)
            predictions=DTE(user_embeddings,item_embeddings,
                                  revw_user_features,revw_item_features,pvs,pos,pos_mode=1)

            loss = -(torch.sum(torch.log(predictions) * labels))
            loss.backward()

            revw_optimizer.step()
            drr_optimizer.step()

            loss_sum+=loss.item()
            count+=1

        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
              time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("LOSS: {:.3f}".format(loss_sum / count))

        ########### EVALUATION ON VALIDATION SET ###########
        revw_model.eval()
        DTE.eval()

        initial_labels_list = []
        labels_list = []
        with torch.no_grad():
            for user_embeddings, item_embeddings, pvs, pos, labels, text_users, text_items in val_loader:

                user_embeddings = user_embeddings.cuda()
                item_embeddings = item_embeddings.cuda()
                pvs = pvs.cuda()
                pos = pos.cuda()
                labels = labels.cuda()
                text_users = text_users.cuda()
                text_items = text_items.cuda()

                revw_user_features, revw_item_features, _, _ = revw_model(text_users, text_items)
                predictions = DTE(user_embeddings, item_embeddings,
                                    revw_user_features, revw_item_features, pvs, pos, pos_mode=1)

                for i in range(predictions.size()[0]):
                    rankSeq = np.argsort(-predictions[i].cpu().numpy())
                    initial_labels_list.append(labels[i].cpu().numpy().tolist())
                    labels_list.append(labels[i].cpu().numpy()[rankSeq].tolist())

        p5 = metric.evaluate(labels_list, False)
        print("%s=%0.2f" % ("p@5", p5))
        if p5 > best_p5:  # the best model evaluated on the validation set will be saved
            best_p5 = p5
            best_epoch = epoch
            if not os.path.exists('./models/{}'.format(opt.dataset)):
                os.mkdir('./models/{}'.format(opt.dataset))
            torch.save(revw_model, './models/{}/revw_model.pth'.format(opt.dataset))
            torch.save(DTE, './models/{}/DTE.pth'.format(opt.dataset))
        print("End. Best epoch {:03d}: p@5 = {:.3f}".format(
            best_epoch, best_p5))
            
    running_time_end = time.time()
    print("Total Training time= " +  time.strftime("%H: %M: %S", time.gmtime(running_time_end - running_time_start)))  

def test():
    user_embeddings, item_embeddings, pvs, labels, text_users, text_items, train_num, test_num,val_num = dataset.utilities.load_all(
        opt.dataset,opt.seq_len)

    test_user_embeddings = [user_embeddings[i] for i in test_num]
    test_item_embeddings = [item_embeddings[i] for i in test_num]
    test_pvs = [pvs[i] for i in test_num]
    test_labels = [labels[i] for i in test_num]
    test_text_users = [text_users[i] for i in test_num]
    test_text_items = [text_items[i] for i in test_num]

    test_dataset = dataset.myDataset.Dataset(
        opt.seq_len, test_user_embeddings, test_item_embeddings, test_pvs, test_labels, test_text_users,
        test_text_items)

    test_loader = data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    ########################### CREATE MODEL #################################

    revw_model = model.IDM.Co_attention(768, opt.emb_size, opt.revw_dropout,
                                                 False)
    revw_model.cuda()

    DTE = model.DTE.Model(opt.seq_len, opt.emb_size, opt.d_feature, d_model=opt.d_model,
                                         d_inner_hid=opt.d_inner_hid, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v,
                                         layers=opt.n_layers, dropout=opt.drr_dropout)
    DTE.cuda()


    revw_model = torch.load('./models/{}/revw_model.pth'.format(opt.dataset))
    DTE = torch.load('./models/{}/DTE.pth'.format(opt.dataset))

    revw_model.eval()
    DTE.eval()
    initial_labels_list = []
    labels_list = []

    for user_embeddings, item_embeddings, pvs, pos, labels, text_users, text_items in test_loader:

        user_embeddings = user_embeddings.cuda()
        item_embeddings = item_embeddings.cuda()
        pvs = pvs.cuda()
        pos = pos.cuda()
        labels = labels.cuda()
        text_users = text_users.cuda()
        text_items = text_items.cuda()

        revw_user_features, revw_item_features, _, _ = revw_model(text_users, text_items)
        predictions = DTE(user_embeddings, item_embeddings,
                                revw_user_features, revw_item_features, pvs, pos, pos_mode=1)

        for i in range(predictions.size()[0]):
            rankSeq=np.argsort(-predictions[i].cpu().detach().numpy())
            initial_labels_list.append(labels[i].cpu().detach().numpy().tolist())
            labels_list.append(labels[i].cpu().detach().numpy()[rankSeq].tolist())


    metric.evaluate(initial_labels_list,True)
    print('Evaluation for testset begins...')
    metric.evaluate(labels_list,True)

if __name__=='__main__':
    if opt.test:
        test()
    else:
        train()
