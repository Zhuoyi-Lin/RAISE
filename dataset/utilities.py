import json
import gzip
import scipy.sparse as sp
import numpy as np
import csv

def dictToFile(dict,path):
    print("Writing to {}".format(path))
    with gzip.open(path, 'w') as f:
        f.write(json.dumps(dict))

def dictFromFileUnicode(path):
    print("Loading {}".format(path))
    with gzip.open(path, 'r') as f:
        return json.loads(f.read())

def load_set(fp):
    data= []
    with open(fp,'r') as f:
      reader = csv.reader(f, delimiter='\t')
      for r in reader:
        data.append(r)
    return data

def load_all(dataset,seq_len):

    data_path3 = '../NCF/outputs/{}/env_rsplit.gz'.format(dataset)
    env3 = dictFromFileUnicode(data_path3)
    train_num=env3['train_num']
    test_num=env3['test_num']
    val_num=env3['val_num']

    data_path1 = '../NCF/outputs/{}/env_{}.gz'.format(dataset,seq_len)
    env1 = dictFromFileUnicode(data_path1)
    initail_lists = env1['initail_lists']
    embed_users=env1['embed_users']
    embed_items=env1['embed_items']
    pvs=env1['pvs']

    item_embeddings=[]
    user_embeddings = []
    for j in range(len(initail_lists)):
        initail_list=initail_lists[j]
        user_embedding=[]
        item_embedding = []
        for i in initail_list:
            user_embedding.append(embed_users[j])
            item_embedding.append(embed_items[i])
        item_embeddings.append(item_embedding)
        user_embeddings.append(user_embedding)

    data_path2 = '../datasets/{}/env.gz'.format(dataset)
    env2 = dictFromFileUnicode(data_path2)

    print('embedding start...')

    user_text =load_set('../datasets/{}/user_text.txt'.format(dataset))
    item_text =load_set('../datasets/{}/item_text.txt'.format(dataset))
    for i in range(len(user_text)):
        user_text[i] = [eval(j) for j in user_text[i]]
    for i in range(len(item_text)):
        item_text[i] = [eval(j) for j in item_text[i]]
    text_users = []
    text_items = []
    for i in range(len(initail_lists)):
        initail_list = initail_lists[i]
        text_user = []
        text_item = []
        for j in initail_list:
            text_user.append(user_text[i])
            text_item.append(item_text[j])
        text_users.append(text_user)
        text_items.append(text_item)

    print('embedding finish.')

    labels=[]
    interactions=env2['train']+env2['dev']+env2['test']
    interaction_mat = sp.dok_matrix((len(user_text), len(item_text)), dtype=np.float32)
    for x in interactions:
        interaction_mat[x[0], x[1]] = 1.0
    for i in range(len(initail_lists)):
        label=[]
        initail_list=initail_lists[i]
        for j in initail_list:
            if (i,j) in interaction_mat:
                label.append(1)
            else:
                label.append(0)
        labels.append(label)

    return user_embeddings,item_embeddings,pvs,labels,text_users,text_items,train_num,test_num,val_num

