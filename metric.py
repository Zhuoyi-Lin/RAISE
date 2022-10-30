import math
import numpy as np
def calc_precision_at_k(r, k):
    k = min(len(r), k)
    r = np.asarray(r)[:k] != 0
    return np.mean(r)

def calc_average_precision_at_k(labels, k):
    n = min(len(labels),k)
    labels = labels[:n]
    p = []
    p_cnt = 0
    for i in range(n):
        if labels[i]>0:
            p_cnt+=1
            p.append(p_cnt*1.0/(i+1))
    if p_cnt > 0:
        return sum(p)/p_cnt
    else:
        return 0.0

def calc_hr_at_k(labels, k):
    n = min(len(labels),k)
    labels = labels[:n]
    p_cnt = 0
    for i in range(n):
        if labels[i]>0:
            p_cnt+=1
    if sum(labels) > 0:
        return p_cnt*1.0/sum(labels)
    else:
        return 0.0

def calc_ndcg_at_k(labels,k):
    dcg=0.0
    for i in range(k):
        dcg+=labels[i]/math.log(i+2,2)
    num=min(int(sum(labels)),k)
    idcg = 0.0
    for i in range(num):
        idcg += 1 / math.log(i + 2, 2)
    if idcg>0:
        return dcg/idcg
    else:
        return 0.0


def make_metric_dict():
    return { 'p@1':0,'p@5':0, 'p@10':0,'p@20':0,'p@30':0 ,
             'map@1':0,'map@5':0, 'map@10':0,'map@20':0,'map@30':0 ,
             'hr@1':0,'hr@5':0,'hr@10':0,'hr@20':0,'hr@30':0,
             'ndcg@1':0,'ndcg@5': 0, 'ndcg@10': 0, 'ndcg@20': 0, 'ndcg@30': 0}

def evaluate(labels_list,out):
    metric_keys = [ 'p@1','p@5','p@10','p@20','p@30',
                    'map@1','map@5', 'map@10', 'map@20','map@30',
                    'hr@1','hr@5','hr@10','hr@20','hr@30',
                    'ndcg@1','ndcg@5','ndcg@10','ndcg@20','ndcg@30']
    cnt = 0
    d = make_metric_dict()
    for labels in labels_list:
        d['p@1'] += calc_precision_at_k(labels, 1)
        d['p@5'] += calc_precision_at_k(labels, 5)
        d['p@10'] += calc_precision_at_k(labels, 10)
#        d['p@20'] += calc_precision_at_k(labels, 20)
#        d['p@30'] += calc_precision_at_k(labels, 30)
        d['map@1'] += calc_average_precision_at_k(labels, 1)
        d['map@5'] += calc_average_precision_at_k(labels, 5)
        d['map@10'] += calc_average_precision_at_k(labels, 10)
#        d['map@20'] += calc_average_precision_at_k(labels, 20)
#        d['map@30'] += calc_average_precision_at_k(labels, 30)
        d['hr@1'] += calc_hr_at_k(labels, 1)
        d['hr@5'] += calc_hr_at_k(labels, 5)
        d['hr@10'] += calc_hr_at_k(labels, 10)
#        d['hr@20'] += calc_hr_at_k(labels, 20)
#        d['hr@30'] += calc_hr_at_k(labels, 30)
        d['ndcg@1'] += calc_ndcg_at_k(labels, 1)
        d['ndcg@5'] += calc_ndcg_at_k(labels, 5)
        d['ndcg@10'] += calc_ndcg_at_k(labels, 10)
#        d['ndcg@20'] += calc_ndcg_at_k(labels, 20)
#        d['ndcg@30'] += calc_ndcg_at_k(labels, 30)
        cnt+=1
    if out:
        for key in metric_keys:
            print("%s=%0.2f" % (key, d[key] * 100.0/cnt))
            #print("%0.2f" % (d[key] * 100.0 / cnt),end=',')
        #print('\n')
    return d['p@5'] * 100.0 / cnt
