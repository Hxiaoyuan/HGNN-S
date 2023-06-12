import json
import torch.nn.functional as f
import torch
import random
import numpy as np
import argparse
from itertools import *
import re
import matplotlib.pyplot as plt
from pandas.plotting import radviz

parser = argparse.ArgumentParser(description='application data process')
parser.add_argument('--C_n', type=int, default=4,
                    help='number of node class label')
parser.add_argument('--data_path', type=str, default='../../data/test',
                    help='path to data')
parser.add_argument('--embed_d', type=int, default=128,
                    help='embedding dimension')

args = parser.parse_args()
print(args)


def evaluation(name):
    #
    with open('../../data/test/name_to_pubs_test_100.json', 'r', encoding='utf-8') as f:
        test_dic = json.load(f)
    name_disambiguation_dic = test_dic.get(name, None)
    with open('../../data/test/pubs_raw.json', 'r', encoding='utf-8') as f:
        test_info = json.load(f)
    assert name_disambiguation_dic is not None
    with open('../../data/test/paper_to_index.txt', 'r', encoding='utf-8') as f:
        paper_to_index = json.load(f)

    test_target = []
    test_author = []
    test_cluster_target = {}
    test_pre = []
    test_paper = []
    test_cluster_target = {}
    __author_label = 0
    for key, item in name_disambiguation_dic.items():
        if len(item) < 5:
            continue
        for paper_id in item:
            paper_id = paper_id.split('-')[0]
            if test_info.get(paper_id, '') is not '' and test_info.get(paper_id).get('abstract', '') is not '':
                if test_cluster_target.get(paper_to_index[paper_id], None) is None:
                    test_target.append(__author_label)
                    test_paper.append(paper_to_index[paper_id])
        __author_label += 1

    precision, recall, f1 = cluster_paper(test_paper, test_target, __author_label)

    print(str(__author_label) + "\t" + str(name) + "\t" + str(precision) + "\t" + str(recall) + "\t" + str(f1))


def cluster_paper(paper_indexs, test_target, author_size):
    from sklearn.cluster import KMeans, AgglomerativeClustering
    b_embed = np.around(np.random.normal(0, 0, [len(paper_indexs), args.embed_d]), 4)

    embed_f = open(args.data_path + "/node_embedding.txt", "r")
    for line in islice(embed_f, 0, None):
        line = line.strip()
        node_id = re.split(' ', line)[0]
        if len(node_id) and (node_id[0] in ('a', 'p', 'v')):
            type_label = node_id[0]
            index = int(node_id[1:])
            embed = np.asarray(re.split(' ', line)[1:], dtype='float32')
            if type_label == 'p' and index in paper_indexs:
                b_embed[paper_indexs.index(index)] = embed

    embed_f.close()

    kmeans = AgglomerativeClustering(n_clusters=author_size).fit(b_embed)

    return pairwise_precision_recall_f1(kmeans.labels_, test_target)


pre_list = []
rec_list = []
f1_list = []


def pairwise_precision_recall_f1(preds, truths):
    tp = 0
    fp = 0
    fn = 0
    n_samples = len(preds)
    for i in range(n_samples - 1):
        pred_i = preds[i]
        for j in range(i + 1, n_samples):
            pred_j = preds[j]
            if pred_i == pred_j:
                if truths[i] == truths[j]:
                    tp += 1
                else:
                    fp += 1
            elif truths[i] == truths[j]:
                fn += 1
    tp_plus_fp = tp + fp
    tp_plus_fn = tp + fn
    if tp_plus_fp == 0:
        precision = 0.
    else:
        precision = tp / tp_plus_fp
    if tp_plus_fn == 0:
        recall = 0.
    else:
        recall = tp / tp_plus_fn
    if not precision or not recall:
        f1 = 0.
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    pre_list.append(precision)
    rec_list.append(recall)
    f1_list.append(f1)
    return precision, recall, f1


if __name__ == '__main__':
    execule = ['guotong_du', 'yongjian_tang', 'li_zhu_wu', 'ruijin_liao', 'fuchu_he', 'yongqing_huang', 'shengkai_gong',
               'fosong_wang', 'yin_shi', 'wei_jun_zhang', 'kwok_fai_so']
    with open('../../data/test/name_to_pubs_test_100.json') as f:
        names = json.load(f).keys()
        for index, name in enumerate(names):
            if name in execule:
                continue
            evaluation(name)
    pre_total = 0.0
    rec_total = 0.0
    f1_total = 0.0
    for i in pre_list:
        pre_total += i
    for i in rec_list:
        rec_total += i
    for i in f1_list:
        f1_total += i
    print('pre_mean:{} \n rec_mean:{} \n f1_mean:{}'.format(pre_total / len(pre_list), rec_total / len(rec_list),
                                                            f1_total / len(f1_list)))
