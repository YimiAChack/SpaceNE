#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import numpy as np
import math
import pickle


def node_comm_id(node, tree):
    comm_label = list()
    while len(tree[node].coverst) > 0:
        label = tree[node].coverst[0]
        comm_label.append(label)
        node = label
    return comm_label


def build_hierach_network(tree, leafnode_num):
    n = len(tree)
    leafnode_com_ids = dict()
    for i in range(leafnode_num):
        leafnode_com_ids[i] = node_comm_id(i, tree)

    hi_network = dict()
    for i in range(leafnode_num - 1, n):
        for j in leafnode_com_ids.keys():
            if i in leafnode_com_ids[j]:
                if i in hi_network.keys():
                    hi_network[i].append(j)
                else:
                    hi_network[i] = [j]
    # print(hi_network)
    return hi_network


def pkl_line_to_spaceNE(in_path):
    pkl_file = open(in_path, 'rb')
    node_embedding = pickle.load(pkl_file)['embeddings']
    arr_len = len(node_embedding)
    embed_dict = dict()
    for i in range(arr_len):
        embed_dict[i] = node_embedding[i]
    return embed_dict


def pkl_line_to_spaceNE_2(init_embeddings):
    embed_dict = dict()
    for i in range(len(init_embeddings)):
        embed_dict[i] = init_embeddings[i]
    return embed_dict


def build_X(root, tree, node_embedding, hi_network, init_embeddings):
    X = list()

    if len(node_embedding.keys()) == 0:
        node_embedding[root] = pkl_line_to_spaceNE_2(init_embeddings)
        for i in tree[root].childst:
            com_embed = list()
            for j in hi_network[i]:
                com_embed.append(node_embedding[root][j])
            X.append(np.array(com_embed).astype('float32'))
    else:
        for i in tree[root].childst:
            com_embed = list()
            for j in hi_network[i]:
                com_embed.append(node_embedding[root][j])

            X.append(np.array(com_embed))

    return X, node_embedding
