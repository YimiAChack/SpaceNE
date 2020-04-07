#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pickle
import numpy as np
import networkx as nx
import os
import re
import json
import hierarch_network as hn



def save_resX_txt(reconstruction_X, path):
    arr = np.array([reconstruction_X[0]])
    for i in range(1, len(reconstruction_X.keys())):
        arr = np.row_stack((arr, reconstruction_X[i]))
    np.savetxt(path, arr)


def load_tree(file_path):
    G = nx.DiGraph()
    n, m = None, None  # node num, edge num +1
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip().strip('\n')
            if len(line) == 0:
                continue
            items = line.split('\t')
            if len(items) != 2:
                continue
            if n is None:
                n, m = int(items[0]), int(items[1])
            else:
                G.add_edge(int(items[0]), int(items[1]))

    return G, n, m


def embed_array_to_dict(comm, projection):
    embed = dict()
    for (index, i) in enumerate(comm):
        embed[i] = projection[index]
    return embed


def load_json_file(file_path):
    with open(file_path, "r") as f:
        s = f.read()
        s = re.sub('\s', "", s)
    return json.loads(s)


def save_res(basevectors, node_embedding, tree, res_X, params):
    out1 = open(params["base_path"] + params["model_save"]["basevectors"], 'wb')
    pickle.dump(basevectors, out1)
    out1.close()
    out2 = open(params["base_path"] + params["model_save"]["multilayer_embeddings"], 'wb')
    pickle.dump(node_embedding, out2)
    out2.close()
    out3 = open(params["base_path"] + params["model_save"]["tree"], 'wb')
    pickle.dump(tree, out3)
    out3.close()

    arr = np.array([res_X[0]])
    for i in range(1, len(res_X.keys())):
        arr = np.row_stack((arr, res_X[i]))
    np.save(params["base_path"] + params["model_save"]["node_embeddings"], arr)
    print("save node embeddngs at ", params["base_path"] + params["model_save"]["node_embeddings"])


def reconstruction_X_top(basevectors, node_embedding, tree, leafnode_num):
    X_res = dict()
    for i in range(leafnode_num):
        com_ids = hn.node_comm_id(i, tree)
        com_ids_copy = com_ids[:-1]  # last one is root, no basevector

        com_ids.reverse()
        # print(com_ids_copy)
        for (index, j) in enumerate(com_ids):
            if index == 0:
                resconstruct_embed = np.array(node_embedding[j][i])
            else:
                resconstruct_embed = np.dot(basevectors[j].transpose(), resconstruct_embed.transpose())
        for (index, j) in enumerate(com_ids_copy):
            resconstruct_embed = np.dot(basevectors[j], resconstruct_embed)

        X_res[i] = resconstruct_embed
    return X_res


def X_comm_to_arr(X):
    out = list()
    for i in X:
        i = i.tolist()
        for j in i:
            out.append(j)
    c = np.array(out)
    return np.array(out)


def create_alias_table(area_ratio):
    """
    :param area_ratio: sum(area_ratio)=1
    :return: accept,alias
    """
    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []
    area_ratio_ = np.array(area_ratio) * l
    for i, prob in enumerate(area_ratio_):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = area_ratio_[small_idx]
        alias[small_idx] = large_idx
        area_ratio_[large_idx] = area_ratio_[large_idx] - \
                                 (1 - area_ratio_[small_idx])
        if area_ratio_[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1

    return accept, alias


def alias_sample(accept, alias):
    """
    :param accept:
    :param alias:
    :return: sample index
    """
    N = len(accept)
    i = int(np.random.random() * N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]


def preprocess_nxgraph(graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx


def partition_dict(vertices, workers):
    batch_size = (len(vertices) - 1) // workers + 1
    part_list = []
    part = []
    count = 0
    for v1, nbs in vertices.items():
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list


def partition_list(vertices, workers):
    batch_size = (len(vertices) - 1) // workers + 1
    part_list = []
    part = []
    count = 0
    for v1, nbs in enumerate(vertices):
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list


def partition_num(num, workers):
    if num % workers == 0:
        return [num // workers] * workers
    else:
        return [num // workers] * workers + [num % workers]

        # load_data_georgetown()
        # load_y_georgetown()
