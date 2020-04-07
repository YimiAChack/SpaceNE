#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os
import networkx as nx
from utils import *


class Node:
    def __init__(self, id, childst, coverst):
        self.id = id
        self.childst = childst
        self.coverst = coverst

    def __str__(self):
        line = "id: %s\n" % self.id
        line += "childst: %s\n" % self.childst
        line += "coverst: %s\n" % self.coverst
        return line


def dfs(u, tree):
    if len(tree[u].childst) == 0:
        # tree[u].coverst = set([u])
        return
    for v in tree[u].childst:
        dfs(v, tree)
        tree[v].coverst = [u]
        # tree[u].coverst = tree[u].coverst | tree[v].coverst


def extract_hierarchy(path_data):
    g, n, m = load_tree(path_data)

    tree = [None] * n  # id start from 0
    for u in g:
        childst = [int(m) for m in list(set(g[u].keys()))]
        cur_u = int(u)
        tree[cur_u] = Node(cur_u, childst, [])

    dfs(n - 1, tree)

    leafnode_num = 0

    for i in range(len(tree)):
        if len(tree[i].childst) == 0:
            leafnode_num += 1

    return tree, n, leafnode_num
