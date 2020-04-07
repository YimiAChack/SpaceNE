#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
import networkx as nx
import itertools
import math
import random
import numpy as np
from joblib import Parallel, delayed
from gensim.models import Word2Vec
from utils import alias_sample, create_alias_table, partition_num


class RandomWalker:
    def __init__(self, G, p=1, q=1):
        self.G = G
        self.p = p
        self.q = q

    def deepwalk_walk(self, walk_length, start_node):
        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def node2vec_walk(self, walk_length, start_node):
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    edge = (prev, cur)
                    next_node = cur_nbrs[alias_sample(alias_edges[edge][0], alias_edges[edge][1])]
                    walk.append(next_node)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):
        G = self.G
        nodes = list(G.nodes())
        res = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length) for num in
            partition_num(num_walks, workers))

        walks = list(itertools.chain(*res))

        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                if self.p == 1 and self.q == 1:
                    walks.append(self.deepwalk_walk(walk_length = walk_length, start_node = v))
                else:
                    walks.append(self.node2vec_walk(walk_length = walk_length, start_node = v))
        return walks

    def get_alias_edge(self, t, v):
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for x in G.neighbors(v):
            weight = G[v][x].get('weight', 1.0)  # w_vx
            if x == t:  # d_tx == 0
                unnormalized_probs.append(weight / p)
            elif G.has_edge(x, t):  # d_tx == 1
                unnormalized_probs.append(weight)
            else:  # d_tx > 1
                unnormalized_probs.append(weight / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return create_alias_table(normalized_probs)

    def preprocess_transition_probs(self):
        G = self.G

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr].get('weight', 1.0) for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = create_alias_table(normalized_probs)

        alias_edges = {}

        for edge in G.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


class Walks:
    def __init__(self, params, workers=1):
        self.graph = nx.read_edgelist(params["base_path"] + params["path_edges"], create_using=nx.Graph())
        self.w2v_model = None
        self._embeddings = {}

        self.walker = RandomWalker(self.graph, p=1, q=1)
        self.sentences = self.walker.simulate_walks(num_walks=params["SpaceNE"]["num_walks"],
                                                    walk_length=params["SpaceNE"]["walk_length"], workers=workers,
                                                    verbose=1)

        self.train(window_size=5, iter=3, embed_size = params["SpaceNE"]["dimension"])

    def train(self, embed_size=64, window_size=5, workers=3, iter=5, **kwargs):
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter

        model = Word2Vec(**kwargs)
        print("Learning init embedding done!")

        self.w2v_model = model
        return model

    def get_embeddings(self, ):
        if self.w2v_model is None:
            print("model not trained!")
            return {}

        embeddings_dict = {}
        for word in self.graph.nodes():
            embeddings_dict[word] = self.w2v_model.wv[word]

        embeddings = []
        for i in range(self.graph.number_of_nodes()):
            embeddings.append(embeddings_dict[str(i)])

        return np.array(embeddings)
