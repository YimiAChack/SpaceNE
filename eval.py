import networkx as nx
import numpy as np
import random
from sklearn import metrics

embedding_file = "./data/output/georgetown/reconstruction_X.pkl"

print(embedding_file)

edge_file = "./data/input/georgetown/edges_georgetown.txt"

# ratio = 0.7
ratio = 0.3


def sample_edges(G, ratio):
    # sample positive and neg edges
    pos_edges, neg_edges = [], []

    num_nodes = G.number_of_nodes()

    for _ in range(int(ratio * num_nodes)):
        u, v = random.choice(list(G.edges()))
        G.remove_edge(u, v)
        pos_edges.append((u, v))

    cnt = 0
    edge_list = list(G.edges())
    while cnt <= ratio * num_nodes:
        u, v = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
        if (u, v) in edge_list or (v, u) in edge_list or (u, v) in pos_edges or (v, u) in pos_edges or (
        u, v) in neg_edges or (v, u) in neg_edges:
            continue
        neg_edges.append((u, v))
        cnt += 1

    return pos_edges, neg_edges


def link_prediction(edge_file, ratio, embedding_file):
    G = nx.read_edgelist(edge_file, create_using=nx.Graph())
    pos_edges, neg_edges = sample_edges(G, ratio)

    # embeddings = np.loadtxt(embedding_file)
    embeddings = np.load(embedding_file)
    num_pos = len(pos_edges)

    scores = []
    for u, v in pos_edges:
        score = np.dot(embeddings[int(u)], embeddings[int(v)]) / (
        np.linalg.norm(embeddings[int(u)]) * np.linalg.norm(embeddings[int(v)]))
        # score = np.linalg.norm(embeddings[u] - embeddings[v])
        scores.append(score)
    for u, v in neg_edges:
        score = np.dot(embeddings[int(u)], embeddings[int(v)]) / (
        np.linalg.norm(embeddings[int(u)]) * np.linalg.norm(embeddings[int(v)]))
        # score = np.linalg.norm(embeddings[u] - embeddings[v])
        scores.append(score)

    argsorted = np.argsort(scores)

    num = len(argsorted)

    label = []
    for _ in range(num // 2):
        label.append(1)
    for _ in range(num - num // 2):
        label.append(0)
    label = np.array(label)

    pred = np.zeros(num)
    for i in range(num // 2 - 1, num):
        pred[argsorted[i]] = 1

    recall = metrics.recall_score(label, pred)
    fpr, tpr, thresholds = metrics.roc_curve(label, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print(auc, recall)

    return auc, recall


aucs, recalls = [], []
for _ in range(5):
    auc, recall = link_prediction(edge_file, ratio, embedding_file)
    aucs.append(auc)
    recalls.append(recall)

print("avg auc: ", round(np.mean(aucs) * 100, 2))
print("avg recall: ", round(np.mean(recalls) * 100, 2))
