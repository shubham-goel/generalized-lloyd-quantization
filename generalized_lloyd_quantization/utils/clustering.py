import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def cluster_kruskal(vertices, edges, num_clusters):
    assert(len(vertices) >= num_clusters)
    parent = dict()
    rank = dict()

    def make_set(vertice):
        parent[vertice] = vertice
        rank[vertice] = 0

    def find(vertice):
        if parent[vertice] != vertice:
            parent[vertice] = find(parent[vertice])
        return parent[vertice]

    def union(vertice1, vertice2):
        root1 = find(vertice1)
        root2 = find(vertice2)
        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            else:
                parent[root1] = root2
            if rank[root1] == rank[root2]: rank[root2] += 1

    for vertice in vertices:
        make_set(vertice)
    minimum_spanning_tree = set()
    edges = list(edges)
    edges.sort()
    #print edges
    k = len(vertices)
    min_cluster_distance = edges[0][0]
    for edge in edges:
        if(k == num_clusters):
            break
        weight, vertice1, vertice2 = edge
        if find(vertice1) != find(vertice2):
            union(vertice1, vertice2)
            minimum_spanning_tree.add(edge)
            k = k-1
            min_cluster_distance = weight
            # print("min_cluster_distance({} cl) = {}".format(k, min_cluster_distance))
    assignments = np.array([find(vertice) for vertice in vertices])
    _, assignments = np.unique(assignments, return_inverse=True)
    return assignments

def get_clusters(A, num_clusters, algo='stoer_wagner'):
    # Cluster columns of A
    if num_clusters == A.shape[1]:
        return [[i] for i in range(num_clusters)]
    if algo == 'kruskal':
        # Use kruskal's algorithm to find maximally spaced clusters
        D = -1* np.abs(A.T @ A)
        num_vertices = A.shape[1]
        vertices = list(range(num_vertices))
        edges = [(D[i,j],i,j) for i in range(num_vertices) for j in range(num_vertices) if (i<j)]
        cluster_assignments = cluster_kruskal(vertices, edges, num_clusters)
        clusters = [[] for c in range(num_clusters)]
        for p in range(A.shape[1]):
            clusters[cluster_assignments[p]].append(p)
        return clusters
    elif algo == 'stoer_wagner':
        D = np.abs(A.T @ A)
        G = nx.from_numpy_matrix(D)
        assert(num_clusters < G.order())

        total_cut_val = 0
        cut_val, partition = nx.stoer_wagner(G)
        subgraphs = {G: (cut_val, partition)}
        while num_clusters > len(subgraphs):
            last_iteration = (num_clusters==(len(subgraphs)+1))
            # Pick smallest cut sg; divide
            min_g = None
            min_val = float('inf')
            for g in subgraphs:
                if subgraphs[g][0] < min_val:
                    min_val = subgraphs[g][0]
                    min_g = g
            # print("New stoer_wagner cut of value={}".format(min_val))
            total_cut_val += min_val
            v1,v2 = subgraphs[min_g][1]
            g1 = G.subgraph(v1)
            g2 = G.subgraph(v2)
            del subgraphs[min_g]
            subgraphs[g1] = nx.stoer_wagner(g1) if (g1.order()>1) and (not last_iteration) else (float('inf'), None)
            subgraphs[g2] = nx.stoer_wagner(g2) if (g2.order()>1) and (not last_iteration) else (float('inf'), None)

        print("Total stoer_wagner cut_value={}".format(total_cut_val))
        clusters = [list(g) for g in subgraphs]
        return clusters
    else:
        raise ValueError("Unrecognized algorith: {}".format(algo))

if __name__ == "__main__":
    dict_file = '../../../data/sc_dictionary_8x8_lamda0point1_Field.p'
    A = pickle.load(open(dict_file, 'rb'))
    assert(isinstance(A, np.ndarray))
    clusters = get_clusters(A, int(A.shape[1]/3))
