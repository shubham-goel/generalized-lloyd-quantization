import pickle
import numpy as np


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
            print("min_cluster_distance({} cl) = {}".format(k, min_cluster_distance))
    assignments = np.array([find(vertice) for vertice in vertices])
    _, assignments = np.unique(assignments, return_inverse=True)
    return assignments

def get_cluster_assignments(A, num_clusters):
    # Cluster columns of A
    A_normal = A/np.linalg.norm(A, axis=0)[None,:]  # Column-normalized
    D = -1* np.abs(A_normal.T @ A_normal)

    # Use kruskal's algorithm to find maximally spaced clusters
    num_vertices = A.shape[1]
    vertices = list(range(num_vertices))
    edges = [(D[i,j],i,j) for i in range(num_vertices) for j in range(num_vertices) if (i<j)]

    return cluster_kruskal(vertices, edges, num_clusters)

if __name__ == "__main__":
    dict_file = 'sc_dictionary_16x16_lamda0point1_Field.p'
    A = pickle.load(open(dict_file, 'rb'))
    assert(isinstance(A, np.ndarray))
    cluster_assignments = get_cluster_assignments(A, int(A.shape[1]/3))
