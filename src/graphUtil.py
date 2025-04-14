import copy

import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt

def generate_graph(n):
    indexMax = 5*n
    positions = np.random.uniform(0, indexMax, size=(n, 2))
    positions_dict = {i : tuple(positions[i-1]) for i in range(1, n+1)}

    names = list(range(1, n+1))

    G = nx.complete_graph(names)
    for u, v in G.edges():
        x1, y1 = positions_dict[u]
        x2, y2 = positions_dict[v]
        distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        G[u][v]['weight'] = np.floor(distance) if distance > 0 else 1
    return G

def satisfies_triangle_inequality(G):
    for u in G.nodes():
        for v in G.neighbors(u):
            for w in G.neighbors(v):
                if w == u or not G.has_edge(u, w):
                    continue
                uw = G[u][w]['weight']
                vw = G[v][w]['weight']
                uv = G[u][v]['weight']
                if uw > uv + vw:
                    print("triangle inequality is not satisfied")
                    return False
    return True

from itertools import combinations
def satisfies_triangle_inequality_2(G):
    for u, v, w in combinations(G.nodes, 3):  # Get all triplets of nodes
        d_uv = G[u][v]['weight']
        d_vw = G[v][w]['weight']
        d_uw = G[u][w]['weight']

        # Triangle inequality conditions
        if not (d_uv + d_vw >= d_uw and
                d_uv + d_uw >= d_vw and
                d_vw + d_uw >= d_uv):
            return False  # If any condition fails, return False
    return True

def select_blocked_edges(G, k):
    if not nx.is_connected(G):
        raise ValueError('Graph is not connected')

    G_copy = copy.deepcopy(G)

    blocked_edges = []
    edges = list(G.edges)
    random.shuffle(edges)

    count = 0

    for u, v in edges:
        G_copy.remove_edge(u, v)
        if nx.is_connected(G_copy):
            blocked_edges.append((u, v))
            count += 1
            if count == k:
                break
        else:
            G_copy.add_edge(u, v)

    if count < k:
        raise ValueError('Not enough edges')

    return blocked_edges

def apply_blockages(G, blocked_edges):
    for u, v in G.edges:
        if (u, v) in blocked_edges:
            G[u][v]['blocked']=True
        else:
            G[u][v]['blocked']=False
    return

def generate_blocked_graph(n, k):
    graph = generate_graph(n)
    blocked_edges = select_blocked_edges(graph, k)
    apply_blockages(graph, blocked_edges)

    return graph


def drawGraph(G):
    pos = nx.spring_layout(G, seed=42)

    blocked_edges = []
    unblocked_edges = []
    for u, v, attr in G.edges(data=True):
        if attr.get('blocked'):
            blocked_edges.append((u, v))
        else:
            unblocked_edges.append((u, v))

    nx.draw_networkx_nodes(G, pos, node_color='black')
    nx.draw_networkx_edges(G, pos, edgelist=blocked_edges, edge_color='red', width=4, style='dashed')
    nx.draw_networkx_edges(G, pos, edgelist=unblocked_edges, edge_color='gray', width=2)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
    plt.title('Graph')
    plt.show()


def get_total_weight(graph, path):
    sum_weight = 0
    it = iter(path)
    try:
        end = next(it)
    except StopIteration:
        return 0

    for vertex in it:
        start = end
        end = vertex
        sum_weight += graph[start][end]['weight']

    return sum_weight


def get_unblocked_subgraph(graph):
    graph_copy = graph.copy()
    blocked_edges = [
        (u, v) for u, v, attr in graph_copy.edges(data=True)
        if attr.get('blocked') is True
    ]
    graph_copy.remove_edges_from(blocked_edges)
    return graph_copy



if __name__ == '__main__':
    n = random.randint(100, 200)
    G = generate_graph(5)
    print(satisfies_triangle_inequality_2(G))

    blocked_edges_for_instance = select_blocked_edges(G, 4)
    apply_blockages(G, blocked_edges_for_instance)
    drawGraph(G)
    print(len(G.edges))
    Gprime = get_unblocked_subgraph(G)
    print(len(Gprime.edges))
    drawGraph(Gprime)