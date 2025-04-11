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
        G[u][v]['weight'] = distance if distance > 0 else 1
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

def generate_graph_blocked(n, k):
    graph = generate_graph(n)
    blocked_edges = select_blocked_edges(graph, k)
    apply_blockages(graph, blocked_edges)

    print(f"Create a complete graph with {n} nodes and {k} blocked edges")

    return graph

def drawGraph(G):
    pos = nx.spring_layout(G, seed=42)

    blocked_edges = []
    unblocked_edges = []
    for u, v, attr in G.edges(data=True):
        if attr.get('blocked') == True:
            blocked_edges.append((u, v))
        else:
            unblocked_edges.append((u, v))

    nx.draw_networkx_nodes(G, pos, node_color='black')
    nx.draw_networkx_edges(G, pos, edgelist=blocked_edges, edge_color='red', width=4, style='dashed')
    nx.draw_networkx_edges(G, pos, edgelist=unblocked_edges, edge_color='gray', width=2)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
    plt.title('Graph')
    plt.show()

if __name__ == '__main__':
    n = random.randint(100, 200)
    G = generate_graph(16)
    print(satisfies_triangle_inequality_2(G))
    #drawGraph(G)
    blocked_edges = select_blocked_edges(G, 11)
    apply_blockages(G, blocked_edges)
    for u, v in G.edges:
        if G.get_edge_data(u, v)['blocked'] == None:
            break
        print(f"u: {u}, v: {v}, blocked: {G.get_edge_data(u, v)['blocked']}")
    #drawGraph(G)