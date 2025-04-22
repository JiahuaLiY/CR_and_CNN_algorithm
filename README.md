# CR and CNN algorithms
## Description
We have implemented two algorithms - **CR (Cyclic Routing)** & **CNN (Christofides Nearest Neighbor)** - to solve the **k-CCTP (k-Covering Canadian Traveller Problem)**, introduced by Chung-Shou Liao and Yamming Huang in 2014 [[hung-Shou Liao and Yamming Huang. The covering canadian traveller problem. Theor.
Comp. Sci., 530:80–88, 2014]](https://www.sciencedirect.com/science/article/pii/S0304397514001327).

The **k-CCTP** is a generalization of the classical **CTP (Canadian Traveller Problem)**, where the goal is to determine a strategy (a shortest tour) to visit a set of locations in a given graph and return to the origin as quickly as possible.

## Usage
We have two main sources files:
* **algorithms.py**: This file contains all the implementations of the algorithms used in this project, including **CR**, **CNN**, **Christofides**, **Dijkstra** and **Prim**.
* **graphUtil.py**: This file contains all the graph utility functions, including **the creation of test instances**.

```python
from src.algorithms import CR, CNN
from src.graphUtil import generate_blocked_graph, get_total_weight
import random
import numpy as np

# Set a random seed.
seed = 0
random.seed(seed)
np.random.seed(seed)

# Create a test instance that respects the algorithm's initial assumptions and constranits.
n = random.randint(10, 20)
# k must not be too large, otherwise the function 'generate_blocked_graph' may raise an error, because it cannot build a validated instance.
k = random.randint(1, n - 2)
graph = generate_blocked_graph(n, k) # return: a nx.Graph instance.

print(f"Is graph satisfies triangular inequality={satisfies_triangle_inequality_2(graph)}")
# [Out]
# Is graph satisfies triangular inequality=True

src = random.choice(list(graph.nodes))

# Create algorithm classes.
# All class constructors in algorithms.py take no arguments, it's just a general class to solve the dedicated problem.
cr = CR()
cnn = CNN()

# All class in algorithms.py contain an apply method which takes at least one argument which is the input graph.

# These two algorithms and the Christofides algorithm take an optional "display" argument,
# which is set to True by default, to visualize the steps the algorithme takes to solve the problem.
crTour = cr.apply(graph, src, display=True)
crPathWeight = get_total_weight(graph, crTour)

print(f"cr tour={crTour}, with path weight={crPathWeight}")

# [Out]
# MST: {1: 2, 2: 15, 3: 12, 4: 12, 5: 6, 6: 2, 7: 4, 8: 13, 9: 14, 10: 7, 11: 10, 12: 1, 13: 16, 14: 16, 15: None, 16: 12}
# Odd degree vertices: [2, 15, 3, 5, 8, 9, 11, 16]
# Perfect minimum matching: {(2, 3), (16, 9), (5, 11), (15, 8)}
# Eulerian tour: [15, 8, 13, 16, 9, 14, 16, 12, 4, 7, 10, 11, 5, 6, 2, 3, 12, 1, 2, 15]
# Hamiltonian circuit: [15, 8, 13, 16, 9, 14, 12, 4, 7, 10, 11, 5, 6, 2, 3, 1, 15]

# E1': {(14, 12), (4, 7)}
# P1: 15 (0) - 8 (1) - 13 (2) - 16 (3) - 9 (4) - 14 (5) - 12 (6) - 4 (7) - 7 (8) - 10 (9) - 11 (10) - 5 (11) - 6 (12) - 2 (13) - 3 (14) - 1 (15)
# Pcr: 15 (0) - 8 (1) - 13 (2) - 16 (3) - 9 (4) - 14 (5) - 4 (7) - 10 (9) - 11 (10) - 5 (11) - 6 (12) - 2 (13) - 3 (14) - 1 (15)
# Unvisited vertices: {7, 12}

# E2': {(1, 12)}
# P2: 1 (15) - 12 (6) - 7 (8)
# Pcr: 15 (0) - 8 (1) - 13 (2) - 16 (3) - 9 (4) - 14 (5) - 4 (7) - 10 (9) - 11 (10) - 5 (11) - 6 (12) - 2 (13) - 3 (14) - 1 (15) - 15 (0) - 12 (6) - 7 (8)
# Unvisited vertices: set()

# E3': set()
# P3: 7 (8) - 15 (0)
# Pcr: 15 (0) - 8 (1) - 13 (2) - 16 (3) - 9 (4) - 14 (5) - 4 (7) - 10 (9) - 11 (10) - 5 (11) - 6 (12) - 2 (13) - 3 (14) - 1 (15) - 15 (0) - 12 (6) - 7 (8) - 15 (0)
# Unvisited vertices: set()

# cr tour=[15, 8, 13, 16, 9, 14, 4, 10, 11, 5, 6, 2, 3, 1, 15, 12, 7, 15], with path weight=410.10023962911754

print()
print("########################################")
print()

cnnTour = cnn.apply(graph, src, display=True)
cnnPathWeight = get_total_weight(graph, cnnTour)
print(f"cr tour={cnnTour}, with path weight={cnnPathWeight}")

# [Out]
# MST: {1: 2, 2: 15, 3: 12, 4: 12, 5: 6, 6: 2, 7: 4, 8: 13, 9: 14, 10: 7, 11: 10, 12: 1, 13: 16, 14: 16, 15: None, 16: 12}
# Odd degree vertices: [2, 15, 3, 5, 8, 9, 11, 16]
# Perfect minimum matching: {(2, 3), (16, 9), (5, 11), (15, 8)}
# Eulerian tour: [15, 8, 13, 16, 9, 14, 16, 12, 4, 7, 10, 11, 5, 6, 2, 3, 12, 1, 2, 15]
# Hamiltonian circuit: [15, 8, 13, 16, 9, 14, 12, 4, 7, 10, 11, 5, 6, 2, 3, 1, 15]

# P: 15 - 8 - 13 - 16 - 9 - 14 - 12 - 4 - 7 - 10 - 11 - 5 - 6 - 2 - 3 - 1
# Us: [15, 12, 7]
# P1: 15 - 8 - 13 - 16 - 9 - 14 - 4 - 10 - 11 - 5 - 6 - 2 - 3 - 1 - 15
# P2: 15 - 12 - 7 - 15

# cr tour=[15, 8, 13, 16, 9, 14, 4, 10, 11, 5, 6, 2, 3, 1, 15, 12, 7, 15], with path weight=410.10023962911754
```

Some manual construction tests:
```python
from src.algorithms import CR, CNN, Christofides
import networkx as nx

christofodesGraph = nx.Graph()
edges = [ ("A", "B", 2), ("A", "C", 1), ("A", "D", 3), ("A", "E", 2),
          ("B", "C", 1), ("B", "D", 2), ("B", "E", 3),
          ("C", "D", 2), ("C", "E", 3),
          ("D", "E", 2) ]
for u, v, w in edges:
    christofodesGraph.add_edge(u, v, weight=w)
christofides = Christofides()
christofidesTour = christofides.apply(christofodesGraph, "A", display=False)
christofidesTourPrime = nx.algorithms.approximation.christofides(christofodesGraph)

christofidesPathWeight = get_total_weight(christofodesGraph, christofidesTour)
christofidesPathWeightPrime = get_total_weight(christofodesGraph, christofidesTourPrime)

print(f"{christofidesPathWeight} vs {christofidesPathWeightPrime}")
# [Out]
# ['A', 'E', 'D', 'B', 'C', 'A'] vs ['A', 'E', 'D', 'B', 'C', 'A']
# 8 vs 8

graph = nx.Graph()
for u in range(1, 17):
    for v in range(u + 1, 17):
        graph.add_edge(u, v, blocked=False, weight=1)
blockedEdges = [ (3, 4), (3, 5), (7, 8), (9, 10), (12, 13), (12, 14),
                 (16, 4), (4, 5), (8, 10), (13, 14),
                 (13, 10), (10, 5), (5, 14),
                 (14, 1) ]
for u, v in blockedEdges:
    graph[u][v]["blocked"] = True

crTour = CR().apply(graph, 1, display=False)
crPathWeight = get_total_weight(graph, crTour)

cnnTour = CNN().apply(graph, 1, display=False)
cnnPathWeight = get_total_weight(graph, cnnTour)

print(f"cr tour={crTour}, with path weight={crPathWeight}")
print(f"cnn tour={cnnTour}, with path weight={cnnPathWeight}")
# [Out]
# cr tour=[1, 16, 9, 8, 11, 6, 12, 5, 13, 4, 14, 3, 15, 2, 10, 7, 1], with path weight=16
# cnn tour=[1, 16, 9, 8, 11, 6, 12, 5, 13, 4, 14, 3, 15, 2, 1, 7, 10, 1], with path weight=17
```

## References
[[hung-Shou Liao and Yamming Huang. The covering canadian traveller problem. Theor.
Comp. Sci., 530:80–88, 2014]](https://www.sciencedirect.com/science/article/pii/S0304397514001327)

[[Niklas Hahn, Michalis Xefteris, The Covering Canadian Traveller Problem Revisited, MFCS
2023.]](https://arxiv.org/abs/2304.14319)