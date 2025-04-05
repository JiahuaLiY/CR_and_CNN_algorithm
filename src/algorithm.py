import networkx as nx
import heapq

# from random import choices
# from collections import deque
from copy import deepcopy

class Prim(object):
    def pushAdjacentEdges(self, u, graph, condidates, visited):
        """Push all adjacent edges into a heap."""
        for v in graph.neighbors(u):
            if not visited[v]:
                w = graph.get_edge_data(u, v)["weight"]
                heapq.heappush(condidates, (w, u, v))

    def getValidatedMinEdge(self, condidates, visited):
        """Get a validated minimum weight edge."""
        while condidates:
            _, u, v = heapq.heappop(condidates)
            if not visited[v]:
                return u, v

    def apply(self, graph, src):
        parent = { v: None for v in graph.nodes }
        visited = { v: False for v in graph.nodes }
        visited[src] = True
        condidates = []
        self.pushAdjacentEdges(src, graph, condidates, visited)
        while True:
            edge = self.getValidatedMinEdge(condidates, visited)
            # If the edge is None, it means there's no edge left, we done.
            if edge is None:
                return parent
            u, v = edge
            parent[v] = u
            visited[v] = True
            self.pushAdjacentEdges(v, graph, condidates, visited)


class GreedyPerfecMintMatch(object):
    def apply(self, graph):
        visited = { v: False for v in graph.nodes }
        perfectMinMatch = []
        sortedEdges = sorted(graph.edges.data("weight"), key=lambda x: x[-1])
        for u, v, _ in sortedEdges:
            if not visited[u] and not visited[v]:
                visited[u] = visited[v] = True
                perfectMinMatch.append((u, v))
        return perfectMinMatch


class Christofides(object):
    def __init__(self):
        self._mstAlgorithm = Prim()
        self._perfectMinMatchAlgorithm = GreedyPerfecMintMatch()

    def findEulerianTour(self, eulerianGraph, src):
        """Find an Eulerian tour."""
        # Example of execution to see why we need to backtrack.
        # 1 -- 2 -- 4
        # |   | |   |
        # |    3    |
        # -----------
        #########################################################
        # stack=[1], backtrack=[], eulerianTour=[]
        # stack=[2], backtrack=[1], eulerianTour=[]
        # stack=[4], backtrack=[1, 2], eulerianTour=[]
        # stack=[1], backtrack=[1, 2, 4], eulerianTour=[]
        # stack=[4], backtrack=[1, 2], eulerianTour=[1]
        # stack=[2], backtrack=[1], eulerianTour=[1, 4]
        # stack=[3], backtrack=[1, 2], eulerianTour=[1, 4]
        # stack=[2], backtrack=[1, 2, 3], eulerianTour=[1, 4]
        # stack=[3], backtrack=[1, 2], eulerianTour=[1, 4, 2]
        # stack=[2], backtrack=[1], eulerianTour=[1, 4, 2, 3]
        # stack=[1], backtrack=[], eulerianTour=[1, 4, 2, 3, 2]
        # stack=[], backtrack=[], eulerianTour=[1, 4, 2, 3, 2, 1]
        #########################################################
        # solution = [1, 4, 2, 3, 2, 1].
        # But if we don't backtrack, we may end up with [1, 2, 4, 1].
        backtrack = []
        eulerianTour = []
        stack = [src]
        while stack:
            u = stack.pop()
            if eulerianGraph.degree(u) > 0:
                v = next(iter(eulerianGraph[u]))
                backtrack.append(u)
                stack.append(v)
                eulerianGraph.remove_edge(u, v)
            else:
                eulerianTour.append(u)
                if backtrack:
                    stack.append(backtrack.pop())
        # The Eulerian tour is indeed reversed, but it doesn't matter.
        return eulerianTour
    
    def buildHamiltonianTour(self, eulerianTour):
        """Build a Hamiltonian circuit from an Eulerian tour."""
        visited = { v: False for v in eulerianTour }
        hamiltonianTour = []
        for v in eulerianTour:
            if not visited[v]:
                visited[v] = True
                hamiltonianTour.append(v)
        hamiltonianTour.append(eulerianTour[-1])
        assert hamiltonianTour[0] == hamiltonianTour[-1]
        return hamiltonianTour

    def apply(self, graph, src):
        # Compute a MST.
        parent = self._mstAlgorithm.apply(graph, src)
        print(f"{parent=}")

        # Count the degree of each node in MST and filter only nodes with odd degree.
        degreeCnt = dict()
        for u, v in parent.items():
            if v is not None:
                degreeCnt[u] = degreeCnt.get(u, 0) + 1
                degreeCnt[v] = degreeCnt.get(v, 0) + 1
        oddDegreeNodes = list(filter(lambda v: degreeCnt[v] & 1, degreeCnt.keys()))
        print(f"{oddDegreeNodes=}, {degreeCnt=}")

        # Compute the perfect min match.
        oddVerticesGraph = graph.subgraph(oddDegreeNodes)
        perfectMinMatch = self._perfectMinMatchAlgorithm.apply(oddVerticesGraph)
        print(f"{perfectMinMatch=}")

        # Build the Eulerian graph.
        eulerianGraph = nx.MultiGraph()
        eulerianGraph = nx.MultiGraph()
        for u, v in parent.items():
            if v is not None:
                eulerianGraph.add_edge(u, v)
        for u, v in perfectMinMatch:
            eulerianGraph.add_edge(u, v)
        print(eulerianGraph.edges)
        # Find an Eulerian tour.
        eulerianTour = self.findEulerianTour(eulerianGraph, src)
        print(f"{eulerianTour=} with size={len(eulerianTour)}")

        # Build a Hamiltonian circuit.
        hamiltonianCircuit = self.buildHamiltonianTour(eulerianTour)
        print(f"{hamiltonianCircuit=} with size={len(hamiltonianCircuit)}")
        return hamiltonianCircuit


class CR(object):
    def __init__(self):
        self._christofides = Christofides()
    
    def shortcut(self,
                 graph,
                 Pcr,
                 P,
                 Pm,
                 unvisitedVertices,
                 vertexToIndex,
                 clockwiseDirection):
        """Perform the shortcut procedure."""
        unvisitedVertices = deepcopy(unvisitedVertices)
        nbUnvisitedVertices = len(unvisitedVertices)
        blockedEdges = set()
        step = 1 if clockwiseDirection else -1
        i, j = 0, 1
        while j <= nbUnvisitedVertices:
            vi, vj = Pm[i], Pm[j]
            if not graph.get_edge_data(vi, vj)["blocked"]:
                unvisitedVertices.remove(vj)
                Pcr.append(vj)
                i = j
                j = i + 1
            else:
                blockedEdges.add((vi, vj))
                # Try to find an internal vertex vl such that the path vi - vl - vj
                # is not blocked.
                # l satisfies the contraint: i < l < j (if clockwise direction) 
                # or i > l > j (if counterclockwise direction).
                l = (vertexToIndex[vi] + step) % len(P)
                vl = P[l]
                while vl != vj:
                    if graph.get_edge_data(vi, vl)["blocked"]:
                        blockedEdges.add((vi, vl))
                    elif graph.get_edge_data(vl, vj)["blocked"]:
                        blockedEdges.add((vl, vj))
                    else:
                        break
                    l += step
                    vl = P[l]
                if vl != vj:
                    unvisitedVertices.remove(vj)
                    Pcr.append(vl)
                    Pcr.append(vj)
                    i = j
                    j = i + 1
                else:
                    j += 1
        return unvisitedVertices, blockedEdges

    def buildShortcutPath(self,
                          Pcr,
                          P,
                          unvisitedVertices,
                          vertexToIndex):
        """Build the shortcut path Pm through unvisited vertices following
        the initial order of the tour P."""
        start = Pcr[-1]
        Pm = [start]
        startIndex = vertexToIndex[start]
        index = (startIndex + 1) % len(P)
        while index != startIndex:
            v = P[index]
            if v in unvisitedVertices:
                Pm.append(v)
            index = (index + 1) % len(P)
        return Pm
    
    def apply(self, graph: nx.Graph, src):
        Pcr = [src]
        P = list(range(1, 17)) # test.
        # P = self._christofides.apply(graph, src)[:-1]
        vertexToIndex = { v: i for i, v in enumerate(P) }
        unvisitedVertices = set(graph.nodes)
        unvisitedVertices.remove(src)
        clockwiseDirection = True
        Pm, PmMinus = None, None
        m = 1
        while unvisitedVertices:
            Pm = self.buildShortcutPath(Pcr, P, unvisitedVertices, vertexToIndex)
            if m == 1 or Pm[0] == PmMinus[-1]:
                # Perform the shortcut procedure in the same direction as that in the (m - 1)th round.
                unvisitedVerticesAfter, blockedEdges = self.shortcut(graph,
                                                                     Pcr,
                                                                     P,
                                                                     Pm,
                                                                     unvisitedVertices,
                                                                     vertexToIndex,
                                                                     clockwiseDirection)
                # If both set are equal, it means there's no non-blocking path in the current direction ?
                # to do
                if unvisitedVerticesAfter == unvisitedVertices:
                    clockwiseDirection = not clockwiseDirection
                    Pm[1:] = Pm[:0:-1]
                    unvisitedVertices, blockedEdges = self.shortcut(graph,
                                                                    Pcr,
                                                                    P,
                                                                    Pm,
                                                                    unvisitedVertices,
                                                                    vertexToIndex,
                                                                    clockwiseDirection)
                unvisitedVertices = unvisitedVerticesAfter
            else:
                # Perform the shortcut procedure in the opposite direction to that in the (m - 1)th round.
                clockwiseDirection = not clockwiseDirection
                # reverse the shortcut path, but keeps the first element.
                Pm[1:] = Pm[:0:-1]
                unvisitedVertices, blockedEdges = self.shortcut(graph,
                                                                Pcr,
                                                                P,
                                                                Pm,
                                                                unvisitedVertices,
                                                                vertexToIndex,
                                                                clockwiseDirection)
            print(f"E'{m}={blockedEdges}")
            print(f"{Pm=}")
            print(f"{Pcr=}")
            print()
            PmMinus = Pm
            m += 1
        # to do
        return Pcr


class CNN(object):
    def __init__(self):
        self._christofides = Christofides()
    pass

if __name__ == "__main__":
    christofides = Christofides()
    graph = nx.Graph()
    edges = [ ("A", "B", 2), ("A", "C", 1), ("A", "D", 3), ("A", "E", 2),
              ("B", "C", 1), ("B", "D", 2), ("B", "E", 3),
              ("C", "D", 2), ("C", "E", 3),
              ("D", "E", 2) ]
    for u, v, w in edges:
        graph.add_edge(u, v, weight=w)
    print(christofides.apply(graph, "A"))

    print()

    graph = nx.Graph()
    for u, v, w in [(1, 2, 1), (1, 3, 1), (1, 4, 1), (1, 5, 2), (2, 3, 1), (2, 5, 1), (2, 4, 2),
                    (3, 4, 1), (3, 5, 1), (4, 5, 1)]:
        graph.add_edge(u, v, weight=w)
    print(christofides.apply(graph, 1))

    graph = nx.Graph()
    edges = [ (1, 2), () ]
    for u in range(1, 17):
        for v in range(u, 17):
            graph.add_edge(u, v, blocked=False)
    blockedEdges = [ (3, 4), (3, 5), (7, 8), (9, 10), (12, 13), (12, 14),
                     (16, 4), (4, 5), (8, 10), (13, 14),
                     (13, 10), (10, 5), (5, 14),
                     (14, 1) ]
    for u, v in blockedEdges:
        graph[u][v]["blocked"] = True

    print(CR().apply(graph, 1))
    pass

