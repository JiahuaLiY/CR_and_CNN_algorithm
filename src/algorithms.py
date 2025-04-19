import networkx as nx
import heapq

# from random import choices
# from collections import deque
from copy import deepcopy

class Prim(object):
    def pushAdjacentEdges(self, u, graph, condidates, visited):
        """Push all edges from vertex u to unvisited neighbors into the heap."""
        for v in graph.neighbors(u):
            if not visited[v]:
                w = graph.get_edge_data(u, v)["weight"]
                heapq.heappush(condidates, (w, u, v))

    def getValidatedMinEdge(self, condidates, visited):
        """Return a validated minimum weight edge."""
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
        # self._perfectMinMatchAlgorithm = GreedyPerfecMintMatch()

    def findEulerianTour(self, eulerianGraph: nx.MultiGraph, src):
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
        return eulerianTour
    
    def buildHamiltonianCircuit(self, eulerianTour):
        """Build a Hamiltonian circuit from an Eulerian tour."""
        visited = { v: False for v in eulerianTour }
        hamiltonianCircuit = []
        for v in eulerianTour:
            if not visited[v]:
                visited[v] = True
                hamiltonianCircuit.append(v)
        hamiltonianCircuit.append(eulerianTour[-1])
        assert hamiltonianCircuit[0] == hamiltonianCircuit[-1]
        return hamiltonianCircuit

    def apply(self, graph, src, display=True):
        def log(mst, oddDegreeVertices, perfectMinMatch, eulerianTour, hamiltonianCircuit):
            if not display:
                return
            print(f"MST: {mst}")
            print(f"Odd degree vertices: {oddDegreeVertices}")
            print(f"Perfect minimum matching: {perfectMinMatch}")
            print(f"Eulerian tour: {eulerianTour}")
            print(f"Hamiltonian circuit: {hamiltonianCircuit}")
            print()
        # Compute a MST.
        parent = self._mstAlgorithm.apply(graph, src)
        # print(f"{parent=}")

        # Count the degree of each node in MST and filter only nodes with odd degree.
        degreeCnt = dict()
        for u, v in parent.items():
            if v is not None:
                degreeCnt[u] = degreeCnt.get(u, 0) + 1
                degreeCnt[v] = degreeCnt.get(v, 0) + 1
        oddDegreeNodes = list(filter(lambda v: degreeCnt[v] & 1, degreeCnt.keys()))
        # print(f"{oddDegreeNodes=}, {degreeCnt=}")

        # Compute the perfect min match.
        oddVerticesGraph = graph.subgraph(oddDegreeNodes)
        # perfectMinMatch = self._perfectMinMatchAlgorithm.apply(oddVerticesGraph)
        perfectMinMatch = nx.min_weight_matching(oddVerticesGraph)
        # print(f"{perfectMinMatch=}")

        # Build the Eulerian graph.
        eulerianGraph = nx.MultiGraph()
        for u, v in parent.items():
            if v is not None:
                eulerianGraph.add_edge(u, v)
        for u, v in perfectMinMatch:
            eulerianGraph.add_edge(u, v)

        # Find an Eulerian tour.
        eulerianTour = self.findEulerianTour(eulerianGraph, src)
        # print(f"{eulerianTour=} with size={len(eulerianTour)}")

        # Build a Hamiltonian circuit.
        hamiltonianCircuit = self.buildHamiltonianCircuit(eulerianTour)
        # print(f"{hamiltonianCircuit=} with size={len(hamiltonianCircuit)}")

        log(parent, oddDegreeNodes, perfectMinMatch, eulerianTour, hamiltonianCircuit)
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
                 blockedEdges,
                 vertexToIndex,
                 clockwiseDirection):
        """Perform the shortcut procedure."""
        unvisitedVertices = deepcopy(unvisitedVertices)
        nbUnvisitedVertices = len(unvisitedVertices)
        newBlockedEdges = set()
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
                newBlockedEdges.add((vi, vj))
                # Try to find an intermediate vertex vl such that the path vi - vl - vj is not blocked.
                # l satisfies the contraint: i < l < j (if clockwise direction) 
                # or i > l > j (if counterclockwise direction).
                l = (vertexToIndex[vi] + step) % len(P)
                vl = P[l]
                while vl != vj:
                    if graph.get_edge_data(vi, vl)["blocked"]:
                        if (vi, vl) not in blockedEdges and (vl, vi) not in blockedEdges:
                            newBlockedEdges.add((vi, vl))
                    elif graph.get_edge_data(vl, vj)["blocked"]:
                        if (vl, vj) not in blockedEdges and (vj, vl) not in blockedEdges:
                            newBlockedEdges.add((vl, vj))
                    else:
                        break
                    l = (l + step) % len(P)
                    vl = P[l]
                if vl != vj:
                    unvisitedVertices.remove(vj)
                    Pcr.append(vl)
                    Pcr.append(vj)
                    i = j
                    j = i + 1
                else:
                    j += 1
        return unvisitedVertices, newBlockedEdges

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

    def apply(self, graph, src, display=True):
        def log(Pcr, Pm, unvisitedVertices, blockedEdges, m):
            if not display:
                return
            print(f"E{m}': {blockedEdges}")
            print(f"P{m}: {" - ".join(map(lambda v: f"{v} ({vertexToIndex[v]})", Pm))}")
            print(f"Pcr: {" - ".join(map(lambda v: f"{v} ({vertexToIndex[v]})", Pcr))}")
            print(f"Unvisited vertices: {unvisitedVertices}")
            print()
        Pcr = [src]
        P = self._christofides.apply(graph, src, display=display)[:-1]
        vertexToIndex = { v: i for i, v in enumerate(P) }
        unvisitedVertices = set(graph.nodes)
        unvisitedVertices.remove(src)
        blockedEdges = set()
        clockwiseDirection = True
        Pm, PmMinus = None, None
        m = 1
        while unvisitedVertices:
            Pm = self.buildShortcutPath(Pcr, P, unvisitedVertices, vertexToIndex)
            if m == 1 or Pm[0] == PmMinus[-1]:
                # Perform the shortcut procedure in the same direction as that in the (m - 1)th round.
                unvisitedVerticesAfter, newBlockedEdges = self.shortcut(graph,
                                                                        Pcr,
                                                                        P,
                                                                        Pm,
                                                                        unvisitedVertices,
                                                                        blockedEdges,
                                                                        vertexToIndex,
                                                                        clockwiseDirection)
                # If both sets are equal, it means there's no non-blocking path in the current direction.
                if unvisitedVerticesAfter == unvisitedVertices:
                    clockwiseDirection = not clockwiseDirection
                    Pm[1:] = Pm[:0:-1]
                    unvisitedVerticesAfter, newBlockedEdges = self.shortcut(graph,
                                                                            Pcr,
                                                                            P,
                                                                            Pm,
                                                                            unvisitedVertices,
                                                                            blockedEdges,
                                                                            vertexToIndex,
                                                                            clockwiseDirection)
                unvisitedVertices = unvisitedVerticesAfter
            else:
                # Perform the shortcut procedure in the opposite direction to that in the (m - 1)th round.
                clockwiseDirection = not clockwiseDirection
                # reverse the shortcut path, but keeps the first element.
                Pm[1:] = Pm[:0:-1]
                unvisitedVerticesAfter, newBlockedEdges = self.shortcut(graph,
                                                                        Pcr,
                                                                        P,
                                                                        Pm,
                                                                        unvisitedVertices,
                                                                        blockedEdges,
                                                                        vertexToIndex,
                                                                        clockwiseDirection)
                # Here we must not have equality between 'unvisitedVerticesAfter' and 'unvisitedVertices',
                # since thepretically if we have equality, it means that graph is badly constructed and does not
                # respect the initial hypothesis, in particular k < n - 1.
                assert unvisitedVerticesAfter != unvisitedVertices
                unvisitedVertices = unvisitedVerticesAfter

            log(Pcr, Pm, unvisitedVertices, newBlockedEdges, m)
            blockedEdges.update(newBlockedEdges)
            PmMinus = Pm
            m += 1
        end = Pcr[-1]
        # If the direct path from the end to the src is not blocked, then return directly to the src. 
        if not graph.get_edge_data(end, src)["blocked"]:
            Pcr.append(src)
        # Try to find a path through an intermediate vertex v such that end - v - src is not blocked.
        else:
            endIndex = vertexToIndex[end]
            index = (endIndex + 1) % len(P)
            while index != endIndex:
                v = P[index]
                if not graph.get_edge_data(end, v)["blocked"] and \
                    not graph.get_edge_data(v, src)["blocked"]:
                    Pcr.append(v)
                    Pcr.append(src)
                    break
                index = (index + 1) % len(P)
        log(Pcr, [end, src], unvisitedVertices, set(), m)
        return Pcr


class Dijkstra(object):
    def buildPath(self, parent, src, dest):
        """Build the shortest path from source to destination."""
        path = [dest]
        v = dest
        while v != src:
            u = parent[v]
            path.append(u)
            v = u
        return path
    
    def getMinVertex(self, unvisitedVertices, distances):
        """Return the unvisited vertex with the smallest known distance."""
        minVertex = None
        minDistance = float("inf")
        for v in unvisitedVertices:
            if distances[v] < minDistance:
                minVertex = v
                minDistance = distances[v]
        if minVertex is not None:
            unvisitedVertices.remove(minVertex)
        return minVertex
    
    def apply(self, graph, src, dest):
        unvisitedVertices = set(graph.nodes)
        parent = { v: None for v in graph.nodes }
        distances = { v: float("inf") for v in graph.nodes }
        distances[src] = 0
        while unvisitedVertices:
            u = self.getMinVertex(unvisitedVertices, distances)
            # If u is None, then this means there is no path from source to destination.
            if u is None:
                return None
            
            if u == dest:
                break
            for v in graph.neighbors(u):
                newDistance = distances[u] + graph.get_edge_data(u, v)["weight"]
                if newDistance < distances[v]:
                    distances[v] = newDistance
                    parent[v] = u
        return self.buildPath(parent, src, dest)


class CNN(object):
    def __init__(self):
        self._christofides = Christofides()
        self._shortestPath = Dijkstra()
        # self._nn = NearestNeighbor()
    
    def shortcut(self, graph, P):
        """Perform the shortcut procedure."""
        unvisitedVertices = [P[0]]
        blockedEdges = set()
        Pprime = [P[0]]
        i, j = 0, 1
        nbVertices = graph.number_of_nodes()
        while j < nbVertices:
            vi, vj = P[i], P[j]
            for x in graph.neighbors(vi):
                if graph.get_edge_data(vi, x)["blocked"]:
                    if (vi, x) not in blockedEdges and (x, vi) not in blockedEdges:
                        blockedEdges.add((vi, x))

            if not graph.get_edge_data(vi, vj)["blocked"]:
                Pprime.append(vj)
                i = j
            else:
                unvisitedVertices.append(vj)
            j += 1
        for x in graph.neighbors(vj):
            if graph.get_edge_data(vj, x)["blocked"]:
                if (vj, x) not in blockedEdges and (x, vj) not in blockedEdges:
                    blockedEdges.add((vj, x))
        if graph.get_edge_data(P[i], P[0])["blocked"]:
            Pprime += Pprime[-2::-1] # Reverse path.
        else:
            Pprime.append(P[0])
        
        GStar = graph.copy()
        GStar.remove_edges_from(blockedEdges)
        return GStar, unvisitedVertices, Pprime

    def compress(self, graphStar, unvisitedVertices, graph):
        """Build the induced graph which contains all unvisited vertices including
        the source, and for all vertices u, v in the graph, they have at most 2 connecting edges."""
        setOfUnvisitedVertices = set(unvisitedVertices)

        # Build G', which is the induced graph containing all the edges connecting 2 unvisited vertices.
        Eprime = set()
        Gprime = nx.MultiGraph()
        Gprime.add_nodes_from(unvisitedVertices)
        for u, v, data in graph.edges(data=True):
            if u in setOfUnvisitedVertices and v in setOfUnvisitedVertices:
                Gprime.add_edge(u, v, **data)
                Eprime.add((u, v))

        # Build H, which is the graph containing all non-blocking edges.
        H = nx.Graph()
        H.add_nodes_from(graph.nodes)
        for u, v, data in graphStar.edges(data=True):
            if (u, v) not in Eprime and (v, u) not in Eprime:
                assert not data["blocked"]
                H.add_edge(u, v, **data)

        nbUnvisitedVertices = len(unvisitedVertices)
        for i in range(nbUnvisitedVertices):
            for j in range(i + 1, nbUnvisitedVertices):
                vi = unvisitedVertices[i]
                vj = unvisitedVertices[j]
                
                # Compute the shortest path from vi to vj using at least 1 visited intermediate vertex,
                # since H contains only non-blocking edges, we guarantee that the shortest path is accessible.
                shortestPath = self._shortestPath.apply(H, vi, vj)
                if shortestPath is None:
                    continue

                cost = sum(H.get_edge_data(u, v)["weight"]
                           for u, v in zip(shortestPath, shortestPath[1:]))
                # To facilitate redevelopment, we decided to store the shortest path directly in the edge data,
                # but other approaches are also possible.
                Gprime.add_edge(vi,
                                vj,
                                weight = cost,
                                blocked = False,
                                shortest_path = shortestPath)
        return Gprime

    def nearestNeighbor(self, graph: nx.MultiGraph, src):
        """Apply the nearest neighbor algorithm."""
        P = [src]
        unvisitedVertices = set(graph.nodes)
        curVertex = src
        while unvisitedVertices:
            unvisitedVertices.remove(curVertex)

            # Try to find a nearest neighbor which can be reached either by using the connecting edge directly,
            # or by using the shortest path.
            # Note that it is possible that neither of these 2 options is feasible,
            # as the instance described in the paper - Niklas Hahn, Michalis Xefteris, The Covering Canadian Traveller Problem Revisited, MFCS 2023.
            path = None
            nearestVertex = None
            minWeight = float("inf")
            for v in graph.neighbors(curVertex):
                if v not in unvisitedVertices:
                    continue

                for data in graph.get_edge_data(curVertex, v).values():
                    weight = data["weight"]
                    # If the edge is blocked or if the weight of this edge is no better than that
                    # of the edge already found, ignore this edge.
                    if data["blocked"] or weight >= minWeight:
                        continue
                    nearestVertex = v
                    minWeight = weight
                    if "shortest_path" not in data:
                        path = [v]
                    else:
                        path = data["shortest_path"]
                        assert path[0] == curVertex or path[-1] == curVertex
                        if path[0] == curVertex:
                            path = path[1:]
                        else:
                            path = path[-2::-1]
            if nearestVertex is None:
                # If 'nearestVertex' is None (i.e. there's no nearest neighbor) and 'unvisitedVertices' is not empty,
                # then we end up in the situation described above, in which case, we stop the algorithm, it's the limit of NN.
                assert not unvisitedVertices
                break
            P.extend(path)
            curVertex = nearestVertex
        # Same procedure as above.
        path = None
        minWeight = float("inf")
        for data in graph.get_edge_data(curVertex, src).values():
            weight = data["weight"]
            # If the edge is blocked or if the weight of this edge is no better than that
            # of the edge already found, ignore this edge.
            if data["blocked"] or weight > minWeight:
                continue
            minWeight = weight
            if "shortest_path" not in data:
                path = [v]
            else:
                path = data["shortest_path"]
                assert path[0] == curVertex or path[-1] == curVertex
                if path[0] == curVertex:
                    path = path[1:]
                else:
                    path = path[-2::-1]
        if path is None:
            P += P[-2::-1]
        else:
            P.extend(path)
        return P

    def apply(self, graph, src, display=True):
        def log(P, Us, P1, P2):
            if not display:
                return
            print(f"P: {" - ".join(map(str, P))}")
            print(f"Us: {Us}")
            print(f"P1: {" - ".join(map(str, P1))}")
            print(f"P2: {" - ".join(map(str, P2))}")
            print()
        P = self._christofides.apply(graph, src, display=display)[:-1]
        # print(f"\n{P=}")
        # P = list(range(1, 17)) # test

        GStar, Us, P1 = self.shortcut(graph, P)
        # print(f"{Us=}")
        # print(f"{P1=}")
        # If the size of Us is less than 2, i.e. we have visited all the vertices,
        # (because Us = {src} U { all unvisited vertices }), so wa can just return P1 as the result.
        if len(Us) < 2:
            log(P, Us, P1, [])
            return P1
        
        GPrime = self.compress(GStar, Us, graph)
        # for u, v, d in GPrime.edges(data=True):
        #     print(u, v, d)

        P2 = self.nearestNeighbor(GPrime, src)
        # print(f"{P2=}")
        log(P, Us, P1, P2)
        return P1 + P2[1:]

