from src.algorithms import CR, CNN
from src.graphUtil import generate_blocked_graph

import networkx as nx
import tqdm
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def pathWeight(graph: nx.Graph, path):
    weight = 0
    for u, v in zip(path, path[1:]):
        data = graph.get_edge_data(u, v)
        assert not data["blocked"]
        weight += data["weight"]
    return weight

def runWeightComparaison(rangeOfNodes, kFormula, nbRepats, fileName=None):
    cr = CR()
    cnn = CNN()
    data = []
    for nbNode in rangeOfNodes:
        kBlockage = kFormula(nbNode)
        for _ in tqdm.tqdm(range(nbRepats), desc=f"Size {nbNode}"):
            graph = generate_blocked_graph(nbNode, kBlockage)
            src = random.choice(list(graph.nodes))

            crTour = cr.apply(graph, src, display=False)
            assert crTour[0] == crTour[-1]
            crWeight = pathWeight(graph, crTour)
            data.append({
                "algorithm": "CR",
                "number of nodes": nbNode,
                "weight": crWeight
            })

            cnnTour = cnn.apply(graph, src, display=False)
            assert cnnTour[0] == cnnTour[-1]
            cnnWeight = pathWeight(graph, cnnTour)
            data.append({
                "algorithm": "CNN",
                "number of nodes": nbNode,
                "weight": cnnWeight
            })
    dataFrame = pd.DataFrame(data)

    plt.figure()
    sns.boxplot(
        x="number of nodes",
        y="weight",
        hue="algorithm",
        data=dataFrame,
        palette={ "CR": "lightcoral", "CNN": "lightblue" },
        width=0.6
    )

    plt.title("Comparaison of path weights: CR vs CNN algorithms")
    plt.xlabel("Number of nodes")
    plt.ylabel("Path weight")
    plt.legend(title="Algorithm")

    if fileName is not None:
        plt.savefig(fileName)
    plt.show()


if __name__ == "__main__":
    # runWeightComparaison(np.arange(10, 52, 2),
    #                      lambda x: x - 2,
    #                      100,
    #                      "small_k.png")
    
    runWeightComparaison(np.arange(50, 210, 10),
                         lambda x: .2 * (x**2),
                         50,
                         "large_k.png")
    pass