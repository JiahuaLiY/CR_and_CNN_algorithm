import time

from src.algorithms import CR, CNN
from src.graphUtil import generate_blocked_graph, get_total_weight

import networkx as nx
import tqdm
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

def runComparaison(rangeOfNodes, kFormula, nbGraph, nbRepats, fileName=None):
    cr = CR()
    cnn = CNN()

    if fileName is not None:
        header_needed = not os.path.isfile(fileName)

    for nbNode in rangeOfNodes:
        kBlockage = kFormula(nbNode)
        for _ in range(nbGraph):
            graph = generate_blocked_graph(nbNode, kBlockage)
            for _ in tqdm.tqdm(range(nbRepats), desc=f"Size {nbNode}"):
                data = []
                src = random.choice(list(graph.nodes))
                start = time.perf_counter()
                crTour = cr.apply(graph, src, display=False)
                CRrun_time = time.perf_counter() - start
                assert crTour[0] == crTour[-1]
                crWeight = get_total_weight(graph, crTour)
                data.append({
                    "algorithm": "CR",
                    "number of nodes": nbNode,
                    "weight": crWeight,
                    "run time": CRrun_time
                })

                start = time.perf_counter()
                cnnTour = cnn.apply(graph, src, display=False)
                CNNrun_time = time.perf_counter() - start
                assert cnnTour[0] == cnnTour[-1]
                cnnWeight = get_total_weight(graph, cnnTour)
                data.append({
                    "algorithm": "CNN",
                    "number of nodes": nbNode,
                    "weight": cnnWeight,
                    "run time": CNNrun_time
                })

                if fileName is not None:
                    pd.DataFrame(data).to_csv(fileName,
                                            mode='a',
                                            header=header_needed,
                                            index=False)
                    header_needed = False

def runKComparaison(nbNode, kratios, nbGraph, nbRepats, fileName=None):
    cr = CR()
    cnn = CNN()

    if fileName is not None:
        header_needed = not os.path.isfile(fileName)

    for kratio in kratios:
        kBlockage = round(kratio*nbNode) if kratio < 1 else kratio
        for _ in range(nbGraph):
            graph = generate_blocked_graph(nbNode, kBlockage)
            for _ in tqdm.tqdm(range(nbRepats), desc=f"k {kBlockage}"):
                data = []
                src = random.choice(list(graph.nodes))
                start = time.perf_counter()
                crTour = cr.apply(graph, src, display=False)
                CRrun_time = time.perf_counter() - start
                assert crTour[0] == crTour[-1]
                crWeight = get_total_weight(graph, crTour)
                data.append({
                    "algorithm": "CR",
                    "number of nodes": nbNode,
                    "k" : kBlockage,
                    "weight": crWeight,
                    "run time": CRrun_time
                })

                start = time.perf_counter()
                cnnTour = cnn.apply(graph, src, display=False)
                CNNrun_time = time.perf_counter() - start
                assert cnnTour[0] == cnnTour[-1]
                cnnWeight = get_total_weight(graph, cnnTour)
                data.append({
                    "algorithm": "CNN",
                    "number of nodes": nbNode,
                    "k": kBlockage,
                    "weight": cnnWeight,
                    "run time": CNNrun_time
                })

                if fileName is not None:
                    pd.DataFrame(data).to_csv(fileName,
                                            mode='a',
                                            header=header_needed,
                                            index=False)
                    header_needed = False

    # plt.figure()
    # sns.boxplot(
    #     x="number of nodes",
    #     y="weight",
    #     hue="algorithm",
    #     data=dataFrame,
    #     palette={ "CR": "lightcoral", "CNN": "lightblue" },
    #     width=0.6
    # )
    #
    # plt.title("Comparaison of path weights: CR vs CNN algorithms")
    # plt.xlabel("Number of nodes")
    # plt.ylabel("Path weight")
    # plt.legend(title="Algorithm")
    #
    # if fileName is not None:
    #     plt.savefig(fileName)
    # plt.show()


if __name__ == "__main__":
    # runComparaison(np.arange(50, 210, 10),
    #                lambda x: .2 * (x**2), 50,
    #                5,
    #                "large_k50graph.csv")

    # runComparaison(np.arange(10, 52, 2),
    #                lambda x: x-2, 50,
    #                5,
    #                "fix_k_vary_n50graph.csv")


    # nbNode = 50
    # k_ratios = list(np.arange(0.1, 1, 0.1))
    # k_ratios.append(nbNode-2)
    # runKComparaison(nbNode,
    #                 k_ratios,
    #                 50,
    #                 5,
    #                 "fix_n_vary_k50graph.csv")


    runComparaison(np.arange(50, 210, 10),
                   lambda x: .2 * (x**2), 50,
                   1,
                    "large_k50graph.csv")

    runComparaison(np.arange(10, 52, 2),
                   lambda x: x-2, 50,
                   1,
                   "fix_k_vary_n50graph.csv")


    nbNode = 50
    k_ratios = list(np.arange(0.1, 1, 0.1))
    k_ratios.append(nbNode-2)
    runKComparaison(nbNode,
                    k_ratios,
                    50,
                    1,
                    "fix_n_vary_k50graph.csv")
    pass