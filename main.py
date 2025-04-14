from src.algorithms import CR, CNN
from src.graphUtil import generate_blocked_graph

from tqdm import tqdm
import random

cr = CR()
cnn = CNN()
def sumWeight(G, path):
    sum = 0
    for u, v in zip(path, path[1:]):
        assert not G.get_edge_data(u, v)["blocked"]
        sum += G.get_edge_data(u, v)["weight"]
    return sum

for _ in tqdm(range(1000)):
    n = random.randint(50, 100)
    k = random.randint(14 * n, 15 * n)
    G = generate_blocked_graph(n, k)

    src = random.choice(list(G.nodes))

    crTour = cr.apply(G, src, display=False)
    assert len(set(crTour)) == n
    assert crTour[0] == crTour[-1]

    cnnTour = cnn.apply(G, src, display=False)
    assert len(set(cnnTour)) == n
    assert cnnTour[0] == cnnTour[-1]

    crResult = sumWeight(G, crTour)
    cnnResult = sumWeight(G, cnnTour)
    if crResult < cnnResult:
        print("cr")
    else:
        print("cnn")