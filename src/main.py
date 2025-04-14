import algorithms as algo
import src.graphUtil as Util
import numpy as np
import time
import pandas as pd
from tqdm import tqdm

# expérmentation

def experiment(algorithm, graph, source):
    start = time.time()
    path = algorithm.apply(graph, source, display=False)
    end = time.time()
    weight = Util.get_total_weight(graph, path)
    return weight, end - start


def run_experiments(ns, k_ratios, repeats):
    cr = algo.CR()
    cnn = algo.CNN()
    christofides = algo.Christofides()

    results = []
    total_tasks = sum(1 for _ in ns for _ in k_ratios) * repeats * 3  # 3 algorithms
    task_counter = 0
    start_time = time.time()

    for n in ns:
        for kr in k_ratios:
            k = int(kr * n)

            tqdm.write(f"\n[start] n={n}, k/n={kr:.2f}, k={k}")
            for i in range(repeats):
                graph = Util.generate_blocked_graph(n, k)
                source = np.random.randint(1, n+1)

                data = {'n': n,
                        'k': k,
                        'repeat': i,
                        'k_ration': kr}

                for name, alg in [('CR', cr), ('CNN', cnn), ('Christofides', christofides)]:
                    try:
                        weight, duration = experiment(alg, graph, source)
                        data.update({
                            f'{name}_weight': weight,
                            f'{name}_time': duration
                        })

                        task_counter += 1
                        elapsed = time.time() - start_time
                        avg_time = elapsed / task_counter
                        remaining = avg_time * (total_tasks - task_counter)
                        tqdm.write(f"[Progress] {task_counter}/{total_tasks} - {name} done, "
                                   f"time used: {duration:.2f}s, elapsed: {elapsed:.1f}s, estimated remaining: {remaining:.1f}s")
                    except Exception as e:
                        print(f"Error: {e} on n={n}, k={k}, repeat={i}, alg={name}")
                        continue

                results.append(data)

    return pd.DataFrame(results)


if __name__ == '__main__':
    ns = [50,60,70,80,90,100,120,140,160,180,200,300,400,500]
    k_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    repeats = 10

    df = run_experiments(ns, k_ratios, repeats)
    df.to_csv("experiment_results.csv", index=False)
    print("Experiments completed and results saved.")
