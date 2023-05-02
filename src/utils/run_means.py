import gzip
import pickle
from collections import defaultdict
import numpy as np


if __name__ == "__main__":

    for stop in [100000,1000000]:
        for k in [10,100,1000]:
            means = []
            
            for eid in ["experiment_1", "experiment_4","experiment_5"]:
                path = f'results/{eid}'
                rewards = []
                with gzip.open(f"{path}/rewards.pkl.gz") as fr:
                    try:
                        while True:
                            rewards.extend(pickle.load(fr))
                    except EOFError:
                        pass
                
                # store smiles in a list 
                smiles = []
                with gzip.open(f"{path}/smiles.pkl.gz") as fr:
                    try:
                        while True:
                            smiles.extend(pickle.load(fr))
                    except EOFError:
                        pass
            
                is_sampled = defaultdict(lambda:False)

                # change to numpy array
                rewards = np.array(rewards)
                
                top_k = np.sort(rewards[:k])

                top_k_avg = [np.mean(top_k)]

                for l in range(k+1,len(rewards)):
                    if rewards[l] > top_k[0] and not is_sampled[smiles[l]]:
                        top_k[0] = rewards[l]
                        is_sampled[smiles[l]] = True
                    top_k_avg.append(np.mean(top_k))
                    top_k = np.sort(top_k)
                    if l == stop-1:
                        means.append(np.mean(top_k))
                        break
            
            print(f"For k={k} after {stop} samples:")
            print(np.mean(means))
            print(np.std(means))
            print("")