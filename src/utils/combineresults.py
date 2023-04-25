import pickle
import gzip
import os

def combine_results(experiment_id, statistic):
    results_path = f"results/{experiment_id}"
    os.makedirs(results_path,exist_ok=True)
    i = 0
    with gzip.open(f"results/{experiment_id}_base/{statistic}.pkl.gz") as fr:
        try:
            while True:
                if i == 247:
                    break
                pickle.dump(pickle.load(fr),
                    gzip.open(f"{results_path}/{statistic}.pkl.gz", 'ab'))
                i += 1
        except EOFError:
            pass
    
    with gzip.open(f"results/{experiment_id}_new/{statistic}.pkl.gz") as fr:
        try:
            while True:
                pickle.dump(pickle.load(fr),
                    gzip.open(f"{results_path}/{statistic}.pkl.gz", 'ab'))
        except EOFError:
            pass

if __name__ == "__main__":
    experiment_id = "experiment_2"
    
    combine_results(experiment_id, "rewards")
    combine_results(experiment_id, "trajectories")
    combine_results(experiment_id, "smiles")
    combine_results(experiment_id, "losses")