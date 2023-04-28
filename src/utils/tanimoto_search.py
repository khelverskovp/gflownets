import gzip
import pickle
import numpy as np
import time
import os
import logging
from rdkit import Chem, DataStructs

def is_diverse_tanimoto(smi, diverse_modes, S):
    fps = Chem.RDKFingerprint(Chem.MolFromSmiles(smi))
    for smi_ref in diverse_modes:
        fps_ref = Chem.RDKFingerprint(Chem.MolFromSmiles(smi_ref))
        tanimoto_sim = DataStructs.FingerprintSimilarity(fps,fps_ref)
        if tanimoto_sim >= S:
            return False
    return True

def compute_tanimoto_counts(T, S, experiment_id):
    results_path = f"results/{experiment_id}"
    os.makedirs(results_path,exist_ok=True)

    #get rewards for experiment
    rewards = []
    with gzip.open(f"results/{experiment_id}/rewards.pkl.gz") as fr:
        try:
            while True:
                rewards.extend(pickle.load(fr))
        except EOFError:
            pass
    
    smiles = []
    with gzip.open(f"results/{experiment_id}/smiles.pkl.gz") as fr:
        try:
            while True:
                smiles.extend(pickle.load(fr))
        except EOFError:
            pass
    
    
    rewards = np.array(rewards)
    smiles = np.array(smiles)
    
    start_time = time.time()
    elapsed_time = 0
    convert_indices = np.arange(len(rewards))[rewards>T]
    _,idx = np.unique(smiles[rewards > T], return_index=True)
    # make sure we look at smiles in correct order
    smiles_idx = np.sort(idx)
    smiles = smiles[smiles_idx]
    
    diverse_modes = set()
    diverse_tanimoto = np.zeros(len(rewards))
    logger = logging.getLogger(__name__)

    logger.info(f"Number of unique smiles: {len(smiles)}")
    for i, (smi, smi_idx) in enumerate(zip(smiles,smiles_idx)):
        if is_diverse_tanimoto(smi, diverse_modes, S):
            diverse_modes.add(smi)
            diverse_tanimoto[convert_indices[smi_idx]] = 1
        if i % 100 == 0:
            end_time = time.time()
            elapsed_time += end_time - start_time
            logger.info(i, len(diverse_modes), "Total time is ", elapsed_time, "seconds")
            start_time = end_time
        if i % 1000 == 0 and i != 0:
            pickle.dump({"tanimoto": np.cumsum(diverse_tanimoto),
                 "T": T,
                 "S": S,
                 "iterations": i},
                gzip.open(f"{results_path}/tanimoto_counts.pkl.gz", 'ab'))
    


if __name__ == "__main__":
    experiment_id = "experiment_1"

    T = 7
    S = 0.7
    compute_tanimoto_counts(T, S, experiment_id)
