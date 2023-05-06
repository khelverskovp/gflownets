import gzip
import pickle
import numpy as np
import time
import os
import logging
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import torch
import matplotlib.pyplot as plt

fingerprints = dict()
diverse_modes = []
representative_mols = []
tanimoto_sims = []

def update_diverse_tanimoto(smi, S, fp_type):
    fps = fingerprints[smi]
    idx = 0
    most_sim = 0
    for i, smi_ref in enumerate(representative_mols):
        fps_ref = fingerprints[smi_ref]
        tanimoto_sim = DataStructs.TanimotoSimilarity(fps,fps_ref)
        if tanimoto_sim >= most_sim:
            idx = i
            most_sim = tanimoto_sim

    # if the most similar mode has a Tanimoto sim of less than S it should be a new mode
    if most_sim < S:
        diverse_modes.append([smi])
        representative_mols.append(smi)
        tanimoto_sims.append(np.ones((1,1)))
    else:
        # add the molecule to the mode with the highest similarity
        diverse_modes[idx].append(smi)
        # update tanimoto similarity
        N = len(diverse_modes[idx])
        new_tanimoto = np.ones((N,N))
        new_tanimoto[:N-1,:N-1] = tanimoto_sims[idx]
        for i in range(N-1):
            fps_ref = fingerprints[diverse_modes[idx][i]]
            tanimoto_sim = DataStructs.TanimotoSimilarity(fps,fps_ref)
            new_tanimoto[i,N-1] = tanimoto_sim
            new_tanimoto[N-1,i] = tanimoto_sim
        tanimoto_sims[idx] = new_tanimoto

        # update representative molecule
        representative_mols[idx] = diverse_modes[idx][np.argmax(np.sum(tanimoto_sims[idx], 0))]

def compute_tanimoto_counts(T, S, experiment_id):
    global fingerprints, diverse_modes, representative_mols, tanimoto_sims
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
<<<<<<< HEAD
=======
    
    # compute all fingerprints
    for smi in smiles:
        if fingerprint_type == "rdk":
            fingerprints[smi] = Chem.RDKFingerprint(Chem.MolFromSmiles(smi))
        elif fingerprint_type == "morgan":
            fingerprints[smi] = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 3, nBits=2048)
        elif fingerprint_type == "bengio":
            fingerprints[smi] = Chem.RDKFingerprint(Chem.MolFromSmiles(smi))
>>>>>>> ab8998a81bf6b767b4865b76e265d3fcdc1f0180

    logger = logging.getLogger(__name__)

    logger.info(f"Number of unique smiles: {len(smiles)}")
    
    # fingerprint types
    fp_types = ["rdk", "morgan"]
    fpsizes = [512,1024,2048]

    tanimoto_counts = dict()

    # compute all fingerprints
    for fp_type in fp_types:
        for fpsize in fpsizes:
            print(f"STARTING {fp_type} with {fpsize} bits")

            # load fngerprints
            for smi in smiles:
                if fp_type == "rdk":
                    fingerprints[smi] = Chem.RDKFingerprint(Chem.MolFromSmiles(smi), fpSize=fpsize)
                elif fp_type == "morgan":
                    fingerprints[smi] = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 3, nBits=fpsize)

            diverse_tanimoto = np.zeros(len(rewards))
            
            for i, (smi, smi_idx) in enumerate(zip(smiles,smiles_idx)):
                old_length = len(diverse_modes)
                update_diverse_tanimoto(smi, S, fp_type)
                diverse_tanimoto[convert_indices[smi_idx]] = len(diverse_modes) > old_length
                if i % 1000 == 0:
                    end_time = time.time()
                    elapsed_time += end_time - start_time
                    logger.info(f"Iteration: {i}, n_modes: {len(diverse_modes)}, Total time is {elapsed_time} seconds")
                    start_time = end_time
            
            tanimoto_counts[f"{fp_type}{fpsize}"] = np.cumsum(diverse_tanimoto)
        
            # reset variables
            fingerprints = dict()
            diverse_modes = []
            representative_mols = []
            tanimoto_sims = []

    pickle.dump({"tanimoto": tanimoto_counts,
                "T": T,
                "S": S},
                gzip.open(f"{results_path}/tanimoto_counts_{T}.pkl.gz", 'wb'))
    


if __name__ == "__main__":
    experiment_id = "experiment_5"

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    T = 7
    S = 0.7
    compute_tanimoto_counts(T, S, "rdk", experiment_id)
    compute_tanimoto_counts(T,S, "rdk","experiment_4")
    compute_tanimoto_counts(T,S,"rdk", "experiment_5")
