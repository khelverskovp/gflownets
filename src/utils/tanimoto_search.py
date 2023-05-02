import gzip
import pickle
import numpy as np
import time
import os
import logging
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import torch

fingerprints = dict()
diverse_modes = []
representative_mols = []
tanimoto_sims = []

fpe = [None]
FP_CONFIG = {
    "mol_fp_len": 512,
    "mol_fp_radiis": [3],
    "stem_fp_len": 64,
    "stem_fp_radiis": [4, 3, 2]
}

# copied directly from initial git repo
class FPEmbedding_v2:
    def __init__(self, mol_fp_len, mol_fp_radiis, stem_fp_len, stem_fp_radiis):
        self.mol_fp_len = mol_fp_len
        self.mol_fp_radiis = mol_fp_radiis
        self.stem_fp_len = stem_fp_len
        self.stem_fp_radiis = stem_fp_radiis

    def __call__(self, molecule):
        mol = molecule.mol
        mol_fp = get_fp(mol, self.mol_fp_len, self.mol_fp_radiis)

        # get fingerprints and also handle empty case
        stem_fps = [get_fp(mol, self.stem_fp_len, self.stem_fp_radiis, [idx])
                    for idx in molecule.stem_atmidxs]

        jbond_fps = [(get_fp(mol, self.stem_fp_len, self.stem_fp_radiis, [idx[0]]) +
                     get_fp(mol, self.stem_fp_len, self.stem_fp_radiis, [idx[1]]))/2.
                     for idx in molecule.jbond_atmidxs]

        if len(stem_fps) > 0:
            stem_fps = np.stack(stem_fps, 0)
        else:
            stem_fps = np.empty(shape=[0, self.stem_fp_len * len(self.stem_fp_radiis)], dtype=np.float32)
        if len(jbond_fps) > 0:
            jbond_fps = np.stack(jbond_fps, 0)
        else:
            jbond_fps = np.empty(shape=[0, self.stem_fp_len * len(self.stem_fp_radiis)], dtype=np.float32)
        return mol_fp, stem_fps, jbond_fps


def mol2fp(mol, mdp):
    if fpe[0] is None:
        fpe[0] = chem.FPEmbedding_v2(
            FP_CONFIG['mol_fp_len'],
            FP_CONFIG['mol_fp_radiis'],
            FP_CONFIG['stem_fp_len'],
            FP_CONFIG['stem_fp_radiis'])
    # ask for non-empty stem and bond embeddings so that they at least
    # have shape (1, n), rather than (0, n) if there are not stems/bonds
    return list(map(torch.tensor,fpe[0](mol, non_empty=True)))  # mol_fp, stem_fps, jbond_fps

def update_diverse_tanimoto(smi, S):
    fps = fingerprints[smi]
    idx = 0
    most_sim = 0
    for i, smi_ref in enumerate(representative_mols):
        fps_ref = fingerprints[smi_ref]
        tanimoto_sim = DataStructs.FingerprintSimilarity(fps,fps_ref)
        #tanimoto_sim = DataStructs.TanimotoSimilarity(fps,fps_ref)
        if tanimoto_sim >= most_sim:
            idx = i
            most_sim = tanimoto_sim
    if most_sim != 0:
        print(most_sim)
        raise Exception
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
            #tanimoto_sim = DataStructs.FingerprintSimilarity(fps,fps_ref)
            tanimoto_sim = DataStructs.TanimotoSimilarity(fps,fps_ref)
            new_tanimoto[i,N-1] = tanimoto_sim
            new_tanimoto[N-1,i] = tanimoto_sim
        tanimoto_sims[idx] = new_tanimoto

        # update representative molecule
        representative_mols[idx] = diverse_modes[idx][np.argmax(np.sum(tanimoto_sims[idx], 0))]

def compute_tanimoto_counts(T, S, fingerprint_type, experiment_id):
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
    
    # compute all fingerprints
    for smi in smiles:
        if fingerprint_type == "rdk":
            fingerprints[smi] = Chem.RDKFingerprint(Chem.MolFromSmiles(smi))
        elif fingerprint_type == "morgan":
            fingerprints[smi] = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 3, nBits=2048)
        elif fingerprint_type == "bengio":
            fingerprints[smi] = 

    diverse_tanimoto = np.zeros(len(rewards))
    logger = logging.getLogger(__name__)

    logger.info(f"Number of unique smiles: {len(smiles)}")
    for i, (smi, smi_idx) in enumerate(zip(smiles,smiles_idx)):
        old_length = len(diverse_modes)
        update_diverse_tanimoto(smi, S)
        diverse_tanimoto[convert_indices[smi_idx]] = len(diverse_modes) > old_length
        if i % 1000 == 0:
            end_time = time.time()
            elapsed_time += end_time - start_time
            logger.info(f"{i} {len(diverse_modes)} Total time is {elapsed_time} seconds")
            start_time = end_time
        if i % 1000 == 0 and i != 0:
            pickle.dump({"tanimoto": np.cumsum(diverse_tanimoto),
                 "T": T,
                 "S": S,
                 "iterations": i},
                gzip.open(f"{results_path}/tanimoto_counts_{T}.pkl.gz", 'wb'))
    
    pickle.dump({"tanimoto": np.cumsum(diverse_tanimoto),
                 "T": T,
                 "S": S,
                 "iterations": i},
                gzip.open(f"{results_path}/tanimoto_counts_{T}.pkl.gz", 'wb'))
    


if __name__ == "__main__":
    experiment_id = "experiment_1"

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    T = 7
    S = 0.7
    compute_tanimoto_counts(T, S, experiment_id)
