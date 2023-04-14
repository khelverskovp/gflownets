import pickle
import torch
import gzip
import time
import numpy as np


if __name__ == "__main__":
    #To load from pickle file
    data = []
    with gzip.open("results/experiment_1/losses.pkl.gz") as fr:
        try:
            while True:
                data.append(pickle.load(fr))
        except EOFError:
            pass

    
    print(data)
    