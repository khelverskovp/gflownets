# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

@click.command()
@click.argument('input_filepath', default="data/processed" ,type=click.Path(exists=True))
def main(input_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    # load data points
    filename = "docked_mols.csv"
    path = f"{input_filepath}/{filename}"

    df = pd.read_csv(path)

    columns = df.columns
    
    # the remaining columns contains lists of numbers
    # they are in string form however in the native dataframe
    # should be converted to list type 
    for name in columns[2:]:
        df.loc[:,name] = df[name].apply(json.loads)

    bpath = "data/raw/blocks_PDB_105.json"
    blocks = pd.read_json(bpath)

    # load smile names and attachment points
    block_rs = blocks["block_r"].to_list()

    print(np.mean(-df.dockscore*((-df.dockscore) > 0)))
    print(np.std(-df.dockscore*((-df.dockscore) > 0)))

    """ for i in range(len(df.smiles)):
        done = False
        for j in range(i+1,len(df.smiles)):
            if df.smiles[i] == df.smiles[j]:
                print(df.iloc[i])
                print(df.iloc[j])
                done = True
                break
        if done:
            break """
    """ df.dockscore[df.dockscore > 0] = 0
    print(np.mean(df.dockscore[df.dockscore < 0]), np.std(df.dockscore[df.dockscore < 0]))
    print(df.iloc[59])
    print(df.iloc[605]) """
    """ target_norm = [-8.6,1.1]
    temp = np.copy(df.dockscore)
    temp[temp > 0] = 0
    vals = 4-(temp-target_norm[0])/target_norm[1]
    print(np.mean(4-(temp-target_norm[0])/target_norm[1]))
    print(np.sum(vals >= 8))

    dtrue = {key: 0 for key in range(1,13)}
    for blockidx in df.blockidxs:
        dtrue[len(blockidx)] += 1

    d = {key: 0 for key in range(1,13)}
    for blockidx in df.blockidxs[vals >= 7.5]:
        d[len(blockidx)] += 1
    
    print("Total:",dtrue)
    print("Total with reward > 7.5:", d)

    d = {key: round(val / dtrue[key] * 100,3) for key, val in d.items()}

    print("Percentages:",d)

    pdf, bins = np.histogram(vals, density=True)
    pdf = pdf / np.sum(pdf)

    # Plot the empirical PDF
    plt.plot(bins[:-1], pdf, 'b', linewidth=2)
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.title('Empirical PDF')
    plt.show() """

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
