# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
import json

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

    

    # print descriptive statistics
    
    # print length of dataset
    print(f"Length of dataset: {len(df['smiles'])}")
    
    # print attributes
    print(f"Attribute names: {df.columns.values}")

    #print(df.iloc[4])

    # print specific data point
    #print(df.iloc[316932])
    #print(df.iloc[316933])

    #print(sum(df["dockscore"][df["dockscore"] <= 0]) / len(df["dockscore"]))
    """ done = False
    for i, (bids, jbs) in enumerate(zip(df["blockidxs"],df["jbonds"])):
        if 33 in bids:
            bidx = bids.index(33)
            for jb in jbs:
                if (jb[0] == bidx and jb[2] == 6) or (jb[1] == bidx and jb[3] == 6):
                    print(df["smiles"][i])
                    print(i)
                    done = True
                    break
        if done:
            break """
    
    smiles = []
    blocks = pd.read_json("data/processed/blocks_PDB_105.json")
    block_smi = blocks["block_smi"].to_list()
    block_rs = blocks["block_r"].to_list()

    count = 0
    print(len(np.unique(block_smi)))
    for i in range(len(block_smi)):
        count += len(np.unique(block_rs[i]))
        print(block_smi[i],block_rs[i])
        smiles.append(block_smi[i])
    
    print(count)
        



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
