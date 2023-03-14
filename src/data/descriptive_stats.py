# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    # dockscore
    """ print("-------------")
    print(f"| DOCKSCORE |")
    print("-------------")
    print(f'Mean: {np.mean(df.dockscore)}')
    print(f'Standard deviation: {np.std(df.dockscore)}')
    print(f'Min.: {np.min(df.dockscore)}')
    print(f'Max.: {np.max(df.dockscore)}')

    print("")

    print(f"Number of datapoints in [-16.00; 0[: {np.sum((df.dockscore >= -17) & (df.dockscore < 0))}")
    print(f"Number of datapoints in [0; 10[: {np.sum((df.dockscore >= 0) & (df.dockscore < 10))}")
    print(f"Number of datapoints in [10; 20[: {np.sum((df.dockscore >= 10) & (df.dockscore < 20))}")
    print(f"Number of datapoints in [20; 30[: {np.sum((df.dockscore >= 20) & (df.dockscore < 30))}")
    print(f"Number of datapoints in [30; 200[: {np.sum((df.dockscore >= 30) & (df.dockscore < 200))}")
    print(f"Number of datapoints in [200; 400[: {np.sum((df.dockscore >= 200) & (df.dockscore < 400))}")

    print("")


    # print length of dataset
    print(f"Length of dataset: {len(df['smiles'])}")

    
    print("")

    # indices of lowest and highest scoring molecule
    print(f"Lowest scoring molecule has index: {np.argmin(df.dockscore)}")
    print(f"Highest scoring molecule has index: {np.argmax(df.dockscore)}") """

    """ # blockidxs and slices
    # sizes = np.array([len(bids) for bids in df.blockidxs])
    sizes = np.array([s[-1] for s in df.slices])
    minlength = np.min(sizes)
    maxlength = np.max(sizes)

    print("-------------")
    # print(f"| Number of blocks |")
    print(f"| Number of non-hydrogen atoms |")
    print("-------------")

    print(f'Mean: {np.mean(sizes)}')
    print(f'Standard deviation: {np.std(sizes)}')
    print(f'Min.: {minlength}')
    print(f'Max.: {maxlength}')

    # count number of data points of different number of blocks
    occs = range(minlength,maxlength+1)
    counts = [sum(sizes == i) for i in occs]
    
    plt.figure(figsize=(20,16))

    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('axes', labelsize=20)
    plt.rc('axes', titlesize=25)

    plt.bar(occs,counts)
    # plt.xticks(occs)
    
    # plt.xlabel("Number of blocks")
    plt.xlabel("Number of non-hydrogen atoms")
    plt.ylabel("Number of molecules")

    # plt.title("Distribution of number of blocks in molecules in the ZINC dataset", fontweight="bold")
    plt.title("Distribution of number of non-hydrogen atoms in molecules in the ZINC dataset", fontweight="bold")

    # save
    # path = "reports/figures/molsizedist.png"
    path = "reports/figures/molsizeatomdist.png"
    plt.savefig(path) """
    
    """ negated_dockscore = -1 * df.dockscore
    mean, std = np.mean(negated_dockscore), np.std(negated_dockscore)
    normalized_dockscore = (negated_dockscore - mean) / std

    blocksizes = np.array([len(bids) for bids in df.blockidxs])
    normalized_blocksizes = (blocksizes - np.mean(blocksizes)) / np.std(blocksizes)

    atomsizes = np.array([s[-1] for s in df.slices])
    normalized_atomsizes = (atomsizes - np.mean(atomsizes)) / np.std(atomsizes)
    
    cov = np.cov([normalized_dockscore,normalized_blocksizes,normalized_atomsizes])

    plt.title("Covariance matrix for variables",fontweight="bold")
   
    sns.heatmap(cov, annot=True)
    
    pos = [0.5,1.5,2.5]
    labels = ["Negated dockscore","Number of blocks", "Number of atoms"]

    plt.xticks(pos,labels,rotation=45)
    plt.yticks(pos,labels,rotation=0)

    plt.subplots_adjust(left=0.25, bottom=0.25)

    path = "reports/figures/covmatrix.png"
    plt.savefig(path) """

    # checks for duplicates
    """ unique = len(np.unique(df.smiles))
    duplicates = len(df.smiles) - unique
    print(f"Number of unique molecules: {unique}")
    print(f"Number of duplicate molecules: {duplicates}")

    # check for nans
    nans =  df.isna().sum().sum() 

    print(f"Number of NANS: {nans}") """

    for i in range(len(df.smiles)):
        done = False
        for j in range(i+1,len(df.smiles)):
            if df.smiles[i] == df.smiles[j]:
                print(df.iloc[i])
                print(df.iloc[j])
                break
        if done:
            break




    



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
