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
    logger.info('making preprocessed data set from raw data')


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
    """ print("Dockscore statistics:")
    print(f"Min. value: {np.min(df['dockscore'])}")
    print(f"Max. value: {np.max(df['dockscore'])}")
    print(f"Mean: {np.mean(df['dockscore'])}")
    print(f"Standard deviation: {np.std(df['dockscore'])}") """

    print(np.max(df["blockidxs"]))




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
