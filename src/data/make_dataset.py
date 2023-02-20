# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
import json

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making preprocessed data set from raw data')


    # filename for raw data file
    filename = "docked_mols.h5"
    path = f"{input_filepath}/{filename}"

    # load using pandas
    store = pd.HDFStore(path,"r")
    df = store.select("df")
    
    # important attributes
    # smiles - chemical formula in smile notation
    # dockscore - AutoDock score of the specific molecule
    # blockidxs - index of the block
    # slices
    # jbonds
    # stems
    
    # column names
    columns = ["smiles","dockscore","blockidxs","slices","jbonds","stems"]

    # retrieve row names
    smiles = df.index.to_numpy()

    # change datatype of dockscore to float64
    dockscore = df.dockscore.astype("float64")
    
    # store in nparray to give as input data for data frame
    data = np.array([smiles,dockscore,df.blockidxs,df.slices,df.jbonds,df.stems])

    # put into dataframe
    df_processed = pd.DataFrame(data.T,columns=columns)
    
    # save to a csv file
    output_filename = "docked_mols.csv"
    df_processed.to_csv(f"{output_filepath}/{output_filename}",index=False)

    # close file
    store.close()
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
