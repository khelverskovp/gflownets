# this file serves to describe preliminary statistics about the dataset


# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import h5py
import numpy as np

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


    # filename for raw data file
    filename = "docked_mols.h5"
    path = f"{input_filepath}/{filename}"

    # read h5 file
    with h5py.File(path, "r") as f:
        # Print all root level object names (aka keys) 
        print(f"Keys: {f.keys()}\n")
        
        # Retrieve header objects from dataset
        key = list(f.keys())[0]
        data = f[key]
        headers = list(data)
        
        # print headers
        print(f"Dataframe headers: {headers}")
        
        for header in headers:
            # retrieve data points for each category
            datapoints = data[header][:] if header != "block1_values" else data[header][:][0]

            print(f"\nHeader: {header}\nShape: {datapoints.shape}")

            print(f"Data: \n{datapoints}")
        
        
        
        


    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
