Codebase for Bachelor Project about Generative Flow Networks (GFlowNets)
==============================
By Kasper Helverskov Petersen and Marcus Roberto Nielsen

Implementation of GFlowNet for our Bachelor Project at DTU. Our code is inspired by the implementation in [GFNOrg/gflownet](GFNOrg/gflownet) by @bengioe, @MJ10 and @MKorablyov for the initial [paper](https://arxiv.org/abs/2106.04399) about GFlowNets.

## Molecule Experiments
-----------
For the experiments we used Python 3.10.2 and CUDA 11.6. If you have CUDA 11.6 configured, you can run pip install -r requirements.txt. You can also change requirements.txt to match your CUDA version. (Replace cu116 to cuXXX, where XXX is your CUDA version).

The compressed dataset consisting of 300k molecules can be found in `data/raw`

## Project Organization
------------

    ├── data
    │   ├── processed      <- Rewards from the proxy dataset
    │   └── raw            <- The original data in docked_mols.h5. json file for blocks
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── iterations         <- experiments with toymodels
    │
    ├── models             
    │   ├── pretrained_proxy <- pretrained model parameters for proxy
    │   └── runs           <- fully trained models for each experiment
    │
    ├── reports            
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── results            <- Logged results for each experiment run (e.g rewards and losses during a run)   
    ├── src                <- Source code used in this project. (most important files are shown below)
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or preprocess data
    │   │
    │   ├── models         
    │   │   │
    │   │   ├── model.py   <- defining GFlowNet model class
    │   │   └── train_model.py <- training loop of GFlowNet
    │   │
    │   ├── utils          <- helper functions
    │   │   │                
    │   │   ├── chem.py    <- onehot encoding, BlockMolecule to rdkit format, conversion to pytorch geometric format
    │   │   ├── mols.py    <- creating a BlockMolecule class to define molecules, adding and removing blocks, functions to compute parent states
    │   │   ├── plots.py    <- plot code
    │   │   └── proxy.py    <- define proxy model class, needed to load pretrained proxy
    │   │
    │   └── visualization  <- code to visualize various results or molecules          
    │
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data`
    ├── README.md          <- The top-level README for developers using this project.
    ├── jobscript.sh       <- batch script to run on hpc
    ├── requirements.txt   <- The requirements file for virtual environment
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>