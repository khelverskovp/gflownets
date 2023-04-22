#!/bin/sh
### General options
### –- specify queue --
# BSUB -q gpuv100
### -- set the job Name --
# BSUB -J gflowjob
### -- ask for number of cores (default: 1) --
# BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
# BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
# BSUB -W 1:10
# request 32GB of system-memory
# BSUB -R "rusage[mem=32GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o outputs_hpc/gpu_%J.out
#BSUB -e outputs_hpc/gpu_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/11.6

# /appl/cuda/10.1.0/samples/bin/x86_64/linux/release/deviceQuery

# activate virtual environment
source /zhome/45/e/155478/gflownets/gflowenv/bin/activate
### source /zhome/7a/6/154675/gflowenv/bin/activate

python3 src/models/train_model.py

bsub < jobscript.sh
