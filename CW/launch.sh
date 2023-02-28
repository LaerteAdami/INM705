#!/bin/bash
#SBATCH --mail-user=laerte.adami@city.ac.uk       # useful to get an email when jobs starts
#SBATCH --mail-type=BEGIN
#SBATCH --job-name=my-jupyter                     # Job name
#SBATCH --partition=gengpu                        # Select the correct partition.
#SBATCH --nodes=1                                 # Run on 2 nodes (each node has 48 cores)
#SBATCH --ntasks-per-node=2
#SBATCH --mem=32GB                                # Expected memory usage (0 means use all available memory)
#SBATCH --time=72:00:00                           # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                              # Use one gpu.
#SBATCH -e jupyterURL                             # This file will contain the URL to connect to jupyter
#SBATCH -o nodename                               # This file will contain the hostname of the node needed to create tunnel
   
source /opt/flight/etc/setup.sh
flight env activate gridware
module add aij
#Write node hostname into nodename file
echo $SLURM_JOB_NODELIST
#Run jupyter lab
papermill INM705_notebook.ipynb 28_02_fcn_10classes.ipynb