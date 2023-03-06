#!/bin/bash
#PBS -P v45
#PBS -q normal
#PBS -M john.reilly@utas.edu.au
#PBS -m abe
#PBS -l ncpus=8
#PBS -l mem=192GB
#PBS -l jobfs=100GB
#PBS -l walltime=2:00:00
#PBS -l software=python
#PBS -l wd
#PBS -l storage=gdata/ik11+gdata/v45+gdata/hh5+gdata/cj50+gdata/ua8+scratch/v45
#PBS -j oe

# Load conda environment
module unload conda
module use /g/data/hh5/public/modules
module load conda/analysis3

# set filename of main code
filecode=gen_segments

# run python application
python3 -u $filecode.py > $PBS_JOBID.log

