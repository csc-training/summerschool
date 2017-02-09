#!/bin/bash
#SBATCH -J my_job_name
#SBATCH -o %J.out
#SBATCH -e %J.err
#SBATCH -n 24
#SBATCH -ptest
#SBATCH -t 5


