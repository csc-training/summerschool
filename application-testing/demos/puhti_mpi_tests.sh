#!/bin/bash

#SBATCH --job-name=heateq_test
#SBATCH --account=project_2002078
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=test
#SBATCH --time=00:05:00
#SBATCH --ntasks=4

module load gcc/11.3.0 cmake/3.23.1 openmpi/4.1.4

proj_dir=/scratch/project_2002078/$USER/heat-equation

if [ -d $proj_dir ]; then
	cd $proj_dir/ctest-gtest
	make clean
else
	mkdir $proj_dir
	cp -r $HOME/Code/heat-equation/common $proj_dir/
	cp -r $HOME/Code/heat-equation/ctest-gtest $proj_dir/
fi

## Build
srun \
	--ntasks=1 \
	-D $proj_dir/ctest-gtest \
	make

## Run
srun \
	-D $proj_dir/ctest-gtest/build/Release/tests/mpi_tests/ \
	./heattests_mpi_tests

