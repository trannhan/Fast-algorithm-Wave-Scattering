#!/bin/bash
#PBS -N scat5b
#PBS -q normal
#PBS -l nodes=35:ppn=16:native:noflash
#PBS -l walltime=02:00:00
#PBS -m abe
#PBS -M ttnhan@hotmail.com
#PBS -V
#PBS -v Catalina_maxhops=None
cd $PBS_O_WORKDIR
mkdir -p scat5b
cp scat5b.sh scat5b/
ibrun --npernode 5 ./scat -n 5000000000 -p 8000 -c 27000 -k 0.5 -ksp_cg_type symmetric > scat5b/scat5b.out 2> scat5b/scat5b.err
