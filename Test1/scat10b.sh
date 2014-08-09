#!/bin/bash
#PBS -N scat10b
#PBS -q normal
#PBS -l nodes=74:ppn=16:native:noflash
#PBS -l walltime=02:00:00
#PBS -m abe
#PBS -M ttnhan@hotmail.com
#PBS -V
#PBS -v Catalina_maxhops=None
cd $PBS_O_WORKDIR
mkdir -p scat10b
cp scat10b.sh scat10b/
ibrun --npernode 5 ./scat -n 10000000000 -p 8000 -c 64000 -k 0.5 -ksp_cg_type symmetric > scat10b/scat10b.out 2> scat10b/scat10b.err
