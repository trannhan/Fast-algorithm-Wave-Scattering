#!/bin/bash
#PBS -N scat3b
#PBS -q normal
#PBS -l nodes=18:ppn=16:native:noflash
#PBS -l walltime=03:00:00
#PBS -m abe
#PBS -M ttnhan@hotmail.com
#PBS -V
#PBS -v Catalina_maxhops=None
cd $PBS_O_WORKDIR
mkdir -p scat3b
cp scat3b.sh scat3b/
ibrun --npernode 5 ./scat -n 3000000000 -p 8000 -c 27000 -k 0.5 -ksp_cg_type symmetric > scat3b/scat3b.out 2> scat3b/scat3b.err
