#!/bin/bash
#PBS -N scat9b
#PBS -q normal
#PBS -l nodes=64:ppn=16:native:noflash
#PBS -l walltime=02:00:00
#PBS -m abe
#PBS -M ttnhan@hotmail.com
#PBS -V
#PBS -v Catalina_maxhops=None
cd $PBS_O_WORKDIR
mkdir -p scat9b
cp scat9b.sh scat9b/
ibrun --npernode 5 ./scat -n 9000000000 -p 8000 -c 27000 -k 0.5 -ksp_cg_type symmetric > scat9b/scat9b.out 2> scat9b/scat9b.err
