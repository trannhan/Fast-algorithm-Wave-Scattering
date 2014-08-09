#!/bin/bash
#PBS -N scat8b
#PBS -q normal
#PBS -l nodes=58:ppn=16:native:noflash
#PBS -l walltime=02:00:00
#PBS -m abe
#PBS -M ttnhan@hotmail.com
#PBS -V
#PBS -v Catalina_maxhops=None
cd $PBS_O_WORKDIR
mkdir -p scat8b
cp scat8b.sh scat8b/
ibrun --npernode 5 ./scat -n 8000000000 -p 8000 -c 27000 -k 0.5 -ksp_cg_type symmetric > scat8b/scat8b.out 2> scat8b/scat8b.err
