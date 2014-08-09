#!/bin/bash
#PBS -N scat2b
#PBS -q normal
#PBS -l nodes=15:ppn=16:native:noflash
#PBS -l walltime=01:00:00
#PBS -m abe
#PBS -M ttnhan@hotmail.com
#PBS -V
#PBS -v Catalina_maxhops=None
cd $PBS_O_WORKDIR
mkdir -p scat2b
cp scat2b.sh scat2b/
ibrun --npernode 8 ./scat -n 2000000000 -p 8000 -c 27000 -k 0.5 -ksp_cg_type symmetric -view_solution > scat2b/scat2b.out 2> scat2b/scat2b.err
