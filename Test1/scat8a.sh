#!/bin/bash
#PBS -N scat8a
#PBS -q normal
#PBS -l nodes=2:ppn=16:native:noflash
#PBS -l walltime=00:05:00
#PBS -m abe
#PBS -M ttnhan@hotmail.com
#PBS -V
#PBS -v Catalina_maxhops=None
cd $PBS_O_WORKDIR
mkdir -p scat8a
cp scat8a.sh scat8a/
ibrun -N 2 -npernode 10 ./scat -n 100000000 -p 8000 -c 27000 -k 0.5 -ksp_cg_type symmetric -ksp_monitor -log_summary -malloc_log > scat8a/scat8a.out 2> scat8a/scat8a.err
