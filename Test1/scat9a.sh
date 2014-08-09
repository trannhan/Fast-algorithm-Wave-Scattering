#!/bin/bash
#PBS -N scat9a
#PBS -q normal
#PBS -l nodes=8:ppn=16:native:noflash
#PBS -l walltime=00:10:00
#PBS -m abe
#PBS -M ttnhan@hotmail.com
#PBS -V
#PBS -v Catalina_maxhops=None
cd $PBS_O_WORKDIR
mkdir -p scat9a
cp scat9a.sh scat9a/
mpirun_rsh -hostfile $PBS_NODEFILE -np 128 ./scat -n 1000000000 -p 8000 -c 64000 -k 0.5 -view_solution > scat9a/scat9a.out 2> scat9a/scat9a.err
