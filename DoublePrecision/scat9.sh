#!/bin/bash
#PBS -N scat9
#PBS -q normal
#PBS -l nodes=15:ppn=16:native
#PBS -l walltime=1:00:00
#PBS -m abe
#PBS -M ttnhan@hotmail.com
#PBS -V
#PBS -v Catalina_maxhops=None
cd $PBS_O_WORKDIR
mpirun_rsh -hostfile $PBS_NODEFILE -np 240 ./scat -n 1000000000 -p 8000 -c 27000 -k 0.5 -log_summary
