#!/bin/bash
#PBS -N scat10
#PBS -q normal
#PBS -l nodes=60:ppn=16:native:noflash
#PBS -l walltime=12:00:00
#PBS -m abe
#PBS -M ttnhan@hotmail.com
#PBS -V
#PBS -v Catalina_maxhops=None
cd $PBS_O_WORKDIR
mpirun_rsh -hostfile $PBS_NODEFILE -np 960 ./scat -n 10000000000 -p 27000 -c 125000 -k 0.5 -log_summary
