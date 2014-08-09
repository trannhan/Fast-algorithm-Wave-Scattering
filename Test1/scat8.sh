#!/bin/bash
#PBS -N scat8
#PBS -q normal
#PBS -l nodes=2:ppn=16:native
#PBS -l walltime=00:05:00
#PBS -m abe
#PBS -M ttnhan@hotmail.com
#PBS -V
#PBS -v Catalina_maxhops=None
cd $PBS_O_WORKDIR
mkdir -p scat8
cp scat8.sh scat8/
mpirun_rsh -hostfile $PBS_NODEFILE -np 32 ./scat -n 100000000 -p 8000 -c 27000 -k 0.5 -ksp_cg_type symmetric -ksp_monitor -memory_info -malloc > scat8/scat8.out 2> scat8/scat8.err
