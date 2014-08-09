#PBS -q vsmp
#PBS -l nodes=1:ppn=16:vsmp
#PBS -N scat
#PBS -o scat.out
#PBS -e scat.err
#PBS -M ttnhan@gmail.com
#PBS -m abe
#PBS -V
export PATH=/opt/ScaleMP/mpich2/1.3.2/bin:$PATH
export VSMP_PLACEMENT=PACKED
export VSMP_VERBOSE=YES
export VSMP_MEM_PIN=YES
vsmputil --unpinall
cd /home/nhantran/work/ScatFFT64
time mpirun -np 64 ./scat -n 1000000000 -k 0.5 -p 8000 -c 27000 -log_summary
