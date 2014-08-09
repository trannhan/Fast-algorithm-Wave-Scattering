Compile:
-------------------------------------------------------------------------------------------------------------------------------------
.build
or: 
make


Run:
-------------------------------------------------------------------------------------------------------------------------------------
mpirun -n <number_of_processors> ./<binary_file> -n <size> -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol> -log_summary
or:
mpiexec -n <number_of_processors> ./<binary_file> <options>


Help:
-------------------------------------------------------------------------------------------------------------------------------------
scat -h


Submit job:
-------------------------------------------------------------------------------------------------------------------------------------
qsub run.sh
qsub -q '*@@elf*' scat9a.sh
qsub -q '*@@elves' scat9a.sh

Requirements: 
-------------------------------------------------------------------------------------------------------------------------------------
MPICH, LAPACK, BLAS, PETSC, mpiCC, ACML, FFTW
locate libacml.so


Top iterative methods with respect to least error(from top to bottom):
-------------------------------------------------------------------------------------------------------------------------------------
bcgs, bcgsl,fbcgs
cgs
tfqmr 
cr
gmres, lgmres, fgmres, specest  
cg (runs fastest)
minres
symmlq 


Notes
-------------------------------------------------------------------------------------------------------------------------------------
1. Single precision: use cg
   Double precision: use gmres

2. use #include 'mpif.h' to work with mpif90.gfortran. Do not use "use mpi"

3. Configure PETSC: 
   ./configure --with-scalar-type=complex --with-precision=double --with-pthreadclasses --with-debugging=no --with-fortran-kernels=generic
   ./configure --with-scalar-type=complex --with-precision=single --with-debugging=no '--with-64-bit-indices=1'
python2 './configure' '--with-scalar-type=complex' '--with-precision=double' '--with-debugging=no' '--with-fftw=1' '--download-fftw' '--download-f-blas-lapack' '--with-fortran-kernels=generic'
(--with-scalar-type=real,complex --with-precision=single,double,longdouble,int,matsingle)
(--with-cc=mpicc or --with-mpi-dir [and not --with-cc=gcc])
(--download-fftw or --with-fftw-dir=/path/to/your/fftw3)

4. How can I determine the condition number of a matrix?
   For small matrices, the condition number can be reliably computed using -pc_type svd -pc_svd_monitor. 
   For larger matrices, you can run with -pc_type none -ksp_type gmres -ksp_monitor_singular_value -ksp_gmres_restart 1000 to get approximations 
to the condition number of the operator. This will generally be accurate for the largest singular values, but may overestimate the smallest 
singular value unless the method has converged. Make sure to avoid restarts. To estimate the condition number of the preconditioned operator, 
use -pc_type somepc in the last command.

5. How can I determine the norm of a matrix?
   MatNorm
   -ksp_compute_eigenvalues