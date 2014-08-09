/* Program usage:  mpiexec -n <procs> scat [-help] [all PETSc options] */

static char help[] = "Solves particle-scattering problem in parallel.\n\
References:\n\
[595]  A.G. Ramm,  Wave scattering by many small bodies and creating materials with a desired refraction coefficient, Afrika Matematika, 22, N1, (2011), 33-55.\n\
Input parameters include:\n\
  -a <particle_radius>         : particle radius\n\
  -d <particles_distance>      : distance between neighboring particles, 1 >> d >> a, default value cuberoot[a^(2-Kappa)]\n\
  -n <total_particles>         : total number of particles, n = O(1/(a^(2-Kappa)))\n\
  -p <total_cubes>             : total number of embeded small cubes in the domain containing all particles (for solving the reduced system)\n\
  -c <total_collocation_points>: total number of collocation points in the domain containing all particles (for solving the integral equation)\n\
  -k <kappa>                   : power constant with respect to the radius of particles, kappa is in [0,1), default value 0.99\n\
  -vol <volume>                : volume of the domain that contains all particles, default value 1\n\
  -ori <original_refraction>   : original refraction coefficient, default value 1\n\
  -des <desired_refraction>    : desired refraction coefficient, default value sqrt(0.2)\n\
  -dis <distribution>          : distribution of particles, default value 1 for uniform distribution\n\
  -view_RHS                    : write RHS vector to stdout\n\
  -view_solution               : write solution vector to stdout\n\
  -standard                    : solve the original system in the standard way, not using convolution\n\n";

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h    - base PETSc routines   petscvec.h - vectors
     petscmat.h    - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/

//#include <petscksp.h>
#include "scattering3DS.h"
#include "scattering3DP.h"
#include "scatteringIE_Riemann.h"

using namespace std;

#define NORMTYPE NORM_INFINITY

extern inline PetscErrorCode ScatSMatMultFFT(Mat,Vec,Vec);
extern inline PetscErrorCode ScatPMatMult(Mat,Vec,Vec);
extern inline PetscErrorCode ScatIEMatMult(Mat,Vec,Vec);
extern PetscErrorCode DiffP(Vec,Vec,Vec,Vec);
extern PetscErrorCode DiffPCollocation(Vec,Vec,Vec,Vec);
extern PetscErrorCode DiffIE(Vec,Vec,Vec,Vec);
extern inline PetscErrorCode FFTPaddedGreenCube(PetscInt[]);
extern inline PetscErrorCode FFTInit();
extern inline PetscErrorCode FFTFree();
extern inline PetscErrorCode VecCopyN1(Vec,Vec,PetscInt);
extern void GetTime(bool);


Scattering3DS<std::vector<PetscReal>,std::vector<PetscScalar>,PetscScalar,PetscReal,PetscInt,PetscInt> 		Scat3DS;
Scattering3DP<std::vector<PetscReal>,std::vector<PetscScalar>,PetscScalar,PetscReal,PetscInt,PetscInt> 		Scat3DP;
ScatteringIE_Riemann<std::vector<PetscReal>,std::vector<PetscScalar>,PetscScalar,PetscReal,PetscInt,PetscInt> 	ScatIE;
PetscLogDouble  mem;

//FFT:
Vec 	      RealSpaceCube;
fftwf_plan    fplan,bplan;
PetscInt      dim[3];
fftwf_complex *data_in;
ptrdiff_t     alloc_local,local_n0,local_0_start;

//MPI:
int           rank;
int           size;
time_t        time1, time2;  //Time usage 


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Vec            FFTx,y,z,u0,u0P,u0IE,Distance,nParticlePerCube;          //approx solution, RHS, solution-difference 
  //Vec          FFTr,r,rP,rIE;                                           //residual 
  Mat            B,IE,AFFT;           		                          //linear system matrix 
  KSP            kspP,kspIE,kspFFT;      	                  	  //linear solver context 
  //PC           pc;          		                                  //Preconditioner context 
  PetscReal      norm,RHSnorm, Tol=1e-3, aTol;          		           
  PetscInt       its;           		                          //number of iterations reached 
  PetscInt       Istart,Iend;
  PetscErrorCode ierr;
  KSPConvergedReason reason;
  KSPType 	 ksptype;
  PetscScalar    *xa;
  PetscBool      flg = PETSC_FALSE;
  PetscViewer    viewer;
  
  //Scattering3D:
  PetscInt       M = 0;                 			  	   //Total particles
  PetscReal      a = 0;                 			  	   //Particle radius
  PetscReal      ParticleDistance = 0;
  PetscReal      VolQ = 0;              			  	   //Volume of the domain Q that contains all particles
  PetscReal      Kappa = -1;             			  	   //Power const with respect to the radius of particles: Kappa in [0,1)
  std::vector<PetscReal> WaveDirection(3,0);WaveDirection[0] = 1;          // WaveDirection is a unit vector that indicates the direction of plane wave
  PetscScalar    OriginalRefractionCoef = 0;
  PetscScalar    DesiredRefractionCoef = 0;
  PetscReal      Distribution = 0;
  PetscInt       TotalCubes = 0;
  PetscInt       N = 0;                 			  	   //Number of sub domains used for solving IE


#if defined(PETSC_USE_LOG)
  PetscLogStage  stage;
#endif    

  PetscInitialize(&argc,&args,(char *)0,help);
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);
  //printf ("Hello from task %d! Total cores: %d\n",rank,size);

  time(&time1);

  ierr = PetscOptionsGetReal(PETSC_NULL,"-a",&a,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-d",&ParticleDistance,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-p",&TotalCubes,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-c",&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-k",&Kappa,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-vol",&VolQ,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(PETSC_NULL,"-ori",&OriginalRefractionCoef,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(PETSC_NULL,"-des",&DesiredRefractionCoef,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-dis",&Distribution,PETSC_NULL);CHKERRQ(ierr);

  //Scatering init 
  ValidateInput(Kappa,VolQ,a,ParticleDistance,M,OriginalRefractionCoef,DesiredRefractionCoef,Distribution,TotalCubes,N);
  Scat3DS.Input(a,Kappa,WaveDirection,ParticleDistance,M,OriginalRefractionCoef,DesiredRefractionCoef,Distribution,VolQ);
  Scat3DP.Input(a,Kappa,WaveDirection,ParticleDistance,M,OriginalRefractionCoef,DesiredRefractionCoef,Distribution,VolQ,TotalCubes,N); 
  ScatIE.Input(a,Kappa,WaveDirection,ParticleDistance,M,OriginalRefractionCoef,DesiredRefractionCoef,Distribution,VolQ,N); 
  Scat3DS.Init();
  Scat3DP.Init();
  ScatIE.Init();
  if(rank==0)
  {
      Output<std::vector<PetscReal>,PetscScalar,PetscReal,PetscInt,PetscInt>(Kappa,VolQ,a,ParticleDistance,M,WaveDirection,\
                                                                             OriginalRefractionCoef,DesiredRefractionCoef,Distribution,TotalCubes,Scat3DS.BoundaryImpedance,N);
  }

  PetscPrintf(PETSC_COMM_WORLD,"\nInitializing done:");
  ierr = PetscMemoryGetCurrentUsage(&mem);CHKERRQ(ierr);PetscPrintf(PETSC_COMM_WORLD,"\tMem used: %G \t",mem);GetTime(0);

  //----------------------------------------SET UP FFTW FOR SYSTEMS S---------------------------------------------------
  ierr = PetscLogStageRegister("Conv FFT", &stage);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr); 

  dim[0]=2*Scat3DS.NumParticlePerSide-2;
  dim[1]=dim[0];
  dim[2]=dim[0];

  fftwf_mpi_init();
  ierr = FFTInit();CHKERRQ(ierr);

  ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,M,M,(void*)0,&AFFT);CHKERRQ(ierr);
  ierr = MatSetUp(AFFT);CHKERRQ(ierr);
  ierr = MatShellSetOperation(AFFT,MATOP_MULT,(void(*)(void))ScatSMatMultFFT);CHKERRQ(ierr);

  //ierr = MatSetOption(AFFT,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);

  //Setup RHS:
  ierr = VecCreate(PETSC_COMM_WORLD,&u0);CHKERRQ(ierr);
  ierr = VecSetSizes(u0,PETSC_DECIDE,M);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u0);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(u0,&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetArray(u0,&xa);CHKERRQ(ierr);
  for(PetscInt s = Istart; s <Iend ; s++)
  {
      xa[s-Istart] = Scat3DS.InitField(s);      
  }
  ierr = VecRestoreArray(u0,&xa);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Setting up RHS done:");
  PetscMemoryGetCurrentUsage(&mem);PetscPrintf(PETSC_COMM_WORLD,"\tMem used: %G \t",mem);GetTime(0);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-view_RHS",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nVector Right-hand-side of the original linear system:\n");CHKERRQ(ierr);
    ierr = VecView(u0,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);    
  }

  //Setup vector solution:
  ierr = VecDuplicate(u0,&FFTx);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&kspFFT);CHKERRQ(ierr);
  ierr = KSPSetOperators(kspFFT,AFFT,AFFT,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetType(kspFFT,KSPCG);CHKERRQ(ierr);
  ierr = KSPSetTolerances(kspFFT,Tol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(kspFFT);CHKERRQ(ierr);	

  ierr = KSPGetType(kspFFT,&ksptype);CHKERRQ(ierr);
  ierr = KSPGetTolerances(kspFFT,&Tol,&aTol,&norm,&its);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"\nComputing:\nTolerance:\t%G\nMethod:\t\t%s\n",Tol,ksptype);
  ierr = KSPSolve(kspFFT,u0,FFTx);CHKERRQ(ierr);

  //ierr = VecDuplicate(FFTx,&FFTr);CHKERRQ(ierr);
  //ierr = MatMult(AFFT,FFTx,FFTr); CHKERRQ(ierr);
  //ierr = VecAXPY(FFTr,-1.0,u0);CHKERRQ(ierr);
  //ierr = VecNorm(FFTr,NORMTYPE,&norm);CHKERRQ(ierr);
  ierr = VecNorm(u0,NORM_2,&RHSnorm);CHKERRQ(ierr);
  ierr = KSPGetResidualNorm(kspFFT,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(kspFFT,&its);CHKERRQ(ierr);
  //norm *= sqrt(1.0/M); //Scale the norm

  ierr = KSPGetConvergedReason(kspFFT,&reason);CHKERRQ(ierr);
  if (reason<0) {
     ierr = PetscPrintf(PETSC_COMM_WORLD,"\nCAUTION: The original linear system is divergent!");CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSolving the original linear system (FFT):\nNorm of r=b-Ax:\t\t%G\nRelative error |r|/|b|:\t%G\nIterations:\t\t%D\n",norm,norm/RHSnorm,its);CHKERRQ(ierr);

  GetTime(0);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-view_solution",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nVector solution of the original linear system (FFT):");CHKERRQ(ierr);
    //ierr = VecView(FFTx,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    //Write output file with PETSC_VIEWER_BINARY_MATLAB format:
    //NOTE: the output generated with this viewer can be loaded into MATLAB using $PETSC_DIR/bin/matlab/PetscReadBinaryMatlab.m
    //ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"FFTx.output",&viewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"FFTx.bin",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
    ierr = VecView(FFTx,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," Wrote to FFTx.bin\n");CHKERRQ(ierr);
  }

  ierr = PetscLogStagePop();CHKERRQ(ierr); 

  ierr = FFTFree();CHKERRQ(ierr);
  //ierr = VecDestroy(&u0);CHKERRQ(ierr);
  //ierr = VecDestroy(&FFTr);CHKERRQ(ierr);
  ierr = MatDestroy(&AFFT);CHKERRQ(ierr);
  ierr = KSPDestroy(&kspFFT);CHKERRQ(ierr);
 
  //----------------------------------------SOLVING THE REDUCED SYSTEM P-------------------------------------------------------
  ierr = PetscLogStageRegister("Reduced system", &stage);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);

  ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,TotalCubes,TotalCubes,(void*)0,&B);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  ierr = MatShellSetOperation(B,MATOP_MULT,(void(*)(void))ScatPMatMult);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&u0P);CHKERRQ(ierr);
  ierr = VecSetSizes(u0P,PETSC_DECIDE,TotalCubes);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u0P);CHKERRQ(ierr);
  ierr = VecDuplicate(u0P,&y);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(u0P,&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetArray(u0P,&xa);CHKERRQ(ierr);
  for(PetscInt s = Istart; s <Iend ; s++)
  {
      xa[s-Istart] = Scat3DP.InitField(s);      
  }
  ierr = VecRestoreArray(u0P,&xa);CHKERRQ(ierr);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-view_RHS",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nVector Right-hand-side of the reduced linear system:\n");CHKERRQ(ierr);
    ierr = VecView(u0P,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);    
  }

  ierr = KSPCreate(PETSC_COMM_WORLD,&kspP);CHKERRQ(ierr);
  ierr = KSPSetOperators(kspP,B,B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetType(kspP,KSPCG);CHKERRQ(ierr);
  ierr = KSPSetTolerances(kspP,Tol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(kspP);CHKERRQ(ierr);

  ierr = KSPSolve(kspP,u0P,y);CHKERRQ(ierr);

  //ierr = VecDuplicate(y,&rP);CHKERRQ(ierr);
  //ierr = MatMult(B,y,rP); CHKERRQ(ierr);
  //ierr = VecAXPY(rP,-1.0,u0P);CHKERRQ(ierr);
  //ierr = VecNorm(rP,NORMTYPE,&norm);CHKERRQ(ierr);
  ierr = VecNorm(u0P,NORM_2,&RHSnorm);CHKERRQ(ierr);
  ierr = KSPGetResidualNorm(kspP,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(kspP,&its);CHKERRQ(ierr);  
  //norm *= sqrt(1.0/TotalCubes); //Scale the norm

  ierr = KSPGetConvergedReason(kspP,&reason);CHKERRQ(ierr);
  if (reason<0) {
     ierr = PetscPrintf(PETSC_COMM_WORLD,"\nCAUTION: The reduced linear system is divergent!");CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSolving the reduced linear system:\nNorm of r=b-Ax:\t\t%G\nRelative error |r|/|b|:\t%G\nIterations:\t\t%D\n",norm,norm/RHSnorm,its);CHKERRQ(ierr);

  GetTime(0);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-view_solution",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nVector Solution of the reduced linear system:");CHKERRQ(ierr);
    //ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    //Write output file with PETSC_VIEWER_BINARY_MATLAB format:
    //NOTE: the output generated with this viewer can be loaded into MATLAB using $PETSC_DIR/bin/matlab/PetscReadBinaryMatlab.m
    //ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"y.output",&viewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"y.bin",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
    ierr = VecView(y,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," Wrote to y.bin\n");CHKERRQ(ierr);
  }

  ierr = PetscLogStagePop();CHKERRQ(ierr); 

  //ierr = VecDestroy(&rP);CHKERRQ(ierr);
  ierr = VecDestroy(&u0P);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = KSPDestroy(&kspP);CHKERRQ(ierr);

//----------------------------------------SOLVING THE INTEGRAL EQUATION-------------------------------------------------------

  ierr = PetscLogStageRegister("Integral Equation", &stage);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);

  ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,N,(void*)0,&IE);CHKERRQ(ierr);
  ierr = MatSetUp(IE);CHKERRQ(ierr);
  ierr = MatShellSetOperation(IE,MATOP_MULT,(void(*)(void))ScatIEMatMult);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&u0IE);CHKERRQ(ierr);
  ierr = VecSetSizes(u0IE,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u0IE);CHKERRQ(ierr);
  ierr = VecDuplicate(u0IE,&z);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(u0IE,&Istart,&Iend);CHKERRQ(ierr);
  ierr = VecGetArray(u0IE,&xa);CHKERRQ(ierr);
  for(PetscInt s = Istart; s <Iend ; s++)
  {
      xa[s-Istart] = ScatIE.InitField(s);      
  }
  ierr = VecRestoreArray(u0IE,&xa);CHKERRQ(ierr);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-view_RHS",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nVector Right-hand-side of the integral equation:\n");CHKERRQ(ierr);
    ierr = VecView(u0IE,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);    
  }

  ierr = KSPCreate(PETSC_COMM_WORLD,&kspIE);CHKERRQ(ierr);
  ierr = KSPSetOperators(kspIE,IE,IE,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetType(kspIE,KSPCG);CHKERRQ(ierr);
  ierr = KSPSetTolerances(kspIE,Tol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(kspIE);CHKERRQ(ierr);

  ierr = KSPSolve(kspIE,u0IE,z);CHKERRQ(ierr);

  //ierr = VecDuplicate(z,&rIE);CHKERRQ(ierr);
  //ierr = MatMult(IE,z,rIE); CHKERRQ(ierr);
  //ierr = VecAXPY(rIE,-1.0,u0IE);CHKERRQ(ierr);
  //ierr = VecNorm(rIE,NORMTYPE,&norm);CHKERRQ(ierr);
  ierr = VecNorm(u0IE,NORM_2,&RHSnorm);CHKERRQ(ierr);
  ierr = KSPGetResidualNorm(kspIE,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(kspIE,&its);CHKERRQ(ierr);  
  //norm *= sqrt(1.0/N);  //Scale the norm

  ierr = KSPGetConvergedReason(kspIE,&reason);CHKERRQ(ierr);
  if (reason<0) {
     ierr = PetscPrintf(PETSC_COMM_WORLD,"\nCAUTION: The integral equation has no solution!");CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSolving the integral equation:\nNorm of r=b-Ax:\t\t%G\nRelative error |r|/|b|:\t%G\nIterations:\t\t%D\n",norm,norm/RHSnorm,its);CHKERRQ(ierr);

  GetTime(0);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-view_solution",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nVector Solution of the integral equation:");CHKERRQ(ierr);
    //ierr = VecView(z,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    //Write output file with PETSC_VIEWER_BINARY_MATLAB format:
    //NOTE: the output generated with this viewer can be loaded into MATLAB using $PETSC_DIR/bin/matlab/PetscReadBinaryMatlab.m
    //ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"z.output",&viewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"z.bin",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer,PETSC_VIEWER_BINARY_MATLAB);CHKERRQ(ierr);
    ierr = VecView(z,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," Wrote to z.bin\n");CHKERRQ(ierr);
  }

  ierr = PetscLogStagePop();CHKERRQ(ierr); 

  //ierr = VecDestroy(&rIE);CHKERRQ(ierr);
  ierr = VecDestroy(&u0IE);CHKERRQ(ierr);
  ierr = MatDestroy(&IE);CHKERRQ(ierr);
  ierr = KSPDestroy(&kspIE);CHKERRQ(ierr);


  //----------------------------------------COMPARE SOLUTIONS BETWEEN S, P SYSTEMS WITH IE-------------------------------------------------------

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nComparing solutions:");CHKERRQ(ierr);
  //FFTx: solution to S
  //y   : solution to P, size(y) << size(FFTx)
  //z   : solution to IE, size(y) <= size(z) < size(FFTx)

  //Compare S & P:
  ierr = VecDuplicate(y,&Distance);CHKERRQ(ierr);
  ierr = VecDuplicate(y,&nParticlePerCube);CHKERRQ(ierr);

  ierr = DiffP(FFTx,y,Distance,nParticlePerCube);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(Distance,Distance,nParticlePerCube);CHKERRQ(ierr);
  ierr = VecNorm(Distance,NORMTYPE,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n(Original system)   vs (Reduced system):\t%G",norm);CHKERRQ(ierr);

  ierr = VecDestroy(&Distance);CHKERRQ(ierr);
  ierr = VecDestroy(&nParticlePerCube);CHKERRQ(ierr);

  //Compare S & IE:
  ierr = VecDuplicate(z,&Distance);CHKERRQ(ierr);
  ierr = VecDuplicate(z,&nParticlePerCube);CHKERRQ(ierr);

  ierr = DiffIE(FFTx,z,Distance,nParticlePerCube);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(Distance,Distance,nParticlePerCube);CHKERRQ(ierr);
  ierr = VecNorm(Distance,NORMTYPE,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n(Integral equation) vs (Original system):\t%G",norm);CHKERRQ(ierr);

  ierr = VecDestroy(&Distance);CHKERRQ(ierr);
  ierr = VecDestroy(&nParticlePerCube);CHKERRQ(ierr);

  //Compare P & IE:
  ierr = VecDuplicate(y,&Distance);CHKERRQ(ierr);
  ierr = VecDuplicate(y,&nParticlePerCube);CHKERRQ(ierr);

  ierr = DiffPCollocation(z,y,Distance,nParticlePerCube);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(Distance,Distance,nParticlePerCube);CHKERRQ(ierr);
  ierr = VecNorm(Distance,NORMTYPE,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n(Integral equation) vs (Reduced system):\t%G",norm);CHKERRQ(ierr);

  ierr = VecDestroy(&Distance);CHKERRQ(ierr);
  ierr = VecDestroy(&nParticlePerCube);CHKERRQ(ierr);

  //Compare S & u0:
  ierr = VecAXPY(u0,-1.0,FFTx);CHKERRQ(ierr);
  ierr = VecNorm(u0,NORMTYPE,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n(Original system)   vs (Incident field):\t%G\n",norm);CHKERRQ(ierr);

  GetTime(1); 

  ierr = VecDestroy(&FFTx);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&z);CHKERRQ(ierr);
  ierr = VecDestroy(&u0);CHKERRQ(ierr);

  PetscFinalize();
  
  return 0;
}

////////////////////////////////////////////////////////////////////////////////////

#undef __FUNCT__
#define __FUNCT__ "ScatPMatMult"
inline PetscErrorCode ScatPMatMult(Mat mat,Vec xx,Vec yy)
{
  PetscInt       n,s,t;
  PetscScalar    tmp,v;
  PetscInt       xstart,xend;

  PetscFunctionBegin;
  VecGetOwnershipRange(xx,&xstart,&xend);
  VecGetSize(yy,&n);
  VecZeroEntries(yy);

  for(s = 0; s <n ; s++)
  {
      tmp = 0;
      for(t = xstart; t < xend; t++)
      {
          VecGetValues(xx,1,&t,&v);
          tmp += Scat3DP.CoefMatFast(s,t)*v;
      }
      VecSetValues(yy,1,&s,&tmp,ADD_VALUES);
  }
  VecAssemblyBegin(yy);
  VecAssemblyEnd(yy);

  PetscFunctionReturn(0);
}

////////////////////////////////////////////////////////////////////////////////////

#undef __FUNCT__
#define __FUNCT__ "ScatIEMatMult"
inline PetscErrorCode ScatIEMatMult(Mat mat,Vec xx,Vec yy)
{
  PetscInt       n,s,t;
  PetscScalar    tmp,v;
  PetscInt       xstart,xend;

  PetscFunctionBegin;
  VecGetOwnershipRange(xx,&xstart,&xend);
  VecGetSize(yy,&n);
  VecZeroEntries(yy);

  for(s = 0; s <n ; s++)
  {
      tmp = 0;
      for(t = xstart; t < xend; t++)
      {
          VecGetValues(xx,1,&t,&v);
          tmp += ScatIE.CoefMatFast(s,t)*v;
      }
      VecSetValues(yy,1,&s,&tmp,ADD_VALUES);
  }
  VecAssemblyBegin(yy);
  VecAssemblyEnd(yy);

  PetscFunctionReturn(0);
}


////////////////////////////////////////////////////////////////////////////////////

#undef __FUNCT__
#define __FUNCT__ "DiffP"
PetscErrorCode DiffP(Vec originalVec, Vec reducedVec, Vec Distance, Vec nParticlePerCube)
{
    // Find the distance between 2 solutions originalVec and reducedVec
    PetscInt    s,t;
    PetscInt    n,xstart,xend;
    PetscScalar tmp,v,diff;
    Vec         vec;
    VecScatter  scat;
    IS          is;

    PetscFunctionBegin;
    VecGetOwnershipRange(originalVec,&xstart,&xend);
    VecGetSize(reducedVec,&n);

    VecCreate(PETSC_COMM_SELF,&vec);
    VecSetSizes(vec,PETSC_DECIDE,n);
    VecSetFromOptions(vec);

    ISCreateStride(PETSC_COMM_WORLD,n,0,1,&is);
    VecScatterCreate(reducedVec,PETSC_NULL,vec,is,&scat);
    //VecScatterCreateToAll(reducedVec,&scat,&vec);
    VecScatterBegin(scat,reducedVec,vec,INSERT_VALUES,SCATTER_FORWARD);
    VecScatterEnd(scat,reducedVec,vec,INSERT_VALUES,SCATTER_FORWARD);

    for(s=xstart;s<xend;s++)
    {
        t = Scat3DP.FindCube(s);  
        VecGetValues(vec,1,&t,&tmp);
        VecGetValues(originalVec,1,&s,&v);

        diff = cabs(v - tmp);
        VecSetValues(Distance,1,&t,&diff,ADD_VALUES);
        tmp = 1;
        VecSetValues(nParticlePerCube,1,&t,&tmp,ADD_VALUES);        
    }   

    VecDestroy(&vec);
    VecScatterDestroy(&scat);
    ISDestroy(&is);

    PetscFunctionReturn(0);
}

////////////////////////////////////////////////////////////////////////////////////

#undef __FUNCT__
#define __FUNCT__ "DiffPCollocation"
PetscErrorCode DiffPCollocation(Vec originalVec, Vec reducedVec, Vec Distance, Vec nCollocationPointsPerCube)
{
    // Find the distance between 2 solutions originalVec and reducedVec
    PetscInt    s,t;
    PetscInt    n,xstart,xend;
    PetscScalar tmp,v,diff;
    Vec         vec;
    VecScatter  scat;
    IS          is;

    PetscFunctionBegin;
    VecGetOwnershipRange(originalVec,&xstart,&xend);
    VecGetSize(reducedVec,&n);

    VecCreate(PETSC_COMM_SELF,&vec);
    VecSetSizes(vec,PETSC_DECIDE,n);
    VecSetFromOptions(vec);

    ISCreateStride(PETSC_COMM_WORLD,n,0,1,&is);
    VecScatterCreate(reducedVec,PETSC_NULL,vec,is,&scat);
    //VecScatterCreateToAll(reducedVec,&scat,&vec);
    VecScatterBegin(scat,reducedVec,vec,INSERT_VALUES,SCATTER_FORWARD);
    VecScatterEnd(scat,reducedVec,vec,INSERT_VALUES,SCATTER_FORWARD);

    for(s=xstart;s<xend;s++)
    {
        t = Scat3DP.FindCubeOfCollocation(s);  
        VecGetValues(vec,1,&t,&tmp);
        VecGetValues(originalVec,1,&s,&v);

        diff = cabs(v - tmp);
        VecSetValues(Distance,1,&t,&diff,ADD_VALUES);
        tmp = 1;
        VecSetValues(nCollocationPointsPerCube,1,&t,&tmp,ADD_VALUES);        
    }   

    VecDestroy(&vec);
    VecScatterDestroy(&scat);
    ISDestroy(&is);

    PetscFunctionReturn(0);
}

////////////////////////////////////////////////////////////////////////////////////

#undef __FUNCT__
#define __FUNCT__ "DiffIE"
PetscErrorCode DiffIE(Vec originalVec, Vec reducedVec, Vec Distance, Vec nParticlePerCube)
{
    // Find the distance between 2 solutions originalVec and reducedVec
    PetscInt    s,t;
    PetscInt    n,xstart,xend;
    PetscScalar tmp,v,diff;
    Vec         vec;
    VecScatter  scat;
    IS          is;

    PetscFunctionBegin;
    VecGetOwnershipRange(originalVec,&xstart,&xend);
    VecGetSize(reducedVec,&n);

    VecCreate(PETSC_COMM_SELF,&vec);
    VecSetSizes(vec,PETSC_DECIDE,n);
    VecSetFromOptions(vec);

    ISCreateStride(PETSC_COMM_WORLD,n,0,1,&is);
    VecScatterCreate(reducedVec,PETSC_NULL,vec,is,&scat);
    //VecScatterCreateToAll(reducedVec,&scat,&vec);
    VecScatterBegin(scat,reducedVec,vec,INSERT_VALUES,SCATTER_FORWARD);
    VecScatterEnd(scat,reducedVec,vec,INSERT_VALUES,SCATTER_FORWARD);

    for(s=xstart;s<xend;s++)
    {
        t = ScatIE.FindCube(s);  
        VecGetValues(vec,1,&t,&tmp);
        VecGetValues(originalVec,1,&s,&v);

        diff = cabs(v - tmp);
        VecSetValues(Distance,1,&t,&diff,ADD_VALUES);
        tmp = 1;
        VecSetValues(nParticlePerCube,1,&t,&tmp,ADD_VALUES);        
    }   
  
    VecDestroy(&vec);
    VecScatterDestroy(&scat);
    ISDestroy(&is);

    PetscFunctionReturn(0);
}

////////////////////////////////////////////////////////////////////////////////////

#undef __FUNCT__
#define __FUNCT__ "ScatSMatMultFFT"
inline PetscErrorCode ScatSMatMultFFT(Mat mat,Vec xx,Vec yy)
{
  PetscInt	  N;
  Vec             FFTVec;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  
  ierr = VecGetSize(xx,&N);CHKERRQ(ierr);

  //Convert to use 3D convolution theorem:
  ierr = VecSet(RealSpaceCube,0);CHKERRQ(ierr);
  ierr = VecCopyN1(xx,RealSpaceCube,N);CHKERRQ(ierr);

  // Apply FFTW_FORWARD for xx:
  fftwf_execute(fplan);
  ierr = PetscMemoryGetCurrentUsage(&mem);CHKERRQ(ierr);PetscPrintf(PETSC_COMM_WORLD,"Apply FFT:\t\tMem used: %G \t",mem);GetTime(0);

  ierr = VecDuplicate(RealSpaceCube,&FFTVec);CHKERRQ(ierr);
  ierr = VecCopy(RealSpaceCube,FFTVec);CHKERRQ(ierr);

  //Prepare the matrix mat to multiply with xx, mat*xx=yy:
  ierr = FFTPaddedGreenCube(dim);CHKERRQ(ierr);
  ierr = PetscMemoryGetCurrentUsage(&mem);CHKERRQ(ierr);PetscPrintf(PETSC_COMM_WORLD,"Pad Green cube:\t\tMem used: %G \t",mem);GetTime(0);

  //Matrix-vector multiplication in Fourier space:
  ierr = VecPointwiseMult(RealSpaceCube,RealSpaceCube,FFTVec);CHKERRQ(ierr);
  ierr = PetscMemoryGetCurrentUsage(&mem);CHKERRQ(ierr);PetscPrintf(PETSC_COMM_WORLD,"Multiply vectors:\tMem used: %G \t",mem);GetTime(0);

  ierr = VecDestroy(&FFTVec);CHKERRQ(ierr);

  // Apply FFTW_BACKWARD: 
  fftwf_execute(bplan);
  ierr = PetscMemoryGetCurrentUsage(&mem);CHKERRQ(ierr);PetscPrintf(PETSC_COMM_WORLD,"Apply iFFT:\t\tMem used: %G \t",mem);GetTime(0);

  ierr = VecCopyN1(RealSpaceCube,yy,N);CHKERRQ(ierr);
  ierr = PetscMemoryGetCurrentUsage(&mem);CHKERRQ(ierr);PetscPrintf(PETSC_COMM_WORLD,"Assembly result:\tMem used: %G \t",mem);GetTime(0);

  PetscFunctionReturn(0);
}

////////////////////////////////////////////////////////////////////////////////////

#undef __FUNCT__
#define __FUNCT__ "FFTPaddedGreenCube"
inline PetscErrorCode FFTPaddedGreenCube(PetscInt dim[])
{
    PetscInt 	   slap1,slap2,slap,m1,m2,m3,m,mm,N;
    PetscReal      scale;
    PetscScalar    ***xa;
    PetscInt       Istart,Iend,nLocalSlaps,nEntriesPerSlap;
    PetscErrorCode ierr;
    PetscBool      flg = PETSC_FALSE;

    PetscFunctionBegin;

    N = Scat3DS.NumParticlePerSide;

    //Create RealSpaceCube of size dim: 
    VecGetOwnershipRange(RealSpaceCube,&Istart,&Iend);
    VecGetLocalSize(RealSpaceCube,&nLocalSlaps);
    
    nEntriesPerSlap = dim[0]*dim[1];
    nLocalSlaps = nLocalSlaps/nEntriesPerSlap;
    if((Istart>=0) && (Iend>0))
    {
    	slap1 = Istart/nEntriesPerSlap;
    	slap2 = Iend/nEntriesPerSlap;
    }
    else
    {
	slap1 = ceil(dim[2]/(PetscReal)size)*rank;
	slap2 = slap1 + nLocalSlaps;
	cout<<"\nIstart = "<<Istart<<"\nIend = "<<Iend<<"\nslap1 = "<<slap1<<"\nslap2 = "<<slap2<<"\nslap2-slap1 = "<<slap2-slap1<<"\nnLocalSlaps = "<<nLocalSlaps<<"\n";
    }

    ierr = VecGetArray3d(RealSpaceCube,dim[0],dim[1],nLocalSlaps,0,0,slap1,&xa);CHKERRQ(ierr);

    //Copy Green cube of size NxNxN:
    slap = (N>slap2)?(slap2):(N);
    for(m3=slap1;m3<slap;m3++)
    {
         for(m2=0;m2<N;m2++)
         {
	     for(m1=0;m1<N;m1++)
	     {		
 		  xa[m1][m2][m3] = Scat3DS.Green3DF(m1,m2,m3);
	     }
         }
    }
    if(rank==0)
	xa[0][0][0] = 1;

    //Pad the Green cube to have size dim=pow(2*N-2,3) for convolution:
    slap = (N>slap2)?(slap2):(N);
    for(m3=slap1;m3<slap;m3++)    
    {
         for(m2=0;m2<N;m2++)
         {
	     for(m1=N;m1<dim[0];m1++)
	     {				
 		  xa[m1][m2][m3] = Scat3DS.Green3DF(2*N-m1-1,m2,m3);		
	     }
         }
    }
    slap = (N>slap2)?(slap2):(N);
    for(m3=slap1;m3<slap;m3++)
    {
         for(m2=N;m2<dim[1];m2++)
         {
	     for(m1=0;m1<N;m1++)
	     {				
	    	  xa[m1][m2][m3] = Scat3DS.Green3DF(m1,2*N-m2-1,m3);					
	     }
	     for(m1=N;m1<dim[0];m1++)
	     {				
		  m = 2*N-m1-1;
	    	  xa[m1][m2][m3] = Scat3DS.Green3DF(m,2*N-m2-1,m3);					
	     }
         }
    }
    slap = (N>slap1)?(N):(slap1);
    for(m3=slap;m3<slap2;m3++)
    {
       for(m2=0;m2<N;m2++)
       {
	     for(m1=0;m1<N;m1++)
	     {				
	    	  xa[m1][m2][m3] = Scat3DS.Green3DF(m1,m2,2*N-m3-1);					
	     }
	     for(m1=N;m1<dim[0];m1++)
	     {				
		  m = 2*N-m1-1;
	    	  xa[m1][m2][m3] = Scat3DS.Green3DF(m,m2,2*N-m3-1);					
	     }
       }
       for(m2=N;m2<dim[1];m2++)
       {
	     mm = 2*N-m2-1;
	     for(m1=0;m1<N;m1++)
	     {				
	    	  xa[m1][m2][m3] = Scat3DS.Green3DF(m1,mm,2*N-m3-1);					
	     }
	     for(m1=N;m1<dim[0];m1++)
	     {				
		  m = 2*N-m1-1;
	    	  xa[m1][m2][m3] = Scat3DS.Green3DF(m,mm,2*N-m3-1);					
	     }
       }
    }
    ierr = VecRestoreArray3d(RealSpaceCube,dim[0],dim[1],nLocalSlaps,0,0,slap1,&xa);CHKERRQ(ierr);

    flg  = PETSC_FALSE;
    ierr = PetscOptionsGetBool(PETSC_NULL,"-view_Green",&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
       ierr = PetscPrintf(PETSC_COMM_WORLD,"\nPadded Green cube:\n");CHKERRQ(ierr);
       ierr = VecView(RealSpaceCube,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);    
	//VecView3D(RealSpaceCube,dim[0]);
    }

    // Apply FFTW_FORWARD for the Green cube:
    fftwf_execute(fplan);

    //FFTW computes an unnormalized DFT, need to scale:
    scale = dim[0]*dim[1]*dim[2];
    scale = 1.0/(PetscReal)scale;
    ierr = VecScale(RealSpaceCube,scale);CHKERRQ(ierr);

    flg  = PETSC_FALSE;
    ierr = PetscOptionsGetBool(PETSC_NULL,"-view_GreenFFT",&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\nFFT of padded Green cube:\n");CHKERRQ(ierr);
      ierr = VecView(RealSpaceCube,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);    
      //VecView3D(RealSpaceCube,dim[0]);
    }

    PetscFunctionReturn(0);
}

////////////////////////////////////////////////////////////////////////////////////

#undef __FUNCT__
#define __FUNCT__ "GetTime"
void GetTime(bool final)
{
    //Check total time used:
    if(rank==0)  
    {
       time(&time2);  
       checkTime(time1,time2,final);
    }
}

////////////////////////////////////////////////////////////////////////////////////

#undef __FUNCT__
#define __FUNCT__ "FFTInit"
inline PetscErrorCode FFTInit()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  alloc_local = fftwf_mpi_local_size_3d(dim[0],dim[1],dim[2],PETSC_COMM_WORLD,&local_n0,&local_0_start);
  data_in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)*alloc_local);
  ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,(PetscInt)local_n0*dim[1]*dim[2],(PetscInt)(dim[0]*dim[1]*dim[2]),(const PetscScalar*)data_in,&RealSpaceCube);CHKERRQ(ierr);

  fplan = fftwf_mpi_plan_dft_3d(dim[0],dim[1],dim[2],data_in,data_in,PETSC_COMM_WORLD,FFTW_FORWARD,FFTW_ESTIMATE);
  bplan = fftwf_mpi_plan_dft_3d(dim[0],dim[1],dim[2],data_in,data_in,PETSC_COMM_WORLD,FFTW_BACKWARD,FFTW_ESTIMATE);

  ierr = PetscMemoryGetCurrentUsage(&mem);CHKERRQ(ierr);PetscPrintf(PETSC_COMM_WORLD,"Setting up FFT:\t\tMem used: %G \t",mem);GetTime(0);

  PetscFunctionReturn(0);
}

////////////////////////////////////////////////////////////////////////////////////

#undef __FUNCT__
#define __FUNCT__ "FFTFree"
inline PetscErrorCode FFTFree()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = VecDestroy(&RealSpaceCube);CHKERRQ(ierr);
  fftwf_destroy_plan(fplan);
  fftwf_destroy_plan(bplan);
  fftwf_free(data_in); 
  fftwf_mpi_cleanup();

  //ierr = PetscMemoryGetCurrentUsage(&mem);CHKERRQ(ierr);PetscPrintf(PETSC_COMM_WORLD,"Free FFT memory:\tMem used: %G \t",mem);GetTime(0);

  PetscFunctionReturn(0);
}

////////////////////////////////////////////////////////////////////////////////////

#undef __FUNCT__
#define __FUNCT__ "VecCopyN1"
inline PetscErrorCode VecCopyN1(Vec xx,Vec yy,PetscInt N)
{
    PetscInt       j;
    PetscInt       Istart,Iend;
    PetscScalar    tmp;
    PetscErrorCode ierr;

    PetscFunctionBegin;

    ierr = VecGetOwnershipRange(xx,&Istart,&Iend);CHKERRQ(ierr);
    if(Istart>=0 && Istart<N)
    {       
       Iend = (Iend<N)?(Iend):(N);
       for(j=Istart;j<Iend;j++)
       {
	  ierr = VecGetValues(xx,1,&j,&tmp);CHKERRQ(ierr);
          ierr = VecSetValues(yy,1,&j,&tmp,INSERT_VALUES);CHKERRQ(ierr);		  	
       }
    }
    ierr = PetscMemoryGetCurrentUsage(&mem);CHKERRQ(ierr);PetscPrintf(PETSC_COMM_WORLD,"Copy internal vector:\tMem used: %G \t",mem);GetTime(0);
    ierr = VecAssemblyBegin(yy);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(yy);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}