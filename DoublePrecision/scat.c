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
extern PetscErrorCode FFTPaddedGreenCube(PetscInt[]);

Scattering3DS<std::vector<PetscReal>,std::vector<PetscScalar>,PetscScalar,PetscReal,PetscInt,PetscInt> 		Scat3DS;
Scattering3DP<std::vector<PetscReal>,std::vector<PetscScalar>,PetscScalar,PetscReal,PetscInt,PetscInt> 		Scat3DP;
ScatteringIE_Riemann<std::vector<PetscReal>,std::vector<PetscScalar>,PetscScalar,PetscReal,PetscInt,PetscInt> 	ScatIE;
Vec 	      RealSpaceCube,FFTPaddedGreen;
fftw_plan     fplan,bplan;

//MPI:
int           rank;
int           size;


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Vec            y,z,rC,u0,u0P,u0IE,Distance,nParticlePerCube;     	   /* approx solution, RHS, solution-difference */
  //Vec            r,rP,rIE;                                         	   /* residual */
  Mat            B,IE,AFFT;           		                   	   /* linear system matrix */
  KSP            kspP,kspIE,kspFFT;      	                           /* linear solver context */
  //PC             pc;          		                           /* Preconditioner context */
  PetscReal      norm, Tol=1e-3;          		             
  PetscInt       its;           		                           /* number of iterations reached */
  PetscInt       Istart,Iend;
  PetscErrorCode ierr;
  KSPConvergedReason reason;
  
  //Scattering3D:
  PetscInt       M = 0;                 				   //Total particles
  PetscReal      a = 0;                 				   //Particle radius
  PetscReal      ParticleDistance = 0;
  PetscReal      VolQ = 0;              				   //Volume of the domain Q that contains all particles
  PetscReal      Kappa = -1;             				   //Power const with respect to the radius of particles: Kappa in [0,1)
  std::vector<PetscReal> WaveDirection(3,0);WaveDirection[0] = 1; 	   // WaveDirection is a unit vector that indicates the direction of plane wave
  PetscScalar    OriginalRefractionCoef = 0;
  PetscScalar    DesiredRefractionCoef = 0;
  PetscReal      Distribution = 0;
  PetscInt       TotalCubes = 0;
  PetscInt       N = 0;                 				   //Number of sub domains used for solving IE
  PetscScalar    tmp;
  PetscScalar    *val;
  PetscInt       j, *pos;
  time_t         time1, time2;                                    	   //Time usage 
  PetscBool      flg = PETSC_FALSE;

  //FFTW:
  PetscInt 	 dim[3];
  fftw_complex   *data_in;
  ptrdiff_t      alloc_local,local_n0,local_0_start;
  Vec		 FFTx;
  //Vec		 FFTr;


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

  PetscPrintf(PETSC_COMM_WORLD,"\nInitializing done!\n");
  //Check total time used:
  if(rank==0)  
  {
      time(&time2);  
      checkTime(time1,time2,0);
  }

  //----------------------------------------SET UP FFTW FOR SYSTEMS S---------------------------------------------------
  ierr = PetscLogStageRegister("Conv FFT", &stage);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);

  dim[0]=2*Scat3DS.NumParticlePerSide-2;
  dim[1]=dim[0];
  dim[2]=dim[0];   

  PetscPrintf(PETSC_COMM_WORLD,"\nSetting up FFT ");
  fftw_mpi_init();
  alloc_local = fftw_mpi_local_size_3d(dim[0],dim[1],dim[2],PETSC_COMM_WORLD,&local_n0,&local_0_start);
  data_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
  ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,(PetscInt)local_n0*dim[1]*dim[2],(PetscInt)(dim[0]*dim[1]*dim[2]),(const PetscScalar*)data_in,&RealSpaceCube);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,".");

  fplan = fftw_mpi_plan_dft_3d(dim[0],dim[1],dim[2],data_in,data_in,PETSC_COMM_WORLD,FFTW_FORWARD,FFTW_ESTIMATE);
  bplan = fftw_mpi_plan_dft_3d(dim[0],dim[1],dim[2],data_in,data_in,PETSC_COMM_WORLD,FFTW_BACKWARD,FFTW_ESTIMATE);
  PetscPrintf(PETSC_COMM_WORLD,".");

  FFTPaddedGreenCube(dim);
  PetscPrintf(PETSC_COMM_WORLD," done!\n");
  //Check total time used:
  if(rank==0)  
  {
      time(&time2);  
      checkTime(time1,time2,0);
  }

  ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,M,M,(void*)0,&AFFT);CHKERRQ(ierr);
  ierr = MatSetUp(AFFT);CHKERRQ(ierr);
  ierr = MatShellSetOperation(AFFT,MATOP_MULT,(void(*)(void))ScatSMatMultFFT);CHKERRQ(ierr);

  //ierr = MatSetOption(AFFT,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);

  //Setup RHS:
  ierr = VecCreate(PETSC_COMM_WORLD,&u0);CHKERRQ(ierr);
  ierr = VecSetSizes(u0,PETSC_DECIDE,M);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u0);CHKERRQ(ierr);

  VecGetOwnershipRange(u0,&Istart,&Iend);
  ierr = PetscMalloc((Iend-Istart)*sizeof(PetscScalar),&val);CHKERRQ(ierr); 
  ierr = PetscMalloc((Iend-Istart)*sizeof(PetscInt),&pos);CHKERRQ(ierr); 
  j = 0;
  for(PetscInt s = Istart; s <Iend ; s++)
  {
      val[j] = Scat3DS.InitField(s);
      pos[j] = j+Istart;
      j++;
  }
  VecSetValues(u0,Iend-Istart,pos,val,INSERT_VALUES);
  ierr = VecAssemblyBegin(u0);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(u0);CHKERRQ(ierr);
  ierr = PetscFree(val);CHKERRQ(ierr);
  ierr = PetscFree(pos);CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_WORLD,"\nSetting up RHS done!\n");
  //Check total time used:
  if(rank==0)  
  {
      time(&time2);  
      checkTime(time1,time2,0);
  }

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
  ierr = KSPSetType(kspFFT,KSPGMRES);CHKERRQ(ierr);
  ierr = KSPSetTolerances(kspFFT,Tol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(kspFFT);CHKERRQ(ierr);	

  ierr = KSPSolve(kspFFT,u0,FFTx);CHKERRQ(ierr);

  //ierr = VecDuplicate(FFTx,&FFTr);CHKERRQ(ierr);
  //ierr = MatMult(AFFT,FFTx,FFTr); CHKERRQ(ierr);
  //ierr = VecAXPY(FFTr,-1.0,u0);CHKERRQ(ierr);
  //ierr = VecNorm(FFTr,NORMTYPE,&norm);CHKERRQ(ierr);
  ierr = KSPGetResidualNorm(kspFFT,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(kspFFT,&its);CHKERRQ(ierr);
  //norm *= sqrt(1.0/M); //  Scale the norm

  ierr = KSPGetConvergedReason(kspFFT,&reason);CHKERRQ(ierr);
  if (reason<0) {
     ierr = PetscPrintf(PETSC_COMM_WORLD,"\nCAUTION: The original linear system is divergent!");CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSolving the original linear system (FFT):\nNorm of error:\t%G\nIterations:\t%D\n",norm,its);CHKERRQ(ierr);

  //Check total time used:
  if(rank==0)  
  {
      time(&time2);  
      checkTime(time1,time2,0);
  }

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-view_solution",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nVector solution of the original linear system (FFT):\n");CHKERRQ(ierr);
    ierr = VecView(FFTx,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  ierr = PetscLogStagePop();CHKERRQ(ierr); 

  ierr = VecDestroy(&RealSpaceCube);CHKERRQ(ierr);
  ierr = VecDestroy(&FFTPaddedGreen);CHKERRQ(ierr);
  fftw_destroy_plan(fplan);
  fftw_destroy_plan(bplan);
  fftw_free(data_in); 
  fftw_mpi_cleanup();
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

  VecGetOwnershipRange(u0P,&Istart,&Iend);
  for(PetscInt s = Istart; s <Iend; s++)
  {
      tmp = Scat3DP.InitField(s);
      VecSetValues(u0P,1,&s,&tmp,INSERT_VALUES);
  }
  ierr = VecAssemblyBegin(u0P);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(u0P);CHKERRQ(ierr);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-view_RHS",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nVector Right-hand-side of the reduced linear system:\n");CHKERRQ(ierr);
    ierr = VecView(u0P,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);    
  }

  ierr = KSPCreate(PETSC_COMM_WORLD,&kspP);CHKERRQ(ierr);
  ierr = KSPSetOperators(kspP,B,B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetType(kspP,KSPGMRES);CHKERRQ(ierr);
  ierr = KSPSetTolerances(kspP,Tol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(kspP);CHKERRQ(ierr);

  ierr = KSPSolve(kspP,u0P,y);CHKERRQ(ierr);

  //ierr = VecDuplicate(y,&rP);CHKERRQ(ierr);
  //ierr = MatMult(B,y,rP); CHKERRQ(ierr);
  //ierr = VecAXPY(rP,-1.0,u0P);CHKERRQ(ierr);
  //ierr = VecNorm(rP,NORMTYPE,&norm);CHKERRQ(ierr);
  ierr = KSPGetResidualNorm(kspP,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(kspP,&its);CHKERRQ(ierr);
  /* Scale the norm */
  /*  norm *= sqrt(1.0/TotalCubes); */

  ierr = KSPGetConvergedReason(kspP,&reason);CHKERRQ(ierr);
  if (reason<0) {
     ierr = PetscPrintf(PETSC_COMM_WORLD,"\nCAUTION: The reduced linear system is divergent!");CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSolving the reduced linear system:\nNorm of error:\t%G\nIterations:\t%D\n",norm,its);CHKERRQ(ierr);

  //Check total time used:
  if(rank==0)  
  {
      time(&time2);  
      checkTime(time1,time2,0);
  }

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-view_solution",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nVector Solution of the reduced linear system:\n");CHKERRQ(ierr);
    ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
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

  VecGetOwnershipRange(u0IE,&Istart,&Iend);
  for(PetscInt s = Istart; s <Iend ; s++)
  {
      tmp = ScatIE.InitField(s);
      VecSetValues(u0IE,1,&s,&tmp,INSERT_VALUES);
  }
  ierr = VecAssemblyBegin(u0IE);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(u0IE);CHKERRQ(ierr);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-view_RHS",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nVector Right-hand-side of the integral equation:\n");CHKERRQ(ierr);
    ierr = VecView(u0IE,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);    
  }

  ierr = KSPCreate(PETSC_COMM_WORLD,&kspIE);CHKERRQ(ierr);
  ierr = KSPSetOperators(kspIE,IE,IE,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetType(kspIE,KSPGMRES);CHKERRQ(ierr);
  ierr = KSPSetTolerances(kspIE,Tol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(kspIE);CHKERRQ(ierr);

  ierr = KSPSolve(kspIE,u0IE,z);CHKERRQ(ierr);

  //ierr = VecDuplicate(z,&rIE);CHKERRQ(ierr);
  //ierr = MatMult(IE,z,rIE); CHKERRQ(ierr);
  //ierr = VecAXPY(rIE,-1.0,u0IE);CHKERRQ(ierr);
  //ierr = VecNorm(rIE,NORMTYPE,&norm);CHKERRQ(ierr);
  ierr = KSPGetResidualNorm(kspIE,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(kspIE,&its);CHKERRQ(ierr);
  /* Scale the norm */
  /*  norm *= sqrt(1.0/N); */

  ierr = KSPGetConvergedReason(kspIE,&reason);CHKERRQ(ierr);
  if (reason<0) {
     ierr = PetscPrintf(PETSC_COMM_WORLD,"\nCAUTION: The integral equation has no solution!");CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSolving the integral equation:\nNorm of error:\t%G\nIterations:\t%D\n",norm,its);CHKERRQ(ierr);

  //Check total time used:
  if(rank==0)  
  {
      time(&time2);  
      checkTime(time1,time2,0);
  }

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-view_solution",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nVector Solution of the integral equation:\n");CHKERRQ(ierr);
    ierr = VecView(z,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  ierr = PetscLogStagePop();CHKERRQ(ierr); 

  //ierr = VecDestroy(&rIE);CHKERRQ(ierr);
  ierr = VecDestroy(&u0IE);CHKERRQ(ierr);
  ierr = MatDestroy(&IE);CHKERRQ(ierr);
  ierr = KSPDestroy(&kspIE);CHKERRQ(ierr);


  //----------------------------------------COMPARE SOLUTIONS BETWEEN S, P SYSTEMS WITH IE-------------------------------------------------------

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nComparing solutions:");CHKERRQ(ierr);
  //x: solution to S
  //y: solution to P, size(y) << size(x)
  //z: solution to IE, size(y) <= size(z) <= size(x)

  //Compare S & P:
  ierr = VecDuplicate(y,&Distance);CHKERRQ(ierr);
  ierr = VecDuplicate(y,&nParticlePerCube);CHKERRQ(ierr);
  ierr = VecDuplicate(y,&rC);CHKERRQ(ierr);

  ierr = DiffP(FFTx,y,Distance,nParticlePerCube);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(rC,Distance,nParticlePerCube);CHKERRQ(ierr);
  ierr = VecNorm(rC,NORMTYPE,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n(Original system)   vs (Reduced system):\t%G",norm);CHKERRQ(ierr);

  ierr = VecDestroy(&Distance);CHKERRQ(ierr);
  ierr = VecDestroy(&nParticlePerCube);CHKERRQ(ierr);
  ierr = VecDestroy(&rC);CHKERRQ(ierr);

  //Compare S & IE:
  ierr = VecDuplicate(z,&Distance);CHKERRQ(ierr);
  ierr = VecDuplicate(z,&nParticlePerCube);CHKERRQ(ierr);
  ierr = VecDuplicate(z,&rC);CHKERRQ(ierr);

  ierr = DiffIE(FFTx,z,Distance,nParticlePerCube);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(rC,Distance,nParticlePerCube);CHKERRQ(ierr);
  ierr = VecNorm(rC,NORMTYPE,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n(Integral equation) vs (Original system):\t%G",norm);CHKERRQ(ierr);

  ierr = VecDestroy(&Distance);CHKERRQ(ierr);
  ierr = VecDestroy(&nParticlePerCube);CHKERRQ(ierr);
  ierr = VecDestroy(&rC);CHKERRQ(ierr);

  //Compare P & IE:
  ierr = VecDuplicate(y,&Distance);CHKERRQ(ierr);
  ierr = VecDuplicate(y,&nParticlePerCube);CHKERRQ(ierr);
  ierr = VecDuplicate(y,&rC);CHKERRQ(ierr);

  ierr = DiffPCollocation(z,y,Distance,nParticlePerCube);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(rC,Distance,nParticlePerCube);CHKERRQ(ierr);
  ierr = VecNorm(rC,NORMTYPE,&norm);CHKERRQ(ierr);
  //VecAXPY(z,-1,y);
  //ierr = VecNorm(z,NORMTYPE,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n(Integral equation) vs (Reduced system):\t%G",norm);CHKERRQ(ierr);

  ierr = VecDestroy(&Distance);CHKERRQ(ierr);
  ierr = VecDestroy(&nParticlePerCube);CHKERRQ(ierr);
  ierr = VecDestroy(&rC);CHKERRQ(ierr);

  //Compare S & u0:
  VecAXPY(u0,-1,FFTx);
  ierr = VecNorm(u0,NORMTYPE,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n(Original system)   vs (Incident field):\t%G\n",norm);CHKERRQ(ierr);

  //Check total time used:
  if(rank==0)  
  {
      time(&time2);  
      checkTime(time1,time2,true);
  } 

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

        diff = v - tmp;
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

        diff = v - tmp;
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

        diff = v - tmp;
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
  PetscInt	n;

  PetscFunctionBegin;

  PetscPrintf(PETSC_COMM_WORLD,"\nComputing ");
  VecGetSize(xx,&n);

  //Convert to use 3D convolution theorem:
  VecZeroEntries(RealSpaceCube);
  VecCopyN(xx,RealSpaceCube,n);
  PetscPrintf(PETSC_COMM_WORLD,".");

  // Apply FFTW_FORWARD for xx:
  fftw_execute(fplan);
  PetscPrintf(PETSC_COMM_WORLD,".");

  //Matrix-vector multiplication in Fourier space
  VecPointwiseMult(RealSpaceCube,RealSpaceCube,FFTPaddedGreen);
  PetscPrintf(PETSC_COMM_WORLD,".");

  // Apply FFTW_BACKWARD 
  fftw_execute(bplan);
  PetscPrintf(PETSC_COMM_WORLD,".\n");

  VecCopyN(RealSpaceCube,yy,n);

  PetscFunctionReturn(0);
}

////////////////////////////////////////////////////////////////////////////////////

#undef __FUNCT__
#define __FUNCT__ "FFTPaddedGreenCube"
PetscErrorCode FFTPaddedGreenCube(PetscInt dim[])
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
    PetscPrintf(PETSC_COMM_WORLD,".");

    flg  = PETSC_FALSE;
    ierr = PetscOptionsGetBool(PETSC_NULL,"-view_Green",&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
       ierr = PetscPrintf(PETSC_COMM_WORLD,"\nPadded Green cube:\n");CHKERRQ(ierr);
       ierr = VecView(RealSpaceCube,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);    
	//VecView3D(RealSpaceCube,dim[0]);
    }

    // Apply FFTW_FORWARD for the Green cube:
    fftw_execute(fplan);
    PetscPrintf(PETSC_COMM_WORLD,".");

    //FFTW computes an unnormalized DFT, need to scale
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

    ierr = VecDuplicate(RealSpaceCube,&FFTPaddedGreen);CHKERRQ(ierr);
    VecCopy(RealSpaceCube,FFTPaddedGreen);

    PetscFunctionReturn(0);
}
