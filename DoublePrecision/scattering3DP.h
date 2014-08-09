#include "common.h"

template < class RealVector, class ComplexVector, class Complex, class Real, class Long, class Integer >
class Scattering3DP 
{
    public:
        Integer        NumParticlePerSide;
        Integer        NumParticlePerPlane;
        Integer        NumCubePerSide;
        Integer        NumCubePerPlane;
        Integer        NumParticlePerSmallCubeSide;
        Real           DomainSize;
        Real           SmallCubeSize;
        Real           ParticleRadius;
        Real           ParticleDistance;
        Real           Kappa;
        Long           TotalParticles;
        Integer        TotalCubes;
        Integer        TotalCollocationPoints;
        RealVector     WaveDirection;
        Complex        OriginalRefractionCoef;
        Complex        DesiredRefractionCoef;
        Real           Distribution;
        Real           VolQ;             //Volume of the cube that contains all particles
        Real           SmallCubeVol;
        Complex        p;     
        Complex        h;
        Complex        BoundaryImpedance;

    private:
        Complex        ha2k;             //coeficient of entries in Matrix A
        Integer 	  NumCollocationPointsPerSide;
        Integer 	  NumCollocationPointsPerPlane;
        Integer 	  NumCollocationPointsPerSmallCubeSide;
        RealVector     x;                //Subcubes 3D position in a big cube
        RealVector     y;
        RealVector     z;
        Complex        ***GreenCube;
	Integer        GreenSize;

    public:
        Scattering3DP()
        {

        }

        virtual ~Scattering3DP()
        {
	     DeleteGreenCube();
        }

        void Input(Real ParticleRadius, Real Kappa, RealVector WaveDirection, Real ParticleDistance, Long TotalParticles,\
                   Complex OriginalRefractionCoef, Complex DesiredRefractionCoef, Real Distribution, Real VolQ, Integer TotalCubes, \
                   Integer TotalCollocationPoints)
        {
            this->ParticleRadius = ParticleRadius;
            this->Kappa = Kappa;
            this->WaveDirection = WaveDirection;
            this->ParticleDistance = ParticleDistance;
            this->TotalParticles = TotalParticles;
            this->OriginalRefractionCoef = OriginalRefractionCoef;
            this->DesiredRefractionCoef = DesiredRefractionCoef;
            this->Distribution = Distribution;
            this->VolQ = VolQ; 
            this->TotalCubes = TotalCubes;  
            this->TotalCollocationPoints = TotalCollocationPoints;      
        }

        void Init()
        {            
            // Number of particles on a side of a cube of size 1
            NumParticlePerSide = round(pow(TotalParticles,1.0/3));
            NumParticlePerPlane = pow(NumParticlePerSide,2);
            NumCubePerSide = ceil(pow(TotalCubes,1.0/3));
            NumCubePerPlane = pow(NumCubePerSide,2);
            //DomainSize = (NumParticlePerSide-1)*ParticleDistance;
	     DomainSize = cbrt(VolQ);
            SmallCubeSize = DomainSize/NumCubePerSide;
            SmallCubeVol = pow(SmallCubeSize,3);
            NumParticlePerSmallCubeSide = floor(NumParticlePerSide/NumCubePerSide);

            NumCollocationPointsPerSide = round(pow(TotalCollocationPoints,1.0/3));
            NumCollocationPointsPerPlane = pow(NumCollocationPointsPerSide,2);
            NumCollocationPointsPerSmallCubeSide = floor(NumCollocationPointsPerSide/NumCubePerSide);

            //Cubes positions            
            UniformDistributeCubes();
                      
            p = pow(K,2)*(cpow(OriginalRefractionCoef,2) - cpow(DesiredRefractionCoef,2));
            Real h1 = creal(p)/(PI4*Distribution);
            Real h2 = cimag(p)/(PI4*Distribution);
            h = (h1+I*h2);
            ha2k = h*Distribution*SmallCubeVol;
            BoundaryImpedance = h/pow(ParticleRadius,Kappa);

	     Green3DCube(NumCubePerSide);
        }

    	 inline const Integer Index2Order(Integer m1,Integer m2,Integer m3)
	 {
        	return (m1+m2*NumCubePerSide+m3*NumCubePerPlane);
    	 }

    	 inline void Order2Index(Integer m,Integer& m1,Integer& m2,Integer& m3)
	 {
              Integer red1;

        	m3 = floor(m/NumCubePerPlane);
        	red1 = m%NumCubePerPlane;
        	m2 = floor(red1/NumCubePerSide);
        	m1 = red1%NumCubePerSide;
    	 }

        inline void Cube2Position(Long m,Real& xm,Real& ym,Real& zm)
        {
		Integer m1,m2,m3;

        	Order2Index(m,m1,m2,m3);
            	xm = x[m1];
            	ym = y[m2];
            	zm = z[m3];
    	 }

        void UniformDistributeCubes()
        {
            // Set the position for each cube (uniformly distributed)
            Real x0,y0,z0,t;

            x0 = SmallCubeSize/2;
            y0 = x0;
            z0 = x0;

            // The first small cube [x1,y1,z1] is at the origin
            x = RealVector(NumCubePerSide);
            y = RealVector(NumCubePerSide);
            z = RealVector(NumCubePerSide);
            
            for(Integer s = 0; s < NumCubePerSide; s++)
            {
                t = SmallCubeSize*s;
                x[s] = x0 + t;
                y[s] = y0 + t;
                z[s] = z0 + t;
            }
        }

    	 inline void ParticleOrder2Index(Integer m,Integer& m1,Integer& m2,Integer& m3)
	 {
              Integer red1;

        	m3 = floor(m/NumParticlePerPlane);
        	red1 = m%NumParticlePerPlane;
        	m2 = floor(red1/NumParticlePerSide);
        	m1 = red1%NumParticlePerSide;
    	 }

    	 inline void CollocationPointOrder2Index(Integer m,Integer& m1,Integer& m2,Integer& m3)
	 {
              Integer red1;

        	m3 = floor(m/NumCollocationPointsPerPlane);
        	red1 = m%NumCollocationPointsPerPlane;
        	m2 = floor(red1/NumCollocationPointsPerSide);
        	m1 = red1%NumCollocationPointsPerSide;
    	 }

        Integer FindCube(Integer ParticleNumber)
        {
            //Find cube number that contains particle ParticleNumber        
        
            Integer x1,x2,x3,CubeNumber;

            ParticleOrder2Index(ParticleNumber,x1,x2,x3);

            //Find the index of the small cube containing particle ParticleNumber            
            x1 = floor(x1/NumParticlePerSmallCubeSide); 
            x2 = floor(x2/NumParticlePerSmallCubeSide); 
            x3 = floor(x3/NumParticlePerSmallCubeSide); 

            if(x1>=NumCubePerSide)
                 x1 = NumCubePerSide - 1;
            if(x2>=NumCubePerSide)
                 x2 = NumCubePerSide - 1;
            if(x3>=NumCubePerSide)
                 x3 = NumCubePerSide - 1;

            CubeNumber = x1 + x2*NumCubePerSide + x3*NumCubePerPlane;

            return CubeNumber;                 
        }

        Integer FindCubeOfCollocation(Integer CollocationPoint)
        {
            //Find cube number that contains CollocationPoint
        
            Integer x1,x2,x3,CubeNumber;

	     CollocationPointOrder2Index(CollocationPoint,x1,x2,x3);

            //Find the index of the small cube containing the CollocationPoint            
            x1 = floor(x1/NumCollocationPointsPerSmallCubeSide); 
            x2 = floor(x2/NumCollocationPointsPerSmallCubeSide); 
            x3 = floor(x3/NumCollocationPointsPerSmallCubeSide); 

            if(x1>=NumCubePerSide)
                 x1 = NumCubePerSide - 1;
            if(x2>=NumCubePerSide)
                 x2 = NumCubePerSide - 1;
            if(x3>=NumCubePerSide)
                 x3 = NumCubePerSide - 1;

            CubeNumber = x1 + x2*NumCubePerSide + x3*NumCubePerPlane;

            return CubeNumber;                 
        }


	 inline const Real Length(Real x,Real y,Real z)
	 {
        	return sqrt(x*x + y*y + z*z);
    	 }

        inline const Complex InitField(Long s)
        {
            // Create an inittial field u0 satisfying Helmholtz equation in R^3
	     Real xs,ys,zs;

	     Cube2Position(s,xs,ys,zs);
            return cexp(I*K*(WaveDirection[0]*xs + WaveDirection[1]*ys + WaveDirection[2]*zs)); 
        }
/*
        //Matrix A in Ax=u0 (Performance tune)
        inline const Complex CoefMatFast(Long s, Long t)
        {
            // Generate value for entry A(i,j) in Au=u0
            Real xs,ys,zs,xt,yt,zt;

            if (s==t)
               return 1;

	     Cube2Position(s,xs,ys,zs);
	     Cube2Position(t,xt,yt,zt);
            Real r = Length(xs-xt,ys-yt,zs-zt);

            if(r<=ZERO)
                return 0;

            //Green function*PI4:
            Complex G = cexp(I*K*r)/r;

            return (G*ha2k);
        }
*/
        //Matrix A in Ax=u0 (Performance tune)
        inline const Complex CoefMatFast(Long s, Long t)
        {
            // Generate value for entry A(i,j) in Au=u0

	     Integer s1,s2,s3,t1,t2,t3;

	     Order2Index(s,s1,s2,s3);
	     Order2Index(t,t1,t2,t3);

	     return GreenCube[abs(s1-t1)][abs(s2-t2)][abs(s3-t3)];
        }

        Complex Green3DF(Integer m1,Integer m2,Integer m3)
 	 {
	     //Create a Green function in 3D
        
            Real r = Length(x[m1],y[m2],z[m3]);

            return (ha2k*cexp(I*K*r)/r);
    	 }

        void Green3DCube(Integer N)
        {
	     //Create a Green cube in 3D

	     GreenCube = new Complex**[N];
	     for(Integer m1 = 0; m1 < N; m1++)
	     {
	         GreenCube[m1] = new Complex*[N];
		  for(Integer m2 = 0; m2 < N; m2++)
	         {
		      GreenCube[m1][m2] = new Complex[N];
		  }
	     }	     

	     for(Integer m1 = 0; m1 < N; m1++)
	     {
		  for(Integer m2 = m1; m2 < N; m2++)
	         {
		      for(Integer m3 = 0; m3 < N; m3++)
	     	      {
			   GreenCube[m1][m2][m3] = Green3DF(m1,m2,m3);
			   GreenCube[m2][m1][m3] = GreenCube[m1][m2][m3];
		      }
		  }
	     }
            GreenCube[0][0][0] = 1;
	     GreenSize = NumCubePerSide;
	 }

       void DeleteGreenCube()
       {
	     if(GreenCube == NULL)
		 return;

	     Integer m1,m2;

	     for(m1 = 0; m1 < GreenSize; m1++)
	     {
		  for(m2 = 0; m2 < GreenSize; m2++)
	         {
		      delete [] GreenCube[m1][m2];
		  }
	         delete [] GreenCube[m1];
	     }
	     delete [] GreenCube;
	     GreenCube = NULL;
	}

};

