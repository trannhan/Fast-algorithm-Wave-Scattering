#include "common.h"

template < class RealVector, class ComplexVector, class Complex, class Real, class Long, class Integer >
class Scattering3DS 
{
    public:
        Integer        NumParticlePerSide;
        Integer        NumParticlePerPlane;
        Real           ParticleRadius;
        Real           ParticleDistance;
        Real           Kappa;
        Long           TotalParticles;
        RealVector     WaveDirection;
        Complex        OriginalRefractionCoef;
        Complex        DesiredRefractionCoef;
        Real           Distribution;
        Real           VolQ;  //Volume of the cube that contains all particles
        Complex        p;     
        Complex        h;
        Complex        BoundaryImpedance;
        //Complex        ***GreenCube;
	//Integer        GreenSize;
        //Complex        *GreenVec;

    private:
        //Performance tune:
        Real           a2k;      //coeficient of entries in Matrix A
        Complex        ha2k;     //coeficient of entries in Matrix A
        RealVector     x;        //Particles 3D position in a big cube
        RealVector     y;
        RealVector     z;

    public:
        Scattering3DS()
        {

        }

        virtual ~Scattering3DS()
        {
	     //DeleteGreenCube();
        }

        void Input(Real ParticleRadius, Real Kappa, RealVector WaveDirection, Real ParticleDistance, Long TotalParticles,\
                   Complex OriginalRefractionCoef, Complex DesiredRefractionCoef, Real Distribution, Real VolQ)
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
        }

        void Init()
        {            
            // Number of particles on a side of a cube of size 1
            NumParticlePerSide = round(pow(TotalParticles,1.0/3));
            NumParticlePerPlane = NumParticlePerSide*NumParticlePerSide;

            //Particles positions            
            UniformDistributeParticles();
            
            a2k = pow(ParticleRadius,(2-Kappa));            
            p = pow(K,2)*(cpow(OriginalRefractionCoef,2) - cpow(DesiredRefractionCoef,2));
            Real h1 = creal(p)/(PI4*Distribution);
            Real h2 = cimag(p)/(PI4*Distribution);
            h = (h1+I*h2);
            ha2k = h*a2k;
            BoundaryImpedance = h/pow(ParticleRadius,Kappa);

	    //Green3DCube(NumParticlePerSide);
	    //PaddedGreen3DCube(NumParticlePerSide);
	    //CreateGreenVec();
        }

    	 inline const Integer Index2Order(Integer m1,Integer m2,Integer m3)
	 {
            return (m1+m2*NumParticlePerSide+m3*NumParticlePerPlane);
    	 }

    	 inline void Order2Index(Integer m,Integer& m1,Integer& m2,Integer& m3)
	 {
            Integer red1;

            m3 = floor(m/NumParticlePerPlane);
            red1 = m%NumParticlePerPlane;
            m2 = floor(red1/NumParticlePerSide);
            m1 = red1%NumParticlePerSide;
    	 }

        inline void Particle2Position(Long m,Real& xm,Real& ym,Real& zm)
        {
	    Integer m1,m2,m3;

            Order2Index(m,m1,m2,m3);
/*
            xm = (m1)*ParticleDistance;
            ym = (m2)*ParticleDistance;
            zm = (m3)*ParticleDistance;
*/
            xm = x[m1];
            ym = y[m2];
            zm = z[m3];
    	 }

        void UniformDistributeParticles()
        {
            // Set the position for each particle (uniformly distributed)
            Real t;

            // The first particle [x1,y1,z1] is at the origin
            x = RealVector(NumParticlePerSide);
            y = RealVector(NumParticlePerSide);
            z = RealVector(NumParticlePerSide);
            
            for(Integer s = 1; s < NumParticlePerSide; s++)
            {
		t = ParticleDistance*s;
                x[s] = t;
                y[s] = t;
                z[s] = t;
            }
        }

	inline const Real Length(Real x,Real y,Real z)
	{
            return sqrt(x*x + y*y + z*z);
    	}

        inline const Complex InitField(Long m)
        {
            // Create an inittial field u0 satisfying Helmholtz equation in R^3
	    Integer m1,m2,m3;
             
            Order2Index(m,m1,m2,m3); 
            Real tmp = (K*(WaveDirection[0]*x[m1] + WaveDirection[1]*y[m2] + WaveDirection[2]*z[m3]));           
            return cexp(I*tmp);
        }

        inline const Complex InitField(Long m1, Long m2, Long m3)
        {
            // Create an inittial field u0 satisfying Helmholtz equation in R^3

            return cexp(I*K*(WaveDirection[0]*x[m1] + WaveDirection[1]*y[m2] + WaveDirection[2]*z[m3])); 
        }
/*
        //Matrix A in Ax=u0 (Performance tune)
        inline const Complex CoefMatFast(Long s, Long t)
        {
            // Generate value for entry A(i,j) in Au=u0

	    Integer s1,s2,s3,t1,t2,t3;

	    Order2Index(s,s1,s2,s3);
	    Order2Index(t,t1,t2,t3);

	    return GreenCube[abs(s1-t1)][abs(s2-t2)][abs(s3-t3)];
        }
*/
        inline const Complex Green3DF(Integer m1,Integer m2,Integer m3)
 	{
	    //Create a Green function in 3D
        
            Real r = Length(x[m1],y[m2],z[m3]);

            return cexp(I*K*r)*(ha2k/r);
    	}
/*
        void Green3DCube(Integer N)
        {
	     //Create a Green cube in 3D
	     Integer m1,m2,m3;

	     GreenCube = new Complex**[N];
	     for(m1 = 0; m1 < N; m1++)
	     {
	         GreenCube[m1] = new Complex*[N];
		  for(m2 = 0; m2 < N; m2++)
	         {
		      GreenCube[m1][m2] = new Complex[N];
		  }
	     }	     

	     for(m1 = 0; m1 < N; m1++)
	     {
		  for(m2 = m1; m2 < N; m2++)
	         {
		      for(m3 = 0; m3 < N; m3++)
	     	      {
			   GreenCube[m1][m2][m3] = Green3DF(m1,m2,m3);
			   GreenCube[m2][m1][m3] = GreenCube[m1][m2][m3];
		      }
		  }
	     }
            GreenCube[0][0][0] = 1;
	    GreenSize = N;
	 }

        void PaddedGreen3DCube(Integer n)
        {
	     //Create a Green cube in 3D

	     Integer N = 2*n-2;
	     Integer m1,m2,m3;

	     GreenCube = new Complex**[N];
	     for(m1 = 0; m1 < N; m1++)
	     {
	         GreenCube[m1] = new Complex*[N];
		  for(m2 = 0; m2 < N; m2++)
	         {
		      GreenCube[m1][m2] = new Complex[N];
		  }
	     }	     

	     for(m1 = 0; m1 < n; m1++)
	     {
		  for(m2 = m1; m2 < n; m2++)
	         {
		      for(m3 = 0; m3 < n; m3++)
	     	      {
			   GreenCube[m1][m2][m3] = Green3DF(m1,m2,m3);
			   GreenCube[m2][m1][m3] = GreenCube[m1][m2][m3];
		      }
		  }
	     }
            GreenCube[0][0][0] = 1;
	      
	     //Pad the cube to do convolution:
	     for(m1=n;m1<N;m1++)
	     {
	        for(m2=0;m2<n;m2++)
	        {
		      for(m3=0;m3<n;m3++)
		      {
	 	 	  GreenCube[m1][m2][m3] = GreenCube[2*n-m1-1][m2][m3];
		      }
	        }
	     }
	     for(m1=0;m1<N;m1++)
	     {
	        for(m2=n;m2<N;m2++)
	        {
		      for(m3=0;m3<n;m3++)
		      {
	 	 	  GreenCube[m1][m2][m3] = GreenCube[m1][2*n-m2-1][m3];
		      }
	        }
	     }
	     for(m1=0;m1<N;m1++)
	     {
	        for(m2=0;m2<N;m2++)
	        {
		      for(m3=n;m3<N;m3++)
		      {
	 	 	  GreenCube[m1][m2][m3] = GreenCube[m1][m2][2*n-m3-1];
		      }
	        }
	     }
	     GreenSize = N;
	 }

        void ViewGreenCube()
        {
	    //View original Green cube:
	    string str;

	    for(Integer t=0;t<NumParticlePerSide;t++)
	    {
	       for(Integer j=0;j<NumParticlePerSide;j++)
	       {
		     for(Integer s=0;s<NumParticlePerSide;s++)
		     {				
			 str = (cimag(GreenCube[j][s][t])>=0)?"+":"";
			 cout<<creal(GreenCube[j][s][t])<<str<<cimag(GreenCube[j][s][t])<<"i\t";		  		
		     }
		     cout<<"\n";
	       }
		cout<<"\n";
	    }	
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

        void CreateGreenVec()
        {
	     Integer N = pow(GreenSize,3);
	     Integer m1,m2,m3,m;

	     GreenVec = new Complex[N];
	     m = 0;

	     for(m1=0;m1<GreenSize;m1++)
	     {
	         for(m2=0;m2<GreenSize;m2++)
	         {		     
		     for(m3=0;m3<GreenSize;m3++)
		     {				
		  	  GreenVec[m++] = GreenCube[m1][m2][m3];
		     }
	         }
	     }	
	 }
*/
/*
    	inline const ComplexVector Convert3DTo1D(ComplexCube Cube,Integer N)  
	{      
            ComplexVector Vec = ComplexVector(pow(N,3));
                
	     for(Integer m3 = 1; m3 < N; m3++)
	     {
		  for(Integer m2 = 1; m2 < N; m2++)
	         {
		      for(Integer m1 = 1; m1 < N; m1++)
	     	      {
			   m = Index2Order(m1,m2,m3);
			   Vec(m) = Cube(m1,m2,m3);
		      }
		  }
	     }	
	     return Vec;
    	}

       inline const ComplexCube Convert1DTo3D(ComplexVector Vec,Integer N)
       { 
            n = round(pow(N,1.0/3));
            Cube = ComplexCube(n,n,n);
                
	     for(Integer m = 1; m < N; m++)
	     {
            	  Order2Index(m,m1,m2,m3);
            	  Cube(m1,m2,m3) = Vec(m);
            }
    	}
*/

};
