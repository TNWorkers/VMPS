#ifndef SPINBASE
#define SPINBASE

#include <Eigen/Dense>
#include <complex>

enum SPINOP_LABEL {SX, SY, iSY, SZ, SP, SM};

std::ostream& operator<< (std::ostream& s, SPINOP_LABEL Sa)
{
	if      (Sa==SX)  {s << "Sx";}
	else if (Sa==SY)  {s << "Sy";}
	else if (Sa==iSY) {s << "iSy";}
	else if (Sa==SZ)  {s << "Sz";}
	else if (Sa==SP)  {s << "S+";}
	else if (Sa==SM)  {s << "S-";}
	return s;
}

struct SpinBase
{
	static const Eigen::MatrixXd Scomp (SPINOP_LABEL Sa, size_t D=2)
	{
		assert(Sa != SY and D >= 2);
		
		if (Sa==SX)
		{
			return Sbase(D) + Sbase(D).transpose();
		}
		else if (Sa==iSY)
		{
			return -Sbase(D) + Sbase(D).transpose();
		}
		else if (Sa==SZ) 
		{
			assert(D >= 2);
			Eigen::MatrixXd Mout(D,D);
			Mout.setZero();
			double S = 0.5*(D-1);
			for (size_t i=0; i<D; ++i)
			{
				double M = S-i;
				Mout(i,i) = M;
			}
			return Mout;
		}
		else if (Sa==SP) 
		{
			return 2.*Sbase(D);
		}
		else if (Sa==SM) 
		{
			return 2.*Sbase(D).transpose();
		}
	}
	
	static const Eigen::MatrixXd Sbase (size_t D)
	{
		assert(D >= 2);
		Eigen::MatrixXd Mout(D,D);
		Mout.setZero();
		Mout.diagonal<1>() = Soffdiag(D);
		return Mout;
	}
	
	static const Eigen::VectorXd Soffdiag (size_t D)
	{
		Eigen::VectorXd Vout(D-1);
		double S = 0.5*(D-1);
		
		for (size_t i=0; i<D-1; ++i)
		{
			double m = -S + static_cast<double>(i);
			Vout(i) = 0.5*sqrt(S*(S+1.)-m*(m+1.));
		}
		return Vout;
	}
};

#endif
