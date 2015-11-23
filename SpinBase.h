#ifndef SPINBASE
#define SPINBASE

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <complex>
#include "DmrgTypedefs.h"
#include "qarray.h"
#include "NestedLoopIterator.h"

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

/**This constructs the operators for a Ly Spins with quantum number D.*/
class SpinBase
{
public:
	
	SpinBase(){};
	
	SpinBase(size_t L_input,size_t D_input);

	/**amount of states = \f$D^L\f$*/
	inline size_t dim() const {return N_states;}
	
	/**amount of orbitals*/
	inline size_t orbitals() const  {return N_orbitals;}

	SparseMatrixXd Scomp (SPINOP_LABEL Sa, int orbital=0) const;

	/**Creates the full Heisenberg Hamiltonian on the supersite.
	   \param J : \f$J\f$
	   \param Bz : \f$B_{z}\f$
	   \param Bx : \f$B_{x}\f$
	   \param Jprime : \f$J^{\prime}\f$ (next nearest neigbour interaction)*/
	SparseMatrixXd HeisenbergHamiltonian (double J, double Bz=0., double Bx=0., double Jprime=0.) const;

	/**Returns the qarray for a given index of the basis
	   \param index*/
	qarray<1> qNums(size_t index);
	
	SparseMatrixXd ScompLocal (SPINOP_LABEL Sa) const;
	SparseMatrixXd Sbase () const;
	VectorXd Soffdiag () const;

private:
	
	size_t N_orbitals;
	size_t N_states;
	size_t D;
	
};

SpinBase::
SpinBase (size_t L_input, size_t D_input)
	:N_orbitals(L_input),D(D_input)
{
	assert(N_orbitals >= 1);
	
	N_states = pow(D,N_orbitals);
}

SparseMatrixXd SpinBase::
Scomp (SPINOP_LABEL Sa, int orbital) const
{
    assert(orbital<N_orbitals);

    size_t R = ScompLocal(Sa).rows();
    size_t Nl = R*orbital;
    size_t Nr = R*(N_orbitals-orbital-1);

    SparseMatrixXd Il = MatrixXd::Identity(Nl,Nl).sparseView();
    SparseMatrixXd Ir = MatrixXd::Identity(Nr,Nr).sparseView();

    // all = 0
    if (Nl == 0 and Nr == 0)
    {
        return ScompLocal(Sa);
    }
    // one != 0
    else if (Nl == 0 and Nr != 0)
    {
        return kroneckerProduct(ScompLocal(Sa),Ir);
    }
    else if (Nl != 0 and Nr == 0)
    {
        return kroneckerProduct(Il,ScompLocal(Sa));
    }
    // all != 0
    else
    {
        return kroneckerProduct(Il,kroneckerProduct(ScompLocal(Sa),Ir));
    }

}

SparseMatrixXd SpinBase::
HeisenbergHamiltonian (double J, double Bz, double Bx, double Jprime) const
{
	SparseMatrixXd Mout(N_states,N_states);
	
	for (int i=0; i<N_orbitals-1; ++i) // for all bonds
	{
		if (J != 0.)
		{
			SparseMatrixXd Mout = -J* (Scomp(SZ,i)*Scomp(SZ,i+1) + 0.5* (Scomp(SP,i)*Scomp(SM,i+1) + Scomp(SM,i)*Scomp(SP,i+1)) );
		}
		if (Jprime != 0. and i != N_orbitals-1)
		{
			Mout += -Jprime* (Scomp(SZ,i)*Scomp(SZ,i+2) + 0.5* (Scomp(SP,i)*Scomp(SM,i+2) + Scomp(SM,i)*Scomp(SP,i+2)) );
		}
	}
	if (Bz != 0.)
	{
		for (int i=0; i<N_orbitals; ++i) {Mout += Bz*Scomp(SZ,i);}
	}
	if (Bz != 0.)
	{
		for (int i=0; i<N_orbitals; ++i) {Mout += Bx*Scomp(SX,i);}
	}

	return Mout;
}

qarray<1> SpinBase::
qNums(size_t index)
{
	NestedLoopIterator Nelly(N_orbitals,D);
	int M=0;
	Nelly = index;
	for (size_t i=0; i<N_orbitals; i++)
	{
		M += D - (2*(Nelly(i)+1) - 1); 
	}
	return qarray<1>{M};
}

SparseMatrixXd SpinBase::
ScompLocal (SPINOP_LABEL Sa) const
{
	assert(Sa != SY and D >= 2);
		
	if (Sa==SX)
	{
		return Sbase() + SparseMatrixXd(Sbase().transpose());
	}
	else if (Sa==iSY)
	{
		return -Sbase() + SparseMatrixXd(Sbase().transpose());
	}
	else if (Sa==SZ) 
	{
		assert(D >= 2);
		SparseMatrixXd Mout(D,D);
		double S = 0.5*(D-1);
		for (size_t i=0; i<D; ++i)
		{
			double M = S-i;
			Mout.insert(i,i) = M;
		}
		return Mout;
	}
	else if (Sa==SP) 
	{
		return 2.*Sbase();
	}
	else if (Sa==SM) 
	{
		return SparseMatrixXd(2.*Sbase().transpose());
	}
}
	
SparseMatrixXd SpinBase::
Sbase () const
{
	assert(D >= 2);
	MatrixXd Mtmp(D,D);
	Mtmp.setZero();
	Mtmp.diagonal<1>() = Soffdiag();
	SparseMatrixXd Mout = Mtmp.sparseView();
	return Mout;
}
	
VectorXd SpinBase::
Soffdiag () const
{
	VectorXd Vout(D-1);
	double S = 0.5*(D-1);
		
	for (size_t i=0; i<D-1; ++i)
	{
		double m = -S + static_cast<double>(i);
		Vout(i) = 0.5*sqrt(S*(S+1.)-m*(m+1.));
	}
	return Vout;
}
	
//	static const MatrixXd Scomp (SPINOP_LABEL Sa, size_t D=2)
//	{
//		assert(Sa != SY and D >= 2);
//		
//		if (Sa==SX)
//		{
//			return Sbase(D) + Sbase(D).transpose();
//		}
//		else if (Sa==iSY)
//		{
//			return -Sbase(D) + Sbase(D).transpose();
//		}
//		else if (Sa==SZ) 
//		{
//			assert(D >= 2);
//			MatrixXd Mout(D,D);
//			Mout.setZero();
//			double S = 0.5*(D-1);
//			for (size_t i=0; i<D; ++i)
//			{
//				double M = S-i;
//				Mout(i,i) = M;
//			}
//			return Mout;
//		}
//		else if (Sa==SP) 
//		{
//			return 2.*Sbase(D);
//		}
//		else if (Sa==SM) 
//		{
//			return 2.*Sbase(D).transpose();
//		}
//	}
//	
//	static const MatrixXd Sbase (size_t D)
//	{
//		assert(D >= 2);
//		MatrixXd Mout(D,D);
//		Mout.setZero();
//		Mout.diagonal<1>() = Soffdiag(D);
//		return Mout;
//	}
//	
//	static const VectorXd Soffdiag (size_t D)
//	{
//		VectorXd Vout(D-1);
//		double S = 0.5*(D-1);
//		
//		for (size_t i=0; i<D-1; ++i)
//		{
//			double m = -S + static_cast<double>(i);
//			Vout(i) = 0.5*sqrt(S*(S+1.)-m*(m+1.));
//		}
//		return Vout;
//	}

#endif
