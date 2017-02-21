#ifndef SPINBASE
#define SPINBASE

#include <complex>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/KroneckerProduct>

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

/**This constructs the operators for a L_input spins with local dimension D=2S+1.*/
class SpinBase
{
public:
	
	SpinBase(){};
	
	/**
	\param L_input : amount of sites
	\param D_input : \f$D=2S+1\f$*/
	SpinBase (size_t L_input, size_t D_input);
	
	/**amount of states = \f$D^L\f$*/
	inline size_t dim() const {return N_states;}
	
	/**\f$D=2S+1\f$*/
	inline size_t get_D() const {return D;}
	
	/**amount of orbitals*/
	inline size_t orbitals() const  {return N_orbitals;}
	
	SparseMatrixXd Scomp (SPINOP_LABEL Sa, int orbital=0) const;
	
	/**Creates the full Heisenberg (XXZ) Hamiltonian on the supersite.
	\param Jxy : \f$J_{xy}\f$
	\param Jz : \f$J_{z}\f$
	\param Bz : \f$B_{z}\f$
	\param Bx : \f$B_{x}\f$
	\param PERIODIC: periodic boundary conditions if \p true*/
	SparseMatrixXd HeisenbergHamiltonian (double Jxy, double Jz, double Bz=0., double Bx=0., double K=0., bool PERIODIC=false) const;
	
	SparseMatrixXd HeisenbergHamiltonian (double Jxy, double Jz, const VectorXd &Bz, const VectorXd &Bx, double K, bool PERIODIC=false) const;
	
	/**Returns the qarray for a given index of the basis.
	\param index*/
	qarray<1> qNums (size_t index);
	
	SparseMatrixXd ScompSingleSite (SPINOP_LABEL Sa) const;
	SparseMatrixXd Sbase () const;
	VectorXd       Soffdiag () const;
	
private:
	
	size_t N_orbitals;
	size_t N_states;
	size_t D;
};

SpinBase::
SpinBase (size_t L_input, size_t D_input)
:N_orbitals(L_input), D(D_input)
{
	assert(N_orbitals >= 1);
	assert(D >= 2);
	N_states = pow(D,N_orbitals);
}

SparseMatrixXd SpinBase::
Scomp (SPINOP_LABEL Sa, int orbital) const
{
	assert(orbital<N_orbitals);
	
	size_t R = ScompSingleSite(Sa).rows();
	size_t Nl = pow(R,orbital);
	size_t Nr = pow(R,N_orbitals-orbital-1);
	
	SparseMatrixXd Il = MatrixXd::Identity(Nl,Nl).sparseView();
	SparseMatrixXd Ir = MatrixXd::Identity(Nr,Nr).sparseView();
	
	return kroneckerProduct(Il,kroneckerProduct(ScompSingleSite(Sa),Ir));
}

SparseMatrixXd SpinBase::
HeisenbergHamiltonian (double Jxy, double Jz, const VectorXd &Bz, const VectorXd &Bx, double K, bool PERIODIC) const
{
	assert (Bz.rows() == N_orbitals and Bx.rows() == N_orbitals);
	
	SparseMatrixXd Mout(N_states,N_states);

	for (int i=0; i<N_orbitals-1; ++i) // for all bonds
	{
		if (Jxy != 0.)
		{
			Mout += -0.5*Jxy * (Scomp(SP,i)*Scomp(SM,i+1) + Scomp(SM,i)*Scomp(SP,i+1));
		}
		if (Jz != 0.)
		{
			Mout += -Jz * Scomp(SZ,i)*Scomp(SZ,i+1);
		}
	}
	if (PERIODIC == true and N_orbitals>2)
	{
		if (Jxy != 0.)
		{
			Mout += -0.5*Jxy * (Scomp(SP,0)*Scomp(SM,N_orbitals-1) + Scomp(SM,0)*Scomp(SP,N_orbitals-1));
		}
		if (Jz != 0.)
		{
			Mout += -Jz * Scomp(SZ,0)*Scomp(SZ,N_orbitals-1);
		}
	}
	for (int i=0; i<N_orbitals; ++i)
	{
		if (Bz(i) != 0.) {Mout -= Bz(i) * Scomp(SZ,i);}
	}
	for (int i=0; i<N_orbitals; ++i)
	{
		if (Bx(i) != 0.) {Mout -= Bx(i) * Scomp(SX,i);}
	}
	if (K!=0.)
	{
		for (int i=0; i<N_orbitals; ++i)
		{
			Mout += K * Scomp(SZ,i) * Scomp(SZ,i);
		}
	}
	
	return Mout;
}

SparseMatrixXd SpinBase::
HeisenbergHamiltonian (double Jxy, double Jz, double Bz, double Bx, double K, bool PERIODIC) const
{
	VectorXd Bzvec(N_orbitals); Bzvec.setConstant(Bz);
	VectorXd Bxvec(N_orbitals); Bxvec.setConstant(Bx);
	return HeisenbergHamiltonian(Jxy, Jz, Bzvec, Bxvec, K, PERIODIC);
}

qarray<1> SpinBase::
qNums (size_t index)
{
	NestedLoopIterator Nelly(N_orbitals,D);
	int M = 0;
	Nelly = index;
	for (size_t i=0; i<N_orbitals; i++)
	{
		M += D-(2*(Nelly(i)+1)-1); 
	}
	return qarray<1>{M};
}

SparseMatrixXd SpinBase::
ScompSingleSite (SPINOP_LABEL Sa) const
{
	assert(Sa != SY);
	
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

#endif
