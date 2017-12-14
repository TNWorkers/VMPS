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

/**This constructs the operators for L_input spins with local dimension D=2S+1.*/
template<typename Symmetry>
class SpinBase
{
	typedef SiteOperator<Symmetry,double> OperatorType;
	
public:
	
	SpinBase(){};
	
	/**
	\param L_input : amount of sites
	\param D_input : \f$D=2S+1\f$
	\param N : if true, the particle number is the good quantum number --> trivial for spins --> all operators have qvacuum()
	*/
	SpinBase (size_t L_input, size_t D_input, bool N_input=false);
	
	/**amount of states = \f$D^L\f$*/
	inline size_t dim() const {return N_states;}
	
	/**\f$D=2S+1\f$*/
	inline size_t get_D() const {return D;}
	
	/**amount of orbitals*/
	inline size_t orbitals() const  {return N_orbitals;}
	
	vector<qarray<Symmetry::Nq> > get_basis() const;
	
	typename Symmetry::qType getQ (SPINOP_LABEL Sa) const;
	
	OperatorType Scomp (SPINOP_LABEL Sa, int orbital=0) const;
	
	OperatorType Id() const;
	
	ArrayXd ZeroField() const;
	
	string alignment (double J) const {return (J<0.)? "(AFM)":"(FM)";};
	
	/**Creates the full Heisenberg (XXZ) Hamiltonian on the supersite.
	\param Jxy : \f$J_{xy}\f$
	\param Jz : \f$J_{z}\f$
	\param Bz : \f$B_{z}\f$
	\param Bx : \f$B_{x}\f$
	\param PERIODIC: periodic boundary conditions if \p true*/
	OperatorType HeisenbergHamiltonian (double Jxy, double Jz, double Bz=0., double Bx=0., double Kz=0., double Kx=0., double Dy=0., 
	                                    bool PERIODIC=false) const;
	
	OperatorType HeisenbergHamiltonian (double Jxy, double Jz, 
	                                    const ArrayXd &Bz, const ArrayXd &Bx, const ArrayXd &Kz, const ArrayXd &Kx, double Dy=0., 
	                                    bool PERIODIC=false) const;
	
	SiteOperator<Symmetry,complex<double> > HeisenbergHamiltonian 
	(Array3d J, Array<double,Dynamic,3> B, Array<double,Dynamic,3> K, Array3d D, bool PERIODIC=false) const;
	
private:
	
	SparseMatrixXd ScompSingleSite (SPINOP_LABEL Sa) const;
	SparseMatrixXd Sbase () const;
	VectorXd       Soffdiag () const;
	
	size_t N_orbitals;
	size_t N_states;
	size_t D;
	bool N;
	
	/**Returns the qarray for a given index of the basis.
	\param index*/
	qarray<Symmetry::Nq> qNums (size_t index) const;
};

template<typename Symmetry>
SpinBase<Symmetry>::
SpinBase (size_t L_input, size_t D_input, bool N_input)
	:N_orbitals(L_input), D(D_input), N(N_input)
{
	assert(N_orbitals >= 1);
	assert(D >= 1);
	N_states = pow(D,N_orbitals);
}

template<typename Symmetry>
SiteOperator<Symmetry,double> SpinBase<Symmetry>::
Scomp (SPINOP_LABEL Sa, int orbital) const
{
	assert(orbital<N_orbitals);
	
	size_t R = ScompSingleSite(Sa).rows();
	size_t Nl = pow(R,orbital);
	size_t Nr = pow(R,N_orbitals-orbital-1);
	
	SparseMatrixXd Il = MatrixXd::Identity(Nl,Nl).sparseView();
	SparseMatrixXd Ir = MatrixXd::Identity(Nr,Nr).sparseView();
	SparseMatrixXd Mout = kroneckerProduct(Il,kroneckerProduct(ScompSingleSite(Sa),Ir));
	
	return OperatorType(Mout,getQ(Sa));
}

template<typename Symmetry>
SiteOperator<Symmetry,double> SpinBase<Symmetry>::
Id() const
{
	SparseMatrixXd mat = MatrixXd::Identity(N_states,N_states).sparseView();
	OperatorType Oout(mat,Symmetry::qvacuum());
	return Oout;
}

template<typename Symmetry>
ArrayXd SpinBase<Symmetry>::
ZeroField() const
{
	return ArrayXd::Zero(N_orbitals);
}

template<typename Symmetry>
SiteOperator<Symmetry,double> SpinBase<Symmetry>::
HeisenbergHamiltonian (double Jxy, double Jz, const ArrayXd &Bz, const ArrayXd &Bx, const ArrayXd &Kz, const ArrayXd &Kx, double Dy, bool PERIODIC) const
{
	assert (Bz.rows() == N_orbitals and Bx.rows() == N_orbitals);
	
	SparseMatrixXd Mout(N_states,N_states);
	
	size_t ilast = (PERIODIC == true and N_orbitals>2)? N_orbitals:N_orbitals-1;
	
	for (int i=0; i<ilast; ++i) // for all bonds
	{
		if (Jxy != 0.)
		{
			Mout += -0.5*Jxy * (Scomp(SP,i).data*Scomp(SM,(i+1)%N_orbitals).data + Scomp(SM,i).data*Scomp(SP,(i+1)%N_orbitals).data);
		}
		if (Jz != 0.)
		{
			Mout += -Jz * Scomp(SZ,i).data*Scomp(SZ,(i+1)%N_orbitals).data;
		}
		if (Dy != 0.)
		{
			Mout += Dy * (Scomp(SX,i).data*Scomp(SZ,(i+1)%N_orbitals).data - Scomp(SZ,i).data*Scomp(SX,(i+1)%N_orbitals).data);
		}
	}
	
	for (int i=0; i<N_orbitals; ++i)
	{
		if (Bz(i) != 0.) {Mout -= Bz(i) * Scomp(SZ,i).data;}
	}
	for (int i=0; i<N_orbitals; ++i)
	{
		if (Bx(i) != 0.) {Mout -= Bx(i) * Scomp(SX,i).data;}
	}
	for (int i=0; i<N_orbitals; ++i)
	{
		if (Kz(i)!=0.) {Mout += Kz(i) * Scomp(SZ,i).data * Scomp(SZ,i).data;}
	}
	for (int i=0; i<N_orbitals; ++i)
	{
		if (Kx(i)!=0.) {Mout += Kx(i) * Scomp(SX,i).data * Scomp(SX,i).data;}
	}
	
	OperatorType Oout(Mout,Symmetry::qvacuum());
	return Oout;
}

template<typename Symmetry>
SiteOperator<Symmetry,double> SpinBase<Symmetry>::
HeisenbergHamiltonian (double Jxy, double Jz, double Bz, double Bx, double Kz, double Kx, double Dy, bool PERIODIC) const
{
	ArrayXd Bzorb(N_orbitals); Bzorb = Bz;
	ArrayXd Bxorb(N_orbitals); Bxorb = Bx;
	ArrayXd Kzorb(N_orbitals); Bzorb = Kz;
	ArrayXd Kxorb(N_orbitals); Kxorb = Kx;
	return HeisenbergHamiltonian(Jxy, Jz, Bzorb, Bxorb, Kzorb, Kxorb, Dy, PERIODIC);
}

template<typename Symmetry>
SiteOperator<Symmetry,complex<double> > SpinBase<Symmetry>::
HeisenbergHamiltonian (Array3d J, Array<double,Dynamic,3> B, Array<double,Dynamic,3> K, Array3d D, bool PERIODIC) const
{
	SiteOperator<Symmetry,complex<double> > Oout = 
	HeisenbergHamiltonian(0.,J(2),B.col(2),B.col(0),K.col(2),K.col(0),D(1),PERIODIC).template cast<complex<double> >();
	
	size_t ilast = (PERIODIC == true and N_orbitals>2)? N_orbitals:N_orbitals-1;
	
	for (size_t i=0; i<ilast; ++i) // for all bonds
	{
		if (J(0) != 0.)
		{
			Oout.data += -J(0) * (Scomp(SX,i).data * Scomp(SX,(i+1)%N_orbitals).data).template cast<complex<double> >();
		}
		if (J(1) != 0.)
		{
			Oout.data += +J[1] * (Scomp(iSY,i).data * Scomp(iSY,(i+1)%N_orbitals)).data.template cast<complex<double> >();
		}
		if (D(0) != 0.)
		{
			Oout.data += D(0) * (-1.i) * (Scomp(iSY,i).data * Scomp(SZ,(i+1)%N_orbitals).data 
			                            -Scomp(SZ,i).data * Scomp(iSY,(i+1)%N_orbitals).data).template cast<complex<double> >();
		}
		if (D(2) != 0.)
		{
			Oout.data += D(2) * (-1.i) * (Scomp(SX,i).data * Scomp(iSY,(i+1)%N_orbitals).data 
			                             -Scomp(iSY,i).data * Scomp(SX,(i+1)%N_orbitals).data).template cast<complex<double> >();
		}
	}
	
	// By
	for (int i=0; i<N_orbitals; ++i)
	{
		if (B(i,2) != 0.) {Oout.data -= B(i,2) * (-1.i) * Scomp(iSY,i).data.template cast<complex<double> >();}
	}
	// Ky
	for (int i=0; i<N_orbitals; ++i)
	{
		if (K(i,1) != 0.) {Oout.data -= K(i,1) * (Scomp(iSY,i).data*Scomp(iSY,i).data).template cast<complex<double> >();}
	}
	
	return Oout;
}

template<typename Symmetry>
qarray<Symmetry::Nq> SpinBase<Symmetry>::
qNums (size_t index) const
{
	NestedLoopIterator Nelly(N_orbitals,D);
	int M = 0;
	Nelly = index;
	
	for (size_t i=0; i<N_orbitals; i++)
	{
		M += D-(2*(Nelly(i)+1)-1);
	}
	
	if constexpr(Symmetry::IS_TRIVIAL){ return qarray<0>{}; }
	else if constexpr(Symmetry::Nq == 1)
					 {
						 if(N) { return qarray<1>{0}; }
						 else { return qarray<1>{M}; }
					 }
	//return a dummy quantum number for a second symmetry. Format: {other symmetry, magnetization}
	else if constexpr(Symmetry::Nq == 2) { return qarray<2>{{0,M}}; }
}

template<typename Symmetry>
vector<qarray<Symmetry::Nq> > SpinBase<Symmetry>::
get_basis() const
{
	vector<qarray<Symmetry::Nq> > vout;
	
	for (size_t i=0; i<N_states; ++i)
	{
		vout.push_back(qNums(i));
	}
	
	return vout;
}

template<typename Symmetry>
typename Symmetry::qType SpinBase<Symmetry>::
getQ (SPINOP_LABEL Sa) const
{
	if constexpr(Symmetry::IS_TRIVIAL) {return {};}
	else if constexpr(Symmetry::Nq == 1)
					 {
						 if(N) { return qarray<1>{0}; }
						 else
						 {
							 typename Symmetry::qType out;
							 if      (Sa==SX)  {out = {0};}
							 else if (Sa==SY)  {out = {0};}
							 else if (Sa==iSY) {out = {0};}
							 else if (Sa==SZ)  {out = {0};}
							 else if (Sa==SP)  {out = {+2};}
							 else if (Sa==SM)  {out = {-2};}
							 return out;
						 }
					 }
	else if constexpr(Symmetry::Nq == 2) //return a dummy quantum number for a second symmetry. Format: {other symmetry, magnetization}
					 {
						 typename Symmetry::qType out;
						 if      (Sa==SX)  {out = qarray<2>({0,0});}
						 else if (Sa==SY)  {out = qarray<2>({0,0});}
						 else if (Sa==iSY) {out = qarray<2>({0,0});}
						 else if (Sa==SZ)  {out = qarray<2>({0,0});}
						 else if (Sa==SP)  {out = qarray<2>({0,+2});}
						 else if (Sa==SM)  {out = qarray<2>({0,-2});}
						 return out;
					 }

}

template<typename Symmetry>
SparseMatrixXd SpinBase<Symmetry>::
ScompSingleSite (SPINOP_LABEL Sa) const
{
	assert(Sa != SY);
	
	if (Sa==SX)
	{
		return Sbase()+SparseMatrixXd(Sbase().transpose());
	}
	else if (Sa==iSY)
	{
		return Sbase()-SparseMatrixXd(Sbase().transpose());
	}
	else if (Sa==SZ) 
	{
		assert(D >= 1);
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

template<typename Symmetry>
SparseMatrixXd SpinBase<Symmetry>::
Sbase () const
{
	assert(D >= 1);
	MatrixXd Mtmp(D,D);
	Mtmp.setZero();
	Mtmp.diagonal<1>() = Soffdiag();
	SparseMatrixXd Mout = Mtmp.sparseView();
	return Mout;
}

template<typename Symmetry>
VectorXd SpinBase<Symmetry>::
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
