#ifndef SPINBASE
#define SPINBASE

#include <complex>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/KroneckerProduct>

#include "DmrgTypedefs.h" // for SPIN_INDEX, SPINOP_LABEL
#include "symmetry/qarray.h"
#include "NestedLoopIterator.h" // from HELPERS

#include "symmetry/U0.h"
#include "symmetry/U1.h"

/** 
 * \class SpinBase
  * \ingroup Bases
  *
  * This class provides the local operators for spins (magnitude \p D) for \p N_Orbitals sites.
  *
  * The class is implemented for all combinations of U1 symmetries in the file SpinBase.h.
  * For the different non abelian symmetries, their are template specialisations in the files SpinBase!Symmetry!.h
  *
  * \describe_Symmetry
  *
  */
template<typename Symmetry>
class SpinBase
{
	typedef SiteOperator<Symmetry,double> OperatorType;
	
public:
	
	SpinBase(){};
	
	/**
	 * \param L_input : amount of sites
	 * \param D_input : \f$D=2S+1\f$
	 */
	SpinBase (size_t L_input, size_t D_input);
	
	/**amount of states = \f$D^L\f$*/
	inline size_t dim() const {return N_states;}
	
	/**\f$D=2S+1\f$*/
	inline size_t get_D() const {return D;}
	
	/**amount of orbitals*/
	inline size_t orbitals() const  {return N_orbitals;}

	/**Returns the local basis.*/
	vector<qarray<Symmetry::Nq> > get_basis() const;

	/**Returns the quantum numbers of the operators for the different combinations of U1 symmetries.*/
	typename Symmetry::qType getQ (SPINOP_LABEL Sa) const;
	
	OperatorType Scomp (SPINOP_LABEL Sa, int orbital=0) const;
	
	OperatorType Id() const;
	
	ArrayXd ZeroField() const;
	
	string alignment (double J) const {return (J<0)? "(AFM)":"(FM)";};
	
	/**
	 * Creates the full Heisenberg (XXZ) Hamiltonian on the supersite.
	 * \param Jxy : \f$J^{xy}\f$
	 * \param Jz : \f$J^{z}\f$
	 * \param Bz : \f$B^{z}\f$
	 * \param Bx : \f$B^{x}\f$
	 * \param Kz : \f$K^{z}\f$
	 * \param Kx : \f$K^{x}\f$
	 * \param Dy : \f$D^{y}\f$
	 * \param PERIODIC: periodic boundary conditions if \p true
	 */
	OperatorType HeisenbergHamiltonian (double Jxy, double Jz, double Bz=0., double Bx=0., double Kz=0., double Kx=0., double Dy=0., 
	                                    bool PERIODIC=false) const;

	/**
	 * Creates the full Heisenberg (XXZ) Hamiltonian on the supersite.
	 * \param Jxy : \f$J^{xy}\f$
	 * \param Jz : \f$J^{z}\f$
	 * \param Bz : \f$B^{z}_i\f$
	 * \param Bx : \f$B^{x}_i\f$
	 * \param Kz : \f$K^{z}_i\f$
	 * \param Kx : \f$K^{x}_i\f$
	 * \param Dy : \f$D^{y}\f$
	 * \param PERIODIC: periodic boundary conditions if \p true
	 */
	OperatorType HeisenbergHamiltonian (double Jxy, double Jz, 
	                                    const ArrayXd &Bz, const ArrayXd &Bx, const ArrayXd &Kz, const ArrayXd &Kx, double Dy=0., 
	                                    bool PERIODIC=false) const;

	/**
	 * Creates the full Heisenberg (XYZ) Hamiltonian on the supersite.
	 * \param J : \f$J^{\alpha}\f$, \f$\alpha \in \{x,y,z\} \f$
	 * \param B : \f$B^{\alpha}_i\f$, \f$\alpha \in \{x,y,z\} \f$
	 * \param K : \f$K^{\alpha}_i\f$, \f$\alpha \in \{x,y,z\} \f$
	 * \param D : \f$D^{\alpha}\f$, \f$\alpha \in \{x,y,z\} \f$
	 * \param PERIODIC: periodic boundary conditions if \p true
	 */
	SiteOperator<Symmetry,complex<double> > HeisenbergHamiltonian 
	(Array3d J, Array<double,Dynamic,3> B, Array<double,Dynamic,3> K, Array3d D, bool PERIODIC=false) const;
	
private:
	
	SparseMatrixXd ScompSingleSite (SPINOP_LABEL Sa) const;
	SparseMatrixXd Sbase () const;
	VectorXd       Soffdiag () const;
	
	size_t N_orbitals;
	size_t N_states;
	size_t D;
	
	/**
	 * Returns the qarray for a given index of the basis.
	 * \param index
	 */
	qarray<Symmetry::Nq> qNums (size_t index) const;
};

template<typename Symmetry>
SpinBase<Symmetry>::
SpinBase (size_t L_input, size_t D_input)
:N_orbitals(L_input), D(D_input)
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
	
	if constexpr (Symmetry::IS_TRIVIAL) {return qarray<0>{};}
	else if constexpr (Symmetry::Nq == 1)
	{
		if constexpr (Symmetry::kind()[0] == Sym::KIND::N or Symmetry::kind()[0] == Sym::KIND::T) {return Symmetry::qvacuum();}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::M) {return qarray<1>{M};}
		else {assert(false and "Ill defined KIND of the used Symmetry.");}
	}
	//return a dummy quantum number for a second symmetry. Format: {other symmetry, magnetization}
	else if constexpr(Symmetry::Nq==2)
	{
		if constexpr (Symmetry::kind()[0] == Sym::KIND::N or Symmetry::kind()[0] == Sym::KIND::T) {return qarray<2>{{Symmetry::qvacuum()[0],M}};}
		else if constexpr (Symmetry::kind()[1] == Sym::KIND::N or Symmetry::kind()[1] == Sym::KIND::T) {return qarray<2>{{M,Symmetry::qvacuum()[1]}};}
		else {assert(false and "Ill defined KIND of the used Symmetry.");}
	}
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
	typename Symmetry::qType out;
	
	if constexpr(Symmetry::IS_TRIVIAL) {return {};}
	else if constexpr (Symmetry::Nq == 1)
	{
		if constexpr (Symmetry::kind()[0] == Sym::KIND::N or Symmetry::kind()[0] == Sym::KIND::T) {return Symmetry::qvacuum();}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::M)
		{
			if      (Sa==SX)  {out = {0};}
			else if (Sa==SY)  {out = {0};}
			else if (Sa==iSY) {out = {0};}
			else if (Sa==SZ)  {out = {0};}
			else if (Sa==SP)  {out = {+2};}
			else if (Sa==SM)  {out = {-2};}
			return out;
		}
		else {assert(false and "Ill defined KIND of the used Symmetry.");}
	}
	else if constexpr(Symmetry::Nq == 2) // return a dummy quantum number for a second symmetry. Format: {other symmetry, magnetization}
	{
		if constexpr (Symmetry::kind()[0] == Sym::KIND::N or Symmetry::kind()[0] == Sym::KIND::T)
					 {
						 if      (Sa==SX)  {out = qarray<2>({Symmetry::qvacuum()[0],0});}
						 else if (Sa==SY)  {out = qarray<2>({Symmetry::qvacuum()[0],0});}
						 else if (Sa==iSY) {out = qarray<2>({Symmetry::qvacuum()[0],0});}
						 else if (Sa==SZ)  {out = qarray<2>({Symmetry::qvacuum()[0],0});}
						 else if (Sa==SP)  {out = qarray<2>({Symmetry::qvacuum()[0],+2});}
						 else if (Sa==SM)  {out = qarray<2>({Symmetry::qvacuum()[0],-2});}
					 }
		else if constexpr (Symmetry::kind()[1] == Sym::KIND::N or Symmetry::kind()[1] == Sym::KIND::T)
						  {
							  if      (Sa==SX)  {out = qarray<2>({0,Symmetry::qvacuum()[1]});}
							  else if (Sa==SY)  {out = qarray<2>({0,Symmetry::qvacuum()[1]});}
							  else if (Sa==iSY) {out = qarray<2>({0,Symmetry::qvacuum()[1]});}
							  else if (Sa==SZ)  {out = qarray<2>({0,Symmetry::qvacuum()[1]});}
							  else if (Sa==SP)  {out = qarray<2>({+2,Symmetry::qvacuum()[1]});}
							  else if (Sa==SM)  {out = qarray<2>({-2,Symmetry::qvacuum()[1]});}
						  }
		else {assert(false and "Ill defined KIND of the used Symmetry.");}
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
