#ifndef SPINBASE
#define SPINBASE

/// \cond
#include <complex>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/KroneckerProduct>
#include <unsupported/Eigen/MatrixFunctions>
/// \endcond

#include "DmrgTypedefs.h" // for SPIN_INDEX, SPINOP_LABEL
#include "symmetry/kind_dummies.h"
//include "symmetry/qarray.h"
//include "NestedLoopIterator.h" // from HELPERS

//include "symmetry/U0.h"
//include "symmetry/U1.h"

#include "tensors/SiteOperator.h"

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
	Qbasis<Symmetry> get_structured_basis() const;

	/**Returns the quantum numbers of the operators for the different combinations of U1 symmetries.*/
	typename Symmetry::qType getQ (SPINOP_LABEL Sa) const;

	/**Returns the label of the operators.*/
	std::string label (SPINOP_LABEL Sa) const;
	
	OperatorType Scomp (SPINOP_LABEL Sa, int orbital=0) const;
	
	SiteOperator<Symmetry,complex<double>> Rcomp (SPINOP_LABEL Sa, int orbital=0) const;
	
	OperatorType n (int orbital=0) const;
	
	OperatorType sign (int orb1=0, int orb2=0) const;
	
	OperatorType Id() const;
	
	OperatorType beadz() const;
	
	/**Returns an array of size dim() with zeros.*/
	ArrayXd ZeroField() const { return ArrayXd::Zero(N_orbitals); }
	
	/**Returns an array of size dim()xdim() with zeros.*/
	ArrayXXd ZeroHopping() const { return ArrayXXd::Zero(N_orbitals,N_orbitals); }
	
	string alignment (double J) const {return (J<0)? "(AFM)":"(FM)";};
	
	/**
	 * Creates the full Heisenberg (XXZ) Hamiltonian on the supersite.
	 * \param Jxy : \f$J^{xy}\f$
	 * \param Jz : \f$J^{z}\f$
	 * \param Bz : \f$B^{z}_i\f$
	 * \param Bx : \f$B^{x}_i\f$
	 * \param mu : \f$\mu\f$ (for spinless fermions, couples to n=1/2-Sz)
	 * \param Kz : \f$K^{z}_i\f$
	 * \param Kx : \f$K^{x}_i\f$
	 * \param Dy : \f$D^{y}\f$
	 */
	OperatorType HeisenbergHamiltonian (const ArrayXXd &Jxy, const ArrayXXd &Jz, 
	                                    const ArrayXd &Bz, const ArrayXd &Bx, const ArrayXd &mu, const ArrayXd &Kz, const ArrayXd &Kx, 
	                                    const ArrayXXd &Dy) const;

	/**
	 * Creates the full Heisenberg (XYZ) Hamiltonian on the supersite.
	 * \param J : \f$J^{\alpha}\f$, \f$\alpha \in \{x,y,z\} \f$
	 * \param B : \f$B^{\alpha}_i\f$, \f$\alpha \in \{x,y,z\} \f$
	 * \param K : \f$K^{\alpha}_i\f$, \f$\alpha \in \{x,y,z\} \f$
	 * \param D : \f$D^{\alpha}\f$, \f$\alpha \in \{x,y,z\} \f$
	 */
	SiteOperator<Symmetry,complex<double> > HeisenbergHamiltonian (const std::array<ArrayXXd,3> &J, 
	                                                               const std::array<ArrayXd,3> &B, 
	                                                               const std::array<ArrayXd,3> &K, 
	                                                               const std::array<ArrayXXd,3> &D) const;
	
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
	
	return OperatorType(Mout,getQ(Sa), label(Sa));
}

template<typename Symmetry>
SiteOperator<Symmetry,complex<double> > SpinBase<Symmetry>::
Rcomp (SPINOP_LABEL Sa, int orbital) const
{
	assert(orbital<N_orbitals);
	
	size_t R = ScompSingleSite(Sa).rows();
	size_t Nl = pow(R,orbital);
	size_t Nr = pow(R,N_orbitals-orbital-1);
	assert(Nl==1 and Nr==1); // otherwise error inkroneckerProduct...
	
	SparseMatrixXd Il = MatrixXd::Identity(Nl,Nl).sparseView();
	SparseMatrixXd Ir = MatrixXd::Identity(Nr,Nr).sparseView();
	
	MatrixXcd Mtmp;
	if (Sa==iSY)
	{
		Mtmp = M_PI*2./(double(D)-1.)*MatrixXcd(ScompSingleSite(Sa));
	}
	else
	{
		Mtmp = 1.i*M_PI*2./(double(D)-1.)*MatrixXcd(ScompSingleSite(Sa));
	}
	
	auto Op = Mtmp.exp().sparseView(1.,1e-14); // ref.value, epsilon // 
	
//	cout << "Rcomp=" << Mtmp << endl << endl;
//	cout << "Re=" << Mtmp.exp().real() << endl << endl;
//	cout << "Im=" << Mtmp.exp().imag() << endl << endl;
//	cout << "Op=" << Op << endl << endl;
	
//	cout << Il << endl << endl;
//	cout << Ir << endl << endl;
//	SparseMatrixXcd Mout = kroneckerProduct(Il,kroneckerProduct(Op,Ir));
//	cout << Mout << endl << endl;
//	cout << "Mout: " << Mout.rows() << "x" << Mout.cols() << endl;
	
	return SiteOperator<Symmetry,complex<double>>(Op,getQ(Sa));
}

template<typename Symmetry>
SiteOperator<Symmetry,double> SpinBase<Symmetry>::
n (int orbital) const
{
	assert(orbital<N_orbitals);
	
	size_t R = ScompSingleSite(SZ).rows();
	size_t Nl = pow(R,orbital);
	size_t Nr = pow(R,N_orbitals-orbital-1);
	
	SparseMatrixXd Il = MatrixXd::Identity(Nl,Nl).sparseView();
	SparseMatrixXd Ir = MatrixXd::Identity(Nr,Nr).sparseView();
	SparseMatrixXd I  = MatrixXd::Identity(R,R).sparseView();
	SparseMatrixXd Mout = kroneckerProduct(Il,kroneckerProduct(0.5*I-ScompSingleSite(SZ),Ir));
	
	return OperatorType(Mout,getQ(SZ), "n");
}

template<typename Symmetry>
SiteOperator<Symmetry,double> SpinBase<Symmetry>::
sign (int orb1, int orb2) const
{
	SparseMatrixXd Id = MatrixXd::Identity(N_states,N_states).sparseView();
	SparseMatrixXd Mout = Id;
	
	for (int i=orb1; i<N_orbitals; ++i)
	{
		Mout = Mout * (2.*Scomp(SZ,i).data);
	}
	for (int i=0; i<orb2; ++i)
	{
		Mout = Mout * (2.*Scomp(SZ,i).data);
	}
	
	return OperatorType(Mout,Symmetry::qvacuum(), "sign");
}

template<typename Symmetry>
SiteOperator<Symmetry,double> SpinBase<Symmetry>::
Id() const
{
	SparseMatrixXd mat = MatrixXd::Identity(N_states,N_states).sparseView();
	OperatorType Oout(mat,Symmetry::qvacuum(), "id");
	return Oout;
}

template<typename Symmetry>
SiteOperator<Symmetry,double> SpinBase<Symmetry>::
beadz() const
{
	MatrixXd Szdiag = Scomp(SZ,0).data;
	MatrixXd tmp(D,D); tmp.setZero();
	
	for (int i=0; i<D; ++i)
	{
		// not sure if true for S>1
		tmp(i,i) = exp(1.i*M_PI*2./(double(D)-1.) * Szdiag(i,i)).real();
	}
	SparseMatrixXd mat = tmp.sparseView();
	OperatorType Oout(mat,Symmetry::qvacuum());
	return Oout;
}

template<typename Symmetry>
SiteOperator<Symmetry,double> SpinBase<Symmetry>::
HeisenbergHamiltonian (const ArrayXXd &Jxy, const ArrayXXd &Jz, 
                       const ArrayXd &Bz, const ArrayXd &Bx, const ArrayXd &mu, 
                       const ArrayXd &Kz, const ArrayXd &Kx,
                       const ArrayXXd &Dy) const
{
	assert(Bz.rows() == N_orbitals and Bx.rows() == N_orbitals and Kz.rows() == N_orbitals and Kx.rows() == N_orbitals);
	
	SparseMatrixXd Mout(N_states,N_states);
	
	for (int i=0; i<N_orbitals; ++i)
	for (int j=0; j<i; ++j)
	{
		if (Jxy(i,j) != 0.)
		{
			Mout += 0.5*Jxy(i,j) * (Scomp(SP,i).data*Scomp(SM,j).data + Scomp(SM,i).data*Scomp(SP,j).data);
		}
		if (Jz(i,j) != 0.)
		{
			Mout += Jz(i,j) * Scomp(SZ,i).data*Scomp(SZ,j).data;
		}
		if (Dy(i,j) != 0.)
		{
			Mout += Dy(i,j) * (Scomp(SX,i).data*Scomp(SZ,j).data - Scomp(SZ,i).data*Scomp(SX,j).data);
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
		if (mu(i) != 0.) {Mout -= mu(i) * (0.5*Id().data-Scomp(SZ,i).data);} // for Kitaev chain: -mu*n = -mu*(1/2-Sz)
	}
	for (int i=0; i<N_orbitals; ++i)
	{
		if (Kz(i)!=0.) {Mout += Kz(i) * Scomp(SZ,i).data * Scomp(SZ,i).data;}
	}
	for (int i=0; i<N_orbitals; ++i)
	{
		if (Kx(i)!=0.) {Mout += Kx(i) * Scomp(SX,i).data * Scomp(SX,i).data;}
	}
	
	OperatorType Oout(Mout,Symmetry::qvacuum(), "Hloc");
	return Oout;
}

template<typename Symmetry>
SiteOperator<Symmetry,complex<double> > SpinBase<Symmetry>::
HeisenbergHamiltonian (const std::array<ArrayXXd,3> &J, 
                       const std::array<ArrayXd,3> &B, 
                       const std::array<ArrayXd,3> &K, 
                       const std::array<ArrayXXd,3> &D) const
{
	SiteOperator<Symmetry,complex<double> > Oout = 
	HeisenbergHamiltonian(ZeroHopping(),J[2],B[2],B[0],ZeroField(),K[2],K[0],D[1]).template cast<complex<double> >();
	
	for (size_t i=0; i<N_orbitals; ++i)
	for (size_t j=0; j<i; ++j)
	{
		if (J[0](i,j) != 0.)
		{
			Oout.data += J[0](i,j) * (Scomp(SX,i).data * Scomp(SX,j).data).template cast<complex<double> >();
		}
		if (J[1](i,j) != 0.)
		{
			Oout.data += -J[1](i,j) * (Scomp(iSY,i).data * Scomp(iSY,j).data).template cast<complex<double> >();
		}
		if (D[0](i,j) != 0.)
		{
			Oout.data += D[0](i,j) * (-1.i) * (Scomp(iSY,i).data * Scomp(SZ,j).data 
			                                  -Scomp(SZ,i).data  * Scomp(iSY,j).data).template cast<complex<double> >();
		}
		if (D[2](i,j) != 0.)
		{
			Oout.data += D[2](i,j) * (-1.i) * (Scomp(SX,i).data * Scomp(iSY,j).data 
			                                  -Scomp(iSY,i).data * Scomp(SX,j).data).template cast<complex<double> >();
		}
	}
	
	// By
	for (int i=0; i<N_orbitals; ++i)
	{
		if (B[2](i) != 0.) {Oout.data -= B[2](i) * (-1.i) * Scomp(iSY,i).data.template cast<complex<double> >();}
	}
	// Ky
	for (int i=0; i<N_orbitals; ++i)
	{
		if (K[1](i) != 0.) {Oout.data -= K[1](i) * (Scomp(iSY,i).data*Scomp(iSY,i).data).template cast<complex<double> >();}
	}
	
	return Oout;
}

template<typename Symmetry>
qarray<Symmetry::Nq> SpinBase<Symmetry>::
qNums (size_t index) const
{
	int M = 0;
	int Ndn = 0;
	
	NestedLoopIterator Nelly(N_orbitals,D);
	Nelly = index;
	for (size_t i=0; i<N_orbitals; i++)
	{
		M += D-(2*(Nelly(i)+1)-1);
		Ndn += Nelly(i);
		// for D=2: Ndn=Nelly(i), Nup=1-Nelly(i)
	}
//	cout << "index=" << index << ", M=" << M << ", Ndn=" << Ndn << ", parity=" << posmod<2>(Ndn) << endl;
	
	if constexpr (Symmetry::IS_TRIVIAL) {return qarray<0>{};}
	else if constexpr (Symmetry::Nq == 1) //return either a dummy quantum number or the magnetic quantum number
	{
		if constexpr (Symmetry::kind()[0] == Sym::KIND::N or Symmetry::kind()[0] == Sym::KIND::T) {return Symmetry::qvacuum();}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::M) {return qarray<1>{M};}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::Z2) {return qarray<1>{posmod<2>(Ndn)};} // number of Ndn spins is the number of fermions
		else {assert(false and "Ill-defined KIND of the used Symmetry.");}
	}
	else if constexpr(Symmetry::Nq==2) //return a dummy quantum number for a second symmetry. Either at first place or at second.
	{
		if constexpr (Symmetry::kind()[0] == Sym::KIND::N or Symmetry::kind()[0] == Sym::KIND::T) {return qarray<2>{{Symmetry::qvacuum()[0],M}};}
		else if constexpr (Symmetry::kind()[1] == Sym::KIND::N or Symmetry::kind()[1] == Sym::KIND::T) {return qarray<2>{{M,Symmetry::qvacuum()[1]}};}
		else {assert(false and "Ill-defined KIND of the used Symmetry.");}
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
Qbasis<Symmetry> SpinBase<Symmetry>::
get_structured_basis() const
{
	Qbasis<Symmetry> out;
	out.push_back(Symmetry::qvacuum(),this->dim());
	return out;
}

template<typename Symmetry>
std::string SpinBase<Symmetry>::
label (SPINOP_LABEL Sa) const
{
	std::string out="";
	if      (Sa==SX)  {out = "Sx";}
	else if (Sa==SY)  {out = "Sy";}
	else if (Sa==iSY) {out = "iSy";}
	else if (Sa==SZ)  {out = "Sz";}
	else if (Sa==SP)  {out = "S+";}
	else if (Sa==SM)  {out = "S-";}
	return out;
}

template<typename Symmetry>
typename Symmetry::qType SpinBase<Symmetry>::
getQ (SPINOP_LABEL Sa) const
{
	typename Symmetry::qType out;
	
	if constexpr(Symmetry::IS_TRIVIAL) {return {};}
	else if constexpr (Symmetry::Nq == 1) //return either a dummy quantum number, the magnetic quantum number or the parity
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
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::Z2)
		{
			if      (Sa==SX)  {out = {1};}
			else if (Sa==SY)  {out = {1};}
			else if (Sa==iSY) {out = {1};}
			else if (Sa==SZ)  {out = {0};}
			else if (Sa==SP)  {out = {1};}
			else if (Sa==SM)  {out = {1};}
			return out;
		}
		else {assert(false and "Ill defined KIND of the used Symmetry.");}
	}
	else if constexpr(Symmetry::Nq == 2) //return a dummy quantum number for a second symmetry. Either at first place or at second.
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
	SparseMatrixXd Mout(D,D);
	if (Sa==SX)
	{
		Mout = Sbase()+SparseMatrixXd(Sbase().transpose());
	}
	else if (Sa==iSY)
	{
		Mout = Sbase()-SparseMatrixXd(Sbase().transpose());
	}
	else if (Sa==SZ) 
	{
		assert(D >= 1);
		// SparseMatrixXd Mout(D,D);
		double S = 0.5*(D-1);
		for (size_t i=0; i<D; ++i)
		{
			double M = S-i;
			Mout.insert(i,i) = M;
		}
	}
	else if (Sa==SP) 
	{
		Mout = 2.*Sbase();
	}
	else if (Sa==SM) 
	{
		Mout = SparseMatrixXd(2.*Sbase().transpose());
	}
	return Mout;
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
