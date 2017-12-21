#ifndef FERMIONBASE
#define FERMIONBASE

#include <algorithm>
#include <iterator>
#include <boost/dynamic_bitset.hpp>

#include <Eigen/Core>

#include "bases/SpinBase.h"

/**This basically just constructs the full Hubbard model on \p L_input lattice sites.*/
template<typename Symmetry>
class FermionBase
{
	typedef SiteOperator<Symmetry,double> OperatorType;
	
public:
	
	FermionBase(){};
	
	/**
	\param L_input : the amount of orbitals
	\param U_IS_INFINITE : if \p true, eliminates doubly-occupied sites from the basis*/
	FermionBase (size_t L_input, bool U_IS_INFINITE=false, bool NM_input=false);
	
	/**number of states = \f$4^L\f$ or \f$3^L\f$ for $U=\infty$*/
	inline size_t dim() const {return N_states;}
	
	/**number of orbitals*/
	inline size_t orbitals() const {return N_orbitals;}
	
	///\{
	/**Annihilation operator, for \p N_orbitals=1, this is
	\f$c_{\uparrow} = \left(
	\begin{array}{cccc}
	0 & 1 & 0 & 0\\
	0 & 0 & 0 & 0\\
	0 & 0 & 0 & 1\\
	0 & 0 & 0 & 0\\
	\end{array}
	\right)\f$
	or
	\f$c_{\downarrow} = \left(
	\begin{array}{cccc}
	0 & 0 & 1 & 0\\
	0 & 0 & 0 & -1\\
	0 & 0 & 0 & 0\\
	0 & 0 & 0 & 0\\
	\end{array}
	\right)\f$
	\param sigma : spin index
	\param orbital : orbital index*/
	OperatorType c (SPIN_INDEX sigma, int orbital=0) const;
	
	/**Creation operator.
	\param sigma : spin index
	\param orbital : orbital index*/
	OperatorType cdag (SPIN_INDEX sigma, int orbital=0) const;
	
	/**Occupation number operator
	\param sigma : spin index
	\param orbital : orbital index*/
	OperatorType n (SPIN_INDEX sigma, int orbital=0) const;
	
	/**Total occupation number operator, for \p N_orbitals=1, this is
	\f$n = n_{\uparrow}+n_{\downarrow} = \left(
	\begin{array}{cccc}
	0 & 0 & 0 & 0\\
	0 & 1 & 0 & 0\\
	0 & 0 & 1 & 0\\
	0 & 0 & 0 & 2\\
	\end{array}
	\right)\f$
	\param orbital : orbital index*/
	OperatorType n (int orbital=0) const;
	
	/**Double occupation, for \p N_orbitals=1, this is
	\f$d = n_{\uparrow}n_{\downarrow} = \left(
	\begin{array}{cccc}
	0 & 0 & 0 & 0\\
	0 & 0 & 0 & 0\\
	0 & 0 & 0 & 0\\
	0 & 0 & 0 & 1\\
	\end{array}
	\right)\f$
	*/
	OperatorType d (int orbital=0) const;
	///\}
	
	///\{
	/**
	\param Sa
	\param orbital
	*/
	OperatorType Scomp (SPINOP_LABEL Sa, int orbital=0) const;
	
	/**For \p N_orbitals=1, this is
	\f$s^z = \left(
	\begin{array}{cccc}
	0 & 0 & 0 & 0\\
	0 & 0.5 & 0 & 0\\
	0 & 0 & -0.5 & 0\\
	0 & 0 & 0 & 0\\
	\end{array}
	\right)\f$
	*/
	OperatorType Sz (int orbital=0) const;
	
	/**For \p N_orbitals=1, this is
	\f$s^+ = \left(
	\begin{array}{cccc}
	0 & 0 & 0 & 0\\
	0 & 0 & 1 & 0\\
	0 & 0 & 0 & 0\\
	0 & 0 & 0 & 0\\
	\end{array}
	\right)\f$
	*/
	OperatorType Sp (int orbital=0) const;
	
	/**For \p N_orbitals=1, this is
	\f$s^- = \left(
	\begin{array}{cccc}
	0 & 0 & 0 & 0\\
	0 & 0 & 0 & 0\\
	0 & 1 & 0 & 0\\
	0 & 0 & 0 & 0\\
	\end{array}
	\right)\f$
	*/
	OperatorType Sm (int orbital=0) const;
	
	/**For \p N_orbitals=1, this is
	\f$s^x = \left(
	\begin{array}{cccc}
	0 & 0 & 0 & 0\\
	0 & 0 & 0.5 & 0\\
	0 & 0.5 & 0 & 0\\
	0 & 0 & 0 & 0\\
	\end{array}
	\right)\f$
	*/
	OperatorType Sx (int orbital=0) const;
	
	/**For \p N_orbitals=1, this is
	\f$is^y = \left(
	\begin{array}{cccc}
	0 & 0 & 0 & 0\\
	0 & 0 & 0.5 & 0\\
	0 & -0.5 & 0 & 0\\
	0 & 0 & 0 & 0\\
	\end{array}
	\right)\f$
	*/
	OperatorType iSy (int orbital=0) const;
	///\}
	
	///\{
	/**Fermionic sign for the hopping between two orbitals of nearest-neighbour supersites of a ladder. For \p N_orbitals=1, this is
	\f$(1-2n_{\uparrow})*(1-2n_{\downarrow}) = \left(
	\begin{array}{cccc}
	1 & 0  & 0  & 0\\
	0 & -1 & 0  & 0\\
	0 & 0  & -1 & 0\\
	0 & 0  & 0  & 1\\
	\end{array}
	\right)\f$
	\param orb1 : orbital on supersite i
	\param orb2 : orbital on supersite i+1
	\note \f$f\f$ anticommutes with \f$c\f$, \f$c^{\dagger}\f$ on the same site and commutes with \f$n\f$, \f$S^+\f$ and \f$S^-\f$
	*/
	OperatorType sign (int orb1=0, int orb2=0) const;
	
	OperatorType Id() const;
	
	ArrayXd ZeroField() const;
	
	/**Fermionic sign for the transfer to a particular orbital, needed by HubbardModel::c and HubbardModel::cdag.
	\param orbital : orbital on the supersite*/
	OperatorType sign_local (int orbital) const;
	///\}
	
	/**Creates the full Hubbard Hamiltonian on the supersite.
	\param U : \f$U\f$
	\param t : \f$t\f$
	\param V : \f$V\f$
	\param J : \f$J\f$
	\param PERIODIC: periodic boundary conditions if \p true
	\param Bz : \f$B_z\f$*/
	template<typename Scalar> SiteOperator<Symmetry,Scalar>
	HubbardHamiltonian (double U, Scalar t=1., double V=0., double J=0., double Bz=0., bool PERIODIC=false) const;
	
	/**Creates the full Hubbard Hamiltonian on the supersite with orbital-dependent U.
	\param Uvec : \f$U\f$ for each orbital
	\param mu : \f$E_i\f$ for each orbital
	\param Bzloc : \f$B_z\f$ for each orbital
	\param t : \f$t\f$
	\param V : \f$V\f$
	\param J : \f$J\f$
	\param PERIODIC: periodic boundary conditions if \p true*/
	template<typename Scalar> SiteOperator<Symmetry,Scalar>
	HubbardHamiltonian (ArrayXd Uorb, ArrayXd Eorb, ArrayXd Bzorb, ArrayXd Bxorb, Scalar t=1., double V=0., double J=0., bool PERIODIC=false) const;
	
	vector<qarray<Symmetry::Nq> > get_basis() const;
	
	typename Symmetry::qType getQ (SPIN_INDEX sigma, int Delta=0) const;
	typename Symmetry::qType getQ (SPINOP_LABEL Sa) const;
	
private:
	
	size_t N_orbitals;
	size_t N_states;
	bool NM;
	
	/**Returns the qarray for a given index of the basis
	\param index
	\param NM : If \p true, the format is (N,M), if \p false the format is (Nup,Ndn)*/ 
	qarray<Symmetry::Nq> qNums (size_t index) const;
	
	vector<boost::dynamic_bitset<unsigned char> > basis;
	
	double parity (const boost::dynamic_bitset<unsigned char> &state, int orbital) const;
};

template<typename Symmetry>
FermionBase<Symmetry>::
FermionBase (size_t L_input, bool U_IS_INFINITE, bool NM_input)
	:N_orbitals(L_input),NM(NM_input)
{
	assert(N_orbitals>=1);
	
	size_t locdim = (U_IS_INFINITE)? 3 : 4;
	N_states = pow(locdim,N_orbitals);
	basis.resize(N_states);
	
	vector<int> nUP(4);
	vector<int> nDN(4);
	nUP[0] = 0; nDN[0] = 0;
	nUP[1] = 1; nDN[1] = 0;
	nUP[2] = 0; nDN[2] = 1;
	nUP[3] = 1; nDN[3] = 1;
	
	NestedLoopIterator Nelly(N_orbitals,locdim);
	for (Nelly=Nelly.begin(); Nelly!=Nelly.end(); ++Nelly)
	{
		basis[Nelly.index()].resize(2*N_orbitals);
		for (int i=0; i<N_orbitals; ++i)
		{
			basis[*Nelly][2*i]   = nUP[Nelly(i)];
			basis[*Nelly][2*i+1] = nDN[Nelly(i)];
		}
	}
	
//	 cout << "basis:" << endl;
//	 for (size_t i=0; i<N_states; i++) {cout << basis[i] << endl;}
//	 cout << endl;
}

template<typename Symmetry>
SiteOperator<Symmetry,double> FermionBase<Symmetry>::
c (SPIN_INDEX sigma, int orbital) const
{
	SparseMatrixXd Mout(N_states,N_states);
	int orbital_in_base = 2*orbital+static_cast<int>(sigma);
	
	for (int j=0; j<basis.size(); ++j)
	{
		if (basis[j][orbital_in_base])
		{
			boost::dynamic_bitset<unsigned char> b = basis[j];
			b[orbital_in_base].flip();
			
			auto it = find(basis.begin(), basis.end(), b);
			
			if (it!=basis.end())
			{
				int i = distance(basis.begin(), it);
				Mout.insert(i,j) = parity(b, orbital_in_base);
			}
		}
	}
	
//	cout << "c " << sigma << "=" << endl << Mout << endl;
	
	return OperatorType(Mout, getQ(sigma,-1));
}

template<typename Symmetry>
SiteOperator<Symmetry,double> FermionBase<Symmetry>::
cdag (SPIN_INDEX sigma, int orbital) const
{
//	cout << "cdag " << sigma << "=" << endl << c(sigma,orbital).data.transpose() << endl;
	return OperatorType(c(sigma,orbital).data.transpose(), getQ(sigma,+1));
}

template<typename Symmetry>
SiteOperator<Symmetry,double> FermionBase<Symmetry>::
n (SPIN_INDEX sigma, int orbital) const
{
	SparseMatrixXd Mout(N_states,N_states);
	for (int j=0; j<basis.size(); ++j)
	{
		if (sigma == UP or sigma == DN)
		{
			Mout.insert(j,j) = 1.*basis[j][2*orbital+static_cast<int>(sigma)];
		}
		else if (sigma == UPDN)
		{
			Mout.insert(j,j) = 1.*(basis[j][2*orbital] + basis[j][2*orbital+1]);
		}
	}
	
	OperatorType Oout(Mout,Symmetry::qvacuum());
	return Oout;
}

template<typename Symmetry>
SiteOperator<Symmetry,double> FermionBase<Symmetry>::
n (int orbital) const
{
	return n(UPDN,orbital);
}

template<typename Symmetry>
SiteOperator<Symmetry,double> FermionBase<Symmetry>::
d (int orbital) const
{
	return OperatorType(n(UP,orbital).data*n(DN,orbital).data, Symmetry::qvacuum());
}

template<typename Symmetry>
SiteOperator<Symmetry,double> FermionBase<Symmetry>::
Scomp (SPINOP_LABEL Sa, int orbital) const
{
	assert(Sa != SY);
	if      (Sa==SX)  {return Sx(orbital);}
	else if (Sa==iSY) {return iSy(orbital);}
	else if (Sa==SZ)  {return Sz(orbital);}
	else if (Sa==SP)  {return Sp(orbital);}
	else if (Sa==SM)  {return Sm(orbital);}
}

template<typename Symmetry>
SiteOperator<Symmetry,double> FermionBase<Symmetry>::
Sz (int orbital) const
{
	return OperatorType(0.5*(n(UP,orbital).data-n(DN,orbital).data), getQ(SZ));
}

template<typename Symmetry>
SiteOperator<Symmetry,double> FermionBase<Symmetry>::
Sp (int orbital) const
{
	return OperatorType(cdag(UP,orbital).data*c(DN,orbital).data, getQ(SP));
}

template<typename Symmetry>
SiteOperator<Symmetry,double> FermionBase<Symmetry>::
Sm (int orbital) const
{
	return OperatorType(cdag(DN,orbital).data*c(UP,orbital).data, getQ(SM));
}

template<typename Symmetry>
SiteOperator<Symmetry,double> FermionBase<Symmetry>::
Sx (int orbital) const
{
	return OperatorType(0.5*(Sp(orbital).data+Sm(orbital).data), getQ(SX));
}

template<typename Symmetry>
SiteOperator<Symmetry,double> FermionBase<Symmetry>::
iSy (int orbital) const
{
	return OperatorType(0.5*(Sp(orbital).data-Sm(orbital).data), getQ(iSY));
}

template<typename Symmetry>
SiteOperator<Symmetry,double> FermionBase<Symmetry>::
sign (int orb1, int orb2) const
{
	SparseMatrixXd Id = MatrixXd::Identity(N_states,N_states).sparseView();
	SparseMatrixXd Mout = Id;
	
	for (int i=orb1; i<N_orbitals; ++i)
	{
		Mout = Mout * (Id-2.*n(UP,i).data)*(Id-2.*n(DN,i).data);
	}
	for (int i=0; i<orb2; ++i)
	{
		Mout = Mout * (Id-2.*n(UP,i).data)*(Id-2.*n(DN,i).data);
	}
	
	return OperatorType(Mout,Symmetry::qvacuum());
}

template<typename Symmetry>
SiteOperator<Symmetry,double> FermionBase<Symmetry>::
Id() const
{
	SparseMatrixXd mat = MatrixXd::Identity(N_states,N_states).sparseView();
	OperatorType Oout(mat,Symmetry::qvacuum());
	return Oout;
}

template<typename Symmetry>
ArrayXd FermionBase<Symmetry>::
ZeroField() const
{
	return ArrayXd::Zero(N_orbitals);
}

template<typename Symmetry>
SiteOperator<Symmetry,double> FermionBase<Symmetry>::
sign_local (int orbital) const
{
	SparseMatrixXd Id = MatrixXd::Identity(N_states,N_states).sparseView();
	SparseMatrixXd Mout = Id;
	
	for (int i=0; i<orbital; ++i)
	{
		Mout = Mout * (Id-2.*n(UP,i).data)*(Id-2.*n(DN,i).data);
	}
	
	return OperatorType(Mout,Symmetry::qvacuum());
}

template<typename Symmetry>
template<typename Scalar>
SiteOperator<Symmetry,Scalar> FermionBase<Symmetry>::
HubbardHamiltonian (double U, Scalar t, double V, double J, double Bz, bool PERIODIC) const
{
	SparseMatrix<Scalar> Mout(N_states,N_states);
	
	size_t ilast = (PERIODIC == true and N_orbitals>2)? N_orbitals:N_orbitals-1;
	
	for (int i=0; i<ilast; ++i) // for all bonds
	{
		if (t != 0.)
		{
			SparseMatrix<Scalar> T = -t*(cdag(UP,i).data * c(UP,(i+1)%N_orbitals).data + 
			                             cdag(DN,i).data * c(DN,(i+1)%N_orbitals).data).template cast<Scalar>();
			Mout += -(T+SparseMatrix<Scalar>(T.adjoint()));
		}
		if (V != 0.)
		{
			Mout += V*(n(i).data*n((i+1)%N_orbitals).data).template cast<Scalar>();
		}
		if (J != 0.)
		{
			Mout += J*(0.5*Sp(i).data*Sm((i+1)%N_orbitals).data + 
			           0.5*Sm(i).data*Sp((i+1)%N_orbitals).data + 
			               Sz(i).data*Sz((i+1)%N_orbitals).data).template cast<Scalar>();
		}
	}
	
	if (U != 0. and U != numeric_limits<double>::infinity())
	{
		for (int i=0; i<N_orbitals; ++i) {Mout += U*d(i).data.template cast<Scalar>();}
	}
	if (Bz != 0.)
	{
		for (int i=0; i<N_orbitals; ++i) {Mout -= Bz*Sz(i).data.template cast<Scalar>();}
	}
	
	return SiteOperator<Symmetry,Scalar>(Mout,Symmetry::qvacuum());
}

template<typename Symmetry>
template<typename Scalar>
SiteOperator<Symmetry,Scalar> FermionBase<Symmetry>::
HubbardHamiltonian (ArrayXd Uorb, ArrayXd Eorb, ArrayXd Bzorb, ArrayXd Bxorb, Scalar t, double V, double J, bool PERIODIC) const
{
	SparseMatrix<Scalar> Mout = HubbardHamiltonian(0.,t,V,J,0.,PERIODIC).data;
	
	for (int i=0; i<N_orbitals; ++i)
	{
		if (Uorb.rows() > 0)
		{
			if (Uorb(i) != 0. and Uorb(i) != numeric_limits<double>::infinity())
			{
				Mout += Uorb(i) * d(i).data.template cast<Scalar>();
			}
		}
		if (Eorb.rows() > 0)
		{
			if (Eorb(i) != 0.)
			{
				Mout += Eorb(i) * n(i).data.template cast<Scalar>();
			}
		}
		if (Bzorb.rows() > 0)
		{
			if (Bzorb(i) != 0.)
			{
				Mout += Bzorb(i) * Sz(i).data.template cast<Scalar>();
			}
		}
		if (Bxorb.rows() > 0)
		{
			if (Bxorb(i) != 0.)
			{
				Mout += Bxorb(i) * Sx(i).data.template cast<Scalar>();
			}
		}
	}
	
	return SiteOperator<Symmetry,Scalar>(Mout,Symmetry::qvacuum());
}

template<typename Symmetry>
double FermionBase<Symmetry>::
parity (const boost::dynamic_bitset<unsigned char> &b, int i) const
{
	double out = 1.;
	for (int j=0; j<i; ++j)
	{
		if (b[j]) {out *= -1.;} // switch sign for every particle found between 0 & i
	}
	return out;
}

template<typename Symmetry>
qarray<Symmetry::Nq> FermionBase<Symmetry>::
qNums (size_t index) const
{
	int M=0; int N=0;
	int Nup=0; int Ndn=0;
	
	for (size_t i=0; i<2*N_orbitals; i++)
	{
		if (basis[index][i])
		{
			N+=1;
			if (i%2 == 0) {M+=1; Nup+=1;}
			else          {M-=1; Ndn+=1;}
		}
	}
	
	if constexpr(Symmetry::IS_TRIVIAL) { return qarray<0>{}; }
	if constexpr(Symmetry::Nq == 1) { return qarray<1>{N}; }
	else
	{
		if (NM) {return qarray<Symmetry::Nq>{N,M};}
		else    {return qarray<Symmetry::Nq>{Nup,Ndn};}
	}
}

template<typename Symmetry>
vector<qarray<Symmetry::Nq> > FermionBase<Symmetry>::
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
typename Symmetry::qType FermionBase<Symmetry>::
getQ (SPIN_INDEX sigma, int Delta) const
{
	if constexpr(Symmetry::IS_TRIVIAL) {return {};}
	if constexpr(Symmetry::Nq == 1) //return particle number as good quantum number.
				{
					typename Symmetry::qType out;
					if      (sigma==UP)     {out = {Delta};}
					else if (sigma==DN)     {out = {Delta};}
					else if (sigma==UPDN)   {out = {2*Delta};}
					else if (sigma==NOSPIN) {out = Symmetry::qvacuum();}
					return out;
				}
	if constexpr(Symmetry::Nq == 2)
				{
					typename Symmetry::qType out;
					if (NM)
					{
						if      (sigma==UP)     {out = {Delta,Delta};}
						else if (sigma==DN)     {out = {Delta,-Delta};}
						else if (sigma==UPDN)   {out = {2*Delta,Delta};}
						else if (sigma==NOSPIN) {out = Symmetry::qvacuum();}
					}
					else
					{
						if      (sigma==UP)     {out = {Delta,0};}
						else if (sigma==DN)     {out = {0,Delta};}
						else if (sigma==UPDN)   {out = {Delta,Delta};}
						else if (sigma==NOSPIN) {out = Symmetry::qvacuum();}
					}
					return out;
				}
	static_assert("You inserted a Symmetry which can not be handled by FermionBase.");
}

template<typename Symmetry>
typename Symmetry::qType FermionBase<Symmetry>::
getQ (SPINOP_LABEL Sa) const
{
	assert(Sa != SX and Sa != iSY);
	if constexpr(Symmetry::IS_TRIVIAL) {return {};}
	if constexpr(Symmetry::Nq == 1) { return {{0}}; } //return particle number as good quantum number.
	
	typename Symmetry::qType out;
	
	if (Sa==SZ) {out = {0,0};}
	
	if (NM)
	{
		if (Sa==SP) {out = {0,+2};}
		if (Sa==SM) {out = {0,-2};}
	}
	else
	{
		if (Sa==SP) {out = {+1,-1};}
		if (Sa==SM) {out = {-1,+1};}
	}
	return out;
}

#endif
