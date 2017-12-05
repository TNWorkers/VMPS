#ifndef FERMIONBASE
#define FERMIONBASE

#include <algorithm>
#include <iterator>
#include <boost/dynamic_bitset.hpp>

#include "SpinBase.h"

#include <Eigen/Core>

/**This basically just constructs the full Hubbard model on \p L_input lattice sites.*/
class FermionBase
{
public:
	
	FermionBase(){};
	
	/**
	\param L_input : the amount of orbitals
	\param U_IS_INFINITE : if \p true, eliminates doubly-occupied sites from the basis*/
	FermionBase (size_t L_input, bool U_IS_INFINITE=false);
	
	/**amount of states = \f$4^L\f$*/
	inline size_t dim() const {return N_states;}
	
	/**amount of orbitals*/
	inline size_t orbitals() const  {return N_orbitals;}
	
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
	SparseMatrixXd c (SPIN_INDEX sigma, int orbital=0) const;
	
	/**Creation operator.
	\param sigma : spin index
	\param orbital : orbital index*/
	SparseMatrixXd cdag (SPIN_INDEX sigma, int orbital=0) const;
	
	/**Occupation number operator
	\param sigma : spin index
	\param orbital : orbital index*/
	SparseMatrixXd n (SPIN_INDEX sigma, int orbital=0) const;
	
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
	SparseMatrixXd n (int orbital=0) const;
	
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
	SparseMatrixXd d (int orbital=0) const;
	///\}
	
	///\{
	/**
	\param Sa
	\param orbital
	*/
	SparseMatrixXd Scomp (SPINOP_LABEL Sa, int orbital=0) const;
	
	/**
	The quantum number change which corresponds to the given spin operator.
	\param Sa
	\param NM : If \p true, the format is (N,M), if \p false the format is (Nup,Ndn)
	*/
	qarray<2> Deltaq (SPINOP_LABEL Sa, bool NM=false) const;
	
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
	SparseMatrixXd Sz (int orbital=0) const;
	
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
	SparseMatrixXd Sp (int orbital=0) const;
	
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
	SparseMatrixXd Sm (int orbital=0) const;
	
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
	SparseMatrixXd Sx (int orbital=0) const;
	
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
	SparseMatrixXd iSy (int orbital=0) const;
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
	*/
	SparseMatrixXd sign (int orb1=0, int orb2=0) const;
	
	/**Fermionic sign for the transfer to a particular orbital, needed by HubbardModel::c and HubbardModel::cdag.
	\param orbital : orbital on the supersite*/
	SparseMatrixXd sign_local (int orbital) const;
	///\}
	
	/**Creates the full Hubbard Hamiltonian on the supersite.
	\param U : \f$U\f$
	\param t : \f$t\f$
	\param V : \f$V\f$
	\param J : \f$J\f$
	\param PERIODIC: periodic boundary conditions if \p true
	\param Bz : \f$B_z\f$*/
	template<typename Scalar> SparseMatrix<Scalar> HubbardHamiltonian (double U, Scalar t=1., double V=0., double J=0., double Bz=0., bool PERIODIC=false) const;
	
	/**Creates the full Hubbard Hamiltonian on the supersite with orbital-dependent U.
	\param Uvec : \f$U\f$ for each orbital
	\param onsite : \f$E_i\f$ for each orbital
	\param Bzloc : \f$B_z\f$ for each orbital
	\param t : \f$t\f$
	\param V : \f$V\f$
	\param J : \f$J\f$
	\param PERIODIC: periodic boundary conditions if \p true*/
	template<typename Scalar> SparseMatrix<Scalar> HubbardHamiltonian (ArrayXd Uvec, ArrayXd onsite, ArrayXd Bzloc, 
	                                                                   Scalar t=1., double V=0., double J=0., bool PERIODIC=false) const;
	
	/**Returns the qarray for a given index of the basis
	\param index
	\param NM : If \p true, the format is (N,M), if \p false the format is (Nup,Ndn)*/ 
	qarray<2> qNums (size_t index, bool NM=true);
	
	vector<boost::dynamic_bitset<unsigned char> > basis;
	
private:
	
	size_t N_orbitals;
	size_t N_states;
	
	double parity (const boost::dynamic_bitset<unsigned char> &state, int orbital) const;
};

FermionBase::
FermionBase (size_t L_input, bool U_IS_INFINITE)
:N_orbitals(L_input)
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

SparseMatrixXd FermionBase::
c (SPIN_INDEX sigma, int orbital) const
{
	SparseMatrixXd Mout(N_states,N_states);
	int orbital_in_base = 2*orbital+static_cast<int>(sigma);
	
	for (int j=0; j<basis.size(); ++j)
	{
		if (basis[j][orbital_in_base]) // factor 2 because of ordering 1UP,1DN,2UP,2DN,...
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
	
	return Mout;
}

inline SparseMatrixXd FermionBase::
cdag (SPIN_INDEX sigma, int orbital) const
{
	return c(sigma,orbital).transpose();
}

inline SparseMatrixXd FermionBase::
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
	return Mout;
}

inline SparseMatrixXd FermionBase::
n (int orbital) const
{
//	SparseMatrixXd Mout(N_states,N_states);
//	for (int j=0; j<basis.size(); ++j)
//	{
//		Mout.insert(j,j) = 1.*(basis[j][2*orbital] + basis[j][2*orbital+1]);
//	}
//	return Mout;
	return n(UPDN,orbital);
}

inline SparseMatrixXd FermionBase::
d (int orbital) const
{
	SparseMatrixXd Mtmp = n(UP,orbital)*n(DN,orbital);
	return n(UP,orbital)*n(DN,orbital);
}

SparseMatrixXd FermionBase::
Scomp (SPINOP_LABEL Sa, int orbital) const
{
	assert(Sa != SY);
	if      (Sa==SX)  {return Sx(orbital);}
	else if (Sa==iSY) {return iSy(orbital);}
	else if (Sa==SZ)  {return Sz(orbital);}
	else if (Sa==SP)  {return Sp(orbital);}
	else if (Sa==SM)  {return Sm(orbital);}
}

qarray<2> FermionBase::
Deltaq (SPINOP_LABEL Sa, bool NM) const
{
	assert(Sa != SX and Sa != iSY);
	
	if (Sa==SZ) {return qarray<2>({0,0});}
	
	if (NM)
	{
		if (Sa==SP) {return qarray<2>({0,+2});}
		if (Sa==SM) {return qarray<2>({0,-2});}
	}
	else
	{
		if (Sa==SP) {return qarray<2>({+1,-1});}
		if (Sa==SM) {return qarray<2>({-1,+1});}
	}
}

inline SparseMatrixXd FermionBase::
Sz (int orbital) const
{
	return 0.5*(n(UP,orbital)-n(DN,orbital));
}

inline SparseMatrixXd FermionBase::
Sp (int orbital) const
{
	return cdag(UP,orbital)*c(DN,orbital);
}

inline SparseMatrixXd FermionBase::
Sm (int orbital) const
{
	return cdag(DN,orbital)*c(UP,orbital);
}

inline SparseMatrixXd FermionBase::
Sx (int orbital) const
{
	return 0.5*(Sp(orbital)+Sm(orbital));
}

inline SparseMatrixXd FermionBase::
iSy (int orbital) const
{
	return 0.5*(Sp(orbital)-Sm(orbital));
}

inline SparseMatrixXd FermionBase::
sign (int orb1, int orb2) const
{
	SparseMatrixXd Id = MatrixXd::Identity(N_states,N_states).sparseView();
	SparseMatrixXd Mout = Id;
	
	for (int i=orb1; i<N_orbitals; ++i)
	{
		Mout = Mout * (Id-2.*n(UP,i))*(Id-2.*n(DN,i));
	}
	for (int i=0; i<orb2; ++i)
	{
		Mout = Mout * (Id-2.*n(UP,i))*(Id-2.*n(DN,i));
	}
//	if (orb1 != orb2) {cout << Mout << endl;}
	
	return Mout;
}

inline SparseMatrixXd FermionBase::
sign_local (int orbital) const
{
	SparseMatrixXd Id = MatrixXd::Identity(N_states,N_states).sparseView();
	SparseMatrixXd Mout = Id;
	
	for (int i=0; i<orbital; ++i)
	{
		Mout = Mout * (Id-2.*n(UP,i))*(Id-2.*n(DN,i));
	}
	return Mout;
}

template<typename Scalar>
SparseMatrix<Scalar> FermionBase::
HubbardHamiltonian (double U, Scalar t, double V, double J, double Bz, bool PERIODIC) const
{
	SparseMatrix<Scalar> Mout(N_states,N_states);
	
	for (int i=0; i<N_orbitals-1; ++i) // for all bonds
	{
		if (t != 0.)
		{
			SparseMatrix<Scalar> T = -t*(cdag(UP,i)*c(UP,i+1)+cdag(DN,i)*c(DN,i+1)).cast<Scalar>();
			Mout += -(T+SparseMatrix<Scalar>(T.adjoint()));
		}
		if (V != 0.) {Mout += V*(n(i)*n(i+1)).cast<Scalar>();}
		if (J != 0.)
		{
			Mout += J*(0.5*Sp(i)*Sm(i+1) + 0.5*Sm(i)*Sp(i+1) + Sz(i)*Sz(i+1)).cast<Scalar>();
		}
	}
	if (PERIODIC==true and N_orbitals>2)
	{
		if (t != 0.)
		{
			SparseMatrix<Scalar> T = -t*(cdag(UP,0)*c(UP,N_orbitals-1) + cdag(DN,0)*c(DN,N_orbitals-1)).cast<Scalar>();
			Mout += -(T+SparseMatrix<Scalar>(T.adjoint()));
		}
		if (V != 0.) {Mout += V*(n(0)*n(N_orbitals-1)).cast<Scalar>();}
		if (J != 0.)
		{
			Mout += J*(0.5*Sp(0)*Sm(N_orbitals-1) + 0.5*Sm(0)*Sp(N_orbitals-1) + Sz(0)*Sz(N_orbitals-1)).cast<Scalar>();
		}
	}
	if (U != 0. and U != numeric_limits<double>::infinity())
	{
		for (int i=0; i<N_orbitals; ++i) {Mout += U*d(i).cast<Scalar>();}
	}
	if (Bz != 0.)
	{
		for (int i=0; i<N_orbitals; ++i) {Mout -= Bz*Sz(i).cast<Scalar>();}
	}
	
	return Mout;
}

template<typename Scalar>
SparseMatrix<Scalar> FermionBase::
HubbardHamiltonian (ArrayXd Uloc, ArrayXd onsite, ArrayXd Bzloc, Scalar t, double V, double J, bool PERIODIC) const
{
	SparseMatrix<Scalar> Mout = HubbardHamiltonian(0,t,V,J,0,PERIODIC);
	
	for (int i=0; i<N_orbitals; ++i)
	{
		if (Uloc.rows() > 0)
		{
			if (Uloc(i) != 0. and Uloc(i) != numeric_limits<double>::infinity())
			{
				Mout += Uloc(i) * d(i).cast<Scalar>();
			}
		}
		if (onsite.rows() > 0)
		{
			if (onsite(i) != 0.)
			{
				Mout += onsite(i) * n(i).cast<Scalar>();
			}
		}
		if (Bzloc.rows() > 0)
		{
			if (Bzloc(i) != 0.)
			{
				Mout += Bzloc(i) * Sz(i).cast<Scalar>();
			}
		}
	}
	
	return Mout;
}

double FermionBase::
parity (const boost::dynamic_bitset<unsigned char> &b, int i) const
{
	int out = 1.;
	for (int j=0; j<i; ++j)
	{
		if (b[j]) {out *= -1.;} // switch sign for every particle found between 0 & i
	}
	return out;
}

qarray<2> FermionBase::
qNums(size_t index, bool NM)
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

	if (NM) {return qarray<2>{N,M};}
	else    {return qarray<2>{Nup,Ndn};}
}

#endif
