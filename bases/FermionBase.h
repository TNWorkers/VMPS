#ifndef FERMIONBASE
#define FERMIONBASE

/// \cond
#include <algorithm>
#include <iterator>

#include <boost/dynamic_bitset.hpp>
#include <Eigen/Core>
/// \endcond

#include "symmetry/kind_dummies.h"
#include "DmrgTypedefs.h" // for SPIN_INDEX, SPINOP_LABEL
#include "tensors/SiteOperator.h"
#include "DmrgExternal.h" // for posmod

/** 
 * \class FermionBase
 * \ingroup Bases
 *
 * This class provides the local operators for fermions.
 *
 * The class is implemented for all combinations of U1 symmetries in the file FermionBase.h.
 * For the different non abelian symmetries, there are template specialisations in the files FermionBase!Symmetry!.h
 *
 * \describe_Symmetry
 *
 */
template<typename Symmetry>
class FermionBase
{
	typedef SiteOperator<Symmetry,double> OperatorType;
	
public:
	
	FermionBase(){};
	
	/**
	 * \param L_input : the amount of orbitals
	 * \param U_IS_INFINITE : if \p true, eliminates doubly-occupied sites from the basis
	 */
	FermionBase (size_t L_input, bool U_IS_INFINITE=false);
	
	/**number of states = \f$4^L\f$ or \f$3^L\f$ for \f$U=\infty\f$*/
	inline size_t dim() const {return N_states;}
	
	/**number of orbitals*/
	inline size_t orbitals() const {return N_orbitals;}
	
	///\{
	/**
	 * Annihilation operator, for \p N_orbitals=1, this is
	 * \f$c_{\uparrow} = \left(
	 * \begin{array}{cccc}
	 * 0 & 1 & 0 & 0\\
	 * 0 & 0 & 0 & 0\\
	 * 0 & 0 & 0 & 1\\
	 * 0 & 0 & 0 & 0\\
	 * \end{array}
	 * \right)\f$
	 * or
	 * \f$c_{\downarrow} = \left(
	 *\begin{array}{cccc}
	 * 0 & 0 & 1 & 0\\
	 * 0 & 0 & 0 & -1\\
	 * 0 & 0 & 0 & 0\\
	 * 0 & 0 & 0 & 0\\
	 * \end{array}
	 * \right)\f$
	 * \param sigma : spin index
	 * \param orbital : orbital index
	 */
	OperatorType c (SPIN_INDEX sigma, int orbital=0) const;
	
	/**
	 * Creation operator.
	 * \param sigma : spin index
	 * \param orbital : orbital index
	 */
	OperatorType cdag (SPIN_INDEX sigma, int orbital=0) const;
	
	/**
	 * Occupation number operator
	 * \param sigma : spin index
	 * \param orbital : orbital index
	 */
	OperatorType n (SPIN_INDEX sigma, int orbital=0) const;
	
	/**
	 * Total occupation number operator, for \p N_orbitals=1, this is
	 * \f$n = n_{\uparrow}+n_{\downarrow} = \left(
	 * \begin{array}{cccc}
	 * 0 & 0 & 0 & 0\\
	 * 0 & 1 & 0 & 0\\
	 * 0 & 0 & 1 & 0\\
	 * 0 & 0 & 0 & 2\\
	 * \end{array}
	 * \right)\f$
	 * \param orbital : orbital index
	 */
	OperatorType n (int orbital=0) const;
	
	/**
	 * Double occupation, for \p N_orbitals=1, this is
	 * \f$d = n_{\uparrow}n_{\downarrow} = \left(
	 * \begin{array}{cccc}
	 * 0 & 0 & 0 & 0\\
	 * 0 & 0 & 0 & 0\\
	 * 0 & 0 & 0 & 0\\
	 * 0 & 0 & 0 & 1\\
	 * \end{array}
	 * \right)\f$
	 */
	OperatorType d (int orbital=0) const;
	
	/**
	 * Spinon density \f$n_s=n-2d\f$
	 * \param orbital : orbital index
	 */
	OperatorType ns (int orbital=0) const
	{
		return n(orbital)-2.*d(orbital);
	};
	
	/**
	 * Holon density \f$n_h=2d-n-1=1-n_s\f$
	 * \param orbital : orbital index
	 */
	OperatorType nh (int orbital=0) const
	{
		return 2.*d(orbital)-n(orbital)+Id();
	};
	
	OperatorType cc (int orbital=0) const;
	OperatorType cdagcdag (int orbital=0) const;
	///\}
	
	///\{
	/**
	 * \param Sa
	 * \param orbital
	*/
	OperatorType Scomp (SPINOP_LABEL Sa, int orbital=0) const;
	
	/**
	 * For \p N_orbitals=1, this is
	 * \f$s^z = \left(
	 * \begin{array}{cccc}
	 * 0 & 0 & 0 & 0\\
	 * 0 & 0.5 & 0 & 0\\
	 * 0 & 0 & -0.5 & 0\\
	 * 0 & 0 & 0 & 0\\
	 * \end{array}
	 * \right)\f$
	 */
	OperatorType Sz (int orbital=0) const;
	
	/**
	 * For \p N_orbitals=1, this is
	 * \f$s^+ = \left(
	 * \begin{array}{cccc}
	 * 0 & 0 & 0 & 0\\
	 * 0 & 0 & 1 & 0\\
	 * 0 & 0 & 0 & 0\\
	 * 0 & 0 & 0 & 0\\
	 * \end{array}
	 * \right)\f$
	 */
	OperatorType Sp (int orbital=0) const;
	
	/**
	 * For \p N_orbitals=1, this is
	 * \f$s^- = \left(
	 * \begin{array}{cccc}
	 * 0 & 0 & 0 & 0\\
	 * 0 & 0 & 0 & 0\\
	 * 0 & 1 & 0 & 0\\
	 * 0 & 0 & 0 & 0\\
	 * \end{array}
	 * \right)\f$
	 */
	OperatorType Sm (int orbital=0) const;
	
	/**
	 * For \p N_orbitals=1, this is
	 * \f$s^x = \left(
	 * \begin{array}{cccc}
	 * 0 & 0 & 0 & 0\\
	 * 0 & 0 & 0.5 & 0\\
	 * 0 & 0.5 & 0 & 0\\
	 * 0 & 0 & 0 & 0\\
	 * \end{array}
	 * \right)\f$
	 */
	OperatorType Sx (int orbital=0) const;
	
	/**
	 * For \p N_orbitals=1, this is
	 * \f$is^y = \left(
	 * \begin{array}{cccc}
	 * 0 & 0 & 0 & 0\\
	 * 0 & 0 & 0.5 & 0\\
	 * 0 & -0.5 & 0 & 0\\
	 * 0 & 0 & 0 & 0\\
	 * \end{array}
	 * \right)\f$
	 */
	OperatorType iSy (int orbital=0) const;
	
	OperatorType Tz (int orbital=0) const;
	///\}
	
	///\{
	/**
	 * Fermionic sign for the hopping between two orbitals of nearest-neighbour supersites of a ladder. For \p N_orbitals=1, this is
	 * \f$(1-2n_{\uparrow})*(1-2n_{\downarrow}) = \left(
	 * \begin{array}{cccc}
	 * 1 & 0  & 0  & 0\\
	 * 0 & -1 & 0  & 0\\
	 * 0 & 0  & -1 & 0\\
	 * 0 & 0  & 0  & 1\\
	 * \end{array}
	 * \right)\f$
	 * \param orb1 : orbital on supersite i
	 * \param orb2 : orbital on supersite i+1
	 * \note \f$f\f$ anticommutes with \f$c\f$, \f$c^{\dagger}\f$ on the same site and commutes with \f$n\f$, \f$S^+\f$ and \f$S^-\f$
	 */
	OperatorType sign (int orb1=0, int orb2=0) const;

	/**The identiy operator. */
	OperatorType Id() const;
	
	/**Returns an array of size dim() with zeros.*/
	ArrayXd ZeroField() const;
	
	/**Returns an array of size dim() x dim() with zeros.*/
	ArrayXXd ZeroHopping() const;
	
	/**
	 * Fermionic sign for the transfer to a particular orbital, needed by HubbardModel::c and HubbardModel::cdag.
	 * \param orbital : orbital on the supersite
	 */
	OperatorType sign_local (int orbital) const;
	///\}
	
	/**
	 * Creates the full Hubbard Hamiltonian on the supersite with orbital-dependent U.
	 * \param U : \f$U\f$ for each orbital
	 * \param Uph : particle-hole symmetric \f$U\f$ for each orbital (times \f$(n_{\uparrow}-1/2)(n_{\downarrow}-1/2)+1/4\f$)
	 * \param Eorb : \f$E_i\f$ for each orbital (onsite energy)
	 * \param Bz : \f$B_z\f$ for each orbital
	 * \param Bx : \f$B_x\f$ for each orbital
	 * \param t : \f$t\f$
	 * \param V : \f$V\f$
	 * \param J : \f$J\f$
	 */
	template<typename Scalar> SiteOperator<Symmetry,Scalar>
	HubbardHamiltonian (const Array<Scalar,Dynamic,1> &U, 
	                    const Array<Scalar,Dynamic,1> &Uph, 
	                    const Array<Scalar,Dynamic,1> &Eorb, 
	                    const Array<Scalar,Dynamic,1> &Bz, 
	                    const Array<Scalar,Dynamic,1> &Bx, 
	                    const Array<Scalar,Dynamic,Dynamic> &t, 
	                    const Array<Scalar,Dynamic,Dynamic> &V, 
	                    const Array<Scalar,Dynamic,Dynamic> &J) const;
	
	/**Returns the local basis.*/
	vector<qarray<Symmetry::Nq> > get_basis() const;
	
	/**Returns the quantum numbers of the operators for the different combinations of U1 symmetries.*/
	typename Symmetry::qType getQ (SPIN_INDEX sigma, int Delta=0) const;
	typename Symmetry::qType getQ (SPINOP_LABEL Sa) const;
	
private:
	
	size_t N_orbitals;
	size_t N_states;
	
	/**
	 * Returns the qarray for a given index of the basis
	 * \param index
	 */ 
	qarray<Symmetry::Nq> qNums (size_t index) const;
	
	vector<boost::dynamic_bitset<unsigned char> > basis;
	
	double parity (const boost::dynamic_bitset<unsigned char> &state, int orbital) const;
};

template<typename Symmetry>
FermionBase<Symmetry>::
FermionBase (size_t L_input, bool U_IS_INFINITE)
:N_orbitals(L_input)
{
//	assert(N_orbitals>=1);
	
	size_t locdim = (U_IS_INFINITE)? 3 : 4;
	N_states = pow(locdim,N_orbitals);
	basis.resize(N_states);
	
	vector<int> nUP(locdim);
	vector<int> nDN(locdim);
	nUP[0] = 0; nDN[0] = 0;
	nUP[1] = 1; nDN[1] = 0;
	nUP[2] = 0; nDN[2] = 1;
	if (!U_IS_INFINITE) nUP[3] = 1; nDN[3] = 1;
	
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
	
	// set to vacuum for N_orbitals=0, reset to N_orbitals=1 to avoid segfaults
	if (N_states == 1)
	{
		N_orbitals = 1;
		basis[0].resize(2);
		basis[0][0] = 0;
		basis[0][1] = 0;
	}
	
//	cout << "L_input=" << L_input << ", N_states=" << N_states << endl;
//	cout << "basis:" << endl;
//	for (size_t i=0; i<N_states; i++) {cout << basis[i] << endl;}
//	cout << endl;
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
	
	return OperatorType(Mout, getQ(sigma,-1));
}

template<typename Symmetry>
SiteOperator<Symmetry,double> FermionBase<Symmetry>::
cdag (SPIN_INDEX sigma, int orbital) const
{
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
cc (int orbital) const
{
	return OperatorType(c(DN,orbital).data*c(UP,orbital).data, getQ(UPDN,-1)); // gets multiplied by 2 in getQ
}

template<typename Symmetry>
SiteOperator<Symmetry,double> FermionBase<Symmetry>::
cdagcdag (int orbital) const
{
//	return OperatorType(cdag(UP,orbital).data*cdag(DN,orbital).data, getQ(UPDN,+2));
//	return OperatorType((c(DN,orbital).data*c(UP,orbital).data).transpose(), getQ(UPDN,+2));
	return OperatorType(cc(orbital).data.transpose(), getQ(UPDN,+1));  // gets multiplied by 2 in getQ
}

template<typename Symmetry>
SiteOperator<Symmetry,double> FermionBase<Symmetry>::
Scomp (SPINOP_LABEL Sa, int orbital) const
{
	assert(Sa != SY);
	SiteOperator<Symmetry,double> out;
	if      (Sa==SX)  { out = Sx(orbital); }
	else if (Sa==iSY) { out = iSy(orbital); }
	else if (Sa==SZ)  { out = Sz(orbital); }
	else if (Sa==SP)  { out = Sp(orbital); }
	else if (Sa==SM)  { out = Sm(orbital); }
	return out;
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
Tz (int orbital) const
{
	return 0.5*(n(orbital)-Id());
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
ArrayXXd FermionBase<Symmetry>::
ZeroHopping() const
{
	return ArrayXXd::Zero(N_orbitals,N_orbitals);
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
HubbardHamiltonian (const Array<Scalar,Dynamic,1> &U, 
                    const Array<Scalar,Dynamic,1> &Uph, 
                    const Array<Scalar,Dynamic,1> &Eorb, 
                    const Array<Scalar,Dynamic,1> &Bz, 
                    const Array<Scalar,Dynamic,1> &Bx, 
                    const Array<Scalar,Dynamic,Dynamic> &t, 
                    const Array<Scalar,Dynamic,Dynamic> &V, 
                    const Array<Scalar,Dynamic,Dynamic> &J) const
{
	SparseMatrix<Scalar> Mout(N_states,N_states);
	
	for (int i=0; i<N_orbitals; ++i)
	for (int j=0; j<i; ++j)
	{
		if (t(i,j) != 0.)
		{
			Mout += -t(i,j)*(cdag(UP,i).data * c(UP,j).data + 
			                 cdag(DN,i).data * c(DN,j).data).template cast<Scalar>();
			Mout += -t(j,i)*(cdag(UP,j).data * c(UP,i).data + 
			                 cdag(DN,j).data * c(DN,i).data).template cast<Scalar>();
		}
		if (V(i,j) != 0.)
		{
			Mout += V(i,j) * (n(i).data*n(j).data).template cast<Scalar>();
		}
		if (J(i,j) != 0.)
		{
			Mout += J(i,j) * (0.5*Sp(i).data*Sm(j).data + 
			                  0.5*Sm(i).data*Sp(j).data + 
			                      Sz(i).data*Sz(j).data).template cast<Scalar>();
		}
	}
	
	for (int i=0; i<N_orbitals; ++i)
	{
		if (U(i) != 0. and U(i) != numeric_limits<double>::infinity())
		{
			Mout += U(i) * d(i).data.template cast<Scalar>();
		}
		if (Uph(i) != 0.)
		{
			if (Uph(i) != std::numeric_limits<double>::infinity())
			{
				Mout += Uph(i) * (d(i)-0.5*n(i)+0.5*Id()).data.template cast<Scalar>();
			}
			else
			{
				Mout += Uph(i) * (-0.5*n(i)+0.5*Id()).data.template cast<Scalar>();
			}
		}
		if (Eorb(i) != 0.)
		{
			Mout += Eorb(i) * n(i).data.template cast<Scalar>();
		}
		if (Bz(i) != 0.)
		{
			Mout -= Bz(i) * Sz(i).data.template cast<Scalar>();
		}
		if (Bx(i) != 0.)
		{
			Mout += Bx(i) * Sx(i).data.template cast<Scalar>();
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
	
	if constexpr (Symmetry::IS_TRIVIAL) { return qarray<0>{}; }
	else if constexpr (Symmetry::Nq == 1)
	{
		if constexpr      (Symmetry::kind()[0] == Sym::KIND::N)  {return qarray<1>{N};}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::M)  {return qarray<1>{M};}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::Z2) {return qarray<1>{posmod<2>(N)};}
		else {assert(false and "Ill defined KIND of the used Symmetry.");}
	}
	else if constexpr (Symmetry::Nq == 2)
	{
		if constexpr      (Symmetry::kind()[0] == Sym::KIND::N and Symmetry::kind()[1] == Sym::KIND::M) {return qarray<Symmetry::Nq>{N,M};}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::M and Symmetry::kind()[1] == Sym::KIND::N) {return qarray<Symmetry::Nq>{M,N};}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::Nup and Symmetry::kind()[1] == Sym::KIND::Ndn) {return qarray<Symmetry::Nq>{Nup,Ndn};}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::Ndn and Symmetry::kind()[1] == Sym::KIND::Nup) {return qarray<Symmetry::Nq>{Ndn,Nup};}
		else {assert(false and "Ill defined KIND of the used Symmetry.");}
	}
	// else {static_assert(false,"Three or more symmetries can not be handled from the code.");}
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
	if constexpr (Symmetry::IS_TRIVIAL) {return {};}
	else if constexpr (Symmetry::Nq == 1) 
	{
		if constexpr (Symmetry::kind()[0] == Sym::KIND::N) //return particle number as good quantum number.
		{
			typename Symmetry::qType out;
			if      (sigma==UP)     {out = {Delta};}
			else if (sigma==DN)     {out = {Delta};}
			else if (sigma==UPDN)   {out = {2*Delta};}
			else if (sigma==NOSPIN) {out = Symmetry::qvacuum();}
			return out;
		}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::M) //return magnetization as good quantum number.
		{
			typename Symmetry::qType out;
			if      (sigma==UP)     {out = {Delta};}
			else if (sigma==DN)     {out = {-Delta};}
			else if (sigma==UPDN)   {out = Symmetry::qvacuum();}
			else if (sigma==NOSPIN) {out = Symmetry::qvacuum();}
			return out;
		}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::Z2) //return parity as good quantum number.
		{
			typename Symmetry::qType out;
			if      (sigma==UP)     {out = {posmod<2>(Delta)};}
			else if (sigma==DN)     {out = {posmod<2>(-Delta)};}
			else if (sigma==UPDN)   {out = Symmetry::qvacuum();}
			else if (sigma==NOSPIN) {out = Symmetry::qvacuum();}
			return out;
		}
		else {assert(false and "Ill defined KIND of the used Symmetry.");}
	}
	else if constexpr (Symmetry::Nq == 2)
	{
		typename Symmetry::qType out;
		if constexpr (Symmetry::kind()[0] == Sym::KIND::N and Symmetry::kind()[1] == Sym::KIND::M)
		{
			if      (sigma==UP)     {out = {Delta,Delta};}
			else if (sigma==DN)     {out = {Delta,-Delta};}
			else if (sigma==UPDN)   {out = {2*Delta,0};}
			else if (sigma==NOSPIN) {out = Symmetry::qvacuum();}
		}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::M and Symmetry::kind()[1] == Sym::KIND::N)
		{
			if      (sigma==UP)     {out = {Delta,Delta};}
			else if (sigma==DN)     {out = {-Delta,Delta};}
			else if (sigma==UPDN)   {out = {0,2*Delta};}
			else if (sigma==NOSPIN) {out = Symmetry::qvacuum();}
		}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::Nup and Symmetry::kind()[1] == Sym::KIND::Ndn)
		{
			if      (sigma==UP)     {out = {Delta,0};}
			else if (sigma==DN)     {out = {0,Delta};}
			else if (sigma==UPDN)   {out = {Delta,Delta};}
			else if (sigma==NOSPIN) {out = Symmetry::qvacuum();}
		}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::Ndn and Symmetry::kind()[1] == Sym::KIND::Nup)
		{
			if      (sigma==UP)     {out = {0,Delta};}
			else if (sigma==DN)     {out = {Delta,0};}
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
	if constexpr (!Symmetry::IS_TRIVIAL)
	{
		assert(Sa != SX and Sa != iSY);
	}
	
	if constexpr (Symmetry::IS_TRIVIAL) {return {};}
	else if constexpr (Symmetry::Nq == 1)
	{
		if constexpr (Symmetry::kind()[0] == Sym::KIND::N or 
		              Symmetry::kind()[0] == Sym::KIND::Z2) //return particle number as good quantum number.
		{
			return Symmetry::qvacuum();
		}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::M) //return magnetization as good quantum number.
		{
			typename Symmetry::qType out;
			if (Sa==SZ)      {out = {0};}
			else if (Sa==SP) {out = {+2};}
			else if (Sa==SM) {out = {-2};}
			return out;
		}
		else {assert(false and "Ill defined KIND of the used Symmetry.");}
	}
	else if constexpr (Symmetry::Nq == 2)
	{
		typename Symmetry::qType out;
		if constexpr (Symmetry::kind()[0] == Sym::KIND::N and Symmetry::kind()[1] == Sym::KIND::M)
		{
			if (Sa==SZ) {out = {0,0};}
			else if (Sa==SP) {out = {0,+2};}
			else if (Sa==SM) {out = {0,-2};}
		}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::M and Symmetry::kind()[1] == Sym::KIND::N)
		{
			if (Sa==SZ) {out = {0,0};}
			else if (Sa==SP) {out = {+2,0};}
			else if (Sa==SM) {out = {-2,0};}
		}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::Nup and Symmetry::kind()[1] == Sym::KIND::Ndn)
		{
			if (Sa==SZ) {out = {0,0};}
			else if (Sa==SP) {out = {+1,-1};}
			else if (Sa==SM) {out = {-1,+1};}
		}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::Ndn and Symmetry::kind()[1] == Sym::KIND::Nup)
		{
			if (Sa==SZ) {out = {0,0};}
			else if (Sa==SP) {out = {-1,+1};}
			else if (Sa==SM) {out = {+1,-1};}
		}
		return out;
	}
	static_assert("You inserted a Symmetry which can not be handled by FermionBase.");
}

#endif
