#ifndef SPINLESSFERMIONBASE
#define SPINLESSFERMIONBASE

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
 * \class SpinlessSpinlessFermionBase
 * \ingroup Bases
 *
 * This class provides the local operators for spinless fermions.
 *
 */
template<typename Symmetry>
class SpinlessFermionBase
{
	typedef SiteOperator<Symmetry,double> OperatorType;
	
public:
	
	SpinlessFermionBase(){};
	
	/**
	 * \param L_input : the amount of orbitals
	 */
	SpinlessFermionBase (size_t L_input);
	
	/**number of states = \f$2^L\f$*/
	inline size_t dim() const {return N_states;}
	
	/**number of orbitals*/
	inline size_t orbitals() const {return N_orbitals;}
	
	///\{
	OperatorType c    (int orbital=0) const;
	OperatorType cdag (int orbital=0) const;
	OperatorType n    (int orbital=0) const;
	OperatorType sign (int orb1=0, int orb2=0) const;
	OperatorType Id() const;
	OperatorType Zero() const;
	///\}
	
	/**Returns an array of size dim() with zeros.*/
	ArrayXd ZeroField() const;
	
	/**Returns an array of size dim() x dim() with zeros.*/
	ArrayXXd ZeroHopping() const;
	
	/**Returns the local basis.*/
	vector<qarray<Symmetry::Nq> > get_basis() const;
	
	typename Symmetry::qType getQ (int Delta=0) const;
	
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
SpinlessFermionBase<Symmetry>::
SpinlessFermionBase (size_t L_input)
:N_orbitals(L_input)
{
	size_t locdim = 2;
	N_states = pow(locdim,N_orbitals);
	basis.resize(N_states);
	
	vector<int> n(2);
	n[0] = 0;
	n[1] = 1;
	
	NestedLoopIterator Nelly(N_orbitals,locdim);
	for (Nelly=Nelly.begin(); Nelly!=Nelly.end(); ++Nelly)
	{
		basis[Nelly.index()].resize(N_orbitals);
		for (int i=0; i<N_orbitals; ++i)
		{
			basis[*Nelly][i] = n[Nelly(i)];
		}
	}
	
	// set to vacuum for N_orbitals=0, reset to N_orbitals=1 to avoid segfaults
	if (N_states == 1)
	{
		N_orbitals = 1;
		basis[0].resize(1);
		basis[0][0] = 0;
	}
	
//	cout << "L_input=" << L_input << ", N_states=" << N_states << endl;
//	cout << "basis:" << endl;
//	for (size_t i=0; i<N_states; i++) {cout << basis[i] << endl;}
//	cout << endl;
}

template<typename Symmetry>
SiteOperator<Symmetry,double> SpinlessFermionBase<Symmetry>::
c (int orbital) const
{
	SparseMatrixXd Mout(N_states,N_states);
	
	for (int j=0; j<basis.size(); ++j)
	{
		if (basis[j][orbital])
		{
			boost::dynamic_bitset<unsigned char> b = basis[j];
			b[orbital].flip();
			
			auto it = find(basis.begin(), basis.end(), b);
			
			if (it!=basis.end())
			{
				int i = distance(basis.begin(), it);
				Mout.insert(i,j) = parity(b, orbital);
			}
		}
	}
	
	return OperatorType(Mout, getQ(-1));
}

template<typename Symmetry>
SiteOperator<Symmetry,double> SpinlessFermionBase<Symmetry>::
cdag (int orbital) const
{
	return OperatorType(c(orbital).data.transpose(), getQ(+1));
}

template<typename Symmetry>
SiteOperator<Symmetry,double> SpinlessFermionBase<Symmetry>::
n (int orbital) const
{
	SparseMatrixXd Mout(N_states,N_states);
	for (int j=0; j<basis.size(); ++j)
	{
		Mout.insert(j,j) = 1.*basis[j][orbital];
	}
	
	OperatorType Oout(Mout,Symmetry::qvacuum());
	return Oout;
}

template<typename Symmetry>
SiteOperator<Symmetry,double> SpinlessFermionBase<Symmetry>::
sign (int orb1, int orb2) const
{
	SparseMatrixXd Id = MatrixXd::Identity(N_states,N_states).sparseView();
	SparseMatrixXd Mout = Id;
	
	for (int i=orb1; i<N_orbitals; ++i)
	{
		Mout = Mout * (Id-2.*n(i).data);
	}
	for (int i=0; i<orb2; ++i)
	{
		Mout = Mout * (Id-2.*n(i).data);
	}
	
	return OperatorType(Mout,Symmetry::qvacuum());
}

template<typename Symmetry>
SiteOperator<Symmetry,double> SpinlessFermionBase<Symmetry>::
Id() const
{
	SparseMatrixXd mat = MatrixXd::Identity(N_states,N_states).sparseView();
	OperatorType Oout(mat,Symmetry::qvacuum());
	return Oout;
}

template<typename Symmetry>
SiteOperator<Symmetry,double> SpinlessFermionBase<Symmetry>::
Zero() const
{
	SparseMatrixXd mat = MatrixXd::Zero(N_states,N_states).sparseView();
	OperatorType Oout(mat,Symmetry::qvacuum());
	return Oout;
}

template<typename Symmetry>
ArrayXd SpinlessFermionBase<Symmetry>::
ZeroField() const
{
	return ArrayXd::Zero(N_orbitals);
}

template<typename Symmetry>
ArrayXXd SpinlessFermionBase<Symmetry>::
ZeroHopping() const
{
	return ArrayXXd::Zero(N_orbitals,N_orbitals);
}

template<typename Symmetry>
double SpinlessFermionBase<Symmetry>::
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
qarray<Symmetry::Nq> SpinlessFermionBase<Symmetry>::
qNums (size_t index) const
{
	int N = 0;
	
	for (size_t i=0; i<N_orbitals; i++)
	{
		if (basis[index][i]) {N+=1;}
	}
	
	if constexpr (Symmetry::IS_TRIVIAL) { return qarray<0>{}; }
	else if constexpr (Symmetry::Nq == 1)
	{
		if constexpr      (Symmetry::kind()[0] == Sym::KIND::N)  {return qarray<1>{N};}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::Z2) {return qarray<1>{posmod<2>(N)};}
		else {assert(false and "Ill defined KIND of the used Symmetry.");}
	}
}

template<typename Symmetry>
vector<qarray<Symmetry::Nq> > SpinlessFermionBase<Symmetry>::
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
typename Symmetry::qType SpinlessFermionBase<Symmetry>::
getQ (int Delta) const
{
	if constexpr (Symmetry::IS_TRIVIAL) {return {};}
	else if constexpr (Symmetry::Nq == 1) 
	{
		if constexpr (Symmetry::kind()[0] == Sym::KIND::N) //return particle number as good quantum number.
		{
			return {Delta};
		}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::Z2) //return parity as good quantum number.
		{
			return {posmod<2>(Delta)};
		}
		else {assert(false and "Ill defined KIND of the used Symmetry.");}
	}
	static_assert("You inserted a Symmetry which can not be handled by SpinlessFermionBase.");
}

#endif
