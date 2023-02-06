#ifndef SPINLESSFERMIONBASE
#define SPINLESSFERMIONBASE

/// \cond
#include <algorithm>
#include <iterator>

#include <boost/dynamic_bitset.hpp>
#include <Eigen/Core>
/// \endcond

#include "tensors/SiteOperatorQ.h"
#include "sites/SpinlessFermionSite.h"
#include <boost/dynamic_bitset.hpp>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>

#include "symmetry/kind_dummies.h"
#include "tensors/SiteOperator.h"
#include "DmrgExternal.h" // for posmod

/** 
 * \class SpinlessSpinlessFermionBase
 * \ingroup Bases
 *
 * This class provides the local operators for spinless fermions.
 *
 */
template<typename Symmetry_>
class SpinlessFermionBase  : public SpinlessFermionSite<Symmetry_>
{
	typedef Eigen::Index Index;
	typedef double Scalar;
	
public:
	
	typedef Symmetry_ Symmetry;
	typedef SiteOperatorQ<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > OperatorType;
	typedef typename Symmetry::qType qType;
	
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
	OperatorType c    (std::size_t orbital=0) const;
	OperatorType cdag (std::size_t orbital=0) const;
	OperatorType n    (std::size_t orbital=0) const;
	OperatorType nph  (std::size_t orbital=0) const;
	OperatorType sign (std::size_t orb1=0, std::size_t orb2=0) const;
	OperatorType Id (std::size_t orbital=0) const;
	//OperatorType Zero() const;
	///\}
	
	/**Returns an array of size dim() with zeros.*/
	ArrayXd ZeroField() const { return ArrayXd::Zero(N_orbitals); }
	
	/**Returns an array of size dim()xdim() with zeros.*/
	ArrayXXd ZeroHopping() const { return ArrayXXd::Zero(N_orbitals,N_orbitals); }
	
//	typename Symmetry::qType getQ (int Delta=0) const;
	
	/**Returns the basis.*/
	Qbasis<Symmetry> get_basis() const { return TensorBasis; }
	
private:
	
	OperatorType make_operator(const OperatorType &Op_1s, size_t orbital=0, bool FERMIONIC = false, string label="") const;
	
	size_t N_orbitals;
	size_t N_states;
	
	/**
	 * Returns the qarray for a given index of the basis
	 * \param index
	 */ 
//	qarray<Symmetry::Nq> qNums (size_t index) const;
	
//	vector<boost::dynamic_bitset<unsigned char> > basis;
	
	Qbasis<Symmetry> TensorBasis; //Final basis for N_orbital sites
	
	//double parity (const boost::dynamic_bitset<unsigned char> &state, int orbital) const;
	
	//operators defined on zero orbitals
	OperatorType Id_vac, Zero_vac;
};

template<typename Symmetry_>
SpinlessFermionBase<Symmetry_>::
SpinlessFermionBase (size_t L_input)
:SpinlessFermionSite<Symmetry>(), N_orbitals(L_input)
{
	//create basis for zero orbitals
	typename Symmetry::qType Q=Symmetry::qvacuum();
	Eigen::Index inner_dim = 1;
	Qbasis<Symmetry_> vacuum;
	vacuum.push_back(Q, inner_dim);
	
	// create operators for zero orbitals
	Zero_vac = OperatorType(Symmetry::qvacuum(), vacuum);
	Zero_vac.setZero();
	Id_vac = OperatorType(Symmetry::qvacuum(), vacuum);
	Id_vac.setIdentity();
	
	// create basis for N_orbitals fermionic sites
	if      (N_orbitals == 1) {TensorBasis = this->basis_1s();}
	else if (N_orbitals == 0) {TensorBasis = vacuum;}
	else
	{
		TensorBasis = this->basis_1s().combine(this->basis_1s());
		for (std::size_t o=2; o<N_orbitals; o++)
		{
			TensorBasis = TensorBasis.combine(this->basis_1s());
		}
	}
	
	N_states = TensorBasis.size();
}

template<typename Symmetry_>
SiteOperatorQ<Symmetry_, Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> >
SpinlessFermionBase<Symmetry_>::
make_operator (const OperatorType &Op_1s, size_t orbital, bool FERMIONIC, string label) const
{
	OperatorType out;
	if (N_orbitals == 1) {out = Op_1s; out.label() = label; return out;}
	else if (N_orbitals == 0) {return Zero_vac;}
	else
	{
		OperatorType stringOp;
		if (FERMIONIC) {stringOp = this->F_1s();}
		else {stringOp = this->Id_1s();}
		bool TOGGLE=false;
		if (orbital == 0) {out = OperatorType::outerprod(Op_1s,this->Id_1s(),Op_1s.Q()); TOGGLE=true;}
		else
		{
			if (orbital == 1) {out = OperatorType::outerprod(stringOp,Op_1s,Op_1s.Q()); TOGGLE=true;}
			else {out = OperatorType::outerprod(stringOp,stringOp,Symmetry_::qvacuum());}
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if      (orbital == o)  {out = OperatorType::outerprod(out,Op_1s,Op_1s.Q()); TOGGLE=true; }
			else if (TOGGLE==false) {out = OperatorType::outerprod(out,stringOp,Symmetry_::qvacuum());}
			else if (TOGGLE==true)  {out = OperatorType::outerprod(out,this->Id_1s(),Op_1s.Q());}
		}
		out.label() = label;
		return out;
	}
}

template<typename Symmetry_>
SiteOperatorQ<Symmetry_, Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > 
SpinlessFermionBase<Symmetry_>::
c (std::size_t orbital) const
{
	return make_operator(this->c_1s(),orbital,PROP::FERMIONIC, "c");
}

template<typename Symmetry_>
SiteOperatorQ<Symmetry_, Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> >
SpinlessFermionBase<Symmetry_>::
cdag (std::size_t orbital) const
{
	return make_operator(this->cdag_1s(),orbital,PROP::FERMIONIC, "c");
}

template<typename Symmetry_>
SiteOperatorQ<Symmetry_, Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> >
SpinlessFermionBase<Symmetry_>::
n (std::size_t orbital) const
{
	return make_operator(this->n_1s(), orbital, PROP::NON_FERMIONIC, "n");
}

template<typename Symmetry_>
SiteOperatorQ<Symmetry_, Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> >
SpinlessFermionBase<Symmetry_>::
nph (std::size_t orbital) const
{
	return make_operator(this->nph_1s(), orbital, PROP::NON_FERMIONIC, "n");
}

template <typename Symmetry_>
SiteOperatorQ<Symmetry_,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> >
SpinlessFermionBase<Symmetry_>::
sign (std::size_t orb1, std::size_t orb2) const
{
	OperatorType Oout;
	if (N_orbitals == 1) {Oout = this->F_1s(); Oout.label()="sign"; return Oout;}
	else if (N_orbitals == 0) {return Zero_vac;}
	else
	{
		Oout = Id();
		for (int i=orb1; i<N_orbitals; ++i)
		{
			Oout = Oout * n(i);
		}
		for (int i=0; i<orb2; ++i)
		{
			Oout = Oout * n(i);
		}
		Oout.label() = "sign";
		return Oout;
	}
}

template <typename Symmetry_>
SiteOperatorQ<Symmetry_,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> >
SpinlessFermionBase<Symmetry_>::
Id (std::size_t orbital) const
{
	return make_operator(this->Id_1s(), orbital, PROP::NON_FERMIONIC, "Id");
}

//template<typename Symmetry_>
//SiteOperator<Symmetry,double> SpinlessFermionBase<Symmetry_>::
//Zero() const
//{
//	SparseMatrixXd mat = MatrixXd::Zero(N_states,N_states).sparseView();
//	OperatorType Oout(mat,Symmetry::qvacuum());
//	return Oout;
//}

//template<typename Symmetry_>
//double SpinlessFermionBase<Symmetry_>::
//parity (const boost::dynamic_bitset<unsigned char> &b, int i) const
//{
//	double out = 1.;
//	for (int j=0; j<i; ++j)
//	{
//		if (b[j]) {out *= -1.;} // switch sign for every particle found between 0 & i
//	}
//	return out;
//}

//template<typename Symmetry_>
//qarray<Symmetry::Nq> SpinlessFermionBase<Symmetry_>::
//qNums (size_t index) const
//{
//	int N = 0;
//	
//	for (size_t i=0; i<N_orbitals; i++)
//	{
//		if (basis[index][i]) {N+=1;}
//	}
//	
//	if constexpr (Symmetry::IS_TRIVIAL) { return qarray<0>{}; }
//	else if constexpr (Symmetry::Nq == 1)
//	{
//		if constexpr      (Symmetry::kind()[0] == Sym::KIND::N)  {return qarray<1>{N};}
//		else if constexpr (Symmetry::kind()[0] == Sym::KIND::Z2) {return qarray<1>{posmod<2>(N)};}
//		else {assert(false and "Ill defined KIND of the used Symmetry.");}
//	}
//}

//template<typename Symmetry_>
//vector<qarray<Symmetry::Nq> > SpinlessFermionBase<Symmetry_>::
//get_basis() const
//{
//	vector<qarray<Symmetry::Nq> > vout;
//	
//	for (size_t i=0; i<N_states; ++i)
//	{
//		vout.push_back(qNums(i));
//	}
//	
//	return vout;
//}

//template<typename Symmetry_>
//typename Symmetry::qType SpinlessFermionBase<Symmetry_>::
//getQ (int Delta) const
//{
//	if constexpr (Symmetry::IS_TRIVIAL) {return {};}
//	else if constexpr (Symmetry::Nq == 1) 
//	{
//		if constexpr (Symmetry::kind()[0] == Sym::KIND::N) //return particle number as good quantum number.
//		{
//			return {Delta};
//		}
//		else if constexpr (Symmetry::kind()[0] == Sym::KIND::Z2) //return parity as good quantum number.
//		{
//			return {posmod<2>(Delta)};
//		}
//		else {assert(false and "Ill defined KIND of the used Symmetry.");}
//	}
//	static_assert("You inserted a Symmetry which can not be handled by SpinlessFermionBase.");
//}

#endif
