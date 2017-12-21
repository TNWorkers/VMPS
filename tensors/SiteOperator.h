#ifndef SITEOPERATOR_H_
#define SITEOPERATOR_H_

#include "symmetry/qbasis.h"
#include "tensors/Biped.h"
#include <Eigen/Sparse>

/** \struct SiteOperator
  *
  * Just a struct to add a quantum number to a matrix which forms a plain SiteOperator.
  * For a SiteOperator blocked into different symmetry sectors, see SiteOperatorQ. 
  *
  * \describe_Symmetry
  * \describe_Scalar
  *
  */
template<typename Symmetry, typename Scalar>
struct SiteOperator
{
	SiteOperator() {};
	SiteOperator (const Eigen::SparseMatrix<Scalar> &data_input, const typename Symmetry::qType& Q_input)
	:data(data_input), Q(Q_input)
	{};
	
	typename Symmetry::qType Q = Symmetry::qvacuum();
	Eigen::SparseMatrix<Scalar> data;
	
	void setZero()
	{
		data.setZero();
		Q = Symmetry::qvacuum();
	}
	
	template<typename OtherScalar>
	SiteOperator<Symmetry,OtherScalar> cast() const
	{
		SiteOperator<Symmetry,OtherScalar> Oout;
		Oout.Q = Q;
		Oout.data = data.template cast<OtherScalar>();
		return Oout;
	}

	SiteOperator<Symmetry,Scalar>& operator+= ( const SiteOperator<Symmetry,Scalar>& Op );
	SiteOperator<Symmetry,Scalar>& operator-= ( const SiteOperator<Symmetry,Scalar>& Op );
};
template<typename Symmetry,typename Scalar>
SiteOperator<Symmetry,Scalar>& SiteOperator<Symmetry,Scalar>::operator+= ( const SiteOperator<Symmetry,Scalar>& Op )
{
	*this = *this + Op;
	return *this;
}

template<typename Symmetry,typename Scalar>
SiteOperator<Symmetry,Scalar>& SiteOperator<Symmetry,Scalar>::operator-= ( const SiteOperator<Symmetry,Scalar>& Op )
{
	*this = *this - Op;
	return *this;
}

template<typename Symmetry, typename Scalar>
SiteOperator<Symmetry,Scalar> operator* (const SiteOperator<Symmetry,Scalar> &O1, const SiteOperator<Symmetry,Scalar> &O2)
{
	SiteOperator<Symmetry,Scalar> Oout;
	Oout.data = O1.data * O2.data;
	Oout.Q = O1.Q+O2.Q;
	return Oout;
}

template<typename Symmetry, typename Scalar>
SiteOperator<Symmetry,Scalar> operator+ (const SiteOperator<Symmetry,Scalar> &O1, const SiteOperator<Symmetry,Scalar> &O2)
{
	assert(O1.Q == O2.Q and "For addition of SiteOperators the operator quantum number needs to be the same.");
	SiteOperator<Symmetry,Scalar> Oout;
	Oout.data = O1.data + O2.data;
	Oout.Q = O1.Q;
	return Oout;
}

template<typename Symmetry, typename Scalar>
SiteOperator<Symmetry,Scalar> operator- (const SiteOperator<Symmetry,Scalar> &O1, const SiteOperator<Symmetry,Scalar> &O2)
{
	assert(O1.Q == O2.Q and "For addition of SiteOperators the operator quantum number needs to be the same.");
	SiteOperator<Symmetry,Scalar> Oout;
	Oout.data = O1.data - O2.data;
	Oout.Q = O1.Q;
	return Oout;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
SiteOperator<Symmetry,Scalar> operator* (const OtherScalar &x, const SiteOperator<Symmetry,Scalar> &O)
{
	SiteOperator<Symmetry,Scalar> Oout;
	Oout.data = x * O.data;
	Oout.Q = O.Q;
	return Oout;
}

template<typename Symmetry, typename Scalar>
SiteOperator<Symmetry,Scalar> kroneckerProduct (const SiteOperator<Symmetry,Scalar> &O1, const SiteOperator<Symmetry,Scalar> &O2)
{
	SiteOperator<Symmetry,Scalar> Oout;
	Oout.data = kroneckerProduct(O1.data,O2.data);
	Oout.Q = O1.Q+O2.Q;
	return Oout;
}

#endif
