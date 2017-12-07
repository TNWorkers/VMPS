#ifndef SITEOPERATOR_H_
#define SITEOPERATOR_H_

#include "qbasis.h"
#include "Biped.h"

/** \struct SiteOperator
  *
  * Just a struct to add a quantum number to a matrix which forms a plain SiteOperator.
  * For a SiteOperator blocked into different symmetry sectors, see SiteOperatorQ. 
  *
  * \describe_Symmetry
  * \describe_MatrixType
  *
  */
template<typename Symmetry, typename MatrixType>
struct SiteOperator
{
	SiteOperator() {};
	SiteOperator(const MatrixType& data_in, const typename Symmetry::qType& Q_in):Q(Q_in),data(data_in) {};
	typename Symmetry::qType Q;
	MatrixType data;
};

template<typename Symmetry,typename MatrixType>
SiteOperator<Symmetry,MatrixType> operator* (const SiteOperator<Symmetry,MatrixType>& O1, const SiteOperator<Symmetry,MatrixType>& O2)
{
//	assert(O1.basis() == O2.basis() and "For multiplication of SiteOperatorQs the basis needs to be the same.");
//	assert(O1.Q() == O2.Q() and "For multiplication of SiteOperatorQs the operator quantum number needs to be the same.");
	SiteOperator<Symmetry,MatrixType> out;
	out.data = O1.data * O2.data;
	out.Q = O1.Q+O2.Q;
	return out;
}

#endif
