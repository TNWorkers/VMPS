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

#endif
