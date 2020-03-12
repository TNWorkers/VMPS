#ifndef SITEOPERATOR_H_
#define SITEOPERATOR_H_

/// \cond
#include <Eigen/Sparse>
/// \endcond

#include "numeric_limits.h"

//forward declarations
template<typename Symmetry, typename MatrixType_> class SiteOperatorQ;
template<typename Symmetry> class Qbasis;
template<typename Symmetry, typename MatrixType_> struct Biped;

//include "tensors/Qbasis.h"
//include "tensors/Biped.h"
//include "tensors/SiteOperatorQ.h"

/** \struct SiteOperator
 *
 * \ingroup Tensors
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
	SiteOperator (const Eigen::SparseMatrix<Scalar> &data_input, const typename Symmetry::qType& Q_input, std::string label_input="")
		:data(data_input), Q(Q_input), label(label_input)
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
		Oout.label = label;
		return Oout;
	}
	
	SiteOperator<Symmetry,Scalar>& operator+= ( const SiteOperator<Symmetry,Scalar>& Op );
	SiteOperator<Symmetry,Scalar>& operator-= ( const SiteOperator<Symmetry,Scalar>& Op );
	
	/**
	 * Returns a trivial SiteOperatorQ for an object with has essentialy no symmetry.
	 */
	SiteOperatorQ<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > structured();
	
	void setIdentity();
    
    std::string label = "";
};

template<typename Symmetry,typename Scalar>
SiteOperatorQ<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > SiteOperator<Symmetry,Scalar>::
structured()
{
	Qbasis<Symmetry> basis; basis.push_back(Symmetry::qvacuum(),this->data.rows());
	Biped<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > mat; mat.push_back(Symmetry::qvacuum(),Symmetry::qvacuum(),this->data);
	SiteOperatorQ<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > out(Symmetry::qvacuum(),basis,mat);
	return out;
}

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
    std::stringstream labelstream;
    labelstream << "(" << O1.label << "*" << O2.label << ")";
    Oout.label = labelstream.str();
	return Oout;
}

template<typename Symmetry, typename Scalar>
SiteOperator<Symmetry,Scalar> operator+ (const SiteOperator<Symmetry,Scalar> &O1, const SiteOperator<Symmetry,Scalar> &O2)
{
	assert(O1.Q == O2.Q and "For addition of SiteOperators the operator quantum number needs to be the same.");
	SiteOperator<Symmetry,Scalar> Oout;
	Oout.data = O1.data + O2.data;
	Oout.Q = O1.Q;
    std::stringstream labelstream;
    labelstream << "(" << O1.label << "+" << O2.label << ")";
    Oout.label = labelstream.str();
	return Oout;
}

template<typename Symmetry, typename Scalar>
SiteOperator<Symmetry,Scalar> operator- (const SiteOperator<Symmetry,Scalar> &O1, const SiteOperator<Symmetry,Scalar> &O2)
{
	assert(O1.Q == O2.Q and "For addition of SiteOperators the operator quantum number needs to be the same.");
	SiteOperator<Symmetry,Scalar> Oout;
	Oout.data = O1.data - O2.data;
	Oout.Q = O1.Q;
    std::stringstream labelstream;
    labelstream << "(" << O1.label << "-" << O2.label << ")";
    Oout.label = labelstream.str();
	return Oout;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
SiteOperator<Symmetry,Scalar> operator* (const OtherScalar &x, const SiteOperator<Symmetry,Scalar> &O)
{
	SiteOperator<Symmetry,Scalar> Oout;
	Oout.data = x * O.data;
	Oout.Q = O.Q;
    std::stringstream labelstream;
    if(std::abs(x-1.) > ::mynumeric_limits<double>::epsilon())
    {
        labelstream << "(" << x << "*" << O.label << ")";
    }
    else
    {
        labelstream << O.label;
    }
    Oout.label = labelstream.str();
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

template<typename Symmetry, typename Scalar>
bool operator== (const SiteOperator<Symmetry,Scalar> &O1, const SiteOperator<Symmetry,Scalar> &O2)
{
    if(O1.Q == O2.Q)
    {
        return ((O1.data - O2.data).norm() < ::mynumeric_limits<double>::epsilon());
    }
    return false;
}

template<typename Symmetry, typename Scalar>
SiteOperator<Symmetry, Scalar> operator*(const SiteOperator<Symmetry,Scalar> &op, const Scalar &lambda)
{
    return lambda*op;
}

template<typename Symmetry, typename Scalar>
void SiteOperator<Symmetry,Scalar>::
setIdentity()
{
	Q = Symmetry::qvacuum();
	data.setIdentity();
}

template<typename Symmetry, typename Scalar>
std::vector<SiteOperator<Symmetry,Scalar>> operator*(const std::vector<SiteOperator<Symmetry,Scalar>> &ops, const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &mat)
{
    assert(ops.size() == mat.rows() and "Dimensions of vector and matrix do not match!");
    std::vector<SiteOperator<Symmetry, Scalar>> out;
    for(std::size_t j=0; j<mat.cols(); ++j)
    {
        SiteOperator<Symmetry, Scalar> temp;
        std::size_t i=0;
        for(; std::abs(mat(i,j)) < ::mynumeric_limits<double>::epsilon() and i<mat.rows()-1; ++i){}
        if(i == mat.rows()-1 and mat.row(i).norm() < ::mynumeric_limits<double>::epsilon())
        {
            temp = 0*ops[j];
        }
        else
        {
            temp = ops[i]*mat(i,j);
            ++i;
        }
        for(; i<mat.rows(); ++i)
        {
            if(std::abs(mat(i,j)) > ::mynumeric_limits<double>::epsilon())
            {
                temp += ops[i]*mat(i,j);
            }
        }
        out.push_back(temp);
    }
    return out;
}

template<typename Symmetry, typename Scalar>
std::vector<SiteOperator<Symmetry,Scalar>> operator*(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &mat, const std::vector<SiteOperator<Symmetry,Scalar>> &ops)
{
    assert(ops.size() == mat.cols() and "Dimensions of matrix and vector do not match!");
    std::vector<SiteOperator<Symmetry, Scalar>> out;
    for(std::size_t i=0; i<mat.rows(); ++i)
    {
        SiteOperator<Symmetry, Scalar> temp;
        std::size_t j=0;
        for(; std::abs(mat(i,j)) < ::mynumeric_limits<double>::epsilon() and j<mat.cols()-1; ++j){}
        if(j == mat.cols()-1 and mat.row(i).norm() < ::mynumeric_limits<double>::epsilon())
        {
            temp = 0*ops[i];
        }
        else
        {
            temp = mat(i,j)*ops[j];
            ++j;
        }
        
        for(; j<mat.cols(); ++j)
        {
            if(std::abs(mat(i,j)) > ::mynumeric_limits<double>::epsilon())
            {
                temp += mat(i,j)*ops[j];
            }
        }
        out.push_back(temp);
    }
    return out;
}

#endif
