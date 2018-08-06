#ifndef GENERICLANCZOSWRAPPERS
#define GENERICLANCZOSWRAPPERS

#include <Eigen/Dense>
#include <Eigen/Sparse>
using namespace Eigen;

//#include "HilbertTypedefs.h"

template<typename Scalar>
inline size_t dim (const SparseMatrix<Scalar> &M)
{
	return M.rows();
}

//inline double dot (const VectorXd &V1, const VectorXd &V2)
//{
//	return V1.dot(V2);
//}

//inline double dot (const VectorXcd &V1, const VectorXcd &V2)
//{
//	return (V1.dot(V2)).real();
//}

template<typename Scalar>
inline Scalar dot (const Matrix<Scalar,Dynamic,1> &V1, const Matrix<Scalar,Dynamic,1> &V2)
{
	return V1.dot(V2);
}

template<typename Scalar>
inline Scalar dim (const Matrix<Scalar,Dynamic,1> &V)
{
	return V.rows();
}

template<typename Scalar>
inline void swap (Matrix<Scalar,Dynamic,1> &V1, Matrix<Scalar,Dynamic,1> &V2)
{
	V1.swap(V2);
}

//inline double dot (const VectorXd &V1, const VectorXcd &V2)
//{
//	return (V1.cast<complex<double> >().dot(V2)).real();
//}

//inline double dot (const VectorXcd &V1, const VectorXd &V2)
//{
//	return (V1.dot(V2.cast<complex<double> >())).real();
//}

template<typename Scalar>
inline double norm (const Matrix<Scalar,Dynamic,1> &V)
{
	return V.norm();
}

template<typename Scalar>
inline double squaredNorm (const Matrix<Scalar,Dynamic,1> &V)
{
	return V.squaredNorm();
}

template<typename Scalar>
inline void normalize (Matrix<Scalar,Dynamic,1> &V)
{
	V.normalize();
}

//template<typename Scalar1, typename Scalar2>
//inline void scale_add (Scalar1 alpha, const Matrix<Scalar2,Dynamic,1> &Vin, Matrix<Scalar2,Dynamic,1> &Vout)
//{
//	Vout += alpha * Vin;
//}

//template<typename Scalar1, typename Scalar2>
//inline void scale_subtract (Scalar1 alpha, const Matrix<Scalar2,Dynamic,1> &Vin, Matrix<Scalar2,Dynamic,1> &Vout)
//{
//	Vout -= alpha * Vin;
//}

//template<typename Scalar>
//inline void set_zero (Matrix<Scalar,Dynamic,1> &V)
//{
//	V.setZero();
//}

template<typename Scalar>
inline double norm (const SparseMatrix<Scalar> &M)
{
	return M.norm();
}

template<typename Scalar>
inline double infNorm (const Matrix<Scalar,Dynamic,1> &V1, const Matrix<Scalar,Dynamic,1> &V2)
{
	return (V1-V2).template lpNorm<Eigen::Infinity>();
}

#endif
