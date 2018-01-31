#ifndef HILBERT_HXV
#define HILBERT_HXV

#include "HilbertTypedefs.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
using namespace Eigen;

#include "PerfectShuffle.h"

// real sparse H * real/complex vector
//template<typename Scalar>
//inline void HxV (const SparseMatrixXd &H, const Matrix<Scalar,Dynamic,1> &Vin, Matrix<Scalar,Dynamic,1> &Vout)
template<typename Scalar1, typename Scalar2>
inline void HxV (const SparseMatrix<Scalar1> &H, const Matrix<Scalar2,Dynamic,1> &Vin, Matrix<Scalar2,Dynamic,1> &Vout)
{
	Vout.noalias() = H.template selfadjointView<Upper>() * Vin;
}

// real sparse H * real/complex vector in place
//template<typename Scalar>
//inline void HxV (const SparseMatrixXd &H, Matrix<Scalar,Dynamic,1> &Vinout)
template<typename Scalar1, typename Scalar2>
inline void HxV (const SparseMatrix<Scalar1> &H, Matrix<Scalar2,Dynamic,1> &Vinout)
{
	Vinout = H.template selfadjointView<Upper>() * Vinout;
}

template<typename Scalar>
void addScale (const Scalar alpha, const Matrix<Scalar,Dynamic,1> &Vin, Matrix<Scalar,Dynamic,1> &Vout)
{
	Vout += alpha * Vin;
}

// real sparse O * real/complex vector
template<typename Scalar>
inline void OxV (const SparseMatrixXd &O, const Matrix<Scalar,Dynamic,1> &Vin, Matrix<Scalar,Dynamic,1> &Vout)
{
	Vout.noalias() = O * Vin;
}

// real sparse O * real/complex vector in place
template<typename Scalar>
inline void OxV (const SparseMatrixXd &O, Matrix<Scalar,Dynamic,1> &Vinout)
{
	Vinout = O * Vinout;
}

template<typename Scalar1, typename Scalar2>
inline void chebIter (const SparseMatrix<Scalar1> &H, const Matrix<Scalar2,Dynamic,1> &Vin1, const Matrix<Scalar2,Dynamic,1> &Vin2, Matrix<Scalar2,Dynamic,1> &Vout)
{
	HxV(H,Vin1,Vout);
	Vout -= Vin2;
}

template<typename Scalar1, typename Scalar2>
inline Scalar2 avg (const Matrix<Scalar2,Dynamic,1> &Vbra, const SparseMatrix<Scalar1> &O, const Matrix<Scalar2,Dynamic,1> &Vket)
{
	return Vbra.dot(O*Vket);
}

//// complex sparse H * real/complex vector
//template<typename Scalar>
//inline void HxV (const SparseMatrixXcd &H, const Matrix<Scalar,Dynamic,1> &Vin, Matrix<Scalar,Dynamic,1> &Vout)
//{
//	Vout = H.selfadjointView<Upper>()*Vin;
//}

//// complex sparse H * real/complex vector in place
//template<typename Scalar>
//inline void HxV (const SparseMatrixXcd &H, Matrix<Scalar,Dynamic,1> &Vinout)
//{
//	Vinout = H.selfadjointView<Upper>()*Vinout;
//}

// real diagonal matrix (stored as vector) * real/complex vector
template<typename Scalar>
inline void HxV (const VectorXd &H, const Matrix<Scalar,Dynamic,1> &Vin, Matrix<Scalar,Dynamic,1> &Vout)
{
	Vout.noalias() = H.asDiagonal() * Vin;
}

// real diagonal matrix (stored as vector) * real/complex vector in place
template<typename Scalar>
inline void HxV (const VectorXd &H, Matrix<Scalar,Dynamic,1> &Vinout)
{
	Vinout = H.asDiagonal() * Vinout;
}

// extract vector segment(pos,n) from Vin into Vout
template<typename Scalar>
inline void get_segment (size_t pos, size_t n, const Matrix<Scalar,Dynamic,1> &Vin, Matrix<Scalar,Dynamic,1> &Vout)
{
	Vout = Vin.segment(pos,n);
}

// set vector segment(pos,n) of Vout to Vin
template<typename Scalar>
inline void set_segment (size_t pos, size_t n, const Matrix<Scalar,Dynamic,1> &Vin, Matrix<Scalar,Dynamic,1> &Vout)
{
	Vout.segment(pos,n) = Vin;
}

// Vout = I(dim_left,dim_left) ⊗ H(dimH,dimH) ⊗ I(dim_right,dim_right) * Vin
template<typename MatrixType, typename VectorType, void prod_inPlace (const MatrixType &M, VectorType &Vinout)>
void PotShPlus_algorithm (size_t dim_left, size_t dimH, size_t dim_right, 
                          const MatrixType &H, const VectorType &Vin, VectorType &Vout)
{
	if (dim_right==1 and dim_left>1)
	{
		Vout.resize(dim_left*dimH);
		#pragma omp parallel for
		for (size_t i=0; i<dim_left*dimH; i+=dimH)
		{
			VectorType Vtmp(dimH);
			get_segment(i,dimH, Vin,Vtmp);
			prod_inPlace(H,Vtmp);
			set_segment(i,dimH, Vtmp,Vout);
		}
	}
	else if (dim_left==1 and dim_right==dimH) // works for dim_left>1, but is slower
	{
		Vout.resize(Vin.rows());
		Vout = Vin;
		perfectShuffle(dim_left*dimH, Vout);
		#pragma omp parallel for
		for (size_t i=0; i<dim_left*dim_right*dimH; i+=dimH)
		{
			VectorType Vtmp(dimH);
			get_segment(i,dimH, Vout,Vtmp);
			prod_inPlace(H,Vtmp);
			set_segment(i,dimH, Vtmp,Vout);
		}
		perfectShuffle(dim_right,Vout);
	}
	else
	{
		Vout = Vin;
		size_t base = 0;
		size_t jump = dimH*dim_right;
		
		for (size_t block=0; block<dim_left; ++block)
		{
			#pragma omp parallel for
			for (size_t offset=0; offset<dim_right; ++offset)
			{
				size_t index = base + offset;
				VectorType Vtmp(dimH);
				for (size_t h=0; h<dimH; ++h)
				{
					Vtmp(h) = Vout(index);
					index += dim_right;
				}
				prod_inPlace(H,Vtmp);
				index = base + offset;
				for (size_t h=0; h<dimH; ++h)
				{
					Vout(index) = Vtmp(h);
					index += dim_right;
				}
			}
			base += jump;
		}
	}
}

// Vinout *= I(dim_left,dim_left) ⊗ H(dimH,dimH) ⊗ I(dim_right,dim_right)
template<typename MatrixType, typename VectorType, void prod_inPlace (const MatrixType &M, VectorType &Vinout)>
void PotShPlus_algorithm (size_t dim_left, size_t dimH, size_t dim_right, 
                          const MatrixType &H, VectorType &Vinout)
{
	if (dim_right==1)
	{
		#pragma omp parallel for
		for (size_t i=0; i<dim_left*dimH; i+=dimH)
		{
			VectorType Vtmp(dimH);
			get_segment(i,dimH, Vinout,Vtmp);
			prod_inPlace(H,Vtmp);
			set_segment(i,dimH, Vtmp,Vinout);
		}
	}
	else if (dim_left==1 and dim_right==dimH) // works for dim_left>1, but is slower
	{
		perfectShuffle(dim_left*dimH, Vinout);
		#pragma omp parallel for
		for (size_t i=0; i<dim_left*dim_right*dimH; i+=dimH)
		{
			VectorType Vtmp(dimH);
			get_segment(i,dimH, Vinout,Vtmp);
			prod_inPlace(H,Vtmp);
			set_segment(i,dimH, Vtmp,Vinout);
		}
		perfectShuffle(dim_right,Vinout);
	}
	else
	{
		size_t base = 0;
		size_t jump = dimH*dim_right;
		
		for (size_t block=0; block<dim_left; ++block)
		{
			#pragma omp parallel for
			for (size_t offset=0; offset<dim_right; ++offset)
			{
				size_t index = base + offset;
				VectorType Vtmp(dimH);
				for (size_t h=0; h<dimH; ++h)
				{
					Vtmp(h) = Vinout(index);
					index += dim_right;
				}
				prod_inPlace(H,Vtmp);
				index = base + offset;
				for (size_t h=0; h<dimH; ++h)
				{
					Vinout(index) = Vtmp(h);
					index += dim_right;
				}
			}
			base += jump;
		}
	}
}

//inline double trace (const SparseMatrixXd &M)
//{
//	double out = 0.;
//	for (size_t k=0; k<M.outerSize(); ++k)
//	out += M.coeff(k,k);
//	return out;
//}

#endif
