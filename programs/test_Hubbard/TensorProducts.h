#ifndef TENSORPRODUCTS
#define TENSORPRODUCTS

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/KroneckerProduct>
using namespace Eigen;

#include "HilbertTypedefs.h"

// Mout = M ⊗ Id(dimI,dimI)
template<typename Scalar>
void MkronI (const SparseMatrix<Scalar> &M, size_t dimI, SparseMatrix<Scalar> &Mout, size_t row_offset=0, size_t col_offset=0)
{
	Mout.resize(M.rows()*dimI+row_offset, M.cols()*dimI+col_offset);
	Mout.reserve(M.nonZeros()*dimI);
	
	vector<triplet> tripletList;
	tripletList.reserve(M.nonZeros()*dimI);
	
	for (size_t id=0; id<dimI; ++id)
	for (size_t k=0; k<M.outerSize(); ++k)
	for (typename SparseMatrix<Scalar>::InnerIterator It(M,k); It; ++It)
	{
		size_t i = It.row()*dimI+id;
		size_t j = It.col()*dimI+id;
		tripletList.push_back(Triplet<Scalar>(i+row_offset, j+col_offset, It.value()));
	}
	
	Mout.setFromTriplets(tripletList.begin(), tripletList.end());
}

// Mout = Id(dimI,dimI) ⊗ M
template<typename Scalar>
void IkronM (size_t dimI, const SparseMatrix<Scalar> &M, SparseMatrix<Scalar> &Mout, size_t row_offset=0, size_t col_offset=0)
{
	size_t Mrows = M.rows();
	size_t Mcols = M.cols();
	Mout.resize(Mrows*dimI+row_offset, Mcols*dimI+col_offset);
	Mout.reserve(M.nonZeros()*dimI);
	
	vector<triplet> tripletList;
	tripletList.reserve(M.nonZeros()*dimI);
	
	for (size_t id=0; id<dimI; ++id)
	for (size_t k=0; k<M.outerSize(); ++k)
	for (typename SparseMatrix<Scalar>::InnerIterator It(M,k); It; ++It)
	{
		size_t i = id*Mrows + It.row();
		size_t j = id*Mcols + It.col();
		tripletList.push_back(Triplet<Scalar>(i+row_offset, j+col_offset, It.value()));
	}
	
	Mout.setFromTriplets(tripletList.begin(), tripletList.end());
}

// Mout = Id(dimI,dimI) ⊗ diag(V)
template<typename Scalar>
void IkronM (size_t dimI, const Matrix<Scalar,Dynamic,1> &V, SparseMatrix<Scalar> &Mout)
{
	size_t Vrows = V.rows();
	Mout.resize(Vrows*dimI, Vrows*dimI);
	
	for (size_t id=0; id<dimI; ++id)
	for (size_t it=0; it<Vrows; ++it)
	{
		size_t i = id*Vrows + it;
		Mout.insert(i,i) = V(it);
	}
}

// Mout = M1 ⊕ M2
template<typename Scalar>
void direct_sum (const SparseMatrix<Scalar> &M1, const SparseMatrix<Scalar> &M2, SparseMatrix<Scalar> &Mout)
{
	MkronI(M1,M2.rows(),Mout);
	SparseMatrix<Scalar> Mtemp;
	IkronM(M1.rows(),M2,Mtemp);
	Mout += Mtemp;
}

// tensor product

// Mout = A ⊗ B
template<typename Scalar>
void tensor_product (const SparseMatrix<Scalar> &A, const SparseMatrix<Scalar> &B, SparseMatrix<Scalar> &Mout)
{
	size_t Ar=A.rows(), Ac=A.cols(), Br=B.rows(), Bc=B.cols();
	Mout.resize(Ar*Br, Ac*Bc);
	Mout.reserve(A.nonZeros()*B.nonZeros());
	
	vector<triplet> tripletList;
	tripletList.reserve(A.nonZeros()*B.nonZeros());
	
	for (int kA=0; kA<A.outerSize(); ++kA)
	for (int kB=0; kB<B.outerSize(); ++kB)
	for (typename SparseMatrix<Scalar>::InnerIterator itA(A,kA); itA; ++itA)
	for (typename SparseMatrix<Scalar>::InnerIterator itB(B,kB); itB; ++itB)
	{
		size_t iA=itA.row(), jA=itA.col(), iB=itB.row(), jB=itB.col();
		size_t i=iA*Br+iB;
		size_t j=jA*Bc+jB;
		tripletList.push_back(Triplet<Scalar>(i,j, itA.value()*itB.value()));
	}
	
	Mout.setFromTriplets(tripletList.begin(), tripletList.end());
}

// return A ⊗ B
template<typename Scalar>
SparseMatrix<Scalar> tensor_product (const SparseMatrix<Scalar> &A, const SparseMatrix<Scalar> &B)
{
	SparseMatrix<Scalar> Mout;
	tensor_product(A,B,Mout);
	return Mout;
}

template<typename Scalar>
void write_to_upperLeftCorner (const SparseMatrix<Scalar> &M, size_t rows, size_t cols, SparseMatrix<Scalar> &Mout)
{
	Mout.resize(rows,cols);
	Mout.reserve(M.nonZeros());
	
	vector<triplet> tripletList;
	tripletList.reserve(M.nonZeros());
	
	for (size_t k=0; k<M.outerSize(); ++k)
	for (typename SparseMatrix<Scalar>::InnerIterator It(M,k); It; ++It)
	{
		tripletList.push_back(Triplet<Scalar>(It.row(), It.col(), It.value()));
	}
	
	Mout.setFromTriplets(tripletList.begin(), tripletList.end());
}

#endif
