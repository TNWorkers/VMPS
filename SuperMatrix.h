#ifndef SUPERMATRIX
#define SUPERMATRIX

#include "DmrgTypedefs.h"
#include "MemCalc.h"
#include "boost/multi_array.hpp"

/**Auxiliary matrix of matrices to create an Mpo and MpoQ.*/
template<typename Scalar=double>
class SuperMatrix
{
typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;

public:
	
	MatrixType &operator() (size_t i, size_t j)       {return data[i][j];} // write
	MatrixType  operator() (size_t i, size_t j) const {return data[i][j];} // read
	
//	template<typename OtherScalar>
//	SuperMatrix<Scalar>& operator= (const SuperMatrix<OtherScalar> &M)
//	{
//		size_t Daux = M.auxdim();
//		size_t D = M.D();
//		if (M.rows() == 1)
//		{
//			setRowVector(Daux,D);
//		}
//		else if (M.rows() != 1 and M.cols() == 1)
//		{
//			setColVector(Daux,D);
//		}
//		else
//		{
//			setMatrix(Daux,D);
//		}
//		
//		data = M.data;
//		return *this;
//	}
	
	/**Resizes to a row vector (1,Daux) for the first site.*/
	void setRowVector (size_t Daux, size_t D)
	{
		N_rows = 1;
		N_cols = Daux;
		data.resize(boost::extents[1][Daux]);
		innerResize(D);
	}
	
	/**Resizes to a column vector (Daux,1) for the last site.*/
	void setColVector (size_t Daux, size_t D)
	{
		N_rows = Daux;
		N_cols = 1;
		data.resize(boost::extents[Daux][1]);
		innerResize(D);
	}
	
	/**Resizes to a matrix (Daux,Daux) for all sites save the first and the last.*/
	void setMatrix (size_t Daux, size_t D)
	{
		N_rows = Daux;
		N_cols = Daux;
		data.resize(boost::extents[N_rows][N_cols]);
		innerResize(D);
	}
	
	/**Returns the i-th row.*/
	SuperMatrix<Scalar> row (size_t i)
	{
		SuperMatrix<Scalar> Mout;
		Mout.setRowVector(auxdim(),D());
		for (size_t j=0; j<N_rows; ++j)
		{
			Mout(0,j) = data[i][j];
		}
		return Mout;
	}
	
	/**Returns the i-th column.*/
	SuperMatrix<Scalar> col (size_t i)
	{
		SuperMatrix<Scalar> Mout;
		Mout.setColVector(auxdim(),D());
		for (size_t j=0; j<N_cols; ++j)
		{
			Mout(j,0) = data[j][i];
		}
		return Mout;
	}
	
	/**Sets all submatrices to zero.*/
	void setZero()
	{
		for (size_t i=0; i<N_rows; ++i)
		for (size_t j=0; j<N_cols; ++j)
		{
			data[i][j].setZero();
		}
	}
	
	inline size_t rows() const {return N_rows;}
	inline size_t cols() const {return N_cols;}
	
	/**\describe_Daux*/
	inline size_t auxdim() const
	{
		return max(N_rows,N_cols);
	}
	
	/**\describe_memory*/
	double memory (MEMUNIT memunit=GB) const
	{
		double out = 0.;
		
		for (size_t i=0; i<N_rows; ++i)
		for (size_t j=0; j<N_cols; ++j)
		{
			return out += calc_memory<Scalar>(data[i][j], memunit);
		}
		return out;
	}
	
	/**\describe_D*/
	size_t D() const
	{
		size_t Dres = data[0][0].rows();
		for (size_t i=0; i<N_rows; ++i)
		for (size_t j=0; j<N_cols; ++j)
		{
			assert(data[i][j].rows() == Dres);
			assert(data[i][j].cols() == Dres);
		}
		return Dres;
	}
	
private:
	
	size_t N_rows;
	size_t N_cols;
	
	boost::multi_array<MatrixType,2> data;
	
	void innerResize (size_t D)
	{
		for (size_t i=0; i<N_rows; ++i)
		for (size_t j=0; j<N_cols; ++j)
		{
			data[i][j].resize(D,D);
		}
	}
};

template<typename Scalar>
SuperMatrix<Scalar> tensor_product (const SuperMatrix<Scalar> &M1, const SuperMatrix<Scalar> &M2)
{
	assert(M1.D() == M2.D());
	
	SuperMatrix<Scalar> Mout;
	
	if (M1.rows() == 1)
	{
		Mout.setRowVector(M1.auxdim()*M2.auxdim(), M1.D());
	}
	else if (M1.cols() == 1)
	{
		Mout.setColVector(M1.auxdim()*M2.auxdim(), M1.D());
	}
	else
	{
		Mout.setMatrix(M1.auxdim()*M2.auxdim(), M1.D());
	}
	
	for (size_t r1=0; r1<M1.rows(); ++r1)
	for (size_t c1=0; c1<M1.cols(); ++c1)
	for (size_t r2=0; r2<M2.rows(); ++r2)
	for (size_t c2=0; c2<M2.cols(); ++c2)
	{
		Mout(r1*M2.rows()+r2, c1*M2.cols()+c2) = M1(r1,c1) * M2(r2,c2);
	}
	
	return Mout;
}

template<typename Scalar>
ostream &operator<< (ostream& os, const SuperMatrix<Scalar> &M)
{
	os << showpos;
	for (int i=0; i<M.rows(); ++i)
	{
		for (int n=0; n<M.D(); ++n)
		{
			for (int j=0; j<M.cols(); ++j)
			{
				for (int m=0; m<M.D(); ++m)
				{
					os << M(i,j)(n,m);
				}
				os << "\t";
			}
			if (n!=M.D()-1) {os << endl;}
		}
		if (i!=M.rows()-1) {os << endl;}
	}
	os << noshowpos;
	return os;
}

#endif
