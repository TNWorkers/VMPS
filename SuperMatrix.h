#ifndef SUPERMATRIX
#define SUPERMATRIX

#include "DmrgTypedefs.h"
#include "MemCalc.h"
#include "boost/multi_array.hpp"

/**Auxiliary matrix of matrices to create an Mpo and MpoQ.
\describe_D*/
template<int D, typename Scalar=double>
class SuperMatrix
{
public:
	
	Eigen::Matrix<Scalar,D,D> &operator() (size_t i, size_t j)       {return data[i][j];} // write
	Eigen::Matrix<Scalar,D,D>  operator() (size_t i, size_t j) const {return data[i][j];} // read
	
	/**Resizes to a row vector (1,Daux) for the first site.*/
	void setRowVector (size_t Daux)
	{
		N_rows = 1;
		N_cols = Daux;
		data.resize(boost::extents[N_rows][N_cols]);
	}
	
	/**Resizes to a column vector (Daux,1) for the last site.*/
	void setColVector (size_t Daux)
	{
		N_rows = Daux;
		N_cols = 1;
		data.resize(boost::extents[N_rows][N_cols]);
	}
	
	/**Resizes to a matrix (Daux,Daux) for all sites save the first and the last.*/
	void setMatrix (size_t Daux)
	{
		N_rows = Daux;
		N_cols = Daux;
		data.resize(boost::extents[N_rows][N_cols]);
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
	
	double memory (MEMUNIT memunit=GB) const
	{
		return N_rows * N_cols * calc_memory<double>(D*D, memunit);
	}
	
private:
	
	size_t N_rows;
	size_t N_cols;
	
	boost::multi_array<Eigen::Matrix<Scalar,D,D>,2> data;
};

template<int D, typename Scalar>
SuperMatrix<D,Scalar> tensor_product (const SuperMatrix<D,Scalar> &M1, const SuperMatrix<D,Scalar> &M2)
{
	SuperMatrix<D,Scalar> Mout;
	
	if (M1.rows() == 1)
	{
		Mout.setRowVector(M1.auxdim()*M2.auxdim());
	}
	else if (M1.cols() == 1)
	{
		Mout.setColVector(M1.auxdim()*M2.auxdim());
	}
	else
	{
		Mout.setMatrix(M1.auxdim()*M2.auxdim());
	}
	
	for (size_t r1=0; r1<M1.rows(); ++r1)
	for (size_t c1=0; c1<M1.cols(); ++c1)
	for (size_t r2=0; r2<M2.rows(); ++r2)
	for (size_t c2=0; c2<M2.cols(); ++c2)
	{
		Mout(r1*M2.rows()+r2, c1*M2.cols()+c2) = M1(r1,c1)*M2(r2,c2);
	}
	
	return Mout;
}

template<int D, typename Scalar>
ostream &operator<< (ostream& os, const SuperMatrix<D,Scalar> &M)
{
	MatrixXd Mout(M.rows()*D, M.cols()*D);
	
	for (int i=0; i<M.rows(); ++i)
	for (int j=0; j<M.cols(); ++j)
	{
		Mout.block(i*D,j*D, D,D) = M(i,j);
	}
	os << Mout;
	return os;
}

#endif
