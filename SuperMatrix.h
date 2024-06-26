#ifndef SUPERMATRIX
#define SUPERMATRIX

/// \cond
#include <set>
#include <boost/multi_array.hpp>
/// \endcond

#include "MemCalc.h" // from TOOLS
//include "DmrgTypedefs.h"
//include "DmrgHamiltonianTerms.h"
#include "tensors/SiteOperator.h"


/**Auxiliary matrix of matrices to create an Mpo and MpoQ.*/
template<typename Symmetry, typename Scalar=double>
class SuperMatrix
{
typedef SparseMatrix<double,ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> MatrixType;
typedef SiteOperator<Symmetry,Scalar> OperatorType;
public:
	
	OperatorType &operator() (size_t i, size_t j)       {return data[i][j];} // write
	OperatorType  operator() (size_t i, size_t j) const {return data[i][j];} // read
	
	/**Resizes to a row vector (1,Daux) for the first site.*/
	void set (size_t Daux1, size_t Daux2, size_t D)
	{
		N_rows = Daux1;
		N_cols = Daux2;
		data.resize(boost::extents[Daux1][Daux2]);
		innerResize(D);
		setZero();
	}
	
	/**Resizes to a row vector (1,Daux) for the first site.*/
	void setRowVector (size_t Daux, size_t D)
	{
		set(1,Daux,D);
	}
	
	/**Resizes to a column vector (Daux,1) for the last site.*/
	void setColVector (size_t Daux, size_t D)
	{
		set(Daux,1,D);
	}
	
	/**Resizes to a matrix (Daux,Daux) for all sites save the first and the last.*/
	void setMatrix (size_t Daux, size_t D)
	{
		set(Daux,Daux,D);
	}
	
	/**Returns the i-th row.*/
	SuperMatrix<Symmetry,Scalar> row (size_t i)
	{
		SuperMatrix<Symmetry,Scalar> Mout;
		Mout.setRowVector(N_cols,D());        // instead of auxdim
		for (size_t j=0; j<N_cols; ++j)        //   Bug? N_rows
		{
			Mout(0,j) = data[i][j];
		}
		return Mout;
	}
	
	/**Returns the i-th column.*/
	SuperMatrix<Symmetry,Scalar> col (size_t i)
	{
		SuperMatrix<Symmetry,Scalar> Mout;
		Mout.setColVector(N_rows,D());          // instead of auxdim
		for (size_t j=0; j<N_rows; ++j)         //  Bug? N_cols
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
			data[i][j].data.setZero();
		}
	}
	
	inline size_t rows() const {return N_rows;}
	inline size_t cols() const {return N_cols;}
	
	/**\describe_Daux*/
	inline size_t auxdim() const
	{
        assert(N_rows == N_cols and "auxdim called although SuperMatrix is not quadratic");
		return std::max(N_rows,N_cols);
	}
	
	/**\describe_memory*/
	double memory (MEMUNIT memunit=GB) const
	{
		double out = 0.;
		
		for (size_t i=0; i<N_rows; ++i)
		for (size_t j=0; j<N_cols; ++j)
		{
			return out += calc_memory<Scalar>(data[i][j].data, memunit);
		}
		return out;
	}
	
	/**\describe_D*/
	size_t D() const
	{
		size_t Dres = data[0][0].data.rows();
		for (size_t i=0; i<N_rows; ++i)
		for (size_t j=0; j<N_cols; ++j)
		{
			assert(data[i][j].data.rows() == Dres);
			assert(data[i][j].data.cols() == Dres);
		}
		return Dres;
	}
	
	std::vector<typename Symmetry::qType> calc_qOp() const
	{
		std::set<typename Symmetry::qType> qOps;
		for (size_t i=0; i<N_rows; ++i)
		for (size_t j=0; j<N_cols; ++j)
		{
			qOps.insert(data[i][j].Q);
		}
		
		std::vector<typename Symmetry::qType> out(qOps.size());
		copy(qOps.begin(), qOps.end(), out.begin());
		
		return out;
	}
	
private:
	
	size_t N_rows;
	size_t N_cols;
	
	boost::multi_array<OperatorType,2> data;
	
	void innerResize (size_t D)
	{
		for (size_t i=0; i<N_rows; ++i)
		for (size_t j=0; j<N_cols; ++j)
		{
			data[i][j].data.resize(D,D);
		}
	}
};

template<typename Symmetry, typename Scalar>
SuperMatrix<Symmetry,Scalar> tensor_product (const SuperMatrix<Symmetry,Scalar> &M1, const SuperMatrix<Symmetry,Scalar> &M2)
{
	assert(M1.D() == M2.D());
	
	SuperMatrix<Symmetry,Scalar> Mout;
	
	if (M1.rows() == 1)
	{
		Mout.setRowVector(M1.cols()*M2.cols(), M1.D());             // instead of auxdim
	}
	else if (M1.cols() == 1)
	{
		Mout.setColVector(M1.rows()*M2.rows(), M1.D());             // instead of auxdim
	}
	else
	{
		Mout.set(M1.rows()*M2.rows(), M1.cols()*M2.cols(), M1.D()); // instead of auxdim
	}
	
	for (size_t r1=0; r1<M1.rows(); ++r1)
	for (size_t c1=0; c1<M1.cols(); ++c1)
	for (size_t r2=0; r2<M2.rows(); ++r2)
	for (size_t c2=0; c2<M2.cols(); ++c2)
	{
		Mout(r1*M2.rows()+r2, c1*M2.cols()+c2).data = M1(r1,c1).data * M2(r2,c2).data;
		auto Qsum = Symmetry::reduceSilent(M1(r1,c1).Q, M2(r2,c2).Q);
		// should be only one term, non-Abelian symmetries have a different code
		assert(Qsum.size() == 1 and "tensor_product of SuperMatrices called with wrong symmetry!");
		Mout(r1*M2.rows()+r2, c1*M2.cols()+c2).Q = Qsum[0];
	}
	
	return Mout;
}

template<typename Symmetry, typename Scalar>
SuperMatrix<Symmetry,Scalar> directSum (const SuperMatrix<Symmetry,Scalar> &M1, const SuperMatrix<Symmetry,Scalar> &M2)
{
	SuperMatrix<Symmetry,Scalar> Mout;
	size_t R;
	size_t C;
	
	if (M1.rows()==1 and M2.rows()==1)
	{
		Mout.setRowVector(M1.cols()+M2.cols(), M1.D());         // instead of auxdim
		R = 0;
		C = M1.cols();
	}
	else if (M1.cols()==1 and M2.cols()==1)
	{
		Mout.setColVector(M1.rows()+M2.rows(), M1.D());     // instead of auxdim
		R = M1.rows();
		C = 0;
	}
	else
	{
        Mout.set(M1.rows()+M2.rows(), M1.cols()+M2.cols(), M1.D());            // instead of auxdim
		R = M1.rows();
		C = M1.cols();
	}
	
	for (size_t r1=0; r1<M1.rows(); ++r1)
	for (size_t c1=0; c1<M1.cols(); ++c1)
	{
		Mout(r1,c1) = M1(r1,c1);
	}
	
	for (size_t r2=0; r2<M2.rows(); ++r2)
	for (size_t c2=0; c2<M2.cols(); ++c2)
	{
		Mout(R+r2,C+c2) = M2(r2,c2);
	}
	
	return Mout;
}

template<typename Symmetry, typename Scalar>
std::ostream &operator<< (std::ostream& os, const SuperMatrix<Symmetry,Scalar> &M)
{
	os << std::showpos << std::setprecision(1) << std::fixed;
	for (int i=0; i<M.rows(); ++i)
	{
		for (int n=0; n<M.D(); ++n)
		{
			for (int j=0; j<M.cols(); ++j)
			{
				for (int m=0; m<M.D(); ++m)
				{
					
					os << Matrix<Scalar,Dynamic,Dynamic>(M(i,j).data)(n,m);
				}
				os << " ";
			}
			if (n!=M.D()-1) {os << std::endl;}
		}
		if (i!=M.rows()-1) {os << std::endl << std::endl;}
	}
	os << noshowpos;
	return os;
}

/*template<typename Symmetry, typename Scalar>
SuperMatrix<Symmetry, Scalar> Generator (const HamiltonianTerms<Symmetry, Scalar> &Terms)
{
	typedef SiteOperator<Symmetry,Scalar> OperatorType;
	size_t Daux = 2 + Terms.tight.size() + 2*Terms.nextn.size();
	
	std::vector<OperatorType> col;
	std::vector<OperatorType> row;
	size_t locdim;
	
	if (Terms.local.size()>0)
	{
		locdim = std::get<1>(Terms.local[0]).data.rows();
	}
	else if (Terms.tight.size()>0)
	{
		locdim = std::get<1>(Terms.tight[0]).data.rows();
	}
	else
	{
		locdim = std::get<1>(Terms.nextn[0]).data.rows();
	}
	
	OperatorType Id(Matrix<Scalar,Dynamic,Dynamic>::Identity(locdim,locdim).sparseView(),Symmetry::qvacuum());
	OperatorType Zero(SparseMatrix<Scalar>(locdim,locdim),Symmetry::qvacuum());
	
	// last row (except corner element)
	for (size_t i=0; i<Terms.nextn.size(); ++i)
	{
		row.push_back(Zero);
	}
	for (int i=0; i<Terms.tight.size(); ++i)
	{
		row.push_back(std::get<0>(Terms.tight[i]) * std::get<1>(Terms.tight[i]));
	}
	for (int i=0; i<Terms.nextn.size(); ++i)
	{
		row.push_back(std::get<0>(Terms.nextn[i]) * std::get<1>(Terms.nextn[i]));
	}
	row.push_back(Id);
	
	// first col (except corner element)
	col.push_back(Id);
	for (int i=0; i<Terms.nextn.size(); ++i)
	{
		col.push_back(std::get<2>(Terms.nextn[i]));
	}
	for (int i=0; i<Terms.tight.size(); ++i)
	{
		col.push_back(std::get<2>(Terms.tight[i]));
	}
	for (size_t i=0; i<Terms.nextn.size(); ++i)
	{
		col.push_back(Zero);
	}
	
	SuperMatrix<Symmetry,Scalar> Gout;
	Gout.setMatrix(Daux,locdim);
	Gout.setZero();
	
	for (size_t i=0; i<Daux-1; ++i)
	{
		Gout(i,0)        = col[i];
		Gout(Daux-1,i+1) = row[i];
	}
	
	// corner element : local interaction
	for (int i=0; i<Terms.local.size(); ++i)
	{
		Gout(Daux-1,0) += std::get<0>(Terms.local[i]) * std::get<1>(Terms.local[i]);
	}
	
	// nearest-neighbour transfer
	if (Terms.nextn.size() != 0)
	{
		for (size_t i=0; i<Terms.nextn.size(); ++i)
		{
			Gout(Daux-1-Terms.nextn.size()+i,1+i) = std::get<3>(Terms.nextn[i]);
		}
	}
	
	return Gout;
}*/

#endif
