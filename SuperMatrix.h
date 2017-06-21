#ifndef SUPERMATRIX
#define SUPERMATRIX

#include "DmrgTypedefs.h"
#include "MemCalc.h"
#include "boost/multi_array.hpp"

/**Auxiliary matrix of matrices to create an Mpo and MpoQ.*/
template<typename Symmetry, typename Scalar=double>
class SuperMatrix
{
typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;
typedef SiteOperator<Symmetry,MatrixType> OperatorType;
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
		Mout.setRowVector(auxdim(),D());
		for (size_t j=0; j<N_rows; ++j)
		{
			Mout(0,j) = data[i][j];
		}
		return Mout;
	}
	
	/**Returns the i-th column.*/
	SuperMatrix<Symmetry,Scalar> col (size_t i)
	{
		SuperMatrix<Symmetry,Scalar> Mout;
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
			data[i][j].data.setZero();
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

	vector<typename Symmetry::qType> get_qOp() const
	{
		vector<typename Symmetry::qType> out;
		for (size_t i=0; i<N_rows; ++i)
		for (size_t j=0; j<N_cols; ++j)
		{
			if(auto it = std::find(out.begin(),out.end(),data[i][j].Q); it == out.end())
			{
				out.push_back(data[i][j].Q);
			}
		}
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
		Mout(r1*M2.rows()+r2, c1*M2.cols()+c2).data = M1(r1,c1).data * M2(r2,c2).data;
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
		Mout.setRowVector(M1.auxdim()+M2.auxdim(), M1.D());
		R = 0;
		C = M1.cols();
	}
	else if (M1.cols()==1 and M2.cols()==1)
	{
		Mout.setColVector(M1.auxdim()+M2.auxdim(), M1.D());
		R = M1.rows();
		C = 0;
	}
	else
	{
		Mout.setMatrix(M1.auxdim()+M2.auxdim(), M1.D());
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
ostream &operator<< (ostream& os, const SuperMatrix<Symmetry,Scalar> &M)
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
					os << M(i,j).data(n,m);
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

template<typename Symmetry, typename Scalar>
SuperMatrix<Symmetry, Scalar> Generator (const HamiltonianTerms<Symmetry, Scalar> &Terms)
{
	typedef SiteOperator<Symmetry,SparseMatrix<Scalar> > OperatorType;
	size_t Daux = 2 + Terms.tight.size() + 2*Terms.nextn.size();
	
	vector<OperatorType> col;
	vector<OperatorType> row;
	size_t locdim;
	if (Terms.tight.size()>0)
	{
		locdim = get<1>(Terms.tight[0]).data.rows();
	}
	else if (Terms.nextn.size()>0)
	{
		locdim = get<1>(Terms.nextn[0]).data.rows();
	}
	else
	{
		locdim = get<1>(Terms.local[0]).data.rows();
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
		row.push_back(OperatorType(get<0>(Terms.tight[i]) * get<1>(Terms.tight[i]).data, get<1>(Terms.tight[i]).Q));
	}
	for (int i=0; i<Terms.nextn.size(); ++i)
	{
		row.push_back(OperatorType(get<0>(Terms.nextn[i]) * get<1>(Terms.nextn[i]).data, get<1>(Terms.nextn[i]).Q));
	}
	row.push_back(Id);
	
	// first col (except corner element)
	col.push_back(Id);
	for (int i=0; i<Terms.nextn.size(); ++i)
	{
		col.push_back(get<2>(Terms.nextn[i]));
	}
	for (int i=0; i<Terms.tight.size(); ++i)
	{
		col.push_back(get<2>(Terms.tight[i]));
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
		Gout(i,0).data     = col[i].data;
		Gout(i,0).Q        = col[i].Q;
		Gout(Daux-1,i+1).data = row[i].data;
		Gout(Daux-1,i+1).Q = row[i].Q;
	}
	
	// corner element : local interaction
	for (int i=0; i<Terms.local.size(); ++i)
	{
		Gout(Daux-1,0).data += get<0>(Terms.local[i]) * get<1>(Terms.local[i]).data;
		Gout(Daux-1,0).Q = Gout(Daux-1,0).Q + get<1>(Terms.local[i]).Q; //TODO: This line is only valid for U1. Change it.

	}
	
	// nearest-neighbour transfer
	if (Terms.nextn.size() != 0)
	{
		for (size_t i=0; i<Terms.nextn.size(); ++i)
		{
			Gout(Daux-1-Terms.nextn.size()+i,1+i).data = get<3>(Terms.nextn[i]).data;
			Gout(Daux-1-Terms.nextn.size()+i,1+i).Q = get<3>(Terms.nextn[i]).Q;
		}
	}
	
	return Gout;
}

#endif
