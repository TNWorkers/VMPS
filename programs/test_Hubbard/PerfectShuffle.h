#ifndef PERFECTSHUFFLE
#define PERFECTSHUFFLE

#include <array>
#include <map>

#include "HilbertTypedefs.h"
#include "Chunkomatic.h"

void extEuclid (SparseMatrixXd::Index a, SparseMatrixXd::Index b, SparseMatrixXd::Index &y, SparseMatrixXd::Index &d)
/* calculates a * *x + b * *y = gcd(a, b) = *d */
/* Author: Pate Williams (c) 1997 */
//--------------------
// note: throws away x
// stolen and adapted from: http://www.di-mgt.com.au/euclidean.html
{ 
	SparseMatrixXd::Index q, r;
	SparseMatrixXd::Index y1=1, y2=0;
//	if (b == 0)
//	{
//		*d=a, *y=0;
//		return;
//	}
//	y2=0, y1=1;
	while (b>0)
	{
		q=a/b, r=a-q*b;
		y=y2-q*y1;
		a=b, b=r;
		y2=y1, y1=y;
	}
	d=a, y=y2;
}

//// Handbook of Applied Cryptography p. 608
//// isn't fully correct
//void binaryExtEuclid (SparseMatrixXd::Index a, SparseMatrixXd::Index b, SparseMatrixXd::Index *y, SparseMatrixXd::Index *d)
//{
//	SparseMatrixXd::Index g = 1;
//	while (GSL_IS_EVEN(a) and GSL_IS_EVEN(b))
//	{
//		a /= 2;
//		b /= 2;
//		g *= 2;
//	}
//	SparseMatrixXd::Index u=a, v=b;
//	SparseMatrixXd::Index A=1, B=0, C=0, D=1;
//	while (u!=0)
//	{
//		while (GSL_IS_EVEN(u))
//		{
//			u /= 2;
//			if (A%2==0 and B%2==0)
//			{
//				A /= 2; B /= 2;
//			}
//			else
//			{
//				A = (A+b)/2; B = (B-a)/2;
//			}
//		
//		}
//		while (GSL_IS_EVEN(v))
//		{
//			v /= 2;
//			if (C%2==0 and D%2==0)
//			{
//				C /= 2; D /= 2;
//			}
//			else
//			{
//				C = (C+b)/2; D = (D-a)/2;
//			}
//		}
//	
//		if (u>=v)
//		{
//			u -= v;
//			A -= C; B -= D;
//		}
//		else
//		{
//			v -= u;
//			C -= A; D -= B;
//		}
////		cout << A << "\t" << B << "\t" << C << "\t" << D << endl;
//	}

//	*y = D;
//	*d = v*g;
//}

template<typename Scalar>
inline void rowswap (Matrix<Scalar,Dynamic,1> &Vinout, SparseMatrixXd::Index i, SparseMatrixXd::Index j)
{
	Scalar tmp = Vinout.coeff(j);
	Vinout.coeffRef(j) = Vinout.coeff(i);
	Vinout.coeffRef(i) = tmp;
}

//template<typename VectorType>
//void invol (size_t r, VectorType &Vinout)
//{
//	size_t Nm1 = Vinout.rows()-1;
//	#pragma omp parallel for
//	for (SparseMatrixXd::Index i=1; i<Nm1; ++i)
//	{
//		SparseMatrixXd::Index y, g;
//		extEuclid(Nm1,i, y,g);
////		binaryExtEuclid(Nm1,i, &y,&g);
//		if (y<0) {y += Nm1/g;}
//		SparseMatrixXd::Index j = g*( (r*y) % (Nm1/g) );
//		if (i<j) {rowswap(Vinout,i,j);}
//	}
//}

void calc_invol (size_t r, SparseMatrixXd::Index N, SparseVector<SparseMatrixXd::Index> &Inv)
//void calc_invol (SparseMatrixXd::Index r, SparseMatrixXd::Index N, Matrix<SparseMatrixXd::Index,Dynamic,1> &I) // using Transpositions
{
	Inv.resize(N);
	Inv.reserve(N/2);
	size_t Nm1 = N-1;
//	I(0)=0; I(Nm1)=Nm1; // using Transpositions
	for (SparseMatrixXd::Index i=1; i<Nm1; ++i)
	{
		SparseMatrixXd::Index y,g;
		extEuclid(Nm1,i, y,g);
		if (y<0) {y += Nm1/g;}
		SparseMatrixXd::Index j = g*( (r*y) % (Nm1/g) );
		if (i<j) {Inv.insert(i) = j;}
//		I(i) = (i<j) ? j : i; // using Transpositions
	}
}

template<typename VectorType>
void involSq (SparseMatrixXd::Index k, VectorType &Vinout) // for the case N=k*k
{
	SparseMatrixXd::Index N = Vinout.rows();
	#pragma omp parallel for
	for (SparseMatrixXd::Index i=1; i<N-1; ++i)
	{
		SparseMatrixXd::Index run = i;
		SparseMatrixXd::Index ultimate = run%k;
		run /= k;
		SparseMatrixXd::Index penultimate = run%k;
		SparseMatrixXd::Index j = i + ultimate*(k-1) + penultimate*(1-k);
		
		if (i<j) {rowswap(Vinout,i,j);}
	}
}

typedef std::array<size_t,2> InvolKey;
//typedef array<Transpositions<Dynamic,Dynamic,SparseMatrixXd::Index>,2> InvolPair; // using Transpositions
typedef std::array<SparseVector<SparseMatrixXd::Index>,2> InvolPair;
map<InvolKey,std::shared_ptr<InvolPair> > GlobInvol;

void store_perfectShuffle (size_t k, size_t N)
{
	InvolKey Key = {k,N};

	// using Transpositions
//	std::shared_ptr<InvolPair> Val = std::make_shared<InvolPair>();
//	Matrix<SparseMatrixXd::Index,Dynamic,1> I;
//	calc_invol(1,N,I);
//	(*Val)[0] = Transpositions<Dynamic,Dynamic,SparseMatrixXd::Index>(I);
//	calc_invol(k,N,I);
//	(*Val)[1] = Transpositions<Dynamic,Dynamic,SparseMatrixXd::Index>(I);
//	GlobInvol.insert({Key,Val});
	
	std::shared_ptr<InvolPair> Val = std::make_shared<InvolPair>();
	GlobInvol.insert({Key,Val});
	calc_invol(1,N, (*GlobInvol[Key])[0]);
	calc_invol(k,N, (*GlobInvol[Key])[1]);
}

template<typename VectorType>
void apply_perfectShuffle (size_t k, VectorType &Vinout)
{
	InvolKey Key = {k,static_cast<size_t>(Vinout.rows())};
	
	// using Transpositions
//	Vinout = (*GlobInvol[Key])[0] * Vinout;
//	Vinout = (*GlobInvol[Key])[1] * Vinout;
	
	for (SparseVector<SparseMatrixXd::Index>::InnerIterator it((*GlobInvol[Key])[0]); it; ++it)
	{
		rowswap(Vinout, it.row(), it.value());
	}
	for (SparseVector<SparseMatrixXd::Index>::InnerIterator it((*GlobInvol[Key])[1]); it; ++it)
	{
		rowswap(Vinout, it.row(), it.value());
	}
	
//	incorrect, cannot iterate SparseVector segments
//	int N_chunks;
//	#if defined(_OPENMP)
//	N_chunks = omp_get_max_threads();
//	#else
//	N_chunks = 1;
//	#endif
//	Chunkomatic<SparseMatrixXd::Index> LeChuck((*GlobInvol[Key])[0].nonZeros(), N_chunks);
//	Eigen::Matrix<SparseMatrixXd::Index,Dynamic,2> T = LeChuck.get_limits();
//	cout << "Nnz=" << (*GlobInvol[Key])[0].nonZeros() << endl;
//	T.col(1) -= T.col(0);
//	cout << T << endl;
////	for (int i=0; i<T.rows(); ++i)
////	{
////		cout << (*GlobInvol[Key])[0].block(T(i,0),0,T(i,1),1) << endl;
////	}
////	#pragma omp parallel for
//	for (int i=0; i<T.rows(); ++i)
//	{
//		for (SparseVector<SparseMatrixXd::Index>::InnerIterator it((*GlobInvol[Key])[0].segment(T(i,0),T(i,1))); it; ++it)
//		{
//			cout << T(i,0) << "\t" << T(i,1) << endl;
//			cout << it.row() << "\t" << it.value() << endl;
//			rowswap(Vinout, it.row(), it.value());
//		}
//	}
////	#pragma omp parallel for
//	for (int i=0; i<T.rows(); ++i)
//	{
//		for (SparseVector<SparseMatrixXd::Index>::InnerIterator it((*GlobInvol[Key])[1].segment(T(i,0),T(i,1))); it; ++it)
//		{
//			rowswap(Vinout, it.row(), it.value());
//		}
//	}

//	slow, incorrect
//	int Nnz = (*GlobInvol[Key])[0].nonZeros();
//	Map<VectorXi> values((*GlobInvol[Key])[0].innerIndexPtr(),Nnz);
//	Map<VectorXi> indices((*GlobInvol[Key])[0].valuePtr(),Nnz);
//	#pragma omp parallel for
//	for (int i=0; i<Nnz; ++i)
//	{
//		rowswap(Vinout, indices.coeff(i), values.coeff(i));
//	}
//	values = Map<VectorXi>((*GlobInvol[Key])[1].innerIndexPtr(),Nnz);
//	indices = Map<VectorXi>((*GlobInvol[Key])[1].valuePtr(),Nnz);
//	#pragma omp parallel for
//	for (int i=0; i<Nnz; ++i)
//	{
//		rowswap(Vinout, indices.coeff(i), values.coeff(i));
//	}

//	slow!
//	SparseMatrixXd::Index Nnz0 = (*GlobInvol[Key])[0].nonZeros();
//	SparseMatrixXd::Index Nnz1 = (*GlobInvol[Key])[1].nonZeros();
//	#pragma omp parallel for
//	for (SparseMatrixXd::Index i=0; i<Nnz0; ++i)
//	{
//		SparseMatrixXd::Index j = (*GlobInvol[Key])[0].innerIndexPtr()[i];
//		rowswap(Vinout, (*GlobInvol[Key])[0].coeff(j), j);
//	}
//	#pragma omp parallel for
//	for (SparseMatrixXd::Index i=0; i<Nnz1; ++i)
//	{
//		SparseMatrixXd::Index j = (*GlobInvol[Key])[1].innerIndexPtr()[i];
//		rowswap(Vinout, (*GlobInvol[Key])[1].coeff(j), j);
//	}
}

template<typename VectorType>
void perfectShuffle (size_t k, VectorType &Vinout)
{
	size_t N = Vinout.rows();
	if (N==k*k)
	{
		involSq(k, Vinout);
	}
	else
	{
		// on the fly:
//		invol(1, Vinout);
//		invol(k, Vinout);

		// using storage:
		std::array<size_t,2> Key = {k,N};
		if (GlobInvol.find(Key) != GlobInvol.end())
		{
			apply_perfectShuffle(k,Vinout);
		}
		else
		{
			store_perfectShuffle(k,N); 
			apply_perfectShuffle(k,Vinout);
		}
	}
}

#endif
