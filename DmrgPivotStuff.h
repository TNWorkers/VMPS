#ifndef VANILLA_DMRGHEFFSTUFF
#define VANILLA_DMRGHEFFSTUFF

#include "DmrgTypedefs.h"

template<size_t D>
struct PivotMatrix
{
	vector<MatrixXd> L;
	vector<MatrixXd> R;
	std::array<std::array<SparseMatrixXd,D>,D> W;
	
	size_t dim;
};

template<size_t D>
struct PivotVector
{
	std::array<MatrixXd,D> A;
	
	PivotVector<D>& operator+= (const PivotVector<D> &Vrhs);
	PivotVector<D>& operator-= (const PivotVector<D> &Vrhs);
	PivotVector<D>& operator*= (const double &alpha);
	PivotVector<D>& operator/= (const double &alpha);
};

//-----------<vector arithmetics>-----------
template<size_t D>
PivotVector<D>& PivotVector<D>::operator+= (const PivotVector<D> &Vrhs)
{
	for (size_t s=0; s<D; ++s)
	{
		A[s] += Vrhs.A[s];
	}
	return *this;
}

template<size_t D>
PivotVector<D>& PivotVector<D>::operator-= (const PivotVector<D> &Vrhs)
{
	for (size_t s=0; s<D; ++s)
	{
		A[s] -= Vrhs.A[s];
	}
	return *this;
}

template<size_t D>
PivotVector<D>& PivotVector<D>::operator*= (const double &alpha)
{
	for (size_t s=0; s<D; ++s)
	{
		A[s] *= alpha;
	}
	return *this;
}

template<size_t D>
PivotVector<D>& PivotVector<D>::operator/= (const double &alpha)
{
	for (size_t s=0; s<D; ++s)
	{
		A[s] /= alpha;
	}
	return *this;
}

template<size_t D>
PivotVector<D> operator* (double const &alpha, PivotVector<D> V)
{
	return V *= alpha;
}

template<size_t D>
PivotVector<D> operator* (PivotVector<D> V, double const &alpha)
{
	return V *= alpha;
}

template<size_t D>
PivotVector<D> operator/ (PivotVector<D> V, const double &alpha)
{
	return V /= alpha;
}

template<size_t D>
PivotVector<D> operator+ (const PivotVector<D> &V1, const PivotVector<D> &V2)
{
	PivotVector<D> Vout = V1;
	Vout += V2;
	return Vout;
}

template<size_t D>
PivotVector<D> operator- (const PivotVector<D> &V1, const PivotVector<D> &V2)
{
	PivotVector<D> Vout = V1;
	Vout -= V2;
	return Vout;
}
//-----------</vector arithmetics>-----------

//-----------<matrix*vector>-----------
template<size_t D>
void HxV (const PivotMatrix<D> &H, const PivotVector<D> &Vin, PivotVector<D> &Vout)
{
	for (size_t s=0; s<D; ++s)
	{
		Vout.A[s].resize(Vin.A[s].rows(), Vin.A[s].cols());
		Vout.A[s].setZero();
	}
	
	#pragma omp parallel for
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (int k=0; k<H.W[s1][s2].outerSize(); ++k)
	for (SparseMatrixXd::InnerIterator iW(H.W[s1][s2],k); iW; ++iW)
	{
		if (H.L[iW.row()].rows() != 0 and 
		    H.R[iW.col()].rows() != 0)
		{
			Vout.A[s1].noalias() += iW.value() * (H.L[iW.row()] * Vin.A[s2] * H.R[iW.col()]);
		}
	}
}

template<size_t D>
void HxV (const PivotMatrix<D> &H, PivotVector<D> &Vinout)
{
	PivotVector<D> Vtmp;
	HxV(H,Vinout, Vtmp);
	Vinout = Vtmp;
}
//-----------</matrix*vector>-----------

//-----------<dot & vector norms>-----------
template<size_t D>
double dot (const PivotVector<D> &V1, const PivotVector<D> &V2)
{
	double res = 0.;
	for (size_t s=0; s<D; ++s)
	for (size_t i=0; i<V2.A[s].cols(); ++i)
	{
		res += V1.A[s].col(i).dot(V2.A[s].col(i));
	}
	return res;
}

template<size_t D>
double squaredNorm (const PivotVector<D> &V)
{
	double res = 0.;
	for (size_t s=0; s<D; ++s)
	{
		res += V.A[s].colwise().squaredNorm().sum();
	}
	return res;
}

template<size_t D>
inline double norm (const PivotVector<D> &V)
{
	return sqrt(squaredNorm(V));
}

template<size_t D>
inline void normalize (PivotVector<D> &V)
{
	V /= norm(V);
}

template<size_t D>
double infNorm (const PivotVector<D> &V1, const PivotVector<D> &V2)
{
	double res = 0.;
	for (size_t s=0; s<D; ++s)
	{
		double tmp = (V1.A[s]-V2.A[s]).template lpNorm<Eigen::Infinity>();
		if (tmp>res) {res = tmp;}
	}
	return res;
}
//-----------</dot & vector norms>-----------

//-----------<miscellaneous>-----------
template<size_t D>
inline size_t dim (const PivotMatrix<D> &H)
{
	return H.dim;
}

template<size_t D>
inline double norm (const PivotMatrix<D> &H)
{
	return H.dim;
}

template<size_t D>
void swap (PivotVector<D> &V1, PivotVector<D> &V2)
{
	for (size_t s=0; s<D; ++s)
	{
		V1.A[s].swap(V2.A[s]);
	}
}

#include "LanczosWrappers.h"
#include "RandomVector.h"

template<size_t D>
struct GaussianRandomVector<PivotVector<D>,double>
{
	static void fill (size_t N, PivotVector<D> &Vout)
	{
		for (size_t s=0; s<D; ++s)
		for (size_t a1=0; a1<Vout.A[s].rows(); ++a1)
		for (size_t a2=0; a2<Vout.A[s].cols(); ++a2)
		{
			Vout.A[s](a1,a2) = NormDist(MtEngine);
		}
		normalize(Vout);
	}
};
//-----------</miscellaneous>-----------

#endif
