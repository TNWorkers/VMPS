#ifndef DMRGPIVOTVECTOR
#define DMRGPIVOTVECTOR

#include "numeric_limits.h"
#include "tensors/DmrgContractions.h" // for contract_AA

template<typename Symmetry, typename Scalar>
struct PivotVector
{
	static constexpr std::size_t Nq = Symmetry::Nq;
	
	PivotVector()
	{
		data.resize(1); // needs to be set for 0-site
	};
	
	/**Set from a center matrix.*/
	PivotVector (const Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &C)
	{
		data.resize(1);
		data[0] = C;
	}
	
	/**Set from one A-tensor.*/
	PivotVector (const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &A12)
	:data(A12)
	{}
	
	/**Make contraction of two A-tensors.*/
	PivotVector (const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &A12,
	             const vector<qarray<Symmetry::Nq> > &qloc12,
	             const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &A34,
	             const vector<qarray<Symmetry::Nq> > &qloc34)
	{
		contract_AA(A12, qloc12, A34, qloc34, data);
	}
	
	/**Set blocks as in Vrhs, but do not resize the matrices*/
	void outerResize (const PivotVector &Vrhs)
	{
		data.clear();
		data.resize(Vrhs.data.size());
		for (size_t i=0; i<data.size(); ++i)
		{
			data[i].in = Vrhs.data[i].in;
			data[i].out = Vrhs.data[i].out;
			data[i].dict = Vrhs.data[i].dict;
			data[i].block.resize(Vrhs.data[i].block.size());
			data[i].dim = Vrhs.data[i].dim;
		}
	}
	
	PivotVector<Symmetry,Scalar>& operator+= (const PivotVector<Symmetry,Scalar> &Vrhs);
	PivotVector<Symmetry,Scalar>& operator-= (const PivotVector<Symmetry,Scalar> &Vrhs);
	template<typename OtherScalar> PivotVector<Symmetry,Scalar>& operator*= (const OtherScalar &alpha);
	template<typename OtherScalar> PivotVector<Symmetry,Scalar>& operator/= (const OtherScalar &alpha);
	
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > data;
};
//-----------</definitions>-----------

//-----------<vector arithmetics>-----------
template<typename Symmetry, typename Scalar>
PivotVector<Symmetry,Scalar>& PivotVector<Symmetry,Scalar>::operator+= (const PivotVector<Symmetry,Scalar> &Vrhs)
{
	for (std::size_t s=0; s<data.size(); s++)
	{
		data[s] = data[s] + Vrhs.data[s];
	}
	return *this;
}

template<typename Symmetry, typename Scalar>
PivotVector<Symmetry,Scalar>& PivotVector<Symmetry,Scalar>::
operator-= (const PivotVector<Symmetry,Scalar> &Vrhs)
{
	for (std::size_t s=0; s<data.size(); s++)
	{
		data[s] = data[s] - Vrhs.data[s];
	}
	return *this;
}

template<typename Symmetry, typename Scalar>
template<typename OtherScalar>
PivotVector<Symmetry,Scalar>& PivotVector<Symmetry,Scalar>::
operator*= (const OtherScalar &alpha)
{
	for (size_t s=0; s<data.size(); ++s)
	for (size_t q=0; q<data[s].dim; ++q)
	{
		data[s].block[q] *= alpha;
	}
	return *this;
}

template<typename Symmetry, typename Scalar>
template<typename OtherScalar>
PivotVector<Symmetry,Scalar>& PivotVector<Symmetry,Scalar>::
operator/= (const OtherScalar &alpha)
{
	for (size_t s=0; s<data.size(); ++s)
	for (size_t q=0; q<data[s].dim; ++q)
	{
		data[s].block[q] /= alpha;
	}
	return *this;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
PivotVector<Symmetry,Scalar> operator* (const OtherScalar &alpha, PivotVector<Symmetry,Scalar> V)
{
	return V *= alpha;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
PivotVector<Symmetry,Scalar> operator* (PivotVector<Symmetry,Scalar> V, const OtherScalar &alpha)
{
	return V *= alpha;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
PivotVector<Symmetry,Scalar> operator/ (PivotVector<Symmetry,Scalar> V, const OtherScalar &alpha)
{
	return V /= alpha;
}

template<typename Symmetry, typename Scalar>
PivotVector<Symmetry,Scalar> operator+ (const PivotVector<Symmetry,Scalar> &V1, const PivotVector<Symmetry,Scalar> &V2)
{
	PivotVector<Symmetry,Scalar> Vout = V1;
	Vout += V2;
	return Vout;
}

template<typename Symmetry, typename Scalar>
PivotVector<Symmetry,Scalar> operator- (const PivotVector<Symmetry,Scalar> &V1, const PivotVector<Symmetry,Scalar> &V2)
{
	PivotVector<Symmetry,Scalar> Vout = V1;
	Vout -= V2;
	return Vout;
}

//-----------<dot & vector norms>-----------
template<typename Symmetry, typename Scalar>
Scalar dot (const PivotVector<Symmetry,Scalar> &V1, const PivotVector<Symmetry,Scalar> &V2)
{
	Scalar res = 0;
	for (size_t s=0; s<V2.data.size(); ++s)
	for (size_t q=0; q<V2.data[s].dim; ++q)
	{
		res += (V1.data[s].block[q].adjoint() * V2.data[s].block[q]).trace() * Symmetry::coeff_dot(V1.data[s].out[q]);
	}
	return res;
}

template<typename Symmetry, typename Scalar>
double squaredNorm (const PivotVector<Symmetry,Scalar> &V)
{
	double res = isReal(dot(V,V));
	return res;
}

template<typename Symmetry, typename Scalar>
inline double norm (const PivotVector<Symmetry,Scalar> &V)
{
	return sqrt(squaredNorm(V));
}

template<typename Symmetry, typename Scalar>
inline void normalize (PivotVector<Symmetry,Scalar> &V)
{
	V /= norm(V);
}

template<typename Symmetry, typename Scalar>
double infNorm (const PivotVector<Symmetry,Scalar> &V1, const PivotVector<Symmetry,Scalar> &V2)
{
	double res = 0.;
	for (size_t s=0; s<V1.data.size(); ++s)
	{
		auto Mtmp = V1.data[s] - V2.data[s];
		for (size_t q=0; q<Mtmp.dim; ++q)
		{
			double tmp = Mtmp.block[q].template lpNorm<Eigen::Infinity>();
			if (tmp>res) {res = tmp;}
		}
	}
	return res;
}

template<typename Symmetry, typename Scalar>
inline size_t dim (const PivotVector<Symmetry,Scalar> &V)
{
	size_t out = 0;
	for (size_t s=0; s<V.data.size(); ++s)
	for (size_t q=0; q<V.data[s].dim; ++q)
	{
		out += V.data[s].block[q].size();
	}
	return out;
}

template<typename Symmetry, typename Scalar>
void swap (PivotVector<Symmetry,Scalar> &V1, PivotVector<Symmetry,Scalar> &V2)
{
	for (size_t s=0; s<V1.data.size(); ++s)
	{
		V1.data[s].block.swap(V2.data[s].block);
	}
}

#include "RandomVector.h"

template<typename Symmetry, typename Scalar>
struct GaussianRandomVector<PivotVector<Symmetry,Scalar>,Scalar>
{
	static void fill (size_t N, PivotVector<Symmetry,Scalar> &Vout)
	{
		for (size_t s=0; s<Vout.data.size(); ++s)
		for (size_t q=0; q<Vout.data[s].dim; ++q)
		for (size_t a1=0; a1<Vout.data[s].block[q].rows(); ++a1)
		for (size_t a2=0; a2<Vout.data[s].block[q].cols(); ++a2)
		{
			Vout.data[s].block[q](a1,a2) = threadSafeRandUniform<Scalar>(-1.,1.);
		}
		normalize(Vout);
	}
};

#endif
