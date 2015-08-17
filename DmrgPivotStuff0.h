#ifndef STRAWBERRY_DMRG_HEFF_STUFF_0SITE_WITH_Q
#define STRAWBERRY_DMRG_HEFF_STUFF_0SITE_WITH_Q

//-----------<definitions>-----------
template<size_t Nq, typename Scalar>
struct PivotVector0Q
{
	Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> > A;
	
	PivotVector0Q<Nq,Scalar>& operator+= (const PivotVector0Q<Nq,Scalar> &Vrhs);
	PivotVector0Q<Nq,Scalar>& operator-= (const PivotVector0Q<Nq,Scalar> &Vrhs);
	template<typename OtherScalar> PivotVector0Q<Nq,Scalar>& operator*= (const OtherScalar &alpha);
	template<typename OtherScalar> PivotVector0Q<Nq,Scalar>& operator/= (const OtherScalar &alpha);
};
//-----------</definitions>-----------

//-----------<vector arithmetics>-----------
template<size_t Nq, typename Scalar>
PivotVector0Q<Nq,Scalar>& PivotVector0Q<Nq,Scalar>::
operator+= (const PivotVector0Q<Nq,Scalar> &Vrhs)
{
	transform(A.block.begin(), A.block.end(), 
	          Vrhs.A.block.begin(), A.block.begin(), 
	          std::plus<Matrix<Scalar,Dynamic,Dynamic> >());
	return *this;
}

template<size_t Nq, typename Scalar>
PivotVector0Q<Nq,Scalar>& PivotVector0Q<Nq,Scalar>::
operator-= (const PivotVector0Q<Nq,Scalar> &Vrhs)
{
	transform(A.block.begin(), A.block.end(), 
	          Vrhs.A.block.begin(), A.block.begin(), 
	          std::minus<Matrix<Scalar,Dynamic,Dynamic> >());
	return *this;
}

template<size_t Nq, typename Scalar>
template<typename OtherScalar>
PivotVector0Q<Nq,Scalar>& PivotVector0Q<Nq,Scalar>::
operator*= (const OtherScalar &alpha)
{
	for (size_t q=0; q<A.dim; ++q)
	{
		A.block[q] *= alpha;
	}
	return *this;
}

template<size_t Nq, typename Scalar>
template<typename OtherScalar>
PivotVector0Q<Nq,Scalar>& PivotVector0Q<Nq,Scalar>::
operator/= (const OtherScalar &alpha)
{
	for (size_t q=0; q<A.dim; ++q)
	{
		A.block[q] /= alpha;
	}
	return *this;
}

template<size_t Nq, typename Scalar, typename OtherScalar>
PivotVector0Q<Nq,Scalar> operator* (const OtherScalar &alpha, PivotVector0Q<Nq,Scalar> V)
{
	return V *= alpha;
}

template<size_t Nq, typename Scalar, typename OtherScalar>
PivotVector0Q<Nq,Scalar> operator* (PivotVector0Q<Nq,Scalar> V, const OtherScalar &alpha)
{
	return V *= alpha;
}

template<size_t Nq, typename Scalar, typename OtherScalar>
PivotVector0Q<Nq,Scalar> operator/ (PivotVector0Q<Nq,Scalar> V, const OtherScalar &alpha)
{
	return V /= alpha;
}

template<size_t Nq, typename Scalar>
PivotVector0Q<Nq,Scalar> operator+ (const PivotVector0Q<Nq,Scalar> &V1, const PivotVector0Q<Nq,Scalar> &V2)
{
	PivotVector0Q<Nq,Scalar> Vout = V1;
	Vout += V2;
	return Vout;
}

template<size_t Nq, typename Scalar>
PivotVector0Q<Nq,Scalar> operator- (const PivotVector0Q<Nq,Scalar> &V1, const PivotVector0Q<Nq,Scalar> &V2)
{
	PivotVector0Q<Nq,Scalar> Vout = V1;
	Vout -= V2;
	return Vout;
}
//-----------</vector arithmetics>-----------

//-----------<matrix*vector>-----------
/**Calculates the following contraction:
\dotfile HxV_0site.dot*/
template<size_t Nq, typename Scalar, typename MpoScalar>
void HxV (const PivotMatrixQ<Nq,Scalar,MpoScalar> &H, const PivotVector0Q<Nq,Scalar> &Vin, PivotVector0Q<Nq,Scalar> &Vout)
{
	Vout = Vin;
	Vout.A.setZero();
	
	for (size_t qL=0; qL<H.L.dim; ++qL)
	{
		qarray3<Nq> qupleR = {H.L.out(qL), H.L.in(qL), H.L.mid(qL)};
		auto qR = H.R.dict.find(qupleR);
		
		if (qR != H.R.dict.end())
		{
			qarray2<Nq> qupleAin = {H.L.out(qL), H.L.out(qL)};
			auto qAin = Vin.A.dict.find(qupleAin);
			
			if (qAin != Vin.A.dict.end())
			{
				qarray2<Nq> qupleAout = {H.R.out(qR->second), H.R.out(qR->second)};
				auto qAout = Vout.A.dict.find(qupleAout);
				
				if (qAout != Vout.A.dict.end())
				{
					for (size_t a=0; a<max(H.W[0][0].rows(),H.W[0][0].cols()); ++a)
					{
						Matrix<Scalar,Dynamic,Dynamic> Mtmp;
						
						if (H.L.block[qL][a][0].rows() != 0 and
							H.R.block[qR->second][a][0].rows() !=0)
						{
							optimal_multiply(1., 
							                 H.L.block[qL][a][0],
							                 Vin.A.block[qAin->second],
							                 H.R.block[qR->second][a][0],
							                 Mtmp);
						}
						
						if (Mtmp.rows() != 0)
						{
							Vout.A.block[qAout->second] += Mtmp;
						}
					}
				}
			}
		}
	}
}

template<size_t Nq, typename Scalar, typename MpoScalar>
void HxV (const PivotMatrixQ<Nq,Scalar,MpoScalar> &H, PivotVector0Q<Nq,Scalar> &Vinout)
{
	PivotVector0Q<Nq,Scalar> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}
//-----------</matrix*vector>-----------

//-----------<dot & vector norms>-----------
template<size_t Nq, typename Scalar>
Scalar dot (const PivotVector0Q<Nq,Scalar> &V1, const PivotVector0Q<Nq,Scalar> &V2)
{
	Scalar res = 0.;
	for (size_t q=0; q<V2.A.dim; ++q)
	for (size_t i=0; i<V2.A.block[q].cols(); ++i)
	{
		res += V1.A.block[q].col(i).dot(V2.A.block[q].col(i));
	}
	return res;
}

template<size_t Nq, typename Scalar>
double squaredNorm (const PivotVector0Q<Nq,Scalar> &V)
{
	double res = 0.;
	for (size_t q=0; q<V.A.dim; ++q)
	{
		res += V.A.block[q].colwise().squaredNorm().sum();
	}
	return res;
}

template<size_t Nq, typename Scalar>
inline double norm (const PivotVector0Q<Nq,Scalar> &V)
{
	return sqrt(squaredNorm(V));
}

template<size_t Nq, typename Scalar>
inline void normalize (PivotVector0Q<Nq,Scalar> &V)
{
	V /= norm(V);
}

template<size_t Nq, typename Scalar>
double infNorm (const PivotVector0Q<Nq,Scalar> &V1, const PivotVector0Q<Nq,Scalar> &V2)
{
	double res = 0.;
	for (size_t q=0; q<V1.A.dim; ++q)
	{
		double tmp = (V1.A.block[q]-V2.A.block[q]).template lpNorm<Eigen::Infinity>();
		if (tmp>res) {res = tmp;}
	}
	return res;
}
//-----------</dot & vector norms>-----------

//-----------<miscellaneous>-----------
template<size_t Nq, typename Scalar>
void swap (PivotVector0Q<Nq,Scalar> &V1, PivotVector0Q<Nq,Scalar> &V2)
{
	for (size_t q=0; q<V1.A.dim; ++q)
	{
		V1.A.block[q].swap(V2.A.block[q]);
	}
}

#include "RandomVector.h"

template<size_t Nq, typename Scalar>
struct GaussianRandomVector<PivotVector0Q<Nq,Scalar>,Scalar>
{
	static void fill (size_t N, PivotVector0Q<Nq,Scalar> &Vout)
	{
		for (size_t q=0; q<Vout.A.dim; ++q)
		for (size_t a1=0; a1<Vout.A.block[q].rows(); ++a1)
		for (size_t a2=0; a2<Vout.A.block[q].cols(); ++a2)
		{
			Vout.A.block[q](a1,a2) = threadSafeRandUniform<Scalar>(-1.,1.);
		}
		normalize(Vout);
	}
};
//-----------</miscellaneous>-----------

#endif
