#ifndef STRAWBERRY_DMRG_HEFF_STUFF_0SITE_WITH_Q
#define STRAWBERRY_DMRG_HEFF_STUFF_0SITE_WITH_Q

#include "DmrgExternal.h"
#include "tensors/Biped.h"

//-----------<definitions>-----------
template<typename Symmetry, typename Scalar>
struct PivotVector0
{
	PivotVector0(){};
	
	PivotVector0 (const Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &Crhs)
	:C(Crhs)
	{}
	
	Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > C;
	
	/**Set blocks as in Vrhs, but do not resize the matrices*/
	void outerResize (const PivotVector0 &Vrhs)
	{
		C.clear();
		C.in = Vrhs.C.in;
		C.out = Vrhs.C.out;
		C.dict = Vrhs.C.dict;
		C.block.resize(Vrhs.C.block.size());
		C.dim = Vrhs.C.dim;
	}
	
	PivotVector0<Symmetry,Scalar>& operator+= (const PivotVector0<Symmetry,Scalar> &Vrhs);
	PivotVector0<Symmetry,Scalar>& operator-= (const PivotVector0<Symmetry,Scalar> &Vrhs);
	template<typename OtherScalar> PivotVector0<Symmetry,Scalar>& operator*= (const OtherScalar &alpha);
	template<typename OtherScalar> PivotVector0<Symmetry,Scalar>& operator/= (const OtherScalar &alpha);
};
//-----------</definitions>-----------

//-----------<vector arithmetics>-----------
template<typename Symmetry, typename Scalar>
PivotVector0<Symmetry,Scalar>& PivotVector0<Symmetry,Scalar>::
operator+= (const PivotVector0<Symmetry,Scalar> &Vrhs)
{
	C = C + Vrhs.C;
	return *this;
}

template<typename Symmetry, typename Scalar>
PivotVector0<Symmetry,Scalar>& PivotVector0<Symmetry,Scalar>::
operator-= (const PivotVector0<Symmetry,Scalar> &Vrhs)
{
	C = C - Vrhs.C;
	return *this;
}

template<typename Symmetry, typename Scalar>
template<typename OtherScalar>
PivotVector0<Symmetry,Scalar>& PivotVector0<Symmetry,Scalar>::
operator*= (const OtherScalar &alpha)
{
	for (size_t q=0; q<C.dim; ++q)
	{
		C.block[q] *= alpha;
	}
	return *this;
}

template<typename Symmetry, typename Scalar>
template<typename OtherScalar>
PivotVector0<Symmetry,Scalar>& PivotVector0<Symmetry,Scalar>::
operator/= (const OtherScalar &alpha)
{
	for (size_t q=0; q<C.dim; ++q)
	{
		C.block[q] /= alpha;
	}
	return *this;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
PivotVector0<Symmetry,Scalar> operator* (const OtherScalar &alpha, PivotVector0<Symmetry,Scalar> V)
{
	return V *= alpha;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
PivotVector0<Symmetry,Scalar> operator* (PivotVector0<Symmetry,Scalar> V, const OtherScalar &alpha)
{
	return V *= alpha;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
PivotVector0<Symmetry,Scalar> operator/ (PivotVector0<Symmetry,Scalar> V, const OtherScalar &alpha)
{
	return V /= alpha;
}

template<typename Symmetry, typename Scalar>
PivotVector0<Symmetry,Scalar> operator+ (const PivotVector0<Symmetry,Scalar> &V1, const PivotVector0<Symmetry,Scalar> &V2)
{
	PivotVector0<Symmetry,Scalar> Vout = V1;
	Vout += V2;
	return Vout;
}

template<typename Symmetry, typename Scalar>
PivotVector0<Symmetry,Scalar> operator- (const PivotVector0<Symmetry,Scalar> &V1, const PivotVector0<Symmetry,Scalar> &V2)
{
	PivotVector0<Symmetry,Scalar> Vout = V1;
	Vout -= V2;
	return Vout;
}
//-----------</vector arithmetics>-----------

//-----------<matrix*vector>-----------
/**Calculates the following contraction:
\dotfile HxV_0site.dot*/
template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivotMatrix<Symmetry,Scalar,MpoScalar> &H, const PivotVector0<Symmetry,Scalar> &Vin, PivotVector0<Symmetry,Scalar> &Vout)
{
	Vout.outerResize(Vin);
	
	for (size_t qL=0; qL<H.L.dim; ++qL)
	{
		qarray3<Symmetry::Nq> qupleR = {H.L.out(qL), H.L.in(qL), H.L.mid(qL)};
		auto qR = H.R.dict.find(qupleR);
		
		if (qR != H.R.dict.end())
		{
			qarray2<Symmetry::Nq> qupleAin = {H.L.out(qL), H.L.out(qL)};
			auto qAin = Vin.C.dict.find(qupleAin);
			
			if (qAin != Vin.C.dict.end())
			{
				qarray2<Symmetry::Nq> qupleAout = {H.R.out(qR->second), H.R.out(qR->second)};
				auto qAout = Vout.C.dict.find(qupleAout);
				
				if (qAout != Vout.C.dict.end())
				{
					for (size_t a=0; a<max(H.W[0][0][0].rows(),H.W[0][0][0].cols()); ++a)
					{
						Matrix<Scalar,Dynamic,Dynamic> Mtmp;
						
						if (H.L.block[qL][a][0].rows() != 0 and
						    H.R.block[qR->second][a][0].rows() !=0)
						{
							optimal_multiply(1., 
							                 H.L.block[qL][a][0],
							                 Vin.C.block[qAin->second],
							                 H.R.block[qR->second][a][0],
							                 Mtmp);
						}
						
						if (Mtmp.rows() != 0)
						{
							if (Vout.C.block[qAout->second].rows() != 0)
							{
								Vout.C.block[qAout->second] += Mtmp;
							}
							else
							{
								Vout.C.block[qAout->second] = Mtmp;
							}
						}
					}
				}
			}
		}
	}
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivotMatrix<Symmetry,Scalar,MpoScalar> &H, PivotVector0<Symmetry,Scalar> &Vinout)
{
	PivotVector0<Symmetry,Scalar> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}
//-----------</matrix*vector>-----------

//-----------<dot & vector norms>-----------
template<typename Symmetry, typename Scalar>
Scalar dot (const PivotVector0<Symmetry,Scalar> &V1, const PivotVector0<Symmetry,Scalar> &V2)
{
	Scalar res = 0.;
	for (size_t q=0; q<V2.C.dim; ++q)
	{
		res += (V1.C.block[q].adjoint() * V2.C.block[q]).trace();
	}
	return res;
}

template<typename Symmetry, typename Scalar>
inline double squaredNorm (const PivotVector0<Symmetry,Scalar> &V)
{
	return isReal(dot(V,V));
}

template<typename Symmetry, typename Scalar>
inline double norm (const PivotVector0<Symmetry,Scalar> &V)
{
	return sqrt(squaredNorm(V));
}

template<typename Symmetry, typename Scalar>
inline void normalize (PivotVector0<Symmetry,Scalar> &V)
{
	V /= norm(V);
}

template<typename Symmetry, typename Scalar>
double infNorm (const PivotVector0<Symmetry,Scalar> &V1, const PivotVector0<Symmetry,Scalar> &V2)
{
	double res = 0.;
	for (size_t q=0; q<V1.C.dim; ++q)
	{
		double tmp = (V1.C.block[q]-V2.C.block[q]).template lpNorm<Eigen::Infinity>();
		if (tmp>res) {res = tmp;}
	}
	return res;
}

template<typename Symmetry, typename Scalar>
inline size_t dim (const PivotVector0<Symmetry,Scalar> &V)
{
	size_t out = 0;
	for (size_t q=0; q<V.C.dim; ++q)
	{
		out += V.C.block[q].size();
	}
	return out;
}
//-----------</dot & vector norms>-----------

//-----------<miscellaneous>-----------
template<typename Symmetry, typename Scalar>
void swap (PivotVector0<Symmetry,Scalar> &V1, PivotVector0<Symmetry,Scalar> &V2)
{
	for (size_t q=0; q<V1.C.dim; ++q)
	{
		V1.C.block[q].swap(V2.C.block[q]);
	}
}

#include "RandomVector.h"

template<typename Symmetry, typename Scalar>
struct GaussianRandomVector<PivotVector0<Symmetry,Scalar>,Scalar>
{
	static void fill (size_t N, PivotVector0<Symmetry,Scalar> &Vout)
	{
		for (size_t q=0; q<Vout.C.dim; ++q)
		for (size_t a1=0; a1<Vout.C.block[q].rows(); ++a1)
		for (size_t a2=0; a2<Vout.C.block[q].cols(); ++a2)
		{
			Vout.C.block[q](a1,a2) = threadSafeRandUniform<Scalar>(-1.,1.);
		}
		normalize(Vout);
	}
};
//-----------</miscellaneous>-----------

#endif
