#ifndef VUMPSPIVOTSTUFF
#define VUMPSPIVOTSTUFF

#include "tensors/Biped.h"
#include "pivot/DmrgPivotStuff1.h"
#include "pivot/DmrgPivotStuff0.h"

//-----------<definitions>-----------

/**Structure to update \f$A_C\f$ (eq. 11) with 2-site Hamiltonian. Contains \f$A_L\f$, \f$A_L\f$ and \f$H_L\f$ (= \p L), \f$H_R\f$ (= \p R).
\ingroup VUMPS
*/
template<typename Symmetry, typename Scalar, typename MpoScalar=double>
struct PivumpsMatrix
{
	PivumpsMatrix(){};
	
	Matrix<Scalar,Dynamic,Dynamic> L;
	Matrix<Scalar,Dynamic,Dynamic> R;
	
	std::array<boost::multi_array<MpoScalar,4>,2> h;
	
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > AL;
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > AR;
	
	vector<qarray<Symmetry::Nq> > qloc;
	
	size_t dim;
};

///**Wrapper containing \f$C\f$ for local upates (eq. 16).*/
//template<typename Symmetry, typename Scalar>
//struct PivotVector0
//{
//	PivotVector0(){};
//	
//	PivotVector0 (const Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &Crhs)
//	{
//		C = Crhs;
//		
//		dim = 0;
//		for (size_t q=0; q<A.dim; ++q)
//		{
//			dim += A.block[q].rows() * A.block[q].cols();
//		}
//	}
//	
//	Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > C;
//	
//	size_t dim;
//	
//	///\{
//	/**Linear algebra for \f$C\f$ in the vector space.*/
//	PivotVector0<Symmetry,Scalar>& operator+= (const PivotVector0<Symmetry,Scalar> &Vrhs);
//	PivotVector0<Symmetry,Scalar>& operator-= (const PivotVector0<Symmetry,Scalar> &Vrhs);
//	template<typename OtherScalar> PivotVector0<Symmetry,Scalar>& operator*= (const OtherScalar &alpha);
//	template<typename OtherScalar> PivotVector0<Symmetry,Scalar>& operator/= (const OtherScalar &alpha);
//	///\}
//};

//-----------</definitions>-----------

template<typename Symmetry, typename Scalar, typename MpoScalar>
inline size_t dim (const PivumpsMatrix<Symmetry,Scalar,MpoScalar> &H)
{
	return H.dim;
}

//template<typename Symmetry, typename Scalar>
//PivotVector0<Symmetry,Scalar>& PivotVector0<Symmetry,Scalar>::operator+= (const PivotVector0<Symmetry,Scalar> &Vrhs)
//{
//	transform(C.block.begin(), C.block.end(), 
//	          Vrhs.C.block.begin(), C.block.begin(), 
//	          std::plus<Matrix<Scalar,Dynamic,Dynamic> >());
//	return *this;
//}

//template<typename Symmetry, typename Scalar>
//PivotVector0<Symmetry,Scalar>& PivotVector0<Symmetry,Scalar>::
//operator-= (const PivotVector0<Symmetry,Scalar> &Vrhs)
//{
//	transform(C.block.begin(), C.block.end(), 
//	          Vrhs.C.block.begin(), C.block.begin(), 
//	          std::minus<Matrix<Scalar,Dynamic,Dynamic> >());
//	return *this;
//}

//template<typename Symmetry, typename Scalar>
//template<typename OtherScalar>
//PivotVector0<Symmetry,Scalar>& PivotVector0<Symmetry,Scalar>::
//operator*= (const OtherScalar &alpha)
//{
//	for (size_t q=0; q<C.dim; ++q)
//	{
//		C.block[q] *= alpha;
//	}
//	return *this;
//}

//template<typename Symmetry, typename Scalar>
//template<typename OtherScalar>
//PivotVector0<Symmetry,Scalar>& PivotVector0<Symmetry,Scalar>::
//operator/= (const OtherScalar &alpha)
//{
//	for (size_t q=0; q<C.dim; ++q)
//	{
//		C.block[q] /= alpha;
//	}
//	return *this;
//}

//template<typename Symmetry, typename Scalar, typename OtherScalar>
//PivotVector0<Symmetry,Scalar> operator* (const OtherScalar &alpha, PivotVector0<Symmetry,Scalar> V)
//{
//	return V *= alpha;
//}

//template<typename Symmetry, typename Scalar, typename OtherScalar>
//PivotVector0<Symmetry,Scalar> operator* (PivotVector0<Symmetry,Scalar> V, const OtherScalar &alpha)
//{
//	return V *= alpha;
//}

//template<typename Symmetry, typename Scalar, typename OtherScalar>
//PivotVector0<Symmetry,Scalar> operator/ (PivotVector0<Symmetry,Scalar> V, const OtherScalar &alpha)
//{
//	return V /= alpha;
//}

//template<typename Symmetry, typename Scalar>
//PivotVector0<Symmetry,Scalar> operator+ (const PivotVector0<Symmetry,Scalar> &V1, const PivotVector0<Symmetry,Scalar> &V2)
//{
//	PivotVector0<Symmetry,Scalar> Vout = V1;
//	Vout += V2;
//	return Vout;
//}

//template<typename Symmetry, typename Scalar>
//PivotVector0<Symmetry,Scalar> operator- (const PivotVector0<Symmetry,Scalar> &V1, const PivotVector0<Symmetry,Scalar> &V2)
//{
//	PivotVector0<Symmetry,Scalar> Vout = V1;
//	Vout -= V2;
//	return Vout;
//}

//template<typename Symmetry, typename Scalar>
//Scalar dot (const PivotVector0<Symmetry,Scalar> &V1, const PivotVector0<Symmetry,Scalar> &V2)
//{
//	Scalar res = 0.;
////	for (size_t q=0; q<V2.C.dim; ++q)
////	for (size_t i=0; i<V2.C.block[q].cols(); ++i)
////	{
////		res += V1.C.block[q].col(i).dot(V2.C.block[q].col(i));
////	}
//	for (size_t q=0; q<V2.C.dim; ++q)
//	{
//		res += (V1.C.block[q].adjoint() * V2.C.block[q]).trace();
//	}
//	return res;
//}

//template<typename Symmetry, typename Scalar>
//double squaredNorm (const PivotVector0<Symmetry,Scalar> &V)
//{
//	double res = 0.;
//	for (size_t q=0; q<V.C.dim; ++q)
//	{
//		res += V.C.block[q].colwise().squaredNorm().sum();
//	}
//	return res;
//}

//template<typename Symmetry, typename Scalar>
//inline double norm (const PivotVector0<Symmetry,Scalar> &V)
//{
//	return sqrt(squaredNorm(V));
//}

//template<typename Symmetry, typename Scalar>
//inline void normalize (PivotVector0<Symmetry,Scalar> &V)
//{
//	V /= norm(V);
//}

//template<typename Symmetry, typename Scalar>
//double infNorm (const PivotVector0<Symmetry,Scalar> &V1, const PivotVector0<Symmetry,Scalar> &V2)
//{
//	double res = 0.;
//	for (size_t q=0; q<V1.C.dim; ++q)
//	{
//		double tmp = (V1.C.block[q]-V2.C.block[q]).template lpNorm<Eigen::Infinity>();
//		if (tmp>res) {res = tmp;}
//	}
//	return res;
//}

//template<typename Symmetry, typename Scalar>
//void swap (PivotVector0<Symmetry,Scalar> &V1, PivotVector0<Symmetry,Scalar> &V2)
//{
//	for (size_t q=0; q<V1.C.dim; ++q)
//	{
//		V1.C.block[q].swap(V2.C.block[q]);
//	}
//}

//template<typename Symmetry, typename Scalar>
//struct GaussianRandomVector<PivotVector0<Symmetry,Scalar>,Scalar>
//{
//	static void fill (size_t N, PivotVector0<Symmetry,Scalar> &Vout)
//	{
//		for (size_t q=0; q<Vout.C.dim; ++q)
//		for (size_t a1=0; a1<Vout.C.block[q].rows(); ++a1)
//		for (size_t a2=0; a2<Vout.C.block[q].cols(); ++a2)
//		{
//			Vout.C.block[q](a1,a2) = threadSafeRandUniform<Scalar>(-1.,1.);
//		}
//		normalize(Vout);
//	}
//};

/**Performs the local update of \f$A_C\f$ (eq. 11) with a 2-site Hamiltonian.*/
template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivumpsMatrix<Symmetry,Scalar,MpoScalar> &H, const PivotVector1<Symmetry,Scalar> &Vin, PivotVector1<Symmetry,Scalar> &Vout)
{
	size_t D = H.qloc.size();
	Vout = Vin;
	for (size_t s=0; s<D; ++s)
	{
		Vout.A[s].block[0].setZero();
	}
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	{
		if (H.h[0][s1][s2][s3][s4] != 0.)
		{
			Vout.A[s3].block[0] += H.h[0][s1][s2][s3][s4] * H.AL[s1].block[0].adjoint() * H.AL[s2].block[0] * Vin.A[s4].block[0];
		}
	}
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	{
		if (H.h[1][s1][s2][s3][s4] != 0.)
		{
			Vout.A[s1].block[0] += H.h[1][s1][s2][s3][s4] * Vin.A[s2].block[0] * H.AR[s4].block[0] * H.AR[s3].block[0].adjoint();
		}
	}
	
	for (size_t s=0; s<D; ++s)
	{
		Vout.A[s].block[0] += H.L * Vin.A[s].block[0];
		Vout.A[s].block[0] += Vin.A[s].block[0] * H.R;
	}
}

/**Performs \p HxV in place.*/
template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivumpsMatrix<Symmetry,Scalar,MpoScalar> &H, PivotVector1<Symmetry,Scalar> &Vinout)
{
	PivotVector1<Symmetry,Scalar> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

/**Performs the local update of \f$C\f$ (eq. 16) with an explicit 2-site Hamiltonian.*/
template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivumpsMatrix<Symmetry,Scalar,MpoScalar> &H, const PivotVector0<Symmetry,Scalar> &Vin, PivotVector0<Symmetry,Scalar> &Vout)
{
	size_t D = H.qloc.size();
	
	Vout = Vin;
	Vout.C.setZero();
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	{
		if (H.h[1][s1][s2][s3][s4] != 0.)
		{
			Vout.C.block[0] += H.h[1][s1][s2][s3][s4] * H.AL[s1].block[0].adjoint() * H.AL[s2].block[0] 
			                                          * Vin.C.block[0] 
			                                          * H.AR[s4].block[0] * H.AR[s3].block[0].adjoint();
		}
	}
	
	Vout.C.block[0] += H.L * Vin.C.block[0];
	Vout.C.block[0] += Vin.C.block[0] * H.R;
}

/**Performs \p HxV in place.*/
template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivumpsMatrix<Symmetry,Scalar,MpoScalar> &H, PivotVector0<Symmetry,Scalar> &Vinout)
{
	PivotVector0<Symmetry,Scalar> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

#endif
