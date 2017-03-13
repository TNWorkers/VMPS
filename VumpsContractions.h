#ifndef VANILLA_VUMPSCONTRACTIONS
#define VANILLA_VUMPSCONTRACTIONS

#include "boost/multi_array.hpp"

#include "UmpsQ.h"
#include "MpoQ.h"

template<size_t Nq, typename MatrixType, typename MpoScalar>
MatrixType make_hL (const boost::multi_array<MpoScalar,4> &H2site,
                    const vector<Biped<Nq,MatrixType> > &AL,
                    const vector<qarray<Nq> > &qloc)
{
	MatrixType Mout;
	Mout.resize(AL[0].block[0].cols(), AL[0].block[0].cols());
	Mout.setZero();
	size_t D = qloc.size();
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	{
		if (H2site[s1][s2][s3][s4] != 0.)
		{
			Mout += H2site[s1][s2][s3][s4] * AL[s3].block[0].adjoint()
			                               * AL[s1].block[0].adjoint()
			                               * AL[s2].block[0]
			                               * AL[s4].block[0];
		}
	}
	
	return Mout;
}

template<size_t Nq, typename MatrixType, typename MpoScalar>
MatrixType make_hR (const boost::multi_array<MpoScalar,4> &H2site,
                    const vector<Biped<Nq,MatrixType> > &AR,
                    const vector<qarray<Nq> > &qloc)
{
	MatrixType Mout;
	Mout.resize(AR[0].block[0].rows(), AR[0].block[0].rows());
	Mout.setZero();
	size_t D = qloc.size();
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	{
		if (H2site[s1][s2][s3][s4] != 0.)
		{
			Mout += H2site[s1][s2][s3][s4] * AR[s2].block[0]
			                               * AR[s4].block[0]
			                               * AR[s3].block[0].adjoint()
			                               * AR[s1].block[0].adjoint();
		}
	}
	
	return Mout;
}

template<size_t Nq, typename MatrixType, typename MpoScalar>
MatrixType make_YL (size_t b,
                    const vector<vector<SparseMatrix<MpoScalar> > > &W,
                    const boost::multi_array<MatrixType,LEGLIMIT> &L,
                    const vector<Biped<Nq,MatrixType> > &AL,
                    const vector<qarray<Nq> > &qloc)
{
	size_t D  = qloc.size();
	size_t dW = W.size();
	size_t M  = AL[0].block[0].cols();
	
	MatrixType Mout;
	Mout.resize(M,M);
	Mout.setZero();
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (int k=0; k<W[s1][s2].outerSize(); ++k)
	for (typename SparseMatrix<MpoScalar>::InnerIterator iW(W[s1][s2],k); iW; ++iW)
	{
		size_t a = iW.row();
		
		if (a>b and b==iW.col() and iW.value() != 0.)
		{
			Mout += iW.value() * AL[s1].block[0].adjoint() * L[a][0] * AL[s2].block[0];
		}
	}
	
	return Mout;
}

template<size_t Nq, typename MatrixType, typename MpoScalar>
MatrixType make_YR (size_t a,
                    const vector<vector<SparseMatrix<MpoScalar> > > &W,
                    const boost::multi_array<MatrixType,LEGLIMIT> &R,
                    const vector<Biped<Nq,MatrixType> > &AR,
                    const vector<qarray<Nq> > &qloc)
{
	size_t D  = qloc.size();
	size_t dW = W.size();
	size_t M  = AR[0].block[0].cols();
	
	MatrixType Mout;
	Mout.resize(M,M);
	Mout.setZero();
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (int k=0; k<W[s1][s2].outerSize(); ++k)
	for (typename SparseMatrix<MpoScalar>::InnerIterator iW(W[s1][s2],k); iW; ++iW)
	{
		size_t b = iW.col();
		
		if (a>b and a==iW.row() and iW.value() != 0.)
		{
			Mout += iW.value() * AR[s1].block[0] * R[b][0] * AR[s2].block[0].adjoint();
		}
	}
	
	return Mout;
}

template<size_t Nq, typename MatrixType, typename MpoScalar>
double energy_L (const boost::multi_array<MpoScalar,4> &H2site, 
                 const vector<Biped<Nq,MatrixType> > &AL, 
                 const Biped<Nq,MatrixType> &C,
                 const vector<qarray<Nq> > &qloc)
{
	size_t D = qloc.size();
	double res = 0;
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	{
		res += H2site[s1][s2][s3][s4] * (AL[s2].block[0] * 
		                                 AL[s4].block[0] * 
		                                 C.block[0] * 
		                                 C.block[0].adjoint() * 
		                                 AL[s3].block[0].adjoint() * 
		                                 AL[s1].block[0].adjoint()
		                                ).trace();
	}
	return res;
}

template<size_t Nq, typename MatrixType, typename MpoScalar>
double energy_R (const boost::multi_array<MpoScalar,4> &H2site, 
                 const vector<Biped<Nq,MatrixType> > &AR, 
                 const Biped<Nq,MatrixType> &C,
                 const vector<qarray<Nq> > &qloc)
{
	size_t D = qloc.size();
	double res = 0;
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	{
		res += H2site[s1][s2][s3][s4] * (AR[s2].block[0] * 
		                                 AR[s4].block[0] * 
		                                 AR[s3].block[0].adjoint() * 
		                                 AR[s1].block[0].adjoint() * 
		                                 C.block[0] * 
		                                 C.block[0].adjoint()
		                                ).trace();
	}
	return res;
}

template<size_t Nq, typename MatrixType, typename MpoScalar>
MatrixType make_hL (const boost::multi_array<MpoScalar,4> &H2site,
                    const vector<Biped<Nq,MatrixType> > &AL1,
                    const vector<Biped<Nq,MatrixType> > &AL2,
                    const vector<qarray<Nq> > &qloc)
{
	MatrixType Mout;
	Mout.resize(AL1[0].block[0].cols(), AL1[0].block[0].cols());
	Mout.setZero();
	size_t D = qloc.size();
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	{
		if (H2site[s1][s2][s3][s4] != 0.)
		{
			Mout += H2site[s1][s2][s3][s4] * AL2[s3].block[0].adjoint()
			                               * AL1[s1].block[0].adjoint()
			                               * AL1[s2].block[0]
			                               * AL2[s4].block[0];
		}
	}
	
	return Mout;
}

template<size_t Nq, typename MatrixType, typename MpoScalar>
MatrixType make_hR (const boost::multi_array<MpoScalar,4> &H2site,
                    const vector<Biped<Nq,MatrixType> > &AR1,
                    const vector<Biped<Nq,MatrixType> > &AR2,
                    const vector<qarray<Nq> > &qloc)
{
	MatrixType Mout;
	Mout.resize(AR1[0].block[0].rows(), AR1[0].block[0].rows());
	Mout.setZero();
	size_t D = qloc.size();
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	{
		if (H2site[s1][s2][s3][s4] != 0.)
		{
			Mout += H2site[s1][s2][s3][s4] * AR1[s2].block[0]
			                               * AR2[s4].block[0]
			                               * AR2[s3].block[0].adjoint()
			                               * AR1[s1].block[0].adjoint();
		}
	}
	
	return Mout;
}

template<size_t Nq, typename MatrixType>
void shift_L (MatrixType &M,
              const vector<Biped<Nq,MatrixType> > &AL,
              const vector<qarray<Nq> > &qloc)
{
	size_t D = qloc.size();
	MatrixType Mtmp(D,D); Mtmp.setZero();
	
	for (size_t s=0; s<D; ++s)
	{
		Mtmp += AL[s].block[0].adjoint() * M * AL[s].block[0];
	}
	
	M = Mtmp;
}

template<size_t Nq, typename MatrixType>
void shift_R (MatrixType &M,
              const vector<Biped<Nq,MatrixType> > &AR,
              const vector<qarray<Nq> > &qloc)
{
	size_t D = qloc.size();
	MatrixType Mtmp(D,D); Mtmp.setZero();
	
	for (size_t s=0; s<D; ++s)
	{
		Mtmp += AR[s].block[0] * M * AR[s].block[0].adjoint();
	}
	
	M = Mtmp;
}

//-----------<definitions>-----------
template<size_t Nq, typename Scalar, typename MpoScalar=double>
struct PivumpsMatrix
{
	Matrix<Scalar,Dynamic,Dynamic> L;
	Matrix<Scalar,Dynamic,Dynamic> R;
	
	std::array<boost::multi_array<MpoScalar,4>,2> h;
	
	vector<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> > > AL;
	vector<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> > > AR;
	
	vector<qarray<Nq> > qloc;
	
	size_t dim;
};

template<size_t Nq, typename Scalar>
struct PivumpsVector0
{
	Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> > C;
	
	PivumpsVector0<Nq,Scalar>& operator+= (const PivumpsVector0<Nq,Scalar> &Vrhs);
	PivumpsVector0<Nq,Scalar>& operator-= (const PivumpsVector0<Nq,Scalar> &Vrhs);
	template<typename OtherScalar> PivumpsVector0<Nq,Scalar>& operator*= (const OtherScalar &alpha);
	template<typename OtherScalar> PivumpsVector0<Nq,Scalar>& operator/= (const OtherScalar &alpha);
};
//-----------</definitions>-----------

template<size_t Nq, typename Scalar, typename MpoScalar>
inline size_t dim (const PivumpsMatrix<Nq,Scalar,MpoScalar> &H)
{
	return H.dim;
}

template<size_t Nq, typename Scalar>
PivumpsVector0<Nq,Scalar>& PivumpsVector0<Nq,Scalar>::operator+= (const PivumpsVector0<Nq,Scalar> &Vrhs)
{
	transform(C.block.begin(), C.block.end(), 
	          Vrhs.C.block.begin(), C.block.begin(), 
	          std::plus<Matrix<Scalar,Dynamic,Dynamic> >());
	return *this;
}

template<size_t Nq, typename Scalar>
PivumpsVector0<Nq,Scalar>& PivumpsVector0<Nq,Scalar>::
operator-= (const PivumpsVector0<Nq,Scalar> &Vrhs)
{
	transform(C.block.begin(), C.block.end(), 
	          Vrhs.C.block.begin(), C.block.begin(), 
	          std::minus<Matrix<Scalar,Dynamic,Dynamic> >());
	return *this;
}

template<size_t Nq, typename Scalar>
template<typename OtherScalar>
PivumpsVector0<Nq,Scalar>& PivumpsVector0<Nq,Scalar>::
operator*= (const OtherScalar &alpha)
{
	for (size_t q=0; q<C.dim; ++q)
	{
		C.block[q] *= alpha;
	}
	return *this;
}

template<size_t Nq, typename Scalar>
template<typename OtherScalar>
PivumpsVector0<Nq,Scalar>& PivumpsVector0<Nq,Scalar>::
operator/= (const OtherScalar &alpha)
{
	for (size_t q=0; q<C.dim; ++q)
	{
		C.block[q] /= alpha;
	}
	return *this;
}

template<size_t Nq, typename Scalar, typename OtherScalar>
PivumpsVector0<Nq,Scalar> operator* (const OtherScalar &alpha, PivumpsVector0<Nq,Scalar> V)
{
	return V *= alpha;
}

template<size_t Nq, typename Scalar, typename OtherScalar>
PivumpsVector0<Nq,Scalar> operator* (PivumpsVector0<Nq,Scalar> V, const OtherScalar &alpha)
{
	return V *= alpha;
}

template<size_t Nq, typename Scalar, typename OtherScalar>
PivumpsVector0<Nq,Scalar> operator/ (PivumpsVector0<Nq,Scalar> V, const OtherScalar &alpha)
{
	return V /= alpha;
}

template<size_t Nq, typename Scalar>
PivumpsVector0<Nq,Scalar> operator+ (const PivumpsVector0<Nq,Scalar> &V1, const PivumpsVector0<Nq,Scalar> &V2)
{
	PivumpsVector0<Nq,Scalar> Vout = V1;
	Vout += V2;
	return Vout;
}

template<size_t Nq, typename Scalar>
PivumpsVector0<Nq,Scalar> operator- (const PivumpsVector0<Nq,Scalar> &V1, const PivumpsVector0<Nq,Scalar> &V2)
{
	PivumpsVector0<Nq,Scalar> Vout = V1;
	Vout -= V2;
	return Vout;
}

template<size_t Nq, typename Scalar>
Scalar dot (const PivumpsVector0<Nq,Scalar> &V1, const PivumpsVector0<Nq,Scalar> &V2)
{
	Scalar res = 0.;
	for (size_t q=0; q<V2.C.dim; ++q)
	for (size_t i=0; i<V2.C.block[q].cols(); ++i)
	{
		res += V1.C.block[q].col(i).dot(V2.C.block[q].col(i));
	}
	return res;
}

template<size_t Nq, typename Scalar>
double squaredNorm (const PivumpsVector0<Nq,Scalar> &V)
{
	double res = 0.;
	for (size_t q=0; q<V.C.dim; ++q)
	{
		res += V.C.block[q].colwise().squaredNorm().sum();
	}
	return res;
}

template<size_t Nq, typename Scalar>
inline double norm (const PivumpsVector0<Nq,Scalar> &V)
{
	return sqrt(squaredNorm(V));
}

template<size_t Nq, typename Scalar>
inline void normalize (PivumpsVector0<Nq,Scalar> &V)
{
	V /= norm(V);
}

template<size_t Nq, typename Scalar>
double infNorm (const PivumpsVector0<Nq,Scalar> &V1, const PivumpsVector0<Nq,Scalar> &V2)
{
	double res = 0.;
	for (size_t q=0; q<V1.C.dim; ++q)
	{
		double tmp = (V1.C.block[q]-V2.C.block[q]).template lpNorm<Eigen::Infinity>();
		if (tmp>res) {res = tmp;}
	}
	return res;
}

template<size_t Nq, typename Scalar>
void swap (PivumpsVector0<Nq,Scalar> &V1, PivumpsVector0<Nq,Scalar> &V2)
{
	for (size_t q=0; q<V1.C.dim; ++q)
	{
		V1.C.block[q].swap(V2.C.block[q]);
	}
}

template<size_t Nq, typename Scalar>
struct GaussianRandomVector<PivumpsVector0<Nq,Scalar>,Scalar>
{
	static void fill (size_t N, PivumpsVector0<Nq,Scalar> &Vout)
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

template<size_t Nq, typename Scalar, typename MpoScalar>
void HxV (const PivumpsMatrix<Nq,Scalar,MpoScalar> &H, const PivotVectorQ<Nq,Scalar> &Vin, PivotVectorQ<Nq,Scalar> &Vout)
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

template<size_t Nq, typename Scalar, typename MpoScalar>
void HxV (const PivumpsMatrix<Nq,Scalar,MpoScalar> &H, PivotVectorQ<Nq,Scalar> &Vinout)
{
	PivotVectorQ<Nq,Scalar> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

template<size_t Nq, typename Scalar, typename MpoScalar>
void HxV (const PivumpsMatrix<Nq,Scalar,MpoScalar> &H, const PivumpsVector0<Nq,Scalar> &Vin, PivumpsVector0<Nq,Scalar> &Vout)
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

template<size_t Nq, typename Scalar, typename MpoScalar>
void HxV (const PivumpsMatrix<Nq,Scalar,MpoScalar> &H, PivumpsVector0<Nq,Scalar> &Vinout)
{
	PivumpsVector0<Nq,Scalar> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

template<size_t Nq, typename MpoScalar, typename Scalar>
Scalar avg (const UmpsQ<Nq,Scalar> &Vbra, 
            const MpoQ<Nq,MpoScalar> &O, 
            const UmpsQ<Nq,Scalar> &Vket)
{
	Tripod<Nq,Matrix<Scalar,Dynamic,Dynamic> > Bnext;
	Tripod<Nq,Matrix<Scalar,Dynamic,Dynamic> > B;
	
	B.setIdentity(Vbra.get_frst_rows(),Vbra.get_frst_rows(),1,1);
	for (size_t l=0; l<O.length(); ++l)
	{
		GAUGE::OPTION g = (l==0)? GAUGE::C : GAUGE::R;
		contract_L(B, Vbra.A[g][0], O.W_at(l), Vket.A[g][0], O.locBasis(l), Bnext);
		
		B.clear();
		B = Bnext;
		Bnext.clear();
	}
	
	if (B.dim == 1)
	{
		return B.block[0][0][0].trace();
	}
	else
	{
		lout << "Warning: Result of contraction in <φ|O|ψ> has several blocks, returning 0!" << endl;
		lout << "MPS in question: " << Vket.info() << endl;
		lout << "MPO in question: " << O.info() << endl;
		return 0;
	}
}

#endif
