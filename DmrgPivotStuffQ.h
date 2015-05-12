#ifndef STRAWBERRY_DMRGHEFFSTUFF_WITH_Q
#define STRAWBERRY_DMRGHEFFSTUFF_WITH_Q

#include "DmrgTypedefs.h"
#include "Biped.h"
#include "Multipede.h"

//-----------<definitions>-----------
template<size_t D, size_t Nq, typename Scalar>
struct PivotMatrixQ
{
	Tripod<Nq,Matrix<Scalar,Dynamic,Dynamic> > L;
	Tripod<Nq,Matrix<Scalar,Dynamic,Dynamic> > R;
	std::array<std::array<SparseMatrixXd,D>,D> W;
	
	size_t dim;
	
	vector<std::array<size_t,2> > qlhs;
	vector<vector<std::array<size_t,4> > > qrhs;
};

//template<size_t D, size_t Nq, typename Scalar> using PivotVectorQ<D,Nq,Scalar> = std::array<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> >,D>;

//template<size_t D, size_t Nq, typename Scalar>
//std::array<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> >,D>& 
//std::array<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> >,D>::operator*= (const double &alpha)
//{
////	for (size_t s=0; s<D; ++s)
////	for (size_t q=0; q<A[s].dim; ++q)
////	{
////		this[s].block[q] *= alpha;
////	}
////	return *this;
//}

template<size_t D, size_t Nq, typename Scalar>
struct PivotVectorQ
{
	std::array<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> >,D> A;
	
	PivotVectorQ<D,Nq,Scalar>& operator+= (const PivotVectorQ<D,Nq,Scalar> &Vrhs);
	PivotVectorQ<D,Nq,Scalar>& operator-= (const PivotVectorQ<D,Nq,Scalar> &Vrhs);
	PivotVectorQ<D,Nq,Scalar>& operator*= (const double &alpha);
	PivotVectorQ<D,Nq,Scalar>& operator/= (const double &alpha);
};
//-----------</definitions>-----------

//-----------<vector arithmetics>-----------
template<size_t D, size_t Nq, typename Scalar>
PivotVectorQ<D,Nq,Scalar>& PivotVectorQ<D,Nq,Scalar>::operator+= (const PivotVectorQ<D,Nq,Scalar> &Vrhs)
{
	for (size_t s=0; s<D; ++s)
	{
		transform(A[s].block.begin(), A[s].block.end(), Vrhs.A[s].block.begin(), A[s].block.begin(), std::plus<MatrixXd>());
	}
	return *this;
}

template<size_t D, size_t Nq, typename Scalar>
PivotVectorQ<D,Nq,Scalar>& PivotVectorQ<D,Nq,Scalar>::operator-= (const PivotVectorQ<D,Nq,Scalar> &Vrhs)
{
	for (size_t s=0; s<D; ++s)
	{
		transform(A[s].block.begin(), A[s].block.end(), Vrhs.A[s].block.begin(), A[s].block.begin(), std::minus<MatrixXd>());
	}
	return *this;
}

template<size_t D, size_t Nq, typename Scalar>
PivotVectorQ<D,Nq,Scalar>& PivotVectorQ<D,Nq,Scalar>::operator*= (const double &alpha)
{
	for (size_t s=0; s<D; ++s)
	for (size_t q=0; q<A[s].dim; ++q)
	{
		A[s].block[q] *= alpha;
	}
	return *this;
}

template<size_t D, size_t Nq, typename Scalar>
PivotVectorQ<D,Nq,Scalar>& PivotVectorQ<D,Nq,Scalar>::operator/= (const double &alpha)
{
	for (size_t s=0; s<D; ++s)
	for (size_t q=0; q<A[s].dim; ++q)
	{
		A[s].block[q] /= alpha;
	}
	return *this;
}

template<size_t D, size_t Nq, typename Scalar>
PivotVectorQ<D,Nq,Scalar> operator* (double const &alpha, PivotVectorQ<D,Nq,Scalar> V)
{
	return V *= alpha;
}

template<size_t D, size_t Nq, typename Scalar>
PivotVectorQ<D,Nq,Scalar> operator* (PivotVectorQ<D,Nq,Scalar> V, double const &alpha)
{
	return V *= alpha;
}

template<size_t D, size_t Nq, typename Scalar>
PivotVectorQ<D,Nq,Scalar> operator/ (PivotVectorQ<D,Nq,Scalar> V, const double &alpha)
{
	return V /= alpha;
}

template<size_t D, size_t Nq, typename Scalar>
PivotVectorQ<D,Nq,Scalar> operator+ (const PivotVectorQ<D,Nq,Scalar> &V1, const PivotVectorQ<D,Nq,Scalar> &V2)
{
	PivotVectorQ<D,Nq,Scalar> Vout = V1;
	Vout += V2;
	return Vout;
}

template<size_t D, size_t Nq, typename Scalar>
PivotVectorQ<D,Nq,Scalar> operator- (const PivotVectorQ<D,Nq,Scalar> &V1, const PivotVectorQ<D,Nq,Scalar> &V2)
{
	PivotVectorQ<D,Nq,Scalar> Vout = V1;
	Vout -= V2;
	return Vout;
}
//-----------</vector arithmetics>-----------

//-----------<matrix*vector>-----------
template<size_t D, size_t Nq, typename Scalar>
void HxV (const PivotMatrixQ<D,Nq,Scalar> &H, const PivotVectorQ<D,Nq,Scalar> &Vin, PivotVectorQ<D,Nq,Scalar> &Vout)
{
	Vout = Vin;
	for (size_t s=0; s<D; ++s) {Vout.A[s].setZero();}
	
	#ifndef DMRG_DONT_USE_OPENMP
	#pragma omp parallel for schedule(dynamic)
	#endif
	for (size_t q=0; q<H.qlhs.size(); ++q)
	{
		size_t s1 = H.qlhs[q][0];
		size_t q1 = H.qlhs[q][1];
		
		for (auto irhs=H.qrhs[q].begin(); irhs!=H.qrhs[q].end(); ++irhs)
		{
			size_t s2 = (*irhs)[0];
			size_t q2 = (*irhs)[1];
			size_t qL = (*irhs)[2];
			size_t qR = (*irhs)[3];
			
			for (int k=0; k<H.W[s1][s2].outerSize(); ++k)
			for (SparseMatrixXd::InnerIterator iW(H.W[s1][s2],k); iW; ++iW)
			{
				if (H.L.block[qL][iW.row()][0].rows() != 0 and 
				    H.R.block[qR][iW.col()][0].rows() != 0)
				{
					if (Vout.A[s1].block[q1].rows() != H.L.block[qL][iW.row()][0].rows() or
					    Vout.A[s1].block[q1].cols() != H.R.block[qR][iW.col()][0].cols())
					{
						Vout.A[s1].block[q1].noalias() = iW.value() * 
						                                 (H.L.block[qL][iW.row()][0] * 
						                                  Vin.A[s2].block[q2] * 
						                                  H.R.block[qR][iW.col()][0]);
					}
					else
					{
						Vout.A[s1].block[q1].noalias() += iW.value() * 
						                                  (H.L.block[qL][iW.row()][0] * 
						                                   Vin.A[s2].block[q2] * 
						                                   H.R.block[qR][iW.col()][0]);
					}
				}
			}
		}
	}
}

//template<size_t D, size_t Nq, typename Scalar>
//void careful_HxV (const PivotMatrixQ<D,Nq,Scalar> &H, const PivotVectorQ<D,Nq,Scalar> &Vin, PivotVectorQ<D,Nq,Scalar> &Vout, std::array<qarray<Nq>,D> qloc)
//{
//	Vout = Vin;
//	for (size_t s=0; s<D; ++s) {Vout.A[s].setZero();}
//	
//	//	for (size_t s1=0; s1<D; ++s1)
////	for (size_t s2=0; s2<D; ++s2)
////	for (size_t qL=0; qL<LW[loc].dim; ++qL)
////	{
////		tuple<qarray3<Nq>,size_t,size_t,size_t> ix;
////		bool FOUND_MATCH = AWA(LW[loc].in(qL), LW[loc].out(qL), LW[loc].mid(qL), s1, s2, O.locBasis(), 
////				               Vbra.A[loc], O.W[loc], Vket.A[loc], ix);
////		if (FOUND_MATCH == true)
////		{
////			size_t q1 = get<1>(ix);
////			size_t qW = get<2>(ix);
////			size_t q2 = get<3>(ix);
////			auto qR = RW[loc].dict.find(get<0>(ix));
////	
////			if (qR != RW[loc].dict.end())
////			{
////				for (int k=0; k<O.W[loc][s1][s2].block[qW].outerSize(); ++k)
////				for (SparseMatrixXd::InnerIterator iW(O.W[loc][s1][s2].block[qW],k); iW; ++iW)
////				{
////					if (LW[loc].block[qL][iW.row()][0].rows()         != 0 and 
////						RW[loc].block[qR->second][iW.col()][0].rows() != 0)
////					{
////						Vbra.A[loc][s1].block[q1].noalias() += iW.value() * (LW[loc].block[qL][iW.row()][0] * 
////						                                                     Vket.A[loc][s2].block[q2] * 
////						                                                     RW[loc].block[qR->second][iW.col()][0]);
////					}
////				}
////			}
////		}
////	}
//	
////	#ifndef DMRG_DONT_USE_OPENMP
////	#pragma omp parallel for
////	#endif
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	for (size_t qL=0; qL<H.L.dim; ++qL)
//	{
//		tuple<qarray3<Nq>,size_t,size_t,size_t> ix;
//		bool FOUND_MATCH = AWA(H.L.in(qL), H.L.out(qL), H.L.mid(qL), s1, s2, qloc, Vout.A, H.W, Vin.A, ix);
//		
//		if (FOUND_MATCH == true)
//		{
//			size_t q1 = get<1>(ix);
//			size_t qW = get<2>(ix);
//			size_t q2 = get<3>(ix);
//			auto   qR = H.R.dict.find(get<0>(ix));
//			
//			if (qR != H.R.dict.end())
//			{
//				for (int k=0; k<H.W[s1][s2].block[qW].outerSize(); ++k)
//				for (SparseMatrixXd::InnerIterator iW(H.W[s1][s2].block[qW],k); iW; ++iW)
//				{
//					if (H.L.block[qL][iW.row()][0].rows() != 0 and 
//						H.R.block[qR->second][iW.col()][0].rows() != 0)
//					{
//						if (Vout.A[s1].block[q1].rows() != H.L.block[qL][iW.row()][0].rows() or
//							Vout.A[s1].block[q1].cols() != H.R.block[qR->second][iW.col()][0].cols())
//						{
//							Vout.A[s1].block[q1] = iW.value() * 
//							                                 (H.L.block[qL][iW.row()][0] * 
//							                                  Vin.A[s2].block[q2] * 
//							                                  H.R.block[qR->second][iW.col()][0]);
//						}
//						else
//						{
//							Vout.A[s1].block[q1] += iW.value() * 
//							                                  (H.L.block[qL][iW.row()][0] * 
//							                                   Vin.A[s2].block[q2] * 
//							                                   H.R.block[qR->second][iW.col()][0]);
//						}
//					}
//				}
//			}
//		}
//	}
//}

template<size_t D, size_t Nq, typename Scalar>
void HxV (const PivotMatrixQ<D,Nq,Scalar> &H, PivotVectorQ<D,Nq,Scalar> &Vinout)
{
	PivotVectorQ<D,Nq,Scalar> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}
//-----------</matrix*vector>-----------

//-----------<dot & vector norms>-----------
template<size_t D, size_t Nq, typename Scalar>
double dot (const PivotVectorQ<D,Nq,Scalar> &V1, const PivotVectorQ<D,Nq,Scalar> &V2)
{
	double res = 0.;
	for (size_t s=0; s<D; ++s)
	for (size_t q=0; q<V2.A[s].dim; ++q)
	for (size_t i=0; i<V2.A[s].block[q].cols(); ++i)
	{
		res += V1.A[s].block[q].col(i).dot(V2.A[s].block[q].col(i));
	}
	return res;
}

template<size_t D, size_t Nq, typename Scalar>
double squaredNorm (const PivotVectorQ<D,Nq,Scalar> &V)
{
	double res = 0.;
	for (size_t s=0; s<D; ++s)
	for (size_t q=0; q<V.A[s].dim; ++q)
	{
		res += V.A[s].block[q].colwise().squaredNorm().sum();
	}
	return res;
}

template<size_t D, size_t Nq, typename Scalar>
inline double norm (const PivotVectorQ<D,Nq,Scalar> &V)
{
	return sqrt(squaredNorm(V));
}

template<size_t D, size_t Nq, typename Scalar>
inline void normalize (PivotVectorQ<D,Nq,Scalar> &V)
{
	V /= norm(V);
}

template<size_t D, size_t Nq, typename Scalar>
double infNorm (const PivotVectorQ<D,Nq,Scalar> &V1, const PivotVectorQ<D,Nq,Scalar> &V2)
{
	double res = 0.;
	for (size_t s=0; s<D; ++s)
	for (size_t q=0; q<V1.A[s].dim; ++q)
	{
		double tmp = (V1.A[s].block[q]-V2.A[s].block[q]).template lpNorm<Eigen::Infinity>();
		if (tmp>res) {res = tmp;}
	}
	return res;
}
//-----------</dot & vector norms>-----------

//-----------<miscellaneous>-----------
template<size_t D, size_t Nq, typename Scalar>
inline size_t dim (const PivotMatrixQ<D,Nq,Scalar> &H)
{
	return H.dim;
}

template<size_t D, size_t Nq, typename Scalar>
inline double norm (const PivotMatrixQ<D,Nq,Scalar> &H)
{
	return H.dim;
}

template<size_t D, size_t Nq, typename Scalar>
void swap (PivotVectorQ<D,Nq,Scalar> &V1, PivotVectorQ<D,Nq,Scalar> &V2)
{
	for (size_t s=0; s<D; ++s)
	for (size_t q=0; q<V1.A[s].dim; ++q)
	{
		V1.A[s].block[q].swap(V2.A[s].block[q]);
	}
}

#include "LanczosWrappers.h"
#include "RandomVector.h"

template<size_t D, size_t Nq, typename Scalar>
struct GaussianRandomVector<PivotVectorQ<D,Nq,Scalar>,double>
{
	static void fill (size_t N, PivotVectorQ<D,Nq,Scalar> &Vout)
	{
		for (size_t s=0; s<D; ++s)
		for (size_t q=0; q<Vout.A[s].dim; ++q)
		for (size_t a1=0; a1<Vout.A[s].block[q].rows(); ++a1)
		for (size_t a2=0; a2<Vout.A[s].block[q].cols(); ++a2)
		{
			Vout.A[s].block[q](a1,a2) = NormDist(MtEngine);
		}
		normalize(Vout);
	}
};
//-----------</miscellaneous>-----------

#endif
