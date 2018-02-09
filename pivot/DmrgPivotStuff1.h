#ifndef STRAWBERRY_DMRGHEFFSTUFF_WITH_Q
#define STRAWBERRY_DMRGHEFFSTUFF_WITH_Q

#include "DmrgTypedefs.h"
#include "tensors/Biped.h"
#include "tensors/Multipede.h"

//-----------<definitions>-----------
template<typename Symmetry, typename Scalar, typename MpoScalar=double>
struct PivotMatrix
{
	static constexpr std::size_t Nq = Symmetry::Nq;
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > L;
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > R;
	vector<vector<vector<SparseMatrix<MpoScalar> > > > W;
	
	vector<std::array<size_t,2> >          qlhs;
	vector<vector<std::array<size_t,5> > > qrhs;
	vector<vector<Scalar> > factor_cgcs;
	
	vector<qarray<Nq> > qloc;
	
	// stuff for excited states
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > PL; // PL[n]
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > PR; // PL[n]
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > A0; // A0[n][s]
	double Epenalty = 0;
};

template<typename Symmetry, typename Scalar>
struct PivotVector1
{
	static constexpr std::size_t Nq = Symmetry::Nq;
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > A;
	
	PivotVector1(){};
	
	PivotVector1 (const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Arhs)
	:A(Arhs)
	{}
	
	/**Set blocks as in Vrhs, but do not resize the matrices*/
	void outerResize (const PivotVector1 &Vrhs)
	{
		A.clear();
		A.resize(Vrhs.A.size());
		for (size_t i=0; i<A.size(); ++i)
		{
			A[i].in = Vrhs.A[i].in;
			A[i].out = Vrhs.A[i].out;
			A[i].dict = Vrhs.A[i].dict;
			A[i].block.resize(Vrhs.A[i].block.size());
			A[i].dim = Vrhs.A[i].dim;
		}
	}
	
	PivotVector1<Symmetry,Scalar>& operator+= (const PivotVector1<Symmetry,Scalar> &Vrhs);
	PivotVector1<Symmetry,Scalar>& operator-= (const PivotVector1<Symmetry,Scalar> &Vrhs);
	template<typename OtherScalar> PivotVector1<Symmetry,Scalar>& operator*= (const OtherScalar &alpha);
	template<typename OtherScalar> PivotVector1<Symmetry,Scalar>& operator/= (const OtherScalar &alpha);
};
//-----------</definitions>-----------

//-----------<vector arithmetics>-----------
template<typename Symmetry, typename Scalar>
PivotVector1<Symmetry,Scalar>& PivotVector1<Symmetry,Scalar>::operator+= (const PivotVector1<Symmetry,Scalar> &Vrhs)
{
	for (std::size_t s=0; s<A.size(); s++)
	{
		A[s] = A[s] + Vrhs.A[s];
	}
	return *this;
}

template<typename Symmetry, typename Scalar>
PivotVector1<Symmetry,Scalar>& PivotVector1<Symmetry,Scalar>::
operator-= (const PivotVector1<Symmetry,Scalar> &Vrhs)
{
	for (std::size_t s=0; s<A.size(); s++)
	{
		A[s] = A[s] - Vrhs.A[s];
	}
	return *this;
}

template<typename Symmetry, typename Scalar>
template<typename OtherScalar>
PivotVector1<Symmetry,Scalar>& PivotVector1<Symmetry,Scalar>::
operator*= (const OtherScalar &alpha)
{
	for (size_t s=0; s<A.size(); ++s)
	for (size_t q=0; q<A[s].dim; ++q)
	{
		A[s].block[q] *= alpha;
	}
	return *this;
}

template<typename Symmetry, typename Scalar>
template<typename OtherScalar>
PivotVector1<Symmetry,Scalar>& PivotVector1<Symmetry,Scalar>::
operator/= (const OtherScalar &alpha)
{
	for (size_t s=0; s<A.size(); ++s)
	for (size_t q=0; q<A[s].dim; ++q)
	{
		A[s].block[q] /= alpha;
	}
	return *this;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
PivotVector1<Symmetry,Scalar> operator* (const OtherScalar &alpha, PivotVector1<Symmetry,Scalar> V)
{
	return V *= alpha;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
PivotVector1<Symmetry,Scalar> operator* (PivotVector1<Symmetry,Scalar> V, const OtherScalar &alpha)
{
	return V *= alpha;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
PivotVector1<Symmetry,Scalar> operator/ (PivotVector1<Symmetry,Scalar> V, const OtherScalar &alpha)
{
	return V /= alpha;
}

template<typename Symmetry, typename Scalar>
PivotVector1<Symmetry,Scalar> operator+ (const PivotVector1<Symmetry,Scalar> &V1, const PivotVector1<Symmetry,Scalar> &V2)
{
	PivotVector1<Symmetry,Scalar> Vout = V1;
	Vout += V2;
	return Vout;
}

template<typename Symmetry, typename Scalar>
PivotVector1<Symmetry,Scalar> operator- (const PivotVector1<Symmetry,Scalar> &V1, const PivotVector1<Symmetry,Scalar> &V2)
{
	PivotVector1<Symmetry,Scalar> Vout = V1;
	Vout -= V2;
	return Vout;
}
//-----------</vector arithmetics>-----------

//-----------<matrix*vector>-----------
template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivotMatrix<Symmetry,Scalar,MpoScalar> &H, const PivotVector1<Symmetry,Scalar> &Vin, PivotVector1<Symmetry,Scalar> &Vout)
{
	Vout = Vin;
	for (size_t s=0; s<Vout.A.size(); ++s) {Vout.A[s].setZero();}
	
	#ifndef DMRG_DONT_USE_OPENMP
	#pragma omp parallel for schedule(dynamic)
	#endif
	for (size_t q=0; q<H.qlhs.size(); ++q)
	{
		size_t s1 = H.qlhs[q][0];
		size_t q1 = H.qlhs[q][1];
		for (size_t p=0; p<H.qrhs[q].size(); ++p)
		// for (auto irhs=H.qrhs[q].begin(); irhs!=H.qrhs[q].end(); ++irhs)
		{
			size_t s2 = H.qrhs[q][p][0];
			size_t q2 = H.qrhs[q][p][1];
			size_t qL = H.qrhs[q][p][2];
			size_t qR = H.qrhs[q][p][3];
			size_t k = H.qrhs[q][p][4];
			for (int r=0; r<H.W[s1][s2][k].outerSize(); ++r)
			for (typename SparseMatrix<MpoScalar>::InnerIterator iW(H.W[s1][s2][k],r); iW; ++iW)
			{
				if (H.L.block[qL][iW.row()][0].rows() != 0 and 
				    H.R.block[qR][iW.col()][0].rows() != 0)
				{
					if (Vout.A[s1].block[q1].rows() != H.L.block[qL][iW.row()][0].rows() or
					    Vout.A[s1].block[q1].cols() != H.R.block[qR][iW.col()][0].cols())
					{
						Vout.A[s1].block[q1].noalias() = H.factor_cgcs[q][p] * iW.value() * 
						                                 (H.L.block[qL][iW.row()][0] * 
						                                  Vin.A[s2].block[q2] * 
						                                  H.R.block[qR][iW.col()][0]);
					}
					else
					{
						Vout.A[s1].block[q1].noalias() += H.factor_cgcs[q][p] * iW.value() * 
						                                  (H.L.block[qL][iW.row()][0] * 
						                                   Vin.A[s2].block[q2] * 
						                                   H.R.block[qR][iW.col()][0]);
					}
				}
			}
		}
	}
	
	// project out unwanted states (e.g. to get lower spectrum)
	for (size_t n=0; n<H.A0.size(); ++n)
	{
		Scalar overlap = 0;
		
		for (size_t s=0; s<Vout.A.size(); ++s)
		{
			overlap += (H.PL[n].adjoint() * Vin.A[s] * H.PR[n].adjoint() * H.A0[n][s].adjoint()).block[0].trace();
		}
		
		for (size_t s=0; s<Vout.A.size(); ++s)
		for (size_t qPL=0; qPL<H.PL[n].dim; ++qPL)
		for (size_t qPR=0; qPR<H.PR[n].dim; ++qPR)
		{
			qarray2<Symmetry::Nq> qupleA = {H.PL[n].in[qPL], H.PR[n].out[qPR]};
			auto qA = Vout.A[s].dict.find(qupleA);
		
			qarray2<Symmetry::Nq> qupleA0 = {H.PL[n].out[qPL], H.PR[n].in[qPR]};
			auto qA0 = H.A0[n][s].dict.find(qupleA0);
		
			if (H.PL[n].out[qPL] + H.qloc[s] == H.PR[n].in[qPR] and
				qA0 != H.A0[n][s].dict.end() and
				qA != Vout.A[s].dict.end())
			{
				Vout.A[s].block[qA->second] += overlap * H.Epenalty * H.PL[n].block[qPL] * H.A0[n][s].block[qA0->second] * H.PR[n].block[qPR];
			}
		}
	}
}

//template<typename Symmetry, typename Scalar, typename MpoScalar>
//void careful_HxV (const PivotMatrix<Symmetry,Scalar,MpoScalar> &H, const PivotVector1<Symmetry,Scalar> &Vin, PivotVector1<Symmetry,Scalar> &Vout, std::array<qarray<Nq>,D> qloc)
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

template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivotMatrix<Symmetry,Scalar,MpoScalar> &H, PivotVector1<Symmetry,Scalar> &Vinout)
{
	PivotVector1<Symmetry,Scalar> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}
//-----------</matrix*vector>-----------

//-----------<dot & vector norms>-----------
template<typename Symmetry, typename Scalar>
Scalar dot (const PivotVector1<Symmetry,Scalar> &V1, const PivotVector1<Symmetry,Scalar> &V2)
{
	Biped<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > out = V1.A[0].adjoint().contract(V2.A[0]);
	for (std::size_t s=1; s<V1.A.size(); s++)
	{
		out += V1.A[s].adjoint().contract(V2.A[s]);
	}
	Scalar res = out.trace();
	return res;
}

template<typename Symmetry, typename Scalar>
double squaredNorm (const PivotVector1<Symmetry,Scalar> &V)
{
	double res = isReal(dot(V,V));
	return res;
}

template<typename Symmetry, typename Scalar>
inline double norm (const PivotVector1<Symmetry,Scalar> &V)
{
	return sqrt(squaredNorm(V));
}

template<typename Symmetry, typename Scalar>
inline void normalize (PivotVector1<Symmetry,Scalar> &V)
{
	V /= norm(V);
}

template<typename Symmetry, typename Scalar>
double infNorm (const PivotVector1<Symmetry,Scalar> &V1, const PivotVector1<Symmetry,Scalar> &V2)
{
	double res = 0.;
	for (size_t s=0; s<V1.A.size(); ++s)
	{
		auto Mtmp = V1.A[s] - V2.A[s];
		for (size_t q=0; q<Mtmp.dim; ++q)
		{
			double tmp = Mtmp.block[q].template lpNorm<Eigen::Infinity>();
			if (tmp>res) {res = tmp;}
		}
	}
	return res;
}

template<typename Symmetry, typename Scalar>
inline size_t dim (const PivotVector1<Symmetry,Scalar> &V)
{
	size_t out = 0;
	for (size_t s=0; s<V.A.size(); ++s)
	for (size_t q=0; q<V.A[s].dim; ++q)
	{
		out += V.A[s].block[q].size();
	}
	return out;
}
//-----------</dot & vector norms>-----------

//-----------<miscellaneous>-----------
template<typename Symmetry, typename Scalar, typename MpoScalar>
inline size_t dim (const PivotMatrix<Symmetry,Scalar,MpoScalar> &H)
{
	return 0;
}

// How to calculate the Frobenius norm of this?
template<typename Symmetry, typename Scalar, typename MpoScalar>
inline double norm (const PivotMatrix<Symmetry,Scalar,MpoScalar> &H)
{
	return H.dim;
}

template<typename Symmetry, typename Scalar>
void swap (PivotVector1<Symmetry,Scalar> &V1, PivotVector1<Symmetry,Scalar> &V2)
{
	for (size_t s=0; s<V1.A.size(); ++s)
	{
		V1.A[s].block.swap(V2.A[s].block);
	}
}

#include "RandomVector.h"

template<typename Symmetry, typename Scalar>
struct GaussianRandomVector<PivotVector1<Symmetry,Scalar>,Scalar>
{
	static void fill (size_t N, PivotVector1<Symmetry,Scalar> &Vout)
	{
		for (size_t s=0; s<Vout.A.size(); ++s)
		for (size_t q=0; q<Vout.A[s].dim; ++q)
		for (size_t a1=0; a1<Vout.A[s].block[q].rows(); ++a1)
		for (size_t a2=0; a2<Vout.A[s].block[q].cols(); ++a2)
		{
			Vout.A[s].block[q](a1,a2) = threadSafeRandUniform<Scalar>(-1.,1.);
		}
		normalize(Vout);
	}
};
//-----------</miscellaneous>-----------

#endif
