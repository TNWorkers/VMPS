#ifndef STRAWBERRY_DMRG_HEFF_STUFF_2SITE_WITH_Q
#define STRAWBERRY_DMRG_HEFF_STUFF_2SITE_WITH_Q

#include <vector>
#include "DmrgTypedefs.h"
#include "tensors/Biped.h"
#include "tensors/Multipede.h"
#include "Mps.h"

//-----------<contractions>-----------
template<typename Symmetry, typename Scalar>
void contract_AA (const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &A1, 
                  vector<qarray<Symmetry::Nq> > qloc1, 
                  const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &A2, 
                  vector<qarray<Symmetry::Nq> > qloc2, 
                  vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Apair)
{
//	auto index = [&qloc2] (size_t s1, size_t s2) -> size_t {return s1*qloc2.size()+s2;};
	
	auto tensor_basis = Symmetry::tensorProd(qloc1,qloc2);
	Apair.resize(tensor_basis.size());
	
	for (size_t s1=0; s1<qloc1.size(); ++s1)
	for (size_t s2=0; s2<qloc2.size(); ++s2)
	{
		auto qmerges = Symmetry::reduceSilent(qloc1[s1], qloc2[s2]);
		
		for (const auto &qmerge:qmerges)
		{
			auto qtensor = make_tuple(qloc1[s1], s1, qloc2[s2], s2, qmerge);
			auto s1s2 = distance(tensor_basis.begin(), find(tensor_basis.begin(), tensor_basis.end(), qtensor));
			
			for (size_t q1=0; q1<A1[s1].dim; ++q1)
			{
				auto qmids = Symmetry::reduceSilent(A1[s1].out[q1], qloc2[s2]);
				
				for (const auto &qmid:qmids)
				{
					qarray2<Symmetry::Nq> quple = {A1[s1].out[q1], qmid};
					auto q2 = A2[s2].dict.find(quple);
					
					if (q2 != A2[s2].dict.end())
					{
						Scalar factor_cgc = Symmetry::coeff_Apair(A2[s2].out[q2->second], qloc1[s1], A1[s1].out[q1], 
						                                          qloc2[s2], A1[s1].in[q1], qmerge);
						Matrix<Scalar,Dynamic,Dynamic> Mtmp = factor_cgc * A1[s1].block[q1] * A2[s2].block[q2->second];
						
						qarray2<Symmetry::Nq> qupleApair = {A1[s1].in[q1], A2[s2].out[q2->second]};
						
						auto qApair = Apair[s1s2].dict.find(qupleApair);
						
						if (qApair != Apair[s1s2].dict.end())
						{
							Apair[s1s2].block[qApair->second] += Mtmp;
						}
						else
						{
							Apair[s1s2].push_back(qupleApair, Mtmp);
						}
					}
				}
			}
		}
	}
}

/**for VUMPS 4-site unit cell*/
template<typename Symmetry, typename Scalar>
void contract_AAAA (const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &A1, 
                    vector<qarray<Symmetry::Nq> > qloc1, 
                    const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &A2, 
                    vector<qarray<Symmetry::Nq> > qloc2, 
                    const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &A3, 
                    vector<qarray<Symmetry::Nq> > qloc3, 
                    const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &A4, 
                    vector<qarray<Symmetry::Nq> > qloc4, 
                    boost::multi_array<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> >,4> &Aquartett)
{
	Aquartett.resize(boost::extents[qloc1.size()][qloc2.size()][qloc3.size()][qloc4.size()]);
	
	for (size_t s1=0; s1<qloc1.size(); ++s1)
	for (size_t s2=0; s2<qloc2.size(); ++s2)
	for (size_t s3=0; s3<qloc3.size(); ++s3)
	for (size_t s4=0; s4<qloc4.size(); ++s4)
	for (size_t q1=0; q1<A1[s1].dim; ++q1)
	{
		qarray2<Symmetry::Nq> quple2 = {A1[s1].out[q1], A1[s1].out[q1]+qloc2[s2]};
		auto q2 = A2[s2].dict.find(quple2);
		
		if (q2 != A2[s2].dict.end())
		{
			qarray2<Symmetry::Nq> quple3 = {A2[s2].out[q2->second], A2[s2].out[q2->second]+qloc3[s3]};
			auto q3 = A3[s3].dict.find(quple3);
			
			if (q3 != A3[s3].dict.end())
			{
				qarray2<Symmetry::Nq> quple4 = {A3[s3].out[q3->second], A3[s3].out[q3->second]+qloc4[s4]};
				auto q4 = A4[s4].dict.find(quple4);
				
				if (q4 != A4[s4].dict.end())
				{
					Matrix<Scalar,Dynamic,Dynamic> Mtmp = A1[s1].block[q1] * 
					                                      A2[s2].block[q2->second] * 
					                                      A3[s3].block[q3->second] * 
					                                      A4[s4].block[q4->second];
					
					qarray2<Symmetry::Nq> qupleAquartett = {A1[s1].in[q1], A4[s4].out[q4->second]};
					auto qAquartett = Aquartett[s1][s2][s3][s4].dict.find(qupleAquartett);
					
					if (qAquartett != Aquartett[s1][s2][s3][s4].dict.end())
					{
						Aquartett[s1][s2][s3][s4].block[qAquartett->second] += Mtmp;
					}
					else
					{
						Aquartett[s1][s2][s3][s4].push_back(qupleAquartett, Mtmp);
					}
				}
			}
		}
	}
}
//-----------</contractions>-----------

//-----------<definitions>-----------
template<typename Symmetry, typename Scalar>
struct PivotVector2
{
	PivotVector2(){};
	
	/**Make contraction of two Bipeds.*/
	PivotVector2 (const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &A12,
	              const vector<qarray<Symmetry::Nq> > &qloc12,
	              const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &A34,
	              const vector<qarray<Symmetry::Nq> > &qloc34)
	{
		D12 = qloc12.size();
		D34 = qloc34.size();
		contract_AA(A12, qloc12, A34, qloc34, A);
	}
	
	/**Set blocks as in Vrhs, but do not resize the matrices*/
	void outerResize (const PivotVector2 &Vrhs)
	{
		D12 = Vrhs.D12;
		D34 = Vrhs.D34;
		
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
	
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > A;
	
	size_t D12;
	size_t D34;
	
	PivotVector2<Symmetry,Scalar>& operator+= (const PivotVector2<Symmetry,Scalar> &Vrhs);
	PivotVector2<Symmetry,Scalar>& operator-= (const PivotVector2<Symmetry,Scalar> &Vrhs);
	template<typename OtherScalar> PivotVector2<Symmetry,Scalar>& operator*= (const OtherScalar &alpha);
	template<typename OtherScalar> PivotVector2<Symmetry,Scalar>& operator/= (const OtherScalar &alpha);
	
	size_t index (size_t s1, size_t s3) const
	{
		return s1*D34+s3;
	}
};

template<typename Symmetry, typename Scalar, typename MpoScalar=double>
struct PivotMatrix2
{
	PivotMatrix2 (const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &L_input, 
	              const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &R_input, 
	              const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W12_input, 
	              const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W34_input, 
	              const vector<qarray<Symmetry::Nq> > &qloc12_input, 
	              const vector<qarray<Symmetry::Nq> > &qloc34_input, 
	              const vector<qarray<Symmetry::Nq> > &qOp12_input, 
	              const vector<qarray<Symmetry::Nq> > &qOp34_input)
	:L(L_input), R(R_input), W12(W12_input), W34(W34_input), 
	qloc12(qloc12_input), qloc34(qloc34_input), qOp12(qOp12_input), qOp34(qOp34_input)
	{}
	
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > L;
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > R;
	vector<vector<vector<SparseMatrix<MpoScalar> > > > W12;
	vector<vector<vector<SparseMatrix<MpoScalar> > > > W34;
	
	vector<qarray<Symmetry::Nq> > qloc12;
	vector<qarray<Symmetry::Nq> > qloc34;
	vector<qarray<Symmetry::Nq> > qOp12;
	vector<qarray<Symmetry::Nq> > qOp34;
};
//-----------</definitions>-----------

//-----------<vector arithmetics>-----------
template<typename Symmetry, typename Scalar>
PivotVector2<Symmetry,Scalar>& PivotVector2<Symmetry,Scalar>::operator+= (const PivotVector2<Symmetry,Scalar> &Vrhs)
{
	for (size_t s1=0; s1<D12; ++s1)
	for (size_t s3=0; s3<D34; ++s3)
	{
		transform(A[index(s1,s3)].block.begin(), A[index(s1,s3)].block.end(), 
		          Vrhs.A[index(s1,s3)].block.begin(), A[index(s1,s3)].block.begin(), 
		          std::plus<Matrix<Scalar,Dynamic,Dynamic> >());
	}
	return *this;
}

template<typename Symmetry, typename Scalar>
PivotVector2<Symmetry,Scalar>& PivotVector2<Symmetry,Scalar>::
operator-= (const PivotVector2<Symmetry,Scalar> &Vrhs)
{
	for (size_t s1=0; s1<D12; ++s1)
	for (size_t s3=0; s3<D34; ++s3)
	{
		transform(A[index(s1,s3)].block.begin(), A[index(s1,s3)].block.end(), 
		          Vrhs.A[index(s1,s3)].block.begin(), A[index(s1,s3)].block.begin(), 
		          std::minus<Matrix<Scalar,Dynamic,Dynamic> >());
	}
	return *this;
}

template<typename Symmetry, typename Scalar>
template<typename OtherScalar>
PivotVector2<Symmetry,Scalar>& PivotVector2<Symmetry,Scalar>::
operator*= (const OtherScalar &alpha)
{
	for (size_t s1=0; s1<D12; ++s1)
	for (size_t s3=0; s3<D34; ++s3)
	for (size_t q=0; q<A[index(s1,s3)].dim; ++q)
	{
		A[index(s1,s3)].block[q] *= alpha;
	}
	return *this;
}

template<typename Symmetry, typename Scalar>
template<typename OtherScalar>
PivotVector2<Symmetry,Scalar>& PivotVector2<Symmetry,Scalar>::
operator/= (const OtherScalar &alpha)
{
	for (size_t s1=0; s1<D12; ++s1)
	for (size_t s3=0; s3<D34; ++s3)
	for (size_t q=0; q<A[index(s1,s3)].dim; ++q)
	{
		A[index(s1,s3)].block[q] /= alpha;
	}
	return *this;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
PivotVector2<Symmetry,Scalar> operator* (const OtherScalar &alpha, PivotVector2<Symmetry,Scalar> V)
{
	return V *= alpha;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
PivotVector2<Symmetry,Scalar> operator* (PivotVector2<Symmetry,Scalar> V, const OtherScalar &alpha)
{
	return V *= alpha;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
PivotVector2<Symmetry,Scalar> operator/ (PivotVector2<Symmetry,Scalar> V, const OtherScalar &alpha)
{
	return V /= alpha;
}

template<typename Symmetry, typename Scalar>
PivotVector2<Symmetry,Scalar> operator+ (const PivotVector2<Symmetry,Scalar> &V1, const PivotVector2<Symmetry,Scalar> &V2)
{
	PivotVector2<Symmetry,Scalar> Vout = V1;
	Vout += V2;
	return Vout;
}

template<typename Symmetry, typename Scalar>
PivotVector2<Symmetry,Scalar> operator- (const PivotVector2<Symmetry,Scalar> &V1, const PivotVector2<Symmetry,Scalar> &V2)
{
	PivotVector2<Symmetry,Scalar> Vout = V1;
	Vout -= V2;
	return Vout;
}
//-----------</vector arithmetics>-----------

//-----------<matrix*vector>-----------
template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivotMatrix2<Symmetry,Scalar,MpoScalar> &H, const PivotVector2<Symmetry,Scalar> &Vin, PivotVector2<Symmetry,Scalar> &Vout)
{
	Vout.outerResize(Vin); // set block structure of Vout as in Vin
	
	for (size_t s1=0; s1<H.qloc12.size(); ++s1)
	for (size_t s2=0; s2<H.qloc12.size(); ++s2)
	for (size_t k12=0; k12<H.qOp12.size(); ++k12)
	{
		std::array<typename Symmetry::qType,3> qCheck12 = {H.qloc12[s2],H.qOp12[k12],H.qloc12[s1]};
		if (!Symmetry::validate(qCheck12)) {continue;}
		
		for (size_t s3=0; s3<H.qloc34.size(); ++s3)
		for (size_t s4=0; s4<H.qloc34.size(); ++s4)
		for (size_t k34=0; k34<H.qOp34.size(); ++k34)
		{
			std::array<typename Symmetry::qType,3> qCheck34 = {H.qloc34[s4],H.qOp34[k34],H.qloc34[s3]};
			if (!Symmetry::validate(qCheck34)) {continue;}
			
			for (size_t qL=0; qL<H.L.dim; ++qL)
			{
				vector<tuple<qarray3<Symmetry::Nq>,size_t,size_t> > ixs;
				bool FOUND_MATCH = AAWWAA(H.L.in(qL), H.L.out(qL), H.L.mid(qL), 
				                          s1, s2, H.qloc12, k12, H.qOp12, 
				                          s3, s4, H.qloc34, k34, H.qOp34,
				                          Vout.A, Vin.A, ixs);
				
				if (FOUND_MATCH)
				{
					for (const auto& ix:ixs)
					{
						auto qR = H.R.dict.find(get<0>(ix));
						size_t qA13 = get<1>(ix);
						size_t qA24 = get<2>(ix);
						
						if (qR != H.R.dict.end())
						{
							for (int r12=0; r12<H.W12[s1][s2][k12].outerSize(); ++r12)
							for (typename SparseMatrix<MpoScalar>::InnerIterator iW12(H.W12[s1][s2][k12],r12); iW12; ++iW12)
							for (int r34=0; r34<H.W34[s3][s4][k34].outerSize(); ++r34)
							for (typename SparseMatrix<MpoScalar>::InnerIterator iW34(H.W34[s3][s4][k34],r34); iW34; ++iW34)
							{
								Matrix<Scalar,Dynamic,Dynamic> Mtmp;
								MpoScalar Wfactor = iW12.value() * iW34.value();
								
								if (H.L.block[qL][iW12.row()][0].rows() != 0 and
									H.R.block[qR->second][iW34.col()][0].rows() !=0 and
									iW12.col() == iW34.row())
								{
									optimal_multiply(Wfactor, 
									                 H.L.block[qL][iW12.row()][0],
									                 Vin.A[Vin.index(s2,s4)].block[qA24],
									                 H.R.block[qR->second][iW34.col()][0],
									                 Mtmp);
								}
								
								if (Mtmp.rows() != 0)
								{
									if (Vout.A[Vout.index(s1,s3)].block[qA13].rows() != 0)
									{
										Vout.A[Vout.index(s1,s3)].block[qA13] += Mtmp;
									}
									else
									{
										Vout.A[Vout.index(s1,s3)].block[qA13] = Mtmp;
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivotMatrix2<Symmetry,Scalar,MpoScalar> &H, PivotVector2<Symmetry,Scalar> &Vinout)
{
	PivotVector2<Symmetry,Scalar> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}
//-----------</matrix*vector>-----------

//-----------<dot & vector norms>-----------
template<typename Symmetry, typename Scalar>
Scalar dot (const PivotVector2<Symmetry,Scalar> &V1, const PivotVector2<Symmetry,Scalar> &V2)
{
	Scalar res = 0.;
	for (size_t s1=0; s1<V2.D12; ++s1)
	for (size_t s3=0; s3<V2.D34; ++s3)
	for (size_t q=0; q<V2.A[V2.index(s1,s3)].dim; ++q)
	{
		res += (V1.A[V1.index(s1,s3)].block[q].adjoint() * V2.A[V2.index(s1,s3)].block[q]).trace();
	}
	return res;
}

template<typename Symmetry, typename Scalar>
double squaredNorm (const PivotVector2<Symmetry,Scalar> &V)
{
	return isReal(dot(V,V));
}

template<typename Symmetry, typename Scalar>
inline double norm (const PivotVector2<Symmetry,Scalar> &V)
{
	return sqrt(squaredNorm(V));
}

template<typename Symmetry, typename Scalar>
inline void normalize (PivotVector2<Symmetry,Scalar> &V)
{
	V /= norm(V);
}

template<typename Symmetry, typename Scalar>
double infNorm (const PivotVector2<Symmetry,Scalar> &V1, const PivotVector2<Symmetry,Scalar> &V2)
{
	double res = 0.;
	for (size_t s1=0; s1<V2.A.size(); ++s1)
	for (size_t s3=0; s3<V2.A[s1].size(); ++s3)
	for (size_t q=0; q<V1.A[V1.index(s1,s3)].dim; ++q)
	{
		double tmp = (V1.A[V1.index(s1,s3)].block[q]-V2.A[V2.index(s1,s3)].block[q]).template lpNorm<Eigen::Infinity>();
		if (tmp>res) {res = tmp;}
	}
	return res;
}

template<typename Symmetry, typename Scalar>
inline size_t dim (const PivotVector2<Symmetry,Scalar> &V)
{
	size_t out = 0;
	for (size_t s1=0; s1<V.D12; ++s1)
	for (size_t s3=0; s3<V.D34; ++s3)
	for (size_t q=0; q<V.A[V.index(s1,s3)].dim; ++q)
	{
		out += V.A[V.index(s1,s3)].block[q].size();
	}
	return out;
}
//-----------</dot & vector norms>-----------

//-----------<miscellaneous>-----------
template<typename Symmetry, typename Scalar, typename MpoScalar>
inline size_t dim (const PivotMatrix2<Symmetry,Scalar,MpoScalar> &H)
{
	return 0;
}

// How to calculate the Frobenius norm of this?
template<typename Symmetry, typename Scalar, typename MpoScalar>
inline double norm (const PivotMatrix2<Symmetry,Scalar,MpoScalar> &H)
{
	return 0;
}

template<typename Symmetry, typename Scalar>
void swap (PivotVector2<Symmetry,Scalar> &V1, PivotVector2<Symmetry,Scalar> &V2)
{
	for (size_t s1=0; s1<V2.A.size(); ++s1)
	for (size_t s3=0; s3<V2.A[s1].size(); ++s3)
	for (size_t q=0; q<V1.A[V1.index(s1,s3)].dim; ++q)
	{
		V1.A[V1.index(s1,s3)].block[q].swap(V2.A[V2.index(s1,s3)].block[q]);
	}
}

#include "RandomVector.h"

template<typename Symmetry, typename Scalar>
struct GaussianRandomVector<PivotVector2<Symmetry,Scalar>,Scalar>
{
	static void fill (size_t N, PivotVector2<Symmetry,Scalar> &Vout)
	{
		for (size_t s1=0; s1<Vout.A.size(); ++s1)
		for (size_t s3=0; s3<Vout.A[s1].size(); ++s3)
		for (size_t q=0; q<Vout.A[Vout.index(s1,s3)].dim; ++q)
		for (size_t a1=0; a1<Vout.A[Vout.index(s1,s3)].block[q].rows(); ++a1)
		for (size_t a2=0; a2<Vout.A[Vout.index(s1,s3)].block[q].cols(); ++a2)
		{
			Vout.A[Vout.index(s1,s3)].block[q](a1,a2) = threadSafeRandUniform<Scalar>(-1.,1.);
		}
		normalize(Vout);
	}
};
//-----------</miscellaneous>-----------

//template<typename Symmetry, typename Scalar, typename MpoScalar>
//void HxV (const PivotMatrix<Symmetry,Scalar,MpoScalar> &H1, 
//          const PivotMatrix<Symmetry,Scalar,MpoScalar> &H2, 
//          const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Aket1, 
//          const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Aket2, 
//          const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Abra1, 
//          const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Abra2, 
//          const vector<qarray<Symmetry::Nq> > &qloc1, const vector<qarray<Symmetry::Nq> > &qloc2, 
//          vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > &Apair)
//{
//	Apair.resize(qloc1.size());
//	for (size_t s1=0; s1<qloc1.size(); ++s1)
//	{
//		Apair[s1].resize(qloc2.size());
//	}
//	
//	for (size_t s1=0; s1<qloc1.size(); ++s1)
//	for (size_t s2=0; s2<qloc1.size(); ++s2)
//	for (size_t qL=0; qL<H1.L.dim; ++qL)
//	{
//		tuple<qarray3<Symmetry::Nq>,size_t,size_t> ix12;
//		bool FOUND_MATCH12 = AWA(H1.L.in(qL), H1.L.out(qL), H1.L.mid(qL), s1, s2, qloc1, Abra1, Aket1, ix12);
//		
//		if (FOUND_MATCH12)
//		{
//			qarray3<Symmetry::Nq> quple12 = get<0>(ix12);
//			swap(quple12[0], quple12[1]);
//			size_t qA12 = get<2>(ix12);
//			
//			for (size_t s3=0; s3<qloc2.size(); ++s3)
//			for (size_t s4=0; s4<qloc2.size(); ++s4)
//			{
//				tuple<qarray3<Symmetry::Nq>,size_t,size_t> ix34;
//				bool FOUND_MATCH34 = AWA(quple12[0], quple12[1], quple12[2], s3, s4, qloc2, Abra2, Aket2, ix34);
//				
//				if (FOUND_MATCH34)
//				{
//					qarray3<Symmetry::Nq> quple34 = get<0>(ix34);
//					size_t qA34 = get<2>(ix34);
//					auto qR = H2.R.dict.find(quple34);
//					
//					if (qR != H2.R.dict.end())
//					{
//						if (H1.L.mid(qL) + qloc1[s1] - qloc1[s2] == 
//						    H2.R.mid(qR->second) - qloc2[s3] + qloc2[s4])
//						{
//							for (int k12=0; k12<H1.W[s1][s2][0].outerSize(); ++k12)
//							for (typename SparseMatrix<MpoScalar>::InnerIterator iW12(H1.W[s1][s2][0],k12); iW12; ++iW12)
//							for (int k34=0; k34<H2.W[s3][s4][0].outerSize(); ++k34)
//							for (typename SparseMatrix<MpoScalar>::InnerIterator iW34(H2.W[s3][s4][0],k34); iW34; ++iW34)
//							{
//								Matrix<Scalar,Dynamic,Dynamic> Mtmp;
//								MpoScalar Wfactor = iW12.value() * iW34.value();
//								
//								if (H1.L.block[qL][iW12.row()][0].rows() != 0 and
//									H2.R.block[qR->second][iW34.col()][0].rows() !=0 and
//									iW12.col() == iW34.row())
//								{
////									Mtmp = Wfactor * 
////									       (H1.L.block[qL][iW12.row()][0] * 
////									       Aket1[loc1][s2].block[qA12] * 
////									       Aket2[s4].block[qA34] * 
////									       H2.R.block[qR->second][iW34.col()][0]);
//									optimal_multiply(Wfactor, 
//									                 H1.L.block[qL][iW12.row()][0],
//									                 Aket1[s2].block[qA12],
//									                 Aket2[s4].block[qA34],
//									                 H2.R.block[qR->second][iW34.col()][0],
//									                 Mtmp);
//								}
//								
//								if (Mtmp.rows() != 0)
//								{
//									qarray2<Symmetry::Nq> qupleApair = {H1.L.in(qL), H2.R.out(qR->second)};
//									auto qApair = Apair[s1][s3].dict.find(qupleApair);
//									
//									if (qApair != Apair[s1][s3].dict.end())
//									{
//										Apair[s1][s3].block[qApair->second] += Mtmp;
//									}
//									else
//									{
//										Apair[s1][s3].push_back(qupleApair, Mtmp);
//									}
//								}
//							}
//						}
//					}
//				}
//			}
//		}
//	}
//}

#endif
