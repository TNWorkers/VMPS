#ifndef STRAWBERRY_DMRG_HEFF_STUFF_2SITE_WITH_Q
#define STRAWBERRY_DMRG_HEFF_STUFF_2SITE_WITH_Q

/// \cond
#include <vector>
/// \endcond

//include "DmrgTypedefs.h"
//include "tensors/Biped.h"
//include "tensors/Multipede.h"
//include "Mps.h"
#include "pivot/DmrgPivotVector.h"
#include "pivot/DmrgPivotOverlap2.h"

template<typename Symmetry, typename Scalar, typename MpoScalar=double>
struct PivotMatrix2Terms
{
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > L;
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > R;
	vector<vector<vector<Biped<Symmetry,Eigen::SparseMatrix<MpoScalar,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE>> > > > W12;
	vector<vector<vector<Biped<Symmetry,Eigen::SparseMatrix<MpoScalar,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE>> > > > W34;
	
	vector<qarray<Symmetry::Nq> > qloc12;
	vector<qarray<Symmetry::Nq> > qloc34;
	vector<qarray<Symmetry::Nq> > qOp12;
	vector<qarray<Symmetry::Nq> > qOp34;
};

template<typename Symmetry, typename Scalar, typename MpoScalar=double>
struct PivotMatrix2
{
	PivotMatrix2(){};
	
	PivotMatrix2 (const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &L_input, 
	              const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &R_input, 
	              const vector<vector<vector<Biped<Symmetry,Eigen::SparseMatrix<MpoScalar,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE>> > > >& W12_input,
	              const vector<vector<vector<Biped<Symmetry,Eigen::SparseMatrix<MpoScalar,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE>> > > >& W34_input,
	              const vector<qarray<Symmetry::Nq> > &qloc12_input,
	              const vector<qarray<Symmetry::Nq> > &qloc34_input, 
	              const vector<qarray<Symmetry::Nq> > &qOp12_input, 
	              const vector<qarray<Symmetry::Nq> > &qOp34_input)
	{
		Terms.resize(1);
		Terms[0].L = L_input;
		Terms[0].R = R_input;
		Terms[0].W12 = W12_input;
		Terms[0].W34 = W34_input;
		Terms[0].qloc12 = qloc12_input;
		Terms[0].qloc34 = qloc34_input;
		Terms[0].qOp12 = qOp12_input;
		Terms[0].qOp34 = qOp34_input;
	}
	
//	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > L;
//	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > R;
//	vector<vector<vector<Biped<Symmetry,Eigen::SparseMatrix<MpoScalar,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE>> > > > W12;
//	vector<vector<vector<Biped<Symmetry,Eigen::SparseMatrix<MpoScalar,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE>> > > > W34;
//	
//	vector<qarray<Symmetry::Nq> > qloc12;
//	vector<qarray<Symmetry::Nq> > qloc34;
//	vector<qarray<Symmetry::Nq> > qOp12;
//	vector<qarray<Symmetry::Nq> > qOp34;
	
	vector<PivotMatrix2Terms<Symmetry,Scalar,MpoScalar> > Terms;
	
	//---<pre-calculated, if Terms.size() == 1>---
	vector<std::array<size_t,2> >           qlhs;
	vector<vector<std::array<size_t,12> > > qrhs;
	vector<vector<Scalar> >                 factor_cgcs;
	//--------------------------------
	
	//---<stuff for excited states>---
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > A0proj; // A0proj[n][s]
	double Epenalty = 0;
	//--------------------------------
};

template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivotMatrix2<Symmetry,Scalar,MpoScalar> &H, const PivotVector<Symmetry,Scalar> &Vin, PivotVector<Symmetry,Scalar> &Vout)
{
	Vout = Vin;
	OxV(H,Vin,Vout);
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void OxV (const PivotMatrix2<Symmetry,Scalar,MpoScalar> &H, const PivotVector<Symmetry,Scalar> &Vin, PivotVector<Symmetry,Scalar> &Vout)
{
	for (size_t i=0; i<Vout.data.size(); ++i) {Vout.data[i].setZero();}
	vector<PivotVector<Symmetry,Scalar> > Vt(H.Terms.size());
	for (size_t t=0; t<H.Terms.size(); ++t) Vt[t] = Vout;
	
//	vector<std::array<size_t,2> >           qlhs;
//	vector<vector<std::array<size_t,12> > > qrhs;
//	vector<vector<Scalar> >                 factor_cgcs;
//	precalc_blockStructure<Symmetry,Scalar,Eigen::SparseMatrix<MpoScalar,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> >
//	(H.L, Vin.data, H.W12, H.W34, Vin.data, H.R, H.qloc12, H.qloc34, H.qOp12, H.qOp34, qlhs, qrhs, factor_cgcs);
//	
//	#ifdef DMRG_PIVOT2_PARALLELIZE
//	#pragma omp parallel for schedule(dynamic)
//	#endif
//	for (size_t q=0; q<qlhs.size(); ++q)
//	{
//		size_t s1s3 = qlhs[q][0];//H.qlhs[q][0];
//		size_t q13  = qlhs[q][1];//H.qlhs[q][1];
//		
//		for (size_t p=0; p<qrhs[q].size(); ++p)
//		{
//			size_t s2s4 = qrhs[q][p][0];//H.qrhs[q][p][0];
//			size_t q24  = qrhs[q][p][1];//H.qrhs[q][p][1];
//			size_t qL   = qrhs[q][p][2];//H.qrhs[q][p][2];
//			size_t qR   = qrhs[q][p][3];//H.qrhs[q][p][3];
//			size_t s1   = qrhs[q][p][4];//H.qrhs[q][p][4];
//			size_t s2   = qrhs[q][p][5];//H.qrhs[q][p][5];
//			size_t k12  = qrhs[q][p][6];//H.qrhs[q][p][6];
//			size_t qW12 = qrhs[q][p][7];//H.qrhs[q][p][7];
//			size_t s3   = qrhs[q][p][8];//H.qrhs[q][p][8];
//			size_t s4   = qrhs[q][p][9];//H.qrhs[q][p][9];
//			size_t k34  = qrhs[q][p][10];//H.qrhs[q][p][10];
//			size_t qW34 = qrhs[q][p][11];//H.qrhs[q][p][11];
//			
//			for (int r12=0; r12<H.W12[s1][s2][k12].block[qW12].outerSize(); ++r12)
//			for (typename SparseMatrix<MpoScalar>::InnerIterator iW12(H.W12[s1][s2][k12].block[qW12],r12); iW12; ++iW12)
//			for (int r34=0; r34<H.W34[s3][s4][k34].block[qW34].outerSize(); ++r34)
//			for (typename SparseMatrix<MpoScalar>::InnerIterator iW34(H.W34[s3][s4][k34].block[qW34],r34); iW34; ++iW34)
//			{
//				if (H.L.block[qL][iW12.row()][0].size() != 0 and 
//				    H.R.block[qR][iW34.col()][0].size() != 0 and
//				    Vin.data[s2s4].block[q24].size() != 0 and
//				    iW12.col() == iW34.row())
//				{
////					cout << "s1s3=" << s1s3 << ", q13=" << q13 << ", qL=" << qL 
////					     << ", iW12.row()=" << iW12.row() << ", qR=" << qR << ", iW34.col()=" << iW34.col() << endl;
////					print_size(H.L.block[qL][iW12.row()][0],"H.L.block[qL][iW12.row()][0]");
////					print_size(Vin.data[s2s4].block[q24],"Vin.data[s2s4].block[q24]");
////					print_size(H.R.block[qR][iW34.col()][0],"H.R.block[qR][iW34.col()][0]");
////					cout << endl;
//					
//					if (Vout.data[s1s3].block[q13].rows() != H.L.block[qL][iW12.row()][0].rows() or
//					    Vout.data[s1s3].block[q13].cols() != H.R.block[qR][iW34.col()][0].cols())
//					{
//						Vout.data[s1s3].block[q13].noalias() = factor_cgcs[q][p] * iW12.value() * iW34.value() *
//						                             (H.L.block[qL][iW12.row()][0] * 
//						                              Vin.data[s2s4].block[q24] * 
//						                              H.R.block[qR][iW34.col()][0]);
//					}
//					else
//					{
//						Vout.data[s1s3].block[q13].noalias() += factor_cgcs[q][p] * iW12.value() * iW34.value() *
//						                              (H.L.block[qL][iW12.row()][0] * 
//						                               Vin.data[s2s4].block[q24] * 
//						                               H.R.block[qR][iW34.col()][0]);
//					}
//				}
//			}
//		}
//	}
	
	#ifdef DMRG_PARALLELIZE_TERMS
	#pragma omp parallel for schedule(dynamic)
	#endif
	for (size_t t=0; t<H.Terms.size(); ++t)
	{
		vector<std::array<size_t,2> >           qlhs;
		vector<vector<std::array<size_t,12> > > qrhs;
		vector<vector<Scalar> >                 factor_cgcs;
		
		if (H.Terms.size() == 1)
		{
			qlhs = H.qlhs;
			qrhs = H.qrhs;
			factor_cgcs = H.factor_cgcs;
		}
		else
		{
			precalc_blockStructure<Symmetry,Scalar,Eigen::SparseMatrix<MpoScalar,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> >
			(H.Terms[t].L, Vin.data, H.Terms[t].W12, H.Terms[t].W34, Vin.data, H.Terms[t].R, H.Terms[t].qloc12, H.Terms[t].qloc34, H.Terms[t].qOp12, H.Terms[t].qOp34, qlhs, qrhs, factor_cgcs);
		}
		
		#ifdef DMRG_PIVOT2_PARALLELIZE
		#pragma omp parallel for schedule(dynamic)
		#endif
		for (size_t q=0; q<qlhs.size(); ++q)
		{
			size_t s1s3 = qlhs[q][0];
			size_t q13  = qlhs[q][1];
			
			for (size_t p=0; p<qrhs[q].size(); ++p)
			{
				size_t s2s4 = qrhs[q][p][0];
				size_t q24  = qrhs[q][p][1];
				size_t qL   = qrhs[q][p][2];
				size_t qR   = qrhs[q][p][3];
				size_t s1   = qrhs[q][p][4];
				size_t s2   = qrhs[q][p][5];
				size_t k12  = qrhs[q][p][6];
				size_t qW12 = qrhs[q][p][7];
				size_t s3   = qrhs[q][p][8];
				size_t s4   = qrhs[q][p][9];
				size_t k34  = qrhs[q][p][10];
				size_t qW34 = qrhs[q][p][11];
				
				for (int r12=0; r12<H.Terms[t].W12[s1][s2][k12].block[qW12].outerSize(); ++r12)
				for (typename SparseMatrix<MpoScalar>::InnerIterator iW12(H.Terms[t].W12[s1][s2][k12].block[qW12],r12); iW12; ++iW12)
				for (int r34=0; r34<H.Terms[t].W34[s3][s4][k34].block[qW34].outerSize(); ++r34)
				for (typename SparseMatrix<MpoScalar>::InnerIterator iW34(H.Terms[t].W34[s3][s4][k34].block[qW34],r34); iW34; ++iW34)
				{
					if (H.Terms[t].L.block[qL][iW12.row()][0].size() != 0 and 
						H.Terms[t].R.block[qR][iW34.col()][0].size() != 0 and
						Vin.data[s2s4].block[q24].size() != 0 and
						iW12.col() == iW34.row())
					{
						if (Vt[t].data[s1s3].block[q13].rows() != H.Terms[t].L.block[qL][iW12.row()][0].rows() or
							Vt[t].data[s1s3].block[q13].cols() != H.Terms[t].R.block[qR][iW34.col()][0].cols())
						{
							Vt[t].data[s1s3].block[q13].noalias() = factor_cgcs[q][p] * iW12.value() * iW34.value() *
								                         (H.Terms[t].L.block[qL][iW12.row()][0] * 
								                          Vin.data[s2s4].block[q24] * 
								                          H.Terms[t].R.block[qR][iW34.col()][0]);
						}
						else
						{
							Vt[t].data[s1s3].block[q13].noalias() += factor_cgcs[q][p] * iW12.value() * iW34.value() *
								                          (H.Terms[t].L.block[qL][iW12.row()][0] * 
								                           Vin.data[s2s4].block[q24] * 
								                           H.Terms[t].R.block[qR][iW34.col()][0]);
						}
					}
				}
			}
		}
	}
	
	for (size_t s=0; s<Vout.size(); ++s)
	{
		Vout[s] = Vt[0][s];
	}
	
	#ifdef DMRG_PARALLELIZE_TERMS
	#pragma omp parallel for
	#endif
	for (size_t s=0; s<Vout.size(); ++s)
	for (size_t t=1; t<H.Terms.size(); ++t)
	{
		Vout[s].addScale(1.,Vt[t][s]);
	}
	
	if (H.Terms.size() > 0) for (size_t s=0; s<Vout.size(); ++s) Vout[s] = Vout[s].cleaned();
	
	// project out unwanted states (e.g. to get lower spectrum)
	for (size_t n=0; n<H.A0proj.size(); ++n)
	{
		Scalar overlap = 0;
		for (size_t s=0; s<H.A0proj[n].size(); ++s)
		{
			// Note: Adjoint needed because we need <E0|Psi>, not <Psi|E0>
			overlap += H.A0proj[n][s].adjoint().contract(Vin.data[s]).trace();
		}
//		cout << "overlap=" << overlap << endl;
		
		for (size_t s=0; s<H.A0proj[n].size(); ++s)
		for (size_t q=0; q<H.A0proj[n][s].dim; ++q)
		{
			qarray2<Symmetry::Nq> cmp = {H.A0proj[n][s].in[q], H.A0proj[n][s].out[q]};
			auto qA = Vout.data[s].dict.find(cmp);
//			assert(qA != Vout.data[s].dict.end() and "Error in HxV(PivotMatrix1,PivotVector): projected block not found!");
			
			if (qA != Vout.data[s].dict.end() and H.A0proj[n][s].block[q].size() != 0)
			{
				Vout.data[s].block[qA->second] += H.Epenalty * overlap * H.A0proj[n][s].block[q];
			}
		}
	}
}

//template<typename Symmetry, typename Scalar, typename MpoScalar>
//void OxV (const PivotMatrix2<Symmetry,Scalar,MpoScalar> &H, const PivotVector<Symmetry,Scalar> &Vin, PivotVector<Symmetry,Scalar> &Vout)
//{
//	auto tensor_basis = Symmetry::tensorProd(H.qloc12, H.qloc34);
//	for (size_t i=0; i<Vout.data.size(); ++i)
//	{
//		Vout.data[i].setZero();
//	}
//	
//	Stopwatch Wtot;
//	double ttot=0;
//	double tmult=0;
//	double tcgc=0;
//	double tmatch=0;
//	int N_cgc_calc=0;
//	
//	for (size_t s1=0; s1<H.qloc12.size(); ++s1)
//	for (size_t s2=0; s2<H.qloc12.size(); ++s2)
//	for (size_t k12=0; k12<H.qOp12.size(); ++k12)
//	{
//		if (!Symmetry::validate(qarray3<Symmetry::Nq>{H.qloc12[s2], H.qOp12[k12], H.qloc12[s1]})) {continue;}
//		
//		for (size_t s3=0; s3<H.qloc34.size(); ++s3)
//		for (size_t s4=0; s4<H.qloc34.size(); ++s4)
//		for (size_t k34=0; k34<H.qOp34.size(); ++k34)
//		{
//			if (!Symmetry::validate(qarray3<Symmetry::Nq>{H.qloc34[s4], H.qOp34[k34], H.qloc34[s3]})) {continue;}
//			
//			auto qOps = Symmetry::reduceSilent(H.qOp12[k12], H.qOp34[k34]);
//			
//			for (const auto &qOp:qOps)
//			{
//				auto qmerges13 = Symmetry::reduceSilent(H.qloc12[s1], H.qloc34[s3]);
//				auto qmerges24 = Symmetry::reduceSilent(H.qloc12[s2], H.qloc34[s4]);
//				
//				for (const auto &qmerge13:qmerges13)
//				for (const auto &qmerge24:qmerges24)
//				{
//					auto qtensor13 = make_tuple(H.qloc12[s1], s1, H.qloc34[s3], s3, qmerge13);
//					auto s1s3 = distance(tensor_basis.begin(), find(tensor_basis.begin(), tensor_basis.end(), qtensor13));
//					
//					auto qtensor24 = make_tuple(H.qloc12[s2], s2, H.qloc34[s4], s4, qmerge24);
//					auto s2s4 = distance(tensor_basis.begin(), find(tensor_basis.begin(), tensor_basis.end(), qtensor24));
//					
//					// tensor product of the MPO operators in the physical space
//					Stopwatch<> Wcgc9;
//					Scalar factor_cgc9 = (Symmetry::NON_ABELIAN)? 
//					Symmetry::coeff_buildR(H.qloc12[s2], H.qloc34[s4], qmerge24,
//					                       H.qOp12[k12], H.qOp34[k34], qOp,
//					                       H.qloc12[s1], H.qloc34[s3], qmerge13)
//					                       :1.;
//					tcgc += Wcgc9.time();
//					++N_cgc_calc;
//					if (abs(factor_cgc9) < abs(mynumeric_limits<Scalar>::epsilon())) {continue;}
//					
//					for (size_t qL=0; qL<H.L.dim; ++qL)
//					{
//						vector<tuple<qarray3<Symmetry::Nq>,qarray<Symmetry::Nq>,size_t,size_t> > ixs;
//						Stopwatch Wmatch;
//						bool FOUND_MATCH = AAWWAA(H.L.in(qL), H.L.out(qL), H.L.mid(qL), 
//						                          k12, H.qOp12, k34, H.qOp34,
//						                          s1s3, qmerge13, s2s4, qmerge24,
//						                          Vout.data, Vin.data, ixs);
//						tmatch += Wmatch.time();
//						
//						if (FOUND_MATCH)
//						{
//							for (const auto &ix:ixs)
//							{
//								auto qR = H.R.dict.find(get<0>(ix));
//								auto qW     = get<1>(ix);
//								size_t qA13 = get<2>(ix);
//								size_t qA24 = get<3>(ix);
//								
//								// multiplication of Op12, Op34 in the auxiliary space
//								Stopwatch<> Wcgc6;
//								Scalar factor_cgc6 = (Symmetry::NON_ABELIAN)? 
//								Symmetry::coeff_Apair(H.L.mid(qL), H.qOp12[k12], qW,
//								                      H.qOp34[k34], get<0>(ix)[2], qOp)
//								                      :1.;
//								tcgc += Wcgc6.time();
//								++N_cgc_calc;
//								if (abs(factor_cgc6) < abs(mynumeric_limits<Scalar>::epsilon())) {continue;}
//								
//								if (qR != H.R.dict.end())
//								{
//									// standard coefficient for H*Psi with environments
//									Stopwatch<> WcgcHPsi;
//									Scalar factor_cgcHPsi = (Symmetry::NON_ABELIAN)?
//									Symmetry::coeff_HPsi(Vin.data[s2s4].out[qA24], qmerge24, Vin.data[s2s4].in[qA24],
//									                     H.R.mid(qR->second), qOp, H.L.mid(qL),
//									                     Vout.data[s1s3].out[qA13], qmerge13, Vout.data[s1s3].in[qA13])
//									                     :1.;
//									++N_cgc_calc;
//									tcgc += WcgcHPsi.time();
//									
//									for (int r12=0; r12<H.W12[s1][s2][k12].outerSize(); ++r12)
//									for (typename SparseMatrix<MpoScalar>::InnerIterator iW12(H.W12[s1][s2][k12],r12); iW12; ++iW12)
//									for (int r34=0; r34<H.W34[s3][s4][k34].outerSize(); ++r34)
//									for (typename SparseMatrix<MpoScalar>::InnerIterator iW34(H.W34[s3][s4][k34],r34); iW34; ++iW34)
//									{
//										Matrix<Scalar,Dynamic,Dynamic> Mtmp;
//										auto Wfactor = iW12.value() * iW34.value() * factor_cgc6 * factor_cgc9 * factor_cgcHPsi;
//										
//										if (H.L.block[qL][iW12.row()][0].size() != 0 and
//										    H.R.block[qR->second][iW34.col()][0].size() !=0 and
//										    iW12.col() == iW34.row())
//										{
//											Stopwatch Wmult;
//											optimal_multiply(Wfactor, 
//											                 H.L.block[qL][iW12.row()][0],
//											                 Vin.data[s2s4].block[qA24],
//											                 H.R.block[qR->second][iW34.col()][0],
//											                 Mtmp);
//											tmult += Wmult.time();
//										}
//										
//										if (Mtmp.size() != 0)
//										{
//											if (Vout.data[s1s3].block[qA13].rows() == Mtmp.rows() and
//											    Vout.data[s1s3].block[qA13].cols() == Mtmp.cols())
//											{
//												Vout.data[s1s3].block[qA13] += Mtmp;
//											}
//											else
//											{
//												Vout.data[s1s3].block[qA13] = Mtmp;
//											}
//										}
//									}
//								}
//							}
//						}
//					}
//				}
//			}
//		}
//	}
//	
//	ttot = Wtot.time();
//	
////	cout << "tmult=" << tmult/ttot << ", tcgc=" << tcgc/ttot << ", tmatch=" << tmatch/ttot << ", N_cgc_calc=" << N_cgc_calc << endl;
//}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivotMatrix2<Symmetry,Scalar,MpoScalar> &H, PivotVector<Symmetry,Scalar> &Vinout)
{
	PivotVector<Symmetry,Scalar> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

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

//template<typename Symmetry, typename Scalar, typename MpoScalar>
//void HxV (const PivotMatrix1<Symmetry,Scalar,MpoScalar> &H1, 
//          const PivotMatrix1<Symmetry,Scalar,MpoScalar> &H2, 
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
