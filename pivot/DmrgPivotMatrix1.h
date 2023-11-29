#ifndef DMRGPIVOTMATRIX1
#define DMRGPIVOTMATRIX1

#include "pivot/DmrgPivotVector.h"
#include "tensors/DmrgIndexGymnastics.h"
//include "DmrgTypedefs.h"
//include "tensors/Biped.h"
//include "tensors/Multipede.h"

//-----------<definitions>-----------
template<typename Symmetry, typename Scalar, typename MpoScalar=double>
struct PivotMatrix1Terms
{
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > L;
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > R;
	vector<vector<vector<Biped<Symmetry,Eigen::SparseMatrix<MpoScalar,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> > > > > W;
	
	vector<qarray<Symmetry::Nq> > qloc;
	vector<qarray<Symmetry::Nq> > qOp;
	
//	void save_L (string filename)
//	{
//		lout << termcolor::green << "Saving L to: " << filename << termcolor::reset << std::endl;
//		L.save(filename);
//	}
//	
//	void save_R (string filename)
//	{
//		lout << termcolor::green << "Saving R to: " << filename << termcolor::reset << std::endl;
//		R.save(filename);
//	}
//	
//	void load_L (string filename)
//	{
//		lout << termcolor::green << "Loading L from: " << filename << termcolor::reset << std::endl;
//		L.load(filename);
//	}
//	
//	void load_R (string filename)
//	{
//		lout << termcolor::green << "Loading R from: " << filename << termcolor::reset << std::endl;
//		R.load(filename);
//	}
};

template<typename Symmetry, typename Scalar, typename MpoScalar=double>
struct PivotMatrix1
{
//	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > L;
//	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > R;
//	vector<vector<vector<Biped<Symmetry,Eigen::SparseMatrix<MpoScalar,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> > > > > W;
//	
//	vector<qarray<Symmetry::Nq> > qloc;
//	vector<qarray<Symmetry::Nq> > qOp;
	
	vector<PivotMatrix1Terms<Symmetry,Scalar,MpoScalar> > Terms;
	
	//---<pre-calculated, if Terms.size() == 1>---
	vector<std::array<size_t,2> >          qlhs;
	vector<vector<std::array<size_t,6> > > qrhs;
	vector<vector<Scalar> >                factor_cgcs;
	//--------------------------------
	
	//---<stuff for excited states>---
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > PL; // PL[n]
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > PR; // PL[n]
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > A0proj; // A0proj[n][s]
	double Epenalty = 0;
	//--------------------------------
	
	template<typename OtherScalar>
	PivotMatrix1<Symmetry,OtherScalar,OtherScalar> cast() const
	{
		PivotMatrix1<Symmetry,OtherScalar,OtherScalar> Pout;
		
		Pout.Terms.resize(Terms.size());
		
		for (size_t t=0; t<Terms.size(); ++t)
		{
			Pout.Terms[t].L = Terms[t].L.template cast<Matrix<OtherScalar,Dynamic,Dynamic> >();
			Pout.Terms[t].R = Terms[t].R.template cast<Matrix<OtherScalar,Dynamic,Dynamic> >();
//			Pout.W = W;
			
			Pout.Terms[t].qloc = Terms[t].qloc;
			Pout.Terms[t].qOp  = Terms[t].qOp;
		}
		
		Pout.qlhs = qlhs;
		Pout.qrhs = qrhs;
		Pout.factor_cgcs.resize(factor_cgcs.size());
		for (int i=0; i<factor_cgcs.size(); ++i) Pout.factor_cgcs[i].resize(factor_cgcs[i].size());
		
		for (int i=0; i<factor_cgcs.size(); ++i)
		for (int j=0; j<factor_cgcs[i].size(); ++j)
		{
			Pout.factor_cgcs[i][j] = factor_cgcs[i][j];
		}
		
//		Pout.qloc = qloc;
//		Pout.qOp = qOp;
		
		return Pout;
	}
	
	double memory (MEMUNIT memunit) const
	{
		double res = 0.;
		for (size_t t=0; t<Terms.size(); ++t)
		{
			res += Terms[t].L.memory(memunit);
			res += Terms[t].R.memory(memunit);
			for (size_t s1=0; s1<Terms[t].W.size(); ++s1)
			for (size_t s2=0; s2<Terms[t].W[s1].size(); ++s2)
			for (size_t k=0; k<Terms[t].W[s1][s2].size(); ++k)
			{
				res += Terms[t].W[s1][s2][k].memory(memunit);
			}
		}
		return res;
	}
};

template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivotMatrix1<Symmetry,Scalar,MpoScalar> &H, const PivotVector<Symmetry,Scalar> &Vin, PivotVector<Symmetry,Scalar> &Vout)
{
//	Vout.outerResize(Vin);
	Vout = Vin;
	OxV(H,Vin,Vout);
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void OxV (const PivotMatrix1<Symmetry,Scalar,MpoScalar> &H, const PivotVector<Symmetry,Scalar> &Vin, PivotVector<Symmetry,Scalar> &Vout)
{
	for (size_t s=0; s<Vout.data.size(); ++s) {Vout.data[s].setZero();}
	vector<PivotVector<Symmetry,Scalar> > Vt(H.Terms.size());
	for (size_t t=0; t<H.Terms.size(); ++t) Vt[t] = Vout;
	
	vector<std::array<size_t,2> >          qlhs;
	vector<vector<std::array<size_t,6> > > qrhs;
	vector<vector<Scalar> >                factor_cgcs;
	
//	precalc_blockStructure<Symmetry,Scalar,Eigen::SparseMatrix<MpoScalar,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> > 
//	(H.L, Vin.data, H.W, Vin.data, H.R, H.qloc, H.qOp, qlhs, qrhs, factor_cgcs);
//	//cout << "qlhs.size()=" << qlhs.size() << endl;
//	
//	#ifdef DMRG_PIVOT1_PARALLELIZE
//	#pragma omp parallel for schedule(dynamic)
//	#endif
//	for (size_t q=0; q<qlhs.size(); ++q)
//	{
//		size_t s1 = qlhs[q][0];//H.qlhs[q][0];
//		size_t q1 = qlhs[q][1];//H.qlhs[q][1];
//		
//		for (size_t p=0; p<qrhs[q].size(); ++p)
//		{
//			size_t s2 = qrhs[q][p][0];//H.qrhs[q][p][0];
//			size_t q2 = qrhs[q][p][1];//H.qrhs[q][p][1];
//			size_t qL = qrhs[q][p][2];//H.qrhs[q][p][2];
//			size_t qR = qrhs[q][p][3];//H.qrhs[q][p][3];
//			size_t k  = qrhs[q][p][4];//H.qrhs[q][p][4];
//			size_t qW = qrhs[q][p][5];//H.qrhs[q][p][5];
//			
//			for (int r=0; r<H.W[s1][s2][k].block[qW].outerSize(); ++r)
//			for (typename Eigen::SparseMatrix<MpoScalar,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE>::InnerIterator iW(H.W[s1][s2][k].block[qW],r); iW; ++iW)
//			{
//				if (H.L.block[qL][iW.row()][0].size() != 0 and 
//				    H.R.block[qR][iW.col()][0].size() != 0 and
//				    Vin.data[s2].block[q2].size() !=0)
//				{
////					print_size(H.L.block[qL][iW.row()][0],"H.L.block[qL][iW.row()][0]");
////					print_size(Vin.data[s2].block[q2], "Vin.data[s2].block[q2]");
////					print_size(H.R.block[qR][iW.col()][0], "H.R.block[qR][iW.col()][0]");
////					cout << endl;
//					
//					if (Vout.data[s1].block[q1].rows() != H.L.block[qL][iW.row()][0].rows() or
//					    Vout.data[s1].block[q1].cols() != H.R.block[qR][iW.col()][0].cols())
//					{
//						Vout.data[s1].block[q1].noalias() = factor_cgcs[q][p] * iW.value() * 
//						                                   (H.L.block[qL][iW.row()][0] * 
//						                                    Vin.data[s2].block[q2] * 
//						                                    H.R.block[qR][iW.col()][0]);
//					}
//					else
//					{
//						Vout.data[s1].block[q1].noalias() += factor_cgcs[q][p] * iW.value() * 
//						                                    (H.L.block[qL][iW.row()][0] * 
//						                                     Vin.data[s2].block[q2] * 
//						                                     H.R.block[qR][iW.col()][0]);
//					}
//				}
//			}
//		}
//	}
	
	//Stopwatch<> Timer1;
	
	#ifdef DMRG_PARALLELIZE_TERMS
	#pragma omp parallel for schedule(dynamic)
	#endif
	for (size_t t=0; t<H.Terms.size(); ++t)
	{
		vector<std::array<size_t,2> >          qlhs;
		vector<vector<std::array<size_t,6> > > qrhs;
		vector<vector<Scalar> >                factor_cgcs;
		
		if (H.Terms.size() == 1)
		{
			qlhs = H.qlhs;
			qrhs = H.qrhs;
			factor_cgcs = H.factor_cgcs;
		}
		else
		{
			precalc_blockStructure<Symmetry,Scalar,Eigen::SparseMatrix<MpoScalar,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> >
			(H.Terms[t].L, Vin.data, H.Terms[t].W, Vin.data, H.Terms[t].R, H.Terms[t].qloc, H.Terms[t].qOp, qlhs, qrhs, factor_cgcs);
			//cout << "t=" << t << ", qlhs.size()=" << qlhs.size() << endl;
		}
		
		#ifdef DMRG_PIVOT1_PARALLELIZE
		#pragma omp parallel for schedule(dynamic)
		#endif
		for (size_t q=0; q<qlhs.size(); ++q)
		{
			size_t s1 = qlhs[q][0];//H.Terms[t].qlhs[q][0];
			size_t q1 = qlhs[q][1];//H.Terms[t].qlhs[q][1];
			
			for (size_t p=0; p<qrhs[q].size(); ++p)
			{
				size_t s2 = qrhs[q][p][0];//H.Terms[t].qrhs[q][p][0];
				size_t q2 = qrhs[q][p][1];//H.Terms[t].qrhs[q][p][1];
				size_t qL = qrhs[q][p][2];//H.Terms[t].qrhs[q][p][2];
				size_t qR = qrhs[q][p][3];//H.Terms[t].qrhs[q][p][3];
				size_t k  = qrhs[q][p][4];//H.Terms[t].qrhs[q][p][4];
				size_t qW = qrhs[q][p][5];//H.Terms[t].qrhs[q][p][5];
				
				for (int r=0; r<H.Terms[t].W[s1][s2][k].block[qW].outerSize(); ++r)
				for (typename Eigen::SparseMatrix<MpoScalar,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE>::InnerIterator iW(H.Terms[t].W[s1][s2][k].block[qW],r); iW; ++iW)
				{
//					print_size(H.Terms[t].L.block[qL][iW.row()][0],"H.Terms[t].L.block[qL][iW.row()][0]");
//					print_size(Vin.data[s2].block[q2], "Vin.data[s2].block[q2]");
//					print_size(H.Terms[t].R.block[qR][iW.col()][0], "H.Terms[t].R.block[qR][iW.col()][0]");
//					cout << endl;
					
					if (H.Terms[t].L.block[qL][iW.row()][0].size() != 0 and 
						H.Terms[t].R.block[qR][iW.col()][0].size() != 0 and
						Vin.data[s2].block[q2].size() !=0)
					{
						if (Vt[t].data[s1].block[q1].rows() != H.Terms[t].L.block[qL][iW.row()][0].rows() or
							Vt[t].data[s1].block[q1].cols() != H.Terms[t].R.block[qR][iW.col()][0].cols())
						{
							Vt[t].data[s1].block[q1].noalias() = factor_cgcs[q][p] * iW.value() * 
								                               (H.Terms[t].L.block[qL][iW.row()][0] * 
								                                Vin.data[s2].block[q2] * 
								                                H.Terms[t].R.block[qR][iW.col()][0]);
						}
						else
						{
							Vt[t].data[s1].block[q1].noalias() += factor_cgcs[q][p] * iW.value() * 
								                                (H.Terms[t].L.block[qL][iW.row()][0] * 
								                                 Vin.data[s2].block[q2] * 
								                                 H.Terms[t].R.block[qR][iW.col()][0]);
						}
					}
				}
			}
		}
	}
	
	//double t1 = Timer1.time();
	//cout << "multiplication: " << t1 << endl;
	//Stopwatch<> Timer2;
	
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
	
	//double t2 = Timer2.time();
	//cout << "sum: " << t2 << endl;
	//cout << "t2/t1=" << t2/t1 << endl;
	
//	for (size_t s=0; s<Vin.data.size(); ++s)
//	for (size_t q=0; q<Vin.data[s].dim; ++q)
//	{
//		cout << "Vin inout=" << Vin.data[s].in[q] << ", " << Vin.data[s].out[q] << endl;
//		cout << "Vout inout=" << Vout.data[s].in[q] << ", " << Vout.data[s].out[q] << endl;
//		print_size(Vin.data[s].block[q],"Vin.data[s].block[q]");
//		print_size(Vout.data[s].block[q],"Vout.data[s].block[q]");
//		cout << endl;
//	}
	
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
		
		// explicit variant:
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
//		// using Biped::addScale:
//		for (size_t s=0; s<H.A0proj[n].size(); ++s)
//		{
//			Vout.data[s].addScale(H.Epenalty*overlap, H.A0proj[n][s]);
//		}
	}
}

//template<typename Symmetry, typename Scalar, typename MpoScalar>
//void careful_HxV (const PivotMatrix1<Symmetry,Scalar,MpoScalar> &H, const PivotVector<Symmetry,Scalar> &Vin, PivotVector<Symmetry,Scalar> &Vout, std::array<qarray<Nq>,D> qloc)
//{
//	Vout = Vin;
//	for (size_t s=0; s<D; ++s) {Vout.data[s].setZero();}
//	
//	//	for (size_t s1=0; s1<D; ++s1)
////	for (size_t s2=0; s2<D; ++s2)
////	for (size_t qL=0; qL<LW[loc].dim; ++qL)
////	{
////		tuple<qarray3<Nq>,size_t,size_t,size_t> ix;
////		bool FOUND_MATCH = AWA(LW[loc].in(qL), LW[loc].out(qL), LW[loc].mid(qL), s1, s2, O.locBasis(), 
////				               Vbra.data[loc], O.W[loc], Vket.data[loc], ix);
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
////						Vbra.data[loc][s1].block[q1].noalias() += iW.value() * (LW[loc].block[qL][iW.row()][0] * 
////						                                                     Vket.data[loc][s2].block[q2] * 
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
//		bool FOUND_MATCH = AWA(H.L.in(qL), H.L.out(qL), H.L.mid(qL), s1, s2, qloc, Vout.data, H.W, Vin.data, ix);
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
//						if (Vout.data[s1].block[q1].rows() != H.L.block[qL][iW.row()][0].rows() or
//							Vout.dataA[s1].block[q1].cols() != H.R.block[qR->second][iW.col()][0].cols())
//						{
//							Vout.data[s1].block[q1] = iW.value() * 
//							                                 (H.L.block[qL][iW.row()][0] * 
//							                                  Vin.data[s2].block[q2] * 
//							                                  H.R.block[qR->second][iW.col()][0]);
//						}
//						else
//						{
//							Vout.data[s1].block[q1] += iW.value() * 
//							                                  (H.L.block[qL][iW.row()][0] * 
//							                                   Vin.data[s2].block[q2] * 
//							                                   H.R.block[qR->second][iW.col()][0]);
//						}
//					}
//				}
//			}
//		}
//	}
//}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivotMatrix1<Symmetry,Scalar,MpoScalar> &H, PivotVector<Symmetry,Scalar> &Vinout)
{
	PivotVector<Symmetry,Scalar> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
inline size_t dim (const PivotMatrix1<Symmetry,Scalar,MpoScalar> &H)
{
	return 0;
}

// How to calculate the Frobenius norm of this?
template<typename Symmetry, typename Scalar, typename MpoScalar>
inline double norm (const PivotMatrix1<Symmetry,Scalar,MpoScalar> &H)
{
	return H.dim;
}

#endif
