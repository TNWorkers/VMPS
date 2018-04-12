#ifndef DMRGPIVOTMATRIX1
#define DMRGPIVOTMATRIX1

#include "DmrgTypedefs.h"
#include "tensors/Biped.h"
#include "tensors/Multipede.h"
#include "pivot/DmrgPivotVector.h"

//-----------<definitions>-----------
template<typename Symmetry, typename Scalar, typename MpoScalar=double>
struct PivotMatrix1
{
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > L;
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > R;
	vector<vector<vector<SparseMatrix<MpoScalar> > > > W;
	
	vector<std::array<size_t,2> >          qlhs;
	vector<vector<std::array<size_t,5> > > qrhs;
	vector<vector<Scalar> >                factor_cgcs;
	
	vector<qarray<Symmetry::Nq> > qloc;
	
	// stuff for excited states
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > PL; // PL[n]
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > PR; // PL[n]
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > A0; // A0[n][s]
	double Epenalty = 0;
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
	
	#ifndef DMRG_DONT_USE_OPENMP
	#pragma omp parallel for schedule(dynamic)
	#endif
	for (size_t q=0; q<H.qlhs.size(); ++q)
	{
		size_t s1 = H.qlhs[q][0];
		size_t q1 = H.qlhs[q][1];
		
		for (size_t p=0; p<H.qrhs[q].size(); ++p)
		{
			size_t s2 = H.qrhs[q][p][0];
			size_t q2 = H.qrhs[q][p][1];
			size_t qL = H.qrhs[q][p][2];
			size_t qR = H.qrhs[q][p][3];
			size_t k  = H.qrhs[q][p][4];
			
			for (int r=0; r<H.W[s1][s2][k].outerSize(); ++r)
			for (typename SparseMatrix<MpoScalar>::InnerIterator iW(H.W[s1][s2][k],r); iW; ++iW)
			{
				if (H.L.block[qL][iW.row()][0].size() != 0 and 
				    H.R.block[qR][iW.col()][0].size() != 0)
				{
//					print_size(H.L.block[qL][iW.row()][0],"H.L.block[qL][iW.row()][0]");
//					print_size(Vin.data[s2].block[q2], "Vin.data[s2].block[q2]");
//					print_size(H.R.block[qR][iW.col()][0], "H.R.block[qR][iW.col()][0]");
//					cout << endl;
					
					if (Vout.data[s1].block[q1].rows() != H.L.block[qL][iW.row()][0].rows() or
					    Vout.data[s1].block[q1].cols() != H.R.block[qR][iW.col()][0].cols())
					{
						Vout.data[s1].block[q1].noalias() = H.factor_cgcs[q][p] * iW.value() * 
						                                 (H.L.block[qL][iW.row()][0] * 
						                                  Vin.data[s2].block[q2] * 
						                                  H.R.block[qR][iW.col()][0]);
					}
					else
					{
						Vout.data[s1].block[q1].noalias() += H.factor_cgcs[q][p] * iW.value() * 
						                                  (H.L.block[qL][iW.row()][0] * 
						                                   Vin.data[s2].block[q2] * 
						                                   H.R.block[qR][iW.col()][0]);
					}
				}
			}
		}
	}
	
	// project out unwanted states (e.g. to get lower spectrum)
	// warning: not implemented for SU(2)!
	for (size_t n=0; n<H.A0.size(); ++n)
	{
		Scalar overlap = 0;
		
		for (size_t s=0; s<Vout.data.size(); ++s)
		{
			overlap += (H.PL[n].adjoint() * Vin.data[s] * H.PR[n].adjoint() * H.A0[n][s].adjoint()).block[0].trace();
		}
		
		for (size_t s=0; s<Vout.data.size(); ++s)
		for (size_t qPL=0; qPL<H.PL[n].dim; ++qPL)
		for (size_t qPR=0; qPR<H.PR[n].dim; ++qPR)
		{
			qarray2<Symmetry::Nq> qupleA = {H.PL[n].in[qPL], H.PR[n].out[qPR]};
			auto qA = Vout.data[s].dict.find(qupleA);
		
			qarray2<Symmetry::Nq> qupleA0 = {H.PL[n].out[qPL], H.PR[n].in[qPR]};
			auto qA0 = H.A0[n][s].dict.find(qupleA0);
		
			if (H.PL[n].out[qPL] + H.qloc[s] == H.PR[n].in[qPR] and
				qA0 != H.A0[n][s].dict.end() and
				qA != Vout.data[s].dict.end())
			{
				Vout.data[s].block[qA->second] += overlap * H.Epenalty * H.PL[n].block[qPL] * H.A0[n][s].block[qA0->second] * H.PR[n].block[qPR];
			}
		}
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
