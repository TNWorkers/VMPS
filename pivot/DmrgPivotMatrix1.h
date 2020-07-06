#ifndef DMRGPIVOTMATRIX1
#define DMRGPIVOTMATRIX1

#include "pivot/DmrgPivotVector.h"
//include "DmrgTypedefs.h"
//include "tensors/Biped.h"
//include "tensors/Multipede.h"

//-----------<definitions>-----------
template<typename Symmetry, typename Scalar, typename MpoScalar=double>
struct PivotMatrix1
{
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > L;
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > R;
	vector<vector<vector<Biped<Symmetry,Eigen::SparseMatrix<MpoScalar,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE>> > > > W;
	
	vector<std::array<size_t,2> >          qlhs;
	vector<vector<std::array<size_t,6> > > qrhs;
	vector<vector<Scalar> >                factor_cgcs;
	
	vector<qarray<Symmetry::Nq> > qloc;
	vector<qarray<Symmetry::Nq> > qOp;
	
	// stuff for excited states
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > PL; // PL[n]
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > PR; // PL[n]
	vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > A0proj; // A0proj[n][s]
	double Epenalty = 0;
	
	template<typename OtherScalar>
	PivotMatrix1<Symmetry,OtherScalar,OtherScalar> cast() const
	{
		PivotMatrix1<Symmetry,OtherScalar,OtherScalar> Pout;
		
		Pout.L = L.template cast<Matrix<OtherScalar,Dynamic,Dynamic> >();
		Pout.R = R.template cast<Matrix<OtherScalar,Dynamic,Dynamic> >();
//		Pout.W = W;
		
		Pout.qlhs = qlhs;
		Pout.qrhs = qrhs;
		Pout.factor_cgcs.resize(factor_cgcs.size());
		for (int i=0; i<factor_cgcs.size(); ++i) Pout.factor_cgcs[i].resize(factor_cgcs[i].size());
		
		for (int i=0; i<factor_cgcs.size(); ++i)
		for (int j=0; j<factor_cgcs[i].size(); ++j)
		{
			Pout.factor_cgcs[i][j] = factor_cgcs[i][j];
		}
		
		Pout.qloc = qloc;
		Pout.qOp = qOp;
		
		return Pout;
	}
};

template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivotMatrix1<Symmetry,Scalar,MpoScalar> &H, const PivotVector<Symmetry,Scalar> &Vin, PivotVector<Symmetry,Scalar> &Vout)
{
//	Vout.outerResize(Vin);
	Vout = Vin;
	OxV(H,Vin,Vout);
//	for (size_t s=0; s<Vin.data.size(); ++s)
//	{
//		cout << "s=" << s << endl;
//		cout << Vin.data[s].print(false) << endl;
//		cout << Vout.data[s].print(false) << endl;
//	}
//	cout << endl;
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void OxV (const PivotMatrix1<Symmetry,Scalar,MpoScalar> &H, const PivotVector<Symmetry,Scalar> &Vin, PivotVector<Symmetry,Scalar> &Vout)
{
	for (size_t s=0; s<Vout.data.size(); ++s) {Vout.data[s].setZero();}
	
//	cout << "1site H.qlhs.size()=" << H.qlhs.size() << endl;
	#ifdef DMRG_PIVOT1_PARALLELIZE
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
            size_t qW = H.qrhs[q][p][5];
			
			for (int r=0; r<H.W[s1][s2][k].block[qW].outerSize(); ++r)
			for (typename Eigen::SparseMatrix<MpoScalar,Eigen::ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE>::InnerIterator iW(H.W[s1][s2][k].block[qW],r); iW; ++iW)
			{
				if (H.L.block[qL][iW.row()][0].size() != 0 and 
				    H.R.block[qR][iW.col()][0].size() != 0 and
				    Vin.data[s2].block[q2].size() !=0)
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
