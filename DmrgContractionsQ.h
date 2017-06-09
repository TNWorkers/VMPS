#ifndef STRAWBERRY_DMRGCONTRACTIONS_WITH_Q
#define STRAWBERRY_DMRGCONTRACTIONS_WITH_Q

#include "MpsQ.h"
#include "MpoQ.h"
#include "DmrgIndexGymnastics.h"

/**Contracts a left transfer matrix \p Lold with two MpsQ tensors \p Abra, \p Aket and an MpoQ tensor \p W as follows:
\dotfile contractQ_L.dot
\param Lold
\param Abra
\param W
\param Aket
\param qloc : local basis
\param Lnew : new transfer matrix to be written to
*/
template<typename Symmetry, typename MatrixType, typename MpoScalar>
void contract_L (const Tripod<Symmetry,MatrixType> &Lold, 
                 const vector<Biped<Symmetry,MatrixType> > &Abra, 
                 const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W, 
                 const vector<Biped<Symmetry,MatrixType> > &Aket, 
                 const vector<qarray<Symmetry::Nq> > &qloc, 
                 Tripod<Symmetry,MatrixType> &Lnew)
{
	Lnew.clear();
	Lnew.setZero();
	
	for (size_t s1=0; s1<qloc.size(); ++s1)
	for (size_t s2=0; s2<qloc.size(); ++s2)
	for (size_t qL=0; qL<Lold.dim; ++qL)
	{
		tuple<qarray3<Symmetry::Nq>,size_t,size_t> ix;
		bool FOUND_MATCH = AWA(Lold.in(qL), Lold.out(qL), Lold.mid(qL), s1, s2, qloc, Abra, Aket, ix);
		
		if (FOUND_MATCH == true)
		{
			qarray3<Symmetry::Nq> quple = get<0>(ix);
			swap(quple[0], quple[1]);
			size_t qAbra = get<1>(ix);
			size_t qAket = get<2>(ix);
			
			for (int k=0; k<W[s1][s2][0].outerSize(); ++k)
			for (typename SparseMatrix<MpoScalar>::InnerIterator iW(W[s1][s2][0],k); iW; ++iW)
			{
				size_t a1 = iW.row();
				size_t a2 = iW.col();
				
				if (Lold.block[qL][a1][0].rows() != 0)
				{
//					MatrixType Mtmp = iW.value() *
//					                  (Abra[s1].block[qAbra].adjoint() *
//					                   Lold.block[qL][a1][0] * 
//					                   Aket[s2].block[qAket]);
					MatrixType Mtmp;
					optimal_multiply(iW.value(),
					                 Abra[s1].block[qAbra].adjoint(),
					                 Lold.block[qL][a1][0],
					                 Aket[s2].block[qAket],
					                 Mtmp);
					
					auto it = Lnew.dict.find(quple);
					if (it != Lnew.dict.end())
					{
						if (Lnew.block[it->second][a2][0].rows() != Mtmp.rows() or 
							Lnew.block[it->second][a2][0].cols() != Mtmp.cols())
						{
							Lnew.block[it->second][a2][0] = Mtmp;
						}
						else
						{
							Lnew.block[it->second][a2][0] += Mtmp;
						}
					}
					else
					{
						boost::multi_array<MatrixType,LEGLIMIT> Mtmpvec(boost::extents[W[s1][s2][0].cols()][1]);
						Mtmpvec[a2][0] = Mtmp;
						Lnew.push_back(quple, Mtmpvec);
					}
				}
			}
		}
	}
}

/**Contracts a right transfer matrix \p Rold with two MpsQ tensors \p Abra, \p Aket and an MpoQ tensor \p W as follows:
\dotfile contractQ_R.dot
\param Rold
\param Abra
\param W
\param Aket
\param qloc : local basis
\param Rnew : new transfer matrix to be written to
*/
template<typename Symmetry, typename MatrixType, typename MpoScalar>
void contract_R (const Tripod<Symmetry,MatrixType> &Rold,
                 const vector<Biped<Symmetry,MatrixType> > &Abra, 
                 const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W, 
                 const vector<Biped<Symmetry,MatrixType> > &Aket, 
                 const vector<qarray<Symmetry::Nq> > &qloc, 
                 Tripod<Symmetry,MatrixType> &Rnew)
{
	Rnew.clear();
	Rnew.setZero();
	
	for (size_t s1=0; s1<qloc.size(); ++s1)
	for (size_t s2=0; s2<qloc.size(); ++s2)
	for (size_t qR=0; qR<Rold.dim; ++qR)
	{
		qarray2<Symmetry::Nq> cmp1 = {Rold.out(qR)-qloc[s1], Rold.out(qR)};
		qarray2<Symmetry::Nq> cmp2 = {Rold.in(qR) -qloc[s2], Rold.in(qR)};
		
		auto q1 = Abra[s1].dict.find(cmp1);
		auto q2 = Aket[s2].dict.find(cmp2);
		
		if (q1!=Abra[s1].dict.end() and 
		    q2!=Aket[s2].dict.end())
		{
			qarray<Symmetry::Nq> new_qin  = Aket[s2].in[q2->second]; // A.in
			qarray<Symmetry::Nq> new_qout = Abra[s1].in[q1->second]; // A†.out = A.in
			qarray<Symmetry::Nq> new_qmid = Rold.mid(qR) - qloc[s1] + qloc[s2];
			qarray3<Symmetry::Nq> quple = {new_qin, new_qout, new_qmid};
			
			for (int k=0; k<W[s1][s2][0].outerSize(); ++k)
			for (typename SparseMatrix<MpoScalar>::InnerIterator iW(W[s1][s2][0],k); iW; ++iW)
			{
				size_t a1 = iW.row();
				size_t a2 = iW.col();
				
				if (Rold.block[qR][a2][0].rows() != 0)
				{
//					MatrixType Mtmp = iW.value() *
//					                  (Aket[s2].block[q2->second] * 
//					                   Rold.block[qR][a2][0] * 
//					                   Abra[s1].block[q1->second].adjoint());
					MatrixType Mtmp;
					optimal_multiply(iW.value(),
					                 Aket[s2].block[q2->second],
					                 Rold.block[qR][a2][0],
					                 Abra[s1].block[q1->second].adjoint(),
					                 Mtmp);
					
					auto it = Rnew.dict.find(quple);
					if (it != Rnew.dict.end())
					{
						if (Rnew.block[it->second][a1][0].rows() != Mtmp.rows() or 
							Rnew.block[it->second][a1][0].cols() != Mtmp.cols())
						{
							Rnew.block[it->second][a1][0] = Mtmp;
						}
						else
						{
							Rnew.block[it->second][a1][0] += Mtmp;
						}
					}
					else
					{
						boost::multi_array<MatrixType,LEGLIMIT> Mtmpvec(boost::extents[W[s1][s2][0].rows()][1]);
						Mtmpvec[a1][0] = Mtmp;
						Rnew.push_back(quple, Mtmpvec);
					}
				}
			}
		}
	}
}

/**Calculates the contraction between a left transfer matrix \p L, two MpsQ tensors \p Abra, \p Aket, an MpoQ tensor \p W and a right transfer matrix \p R. Not really that much useful.
\param L
\param Abra
\param W
\param Aket
\param R
\param qloc : local basis
\returns : result of contraction*/
template<typename Symmetry, typename Scalar>
Scalar contract_LR (const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &L,
                    const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Abra, 
                    const vector<vector<vector<SparseMatrixXd> > > &W, 
                    const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Aket, 
                    const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &R, 
                    const vector<qarray<Symmetry::Nq> > &qloc)
{
	Scalar res = 0.;
	
	for (size_t s1=0; s1<qloc.size(); ++s1)
	for (size_t s2=0; s2<qloc.size(); ++s2)
	for (size_t qL=0; qL<L.dim; ++qL)
	{
		tuple<qarray3<Symmetry::Nq>,size_t,size_t> ix;
		bool FOUND_MATCH = AWA(L.in(qL), L.out(qL), L.mid(qL), s1, s2, qloc, Abra, Aket, ix);
		
		if (FOUND_MATCH == true)
		{
			qarray3<Symmetry::Nq> quple = get<0>(ix);
			auto qR = R.dict.find(quple);
			
			if (qR != R.dict.end())
			{
				swap(quple[0], quple[1]);
				size_t qAbra = get<1>(ix);
				size_t qAket = get<2>(ix);
				
				for (int k=0; k<W[s1][s2][0].outerSize(); ++k)
				for (SparseMatrixXd::InnerIterator iW(W[s1][s2][0],k); iW; ++iW)
				{
					size_t a1 = iW.row();
					size_t a2 = iW.col();
					
					if (L.block[qL][a1][0].rows() != 0 and
					    R.block[qR->second][a2][0].rows() != 0)
					{
//						Matrix<Scalar,Dynamic,Dynamic> Mtmp  = iW.value() *
//						                                       (Abra[s1].block[qAbra].adjoint() *
//						                                        L.block[qL][a1][0] * 
//						                                        Aket[s2].block[qAket] * 
//						                                        R.block[qR->second][a2][0]);
//						res += Mtmp.trace();
						
						Matrix<Scalar,Dynamic,Dynamic> Mtmp = L.block[qL][a1][0] * 
						                                      Aket[s2].block[qAket] * 
						                                      R.block[qR->second][a2][0];
						for (size_t i=0; i<Abra[s1].block[qAbra].cols(); ++i)
						{
							res += iW.value() * Abra[s1].block[qAbra].col(i).dot(Mtmp.col(i));
						}
					}
				}
			}
		}
	}
	return res;
}

//template<typename Symmetry, typename MatrixType>
//void contract_LR (const Tripod<Symmetry,MatrixType> &L,
//                  const Tripod<Symmetry,MatrixType> &R, 
//                  const std::array<qarray<Symmetry::Nq>,D> &qloc, 
//                  Tripod<Symmetry,MatrixType> &Bres)
//{
//	Bres.clear();
//	Bres.setZero();
//	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	for (size_t qL=0; qL<L.dim; ++qL)
//	{
//		qarray3<Symmetry::Nq> quple = {L.out(qL), L.in(qL), L.mid(qL)};
//		auto qR = R.dict.find(quple);
//		
//		if (qR != R.dict.end())
//		{
//			if (L.block[qL][a1][0].rows() != 0 and
//			    R.block[qR->second][a2][0].rows() != 0)
//			{
////						cout << Abra[s1].block[qAbra].adjoint().rows() << "\t" << Abra[s1].block[qAbra].adjoint().cols() << endl;
////						cout << L.block[qL][a1][0].rows() << "\t" << L.block[qL][a1][0].cols() << endl;
////						cout << Aket[s2].block[qAket].rows() << "\t" << Aket[s2].block[qAket].cols() << endl;
////						cout << R.block[qR->second][a2][0].rows() << "\t" << R.block[qR->second][a2][0].cols() << endl;
////						cout << endl;
//				
//				MatrixType Mtmp = L.block[qL][a1][0] * R.block[qR->second][a2][0]);
//				
//				cout << Mtmp.rows() << "\t" << Mtmp.cols() << endl << Mtmp << endl << endl;
//				
////						auto it = Bres.dict.find(quple);
////						if (it != Bres.dict.end())
////						{
////							if (Bres.block[it->second][a2][0].rows() != Mtmp.rows() or 
////								Bres.block[it->second][a2][0].cols() != Mtmp.cols())
////							{
////								Bres.block[it->second][a2][0] = Mtmp;
////							}
////							else
////							{
////								Bres.block[it->second][a2][0] += Mtmp;
////							}
////						}
////						else
////						{
////							boost::multi_array<MatrixType,LEGLIMIT> Mtmpvec(boost::extents[W[s1][s2].block[qW].cols()][1]);
////							Mtmpvec[a2][0] = Mtmp;
////							Bres.push_back(quple, Mtmpvec);
////							cout << "in:  " << quple[0] << ", out: " << quple[1] << ", mid: " << quple[2] << endl;
////						}
//			}
//		}
//	}
//}

//template<typename Symmetry, typename MatrixType>
//void dryContract_L (const Tripod<Symmetry,MatrixType> &Lold, 
//                    const vector<Biped<Symmetry,MatrixType> > &Abra, 
//                    const std::array<std::array<Biped<Symmetry,SparseMatrixXd>,D>,D> &W, 
//                    const vector<Biped<Symmetry,MatrixType> > &Aket, 
//                    const std::array<qarray<Symmetry::Nq>,D> &qloc, 
//                    Tripod<Symmetry,MatrixType> &Lnew, 
//                    vector<tuple<qarray3<Symmetry::Nq>,std::array<size_t,8> > > &ix)
//{
//	Lnew.setZero();
//	
//	MatrixType Mtmp(1,1); Mtmp << 1.;
//	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	for (size_t qL=0; qL<Lold.dim; ++qL)
//	{
//		qarray2<Symmetry::Nq> cmp1 = {Lold.in(qL),  Lold.in(qL)+qloc[s1]};
//		qarray2<Symmetry::Nq> cmp2 = {Lold.out(qL), Lold.out(qL)+qloc[s2]};
//		qarray2<Symmetry::Nq> cmpW = {Lold.mid(qL), Lold.mid(qL)+qloc[s1]-qloc[s2]};
//		
//		auto q1 = Abra[s1].dict.find(cmp1);
//		auto q2 = Aket[s2].dict.find(cmp2);
//		auto qW = W[s1][s2].dict.find(cmpW);
//		
//		if (q1!=Abra[s1].dict.end() and 
//		    q2!=Aket[s2].dict.end() and 
//		    qW!=W[s1][s2].dict.end())
//		{
//			qarray<Symmetry::Nq> new_qin  = Abra[s1].out[q1->second]; // A†.in = A.out
//			qarray<Symmetry::Nq> new_qout = Aket[s2].out[q2->second]; // A.in
//			qarray<Symmetry::Nq> new_qmid = W[s1][s2].out[qW->second];
//			qarray3<Symmetry::Nq> quple = {new_qin, new_qout, new_qmid};
//			
//			size_t Wcols = W[s1][s2].block[qW->second].cols();
//			
//			for (int k=0; k<W[s1][s2].block[qW->second].outerSize(); ++k)
//			for (SparseMatrixXd::InnerIterator iW(W[s1][s2].block[qW->second],k); iW; ++iW)
//			{
//				size_t a1 = iW.row();
//				size_t a2 = iW.col();
//				
//				if (Lold.block[qL][a1][0].rows() != 0)
//				{
//					std::array<size_t,9> juple = {s1, s2, q1->second, qW->second, q2->second, qL, a1, a2};
//					ix.push_back(make_tuple(quple,juple));
//					
//					auto it = Lnew.dict.find(quple);
//					if (it != Lnew.dict.end())
//					{
//						if (Lnew.block[it->second][a2][0].rows() == 0)
//						{
//							Lnew.block[it->second][a2][0] = Mtmp;
//						}
//					}
//					else
//					{
//						boost::multi_array<MatrixType,LEGLIMIT> Mtmpvec(boost::extents[Wcols][1]);
//						Mtmpvec[a2][0] = Mtmp;
//						Lnew.push_back(quple, Mtmpvec);
//					}
//				}
//			}
//		}
//	}
//}

//template<typename Symmetry, typename MatrixType>
//void contract_L (const Tripod<Symmetry,MatrixType> &Lold, 
//                 const vector<tuple<qarray3<Symmetry::Nq>,std::array<size_t,8> > > ix, 
//                 const vector<Biped<Symmetry,MatrixType> > &Abra, 
//                 const std::array<std::array<Biped<Symmetry,SparseMatrixXd>,D>,D> &W, 
//                 const vector<Biped<Symmetry,MatrixType> > &Aket, 
//                 const std::array<qarray<Symmetry::Nq>,D> &qloc, 
//                 Tripod<Symmetry,MatrixType> &Lnew)
//{
//	Lnew.setZero();
//	
//	for (size_t i=0; i<ix.size(); ++i)
//	{
//		auto quple = get<0>(ix);
//		size_t s1 = get<1>(ix)[0];
//		size_t s2 = get<1>(ix)[1];
//		size_t q1 = get<1>(ix)[2];
//		size_t qW = get<1>(ix)[3];
//		size_t q2 = get<1>(ix)[4];
//		size_t qL = get<1>(ix)[5];
//		size_t a1 = get<1>(ix)[6];
//		size_t a2 = get<1>(ix)[7];
//		
//		for (int k=0; k<W[s1][s2].block[qW].outerSize(); ++k)
//		for (SparseMatrixXd::InnerIterator iW(W[s1][s2].block[qW],k); iW; ++iW)
//		{
//			size_t a1 = iW.row();
//			size_t a2 = iW.col();
//			
//			if (Lold.block[qL][a1][0].rows() != 0)
//			{
//				MatrixType Mtmp = iW.value() *
//				                  (Abra[s1].block[q1].adjoint() *
//				                   Lold.block[qL][a1][0] * 
//				                   Aket[s2].block[q2]);
//				
//				auto it = Lnew.dict.find(quple);
//				if (it != Lnew.dict.end())
//				{
//					if (Lnew.block[it->second][a2][0].rows() != Mtmp.rows() or 
//						Lnew.block[it->second][a2][0].cols() != Mtmp.cols())
//					{
//						Lnew.block[it->second][a2][0] = Mtmp;
//					}
//					else
//					{
//						Lnew.block[it->second][a2][0] += Mtmp;
//					}
//				}
//				else
//				{
//					boost::multi_array<MatrixType,LEGLIMIT> Mtmpvec(boost::extents[W[s1][s2].block[qW].cols()][1]);
//					Mtmpvec[a2][0] = Mtmp;
//					Lnew.push_back(quple, Mtmpvec);
//				}
//			}
//		}
//	}
//}

/**Calculates the contraction between a left transfer matrix \p Lold, two MpsQ tensors \p Abra, \p Aket and two MpoQ tensors \p Wbot, \p Wtop.
Needed, for example, when calculating \f$\left<H^2\right>\f$ and no MpoQ represenation of \f$H^2\f$ is available.*/
template<typename Symmetry, typename MatrixType, typename MpoScalar>
void contract_L (const Multipede<4,Symmetry,MatrixType> &Lold, 
                 const vector<Biped<Symmetry,MatrixType> > &Abra, 
                 const vector<vector<vector<SparseMatrix<MpoScalar> > > > &Wbot, 
                 const vector<vector<vector<SparseMatrix<MpoScalar> > > > &Wtop, 
                 const vector<Biped<Symmetry,MatrixType> > &Aket, 
                 const vector<qarray<Symmetry::Nq> > &qloc,
                 Multipede<4,Symmetry,MatrixType> &Lnew)
{
	Lnew.setZero();
	
	for (size_t s1=0; s1<qloc.size(); ++s1)
	for (size_t s2=0; s2<qloc.size(); ++s2)
	for (size_t s3=0; s3<qloc.size(); ++s3)
	for (size_t qL=0; qL<Lold.dim; ++qL)
	{
		tuple<qarray4<Symmetry::Nq>,size_t,size_t> ix;
		bool FOUND_MATCH = AWWA(Lold.in(qL), Lold.out(qL), Lold.bot(qL), Lold.top(qL), 
		                        s1, s2, s3, qloc, Abra, Aket, ix);
		auto   quple = get<0>(ix);
		swap(quple[0],quple[1]);
		size_t qAbra = get<1>(ix);
		size_t qAket = get<2>(ix);
		
		if (FOUND_MATCH == true)
		{
			for (int kbot=0; kbot<Wbot[s1][s2][0].outerSize(); ++kbot)
			for (typename SparseMatrix<MpoScalar>::InnerIterator iWbot(Wbot[s1][s2][0],kbot); iWbot; ++iWbot)
			for (int ktop=0; ktop<Wtop[s2][s3][0].outerSize(); ++ktop)
			for (typename SparseMatrix<MpoScalar>::InnerIterator iWtop(Wtop[s2][s3][0],ktop); iWtop; ++iWtop)
			{
				size_t br = iWbot.row();
				size_t bc = iWbot.col();
				size_t tr = iWtop.row();
				size_t tc = iWtop.col();
				MpoScalar Wfactor = iWbot.value() * iWtop.value();
				
				if (Lold.block[qL][br][tr].rows() != 0)
				{
//					MatrixType Mtmp = (iWbot.value() * iWtop.value()) * 
//					                  (Abra[s1].block[qAbra].adjoint() *
//					                   Lold.block[qL][br][tr] * 
//					                   Aket[s3].block[qAket]);
					MatrixType Mtmp;
					optimal_multiply(Wfactor,
					                 Abra[s1].block[qAbra].adjoint(),
					                 Lold.block[qL][br][tr],
					                 Aket[s3].block[qAket],
					                 Mtmp);
					
					if (Mtmp.norm() != 0.)
					{
						auto it = Lnew.dict.find(quple);
						if (it != Lnew.dict.end())
						{
							if (Lnew.block[it->second][bc][tc].rows() != Mtmp.rows() or 
								Lnew.block[it->second][bc][tc].cols() != Mtmp.cols())
							{
								Lnew.block[it->second][bc][tc] = Mtmp;
							}
							else
							{
								Lnew.block[it->second][bc][tc] += Mtmp;
							}
						}
						else
						{
							size_t bcols = Wbot[s1][s2][0].cols();
							size_t tcols = Wtop[s2][s3][0].cols();
							boost::multi_array<MatrixType,LEGLIMIT> Mtmparray(boost::extents[bcols][tcols]);
							Mtmparray[bc][tc] = Mtmp;
							Lnew.push_back(quple, Mtmparray);
						}
					}
				}
			}
		}
	}
}

/**For details see: Stoudenmire, White (2010)*/
template<typename Symmetry, typename MatrixType, typename MpoScalar>
void contract_C0 (vector<qarray<Symmetry::Nq> > qloc,
                  const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W, 
                  const vector<Biped<Symmetry,MatrixType> >   &Aket, 
                  vector<Tripod<Symmetry,MatrixType> >        &Cnext)
{
	Cnext.clear();
	Cnext.resize(qloc.size());
	
	for (size_t s2=0; s2<qloc.size(); ++s2)
	{
		qarray2<Symmetry::Nq> cmpA = {qvacuum<Symmetry::Nq>(), qvacuum<Symmetry::Nq>()+qloc[s2]};
		auto qA = Aket[s2].dict.find(cmpA);
		
		if (qA != Aket[s2].dict.end())
		{
			for (size_t s1=0; s1<qloc.size(); ++s1)
			{
				for (int k=0; k<W[s1][s2][0].outerSize(); ++k)
				for (typename SparseMatrix<MpoScalar>::InnerIterator iW(W[s1][s2][0],k); iW; ++iW)
				{
					MatrixType Mtmp = iW.value() * Aket[s2].block[qA->second];
					
					qarray3<Symmetry::Nq> cmpC = {qvacuum<Symmetry::Nq>(), Aket[s2].out[qA->second], qvacuum<Symmetry::Nq>()+qloc[s1]-qloc[s2]};
					auto qCnext = Cnext[s1].dict.find(cmpC);
					if (qCnext != Cnext[s1].dict.end())
					{
						if (Cnext[s1].block[qCnext->second][iW.col()][0].rows() != Mtmp.rows() or 
							Cnext[s1].block[qCnext->second][iW.col()][0].cols() != Mtmp.cols())
						{
							Cnext[s1].block[qCnext->second][iW.col()][0] = Mtmp;
						}
						else
						{
							Cnext[s1].block[qCnext->second][iW.col()][0] += Mtmp;
						}
					}
					else
					{
						boost::multi_array<MatrixType,LEGLIMIT> Mtmpvec(boost::extents[W[s1][s2][0].cols()][1]);
						Mtmpvec[iW.col()][0] = Mtmp;
						Cnext[s1].push_back({qvacuum<Symmetry::Nq>(), Aket[s2].out[qA->second], qvacuum<Symmetry::Nq>()+qloc[s1]-qloc[s2]}, Mtmpvec);
					}
				}
			}
		}
	}
}

/**For details see: Stoudenmire, White (2010)
\dotfile contract_C.dot*/
template<typename Symmetry, typename MatrixType, typename MpoScalar>
void contract_C (vector<qarray<Symmetry::Nq> > qloc,
                 const vector<Biped<Symmetry,MatrixType> >   &Abra, 
                 const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W, 
                 const vector<Biped<Symmetry,MatrixType> >   &Aket, 
                 const vector<Tripod<Symmetry,MatrixType> >  &C, 
                 vector<Tripod<Symmetry,MatrixType> >        &Cnext)
{
	Cnext.clear();
	Cnext.resize(qloc.size());
	
	for (size_t s=0; s<qloc.size(); ++s)
	for (size_t qC=0; qC<C[s].dim; ++qC)
	{
		qarray2<Symmetry::Nq> cmpU = {C[s].in(qC), C[s].in(qC)+qloc[s]};
		auto qU = Abra[s].dict.find(cmpU);
		
		if (qU != Abra[s].dict.end())
		{
			for (size_t s1=0; s1<qloc.size(); ++s1)
			for (size_t s2=0; s2<qloc.size(); ++s2)
			{
				qarray2<Symmetry::Nq> cmpA = {C[s].out(qC), C[s].out(qC)+qloc[s2]};
				auto qA = Aket[s2].dict.find(cmpA);
				
				if (qA != Aket[s2].dict.end())
				{
					for (int k=0; k<W[s1][s2][0].outerSize(); ++k)
					for (typename SparseMatrix<MpoScalar>::InnerIterator iW(W[s1][s2][0],k); iW; ++iW)
					{
						if (C[s].block[qC][iW.row()][0].rows() != 0)
						{
//							MatrixType Mtmp = iW.value() * (Abra[s].block[qU->second].adjoint() * 
//							                                C[s].block[qC][iW.row()][0] * 
//							                                Aket[s2].block[qA->second]);
							MatrixType Mtmp;
							optimal_multiply(iW.value(),
							                 Abra[s].block[qU->second].adjoint(),
							                 C[s].block[qC][iW.row()][0],
							                 Aket[s2].block[qA->second],
							                 Mtmp);
							
							qarray3<Symmetry::Nq> cmpC = {Abra[s].out[qU->second], Aket[s2].out[qA->second], C[s].mid(qC)+qloc[s1]-qloc[s2]};
							auto qCnext = Cnext[s1].dict.find(cmpC);
							if (qCnext != Cnext[s1].dict.end())
							{
								if (Cnext[s1].block[qCnext->second][iW.col()][0].rows() != Mtmp.rows() or 
									Cnext[s1].block[qCnext->second][iW.col()][0].cols() != Mtmp.cols())
								{
									Cnext[s1].block[qCnext->second][iW.col()][0] = Mtmp;
								}
								else
								{
									Cnext[s1].block[qCnext->second][iW.col()][0] += Mtmp;
								}
							}
							else
							{
								boost::multi_array<MatrixType,LEGLIMIT> Mtmpvec(boost::extents[W[s1][s2][0].cols()][1]);
								Mtmpvec[iW.col()][0] = Mtmp;
								Cnext[s1].push_back({Abra[s].out[qU->second], Aket[s2].out[qA->second], C[s].mid(qC)+qloc[s1]-qloc[s2]}, Mtmpvec);
							}
						}
					}
				}
			}
		}
	}
}

#endif
