#ifndef STRAWBERRY_DMRGCONTRACTIONS_WITH_Q
#define STRAWBERRY_DMRGCONTRACTIONS_WITH_Q

#include <unordered_set>

#include "Biped.h"
#include "Multipede.h"
#include "tensors/DmrgIndexGymnastics.h"
#include "symmetry/functions.h"

/**
 * Contracts a left transfer matrix \p Lold with two MpsQ tensors \p Abra, \p Aket and an MpoQ tensor \p W as follows:
 * \dotfile contractQ_L.dot
 * \param Lold
 * \param Abra
 * \param W
 * \param Aket
 * \param qloc : local basis
 * \param qOp : operator basis
 * \param Lnew : new transfer matrix to be written to
 */
template<typename Symmetry, typename MatrixType, typename MpoScalar>
void contract_L (const Tripod<Symmetry,MatrixType> &Lold, 
                 const vector<Biped<Symmetry,MatrixType> > &Abra, 
                 const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W, 
                 const vector<Biped<Symmetry,MatrixType> > &Aket, 
                 const vector<qarray<Symmetry::Nq> > &qloc,
                 const vector<qarray<Symmetry::Nq> > &qOp, 
                 Tripod<Symmetry,MatrixType> &Lnew)
{
	std::array<typename Symmetry::qType,3> qCheck;
	MpoScalar factor_cgc;
	Lnew.clear();
	Lnew.setZero();
	
	for (size_t s1=0; s1<qloc.size(); ++s1)
	for (size_t s2=0; s2<qloc.size(); ++s2)
	for (size_t k=0; k<qOp.size(); ++k)
	{
		qCheck = {qloc[s2],qOp[k],qloc[s1]};
		if(!Symmetry::validate(qCheck)) {continue;}
		
		for (size_t qL=0; qL<Lold.dim; ++qL)
		{
			vector<tuple<qarray3<Symmetry::Nq>,size_t,size_t> > ix;
			bool FOUND_MATCH = AWA(Lold.in(qL), Lold.out(qL), Lold.mid(qL), s1, s2, qloc, k, qOp, Abra, Aket, ix);
			if (FOUND_MATCH == true)
			{
				for(size_t n=0; n<ix.size(); n++ )
				{
					qarray3<Symmetry::Nq> quple = get<0>(ix[n]);
					swap(quple[0], quple[1]);
					size_t qAbra = get<1>(ix[n]);
					size_t qAket = get<2>(ix[n]);
					if constexpr ( Symmetry::NON_ABELIAN )
					{
						factor_cgc = Symmetry::coeff_buildL(Aket[s2].out[qAket], qloc[s2], Aket[s2].in[qAket],
															quple[2],            qOp[k],   Lold.mid(qL),
															Abra[s1].out[qAbra], qloc[s1], Abra[s1].in[qAbra]);
					}
					else
					{
						factor_cgc = 1.;
					}
					if (std::abs(factor_cgc) < ::mynumeric_limits<MpoScalar>::epsilon()) { continue; }
					for (int r=0; r<W[s1][s2][k].outerSize(); ++r)
						for (typename SparseMatrix<MpoScalar>::InnerIterator iW(W[s1][s2][k],r); iW; ++iW)
						{
							size_t a1 = iW.row();
							size_t a2 = iW.col();

							if (Lold.block[qL][a1][0].rows() != 0)
							{
								MatrixType Mtmp;
								optimal_multiply(factor_cgc*iW.value(),
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
									boost::multi_array<MatrixType,LEGLIMIT> Mtmpvec(boost::extents[W[s1][s2][k].cols()][1]);
									Mtmpvec[a2][0] = Mtmp;
									Lnew.push_back(quple, Mtmpvec);
								}
							}
						}
				}
			}
		}
	}
}

/**
 * Contracts a left transfer matrix \p Lold with two MpsQ tensors \p Abra, \p Aket and an MpoQ tensor \p W as follows:
 * \dotfile contractQ_L.dot
 * \param Lold
 * \param Abra
 * \param V
 * \param Aket
 * \param qloc : local basis
 * \param qOp : operator basis
 * \param Lnew : new transfer matrix to be written to
 */
template<typename Symmetry, typename MatrixType, typename MpoScalar>
void contract_L (const Tripod<Symmetry,MatrixType> &Lold, 
                 const vector<Biped<Symmetry,MatrixType> > &Abra, 
                 const unordered_map<tuple<size_t,size_t,size_t,qarray<Symmetry::Nq>,qarray<Symmetry::Nq> >,SparseMatrix<MpoScalar> > &V, 
                 const vector<Biped<Symmetry,MatrixType> > &Aket, 
                 const vector<qarray<Symmetry::Nq> > &qloc,
                 const vector<qarray<Symmetry::Nq> > &qOp, 
                 Tripod<Symmetry,MatrixType> &Lnew)
{
	std::array<typename Symmetry::qType,3> qCheck;
	MpoScalar factor_cgc;
	Lnew.clear();
	Lnew.setZero();
	
	for (size_t s1=0; s1<qloc.size(); ++s1)
	for (size_t s2=0; s2<qloc.size(); ++s2)
	for (size_t k=0; k<qOp.size(); ++k)
	{
		qCheck = {qloc[s2],qOp[k],qloc[s1]};
		if(!Symmetry::validate(qCheck)) {continue;}
		// cout << "squared: s1=" << qloc[s1] << ", s2=" << qloc[s2] << ", k=" << qOp[k] << ", Lold.dim=" << Lold.dim << endl;
		for (size_t qL=0; qL<Lold.dim; ++qL)
		{
			vector<tuple<qarray3<Symmetry::Nq>,size_t,size_t> > ix;
			bool FOUND_MATCH = AWA(Lold.in(qL), Lold.out(qL), Lold.mid(qL), s1, s2, qloc, k, qOp, Abra, Aket, ix);
			// cout << "squared: ix.size()=" << ix.size() << endl;
		
			if (FOUND_MATCH == true)
			{
				for(size_t n=0; n<ix.size(); n++ )
				{
					qarray3<Symmetry::Nq> quple = get<0>(ix[n]);
					swap(quple[0], quple[1]);
					size_t qAbra = get<1>(ix[n]);
					size_t qAket = get<2>(ix[n]);
					if constexpr ( Symmetry::NON_ABELIAN )
						{
							factor_cgc = Symmetry::coeff_buildL(Aket[s2].out[qAket],qloc[s2],Aket[s2].in[qAket],
																quple[2]           ,qOp[k]  ,Lold.mid(qL),
																Abra[s1].out[qAbra],qloc[s1],Abra[s1].in[qAbra]);
						}
					else
					{
						factor_cgc = 1.;
					}
					if (std::abs(factor_cgc) < ::mynumeric_limits<MpoScalar>::epsilon()) { continue; }
					auto key = make_tuple(s1,s2,k,Lold.mid(qL),quple[2]);
					// auto itV = V.find(key);
					// if(itV != V.end())
					// {
						for (int r=0; r<V.at(key).outerSize(); ++r)
						for (typename SparseMatrix<MpoScalar>::InnerIterator iW(V.at(key),r); iW; ++iW)
						{
							size_t a1 = iW.row();
							size_t a2 = iW.col();
				
							if (Lold.block[qL][a1][0].rows() != 0)
							{
								MatrixType Mtmp;
								// if(abs(Lold.block[qL][a1][0].sum()) < 1.e-7) {cout << "qL=" << qL << ", a1=" << a1 << endl << Lold.block[qL][a1][0] << endl << endl;}
								optimal_multiply(factor_cgc*iW.value(),
												 Abra[s1].block[qAbra].adjoint(),
												 Lold.block[qL][a1][0],
												 Aket[s2].block[qAket],
												 Mtmp);
								// if(quple[0]==qarray<Symmetry::Nq>{2} and quple[1]==qarray<Symmetry::Nq>{2} and (quple[2]==qarray<Symmetry::Nq>{1} or quple[2]==qarray<Symmetry::Nq>{3}) and (a2 == 13 or a2 == 22))
								// {
								// 	cout << "Mtmp for qL=0 and a2=" << a2 << " and a1=" << a1 <<  " is:" << endl << Mtmp << endl;
								// 	cout << "cgc=" << factor_cgc << ", Mpo=" << iW.value() << endl;
								// 	cout << "Abra[s1].block[qAbra].adjoint():" << endl << Abra[s1].block[qAbra].adjoint() << endl << endl;
								// 	cout << "Lold.block[qL][a1][0], in[qL]=" << Lold.in(qL) << ", out[qL]=" << Lold.out(qL)  <<  ", mid[qL]=" << Lold.mid(qL) << ", a1=" << a1 << ":" << endl << Lold.block[qL][a1][0] << endl << endl;
								// 	cout << "Aket[s2].block[qAket]:" << endl << Aket[s2].block[qAket] << endl << endl;
								// 	cout << endl << endl << endl;

								// }

								// if(quple[0]==qarray<Symmetry::Nq>{1} and quple[1]==qarray<Symmetry::Nq>{1} and quple[2]==qarray<Symmetry::Nq>{1} and a2 == 18)
								// {
								// 	cout << "Mtmp for qL=0 and a2=18 and a1=" << a1 <<  " is:" << endl << Mtmp << endl;
								// 	cout << "cgc=" << factor_cgc << ", Mpo=" << iW.value() << endl;
								// 	cout << "Abra[s1].block[qAbra].adjoint():" << endl << Abra[s1].block[qAbra].adjoint() << endl << endl;
								// 	cout << "Lold.block[qL][a1][0], in[qL]=" << Lold.in(qL) << ", out[qL]=" << Lold.out(qL)  <<  ", mid[qL]=" << Lold.mid(qL) << ", a1=" << a1 << ":" << endl << Lold.block[qL][a1][0] << endl << endl;
								// 	cout << "Aket[s2].block[qAket]:" << endl << Aket[s2].block[qAket] << endl << endl;
								// 	cout << endl << endl << endl;

								// }
								if(Mtmp.norm() < ::mynumeric_limits<MpoScalar>::epsilon())
								{
									// cout << "cgc=" << factor_cgc << ", Mpo=" << iW.value() << endl;
									// cout << "Abra[s1].block[qAbra].adjoint():" << endl << Abra[s1].block[qAbra].adjoint() << endl << endl;
									// cout << "Lold.block[qL][a1][0], in[qL]=" << Lold.in(qL) << ", out[qL]=" << Lold.out(qL)  <<  ", mid[qL]=" << Lold.mid(qL) << ", a1=" << a1 << ":" << endl << Lold.block[qL][a1][0] << endl << endl;
									// cout << "Aket[s2].block[qAket]:" << endl << Aket[s2].block[qAket] << endl << endl;
									// cout << endl << endl << endl;
									continue;
								}
					
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
										if(Lnew.block[it->second][a2][0].norm() < ::mynumeric_limits<MpoScalar>::epsilon())
										{
											// Lnew.block[it->second][a2][0].resize(0,0);
											continue;
										}
									}
								}
								else
								{
									boost::multi_array<MatrixType,LEGLIMIT> Mtmpvec(boost::extents[V.at(key).cols()][1]);
									Mtmpvec[a2][0] = Mtmp;
									Lnew.push_back(quple, Mtmpvec);
								}
							}
						}
					// }
					// else {cout << "This happens?" << endl;					cout << "s1=" << qloc[s1] << ", s2=" << qloc[s2] << ", k=" << k << ", qK=" << qOp[k]
					// 		   << ", ql=" << Lold.mid(qL) << ", qr=" << quple[2] << endl << endl;
					// }
				}
			}
		}
	}
}

/**
 * Contracts a right transfer matrix \p Rold with two MpsQ tensors \p Abra, \p Aket and an MpoQ tensor \p W as follows:
 * \dotfile contractQ_R.dot
 * \param Rold
 * \param Abra
 * \param W
 * \param Aket
 * \param qloc : local basis
 * \param qOp : operator basis
 * \param Rnew : new transfer matrix to be written to
 */
template<typename Symmetry, typename MatrixType, typename MpoScalar>
void contract_R (const Tripod<Symmetry,MatrixType> &Rold,
                 const vector<Biped<Symmetry,MatrixType> > &Abra, 
                 const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W, 
                 const vector<Biped<Symmetry,MatrixType> > &Aket, 
                 const vector<qarray<Symmetry::Nq> > &qloc,
                 const vector<qarray<Symmetry::Nq> > &qOp, 
                 Tripod<Symmetry,MatrixType> &Rnew)
{
	std::array<typename Symmetry::qType,3> qCheck;
	MpoScalar factor_cgc;
	Rnew.clear();
	Rnew.setZero();
	
	for (size_t s1=0; s1<qloc.size(); ++s1)
	for (size_t s2=0; s2<qloc.size(); ++s2)
	for (size_t k=0; k<qOp.size(); ++k)
	{
		qCheck = {qloc[s2],qOp[k],qloc[s1]};
		if(!Symmetry::validate(qCheck)) {continue;}
		
		for (size_t qR=0; qR<Rold.dim; ++qR)
		{
			auto qRouts = Symmetry::reduceSilent(Rold.out(qR),Symmetry::flip(qloc[s1]));
			auto qRins = Symmetry::reduceSilent(Rold.in(qR),Symmetry::flip(qloc[s2]));
			
			for(const auto& qRout : qRouts)
			for(const auto& qRin : qRins)
			{
				qarray2<Symmetry::Nq> cmp1 = {qRout, Rold.out(qR)};
				qarray2<Symmetry::Nq> cmp2 = {qRin, Rold.in(qR)};
				
				auto q1 = Abra[s1].dict.find(cmp1);
				auto q2 = Aket[s2].dict.find(cmp2);
				
				if (q1!=Abra[s1].dict.end() and 
				    q2!=Aket[s2].dict.end())
				{
					qarray<Symmetry::Nq> new_qin  = Aket[s2].in[q2->second]; // A.in
					qarray<Symmetry::Nq> new_qout = Abra[s1].in[q1->second]; // A†.out = A.in
					auto qRmids = Symmetry::reduceSilent(Rold.mid(qR),Symmetry::flip(qOp[k]));
					
					for(const auto& new_qmid : qRmids)
					{
						qarray3<Symmetry::Nq> quple = {new_qin, new_qout, new_qmid};
						if constexpr (Symmetry::NON_ABELIAN)
						{
							factor_cgc = Symmetry::coeff_buildR(Aket[s2].out[q2->second],qloc[s2],Aket[s2].in[q2->second],
							                                    Rold.mid(qR),qOp[k],quple[2],
							                                    Abra[s1].out[q1->second],qloc[s1],Abra[s1].in[q1->second]);
						}
						else
						{
							factor_cgc = 1.;
						}
						if (std::abs(factor_cgc) < ::mynumeric_limits<MpoScalar>::epsilon()) { continue; }
						
						for (int r=0; r<W[s1][s2][k].outerSize(); ++r)
							for (typename SparseMatrix<MpoScalar>::InnerIterator iW(W[s1][s2][k],r); iW; ++iW)
							{
								size_t a1 = iW.row();
								size_t a2 = iW.col();
								
								if (Rold.block[qR][a2][0].rows() != 0)
								{
									MatrixType Mtmp;
									optimal_multiply(factor_cgc*iW.value(),
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
										boost::multi_array<MatrixType,LEGLIMIT> Mtmpvec(boost::extents[W[s1][s2][k].rows()][1]);
										Mtmpvec[a1][0] = Mtmp;
										Rnew.push_back(quple, Mtmpvec);
									}
								}
							}
					}
				}
			}
		}
	}
}

template<typename Symmetry, typename MatrixType>
void contract_L (const Biped<Symmetry,MatrixType> &Lold, 
                 const vector<Biped<Symmetry,MatrixType> > &Abra, 
                 const vector<Biped<Symmetry,MatrixType> > &Aket, 
                 const vector<qarray<Symmetry::Nq> > &qloc,
                 Biped<Symmetry,MatrixType> &Lnew)
{
	Lnew.clear();
	Lnew.setZero();
	
	for (size_t s=0; s<qloc.size(); ++s)
	{
		for (size_t qL=0; qL<Lold.dim; ++qL)
		{
			vector<tuple<qarray2<Symmetry::Nq>,size_t,size_t> > ix;
			bool FOUND_MATCH = AA(Lold.in[qL], Lold.out[qL], s, qloc, Abra, Aket, ix);
			
			if (FOUND_MATCH)
			{
				for (size_t n=0; n<ix.size(); n++)
				{
					qarray2<Symmetry::Nq> quple = get<0>(ix[n]);
					swap(quple[0], quple[1]);
					if (!Symmetry::validate(quple)) {continue;}
					size_t qAbra = get<1>(ix[n]);
					size_t qAket = get<2>(ix[n]);
					{
						if (Lold.block[qL].rows() != 0)
						{
							MatrixType Mtmp;
							optimal_multiply(1.,
							                 Abra[s].block[qAbra].adjoint(),
							                 Lold.block[qL],
							                 Aket[s].block[qAket],
							                 Mtmp);
							
							auto it = Lnew.dict.find(quple);
							if (it != Lnew.dict.end())
							{
								if (Lnew.block[it->second].rows() != Mtmp.rows() or 
									Lnew.block[it->second].cols() != Mtmp.cols())
								{
									Lnew.block[it->second] = Mtmp;
								}
								else
								{
									Lnew.block[it->second] += Mtmp;
								}
							}
							else
							{
								Lnew.push_back(quple, Mtmp);
							}
						}
					}
				}
			}
		}
	}
}

template<typename Symmetry, typename MatrixType>
void contract_R (const Biped<Symmetry,MatrixType> &Rold,
                 const vector<Biped<Symmetry,MatrixType> > &Abra, 
                 const vector<Biped<Symmetry,MatrixType> > &Aket, 
                 const vector<qarray<Symmetry::Nq> > &qloc,
                 Biped<Symmetry,MatrixType> &Rnew)
{
	Rnew.clear();
	Rnew.setZero();
	
	for (size_t s=0; s<qloc.size(); ++s)
	{
		for (size_t qR=0; qR<Rold.dim; ++qR)
		{
			auto qRouts = Symmetry::reduceSilent(Rold.out[qR],Symmetry::flip(qloc[s]));
			auto qRins = Symmetry::reduceSilent(Rold.in[qR],Symmetry::flip(qloc[s]));
			
			for(const auto& qRout : qRouts)
			for(const auto& qRin : qRins)
			{
				qarray2<Symmetry::Nq> cmp1 = {qRout, Rold.out[qR]};
				qarray2<Symmetry::Nq> cmp2 = {qRin, Rold.in[qR]};
				auto q1 = Abra[s].dict.find(cmp1);
				auto q2 = Aket[s].dict.find(cmp2);
				
				if (q1!=Abra[s].dict.end() and 
				    q2!=Aket[s].dict.end())
				{
					qarray<Symmetry::Nq> new_qin  = Aket[s].in[q2->second]; // A.in
					qarray<Symmetry::Nq> new_qout = Abra[s].in[q1->second]; // A†.out = A.in
					qarray2<Symmetry::Nq> quple = {new_qin, new_qout};
					if (!Symmetry::validate(quple)) {continue;}
					
					double factor_cgc = Symmetry::coeff_rightOrtho(Abra[s].out[q1->second],
					                                               Abra[s].in[q1->second]);
					
					if (Rold.block[qR].rows() != 0)
					{
						MatrixType Mtmp;
						optimal_multiply(factor_cgc,
						                 Aket[s].block[q2->second],
						                 Rold.block[qR],
						                 Abra[s].block[q1->second].adjoint(),
						                 Mtmp);
						
						auto it = Rnew.dict.find(quple);
						if (it != Rnew.dict.end())
						{
							if (Rnew.block[it->second].rows() != Mtmp.rows() or 
								Rnew.block[it->second].cols() != Mtmp.cols())
							{
								Rnew.block[it->second] = Mtmp;
							}
							else
							{
								Rnew.block[it->second] += Mtmp;
							}
						}
						else
						{
							Rnew.push_back(quple, Mtmp);
						}
					}
				}
			}
		}
	}
}

/**
 * Calculates the contraction between a left transfer matrix \p L, 
 * two MpsQ tensors \p Abra, \p Aket, an MpoQ tensor \p W and a right transfer matrix \p R. Not really that much useful.
 * \param L
 * \param Abra
 * \param W
 * \param Aket
 * \param R
 * \param qloc : local basis
 * \param qOp : operator basis
 * \returns : result of contraction
 * \warning Not working for non-abelian symmetries.
 */
template<typename Symmetry, typename Scalar>
Scalar contract_LR (const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &L,
                    const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Abra, 
                    const vector<vector<vector<SparseMatrixXd> > > &W, 
                    const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Aket, 
                    const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &R, 
                    const vector<qarray<Symmetry::Nq> > &qloc,
                    const vector<qarray<Symmetry::Nq> > &qOp)
{
	Scalar res = 0.;
	std::array<typename Symmetry::qType,3> qCheck;
	Scalar factor_cgc;

	for (size_t s1=0; s1<qloc.size(); ++s1)
	for (size_t s2=0; s2<qloc.size(); ++s2)
	for (size_t k=0; k<qOp.size(); ++k)
	{
		qCheck = {qloc[s2],qOp[k],qloc[s1]};
		if(!Symmetry::validate(qCheck)) {continue;}
		for (size_t qL=0; qL<L.dim; ++qL)
		{
			vector<tuple<qarray3<Symmetry::Nq>,size_t,size_t> > ix;
			bool FOUND_MATCH = AA(L.in(qL), L.out(qL), L.mid(qL), s1, s2, qloc, k, qOp, Abra, Aket, ix);
			if (FOUND_MATCH == true)
			{
				for(size_t n=0; n<ix.size(); n++ )
				{
					qarray3<Symmetry::Nq> quple = get<0>(ix[n]);
					auto qR = R.dict.find(quple);
					
					if (qR != R.dict.end())
					{
						swap(quple[0], quple[1]);
						size_t qAbra = get<1>(ix[n]);
						size_t qAket = get<2>(ix[n]);
						if constexpr ( Symmetry::NON_ABELIAN )
						{
							factor_cgc = Symmetry::coeff_buildL(Aket[s2].out[qAket],qloc[s2],Aket[s2].in[qAket],
																quple[2],qOp[k],L.mid(qL),
																Abra[s1].out[qAbra],qloc[s1],Abra[s1].in[qAbra]);
						}
						else
						{
							factor_cgc = 1.;
						}
						if (std::abs(factor_cgc) < ::mynumeric_limits<Scalar>::epsilon()) { continue; }
						
						for (int r=0; r<W[s1][s2][k].outerSize(); ++r)
						for (SparseMatrixXd::InnerIterator iW(W[s1][s2][k],r); iW; ++iW)
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

/**
 * Calculates the contraction between a right transfer matrix \p Rold, two MpsQ tensors \p Abra, \p Aket and two MpoQ tensors \p Wbot, \p Wtop.
 * Needed, for example, when calculating \f$\left<H^2\right>\f$ and no MpoQ represenation of \f$H^2\f$ is available.
*/
template<typename Symmetry, typename MatrixType, typename MpoScalar>
void contract_R (const Tripod<Symmetry,MatrixType> &Rold,
                 const vector<Biped<Symmetry,MatrixType> > &Abra,
				 const vector<vector<vector<SparseMatrix<MpoScalar> > > > &Wbot,
                 const vector<vector<vector<SparseMatrix<MpoScalar> > > > &Wtop,
                 const vector<Biped<Symmetry,MatrixType> > &Aket,
                 const vector<qarray<Symmetry::Nq> > &qloc,
				 const vector<qarray<Symmetry::Nq> > &qOpBot,
                 const vector<qarray<Symmetry::Nq> > &qOpTop,
				 const Qbasis<Symmetry> &baseRightBot,
				 const Qbasis<Symmetry> &baseRightTop,
				 const Qbasis<Symmetry> &baseLeftBot,
				 const Qbasis<Symmetry> &baseLeftTop,
                 Tripod<Symmetry,MatrixType> &Rnew)
{
	// cout << baseRightTop << endl << baseLeftTop << endl;
	auto leftTopQs = baseLeftTop.unordered_qs();
	auto leftBotQs = baseLeftBot.unordered_qs();

	auto TensorBaseRight = baseRightBot.combine(baseRightTop);
	auto TensorBaseLeft = baseLeftBot.combine(baseLeftTop);

	std::array<typename Symmetry::qType,3> qCheck;

	MpoScalar factor_cgc, factor_merge, factor_check;
	Rnew.clear();
	Rnew.setZero();

	for (size_t s1=0; s1<qloc.size(); ++s1)
	for (size_t s2=0; s2<qloc.size(); ++s2)
	for (size_t s3=0; s3<qloc.size(); ++s3)
	for (size_t k1=0; k1<qOpTop.size(); ++k1)
	for (size_t k2=0; k2<qOpBot.size(); ++k2)
	{
		qCheck = {qloc[s3],qOpTop[k1],qloc[s2]};
		if(!Symmetry::validate(qCheck)) {continue;}
		qCheck = {qloc[s2],qOpBot[k2],qloc[s1]};
		if(!Symmetry::validate(qCheck)) {continue;}

		auto ks = Symmetry::reduceSilent(qOpTop[k1],qOpBot[k2]);
		for(const auto& k : ks)
		{
			qCheck = {qloc[s3],k,qloc[s1]};
			if(!Symmetry::validate(qCheck)) {continue;}

			factor_check = Symmetry::coeff_Apair(qloc[s1],qOpBot[k2],qloc[s2],
												 qOpTop[k1],qloc[s3],k);
			if (std::abs(factor_check) < ::mynumeric_limits<MpoScalar>::epsilon()) { continue; }
			for (size_t qR=0; qR<Rold.dim; ++qR)
			{
				auto qRouts = Symmetry::reduceSilent(Rold.out(qR),Symmetry::flip(qloc[s1]));
				auto qRins = Symmetry::reduceSilent(Rold.in(qR),Symmetry::flip(qloc[s3]));
				auto qrightAuxs = Sym::split<Symmetry>(Rold.mid(qR),baseRightTop.qs(),baseRightBot.qs());
				
				for(const auto& qRout : qRouts)
					for(const auto& qRin : qRins)
					{
						auto q1 = Abra[s1].dict.find({qRout, Rold.out(qR)});
						auto q3 = Aket[s3].dict.find({qRin, Rold.in(qR)});
						if (q1!=Abra[s1].dict.end() and q3!=Aket[s3].dict.end())
						{
							auto qRmids = Symmetry::reduceSilent(Rold.mid(qR),Symmetry::flip(k));
							for(const auto& new_qmid : qRmids)
							{
								qarray3<Symmetry::Nq> quple = {Aket[s3].in[q3->second], Abra[s1].in[q1->second], new_qmid};
								factor_cgc = Symmetry::coeff_buildR(Aket[s3].out[q3->second],qloc[s3],Aket[s3].in[q3->second],
																	Rold.mid(qR),k,new_qmid,
																	Abra[s1].out[q1->second],qloc[s1],Abra[s1].in[q1->second]);
								if (std::abs(factor_cgc) < std::abs(::mynumeric_limits<MpoScalar>::epsilon())) { continue; }
								for(const auto& [qrightAux,qrightAuxP] : qrightAuxs)
								{
									Eigen::Index left2=TensorBaseRight.leftAmount(Rold.mid(qR),{qrightAuxP, qrightAux});
									auto qleftAuxs = Symmetry::reduceSilent(qrightAux,Symmetry::flip(qOpTop[k1]));
									for(const auto& qleftAux : qleftAuxs)
									{
										if(auto it=leftTopQs.find(qleftAux) != leftTopQs.end())
										{
											auto qleftAuxPs = Symmetry::reduceSilent(qrightAuxP,Symmetry::flip(qOpBot[k2]));
											for(const auto& qleftAuxP : qleftAuxPs)
											{
												if(auto it=leftBotQs.find(qleftAuxP) != leftBotQs.end())
												{
													factor_merge = Symmetry::coeff_buildR(qrightAuxP,qrightAux,Rold.mid(qR),
																						 qOpBot[k2],qOpTop[k1],k,
																						 qleftAuxP,qleftAux,new_qmid);
													if (std::abs(factor_merge) < std::abs(::mynumeric_limits<MpoScalar>::epsilon())) { continue; }
													Eigen::Index left1=TensorBaseLeft.leftAmount(new_qmid,{qleftAuxP, qleftAux});
													for (int ktop=0; ktop<Wtop[s2][s3][k1].outerSize(); ++ktop)
													for (typename SparseMatrix<MpoScalar>::InnerIterator iWtop(Wtop[s2][s3][k1],ktop); iWtop; ++iWtop)
													for (int kbot=0; kbot<Wbot[s1][s2][k2].outerSize(); ++kbot)
													for (typename SparseMatrix<MpoScalar>::InnerIterator iWbot(Wbot[s1][s2][k2],kbot); iWbot; ++iWbot)
													{
														size_t br = iWbot.row();
														size_t bc = iWbot.col();
														size_t tr = iWtop.row();
														size_t tc = iWtop.col();
														MpoScalar Wfactor = iWbot.value() * iWtop.value();

														size_t a1 = left1+br*Wtop[s2][s3][k1].rows()+tr;
														size_t a2 = left2+bc*Wtop[s2][s3][k1].cols()+tc;

														if (Rold.block[qR][a2][0].rows() != 0)
														{
															MatrixType Mtmp;
															optimal_multiply(factor_check*factor_merge*factor_cgc*Wfactor,
																			 Aket[s3].block[q3->second],
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
																boost::multi_array<MatrixType,LEGLIMIT> Mtmpvec(boost::extents[TensorBaseLeft.inner_dim(new_qmid)][1]);
																Mtmpvec[a1][0] = Mtmp;
																Rnew.push_back(quple, Mtmpvec);
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
					}
			}
		}
	}
}

/**
 * Calculates the contraction between a left transfer matrix \p Lold, two MpsQ tensors \p Abra, \p Aket and two MpoQ tensors \p Wbot, \p Wtop.
 * Needed, for example, when calculating \f$\left<H^2\right>\f$ and no MpoQ represenation of \f$H^2\f$ is available.
 */
template<typename Symmetry, typename MatrixType, typename MpoScalar>
void contract_L (const Multipede<4,Symmetry,MatrixType> &Lold, 
                 const vector<Biped<Symmetry,MatrixType> > &Abra, 
                 const vector<vector<vector<SparseMatrix<MpoScalar> > > > &Wbot, 
                 const vector<vector<vector<SparseMatrix<MpoScalar> > > > &Wtop, 
                 const vector<Biped<Symmetry,MatrixType> > &Aket, 
                 const vector<qarray<Symmetry::Nq> > &qloc,
				 const vector<qarray<Symmetry::Nq> > &qOpBot,
                 const vector<qarray<Symmetry::Nq> > &qOpTop,
                 Multipede<4,Symmetry,MatrixType> &Lnew)
{
	std::array<typename Symmetry::qType,3> qCheck;

	Lnew.setZero();
	
	for (size_t s1=0; s1<qloc.size(); ++s1)
	for (size_t s2=0; s2<qloc.size(); ++s2)
	for (size_t s3=0; s3<qloc.size(); ++s3)
	for (size_t k1=0; k1<qOpBot.size(); ++k1)
	for (size_t k2=0; k2<qOpTop.size(); ++k2)
	{
		qCheck = {qloc[s2],qOpBot[k1],qloc[s1]};
		if(!Symmetry::validate(qCheck)) {continue;}
		qCheck = {qloc[s3],qOpTop[k2],qloc[s2]};
		if(!Symmetry::validate(qCheck)) {continue;}
		for (size_t qL=0; qL<Lold.dim; ++qL)
		{
			tuple<qarray4<Symmetry::Nq>,size_t,size_t> ix;
			bool FOUND_MATCH = AWWA(Lold.in(qL), Lold.out(qL), Lold.bot(qL), Lold.top(qL), 
									s1, s2, s3, qloc, k1, qOpBot, k2, qOpTop, Abra, Aket, ix);
			auto   quple = get<0>(ix);
			swap(quple[0],quple[1]);
			size_t qAbra = get<1>(ix);
			size_t qAket = get<2>(ix);
		
			if (FOUND_MATCH == true)
			{
				for (int kbot=0; kbot<Wbot[s1][s2][k1].outerSize(); ++kbot)
					for (typename SparseMatrix<MpoScalar>::InnerIterator iWbot(Wbot[s1][s2][k1],kbot); iWbot; ++iWbot)
						for (int ktop=0; ktop<Wtop[s2][s3][k2].outerSize(); ++ktop)
							for (typename SparseMatrix<MpoScalar>::InnerIterator iWtop(Wtop[s2][s3][k2],ktop); iWtop; ++iWtop)
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
											size_t bcols = Wbot[s1][s2][k1].cols();
											size_t tcols = Wtop[s2][s3][k2].cols();
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
}

/**For details see: Stoudenmire, White (2010)*/
template<typename Symmetry, typename MatrixType, typename MpoScalar>
void contract_C0 (vector<qarray<Symmetry::Nq> > qloc,
				  vector<qarray<Symmetry::Nq> > qOp,
                  const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W, 
                  const vector<Biped<Symmetry,MatrixType> >   &Aket, 
                  vector<Tripod<Symmetry,MatrixType> >        &Cnext)
{
	Cnext.clear();
	Cnext.resize(qloc.size());
	std::array<typename Symmetry::qType,3> qCheck;

	for (size_t s2=0; s2<qloc.size(); ++s2)
	{
		qarray2<Symmetry::Nq> cmpA = {Symmetry::qvacuum(), Symmetry::qvacuum()+qloc[s2]};
		auto qA = Aket[s2].dict.find(cmpA);
		
		if (qA != Aket[s2].dict.end())
		{
			for (size_t s1=0; s1<qloc.size(); ++s1)
			for (size_t k=0; k<qOp.size(); ++k)
			{
				qCheck = {qloc[s2],qOp[k],qloc[s1]};
				if(!Symmetry::validate(qCheck)) {continue;}
				for (int r=0; r<W[s1][s2][k].outerSize(); ++r)
				for (typename SparseMatrix<MpoScalar>::InnerIterator iW(W[s1][s2][k],r); iW; ++iW)
				{
					MatrixType Mtmp = iW.value() * Aket[s2].block[qA->second];
					
					qarray3<Symmetry::Nq> cmpC = {Symmetry::qvacuum(), Aket[s2].out[qA->second], Symmetry::qvacuum()+qloc[s1]-qloc[s2]};
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
						boost::multi_array<MatrixType,LEGLIMIT> Mtmpvec(boost::extents[W[s1][s2][k].cols()][1]);
						Mtmpvec[iW.col()][0] = Mtmp;
						Cnext[s1].push_back({Symmetry::qvacuum(), Aket[s2].out[qA->second], Symmetry::qvacuum()+qloc[s1]-qloc[s2]}, Mtmpvec);
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
				 vector<qarray<Symmetry::Nq> > qOp,
                 const vector<Biped<Symmetry,MatrixType> >   &Abra, 
                 const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W, 
                 const vector<Biped<Symmetry,MatrixType> >   &Aket, 
                 const vector<Tripod<Symmetry,MatrixType> >  &C, 
                 vector<Tripod<Symmetry,MatrixType> >        &Cnext)
{
	Cnext.clear();
	Cnext.resize(qloc.size());
	std::array<typename Symmetry::qType,3> qCheck;

	for (size_t s=0; s<qloc.size(); ++s)
	for (size_t qC=0; qC<C[s].dim; ++qC)
	{
		qarray2<Symmetry::Nq> cmpU = {C[s].in(qC), C[s].in(qC)+qloc[s]};
		auto qU = Abra[s].dict.find(cmpU);
		
		if (qU != Abra[s].dict.end())
		{
			for (size_t s1=0; s1<qloc.size(); ++s1)
			for (size_t s2=0; s2<qloc.size(); ++s2)
			for (size_t k=0; k<qOp.size(); ++k)
			{
				qCheck = {qloc[s2],qOp[k],qloc[s1]};
				if(!Symmetry::validate(qCheck)) {continue;}

				qarray2<Symmetry::Nq> cmpA = {C[s].out(qC), C[s].out(qC)+qloc[s2]};
				auto qA = Aket[s2].dict.find(cmpA);
				
				if (qA != Aket[s2].dict.end())
				{
					for (int r=0; r<W[s1][s2][k].outerSize(); ++r)
					for (typename SparseMatrix<MpoScalar>::InnerIterator iW(W[s1][s2][k],r); iW; ++iW)
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


template<typename Symmetry, typename Scalar>
void contract_AA (const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &A1, 
                  vector<qarray<Symmetry::Nq> > qloc1, 
                  const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &A2, 
                  vector<qarray<Symmetry::Nq> > qloc2, 
                  vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Apair)
{
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
						Scalar factor_cgc = Symmetry::coeff_Apair(A1[s1].in[q1], qloc1[s1], A1[s1].out[q1], 
						                                          qloc2[s2], A2[s2].out[q2->second], qmerge);
						
						if (abs(factor_cgc) > abs(mynumeric_limits<Scalar>::epsilon()))
						{
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

#endif
