#ifndef MULTIPEDE_CONTRACTIONS_H_
#define MULTIPEDE_CONTRACTIONS_H_

#include "Biped.h"
#include "Multipede.h"
#include "qbasis.h"
#include "DmrgExternalQ.h"

namespace contractions {

	typedef Eigen::Index Index;
	template<Index Rank, typename Scalar> using TensorType = Eigen::Tensor<Scalar,Rank,Eigen::ColMajor,Index>;
	template<typename Scalar> using MatrixType = Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic>;
	
	template<typename Symmetry, typename Scalar>
	void buildL( const Tripod<Symmetry,MatrixType<Scalar> > &Lold, const std::vector<Biped<Symmetry,MatrixType<Scalar> > > &A,
				 const std::vector<Biped<Symmetry,MatrixType<Scalar> > > &B, const std::vector<std::vector<std::vector<SparseMatrixXd> > > &W,
				 const std::vector<typename Symmetry::qType>& qloc, const std::vector<typename Symmetry::qType>& qOp,
				 Tripod<Symmetry,MatrixType<Scalar> > &Lnew )
	{
		typedef typename Symmetry::qType qType;

		Lnew.clear();
		std::array<qType,3> qCheck,totIndex;
		Scalar factor_cgc;
		for (std::size_t s1=0; s1<qloc.size(); ++s1)
			for (std::size_t s2=0; s2<qloc.size(); ++s2)
				for (std::size_t k=0; k<qOp.size(); ++k)
				{
					qCheck = {qloc[s2],qOp[k],qloc[s1]};
					if(!Symmetry::validate(qCheck)) {continue;}
					for (std::size_t nu=0; nu<Lold.size(); nu++)
					{
						auto pNews = Symmetry::reduceSilent(Lold.index[nu][2],qOp[k]);
						for (const auto& p : pNews)
						{
							auto qAs = Symmetry::reduceSilent(Lold.index[nu][0],qloc[s2]);
							auto qBs = Symmetry::reduceSilent(Lold.index[nu][1],qloc[s1]);
							for (const auto& qA : qAs)
							{
								auto itA = A[s2].dict.find({{Lold.index[nu][0],qA}});
								if (itA == A[s2].dict.end()) {continue;}
								for (const auto& qB : qBs)
								{
									auto itB = B[s1].dict.find({{Lold.index[nu][1],qB}});
									if (itB == B[s1].dict.end()) {continue;}
									if constexpr ( Symmetry::SPECIAL )
										{
											factor_cgc = Symmetry::coeff_buildL(A[s2].out[itA->second],qloc[s2],A[s2].in[itA->second],
																				p,qOp[k],Lold.index[nu][2],
																				B[s1].out[itB->second],qloc[s1],B[s1].in[itB->second]);
										}
									else if constexpr ( Symmetry::HAS_CGC )
										{
											factor_cgc = 1.;
										}
									else
									{
										factor_cgc = 1.;
									}
									if (std::abs(factor_cgc) < ::numeric_limits<Scalar>::epsilon()) { continue; }
									totIndex = {A[s2].out[itA->second],B[s1].out[itB->second],p};
									for (int spInd=0; spInd<W[s1][s2][k].outerSize(); ++spInd)
										for (typename SparseMatrix<Scalar>::InnerIterator iW(W[s1][s2][k],spInd); iW; ++iW)
										{
											Index a1 = iW.row();
											Index a2 = iW.col();
				
											if (Lold.block[nu][a1][0].rows() != 0)
											{
												MatrixType<Scalar> Mtmp;
												optimal_multiply(factor_cgc*iW.value(),
																 B[s1].block[itB->second].adjoint(),
																 Lold.block[nu][a1][0].adjoint(),
																 A[s2].block[itA->second],
																 Mtmp);
												MatrixType<Scalar> Mtmp2 = Mtmp.adjoint();
												
												auto it = Lnew.dict.find(totIndex);
												if (it != Lnew.dict.end())
												{
													if (Lnew.block[it->second][a2][0].rows() != Mtmp2.rows() or 
														Lnew.block[it->second][a2][0].cols() != Mtmp2.cols())
													{
														Lnew.block[it->second][a2][0] = Mtmp2;
													}
													else
													{
														Lnew.block[it->second][a2][0] += Mtmp2;
													}
												}
												else
												{
													boost::multi_array<MatrixType<Scalar>,LEGLIMIT> Mtmpvec(boost::extents[W[s1][s2][k].cols()][1]);
													Mtmpvec[a2][0] = Mtmp2;
													Lnew.push_back(totIndex, Mtmpvec);
												}
											}
										}
								}
							}
						}
					}
				}
		return;
	}

	template<typename Symmetry, typename Scalar>
	void buildR( const Tripod<Symmetry,MatrixType<Scalar> > &Rold, const std::vector<Biped<Symmetry,MatrixType<Scalar> > > &A,
				 const std::vector<Biped<Symmetry,MatrixType<Scalar> > > &B, const std::vector<std::vector<std::vector<SparseMatrixXd> > > &W,
				 const std::vector<typename Symmetry::qType>& qloc, const std::vector<typename Symmetry::qType>& qOp,
				 Tripod<Symmetry,MatrixType<Scalar> > &Rnew )
	{
		typedef typename Symmetry::qType qType;

		Rnew.clear();
		std::array<qType,3> qCheck,totIndex;
		Scalar factor_cgc;
		for (std::size_t s1=0; s1<qloc.size(); ++s1)
			for (std::size_t s2=0; s2<qloc.size(); ++s2)
				for (std::size_t k=0; k<qOp.size(); ++k)
				{
					qCheck = {qloc[s2],qOp[k],qloc[s1]};
					if(!Symmetry::validate(qCheck)) {continue;}
					for (std::size_t nu=0; nu<Rold.size(); nu++)
					{
						auto pNews = Symmetry::reduceSilent(Rold.index[nu][2],Symmetry::flip(qOp[k]));
						for (const auto& p : pNews)
						{
							auto qAs = Symmetry::reduceSilent(Rold.index[nu][0],Symmetry::flip(qloc[s2]));
							auto qBs = Symmetry::reduceSilent(Rold.index[nu][1],Symmetry::flip(qloc[s1]));
							for (const auto& qA : qAs)
							{
								auto itA = A[s2].dict.find({{qA,Rold.index[nu][0]}});
								if (itA == A[s2].dict.end()) {continue;}
								for (const auto& qB : qBs)
								{
									auto itB = B[s1].dict.find({{qB,Rold.index[nu][1]}});
									if (itB == B[s1].dict.end()) {continue;}
									if constexpr ( Symmetry::SPECIAL )
										{
											factor_cgc = Symmetry::coeff_buildR(A[s2].out[itA->second],qloc[s2],A[s2].in[itA->second],
																				Rold.index[nu][2],qOp[k],p,
																				B[s1].out[itB->second],qloc[s1],B[s1].in[itB->second]);
										}
									else if constexpr ( Symmetry::HAS_CGC )
										{
											factor_cgc = 1.;
										}
									else
									{
										factor_cgc = 1.;
									}
									if (std::abs(factor_cgc) < ::numeric_limits<Scalar>::epsilon()) { continue; }
									totIndex = {A[s2].in[itA->second],B[s1].in[itB->second],p};
									for (int spInd=0; spInd<W[s1][s2][k].outerSize(); ++spInd)
										for (typename SparseMatrix<Scalar>::InnerIterator iW(W[s1][s2][k],spInd); iW; ++iW)
										{
											Index a1 = iW.row();
											Index a2 = iW.col();

											if (Rold.block[nu][a2][0].rows() != 0)
											{
												MatrixType<Scalar> Mtmp;
												optimal_multiply(factor_cgc*iW.value(),
																 A[s2].block[itA->second],
																 Rold.block[nu][a2][0],
																 B[s1].block[itB->second].adjoint(),
																 Mtmp);												
												auto it = Rnew.dict.find(totIndex);
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
													boost::multi_array<MatrixType<Scalar>,LEGLIMIT> Mtmpvec(boost::extents[W[s1][s2][k].rows()][1]);
													Mtmpvec[a1][0] = Mtmp;
													Rnew.push_back(totIndex, Mtmpvec);
												}
											}
										}
								}
							}
						}
					}
				}
		return;
	}

	template<typename Symmetry, typename Scalar>
	void HPsi( const Tripod<Symmetry,MatrixType<Scalar> > &L, const Tripod<Symmetry,MatrixType<Scalar> > &R,
			   const std::vector<Biped<Symmetry,MatrixType<Scalar> > > &Aold, const std::vector<std::vector<std::vector<SparseMatrixXd> > > &W,
			   const std::vector<std::array<std::size_t,2> > &qlhs, const std::vector<std::vector<std::array<std::size_t,5> > > &qrhs,
			   const std::vector<std::vector<Scalar> > &factor_cgcs, std::vector<Biped<Symmetry,MatrixType<Scalar> > > &Anew )
	{
		typedef typename Symmetry::qType qType;

		Anew = Aold;
		for (std::size_t s=0; s<Anew.size(); ++s) {Anew[s].setZero();}
		MatrixType<Scalar> Mtmp;
		for (std::size_t q=0; q<qlhs.size(); ++q)
		{
			std::size_t s2 = qlhs[q][0];
			std::size_t nu = qlhs[q][1];
			for (std::size_t p=0; p<qrhs[q].size(); ++p)
			{
				std::size_t s1 = qrhs[q][p][0];
				std::size_t mu = qrhs[q][p][1];
				std::size_t qL = qrhs[q][p][2];
				std::size_t qR = qrhs[q][p][3];
				std::size_t k = qrhs[q][p][4];
				for (int spInd=0; spInd<W[s1][s2][k].outerSize(); ++spInd)
					for (typename SparseMatrix<Scalar>::InnerIterator iW(W[s1][s2][k],spInd); iW; ++iW)
					{
						if (L.block[qL][iW.row()][0].rows() != 0 and R.block[qR][iW.col()][0].rows() != 0)
						{
							if (Anew[s1].block[mu].rows() != L.block[qL][iW.row()][0].cols() or
								Anew[s1].block[mu].cols() != R.block[qR][iW.col()][0].cols())
							{
								Anew[s1].block[mu].noalias() = factor_cgcs[q][p] * iW.value() * 
									(L.block[qL][iW.row()][0].adjoint() * 
									 Aold[s2].block[nu] * 
									 R.block[qR][iW.col()][0]);
							}
							else
							{
								Anew[s1].block[mu].noalias() += factor_cgcs[q][p] * iW.value() * 
									(L.block[qL][iW.row()][0].adjoint() * 
									 Aold[s2].block[nu] * 
									 R.block[qR][iW.col()][0]);
							}
						}
					}
			}
		}
	}
	
	// template<typename Symmetry, typename Scalar>
	// void buildL( const TripodQ<Symmetry,Scalar> &Lold, const TripodQ<Symmetry,Scalar> &A,
	// 			 const TripodQ<Symmetry,Scalar> &B, const MultipedeQ<4,Symmetry,Scalar,-2> &W, TripodQ<Symmetry,Scalar> &Lnew )
	// {
	// 	Lnew.clear();
	// 	typedef typename Symmetry::qType qType;
	// 	Scalar factor_cgc;
	// 	std::array<Eigen::IndexPair<Index>, 1> product_dims1= {Eigen::IndexPair<Index>(0,0)};
	// 	std::array<Eigen::IndexPair<Index>, 1> product_dims2= {Eigen::IndexPair<Index>(0,0)};
	// 	std::array<Eigen::IndexPair<Index>, 3> product_dims3= {Eigen::IndexPair<Index>(0,0),Eigen::IndexPair<Index>(4,2),Eigen::IndexPair<Index>(2,3)};

	// 	std::array<Index,3> new_dimsA;
	// 	TensorType<3,Scalar> Atmp;
		
	// 	std::array<qType,1> searchA, searchB;
	// 	std::array<qType,2> searchW;
	// 	std::array<Index,1> legA, legB; legA[0] = 0; legB[0] = 0;
	// 	std::array<Index,2> legW = {0,1};
	// 	std::array<Index,3> dummy_legs;
	// 	std::iota(dummy_legs.begin(), dummy_legs.end(), Index(0));

	// 	std::array<qType,3> totIndex, checkIndex;
		
	// 	for (std::size_t nu=0; nu<Lold.size(); nu++)
	// 	{
	// 		searchA[0] = Lold.index[nu][0];
	// 		searchB[0] = Lold.index[nu][1];
	// 		auto rangeA = A.dict.equal_range(searchA,legA);
	// 		auto rangeB = B.dict.equal_range(searchB,legB);
	// 		for (auto itA = rangeA.first; itA != rangeA.second; ++itA)
	// 			for (auto itB = rangeB.first; itB != rangeB.second; ++itB)
	// 			{
	// 				std::size_t mu = itA->second;
	// 				std::size_t kappa = itB->second;
	// 				searchW[0] = B.index[kappa][2];
	// 				searchW[1] = A.index[mu][2];
	// 				auto rangeW = W.dict.equal_range(searchW,legW);
	// 				for (auto itW = rangeW.first; itW != rangeW.second; itW++)
	// 				{
	// 					std::size_t lambda = itW->second;
	// 					auto qNews = Symmetry::reduceSilent(Lold.index[nu][2],W.index[lambda][2]);
	// 					for(const auto &q : qNews)
	// 					{
	// 						if constexpr ( Symmetry::SPECIAL )
	// 							{
	// 								factor_cgc = Symmetry::coeff_buildL(A.index[mu][1],A.index[mu][2],A.index[mu][0],
	// 																	q,W.index[lambda][2],Lold.index[nu][2],
	// 																	B.index[kappa][1],B.index[kappa][2],B.index[kappa][0]);
	// 							}
	// 						else if constexpr ( Symmetry::HAS_CGC )
	// 							{
	// 								factor_cgc = 1.;
	// 							}
	// 						else
	// 						{
	// 							checkIndex = {A.index[mu][1],q,B.index[kappa][1]};
	// 							if (Symmetry::validate(checkIndex)) { factor_cgc = 1.; }
	// 							else { factor_cgc = 0.; }
	// 						}
	// 						if (std::abs(factor_cgc) < ::numeric_limits<Scalar>::epsilon()) { continue; }
	// 						new_dimsA = {A.block[mu].dimension(1),B.block[kappa].dimension(1),W.block[lambda].dimension(1)};
	// 						Atmp.resize(new_dimsA);
	// 						device.set(Cores);
	// 						Atmp.device(device.get()) = (((Lold.block[nu].contract(A.block[mu],product_dims1)).
	// 												   contract(B.block[kappa],product_dims2)).contract(W.block[lambda],product_dims3));
	// 						device.set(1);
	// 						totIndex = {A.index[mu][1],B.index[kappa][1],q};
	// 						auto it = Lnew.dict.find(totIndex,dummy_legs);
	// 						if ( it == Lnew.dict.end(dummy_legs) )
	// 						{
	// 							Lnew.push_back(totIndex, factor_cgc*Atmp);
	// 						}
	// 						else
	// 						{
	// 							Lnew.block[it->second] += factor_cgc * Atmp;
	// 						}
	// 					}
	// 				}
	// 			}
	// 	}
	// 	return;
	// }

	// template<typename Symmetry, typename Scalar>
	// void buildR( const TripodQ<Symmetry,Scalar> &Rold, const TripodQ<Symmetry,Scalar> &A,
	// 			 const TripodQ<Symmetry,Scalar> &B, const MultipedeQ<4,Symmetry,Scalar,-2> &W, TripodQ<Symmetry,Scalar> &Rnew )
	// {
	// 	Rnew.clear();
	// 	typedef typename Symmetry::qType qType;
	// 	Scalar factor_cgc;
	// 	std::array<Eigen::IndexPair<Index>, 1> product_dims1= {Eigen::IndexPair<Index>(0,1)};
	// 	std::array<Eigen::IndexPair<Index>, 1> product_dims2= {Eigen::IndexPair<Index>(0,1)};
	// 	std::array<Eigen::IndexPair<Index>, 3> product_dims3= {Eigen::IndexPair<Index>(0,1),Eigen::IndexPair<Index>(2,3),Eigen::IndexPair<Index>(4,2)};

	// 	std::array<Index,3> new_dimsA;
	// 	TensorType<3,Scalar> Atmp;
		
	// 	std::array<qType,1> searchA, searchB;
	// 	std::array<qType,2> searchW;
	// 	std::array<Index,1> legA, legB; legA[0] = 1; legB[0] = 1;
	// 	std::array<Index,2> legW = {0,1};
	// 	std::array<Index,3> dummy_legs;
	// 	std::iota(dummy_legs.begin(), dummy_legs.end(), Index(0));

	// 	std::array<qType,3> totIndex, checkIndex;
		
	// 	for (std::size_t nu=0; nu<Rold.size(); nu++)
	// 	{
	// 		searchA[0] = Rold.index[nu][0];
	// 		searchB[0] = Rold.index[nu][1];
	// 		auto rangeA = A.dict.equal_range(searchA,legA);
	// 		auto rangeB = B.dict.equal_range(searchB,legB);
	// 		for (auto itA = rangeA.first; itA != rangeA.second; ++itA)
	// 			for (auto itB = rangeB.first; itB != rangeB.second; ++itB)
	// 			{
	// 				std::size_t mu = itA->second;
	// 				std::size_t kappa = itB->second;
	// 				searchW[0] = B.index[kappa][2];
	// 				searchW[1] = A.index[mu][2];
	// 				auto rangeW = W.dict.equal_range(searchW,legW);
	// 				for (auto itW = rangeW.first; itW != rangeW.second; itW++)
	// 				{
	// 					std::size_t lambda = itW->second;
	// 					auto qNews = Symmetry::reduceSilent(Rold.index[nu][2],Symmetry::flip(W.index[lambda][2]));
	// 					for(const auto &q : qNews)
	// 					{
	// 						if constexpr ( Symmetry::SPECIAL )
	// 							{
	// 								factor_cgc = Symmetry::coeff_buildR(A.index[mu][1],A.index[mu][2],A.index[mu][0],
	// 																	Rold.index[nu][2],W.index[lambda][2],q,
	// 																	B.index[kappa][1],B.index[kappa][2],B.index[kappa][0]);
	// 							}
	// 						else if constexpr ( Symmetry::HAS_CGC )
	// 							{
	// 								factor_cgc = 1.;
	// 							}
	// 						else
	// 						{
	// 							checkIndex = {A.index[mu][0],q,B.index[kappa][0]};
	// 							if (Symmetry::validate(checkIndex)) { factor_cgc = 1.; }
	// 							else { factor_cgc = 0.; }
	// 						}
	// 						if (std::abs(factor_cgc) < ::numeric_limits<Scalar>::epsilon()) { continue; }
	// 						new_dimsA = {A.block[mu].dimension(0),B.block[kappa].dimension(0),W.block[lambda].dimension(0)};
	// 						Atmp.resize(new_dimsA);
	// 						device.set(Cores);
	// 						Atmp.device(device.get()) = (((Rold.block[nu].contract(A.block[mu],product_dims1)).
	// 												   contract(B.block[kappa],product_dims2)).contract(W.block[lambda],product_dims3));
	// 						device.set(1);
	// 						totIndex = {A.index[mu][0],B.index[kappa][0],q};
	// 						auto it = Rnew.dict.find(totIndex,dummy_legs);
	// 						if ( it == Rnew.dict.end(dummy_legs) )
	// 						{
	// 							Rnew.push_back(totIndex, factor_cgc*Atmp);
	// 						}
	// 						else
	// 						{
	// 							Rnew.block[it->second] += factor_cgc * Atmp;
	// 						}
	// 					}
	// 				}
	// 			}
	// 	}
	// 	return;
	// }

	// template<typename Symmetry, typename Scalar>
	// void HPsi( const TripodQ<Symmetry,Scalar> &L, const TripodQ<Symmetry,Scalar> &R,
	// 		   const TripodQ<Symmetry,Scalar> &Aold, const MultipedeQ<4,Symmetry,Scalar,-2> &W, TripodQ<Symmetry,Scalar> &Anew )
	// {
	// 	Anew.clear();
	// 	typedef typename Symmetry::qType qType;
	// 	Scalar factor_cgc;
	// 	std::array<Eigen::IndexPair<Index>, 1> product_dims1 = {Eigen::IndexPair<Index>(0,0)};
	// 	std::array<Eigen::IndexPair<Index>, 2> product_dims2 = {Eigen::IndexPair<Index>(1,0),Eigen::IndexPair<Index>(3,3)};
	// 	std::array<Eigen::IndexPair<Index>, 2> product_dims3 = {Eigen::IndexPair<Index>(1,0),Eigen::IndexPair<Index>(2,2)};
	// 	std::array<Index,3> shuffle_dims = {0,2,1};
	// 	std::array<Index,3> new_dimsA;
	// 	TensorType<3,Scalar> Atmp;
		
	// 	std::array<qType,1> searchA, searchW;
	// 	std::array<qType,2> searchR;
	// 	std::array<Index,1> legA, legW; legA[0] = 0; legW[0] = 1;
	// 	std::array<Index,2> legR = {0,2};
	// 	std::array<Index,3> dummy_legs;
	// 	std::iota(dummy_legs.begin(), dummy_legs.end(), Index(0));

	// 	std::array<qType,3> totIndex, checkIndex;
		
	// 	for (std::size_t nu=0; nu<L.size(); nu++)
	// 	{
	// 		searchA[0] = L.index[nu][0];
	// 		auto rangeA = Aold.dict.equal_range(searchA,legA);
	// 		for (auto itA = rangeA.first; itA != rangeA.second; ++itA)
	// 		{
	// 			std::size_t mu = itA->second;
	// 			searchW[0] = Aold.index[mu][2];
	// 			auto rangeW = W.dict.equal_range(searchW,legW);
	// 			for (auto itW = rangeW.first; itW != rangeW.second; ++itW)
	// 			{
	// 				std::size_t kappa = itW->second;
	// 				auto qNews = Symmetry::reduceSilent(L.index[nu][2],W.index[kappa][2]);
	// 				for(const auto &q : qNews)
	// 				{
	// 					searchR[0] = Aold.index[mu][1];
	// 					searchR[1] = q;
	// 					auto rangeR = R.dict.equal_range(searchR,legR);
	// 					for (auto itR = rangeR.first; itR != rangeR.second; itR++)
	// 					{
	// 						std::size_t lambda = itR->second;
						
	// 						if constexpr ( Symmetry::SPECIAL )
	// 							{
	// 								factor_cgc = Symmetry::coeff_HPsi(Aold.index[mu][1],Aold.index[mu][2],Aold.index[mu][0],
	// 																  R.index[lambda][2],W.index[kappa][2],L.index[nu][2],
	// 																  R.index[lambda][1],W.index[kappa][0],L.index[nu][1]);
	// 							}
	// 						else if constexpr ( Symmetry::HAS_CGC )
	// 							{
	// 								factor_cgc = 1.;
	// 							}
	// 						else
	// 						{
	// 							checkIndex = {L.index[nu][1],W.index[kappa][0],R.index[lambda][1]};
	// 							if (Symmetry::validate(checkIndex)) { factor_cgc = 1.; }
	// 							else { factor_cgc = 0.; }
	// 						}
	// 						if (std::abs(factor_cgc) < ::numeric_limits<Scalar>::epsilon()) { continue; }
	// 						new_dimsA = {L.block[nu].dimension(1),R.block[lambda].dimension(1),W.block[kappa].dimension(2)};
	// 						Atmp.resize(new_dimsA);
	// 						device.set(Cores);
	// 						Atmp.device(device.get()) = (((L.block[nu].contract(Aold.block[mu],product_dims1)).
	// 													  contract(W.block[kappa],product_dims2)).
	// 													 contract(R.block[lambda],product_dims3)).shuffle(shuffle_dims);
	// 						device.set(1);
	// 						totIndex = {L.index[nu][1],R.index[lambda][1],W.index[kappa][0]};
	// 						auto it = Anew.dict.find(totIndex,dummy_legs);
	// 						if ( it == Anew.dict.end(dummy_legs) )
	// 						{
	// 							Anew.push_back(totIndex, factor_cgc*Atmp);
	// 						}
	// 						else
	// 						{
	// 							Anew.block[it->second] += factor_cgc * Atmp;
	// 						}
	// 					}
	// 				}
	// 			}
	// 		}
	// 	}
	// 	return;
	// }

	// template<typename Symmetry, typename Scalar>
	// MultipedeQ<4,Symmetry,Scalar,0> merge2mpos( const MultipedeQ<4,Symmetry,Scalar,0> &W1, const MultipedeQ<4,Symmetry,Scalar,0> &W2 )
	// {
	// 	typedef typename Symmetry::qType qType;
	// 	assert(W1.size()>0 and W2.size()>0 and "One of the MultipedeQs has zero size.");
	// 	constexpr Index resRank=4;
	// 	std::array<dir,resRank> new_dirs;
	// 	// std::array<string,3> new_names;
	// 	new_dirs[0] = dir::in;
	// 	new_dirs[1] = dir::out;
	// 	new_dirs[2] = dir::in;
	// 	new_dirs[3] = dir::out;

	// 	MultipedeQ<resRank,Symmetry,Scalar,0> Mout(new_dirs); //,new_names);
	// 	for (std::size_t q=0; q<resRank; q++) {Mout.legs[q].place = q;}

	// 	Qbasis<Symmetry> basis1, basis2;
	// 	basis1.pullData(W1,2);
	// 	basis2.pullData(W2,2);
	// 	auto TensorBasis = basis1.combine(basis2);

	// 	//initalize variables needed during the loop.
	// 	std::array<Index,1> legs; legs[0] = 0;
	// 	// std::array<Index,resRank> new_dimsC; //New dimensions for the CGC-Tensor
	// 	std::array<Eigen::IndexPair<Index>, 1> product_dims= {Eigen::IndexPair<Index>(1,0)};
	// 	std::array<Index,resRank> new_dimsA; //New dimensions for the Block-Tensor
	// 	std::array<Index,6> shuffle_dims = {0,3,1,4,2,5}; //Shuffle dimensions
	// 	std::array<std::pair<Index,Index>,4> padding_dims; //Padding dims
	// 	Scalar factor_cgc; //factor which is the coupling coefficient
	// 	TensorType<resRank,Scalar> Atmp,A; //intermediate variable for storing the Block-Tensor
	// 	TensorType<resRank,Scalar> Atmp2; //intermediate variable for storing the Block-Tensor
	// 	TensorType<resRank,Scalar> Aold; //intermediate variable for storing the Block-Tensor
	// 	TensorType<resRank,Scalar> Aold2; //intermediate variable for storing the Block-Tensor

	// 	std::array<Index,resRank+::hidden_dim(resRank)> dummy_legs;
	// 	std::iota(dummy_legs.begin(), dummy_legs.end(), Index(0));
	
	// 	std::array<qType,1> conIndex; //search index after the first contraction
	// 	std::array<qType,resRank+::hidden_dim(resRank)> totIndex; //resulting index for the output MultipedeQ.
	// 	//working loop: Contract and reshape the tensors
	// 	for (std::size_t nu=0; nu<W1.size(); nu++)
	// 	{
	// 		conIndex[0] = W1.index[nu][1];
	// 		auto range = W2.dict.equal_range(conIndex,legs);	
	// 		for (auto its = range.first; its != range.second; ++its)
	// 		{
	// 			std::size_t mu = its->second;
	// 			auto reduce1 = Symmetry::reduceSilent(W1.index[nu][3],W2.index[mu][3]);
	// 			auto reduce2 = Symmetry::reduceSilent(W1.index[nu][2],W2.index[mu][2]);
	// 			auto reduce3 = Symmetry::reduceSilent(W1.index[nu][4],W2.index[mu][4]);
	// 			for (std::size_t kappa=0; kappa<reduce1.size(); kappa++)
	// 				for (std::size_t lambda=0; lambda<reduce2.size(); lambda++)
	// 					for (std::size_t jotta=0; jotta<reduce3.size(); jotta++)
	// 					{
	// 						if constexpr (Symmetry::SPECIAL)
	// 							{
	// 								factor_cgc = Symmetry::coeff_Wpair(W1.index[nu][0],W2.index[mu][1],W1.index[nu][1],
	// 																   W1.index[nu][3],W2.index[mu][3],reduce1[kappa],
	// 																   W1.index[nu][2],W2.index[mu][2],reduce2[lambda],
	// 																   W1.index[nu][4],W2.index[mu][4],reduce3[jotta]);
	// 							}
	// 						else if constexpr (Symmetry::HAS_CGC)
	// 							{
	// 								factor_cgc = 1.;
	// 							}
	// 						else
	// 						{
	// 							factor_cgc = 1.;
	// 						}
	// 						if ( std::abs(factor_cgc) < ::numeric_limits<Scalar>::epsilon() ) { continue; }
	// 						//set totIndex
	// 						totIndex[0] = W1.index[nu][0];
	// 						totIndex[1] = W2.index[mu][1];
	// 						totIndex[2] = reduce2[lambda];
	// 						totIndex[3] = reduce1[kappa];
	// 						totIndex[4] = reduce3[jotta];
	// 						//set new_dims
	// 						new_dimsA[0] = W1.block[nu].dimension(0);
	// 						new_dimsA[1] = W2.block[mu].dimension(1);
	// 						new_dimsA[2] = W1.block[nu].dimension(2)*W2.block[mu].dimension(2);
	// 						new_dimsA[3] = W1.block[nu].dimension(3)*W2.block[mu].dimension(3);
	// 						Atmp.resize(new_dimsA);
	// 						TensorType<6,Scalar> tmp1 = W1.block[nu].contract(W2.block[mu],product_dims);
	// 						TensorType<6,Scalar> tmp2 = tmp1.shuffle(shuffle_dims);
	// 						TensorType<4,Scalar> Atmp = tmp2.reshape(new_dimsA);

	// 						Index left1=TensorBasis.leftAmount(reduce1[kappa],{W1.index[nu][3], W2.index[mu][3]});
	// 						Index right1=TensorBasis.rightAmount(reduce1[kappa],{W1.index[nu][3], W2.index[mu][3]});
	// 						Index left2=TensorBasis.leftAmount(reduce2[lambda],{W1.index[nu][2], W2.index[mu][2]});
	// 						Index right2=TensorBasis.rightAmount(reduce2[lambda],{W1.index[nu][2], W2.index[mu][2]});
	// 						padding_dims = {std::make_pair(0,0),std::make_pair(0,0),
	// 										std::make_pair(left2,right2),std::make_pair(left1,right1)};
	// 						A = Atmp.pad(padding_dims);

	// 						auto it = Mout.dict.find(totIndex,dummy_legs);
	// 						if ( it == Mout.dict.end(dummy_legs) )
	// 						{
	// 							Mout.push_back(totIndex, factor_cgc*A);
	// 						}
	// 						else
	// 						{
	// 							Mout.block[it->second] += factor_cgc * A;
	// 						}
	// 					}
	// 		}
	// 	}
	// 	return Mout;
	// }

	// template<typename Symmetry, typename Scalar>
	// TripodQ<Symmetry,Scalar> calcPfromL( const TripodQ<Symmetry,Scalar> &L, const TripodQ<Symmetry,Scalar> &A, const MultipedeQ<4,Symmetry,Scalar,-2> &W )
	// {
	// 	typedef typename Symmetry::qType qType;
	// 	constexpr Index resRank = 3;
	// 	std::array<dir,resRank> new_dirs = { dir::in, dir::out, dir::in };
	// 	TripodQ<Symmetry,Scalar> Mout(new_dirs);
	// 	for (std::size_t q=0; q<resRank; q++) {Mout.legs[q].place = q;}

	// 	Qbasis<Symmetry> basis1, basis2;
	// 	basis1.pullData(A,1);
	// 	basis2.pullData(W,1);
	// 	auto TensorBasis = basis1.combine(basis2);
	// 	std::array<Eigen::IndexPair<Index>, 1> product_dims1= {Eigen::IndexPair<Index>(0,0)};
	// 	std::array<Eigen::IndexPair<Index>, 2> product_dims2= {Eigen::IndexPair<Index>(1,0),Eigen::IndexPair<Index>(3,3)};
	// 	std::array<Eigen::IndexPair<Index>, 2> product_dims3= {Eigen::IndexPair<Index>(1,1),Eigen::IndexPair<Index>(2,0)};
	// 	std::array<Index,resRank> new_dimsA; //New dimensions for the Block-Tensor
	// 	std::array<Index,4> shuffle_dims = {0,2,1,3}; //Shuffle dimensions
	// 	std::array<std::pair<Index,Index>,3> padding_dims; //Padding dims
	// 	Scalar factor_cgc; //factor which is the coupling coefficient
	// 	TensorType<resRank,Scalar> Atmp; //intermediate variable for storing the Block-Tensor
	// 	TensorType<resRank,Scalar> Atmp2; //intermediate variable for storing the Block-Tensor
	// 	TensorType<resRank,Scalar> Ctmp,Cnow; //intermediate variable for storing the CGC-Tensors

	// 	std::array<qType,resRank> totIndex;
	// 	std::array<qType,1> conIndex1;
	// 	std::array<Index,1> legs1; legs1[0] = 0;
	// 	std::array<qType,1> conIndex2;
	// 	std::array<Index,1> legs2; legs2[0] = 1; // legs2[1] = 3;
	// 	std::array<Index,3> dummy_legs;
	// 	std::iota(dummy_legs.begin(),dummy_legs.end(),0);
	
	// 	for (std::size_t nu=0; nu<L.size(); nu++)
	// 	{
	// 		conIndex1[0] = L.index[nu][0];
	// 		auto range = A.dict.equal_range(conIndex1,legs1);	
	// 		for (auto its = range.first; its != range.second; ++its)
	// 		{
	// 			std::size_t mu = its->second;
	// 			// conIndex2[0] = L.index[nu][2];
	// 			conIndex2[0] = A.index[mu][2];
	// 			auto range2 = W.dict.equal_range(conIndex2,legs2);	
	// 			for (auto its2 = range2.first; its2 != range2.second; ++its2)
	// 			{
	// 				std::size_t kappa = its2->second;
	// 				auto qvec = Symmetry::reduceSilent(L.index[nu][2],W.index[kappa][2]);
	// 				for ( const auto& q: qvec )
	// 				{
	// 					auto qvec2 = Symmetry::reduceSilent(A.index[mu][1],q);

	// 					for ( const auto& q2 : qvec2 )
	// 					{
	// 						if constexpr( Symmetry::SPECIAL )
	// 							{
	// 								factor_cgc = Symmetry::coeff_HPsi(A.index[mu][1],A.index[mu][2],A.index[mu][0], 
	// 																  q,W.index[kappa][2],L.index[nu][2],
	// 																  q2,W.index[kappa][0],L.index[nu][1]);
	// 							}
	// 						else if constexpr( Symmetry::HAS_CGC )
	// 							{
	// 								// Ctmp.resize(new_dimsC);
	// 								auto fuse = Symmetry::reduce(W.index[kappa][1],q,A.index[mu][1]);
	// 								// TensorType<4,Scalar> Cttmp = (L.cgc[nu].contract(A.cgc[mu],product_dims1)).contract(W.cgc[kappa],product_dims2);
	// 								// std::cout << "dim tmp: "; for(const auto& i:Cttmp.dimensions()) {std::cout << i << " ";} std::cout << std::endl;
	// 								// std::cout << "dim fuse: "; for(const auto& i:fuse.dimensions()) {std::cout << i << " ";} std::cout << std::endl;
	// 								Ctmp = (L.cgc[nu].contract(A.cgc[mu],product_dims1)).contract(W.cgc[kappa],product_dims2).contract(fuse,product_dims3);
	// 								factor_cgc = ::sumAbs(Ctmp);
	// 							}
	// 						else
	// 						{
	// 							std::array<qType,3> qCheck = {L.index[nu][1],W.index[kappa][0],q2};
	// 							if (Symmetry::validate(qCheck)) {factor_cgc=1.;}
	// 							else { factor_cgc=0.; }
	// 						}
	// 						// factor_cgc = Symmetry::coeff_buildL(A.index[mu][1],A.index[mu][2],A.index[mu][0], 
	// 						// 									W.index[kappa][1],W.index[kappa][4],W.index[kappa][0],
	// 						// 									q,W.index[kappa][2],L.index[nu][1]);
	// 						if ( std::abs(factor_cgc) < ::numeric_limits<Scalar>::epsilon() ) { continue; }
	// 						//set totIndex
	// 						totIndex[0] = L.index[nu][1];
	// 						totIndex[1] = W.index[kappa][0];
	// 						// totIndex[2] = q2;
	// 						totIndex[2] = A.index[mu][1];
	// 						new_dimsA[0] = L.block[nu].dimension(1);
	// 						new_dimsA[1] = W.block[kappa].dimension(2);
	// 						new_dimsA[2] = A.block[mu].dimension(1)*W.block[kappa].dimension(1);

	// 						Atmp.resize(new_dimsA);
	// 						TensorType<4,Scalar> tmp1 = (L.block[nu].contract(A.block[mu],product_dims1)).contract(W.block[kappa],product_dims2);
	// 						TensorType<4,Scalar> tmp2 = tmp1.shuffle(shuffle_dims);
	// 						Atmp = tmp2.reshape(new_dimsA);
	// 						// Index left=TensorBasis.leftAmount(q,{A.index[mu][1], W.index[kappa][1]});
	// 						// Index right=TensorBasis.rightAmount(q,{A.index[mu][1], W.index[kappa][1]});
	// 						// padding_dims = {std::make_pair(0,0),std::make_pair(0,0),
	// 						// 				std::make_pair(left,right)};
	// 						// Atmp2 = Atmp.pad(padding_dims);
	// 						auto it = Mout.dict.find(totIndex,dummy_legs);
	// 						if constexpr( Symmetry::HAS_CGC )
	// 							{
	// 								if ( it == Mout.dict.end(dummy_legs) )
	// 								{
	// 									Mout.push_back(totIndex, Atmp, Ctmp);
	// 								}
	// 								else
	// 								{
	// 									Cnow = Mout.cgc[it->second];
	// 									factor_cgc = prop_to(Ctmp, Cnow);
	// 									assert( factor_cgc != std::numeric_limits<Scalar>::infinity() and "Error in L" );
	// 									Atmp.device(device.get()) = Atmp * factor_cgc;
	// 									Mout.block[it->second].device(device.get()) += Atmp;
	// 								}
	// 							}
	// 						else
	// 						{
	// 							if ( it == Mout.dict.end(dummy_legs) )
	// 							{
	// 								Mout.push_back(totIndex, factor_cgc*Atmp);
	// 							}
	// 							else
	// 							{
	// 								// std::cout << "dims oldL: "; for (const auto& i:Mout.block[it->second].dimensions()) {std::cout << i << " ";} std::cout << std::endl;
	// 								// std::cout << "dims newL: "; for (const auto& i:Atmp.dimensions()) {std::cout << i << " ";} std::cout << std::endl;
	// 								Mout.block[it->second] += factor_cgc * Atmp;
	// 							}
	// 						}
	// 					}
	// 				}
	// 			}
	// 		}
	// 	}
	// 	return Mout.shuffle({0,2,1});
	// }

	// template<typename Symmetry, typename Scalar>
	// TripodQ<Symmetry,Scalar> calcPfromR( const TripodQ<Symmetry,Scalar> &R, const TripodQ<Symmetry,Scalar> &A, const MultipedeQ<4,Symmetry,Scalar,-2> &W )
	// {
	// 	typedef typename Symmetry::qType qType;
	// 	constexpr Index resRank = 3;
	// 	std::array<dir,resRank> new_dirs = { dir::out, dir::in, dir::in };
	// 	MultipedeQ<resRank,Symmetry,Scalar,0> Mout(new_dirs);
	// 	for (std::size_t q=0; q<resRank; q++) {Mout.legs[q].place = q;}
		
	// 	Qbasis<Symmetry> basis1, basis2;
	// 	basis1.pullData(A,0);
	// 	basis2.pullData(W,0);
	// 	auto TensorBasis = basis1.combine(basis2);

	// 	std::array<Eigen::IndexPair<Index>, 1> product_dims1= {Eigen::IndexPair<Index>(0,1)};
	// 	std::array<Eigen::IndexPair<Index>, 2> product_dims2= {Eigen::IndexPair<Index>(1,1),Eigen::IndexPair<Index>(3,3)};
	// 	std::array<Eigen::IndexPair<Index>, 2> product_dims3= {Eigen::IndexPair<Index>(1,1),Eigen::IndexPair<Index>(2,0)};
	// 	std::array<Index,resRank> new_dimsA; //New dimensions for the Block-Tensor
	// 	std::array<Index,4> shuffle_dims = {0,2,3,1}; //Shuffle dimensions
	// 	std::array<std::pair<Index,Index>,3> padding_dims; //Padding dims
	// 	Scalar factor_cgc; //factor which is the coupling coefficient
	// 	TensorType<resRank,Scalar> Atmp; //intermediate variable for storing the Block-Tensor
	// 	TensorType<resRank,Scalar> Atmp2; //intermediate variable for storing the Block-Tensor
	// 	TensorType<resRank,Scalar> Ctmp,Cnow; //intermediate variable for storing the CGC-Tensors
		
	// 	std::array<qType,resRank> totIndex;
	// 	std::array<qType,1> conIndex1;
	// 	std::array<Index,1> legs1; legs1[0] = 1;
	// 	std::array<qType,1> conIndex2;
	// 	std::array<Index,1> legs2; legs2[0] = 1; //legs2[1] = 3;
	// 	std::array<Index,3> dummy_legs;
	// 	std::iota(dummy_legs.begin(),dummy_legs.end(),0);
	
	// 	for (std::size_t nu=0; nu<R.size(); nu++)
	// 	{
	// 		conIndex1[0] = R.index[nu][0];
	// 		auto range = A.dict.equal_range(conIndex1,legs1);	
	// 		for (auto its = range.first; its != range.second; ++its)
	// 		{
	// 			std::size_t mu = its->second;
	// 			// conIndex2[0] = R.index[nu][2];
	// 			conIndex2[0] = A.index[mu][2];
	// 			auto range2 = W.dict.equal_range(conIndex2,legs2);	
	// 			for (auto its2 = range2.first; its2 != range2.second; ++its2)
	// 			{
	// 				std::size_t kappa = its2->second;
	// 				auto qvec = Symmetry::reduceSilent(R.index[nu][2],Symmetry::flip(W.index[kappa][2]));
	// 				for ( const auto& q: qvec )
	// 				{
	// 					auto qvec2 = Symmetry::reduceSilent(A.index[mu][0],q);
	// 					for ( const auto& q2: qvec2 )
	// 					{
	// 						if constexpr( Symmetry::SPECIAL )
	// 							{
	// 								factor_cgc = Symmetry::coeff_HPsi(A.index[mu][1],A.index[mu][2],A.index[mu][0], 
	// 																  R.index[nu][2],W.index[kappa][2],q,
	// 																  R.index[nu][1],W.index[kappa][0],q2);
	// 							}
	// 						else if constexpr( Symmetry::HAS_CGC )
	// 							{
	// 								// Ctmp.resize(new_dimsC);
	// 								auto fuse = Symmetry::reduce(W.index[kappa][0],q,A.index[mu][0]);
	// 								Ctmp = ((R.cgc[nu].contract(A.cgc[mu],product_dims1)).contract(W.cgc[kappa],product_dims2)).contract(fuse,product_dims3);
	// 								factor_cgc = ::sumAbs(Ctmp);
	// 							}
	// 						else
	// 						{
	// 							std::array<qType,3> qCheck = {q2,W.index[kappa][0],R.index[nu][1]};
	// 							if (Symmetry::validate(qCheck)) {factor_cgc=1.;}
	// 							else { factor_cgc=0.; }
	// 						}
	// 						// factor_cgc = Symmetry::coeff_test(A.index[mu][1],A.index[mu][2],A.index[mu][0], 
	// 						// 								  W.index[kappa][1],W.index[kappa][4],W.index[kappa][0],
	// 						// 								  R.index[nu][1],W.index[kappa][2],q);
	// 						if ( std::abs(factor_cgc) < ::numeric_limits<Scalar>::epsilon() ) { continue; }
	// 						//set totIndex
	// 						totIndex[0] = R.index[nu][1];
	// 						totIndex[1] = W.index[kappa][0];
	// 						// totIndex[2] = q2;
	// 						totIndex[2] = A.index[mu][0];
	// 						new_dimsA[0] = R.block[nu].dimension(1);
	// 						new_dimsA[1] = W.block[kappa].dimension(2);
	// 						new_dimsA[2] = A.block[mu].dimension(0)*W.block[kappa].dimension(0);
	// 						Atmp.resize(new_dimsA);
	// 						TensorType<4,Scalar> tmp1 = (R.block[nu].contract(A.block[mu],product_dims1)).contract(W.block[kappa],product_dims2);
	// 						TensorType<4,Scalar> tmp2 = tmp1.shuffle(shuffle_dims);
	// 						Atmp = tmp2.reshape(new_dimsA);
	// 						// Index left=TensorBasis.leftAmount(q,{A.index[mu][0], W.index[kappa][0]});
	// 						// Index right=TensorBasis.rightAmount(q,{A.index[mu][0], W.index[kappa][0]});
	// 						// padding_dims = {std::make_pair(0,0),std::make_pair(0,0),
	// 						// 				std::make_pair(left,right)};
	// 						// Atmp2 = Atmp.pad(padding_dims);
	// 						auto it = Mout.dict.find(totIndex,dummy_legs);
	// 						if constexpr( Symmetry::HAS_CGC )
	// 							{
	// 								if ( it == Mout.dict.end(dummy_legs) )
	// 								{
	// 									Mout.push_back(totIndex, Atmp2, Ctmp);
	// 								}
	// 								else
	// 								{
	// 									Cnow = Mout.cgc[it->second];
	// 									factor_cgc = prop_to(Ctmp, Cnow);
	// 									assert( factor_cgc != std::numeric_limits<Scalar>::infinity() and "Error in R" );
	// 									Atmp2.device(device.get()) = Atmp2 * factor_cgc;
	// 									Mout.block[it->second].device(device.get()) += Atmp2;
	// 								}
	// 							}
	// 						else
	// 						{
	// 							if ( it == Mout.dict.end(dummy_legs) )
	// 							{
	// 								Mout.push_back(totIndex, factor_cgc*Atmp);
	// 							}
	// 							else
	// 							{
	// 								// std::cout << "dims oldR: "; for (const auto& i:Mout.block[it->second].dimensions()) {std::cout << i << " ";} std::cout << std::endl;
	// 								// std::cout << "dims newR: "; for (const auto& i:Atmp.dimensions()) {std::cout << i << " ";} std::cout << std::endl;
	// 								Mout.block[it->second] += factor_cgc * Atmp;
	// 							}
	// 						}
	// 					}
	// 				}
	// 			}
	// 		}
	// 	}
	// 	return Mout.shuffle({2,0,1});
	// }

} //end namespace contractions
#endif
