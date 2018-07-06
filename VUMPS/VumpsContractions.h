#ifndef VANILLA_VUMPSCONTRACTIONS
#define VANILLA_VUMPSCONTRACTIONS

#include "boost/multi_array.hpp"

#include "tensors/DmrgContractions.h"
#include "VUMPS/Umps.h"
#include "VUMPS/VumpsTransferMatrixAA.h"
#include "Mpo.h"

template<typename Symmetry, typename Scalar>
Eigenstate<Biped<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > >
calc_LReigen (GAUGE::OPTION gauge, 
              const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Aket,
              const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Abra,
              const Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &C,
              const vector<qarray<Symmetry::Nq> > &qlocCell,
              size_t dimK = 100ul)
{
//	TransferMatrixAA<Symmetry,Scalar> T(gauge, Abra, Aket, qlocCell);
//	PivotVector<Symmetry,complex<double> > LRtmp(C.template cast<MatrixXcd>());
//	
//	ArnoldiSolver<TransferMatrixAA<Symmetry,double>,PivotVector<Symmetry,complex<double> > > Arnie;
//	Arnie.set_dimK(dimK);
//	
//	complex<double> lambda;
//	
//	Arnie.calc_dominant(T,LRtmp,lambda);
//	
//	Eigenstate<Biped<Symmetry,Matrix<complex<Scalar>,Dynamic,Dynamic> > > out;
//	out.energy = lambda.real();
//	out.state = LRtmp.data[0];
//	if (abs(lambda.imag()) > 1e-10)
//	{
//		lout << termcolor::red << "Non-zero imaginary part of dominant eigenvalue λ=" << lambda << ", |λ|=" << abs(lambda) << termcolor::reset << endl;
//	}
//	
//	return out;
}

///**Calculates the tensor \f$h_L\f$ (eq. 12) from the explicit 4-legged 2-site Hamiltonian and \f$A_L\f$.*/
template<typename Symmetry, typename MatrixType, typename MpoScalar>
Biped<Symmetry,MatrixType> make_hL (const boost::multi_array<MpoScalar,4> &H2site,
                                    const vector<Biped<Symmetry,MatrixType> > &AL,
                                    const vector<qarray<Symmetry::Nq> > &qloc)
{
	Biped<Symmetry,MatrixType> Mout;
	size_t D = qloc.size();
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	for (size_t q3=0; q3<AL[s3].dim; ++q3)
	{
		auto A1ins = Symmetry::reduceSilent(AL[s3].in[q3], Symmetry::flip(qloc[s1]));
		for (const auto &A1in : A1ins)
		{
			auto it1 = AL[s1].dict.find(qarray2<Symmetry::Nq>{A1in, AL[s3].in[q3]});
			if (it1 != AL[s1].dict.end())
			{
				auto A2outs = Symmetry::reduceSilent(AL[s1].in[it1->second], qloc[s2]);
				for (const auto &A2out : A2outs)
				{
					auto it2 = AL[s2].dict.find(qarray2<Symmetry::Nq>{AL[s1].in[it1->second], A2out});
					if (it2 != AL[s2].dict.end())
					{
						auto A4outs = Symmetry::reduceSilent(AL[s2].out[it2->second], qloc[s4]);
						for (const auto &A4out : A4outs)
						{
							auto it4 = AL[s4].dict.find(qarray2<Symmetry::Nq>{AL[s2].out[it2->second], A4out});
							if (it4 != AL[s4].dict.end())
							{
								MatrixType Mtmp;
								if (H2site[s1][s2][s3][s4] != 0.)
								{
									optimal_multiply(H2site[s1][s2][s3][s4],
									                 AL[s3].block[q3].adjoint(),
									                 AL[s1].block[it1->second].adjoint(),
									                 AL[s2].block[it2->second],
									                 AL[s4].block[it4->second],
									                 Mtmp);
								}
								
								if (Mtmp.size() != 0)
								{
									qarray2<Symmetry::Nq> quple = {AL[s3].out[q3], AL[s4].out[it4->second]};
									assert(quple[0] == quple[1]);
									auto it = Mout.dict.find(quple);
									
									if (it != Mout.dict.end())
									{
										if (Mout.block[it->second].rows() != Mtmp.rows() and
											Mout.block[it->second].cols() != Mtmp.cols())
										{
											Mout.block[it->second] = Mtmp;
										}
										else
										{
											Mout.block[it->second] += Mtmp;
										}
									}
									else
									{
										Mout.push_back(quple, Mtmp);
									}
								}
							}
						}
					}
				}
			}
		}
	}
	
	return Mout;
}

/**Calculates the tensor \f$h_R\f$ (eq. 12) from the explicit 4-legged 2-site Hamiltonian and \f$A_R\f$.*/
template<typename Symmetry, typename MatrixType, typename MpoScalar>
Biped<Symmetry,MatrixType> make_hR (const boost::multi_array<MpoScalar,4> &H2site,
                                    const vector<Biped<Symmetry,MatrixType> > &AR,
                                    const vector<qarray<Symmetry::Nq> > &qloc)
{
	Biped<Symmetry,MatrixType> Mout;
	size_t D = qloc.size();
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	for (size_t q2=0; q2<AR[s2].dim; ++q2)
	{
		auto A4outs = Symmetry::reduceSilent(AR[s2].out[q2], qloc[s4]);
		for (const auto &A4out : A4outs)
		{
			auto it4 = AR[s4].dict.find(qarray2<Symmetry::Nq>{AR[s2].out[q2], A4out});
			if (it4 != AR[s4].dict.end())
			{
				auto A3ins = Symmetry::reduceSilent(AR[s4].out[it4->second], Symmetry::flip(qloc[s3]));
				for (const auto &A3in : A3ins)
				{
					auto it3 = AR[s3].dict.find(qarray2<Symmetry::Nq>{A3in, AR[s4].out[it4->second]});
					if (it3 != AR[s3].dict.end())
					{
						auto A1ins = Symmetry::reduceSilent(AR[s3].in[it3->second], Symmetry::flip(qloc[s1]));
						for (const auto &A1in : A1ins)
						{
							auto it1 = AR[s1].dict.find(qarray2<Symmetry::Nq>{A1in, AR[s3].in[it3->second]});
							if (it1 != AR[s1].dict.end())
							{
								MatrixType Mtmp;
								if (H2site[s1][s2][s3][s4] != 0.)
								{
									optimal_multiply(H2site[s1][s2][s3][s4],
									                 AR[s2].block[q2],
									                 AR[s4].block[it4->second], 
									                 AR[s3].block[it3->second].adjoint(),
									                 AR[s1].block[it1->second].adjoint(),
									                 Mtmp);
								}
								
								if (Mtmp.size() != 0)
								{
									qarray2<Symmetry::Nq> quple = {AR[s2].in[q2], AR[s1].in[it1->second]};
									assert(quple[0] == quple[1]);
									auto it = Mout.dict.find(quple);
									
									if (it != Mout.dict.end())
									{
										if (Mout.block[it->second].rows() != Mtmp.rows() and
											Mout.block[it->second].cols() != Mtmp.cols())
										{
											Mout.block[it->second] = Mtmp;
										}
										else
										{
											Mout.block[it->second] += Mtmp;
										}
									}
									else
									{
										Mout.push_back(quple, Mtmp);
									}
								}
							}
						}
					}
				}
			}
		}
	}
	
	return Mout;
}

/**Calculates the tensor \f$Y_{Ra}\f$ (eq. C17) from the MPO tensor \p W, the left transfer matrix \p L and \f$A_L\f$.*/
template<typename Symmetry, typename MatrixType, typename MpoScalar>
Tripod<Symmetry,MatrixType> make_YL (size_t b,
                                     const Tripod<Symmetry,MatrixType> &Lold, 
                                     const vector<Biped<Symmetry,MatrixType> > &Abra, 
                                     const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W, 
                                     const vector<Biped<Symmetry,MatrixType> > &Aket, 
                                     const vector<qarray<Symmetry::Nq> > &qloc,
                                     const vector<qarray<Symmetry::Nq> > &qOp)
{
	Tripod<Symmetry,MatrixType> Lnew;
	contract_L(Lold, Abra, W, Aket, qloc, qOp, Lnew, false, make_pair(TRIANGULAR,b));
	return Lnew;
}

/**Calculates the tensor \f$Y_{Ra}\f$ (eq. C18) from the MPO tensor \p W, the left transfer matrix \p R and \f$A_R\f$.*/
template<typename Symmetry, typename MatrixType, typename MpoScalar>
Tripod<Symmetry,MatrixType> make_YR (size_t a,
                                     const Tripod<Symmetry,MatrixType> &Rold,
                                     const vector<Biped<Symmetry,MatrixType> > &Abra, 
                                     const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W, 
                                     const vector<Biped<Symmetry,MatrixType> > &Aket, 
                                     const vector<qarray<Symmetry::Nq> > &qloc,
                                     const vector<qarray<Symmetry::Nq> > &qOp)
{
	Tripod<Symmetry,MatrixType> Rnew;
	contract_R(Rold, Abra, W, Aket, qloc, qOp, Rnew, false, make_pair(TRIANGULAR,a));
	return Rnew;
}

///**Calculates the tensor \f$Y_{Ra}\f$ (eq. C17) for a 2-site unit cell.*/
//template<typename Symmetry, typename MatrixType, typename MpoScalar>
//MatrixType make_YL (size_t b,
//                    const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W12,
//                    const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W34,
//                    const boost::multi_array<MatrixType,LEGLIMIT> &L,
//                    const vector<Biped<Symmetry,MatrixType> > &AL1,
//                    const vector<Biped<Symmetry,MatrixType> > &AL2,
//                    const vector<qarray<Symmetry::Nq> > &qloc)
//{
//	size_t D  = qloc.size();
//	size_t M  = AL1[0].block[0].cols();
//	
//	MatrixType Mout;
//	Mout.resize(M,M);
//	Mout.setZero();
//	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	for (int k12=0; k12<W12[s1][s2][0].outerSize(); ++k12)
//	for (typename SparseMatrix<MpoScalar>::InnerIterator iW12(W12[s1][s2][0],k12); iW12; ++iW12)
//	for (size_t s3=0; s3<D; ++s3)
//	for (size_t s4=0; s4<D; ++s4)
//	for (int k34=0; k34<W34[s3][s4][0].outerSize(); ++k34)
//	for (typename SparseMatrix<MpoScalar>::InnerIterator iW34(W34[s3][s4][0],k34); iW34; ++iW34)
//	{
//		if (iW12.col()==iW34.row())
//		{
//			size_t a = iW12.row();
//			
//			if (a>b and b==iW34.col() and iW12.value()!=0. and iW34.value()!=0.)
//			{
//				Mout += iW12.value() * iW34.value() *
//				        AL2[s3].block[0].adjoint() *
//				        AL1[s1].block[0].adjoint() *
//				        L[a][0] *
//				        AL1[s2].block[0] *
//				        AL2[s4].block[0];
//			}
//		}
//	}
//	
//	return Mout;
//}

///**Calculates the tensor \f$Y_{Ra}\f$ (eq. C18) for a 2-site unit cell.*/
//template<typename Symmetry, typename MatrixType, typename MpoScalar>
//MatrixType make_YR (size_t a,
//                    const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W12,
//                    const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W34,
//                    const boost::multi_array<MatrixType,LEGLIMIT> &R,
//                    const vector<Biped<Symmetry,MatrixType> > &AR1,
//                    const vector<Biped<Symmetry,MatrixType> > &AR2,
//                    const vector<qarray<Symmetry::Nq> > &qloc)
//{
//	size_t D  = qloc.size();
//	size_t M  = AR1[0].block[0].cols();
//	
//	MatrixType Mout;
//	Mout.resize(M,M);
//	Mout.setZero();
//	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	for (int k12=0; k12<W12[s1][s2][0].outerSize(); ++k12)
//	for (typename SparseMatrix<MpoScalar>::InnerIterator iW12(W12[s1][s2][0],k12); iW12; ++iW12)
//	for (size_t s3=0; s3<D; ++s3)
//	for (size_t s4=0; s4<D; ++s4)
//	for (int k34=0; k34<W34[s3][s4][0].outerSize(); ++k34)
//	for (typename SparseMatrix<MpoScalar>::InnerIterator iW34(W34[s3][s4][0],k34); iW34; ++iW34)
//	{
//		if (iW12.col()==iW34.row())
//		{
//			size_t b = iW34.col();
//			
//			if (a>b and a==iW12.row() and iW12.value()!=0. and iW34.value()!=0.)
//			{
//				Mout += iW12.value() * iW34.value() *
//				        AR1[s2].block[0] *
//				        AR2[s4].block[0] *
//				        R[b][0] *
//				        AR2[s3].block[0].adjoint() *
//				        AR1[s1].block[0].adjoint();
//			}
//		}
//	}
//	
//	return Mout;
//}

///**Calculates the tensor \f$Y_{La}\f$ (eq. C17) for a 2-site unit cell in a more effective fashion where \p W is a pre-contracted 2-site 4-legged MPO and \p Apair is a pre-contracted pair of A-tensors. The first index of \p W is the row index.*/
//template<typename Symmetry, typename MatrixType, typename Scalar, typename MpoScalar>
//MatrixType make_YL (const vector<tuple<size_t,size_t,size_t,size_t,size_t,MpoScalar> > &W,
//                    const boost::multi_array<MatrixType,LEGLIMIT> &L,
//                    const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Apair,
//                    const vector<qarray<Symmetry::Nq> > &qloc)
//{
//	size_t D  = qloc.size();
//	size_t M  = Apair[0].block[0].cols();
//	
//	MatrixType Mout;
//	Mout.resize(M,M);
//	Mout.setZero();
//	
//	auto index = [&D] (size_t s1, size_t s3) -> int {return s1*D+s3;};
//	
//	for (size_t i=0; i<W.size(); ++i)
//	{
//		size_t a = get<0>(W[i]);
//		size_t s1 = get<1>(W[i]);
//		size_t s2 = get<2>(W[i]);
//		size_t s3 = get<3>(W[i]);
//		size_t s4 = get<4>(W[i]);
//		
//		Mout += get<5>(W[i]) * Apair[index(s1,s3)].block[0].adjoint() * L[a][0] * Apair[index(s2,s4)].block[0];
//	}
//	
//	return Mout;
//}

///**Calculates the tensor \f$Y_{Ra}\f$ (eq. C18) for a 2-site unit cell in a more effective fashion where \p W is a pre-contracted 2-site 4-legged MPO and \p Apair is a pre-contracted pair of A-tensors. The first index of \p W is the column index.*/
//template<typename Symmetry, typename MatrixType, typename Scalar, typename MpoScalar>
//MatrixType make_YR (const vector<tuple<size_t,size_t,size_t,size_t,size_t,MpoScalar> > &W,
//                    const boost::multi_array<MatrixType,LEGLIMIT> &R,
//                    const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &Apair,
//                    const vector<qarray<Symmetry::Nq> > &qloc)
//{
//	size_t D  = qloc.size();
//	size_t M  = Apair[0].block[0].cols();
//	
//	MatrixType Mout;
//	Mout.resize(M,M);
//	Mout.setZero();
//	
//	auto index = [&D] (size_t s1, size_t s3) -> size_t {return s1*D+s3;};
//	
//	for (size_t i=0; i<W.size(); ++i)
//	{
//		size_t b = get<0>(W[i]);
//		size_t s1 = get<1>(W[i]);
//		size_t s2 = get<2>(W[i]);
//		size_t s3 = get<3>(W[i]);
//		size_t s4 = get<4>(W[i]);
//		
//		Mout += get<5>(W[i]) * Apair[index(s2,s4)].block[0] * R[b][0] * Apair[index(s1,s3)].block[0].adjoint();
//	}
//	
//	return Mout;
//}

///**Calculates the tensor \f$Y_{La}\f$ (eq. C17) for a 4-site unit cell in a more effective fashion where \p W is a pre-contracted 4-site 8-legged MPO and \p Aquadruple is a pre-contracted quadruple of A-tensors. The first index of \p W is the row index.*/
//template<typename Symmetry, typename MatrixType, typename Scalar, typename MpoScalar>
//MatrixType make_YL (const vector<tuple<size_t,size_t,size_t,size_t,size_t,size_t,size_t,size_t,size_t,MpoScalar> > &W,
//                    const boost::multi_array<MatrixType,LEGLIMIT> &L,
//                    const boost::multi_array<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> >,4> &Aquadruple, 
//                    const vector<qarray<Symmetry::Nq> > &qloc)
//{
//	size_t D  = qloc.size();
//	size_t M = Aquadruple[0][0][0][0].block[0].cols();
//	
//	MatrixType Mout;
//	Mout.resize(M,M);
//	Mout.setZero();
//	
//	for (size_t i=0; i<W.size(); ++i)
//	{
//		size_t a = get<0>(W[i]);
//		size_t s1 = get<1>(W[i]);
//		size_t s2 = get<2>(W[i]);
//		size_t s3 = get<3>(W[i]);
//		size_t s4 = get<4>(W[i]);
//		size_t s5 = get<5>(W[i]);
//		size_t s6 = get<6>(W[i]);
//		size_t s7 = get<7>(W[i]);
//		size_t s8 = get<8>(W[i]);
//		
//		Mout += get<9>(W[i]) * Aquadruple[s1][s3][s5][s7].block[0].adjoint() * L[a][0] * Aquadruple[s2][s4][s6][s8].block[0];
//	}
//	
//	return Mout;
//}

///**Calculates the tensor \f$Y_{Ra}\f$ (eq. C18) for a 4-site unit cell in a more effective fashion where \p W is a pre-contracted 4-site 8-legged MPO and \p Aquadruple is a pre-contracted quadruple of A-tensors. The first index of \p W is the column index.*/
//template<typename Symmetry, typename MatrixType, typename Scalar, typename MpoScalar>
//MatrixType make_YR (const vector<tuple<size_t,size_t,size_t,size_t,size_t,size_t,size_t,size_t,size_t,MpoScalar> > &W,
//                    const boost::multi_array<MatrixType,LEGLIMIT> &R,
//                    const boost::multi_array<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> >,4> &Aquadruple, 
//                    const vector<qarray<Symmetry::Nq> > &qloc)
//{
//	size_t D  = qloc.size();
//	size_t M = Aquadruple[0][0][0][0].block[0].cols();
//	
//	MatrixType Mout;
//	Mout.resize(M,M);
//	Mout.setZero();
//	
//	for (size_t i=0; i<W.size(); ++i)
//	{
//		size_t b = get<0>(W[i]);
//		size_t s1 = get<1>(W[i]);
//		size_t s2 = get<2>(W[i]);
//		size_t s3 = get<3>(W[i]);
//		size_t s4 = get<4>(W[i]);
//		size_t s5 = get<5>(W[i]);
//		size_t s6 = get<6>(W[i]);
//		size_t s7 = get<7>(W[i]);
//		size_t s8 = get<8>(W[i]);
//		
//		Mout += get<9>(W[i]) * Aquadruple[s2][s4][s6][s8].block[0] * R[b][0] * Aquadruple[s1][s3][s5][s7].block[0].adjoint();
//	}
//	
//	return Mout;
//}

///**Calculates the tensor \f$h_L\f$ (eq. 12) for a 2-site unit cell.*/
//template<typename Symmetry, typename MatrixType, typename MpoScalar>
//MatrixType make_hL (const boost::multi_array<MpoScalar,4> &H2site,
//                    const vector<Biped<Symmetry,MatrixType> > &AL1,
//                    const vector<Biped<Symmetry,MatrixType> > &AL2,
//                    const vector<qarray<Symmetry::Nq> > &qloc)
//{
//	MatrixType Mout;
//	Mout.resize(AL1[0].block[0].cols(), AL1[0].block[0].cols());
//	Mout.setZero();
//	size_t D = qloc.size();
//	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	for (size_t s3=0; s3<D; ++s3)
//	for (size_t s4=0; s4<D; ++s4)
//	{
//		if (H2site[s1][s2][s3][s4] != 0.)
//		{
//			Mout += H2site[s1][s2][s3][s4] * AL2[s3].block[0].adjoint()
//			                               * AL1[s1].block[0].adjoint()
//			                               * AL1[s2].block[0]
//			                               * AL2[s4].block[0];
//		}
//	}
//	
//	return Mout;
//}

///**Calculates the tensor \f$h_R\f$ (eq. 12) for a 2-site unit cell.*/
//template<typename Symmetry, typename MatrixType, typename MpoScalar>
//MatrixType make_hR (const boost::multi_array<MpoScalar,4> &H2site,
//                    const vector<Biped<Symmetry,MatrixType> > &AR1,
//                    const vector<Biped<Symmetry,MatrixType> > &AR2,
//                    const vector<qarray<Symmetry::Nq> > &qloc)
//{
//	MatrixType Mout;
//	Mout.resize(AR1[0].block[0].rows(), AR1[0].block[0].rows());
//	Mout.setZero();
//	size_t D = qloc.size();
//	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	for (size_t s3=0; s3<D; ++s3)
//	for (size_t s4=0; s4<D; ++s4)
//	{
//		if (H2site[s1][s2][s3][s4] != 0.)
//		{
//			Mout += H2site[s1][s2][s3][s4] * AR1[s2].block[0]
//			                               * AR2[s4].block[0]
//			                               * AR2[s3].block[0].adjoint()
//			                               * AR1[s1].block[0].adjoint();
//		}
//	}
//	
//	return Mout;
//}

/**Contracts two MPO tensors (H of length 2) to a 4-legged tensor.*/
template<typename MpHamiltonian, typename Scalar>
boost::multi_array<Scalar,4> make_Warray4 (int b, const MpHamiltonian &H)
{
	size_t D12 = H.locBasis(0).size();
	size_t D34 = H.locBasis(1).size();
	boost::multi_array<Scalar,4> Wout(boost::extents[D12][D12][D34][D34]);
	
	for (size_t s1=0; s1<D12; ++s1)
	for (size_t s2=0; s2<D12; ++s2)
	for (size_t s3=0; s3<D34; ++s3)
	for (size_t s4=0; s4<D34; ++s4)
	for (int k12=0; k12<H.W_at(0)[s1][s2][0].outerSize(); ++k12)
	for (typename SparseMatrix<Scalar>::InnerIterator iW12(H.W_at(0)[s1][s2][0],k12); iW12; ++iW12)
	for (int k34=0; k34<H.W_at(1)[s3][s4][0].outerSize(); ++k34)
	for (typename SparseMatrix<Scalar>::InnerIterator iW34(H.W_at(1)[s3][s4][0],k34); iW34; ++iW34)
	{
		if (iW12.row() == b and iW34.col() == b and 
		    iW12.col() == iW34.row() and
		    H.locBasis(0)[s1]+H.locBasis(1)[s3] == H.locBasis(0)[s2]+H.locBasis(1)[s4])
		{
			Wout[s1][s2][s3][s4] = iW12.value() * iW34.value();
		}
	}
	
	return Wout;
}

// template<typename Symmetry, typename MpoScalar>
// void contract_WW (const vector<unordered_map<tuple<size_t,size_t,size_t,qarray<Symmetry::Nq>,qarray<Symmetry::Nq> >,SparseMatrix<MpoScalar> > > V12,
//                   const vector<qarray<Symmetry::Nq> > &qloc12, 
//                   const vector<qarray<Symmetry::Nq> > &qOp12,
//                   const vector<unordered_map<tuple<size_t,size_t,size_t,qarray<Symmetry::Nq>,qarray<Symmetry::Nq> >,SparseMatrix<MpoScalar> > > V34,
//                   const vector<qarray<Symmetry::Nq> > &qloc34, 
//                   const vector<qarray<Symmetry::Nq> > &qOp34,
// 				  vector<unordered_map<tuple<size_t,size_t,size_t,qarray<Symmetry::Nq>,qarray<Symmetry::Nq> >,SparseMatrix<MpoScalar> > > V,
// 				  vector<qarray<Symmetry::Nq> > &qloc, 
// 				  vector<qarray<Symmetry::Nq> > &qOp)
// {
// 	V.clear();
// 	qloc.clear();
// 	qOp.clear();
	
// 	qOp = Symmetry::reduceSilent(qOp12, qOp34, true);
// 	auto tensor_basis = Symmetry::tensorProd(qloc12, qloc34);
	
// 	qloc.resize(tensor_basis.size());
// 	for (size_t q=0; q<tensor_basis.size(); ++q)
// 	{
// 		qloc[q] = get<4>(tensor_basis[q]);
// 	}
	
// 	W.resize(tensor_basis.size());
	
// 	for (size_t s1=0; s1<qloc12.size(); ++s1)
// 	for (size_t s3=0; s3<qloc34.size(); ++s3)
// 	{
// 		auto qmerges13 = Symmetry::reduceSilent(qloc12[s1], qloc34[s3]);
		
// 		for (const auto &qmerge13:qmerges13)
// 		{
// 			auto qtensor13 = make_tuple(qloc12[s1], s1, qloc34[s3], s3, qmerge13);
// 			auto s1s3 = distance(tensor_basis.begin(), find(tensor_basis.begin(), tensor_basis.end(), qtensor13));
			
// 			for (size_t s2=0; s2<qloc12.size(); ++s2)
// 			for (size_t s4=0; s4<qloc34.size(); ++s4)
// 			{
// 				auto qmerges24 = Symmetry::reduceSilent(qloc12[s2], qloc34[s4]);
				
// 				for (const auto &qmerge24:qmerges24)
// 				{
// 					auto qtensor24 = make_tuple(qloc12[s2], s2, qloc34[s4], s4, qmerge24);
// 					auto s2s4 = distance(tensor_basis.begin(), find(tensor_basis.begin(), tensor_basis.end(), qtensor24));
					
// 					for (size_t k12=0; k12<qOp12.size(); ++k12)
// 					for (size_t k34=0; k34<qOp34.size(); ++k34)
// 					{
// 						auto kmerges = Symmetry::reduceSilent(qOp12[k12], qOp34[k34]);
						
// 						for (const auto &kmerge:kmerges)
// 						{
// 							if (!Symmetry::validate(qarray3<Symmetry::Nq>{qmerge24,kmerge,qmerge13})) {continue;}
							
// 							auto k = distance(qOp.begin(), find(qOp.begin(), qOp.end(), kmerge));
							
// 							auto key12 = make_tuple(s1,s2,k12,Lold.mid(qL),quple[2]);
// 							if(auto it=V12.find(key); it == V.end()) { continue; }
// 							for (int r12=0; r12<V12.at(key).outerSize(); ++r12)

// 							for (int r12=0; r12<W12[s1][s2][k12].outerSize(); ++r12)
// 							for (typename SparseMatrix<MpoScalar>::InnerIterator iW12(W12[s1][s2][k12],r12); iW12; ++iW12)
// 							for (int r34=0; r34<W34[s3][s4][k34].outerSize(); ++r34)
// 							for (typename SparseMatrix<MpoScalar>::InnerIterator iW34(W34[s3][s4][k34],r34); iW34; ++iW34)
// 							{
// 								MpoScalar val = iW12.value() * iW34.value();
								
// 								if (iW12.col() == iW34.row() and abs(val) > 0.)
// 								{
// 									if (W[s1s3][s2s4][k].size() == 0)
// 									{
// 										W[s1s3][s2s4][k].resize(W12[s1][s2][k12].rows(), W34[s3][s4][k34].cols());
// 									}
									
// 									W[s1s3][s2s4][k].coeffRef(iW12.row(),iW34.col()) += val;
// 								}
// 							}
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}
// }


template<typename Symmetry, typename MpoScalar>
void contract_WW (const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W12, 
                  const vector<qarray<Symmetry::Nq> > &qloc12, 
                  const vector<qarray<Symmetry::Nq> > &qOp12,
                  const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W34, 
                  const vector<qarray<Symmetry::Nq> > &qloc34, 
                  const vector<qarray<Symmetry::Nq> > &qOp34,
                        vector<vector<vector<SparseMatrix<MpoScalar> > > > &W, 
                        vector<qarray<Symmetry::Nq> > &qloc, 
                        vector<qarray<Symmetry::Nq> > &qOp)
{
	W.clear();
	qloc.clear();
	qOp.clear();
	
	qOp = Symmetry::reduceSilent(qOp12, qOp34, true);
	auto tensor_basis = Symmetry::tensorProd(qloc12, qloc34);
	
	qloc.resize(tensor_basis.size());
	for (size_t q=0; q<tensor_basis.size(); ++q)
	{
		qloc[q] = get<4>(tensor_basis[q]);
	}
	
	W.resize(tensor_basis.size());
	for (size_t s1s3=0; s1s3<tensor_basis.size(); ++s1s3)
	{
		W[s1s3].resize(tensor_basis.size());
		for (size_t s2s4=0; s2s4<tensor_basis.size(); ++s2s4)
		{
			W[s1s3][s2s4].resize(qOp.size());
		}
	}
	
	for (size_t s1=0; s1<qloc12.size(); ++s1)
	for (size_t s3=0; s3<qloc34.size(); ++s3)
	{
		auto qmerges13 = Symmetry::reduceSilent(qloc12[s1], qloc34[s3]);
		
		for (const auto &qmerge13:qmerges13)
		{
			auto qtensor13 = make_tuple(qloc12[s1], s1, qloc34[s3], s3, qmerge13);
			auto s1s3 = distance(tensor_basis.begin(), find(tensor_basis.begin(), tensor_basis.end(), qtensor13));
			
			for (size_t s2=0; s2<qloc12.size(); ++s2)
			for (size_t s4=0; s4<qloc34.size(); ++s4)
			{
				auto qmerges24 = Symmetry::reduceSilent(qloc12[s2], qloc34[s4]);
				
				for (const auto &qmerge24:qmerges24)
				{
					auto qtensor24 = make_tuple(qloc12[s2], s2, qloc34[s4], s4, qmerge24);
					auto s2s4 = distance(tensor_basis.begin(), find(tensor_basis.begin(), tensor_basis.end(), qtensor24));
					
					for (size_t k12=0; k12<qOp12.size(); ++k12)
					for (size_t k34=0; k34<qOp34.size(); ++k34)
					{
						auto kmerges = Symmetry::reduceSilent(qOp12[k12], qOp34[k34]);
						
						for (const auto &kmerge:kmerges)
						{
							if (!Symmetry::validate(qarray3<Symmetry::Nq>{qmerge24,kmerge,qmerge13})) {continue;}
							
							auto k = distance(qOp.begin(), find(qOp.begin(), qOp.end(), kmerge));
							
							for (int r12=0; r12<W12[s1][s2][k12].outerSize(); ++r12)
							for (typename SparseMatrix<MpoScalar>::InnerIterator iW12(W12[s1][s2][k12],r12); iW12; ++iW12)
							for (int r34=0; r34<W34[s3][s4][k34].outerSize(); ++r34)
							for (typename SparseMatrix<MpoScalar>::InnerIterator iW34(W34[s3][s4][k34],r34); iW34; ++iW34)
							{
								MpoScalar val = iW12.value() * iW34.value();
								
								if (iW12.col() == iW34.row() and abs(val) > 0.)
								{
									if (W[s1s3][s2s4][k].size() == 0)
									{
										W[s1s3][s2s4][k].resize(W12[s1][s2][k12].rows(), W34[s3][s4][k34].cols());
									}
									
									W[s1s3][s2s4][k].coeffRef(iW12.row(),iW34.col()) += val;
								}
							}
						}
					}
				}
			}
		}
	}
}

///**Sums up all elements of a pre-contracted 4-legged MPO to check whether the transfer matrix becomes zero (see text below eq. C20).*/
//template<typename MpHamiltonian, typename Scalar>
//Scalar sum (const boost::multi_array<Scalar,4> &Warray)
//{
//	Scalar Wsum = 0;
//	
//	for (size_t s1=0; s1<Warray.shape()[0]; ++s1)
//	for (size_t s2=0; s2<Warray.shape()[1]; ++s2)
//	for (size_t s3=0; s3<Warray.shape()[2]; ++s3)
//	for (size_t s4=0; s4<Warray.shape()[3]; ++s4)
//	{
//		Wsum += Warray[s1][s2][s3][s4];
//	}
//	
//	return Wsum;
//}

///**Contracts four MPO tensors (H of length 4) to an 8-legged tensor.*/
//template<typename MpHamiltonian, typename Scalar>
//boost::multi_array<Scalar,8> make_Warray8 (int b, const MpHamiltonian &H)
//{
//	size_t D12 = H.locBasis(0).size();
//	size_t D34 = H.locBasis(1).size();
//	size_t D56 = H.locBasis(2).size();
//	size_t D78 = H.locBasis(3).size();
//	boost::multi_array<Scalar,8> Wout(boost::extents[D12][D12][D34][D34][D56][D56][D78][D78]);
//	
//	for (size_t s1=0; s1<D12; ++s1)
//	for (size_t s2=0; s2<D12; ++s2)
//	for (size_t s3=0; s3<D34; ++s3)
//	for (size_t s4=0; s4<D34; ++s4)
//	for (size_t s5=0; s5<D56; ++s5)
//	for (size_t s6=0; s6<D56; ++s6)
//	for (size_t s7=0; s7<D78; ++s7)
//	for (size_t s8=0; s8<D78; ++s8)
//	for (int k12=0; k12<H.W_at(0)[s1][s2][0].outerSize(); ++k12)
//	for (typename SparseMatrix<Scalar>::InnerIterator iW12(H.W_at(0)[s1][s2][0],k12); iW12; ++iW12)
//	for (int k34=0; k34<H.W_at(1)[s3][s4][0].outerSize(); ++k34)
//	for (typename SparseMatrix<Scalar>::InnerIterator iW34(H.W_at(1)[s3][s4][0],k34); iW34; ++iW34)
//	for (int k56=0; k56<H.W_at(2)[s5][s6][0].outerSize(); ++k56)
//	for (typename SparseMatrix<Scalar>::InnerIterator iW56(H.W_at(2)[s5][s6][0],k56); iW56; ++iW56)
//	for (int k78=0; k78<H.W_at(3)[s7][s8][0].outerSize(); ++k78)
//	for (typename SparseMatrix<Scalar>::InnerIterator iW78(H.W_at(3)[s7][s8][0],k78); iW78; ++iW78)
//	{
//		if (iW12.row() == b and iW78.col() == b and 
//		    iW12.col() == iW34.row() and
//		    iW34.col() == iW56.row() and
//		    iW56.col() == iW78.row() and
//		    H.locBasis(0)[s1]+H.locBasis(1)[s3]+H.locBasis(2)[s5]+H.locBasis(3)[s7] 
//		    == 
//		    H.locBasis(0)[s2]+H.locBasis(1)[s4]+H.locBasis(2)[s6]+H.locBasis(3)[s8])
//		{
//			Wout[s1][s2][s3][s4][s5][s6][s7][s8] = iW12.value() * iW34.value() * iW56.value() * iW78.value();
//		}
//	}
//	
//	return Wout;
//}

///**Sums up all elements of a pre-contracted 8-legged MPO to check whether the transfer matrix becomes zero (see text below eq. C20).*/
//template<typename MpHamiltonian, typename Scalar>
//Scalar sum (const boost::multi_array<Scalar,8> &Warray)
//{
//	Scalar Wsum = 0;
//	
//	for (size_t s1=0; s1<Warray.shape()[0]; ++s1)
//	for (size_t s2=0; s2<Warray.shape()[1]; ++s2)
//	for (size_t s3=0; s3<Warray.shape()[2]; ++s3)
//	for (size_t s4=0; s4<Warray.shape()[3]; ++s4)
//	for (size_t s5=0; s5<Warray.shape()[4]; ++s5)
//	for (size_t s6=0; s6<Warray.shape()[5]; ++s6)
//	for (size_t s7=0; s7<Warray.shape()[6]; ++s7)
//	for (size_t s8=0; s8<Warray.shape()[7]; ++s8)
//	{
//		Wsum += Warray[s1][s2][s3][s4][s5][s6][s7][s8];
//	}
//	
//	return Wsum;
//}

#endif
