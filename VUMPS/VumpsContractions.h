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

///**Calculates the tensor \f$h_L\f$ (eq. (12)) from the explicit 4-legged 2-site Hamiltonian and \f$A_L\f$.*/
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

/**Calculates the tensor \f$h_R\f$ (eq. (12)) from the explicit 4-legged 2-site Hamiltonian and \f$A_R\f$.*/
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

/**Calculates the tensor \f$Y_{Ra}\f$ (eq. (C17)) from the MPO tensor \p W, the left transfer matrix \p L and \f$A_L\f$.*/
template<typename Symmetry, typename MatrixType, typename MpoScalar>
Tripod<Symmetry,MatrixType> make_YL (size_t b,
                                     const Tripod<Symmetry,MatrixType> &Lold, 
                                     const vector<vector<Biped<Symmetry,MatrixType> > > &Abra, 
                                     const vector<vector<vector<vector<SparseMatrix<MpoScalar> > > > > &W, 
                                     const bool IS_HAMILTONIAN, 
                                     const vector<vector<Biped<Symmetry,MatrixType> > > &Aket, 
                                     const vector<vector<qarray<Symmetry::Nq> > > &qloc,
                                     const vector<vector<qarray<Symmetry::Nq> > > &qOp)
{
	size_t Lcell = Abra.size();
	Tripod<Symmetry,MatrixType> Lnext;
	Tripod<Symmetry,MatrixType> L = Lold;
	for (size_t l=0; l<Lcell; ++l)
	{
		if (l==Lcell-1)
		{
			contract_L(L, Abra[l], W[l], IS_HAMILTONIAN, Aket[l], qloc[l], qOp[l], Lnext, false, make_pair(FIXED,b));
		}
		else if (l==0)
		{
			contract_L(L, Abra[l], W[l], IS_HAMILTONIAN, Aket[l], qloc[l], qOp[l], Lnext, false, make_pair(TRIANGULAR,b));
		}
		else
		{
			contract_L(L, Abra[l], W[l], IS_HAMILTONIAN, Aket[l], qloc[l], qOp[l], Lnext, false, make_pair(FULL,0));
		}
		L.clear();
		L = Lnext;
		Lnext.clear();
	}
	return L;
}

/**Calculates the tensor \f$Y_{Ra}\f$ (eq. (C18)) from the MPO tensor \p W, the left transfer matrix \p R and \f$A_R\f$.*/
template<typename Symmetry, typename MatrixType, typename MpoScalar>
Tripod<Symmetry,MatrixType> make_YR (size_t a,
                                     const Tripod<Symmetry,MatrixType> &Rold,
                                     const vector<vector<Biped<Symmetry,MatrixType> > > &Abra, 
                                     const vector<vector<vector<vector<SparseMatrix<MpoScalar> > > > > &W, 
                                     const bool &IS_HAMILTONIAN, 
                                     const vector<vector<Biped<Symmetry,MatrixType> > > &Aket, 
                                     const vector<vector<qarray<Symmetry::Nq> > > &qloc,
                                     const vector<vector<qarray<Symmetry::Nq> > > &qOp)
{
	size_t Lcell = Abra.size();
	Tripod<Symmetry,MatrixType> Rnext;
	Tripod<Symmetry,MatrixType> R = Rold;
	for (int l=Lcell-1; l>=0; --l)
	{
		if (l==0)
		{
			contract_R(R, Abra[l], W[l], IS_HAMILTONIAN, Aket[l], qloc[l], qOp[l], Rnext, false, make_pair(FIXED,a));
		}
		else if (l==Lcell-1)
		{
			contract_R(R, Abra[l], W[l], IS_HAMILTONIAN, Aket[l], qloc[l], qOp[l], Rnext, false, make_pair(TRIANGULAR,a));
		}
		else
		{
			contract_R(R, Abra[l], W[l], IS_HAMILTONIAN, Aket[l], qloc[l], qOp[l], Rnext, false, make_pair(FULL,0));
		}
		R.clear();
		R = Rnext;
		Rnext.clear();
	}
	return R;
}

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

#endif
