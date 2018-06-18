#ifndef VANILLA_VUMPSCONTRACTIONS
#define VANILLA_VUMPSCONTRACTIONS

#include "boost/multi_array.hpp"

#include "tensors/DmrgContractions.h"
#include "VUMPS/Umps.h"
#include "Mpo.h"

///**Calculates the tensor \f$h_L\f$ (eq. 12) from the explicit 4-legged 2-site Hamiltonian and \f$A_L\f$.*/
//template<typename Symmetry, typename MatrixType, typename MpoScalar>
//MatrixType make_hL (const boost::multi_array<MpoScalar,4> &H2site,
//                    const vector<Biped<Symmetry,MatrixType> > &AL,
//                    const vector<qarray<Symmetry::Nq> > &qloc)
//{
//	MatrixType Mout;
//	Mout.resize(AL[0].block[0].cols(), AL[0].block[0].cols());
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
//			Mout += H2site[s1][s2][s3][s4] * AL[s3].block[0].adjoint()
//			                               * AL[s1].block[0].adjoint()
//			                               * AL[s2].block[0]
//			                               * AL[s4].block[0];
//		}
//	}
//	
//	return Mout;
//}

///**Calculates the tensor \f$h_R\f$ (eq. 12) from the explicit 4-legged 2-site Hamiltonian and \f$A_R\f$.*/
//template<typename Symmetry, typename MatrixType, typename MpoScalar>
//MatrixType make_hR (const boost::multi_array<MpoScalar,4> &H2site,
//                    const vector<Biped<Symmetry,MatrixType> > &AR,
//                    const vector<qarray<Symmetry::Nq> > &qloc)
//{
//	MatrixType Mout;
//	Mout.resize(AR[0].block[0].rows(), AR[0].block[0].rows());
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
//			Mout += H2site[s1][s2][s3][s4] * AR[s2].block[0]
//			                               * AR[s4].block[0]
//			                               * AR[s3].block[0].adjoint()
//			                               * AR[s1].block[0].adjoint();
//		}
//	}
//	
//	return Mout;
//}

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
//	size_t D  = qloc.size();
//	size_t dW = W.size();
//	size_t M  = AL[0].block[0].cols();
//	
//	MatrixType Mout;
//	Mout.resize(M,M);
//	Mout.setZero();
//	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	for (int k=0; k<W[s1][s2][0].outerSize(); ++k)
//	for (typename SparseMatrix<MpoScalar>::InnerIterator iW(W[s1][s2][0],k); iW; ++iW)
//	{
//		size_t a = iW.row();
//		
//		if (a>b and b==iW.col() and iW.value()!=0.)
//		{
//			Mout += iW.value() * AL[s1].block[0].adjoint() * L[a][0] * AL[s2].block[0];
//		}
//	}
//	
//	return Mout;
	
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
//	size_t D  = qloc.size();
//	size_t dW = W.size();
//	size_t M  = AR[0].block[0].cols();
//	
//	MatrixType Mout;
//	Mout.resize(M,M);
//	Mout.setZero();
//	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	for (int k=0; k<W[s1][s2][0].outerSize(); ++k)
//	for (typename SparseMatrix<MpoScalar>::InnerIterator iW(W[s1][s2][0],k); iW; ++iW)
//	{
//		size_t b = iW.col();
//		
//		if (a>b and a==iW.row() and iW.value()!=0.)
//		{
//			Mout += iW.value() * AR[s2].block[0] * R[b][0] * AR[s1].block[0].adjoint();
//		}
//	}
//	
//	return Mout;
	
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

/**Sums up all elements of a pre-contracted 4-legged MPO to check whether the transfer matrix becomes zero (see text below eq. C20).*/
template<typename MpHamiltonian, typename Scalar>
Scalar sum (const boost::multi_array<Scalar,4> &Warray)
{
	Scalar Wsum = 0;
	
	for (size_t s1=0; s1<Warray.shape()[0]; ++s1)
	for (size_t s2=0; s2<Warray.shape()[1]; ++s2)
	for (size_t s3=0; s3<Warray.shape()[2]; ++s3)
	for (size_t s4=0; s4<Warray.shape()[3]; ++s4)
	{
		Wsum += Warray[s1][s2][s3][s4];
	}
	
	return Wsum;
}

/**Contracts four MPO tensors (H of length 4) to an 8-legged tensor.*/
template<typename MpHamiltonian, typename Scalar>
boost::multi_array<Scalar,8> make_Warray8 (int b, const MpHamiltonian &H)
{
	size_t D12 = H.locBasis(0).size();
	size_t D34 = H.locBasis(1).size();
	size_t D56 = H.locBasis(2).size();
	size_t D78 = H.locBasis(3).size();
	boost::multi_array<Scalar,8> Wout(boost::extents[D12][D12][D34][D34][D56][D56][D78][D78]);
	
	for (size_t s1=0; s1<D12; ++s1)
	for (size_t s2=0; s2<D12; ++s2)
	for (size_t s3=0; s3<D34; ++s3)
	for (size_t s4=0; s4<D34; ++s4)
	for (size_t s5=0; s5<D56; ++s5)
	for (size_t s6=0; s6<D56; ++s6)
	for (size_t s7=0; s7<D78; ++s7)
	for (size_t s8=0; s8<D78; ++s8)
	for (int k12=0; k12<H.W_at(0)[s1][s2][0].outerSize(); ++k12)
	for (typename SparseMatrix<Scalar>::InnerIterator iW12(H.W_at(0)[s1][s2][0],k12); iW12; ++iW12)
	for (int k34=0; k34<H.W_at(1)[s3][s4][0].outerSize(); ++k34)
	for (typename SparseMatrix<Scalar>::InnerIterator iW34(H.W_at(1)[s3][s4][0],k34); iW34; ++iW34)
	for (int k56=0; k56<H.W_at(2)[s5][s6][0].outerSize(); ++k56)
	for (typename SparseMatrix<Scalar>::InnerIterator iW56(H.W_at(2)[s5][s6][0],k56); iW56; ++iW56)
	for (int k78=0; k78<H.W_at(3)[s7][s8][0].outerSize(); ++k78)
	for (typename SparseMatrix<Scalar>::InnerIterator iW78(H.W_at(3)[s7][s8][0],k78); iW78; ++iW78)
	{
		if (iW12.row() == b and iW78.col() == b and 
		    iW12.col() == iW34.row() and
		    iW34.col() == iW56.row() and
		    iW56.col() == iW78.row() and
		    H.locBasis(0)[s1]+H.locBasis(1)[s3]+H.locBasis(2)[s5]+H.locBasis(3)[s7] 
		    == 
		    H.locBasis(0)[s2]+H.locBasis(1)[s4]+H.locBasis(2)[s6]+H.locBasis(3)[s8])
		{
			Wout[s1][s2][s3][s4][s5][s6][s7][s8] = iW12.value() * iW34.value() * iW56.value() * iW78.value();
		}
	}
	
	return Wout;
}

/**Sums up all elements of a pre-contracted 8-legged MPO to check whether the transfer matrix becomes zero (see text below eq. C20).*/
template<typename MpHamiltonian, typename Scalar>
Scalar sum (const boost::multi_array<Scalar,8> &Warray)
{
	Scalar Wsum = 0;
	
	for (size_t s1=0; s1<Warray.shape()[0]; ++s1)
	for (size_t s2=0; s2<Warray.shape()[1]; ++s2)
	for (size_t s3=0; s3<Warray.shape()[2]; ++s3)
	for (size_t s4=0; s4<Warray.shape()[3]; ++s4)
	for (size_t s5=0; s5<Warray.shape()[4]; ++s5)
	for (size_t s6=0; s6<Warray.shape()[5]; ++s6)
	for (size_t s7=0; s7<Warray.shape()[6]; ++s7)
	for (size_t s8=0; s8<Warray.shape()[7]; ++s8)
	{
		Wsum += Warray[s1][s2][s3][s4][s5][s6][s7][s8];
	}
	
	return Wsum;
}

#endif
