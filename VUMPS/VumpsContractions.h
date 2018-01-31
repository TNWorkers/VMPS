#ifndef VANILLA_VUMPSCONTRACTIONS
#define VANILLA_VUMPSCONTRACTIONS

#include "boost/multi_array.hpp"

#include "tensors/DmrgContractions.h"
#include "VUMPS/Umps.h"
#include "Mpo.h"

/**Calculates the tensor \f$h_L\f$ (eq. 12) from the explicit 4-legged 2-site Hamiltonian and \f$A_L\f$.*/
template<typename Symmetry, typename MatrixType, typename MpoScalar>
MatrixType make_hL (const boost::multi_array<MpoScalar,4> &H2site,
                    const vector<Biped<Symmetry,MatrixType> > &AL,
                    const vector<qarray<Symmetry::Nq> > &qloc)
{
	MatrixType Mout;
	Mout.resize(AL[0].block[0].cols(), AL[0].block[0].cols());
	Mout.setZero();
	size_t D = qloc.size();
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	{
		if (H2site[s1][s2][s3][s4] != 0.)
		{
			Mout += H2site[s1][s2][s3][s4] * AL[s3].block[0].adjoint()
			                               * AL[s1].block[0].adjoint()
			                               * AL[s2].block[0]
			                               * AL[s4].block[0];
		}
	}
	
	return Mout;
}

/**Calculates the tensor \f$h_R\f$ (eq. 12) from the explicit 4-legged 2-site Hamiltonian and \f$A_R\f$.*/
template<typename Symmetry, typename MatrixType, typename MpoScalar>
MatrixType make_hR (const boost::multi_array<MpoScalar,4> &H2site,
                    const vector<Biped<Symmetry,MatrixType> > &AR,
                    const vector<qarray<Symmetry::Nq> > &qloc)
{
	MatrixType Mout;
	Mout.resize(AR[0].block[0].rows(), AR[0].block[0].rows());
	Mout.setZero();
	size_t D = qloc.size();
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	{
		if (H2site[s1][s2][s3][s4] != 0.)
		{
			Mout += H2site[s1][s2][s3][s4] * AR[s2].block[0]
			                               * AR[s4].block[0]
			                               * AR[s3].block[0].adjoint()
			                               * AR[s1].block[0].adjoint();
		}
	}
	
	return Mout;
}

/**Calculates the tensor \f$Y_L\f$ (eq. C17) from the MPO tensor \p W, the left transfer matrix \p L and \f$A_L\f$.*/
template<typename Symmetry, typename MatrixType, typename MpoScalar>
MatrixType make_YL (size_t b,
                    const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W,
                    const boost::multi_array<MatrixType,LEGLIMIT> &L,
                    const vector<Biped<Symmetry,MatrixType> > &AL,
                    const vector<qarray<Symmetry::Nq> > &qloc)
{
	size_t D  = qloc.size();
	size_t dW = W.size();
	size_t M  = AL[0].block[0].cols();
	
	MatrixType Mout;
	Mout.resize(M,M);
	Mout.setZero();
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (int k=0; k<W[s1][s2][0].outerSize(); ++k)
	for (typename SparseMatrix<MpoScalar>::InnerIterator iW(W[s1][s2][0],k); iW; ++iW)
	{
		size_t a = iW.row();
		
		if (a>b and b==iW.col() and iW.value()!=0.)
		{
			Mout += iW.value() * AL[s1].block[0].adjoint() * L[a][0] * AL[s2].block[0];
		}
	}
	
	return Mout;
}

/**Calculates the tensor \f$Y_R\f$ (eq. C18) from the MPO tensor \p W, the left transfer matrix \p R and \f$A_R\f$.*/
template<typename Symmetry, typename MatrixType, typename MpoScalar>
MatrixType make_YR (size_t a,
                    const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W,
                    const boost::multi_array<MatrixType,LEGLIMIT> &R,
                    const vector<Biped<Symmetry,MatrixType> > &AR,
                    const vector<qarray<Symmetry::Nq> > &qloc)
{
	size_t D  = qloc.size();
	size_t dW = W.size();
	size_t M  = AR[0].block[0].cols();
	
	MatrixType Mout;
	Mout.resize(M,M);
	Mout.setZero();
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (int k=0; k<W[s1][s2][0].outerSize(); ++k)
	for (typename SparseMatrix<MpoScalar>::InnerIterator iW(W[s1][s2][0],k); iW; ++iW)
	{
		size_t b = iW.col();
		
		if (a>b and a==iW.row() and iW.value()!=0.)
		{
			Mout += iW.value() * AR[s2].block[0] * R[b][0] * AR[s1].block[0].adjoint();
		}
	}
	
	return Mout;
}

/**Calculates the tensor \f$Y_L\f$ (eq. C17) explicitly for a 2-site unit cell.*/
template<typename Symmetry, typename MatrixType, typename MpoScalar>
MatrixType make_YL (size_t b,
                    const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W12,
                    const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W34,
                    const boost::multi_array<MatrixType,LEGLIMIT> &L,
                    const vector<Biped<Symmetry,MatrixType> > &AL1,
                    const vector<Biped<Symmetry,MatrixType> > &AL2,
                    const vector<qarray<Symmetry::Nq> > &qloc)
{
	size_t D  = qloc.size();
	size_t M  = AL1[0].block[0].cols();
	
	MatrixType Mout;
	Mout.resize(M,M);
	Mout.setZero();
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (int k12=0; k12<W12[s1][s2][0].outerSize(); ++k12)
	for (typename SparseMatrix<MpoScalar>::InnerIterator iW12(W12[s1][s2][0],k12); iW12; ++iW12)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	for (int k34=0; k34<W34[s3][s4][0].outerSize(); ++k34)
	for (typename SparseMatrix<MpoScalar>::InnerIterator iW34(W34[s3][s4][0],k34); iW34; ++iW34)
	{
		if (iW12.col()==iW34.row())
		{
			size_t a = iW12.row();
			
			if (a>b and b==iW34.col() and iW12.value()!=0. and iW34.value()!=0.)
			{
				Mout += iW12.value() * iW34.value() *
				        AL2[s3].block[0].adjoint() *
				        AL1[s1].block[0].adjoint() *
				        L[a][0] *
				        AL1[s2].block[0] *
				        AL2[s4].block[0];
			}
		}
	}
	
	return Mout;
}

/**Calculates the tensor \f$Y_R\f$ (eq. C18) explicitly for a 2-site unit cell.*/
template<typename Symmetry, typename MatrixType, typename MpoScalar>
MatrixType make_YR (size_t a,
                    const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W12,
                    const vector<vector<vector<SparseMatrix<MpoScalar> > > > &W34,
                    const boost::multi_array<MatrixType,LEGLIMIT> &R,
                    const vector<Biped<Symmetry,MatrixType> > &AR1,
                    const vector<Biped<Symmetry,MatrixType> > &AR2,
                    const vector<qarray<Symmetry::Nq> > &qloc)
{
	size_t D  = qloc.size();
	size_t M  = AR1[0].block[0].cols();
	
	MatrixType Mout;
	Mout.resize(M,M);
	Mout.setZero();
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (int k12=0; k12<W12[s1][s2][0].outerSize(); ++k12)
	for (typename SparseMatrix<MpoScalar>::InnerIterator iW12(W12[s1][s2][0],k12); iW12; ++iW12)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	for (int k34=0; k34<W34[s3][s4][0].outerSize(); ++k34)
	for (typename SparseMatrix<MpoScalar>::InnerIterator iW34(W34[s3][s4][0],k34); iW34; ++iW34)
	{
		if (iW12.col()==iW34.row())
		{
			size_t b = iW34.col();
			
			if (a>b and a==iW12.row() and iW12.value()!=0. and iW34.value()!=0.)
			{
				Mout += iW12.value() * iW34.value() *
				        AR1[s2].block[0] *
				        AR2[s4].block[0] *
				        R[b][0] *
				        AR2[s3].block[0].adjoint() *
				        AR1[s1].block[0].adjoint();
			}
		}
	}
	
	return Mout;
}

template<typename Symmetry, typename MatrixType, typename Scalar, typename MpoScalar>
MatrixType make_YL (const vector<tuple<size_t,size_t,size_t,size_t,size_t,MpoScalar> > &W,
                    const boost::multi_array<MatrixType,LEGLIMIT> &L,
                    const vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > &Apair,
                    const vector<qarray<Symmetry::Nq> > &qloc)
{
	size_t D  = qloc.size();
	size_t M  = Apair[0][0].block[0].cols();
	
	MatrixType Mout;
	Mout.resize(M,M);
	Mout.setZero();
	
	for (size_t i=0; i<W.size(); ++i)
	{
		size_t a = get<0>(W[i]);
		size_t s1 = get<1>(W[i]);
		size_t s2 = get<2>(W[i]);
		size_t s3 = get<3>(W[i]);
		size_t s4 = get<4>(W[i]);
		
		Mout += get<5>(W[i]) * Apair[s1][s3].block[0].adjoint() * L[a][0] * Apair[s2][s4].block[0];
	}
	
	return Mout;
}

template<typename Symmetry, typename MatrixType, typename Scalar, typename MpoScalar>
MatrixType make_YR (const vector<tuple<size_t,size_t,size_t,size_t,size_t,MpoScalar> > &W,
                    const boost::multi_array<MatrixType,LEGLIMIT> &R,
                    const vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > &Apair,
                    const vector<qarray<Symmetry::Nq> > &qloc)
{
	size_t D  = qloc.size();
	size_t M  = Apair[0][0].block[0].cols();
	
	MatrixType Mout;
	Mout.resize(M,M);
	Mout.setZero();
	
	for (size_t i=0; i<W.size(); ++i)
	{
		size_t b = get<0>(W[i]);
		size_t s1 = get<1>(W[i]);
		size_t s2 = get<2>(W[i]);
		size_t s3 = get<3>(W[i]);
		size_t s4 = get<4>(W[i]);
		
		Mout += get<5>(W[i]) * Apair[s2][s4].block[0] * R[b][0] * Apair[s1][s3].block[0].adjoint();
	}
	
	return Mout;
}

template<typename Symmetry, typename MatrixType, typename Scalar, typename MpoScalar>
MatrixType make_YL (const vector<tuple<size_t,size_t,size_t,size_t,size_t,size_t,size_t,size_t,size_t,MpoScalar> > &W,
                    const boost::multi_array<MatrixType,LEGLIMIT> &L,
                    const boost::multi_array<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> >,4> &Aquartett, 
                    const vector<qarray<Symmetry::Nq> > &qloc)
{
	size_t D  = qloc.size();
	size_t M = Aquartett[0][0][0][0].block[0].cols();
	
	MatrixType Mout;
	Mout.resize(M,M);
	Mout.setZero();
	
	for (size_t i=0; i<W.size(); ++i)
	{
		size_t a = get<0>(W[i]);
		size_t s1 = get<1>(W[i]);
		size_t s2 = get<2>(W[i]);
		size_t s3 = get<3>(W[i]);
		size_t s4 = get<4>(W[i]);
		size_t s5 = get<5>(W[i]);
		size_t s6 = get<6>(W[i]);
		size_t s7 = get<7>(W[i]);
		size_t s8 = get<8>(W[i]);
		
		Mout += get<9>(W[i]) * Aquartett[s1][s3][s5][s7].block[0].adjoint() * L[a][0] * Aquartett[s2][s4][s6][s8].block[0];
	}
	
	return Mout;
}

template<typename Symmetry, typename MatrixType, typename Scalar, typename MpoScalar>
MatrixType make_YR (const vector<tuple<size_t,size_t,size_t,size_t,size_t,size_t,size_t,size_t,size_t,MpoScalar> > &W,
                    const boost::multi_array<MatrixType,LEGLIMIT> &R,
                    const boost::multi_array<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> >,4> &Aquartett, 
                    const vector<qarray<Symmetry::Nq> > &qloc)
{
	size_t D  = qloc.size();
	size_t M = Aquartett[0][0][0][0].block[0].cols();
	
	MatrixType Mout;
	Mout.resize(M,M);
	Mout.setZero();
	
	for (size_t i=0; i<W.size(); ++i)
	{
		size_t b = get<0>(W[i]);
		size_t s1 = get<1>(W[i]);
		size_t s2 = get<2>(W[i]);
		size_t s3 = get<3>(W[i]);
		size_t s4 = get<4>(W[i]);
		size_t s5 = get<5>(W[i]);
		size_t s6 = get<6>(W[i]);
		size_t s7 = get<7>(W[i]);
		size_t s8 = get<8>(W[i]);
		
		Mout += get<9>(W[i]) * Aquartett[s2][s4][s6][s8].block[0] * R[b][0] * Aquartett[s1][s3][s5][s7].block[0].adjoint();
	}
	
	return Mout;
}

//template<typename Symmetry, typename MatrixType, typename Scalar, typename MpoScalar>
//MatrixType make_YL (size_t b,
//                    const vector<vector<SparseMatrix<MpoScalar> > > &W12,
//                    const vector<vector<SparseMatrix<MpoScalar> > > &W34,
//                    const vector<vector<SparseMatrix<MpoScalar> > > &W56,
//                    const vector<vector<SparseMatrix<MpoScalar> > > &W78,
//                    const boost::multi_array<MatrixType,LEGLIMIT> &L,
//                    const boost::multi_array<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> >,4> &Aquartett, 
//                    const vector<qarray<Symmetry::Nq> > &qloc)
//{
//	size_t D  = qloc.size();
////	size_t M  = AL1[0].block[0].cols();
//	size_t M = Aquartett[0][0][0][0].block[0].cols();
//	
//	MatrixType Mout;
//	Mout.resize(M,M);
//	Mout.setZero();
//	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	for (int k12=0; k12<W12[s1][s2].outerSize(); ++k12)
//	for (typename SparseMatrix<MpoScalar>::InnerIterator iW12(W12[s1][s2],k12); iW12; ++iW12)
//	for (size_t s3=0; s3<D; ++s3)
//	for (size_t s4=0; s4<D; ++s4)
//	for (int k34=0; k34<W34[s3][s4].outerSize(); ++k34)
//	for (typename SparseMatrix<MpoScalar>::InnerIterator iW34(W34[s3][s4],k34); iW34; ++iW34)
//	for (size_t s5=0; s5<D; ++s5)
//	for (size_t s6=0; s6<D; ++s6)
//	for (int k56=0; k56<W56[s5][s6].outerSize(); ++k56)
//	for (typename SparseMatrix<MpoScalar>::InnerIterator iW56(W56[s5][s6],k56); iW56; ++iW56)
//	for (size_t s7=0; s7<D; ++s7)
//	for (size_t s8=0; s8<D; ++s8)
//	for (int k78=0; k78<W78[s7][s8].outerSize(); ++k78)
//	for (typename SparseMatrix<MpoScalar>::InnerIterator iW78(W78[s7][s8],k78); iW78; ++iW78)
//	{
//		if (iW12.col()==iW34.row() and 
//		    iW34.col()==iW56.row() and 
//		    iW56.col()==iW78.row())
//		{
//			size_t a = iW12.row();
//			
//			if (a>b and b==iW78.col() and abs(iW12.value())>1e-15 and abs(iW34.value())>1e-15 
//			                          and abs(iW56.value())>1e-15 and abs(iW78.value())>1e-15)
//			{
////				Mout += iW12.value() * iW34.value() *
////				        iW56.value() * iW78.value() *
////				        AL4[s7].block[0].adjoint() *
////				        AL3[s5].block[0].adjoint() *
////				        AL2[s3].block[0].adjoint() *
////				        AL1[s1].block[0].adjoint() *
////				        L[a][0] *
////				        AL1[s2].block[0] *
////				        AL2[s4].block[0] *
////				        AL3[s6].block[0] *
////				        AL4[s8].block[0];
//				Mout += iW12.value() * iW34.value() *
//				        iW56.value() * iW78.value() *
//				        Aquartett[s1][s3][s5][s7].block[0].adjoint() * 
//				        L[a][0] *
//				        Aquartett[s2][s4][s6][s8].block[0];
//			}
//		}
//	}
//	
//	return Mout;
//}

//template<typename Symmetry, typename MatrixType, typename Scalar, typename MpoScalar>
//MatrixType make_YR (size_t a,
//                    const vector<vector<SparseMatrix<MpoScalar> > > &W12,
//                    const vector<vector<SparseMatrix<MpoScalar> > > &W34,
//                    const vector<vector<SparseMatrix<MpoScalar> > > &W56,
//                    const vector<vector<SparseMatrix<MpoScalar> > > &W78,
//                    const boost::multi_array<MatrixType,LEGLIMIT> &R,
//                    const boost::multi_array<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> >,4> &Aquartett, 
//                    const vector<qarray<Symmetry::Nq> > &qloc)
//{
//	size_t D  = qloc.size();
//	size_t M = Aquartett[0][0][0][0].block[0].cols();
//	
//	MatrixType Mout;
//	Mout.resize(M,M);
//	Mout.setZero();
//	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	for (int k12=0; k12<W12[s1][s2].outerSize(); ++k12)
//	for (typename SparseMatrix<MpoScalar>::InnerIterator iW12(W12[s1][s2],k12); iW12; ++iW12)
//	for (size_t s3=0; s3<D; ++s3)
//	for (size_t s4=0; s4<D; ++s4)
//	for (int k34=0; k34<W34[s3][s4].outerSize(); ++k34)
//	for (typename SparseMatrix<MpoScalar>::InnerIterator iW34(W34[s3][s4],k34); iW34; ++iW34)
//	for (size_t s5=0; s5<D; ++s5)
//	for (size_t s6=0; s6<D; ++s6)
//	for (int k56=0; k56<W56[s5][s6].outerSize(); ++k56)
//	for (typename SparseMatrix<MpoScalar>::InnerIterator iW56(W56[s5][s6],k56); iW56; ++iW56)
//	for (size_t s7=0; s7<D; ++s7)
//	for (size_t s8=0; s8<D; ++s8)
//	for (int k78=0; k78<W78[s7][s8].outerSize(); ++k78)
//	for (typename SparseMatrix<MpoScalar>::InnerIterator iW78(W78[s7][s8],k78); iW78; ++iW78)
//	{
//		if (iW12.col()==iW34.row() and
//		    iW34.col()==iW56.row() and 
//		    iW56.col()==iW78.row())
//		{
//			size_t b = iW78.col();
//			
//			if (a>b and a==iW12.row() and abs(iW12.value())>1e-15 and abs(iW34.value())>1e-15 
//			                          and abs(iW56.value())>1e-15 and abs(iW78.value())>1e-15)
//			{
////				Mout += iW12.value() * iW34.value() *
////				        iW56.value() * iW78.value() *
////				        AR1[s2].block[0] *
////				        AR2[s4].block[0] *
////				        AR3[s6].block[0] *
////				        AR4[s8].block[0] *
////				        R[b][0] *
////				        AR4[s7].block[0].adjoint() *
////				        AR3[s5].block[0].adjoint() *
////				        AR2[s3].block[0].adjoint() *
////				        AR1[s1].block[0].adjoint();
//				Mout += iW12.value() * iW34.value() *
//				        iW56.value() * iW78.value() *
//				        Aquartett[s2][s4][s6][s8].block[0] * 
//				        R[b][0] *
//				        Aquartett[s1][s3][s5][s7].block[0].adjoint();
//			}
//		}
//	}
//	
//	return Mout;
//}

//template<typename Symmetry, typename MatrixType, typename MpoScalar>
//double energy_L (const boost::multi_array<MpoScalar,4> &H2site, 
//                 const vector<Biped<Symmetry,MatrixType> > &AL, 
//                 const Biped<Symmetry,MatrixType> &C,
//                 const vector<qarray<Symmetry::Nq> > &qloc)
//{
//	size_t D = qloc.size();
//	double res = 0;
//	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	for (size_t s3=0; s3<D; ++s3)
//	for (size_t s4=0; s4<D; ++s4)
//	{
//		res += H2site[s1][s2][s3][s4] * (AL[s2].block[0] * 
//		                                 AL[s4].block[0] * 
//		                                 C.block[0] * 
//		                                 C.block[0].adjoint() * 
//		                                 AL[s3].block[0].adjoint() * 
//		                                 AL[s1].block[0].adjoint()
//		                                ).trace();
//	}
//	return res;
//}

//template<typename Symmetry, typename MatrixType, typename MpoScalar>
//double energy_R (const boost::multi_array<MpoScalar,4> &H2site, 
//                 const vector<Biped<Symmetry,MatrixType> > &AR, 
//                 const Biped<Symmetry,MatrixType> &C,
//                 const vector<qarray<Symmetry::Nq> > &qloc)
//{
//	size_t D = qloc.size();
//	double res = 0;
//	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	for (size_t s3=0; s3<D; ++s3)
//	for (size_t s4=0; s4<D; ++s4)
//	{
//		res += H2site[s1][s2][s3][s4] * (AR[s2].block[0] * 
//		                                 AR[s4].block[0] * 
//		                                 AR[s3].block[0].adjoint() * 
//		                                 AR[s1].block[0].adjoint() * 
//		                                 C.block[0] * 
//		                                 C.block[0].adjoint()
//		                                ).trace();
//	}
//	return res;
//}

/**Calculates the tensor \f$h_L\f$ (eq. 12) explicitly for a 2-site unit cell.*/
template<typename Symmetry, typename MatrixType, typename MpoScalar>
MatrixType make_hL (const boost::multi_array<MpoScalar,4> &H2site,
                    const vector<Biped<Symmetry,MatrixType> > &AL1,
                    const vector<Biped<Symmetry,MatrixType> > &AL2,
                    const vector<qarray<Symmetry::Nq> > &qloc)
{
	MatrixType Mout;
	Mout.resize(AL1[0].block[0].cols(), AL1[0].block[0].cols());
	Mout.setZero();
	size_t D = qloc.size();
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	{
		if (H2site[s1][s2][s3][s4] != 0.)
		{
			Mout += H2site[s1][s2][s3][s4] * AL2[s3].block[0].adjoint()
			                               * AL1[s1].block[0].adjoint()
			                               * AL1[s2].block[0]
			                               * AL2[s4].block[0];
		}
	}
	
	return Mout;
}

/**Calculates the tensor \f$h_R\f$ (eq. 12) explicitly for a 2-site unit cell.*/
template<typename Symmetry, typename MatrixType, typename MpoScalar>
MatrixType make_hR (const boost::multi_array<MpoScalar,4> &H2site,
                    const vector<Biped<Symmetry,MatrixType> > &AR1,
                    const vector<Biped<Symmetry,MatrixType> > &AR2,
                    const vector<qarray<Symmetry::Nq> > &qloc)
{
	MatrixType Mout;
	Mout.resize(AR1[0].block[0].rows(), AR1[0].block[0].rows());
	Mout.setZero();
	size_t D = qloc.size();
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	{
		if (H2site[s1][s2][s3][s4] != 0.)
		{
			Mout += H2site[s1][s2][s3][s4] * AR1[s2].block[0]
			                               * AR2[s4].block[0]
			                               * AR2[s3].block[0].adjoint()
			                               * AR1[s1].block[0].adjoint();
		}
	}
	
	return Mout;
}

//template<typename Symmetry, typename MatrixType>
//void shift_L (MatrixType &M,
//              const vector<Biped<Symmetry,MatrixType> > &AL,
//              const vector<qarray<Symmetry::Nq> > &qloc)
//{
//	size_t D = qloc.size();
//	MatrixType Mtmp(D,D); Mtmp.setZero();
//	
//	for (size_t s=0; s<D; ++s)
//	{
//		Mtmp += AL[s].block[0].adjoint() * M * AL[s].block[0];
//	}
//	
//	M = Mtmp;
//}

//template<typename Symmetry, typename MatrixType>
//void shift_R (MatrixType &M,
//              const vector<Biped<Symmetry,MatrixType> > &AR,
//              const vector<qarray<Symmetry::Nq> > &qloc)
//{
//	size_t D = qloc.size();
//	MatrixType Mtmp(D,D); Mtmp.setZero();
//	
//	for (size_t s=0; s<D; ++s)
//	{
//		Mtmp += AR[s].block[0] * M * AR[s].block[0].adjoint();
//	}
//	
//	M = Mtmp;
//}

//-----------<definitions>-----------
/**Structure to update \f$A_C\f$ (eq. 11). Contains \f$A_L\f$, \f$A_L\f$ and \f$H_L\f$ (= \p L), \f$H_R\f$ (= \p R).*/
template<typename Symmetry, typename Scalar, typename MpoScalar=double>
struct PivumpsMatrix
{
	Matrix<Scalar,Dynamic,Dynamic> L;
	Matrix<Scalar,Dynamic,Dynamic> R;
	
	std::array<boost::multi_array<MpoScalar,4>,2> h;
	
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > AL;
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > AR;
	
	vector<qarray<Symmetry::Nq> > qloc;
	
	size_t dim;
};

/**Wrapper containing \f$C\f$ for local upates (eq. 16).*/
template<typename Symmetry, typename Scalar>
struct PivumpsVector0
{
	Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > C;
	
	///\{
	/**Linear algebra for \f$C\f$ in the vector space.*/
	PivumpsVector0<Symmetry,Scalar>& operator+= (const PivumpsVector0<Symmetry,Scalar> &Vrhs);
	PivumpsVector0<Symmetry,Scalar>& operator-= (const PivumpsVector0<Symmetry,Scalar> &Vrhs);
	template<typename OtherScalar> PivumpsVector0<Symmetry,Scalar>& operator*= (const OtherScalar &alpha);
	template<typename OtherScalar> PivumpsVector0<Symmetry,Scalar>& operator/= (const OtherScalar &alpha);
	///\}
};
//-----------</definitions>-----------

template<typename Symmetry, typename Scalar, typename MpoScalar>
inline size_t dim (const PivumpsMatrix<Symmetry,Scalar,MpoScalar> &H)
{
	return H.dim;
}

template<typename Symmetry, typename Scalar>
PivumpsVector0<Symmetry,Scalar>& PivumpsVector0<Symmetry,Scalar>::operator+= (const PivumpsVector0<Symmetry,Scalar> &Vrhs)
{
	transform(C.block.begin(), C.block.end(), 
	          Vrhs.C.block.begin(), C.block.begin(), 
	          std::plus<Matrix<Scalar,Dynamic,Dynamic> >());
	return *this;
}

template<typename Symmetry, typename Scalar>
PivumpsVector0<Symmetry,Scalar>& PivumpsVector0<Symmetry,Scalar>::
operator-= (const PivumpsVector0<Symmetry,Scalar> &Vrhs)
{
	transform(C.block.begin(), C.block.end(), 
	          Vrhs.C.block.begin(), C.block.begin(), 
	          std::minus<Matrix<Scalar,Dynamic,Dynamic> >());
	return *this;
}

template<typename Symmetry, typename Scalar>
template<typename OtherScalar>
PivumpsVector0<Symmetry,Scalar>& PivumpsVector0<Symmetry,Scalar>::
operator*= (const OtherScalar &alpha)
{
	for (size_t q=0; q<C.dim; ++q)
	{
		C.block[q] *= alpha;
	}
	return *this;
}

template<typename Symmetry, typename Scalar>
template<typename OtherScalar>
PivumpsVector0<Symmetry,Scalar>& PivumpsVector0<Symmetry,Scalar>::
operator/= (const OtherScalar &alpha)
{
	for (size_t q=0; q<C.dim; ++q)
	{
		C.block[q] /= alpha;
	}
	return *this;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
PivumpsVector0<Symmetry,Scalar> operator* (const OtherScalar &alpha, PivumpsVector0<Symmetry,Scalar> V)
{
	return V *= alpha;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
PivumpsVector0<Symmetry,Scalar> operator* (PivumpsVector0<Symmetry,Scalar> V, const OtherScalar &alpha)
{
	return V *= alpha;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
PivumpsVector0<Symmetry,Scalar> operator/ (PivumpsVector0<Symmetry,Scalar> V, const OtherScalar &alpha)
{
	return V /= alpha;
}

template<typename Symmetry, typename Scalar>
PivumpsVector0<Symmetry,Scalar> operator+ (const PivumpsVector0<Symmetry,Scalar> &V1, const PivumpsVector0<Symmetry,Scalar> &V2)
{
	PivumpsVector0<Symmetry,Scalar> Vout = V1;
	Vout += V2;
	return Vout;
}

template<typename Symmetry, typename Scalar>
PivumpsVector0<Symmetry,Scalar> operator- (const PivumpsVector0<Symmetry,Scalar> &V1, const PivumpsVector0<Symmetry,Scalar> &V2)
{
	PivumpsVector0<Symmetry,Scalar> Vout = V1;
	Vout -= V2;
	return Vout;
}

template<typename Symmetry, typename Scalar>
Scalar dot (const PivumpsVector0<Symmetry,Scalar> &V1, const PivumpsVector0<Symmetry,Scalar> &V2)
{
	Scalar res = 0.;
	for (size_t q=0; q<V2.C.dim; ++q)
	for (size_t i=0; i<V2.C.block[q].cols(); ++i)
	{
		res += V1.C.block[q].col(i).dot(V2.C.block[q].col(i));
	}
	return res;
}

template<typename Symmetry, typename Scalar>
double squaredNorm (const PivumpsVector0<Symmetry,Scalar> &V)
{
	double res = 0.;
	for (size_t q=0; q<V.C.dim; ++q)
	{
		res += V.C.block[q].colwise().squaredNorm().sum();
	}
	return res;
}

template<typename Symmetry, typename Scalar>
inline double norm (const PivumpsVector0<Symmetry,Scalar> &V)
{
	return sqrt(squaredNorm(V));
}

template<typename Symmetry, typename Scalar>
inline void normalize (PivumpsVector0<Symmetry,Scalar> &V)
{
	V /= norm(V);
}

template<typename Symmetry, typename Scalar>
double infNorm (const PivumpsVector0<Symmetry,Scalar> &V1, const PivumpsVector0<Symmetry,Scalar> &V2)
{
	double res = 0.;
	for (size_t q=0; q<V1.C.dim; ++q)
	{
		double tmp = (V1.C.block[q]-V2.C.block[q]).template lpNorm<Eigen::Infinity>();
		if (tmp>res) {res = tmp;}
	}
	return res;
}

template<typename Symmetry, typename Scalar>
void swap (PivumpsVector0<Symmetry,Scalar> &V1, PivumpsVector0<Symmetry,Scalar> &V2)
{
	for (size_t q=0; q<V1.C.dim; ++q)
	{
		V1.C.block[q].swap(V2.C.block[q]);
	}
}

template<typename Symmetry, typename Scalar>
struct GaussianRandomVector<PivumpsVector0<Symmetry,Scalar>,Scalar>
{
	static void fill (size_t N, PivumpsVector0<Symmetry,Scalar> &Vout)
	{
		for (size_t q=0; q<Vout.C.dim; ++q)
		for (size_t a1=0; a1<Vout.C.block[q].rows(); ++a1)
		for (size_t a2=0; a2<Vout.C.block[q].cols(); ++a2)
		{
			Vout.C.block[q](a1,a2) = threadSafeRandUniform<Scalar>(-1.,1.);
		}
		normalize(Vout);
	}
};

/**Performs the local update of \f$A_C\f$ (eq. 11) with an explicit 2-site Hamiltonian.*/
template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivumpsMatrix<Symmetry,Scalar,MpoScalar> &H, const PivotVectorQ<Symmetry,Scalar> &Vin, PivotVectorQ<Symmetry,Scalar> &Vout)
{
	size_t D = H.qloc.size();
	Vout = Vin;
	for (size_t s=0; s<D; ++s)
	{
		Vout.A[s].block[0].setZero();
	}
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	{
		if (H.h[0][s1][s2][s3][s4] != 0.)
		{
			Vout.A[s3].block[0] += H.h[0][s1][s2][s3][s4] * H.AL[s1].block[0].adjoint() * H.AL[s2].block[0] * Vin.A[s4].block[0];
		}
	}
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	{
		if (H.h[1][s1][s2][s3][s4] != 0.)
		{
			Vout.A[s1].block[0] += H.h[1][s1][s2][s3][s4] * Vin.A[s2].block[0] * H.AR[s4].block[0] * H.AR[s3].block[0].adjoint();
		}
	}
	
	for (size_t s=0; s<D; ++s)
	{
		Vout.A[s].block[0] += H.L * Vin.A[s].block[0];
		Vout.A[s].block[0] += Vin.A[s].block[0] * H.R;
	}
}

/**Performs \p HxV in place.*/
template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivumpsMatrix<Symmetry,Scalar,MpoScalar> &H, PivotVectorQ<Symmetry,Scalar> &Vinout)
{
	PivotVectorQ<Symmetry,Scalar> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

/**Performs the local update of \f$C\f$ (eq. 16) with an explicit 2-site Hamiltonian.*/
template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivumpsMatrix<Symmetry,Scalar,MpoScalar> &H, const PivumpsVector0<Symmetry,Scalar> &Vin, PivumpsVector0<Symmetry,Scalar> &Vout)
{
	size_t D = H.qloc.size();
	
	Vout = Vin;
	Vout.C.setZero();
	
	for (size_t s1=0; s1<D; ++s1)
	for (size_t s2=0; s2<D; ++s2)
	for (size_t s3=0; s3<D; ++s3)
	for (size_t s4=0; s4<D; ++s4)
	{
		if (H.h[1][s1][s2][s3][s4] != 0.)
		{
			Vout.C.block[0] += H.h[1][s1][s2][s3][s4] * H.AL[s1].block[0].adjoint() * H.AL[s2].block[0] 
			                                          * Vin.C.block[0] 
			                                          * H.AR[s4].block[0] * H.AR[s3].block[0].adjoint();
		}
	}
	
	Vout.C.block[0] += H.L * Vin.C.block[0];
	Vout.C.block[0] += Vin.C.block[0] * H.R;
}

/**Performs \p HxV in place.*/
template<typename Symmetry, typename Scalar, typename MpoScalar>
void HxV (const PivumpsMatrix<Symmetry,Scalar,MpoScalar> &H, PivumpsVector0<Symmetry,Scalar> &Vinout)
{
	PivumpsVector0<Symmetry,Scalar> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

/**Calculates the matrix element between two UMPS and an MPO. Goes from the left and uses \f$A_C\f$ and \f$A_R\f$.*/
template<typename Symmetry, typename MpoScalar, typename Scalar>
Scalar avg (const Umps<Symmetry,Scalar> &Vbra, 
            const Mpo<Symmetry,MpoScalar> &O, 
            const Umps<Symmetry,Scalar> &Vket)
{
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Bnext;
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > B;
	
	B.setIdentity(Vbra.get_frst_rows(),Vbra.get_frst_rows(),1,1);
	for (size_t l=0; l<O.length(); ++l)
	{
		GAUGE::OPTION g = (l==0)? GAUGE::C : GAUGE::R;
		contract_L(B, Vbra.A_at(g,l%Vket.length()), O.W_at(l), Vket.A_at(g,l%Vket.length()), O.locBasis(l), O.opBasis(l), Bnext);
		
		B.clear();
		B = Bnext;
		Bnext.clear();
	}
	
	if (B.dim == 1)
	{
		return B.block[0][0][0].trace();
	}
	else
	{
		lout << "Warning: Result of contraction in <φ|O|ψ> has several blocks, returning 0!" << endl;
		lout << "MPS in question: " << Vket.info() << endl;
		lout << "MPO in question: " << O.info() << endl;
		return 0;
	}
}

/**Calculates the matrix element for a vector of MPOs, summing up the result.*/
template<typename Symmetry, typename MpoScalar, typename Scalar>
Scalar avg (const Umps<Symmetry,Scalar> &Vbra, 
            const vector<Mpo<Symmetry,MpoScalar> > &O, 
            const Umps<Symmetry,Scalar> &Vket)
{
	Scalar out = 0;
	
	for (int t=0; t<O.size(); ++t)
	{
//		cout << "partial val=" << avg(Vbra,O[t],Vket) << endl;
		out += avg(Vbra,O[t],Vket);
	}
	return out;
}

#endif
