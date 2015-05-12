#ifndef VANILLA_DMRG_EXTERNAL
#define VANILLA_DMRG_EXTERNAL

#include "unsupported/Eigen/KroneckerProduct"
#include <algorithm>

#include "Mps.h"
#include "Mpo.h"
#include "DmrgContractions.h"

template<size_t D, typename Scalar>
Scalar dot (const Mps<D,Scalar> &Vbra, const Mps<D,Scalar> &Vket)
{
	return Vbra.dot(Vket);
}

template<size_t D, typename Scalar> 
void swap (Mps<D,Scalar> &V1, Mps<D,Scalar> &V2)
{
	V1.swap(V2);
}

template<size_t D, typename Scalar>
Scalar avg (const Mps<D,Scalar> &Vbra, 
            const Mpo<D> &O, 
            const Mps<D,Scalar> &Vket, 
            bool USE_SQUARE = false, 
            DMRG::DIRECTION::OPTION DIR = DMRG::DIRECTION::RIGHT)
{
	vector<Matrix<Scalar,Dynamic,Dynamic> > Bnext;
	vector<Matrix<Scalar,Dynamic,Dynamic> > B(1);
	
	(USE_SQUARE == false)? Bnext.resize(O.auxdim()) : Bnext.resize(O.auxdim()*O.auxdim());
	
	B[0].resize(1,1);
	B[0](0,0) = 1.;
	
	if (DIR == DMRG::DIRECTION::RIGHT)
	{
		for (size_t l=0; l<O.length(); ++l)
		{
			if (USE_SQUARE == true)
			{
				contract_L(B, Vbra.A_at(l), O.Wsq_at(l), Vket.A_at(l), Bnext);
			}
			else
			{
				contract_L(B, Vbra.A_at(l), O.W_at(l), Vket.A_at(l), Bnext);
			}
			B = Bnext;
		}
	}
	else
	{
		for (size_t l=O.length(); l-->0;)
		{
			if (USE_SQUARE == true)
			{
				contract_R(B, Vbra.A_at(l), O.Wsq_at(l), Vket.A_at(l), Bnext);
			}
			else
			{
				contract_R(B, Vbra.A_at(l), O.W_at(l), Vket.A_at(l), Bnext);
			}
			B = Bnext;
		}
	}
	
	assert(B[0].rows() == 1 and B[0].cols() == 1 and 
	       "Result of contraction in <φ|O|ψ> is not a scalar!");
	
	for (size_t a=1; a<B.size(); ++a)
	{
		assert(B[a].norm() == 0. and 
		       "Result of contraction in <φ|O|ψ> is not a scalar!");
	}
	
	return B[0](0,0);
}

template<size_t D, typename Scalar>
Scalar avg (const Mps<D,Scalar> &Vbra, 
            const Mpo<D> &Obra,
            const Mpo<D> &Oket, 
            const Mps<D,Scalar> &Vket)
{
	assert(Obra.auxdim() == Oket.auxdim());
	size_t Daux = Oket.auxdim();
	vector<vector<Matrix<Scalar,Dynamic,Dynamic> > > B(1);
	vector<vector<Matrix<Scalar,Dynamic,Dynamic> > > Bnext(Daux);
	
	for (size_t i=0; i<Daux; ++i) {Bnext[i].resize(Daux);}
	B[0].resize(1);
	B[0][0].resize(1,1);
	B[0][0](0,0) = 1.;
	
	for (size_t l=0; l<Oket.length(); ++l)
	{
		contract_L(B, Vbra.A_at(l), Obra.W_at(l), Oket.W_at(l), Vket.A_at(l), Bnext);
		B = Bnext;
	}
	
	assert(B[0][0].rows() == 1 and 
	       B[0][0].cols() == 1 and
	       "Result of contraction in <φ|O1*O2|ψ> is not a scalar!");
	
	return B[0][0](0,0);
}

template<size_t D, typename Scalar> 
void HxV (const Mpo<D> &H, const Mps<D,Scalar> &Vin, Mps<D,Scalar> &Vout)
{
//	Stopwatch Chronos;
//	Vout.resize(H.length(), Vin.calc_Dmax()*Daux);
//	
//	for (int l=0; l<H.length(); ++l)
//	{
//		for (int s1=0; s1<D; ++s1)
//		{
//			Vout.A[l][s1] = KroneckerProduct<SparseMatrixXd,Matrix<Scalar,Dynamic,Dynamic> >(H.W[l][s1][0], Vin.A[l][0]);
//			for (int s2=1; s2<D; ++s2)
//			{
//				Vout.A[l][s1] += KroneckerProduct<SparseMatrixXd,Matrix<Scalar,Dynamic,Dynamic> >(H.W[l][s1][s2], Vin.A[l][s2]);
//			}
//		}
//		
//		if (l>0)
//		{
//			Vout.rightSweepStep(l-1, DMRG::BROOM::SVD);
//		}
//	}
//	cout << Vout.info() << endl;
//	Chronos.check("HxV");
	
	Stopwatch Chronos;
	MpsCompressor<D,Scalar> Compadre;
	Compadre.varCompress(H,Vin, Vout, Vin.calc_Dmax(), 1e-4);
	Chronos.check("HxV");
	cout << Compadre.info() << endl;
	cout << endl;
}

template<size_t D, typename Scalar> 
void HxV (const Mpo<D> &H, Mps<D,Scalar> &Vinout)
{
	Mps<D,Scalar> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

#endif
