#ifndef STRAWBERRY_DMRGEXTERNAL_WITH_Q
#define STRAWBERRY_DMRGEXTERNAL_WITH_Q

#include "MpsQ.h"
#include "MpoQ.h"
#include "DmrgContractionsQ.h"

/**@file
\brief External functions to manipulate MpsQ and MpoQ objects.*/

/**Calculates the scalar product \f$\left<\Psi_{bra}|\Psi_{ket}\right>\f$.
\param Vbra : input \f$\left<\Psi_{bra}\right|\f$
\param Vket : input \f$\left|\Psi_{ket}\right>\f$*/
template<size_t D, size_t Nq, typename Scalar>
Scalar dot (const MpsQ<D,Nq,Scalar> &Vbra, const MpsQ<D,Nq,Scalar> &Vket)
{
	return Vbra.dot(Vket);
}

/**Swaps two MpsQ.*/
template<size_t D, size_t Nq, typename Scalar> 
void swap (MpsQ<D,Nq,Scalar> &V1, MpsQ<D,Nq,Scalar> &V2)
{
	V1.swap(V2);
}

//template<size_t D, size_t Nq, typename Scalar>
//Scalar avg_ (const MpsQ<D,Nq,Scalar> &Vbra, 
//             const MpoQ<D,Nq> &O, 
//             const MpsQ<D,Nq,Scalar> &Vket, 
//             bool USE_SQUARE = false, 
//             DMRG::DIRECTION::OPTION DIR = DMRG::DIRECTION::RIGHT)
//{
//	Tripod<Nq,Matrix<Scalar,Dynamic,Dynamic> > Bnext;
//	Tripod<Nq,Matrix<Scalar,Dynamic,Dynamic> > B;
//	
//	if (DIR == DMRG::DIRECTION::RIGHT)
//	{
//		B.setVacuum();
//		for (size_t l=0; l<O.length(); ++l)
//		{
//			if (USE_SQUARE == true)
//			{
//				contract_L(B, Vbra.A_at(l), O.Wsq_at(l), Vket.A_at(l), O.locBasis(), Bnext);
//			}
//			else
//			{
//				contract_L(B, Vbra.A_at(l), O.W_at(l), Vket.A_at(l), O.locBasis(), Bnext);
//			}
//			B.clear();
//			B = Bnext;
//			Bnext.clear();
//		}
//	}
//	else
//	{
//		B.setTarget(qarray3<Nq>{Vket.Qtarget(), Vbra.Qtarget(), O.Qtarget()});
////		for (int l=O.length()-1; l>=0; --l)
//		for (size_t l=O.length()-1; l!=-1; --l)
//		{
//			if (USE_SQUARE == true)
//			{
//				contract_R(B, Vbra.A_at(l), O.Wsq_at(l), Vket.A_at(l), O.locBasis(), Bnext);
//			}
//			else
//			{
//				contract_R(B, Vbra.A_at(l), O.W_at(l), Vket.A_at(l), O.locBasis(), Bnext);
//			}
//			B.clear();
//			B = Bnext;
//			Bnext.clear();
//		}
//	}
//	
//	assert(B.dim == 1 and 
//	       B.block[0][0][0].rows() == 1 and 
//	       B.block[0][0][0].cols() == 1 and
//	       "Result of contraction in <φ|O|ψ> is not a scalar!");
//	
//	return B.block[0][0][0](0,0);
//}

/**Calculates the expectation value \f$\left<\Psi_{bra}|O|\Psi_{ket}\right>\f$
\param Vbra : input \f$\left<\Psi_{bra}\right|\f$
\param O : input MpoQ
\param Vket : input \f$\left|\Psi_{ket}\right>\f$
\param USE_SQUARE : If \p true, uses the square of \p O stored in \p O itself. Call MpoQ::check_SQUARE() first to see whether it was calculated.
\param DIR : whether to contract going left or right (should obviously make no difference, useful for testing purposes)
*/
template<size_t D, size_t Nq, typename MpoScalar, typename Scalar>
Scalar avg (const MpsQ<D,Nq,Scalar> &Vbra, 
            const MpoQ<D,Nq,MpoScalar> &O, 
            const MpsQ<D,Nq,Scalar> &Vket, 
            bool USE_SQUARE = false, 
            DMRG::DIRECTION::OPTION DIR = DMRG::DIRECTION::RIGHT)
{
	Tripod<Nq,Matrix<Scalar,Dynamic,Dynamic> > Bnext;
	Tripod<Nq,Matrix<Scalar,Dynamic,Dynamic> > B;
	
	if (DIR == DMRG::DIRECTION::RIGHT)
	{
		B.setVacuum();
		for (size_t l=0; l<O.length(); ++l)
		{
			if (USE_SQUARE == true)
			{
				contract_L(B, Vbra.A_at(l), O.Wsq_at(l), Vket.A_at(l), O.locBasis(), Bnext);
			}
			else
			{
				contract_L(B, Vbra.A_at(l), O.W_at(l), Vket.A_at(l), O.locBasis(), Bnext);
			}
			B.clear();
			B = Bnext;
			Bnext.clear();
		}
	}
	else
	{
		B.setTarget(qarray3<Nq>{Vket.Qtarget(), Vbra.Qtarget(), O.Qtarget()});
//		for (int l=O.length()-1; l>=0; --l)
		for (size_t l=O.length()-1; l!=-1; --l)
		{
			if (USE_SQUARE == true)
			{
				contract_R(B, Vbra.A_at(l), O.Wsq_at(l), Vket.A_at(l), O.locBasis(), Bnext);
			}
			else
			{
				contract_R(B, Vbra.A_at(l), O.W_at(l), Vket.A_at(l), O.locBasis(), Bnext);
			}
			B.clear();
			B = Bnext;
			Bnext.clear();
		}
	}
	
//	cout << "B.dim=" << B.dim << endl;
//	for (size_t q=0; q<B.dim; ++q)
//	{
//		cout << "q=" << B.in(q) << ", " << B.out(q) << ", " << B.mid(q) << endl 
//		<< B.block[q][0][0] << endl;
//	}
	
	assert(B.dim == 1 and 
	       B.block[0][0][0].rows() == 1 and
	       B.block[0][0][0].cols() == 1 and
	       "Result of contraction in <φ|O|ψ> is not a scalar!");
	
	return B.block[0][0][0](0,0);
	
//	Tripod<Nq,Matrix<Scalar,Dynamic,Dynamic> > Lnext;
//	Tripod<Nq,Matrix<Scalar,Dynamic,Dynamic> > L;
//	Tripod<Nq,Matrix<Scalar,Dynamic,Dynamic> > Rnext;
//	Tripod<Nq,Matrix<Scalar,Dynamic,Dynamic> > R;
//	
//	L.setVacuum();
//	R.setTarget(qarray3<Nq>{Vket.Qtarget(), Vbra.Qtarget(), O.Qtarget()});
//	
//	size_t last = O.length()/2;
//	
//	#pragma omp parallel sections
//	{
//		#pragma omp section
//		for (size_t l=0; l<last; ++l)
//		{
//			if (USE_SQUARE == true)
//			{
//				contract_L(L, Vbra.A_at(l), O.Wsq_at(l), Vket.A_at(l), O.locBasis(), Lnext);
//			}
//			else
//			{
//				contract_L(L, Vbra.A_at(l), O.W_at(l), Vket.A_at(l), O.locBasis(), Lnext);
//			}
//			L.clear();
//			L = Lnext;
//			Lnext.clear();
//		}
//		
//		#pragma omp section
//		for (int l=O.length()-1; l>last; --l)
//		{
//			if (USE_SQUARE == true)
//			{
//				contract_R(R, Vbra.A_at(l), O.Wsq_at(l), Vket.A_at(l), O.locBasis(), Rnext);
//			}
//			else
//			{
//				contract_R(R, Vbra.A_at(l), O.W_at(l), Vket.A_at(l), O.locBasis(), Rnext);
//			}
//			R.clear();
//			R = Rnext;
//			Rnext.clear();
//		}
//	}
//	
//	if (USE_SQUARE == true)
//	{
//		return contract_LR(L, Vbra.A_at(last), O.Wsq_at(last), Vket.A_at(last), R, O.locBasis());
//	}
//	else
//	{
//		return contract_LR(L, Vbra.A_at(last), O.W_at(last), Vket.A_at(last), R, O.locBasis());
//	}
}

/**Calculates the expectation value \f$\left<\Psi_{bra}|O_{bra}O_{ket}|\Psi_{ket}\right>\f$
Only a left-to-right contraction is implemented.
\param Vbra : input \f$\left<\Psi_{bra}\right|\f$
\param O1 : input MpoQ
\param O2 : input MpoQ
\param Vket : input \f$\left|\Psi_{ket}\right>\f$
*/
template<size_t D, size_t Nq, typename MpoScalar, typename Scalar>
Scalar avg (const MpsQ<D,Nq,Scalar> &Vbra, 
            const MpoQ<D,Nq,MpoScalar> &O1,
            const MpoQ<D,Nq,MpoScalar> &O2, 
            const MpsQ<D,Nq,Scalar> &Vket)
{
	Multipede<4,Nq,Matrix<Scalar,Dynamic,Dynamic> > B;
	Multipede<4,Nq,Matrix<Scalar,Dynamic,Dynamic> > Bnext;
	
	B.setVacuum();
	for (size_t l=1; l<O2.length()+1; ++l)
	{
		contract_L(B, Vbra.A_at(l-1), O1.W_at(l-1), O2.W_at(l-1), Vket.A_at(l-1), O2.locBasis(), Bnext);
		B.clear();
		B = Bnext;
		Bnext.clear();
	}
	
//	cout << "B.dim=" << B.dim << endl;
//	for (size_t q=0; q<B.dim; ++q)
//	{
//		cout << "q=" << B.in(q) << ", " << B.out(q) << ", " << B.top(q) << ", " << B.bot(q) << endl 
//		<< B.block[q][0][0] << endl;
//	}
	
	assert(B.dim == 1 and 
	       B.block[0][0][0].rows() == 1 and 
	       B.block[0][0][0].cols() == 1 and
	       "Result of contraction in <φ|O1*O2|ψ> is not a scalar!");
	
	return B.block[0][0][0](0,0);
}

/**Apply an MpoQ to an MpsQ \f$\left|\Psi_{out}\right> = H \left|\Psi_{in}\right>\f$ by using the zip-up algorithm (Stoudenmire, White 2010).
\param H : input Hamiltonian
\param Vin : input \f$\left|\Psi_{in}\right>\f$
\param Vout : output \f$\left|\Psi_{out}\right>\f$*/
template<size_t D, size_t Nq, typename MpoScalar, typename Scalar>
void HxV (const MpoQ<D,Nq,MpoScalar> &H, const MpsQ<D,Nq,Scalar> &Vin, MpsQ<D,Nq,Scalar> &Vout)
{
	Stopwatch Chronos;
	
	MpsQCompressor<D,Nq,Scalar,MpoScalar> Compadre(DMRG::VERBOSITY::STEPWISE);
	Compadre.varCompress(H,Vin, Vout, Vin.calc_Dmax(), 1e-8);
	lout << Compadre.info() << endl;
	
	lout << Chronos.info("HxV") << endl;
	lout << "Vout: " << Vout.info() << endl << endl;
}

/**Performs a Chebyshev iteration step \f$\left|\Psi_{out}\right> = 2 \cdot H \left|\Psi_{in,1}\right> - \left|\Psi_{in,2}\right> \f$.
\param H : input Hamiltonian
\param Vin1 : input MpsQ \f$\left|T_{n-1}\right>\f$
\param Vin2 : input MpsQ \f$\left|T_{n-2}\right>\f$
\param Vout : output MpsQ \f$\left|T_{n}\right>\f$*/
template<size_t D, size_t Nq, typename MpoScalar, typename Scalar>
void chebIter (const MpoQ<D,Nq,MpoScalar> &H, const MpsQ<D,Nq,Scalar> &Vin1, const MpsQ<D,Nq,Scalar> &Vin2, MpsQ<D,Nq,Scalar> &Vout)
{
	Stopwatch Chronos;
	MpsQCompressor<D,Nq,Scalar,MpoScalar> Compadre(DMRG::VERBOSITY::STEPWISE);
	Compadre.chebCompress(H,Vin1,Vin2, Vout, Vin1.calc_Dmax());
	lout << Compadre.info() << endl;
	lout << Chronos.info("chebIter") << endl;
	lout << "Vout: " << Vout.info() << endl << endl;
}

template<size_t D, size_t Nq, typename MpoScalar, typename Scalar>
void HxV (const MpoQ<D,Nq,MpoScalar> &H, MpsQ<D,Nq,Scalar> &Vinout)
{
	MpsQ<D,Nq,Scalar> Vtmp;
	HxV(H,Vinout,Vtmp);
	Vinout = Vtmp;
}

template<size_t D, size_t Nq, typename Scalar, typename OtherScalar>
void addScale (const OtherScalar alpha, const MpsQ<D,Nq,Scalar> &Vin, MpsQ<D,Nq,Scalar> &Vout)
{
	Stopwatch Chronos;
	
	MpsQCompressor<D,Nq,Scalar,OtherScalar> Compadre;
	size_t Dstart = Vout.calc_Dmax();
	MpsQ<D,Nq,Scalar> Vtmp = Vout;
	Vtmp.addScale(alpha,Vin,true);
	Compadre.varCompress(Vtmp, Vout, Dstart, 1e-3);
	
	lout << Chronos.info("V+V") << endl;
	lout << "Vin: " << Vin.info() << endl;
	lout << "Vout: " << Vout.info() << endl << endl;
}

template<size_t D, size_t Nq, typename MpoScalar, typename Scalar>
void OxV (const MpoQ<D,Nq,MpoScalar> &O, const MpsQ<D,Nq,Scalar> &Vin, MpsQ<D,Nq,Scalar> &Vout, DMRG::BROOM::OPTION TOOL=DMRG::BROOM::SVD)
{
	std::array<Tripod<Nq,Matrix<Scalar,Dynamic,Dynamic> >,D> C;
	std::array<Tripod<Nq,Matrix<Scalar,Dynamic,Dynamic> >,D> Cnext;
	
	if (TOOL == DMRG::BROOM::QR)
	{
		assert(O.Qtarget() == qvacuum<Nq>() and 
		       "Need a qnumber-conserving operator in OxV for QR option!");
		Vout = Vin;
	}
	else
	{
		Vout.outerResize(O, Vin.Qtarget()+O.Qtarget());
	}
	
	contract_C0(O.locBasis(), O.W_at(0), Vin.A[0], C);
	Vout.set_A_from_C(0,C,TOOL);
	
	for (size_t l=1; l<Vin.length(); ++l)
	{
		contract_C(O.locBasis(), Vout.A[l-1], O.W_at(l), Vin.A[l], C, Cnext);
		
		for (size_t s1=0; s1<D; ++s1)
		{
			C[s1].clear();
			C[s1] = Cnext[s1];
			Cnext[s1].clear();
		}
		
		Vout.set_A_from_C(l,C,TOOL);
	}
	
	Vout.mend();
	Vout.pivot = Vout.length()-1;
}

template<size_t D, size_t Nq, typename MpoScalar, typename Scalar>
void OxV (const MpoQ<D,Nq,MpoScalar> &O, MpsQ<D,Nq,Scalar> &Vinout)
{
	MpsQ<D,Nq,Scalar> Vtmp;
	OxV(O,Vinout,Vtmp);
	Vinout = Vtmp;
}

//template<size_t D, size_t Nq, typename Scalar>
//void OxV_exact (const MpoQ<D,Nq> &O, const MpsQ<D,Nq,Scalar> &Vin, MpsQ<D,Nq,Scalar> &Vout)
//{
//	Vout = MpsQ<D,Nq,Scalar>(Vin.length(), 1, Vin.locbasis(), Vout.Qtarget()+O.Qtarget());
//	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	for (size_t l=0; l<Vout.length(); ++l)
//	for (size_t qA2=0; qA2<Vin.A[l][s2].dim; ++qA2)
//	{
//		Matrix<Scalar,Dynamic,Dynamic> Mtmp = tensor_product(O.W[s1][s2], Vin.A[l][s2].block[qA2]);
//	}
//}

#endif
