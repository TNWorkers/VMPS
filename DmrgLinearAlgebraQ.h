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
template<typename Symmetry, typename Scalar>
Scalar dot (const MpsQ<Symmetry,Scalar> &Vbra, const MpsQ<Symmetry,Scalar> &Vket)
{
	return Vbra.dot(Vket);
}

/**Swaps two MpsQ.*/
template<typename Symmetry, typename Scalar> 
void swap (MpsQ<Symmetry,Scalar> &V1, MpsQ<Symmetry,Scalar> &V2)
{
	V1.swap(V2);
}

/**Calculates the expectation value \f$\left<\Psi_{bra}|O|\Psi_{ket}\right>\f$
\param Vbra : input \f$\left<\Psi_{bra}\right|\f$
\param O : input MpoQ
\param Vket : input \f$\left|\Psi_{ket}\right>\f$
\param USE_SQUARE : If \p true, uses the square of \p O stored in \p O itself. Call MpoQ::check_SQUARE() first to see whether it was calculated.
\param INFINITE_BC : If \p true, uses infinite boundary conditions (sets initial 3-leg tensor to identity instead of vacuum).
\param DIR : whether to contract going left or right (should obviously make no difference, useful for testing purposes)
*/
template<typename Symmetry, typename MpoScalar, typename Scalar>
Scalar avg (const MpsQ<Symmetry,Scalar> &Vbra, 
            const MpoQ<Symmetry,MpoScalar> &O, 
            const MpsQ<Symmetry,Scalar> &Vket, 
            bool USE_SQUARE = false,  
            DMRG::DIRECTION::OPTION DIR = DMRG::DIRECTION::RIGHT)
{
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Bnext;
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > B;
	
	if (DIR == DMRG::DIRECTION::RIGHT)
	{
		B.setVacuum();
		for (size_t l=0; l<O.length(); ++l)
		{
			if (USE_SQUARE == true)
			{
				contract_L(B, Vbra.A_at(l), O.Wsq_at(l), Vket.A_at(l), O.locBasis(l), Bnext);
			}
			else
			{
				contract_L(B, Vbra.A_at(l), O.W_at(l), Vket.A_at(l), O.locBasis(l), Bnext);
			}
			B.clear();
			B = Bnext;
			Bnext.clear();
		}
	}
	else
	{
		B.setTarget(qarray3<Symmetry::Nq>{Vket.Qtarget(), Vbra.Qtarget(), O.Qtarget()});
//		for (int l=O.length()-1; l>=0; --l)
		for (size_t l=O.length()-1; l!=-1; --l)
		{
			if (USE_SQUARE == true)
			{
				contract_R(B, Vbra.A_at(l), O.Wsq_at(l), Vket.A_at(l), O.locBasis(l), Bnext);
			}
			else
			{
				contract_R(B, Vbra.A_at(l), O.W_at(l), Vket.A_at(l), O.locBasis(l), Bnext);
			}
			B.clear();
			B = Bnext;
			Bnext.clear();
		}
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

/**Calculates the expectation value \f$\left<\Psi_{bra}|O_{1}O_{2}|\Psi_{ket}\right>\f$
Only a left-to-right contraction is implemented.
\param Vbra : input \f$\left<\Psi_{bra}\right|\f$
\param O1 : input MpoQ
\param O2 : input MpoQ
\param Vket : input \f$\left|\Psi_{ket}\right>\f$
*/
template<typename Symmetry, typename MpoScalar, typename Scalar>
Scalar avg (const MpsQ<Symmetry,Scalar> &Vbra, 
            const MpoQ<Symmetry,MpoScalar> &O1,
            const MpoQ<Symmetry,MpoScalar> &O2, 
            const MpsQ<Symmetry,Scalar> &Vket)
{
	Multipede<4,Symmetry,Matrix<Scalar,Dynamic,Dynamic> > B;
	Multipede<4,Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Bnext;
	
	B.setVacuum();
	for (size_t l=0; l<O2.length(); ++l)
	{
		contract_L(B, Vbra.A_at(l), O1.W_at(l), O2.W_at(l), Vket.A_at(l), O2.locBasis(l), Bnext);
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
	
	if (B.dim == 1)
	{
		return B.block[0][0][0].trace();
	}
	else
	{
		lout << "Warning: Result of contraction in <φ|O1*O2|ψ> has several blocks, returning 0!" << endl;
		lout << "MPS in question: " << Vket.info() << endl;
		lout << "MPO1 in question: " << O1.info() << endl;
		lout << "MPO2 in question: " << O2.info() << endl;
		return 0;
	}
}

/**Apply an MpoQ to an MpsQ \f$\left|\Psi_{out}\right> = H \left|\Psi_{in}\right>\f$ by using the zip-up algorithm (Stoudenmire, White 2010).
\param H : input Hamiltonian
\param Vin : input \f$\left|\Psi_{in}\right>\f$
\param Vout : output \f$\left|\Psi_{out}\right>\f$
\param VERBOSITY : verbosity level*/
template<typename Symmetry, typename MpoScalar, typename Scalar>
void HxV (const MpoQ<Symmetry,MpoScalar> &H, const MpsQ<Symmetry,Scalar> &Vin, MpsQ<Symmetry,Scalar> &Vout, 
          DMRG::VERBOSITY::OPTION VERBOSITY) //=DMRG::VERBOSITY::HALFSWEEPWISE)
{
	Stopwatch<> Chronos;
	
	MpsQCompressor<Symmetry::Nq,Scalar,MpoScalar> Compadre(VERBOSITY);
	Compadre.varCompress(H,Vin, Vout, Vin.calc_Dmax());
	
	if (VERBOSITY != DMRG::VERBOSITY::SILENT)
	{
		lout << Compadre.info() << endl;
		lout << Chronos.info("HxV") << endl;
		lout << "Vout: " << Vout.info() << endl << endl;
	}
}

/**Performs an orthogonal polynomial iteration step \f$\left|\Psi_{out}\right> = \cdot H \left|\Psi_{in,1}\right> - B \left|\Psi_{in,2}\right>\f$ as needed in the polynomial recursion relation \f$P_n = (C_n x - A_n) P_{n-1} - B_n P_{n-2}\f$.
\warning The Hamiltonian is assumed to be rescaled by \p C_n and \p A_n already.
\param H : input Hamiltonian
\param Vin1 : input MpsQ \f$\left|T_{n-1}\right>\f$
\param polyB : the coefficient before the subtracted vector
\param Vin2 : input MpsQ \f$\left|T_{n-2}\right>\f$
\param Vout : output MpsQ \f$\left|T_{n}\right>\f$
\param VERBOSITY : verbosity level*/
template<typename Symmetry, typename MpoScalar, typename Scalar>
void polyIter (const MpoQ<Symmetry,MpoScalar> &H, const MpsQ<Symmetry,Scalar> &Vin1, double polyB, const MpsQ<Symmetry,Scalar> &Vin2, MpsQ<Symmetry,Scalar> &Vout, 
               DMRG::VERBOSITY::OPTION VERBOSITY=DMRG::VERBOSITY::HALFSWEEPWISE)
{
	Stopwatch<> Chronos;
	MpsQCompressor<Symmetry::Nq,Scalar,MpoScalar> Compadre(VERBOSITY);
	Compadre.polyCompress(H,Vin1,polyB,Vin2, Vout, Vin1.calc_Dmax());
	
	if (VERBOSITY != DMRG::VERBOSITY::SILENT)
	{
		lout << Compadre.info() << endl;
		lout << Chronos.info(make_string("polyIter B=",polyB)) << endl;
		lout << "Vout: " << Vout.info() << endl << endl;
	}
}

template<typename Symmetry, typename MpoScalar, typename Scalar>
void HxV (const MpoQ<Symmetry,MpoScalar> &H, MpsQ<Symmetry,Scalar> &Vinout, 
          DMRG::VERBOSITY::OPTION VERBOSITY=DMRG::VERBOSITY::HALFSWEEPWISE)
{
	MpsQ<Symmetry,Scalar> Vtmp;
	HxV(H,Vinout,Vtmp,VERBOSITY);
	Vinout = Vtmp;
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
void addScale (const OtherScalar alpha, const MpsQ<Symmetry,Scalar> &Vin, MpsQ<Symmetry,Scalar> &Vout, 
               DMRG::VERBOSITY::OPTION VERBOSITY=DMRG::VERBOSITY::SILENT)
{
	Stopwatch<> Chronos;
	MpsQCompressor<Symmetry::Nq,Scalar,OtherScalar> Compadre(VERBOSITY);
	size_t Dstart = Vout.calc_Dmax();
	MpsQ<Symmetry,Scalar> Vtmp = Vout;
	Vtmp.addScale(alpha,Vin,false);
//	Compadre.varCompress(Vtmp, Vout, Dstart, 1e-3, 100, 1, DMRG::COMPRESSION::RANDOM);
	Compadre.varCompress(Vtmp, Vout, Dstart);
	
	if (VERBOSITY != DMRG::VERBOSITY::SILENT)
	{
		lout << Compadre.info() << endl;
		lout << Chronos.info("V+V") << endl;
		lout << "Vout: " << Vout.info() << endl << endl;
	}
}

template<typename Symmetry, typename MpoScalar, typename Scalar>
void OxV (const MpoQ<Symmetry,MpoScalar> &O, const MpsQ<Symmetry,Scalar> &Vin, MpsQ<Symmetry,Scalar> &Vout, DMRG::BROOM::OPTION TOOL) //=DMRG::BROOM::SVD)
{
	vector<Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > C;
	vector<Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > Cnext;
	
	if (TOOL == DMRG::BROOM::QR)
	{
		assert(O.Qtarget() == Symmetry::qvacuum() and 
		       "Need a qnumber-conserving operator in OxV for QR option!");
		Vout = Vin;
	}
	else
	{
		Vout.outerResize(O, Vin.Qtarget()+O.Qtarget());
	}
	
	contract_C0(O.locBasis(0), O.W_at(0), Vin.A[0], C);
	Vout.set_A_from_C(0,C,TOOL);
	
	for (size_t l=1; l<Vin.length(); ++l)
	{
		Stopwatch<> Chronos;
		contract_C(O.locBasis(l), Vout.A[l-1], O.W_at(l), Vin.A[l], C, Cnext);
//		lout << "l=" << l << ", " << Chronos.info("contract_C") << endl;
		
		for (size_t s1=0; s1<O.locBasis(l).size(); ++s1)
		{
			C[s1].clear();
			C[s1] = Cnext[s1];
			Cnext[s1].clear();
		}
		
		Vout.set_A_from_C(l,C,TOOL);
//		lout << "l=" << l << ", " << Chronos.info("set_A_from_C") << endl;
	}
	
	Vout.mend();
	Vout.pivot = Vout.length()-1;
	
//	std::array<Tripod<Nq,Matrix<Scalar,Dynamic,Dynamic> >,D> C;
//	std::array<Tripod<Nq,Matrix<Scalar,Dynamic,Dynamic> >,D> Cnext;
//	
//	if (TOOL == DMRG::BROOM::QR)
//	{
//		assert(O.Qtarget() == qvacuum<Nq>() and 
//		       "Need a qnumber-conserving operator in OxV for QR option!");
//		Vout = Vin;
//	}
//	else
//	{
//		Vout.outerResize(O, Vin.Qtarget()+O.Qtarget());
//	}
//	
//	contract_C0(O.locBasis(0), O.W_at(0), Vin.A[0], C);
//	Vout.set_A_from_C(0,C,TOOL);
//	
//	for (size_t l=1; l<Vin.length(); ++l)
//	{
//		contract_C(O.locBasis(l), Vout.A[l-1], O.W_at(l), Vin.A[l], C, Cnext);
//		
//		for (size_t s1=0; s1<D; ++s1)
//		{
//			C[s1].clear();
//			C[s1] = Cnext[s1];
//			Cnext[s1].clear();
//		}
//		
//		Vout.set_A_from_C(l,C,TOOL);
//	}
//	
//	Vout.mend();
//	Vout.pivot = Vout.length()-1;
}

template<typename Symmetry, typename MpoScalar, typename Scalar>
void OxV (const MpoQ<Symmetry,MpoScalar> &O, MpsQ<Symmetry,Scalar> &Vinout)
{
	MpsQ<Symmetry,Scalar> Vtmp;
	OxV(O,Vinout,Vtmp);
	Vinout = Vtmp;
}

// unfinished:
//template<size_t Nq, typename Scalar>
//void OxV_exact (const MpoQ<D,Nq> &O, const MpsQ<Nq,Scalar> &Vin, MpsQ<Nq,Scalar> &Vout)
//{
//	Vout = MpsQ<Nq,Scalar>(Vin.length(), 1, Vin.locbasis(), Vout.Qtarget()+O.Qtarget());
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
