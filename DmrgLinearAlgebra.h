#ifndef STRAWBERRY_DMRGEXTERNAL_WITH_Q
#define STRAWBERRY_DMRGEXTERNAL_WITH_Q

#include "Mps.h"
#include "Mpo.h"
#include "tensors/DmrgContractions.h"
#include "Stopwatch.h" // from HELPERS

/**@file
\brief External functions to manipulate Mps and Mpo objects.*/

/**
 * Calculates the scalar product \f$\left<\Psi_{bra}|\Psi_{ket}\right>\f$.
 * \param Vbra : input \f$\left<\Psi_{bra}\right|\f$
 * \param Vket : input \f$\left|\Psi_{ket}\right>\f$
 */
template<typename Symmetry, typename Scalar>
Scalar dot (const Mps<Symmetry,Scalar> &Vbra, const Mps<Symmetry,Scalar> &Vket)
{
	return Vbra.dot(Vket);
}

/**Swaps two Mps.*/
template<typename Symmetry, typename Scalar> 
void swap (Mps<Symmetry,Scalar> &V1, Mps<Symmetry,Scalar> &V2)
{
	V1.swap(V2);
}

/**
 * Calculates the expectation value \f$\left<\Psi_{bra}|O|\Psi_{ket}\right>\f$
 * \param Vbra : input \f$\left<\Psi_{bra}\right|\f$
 * \param O : input Mpo
 * \param Vket : input \f$\left|\Psi_{ket}\right>\f$
 * \param USE_SQUARE : If \p true, uses the square of \p O stored in \p O itself. Call Mpo::check_SQUARE() first to see whether it was calculated.
 * \param DIR : whether to contract going left or right (should obviously make no difference, useful for testing purposes)
 */
template<typename Symmetry, typename MpoScalar, typename Scalar>
Scalar avg (const Mps<Symmetry,Scalar> &Vbra, 
            const Mpo<Symmetry,MpoScalar> &O, 
            const Mps<Symmetry,Scalar> &Vket, 
            bool USE_SQUARE = false,  
            DMRG::DIRECTION::OPTION DIR = DMRG::DIRECTION::LEFT)
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
				if constexpr (Symmetry::NON_ABELIAN) { contract_L(B, Vbra.A_at(l), O.Vsq_at(l), Vket.A_at(l), O.locBasis(l), O.opBasisSq(l), Bnext); }
				else { contract_L(B, Vbra.A_at(l), O.Wsq_at(l), Vket.A_at(l), O.locBasis(l), O.opBasisSq(l), Bnext); }
			}
			else
			{
				contract_L(B, Vbra.A_at(l), O.W_at(l), Vket.A_at(l), O.locBasis(l), O.opBasis(l), Bnext);
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
				if constexpr (Symmetry::NON_ABELIAN) { contract_R(B, Vbra.A_at(l), O.Vsq_at(l), Vket.A_at(l), O.locBasis(l), O.opBasisSq(l), Bnext); }
				else {contract_R(B, Vbra.A_at(l), O.Wsq_at(l), Vket.A_at(l), O.locBasis(l), O.opBasisSq(l), Bnext); }
			}
			else
			{
				contract_R(B, Vbra.A_at(l), O.W_at(l), Vket.A_at(l), O.locBasis(l), O.opBasis(l), Bnext);
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
		lout << "dim=" << B.dim << endl;
		lout << "B=" << B.print(true) << endl;
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

/**
 * Calculates the expectation value \f$\left<\Psi_{bra}|O_{1}O_{2}|\Psi_{ket}\right>\f$
 * Only a left-to-right contraction is implemented.
 * \param Vbra : input \f$\left<\Psi_{bra}\right|\f$
 * \param O1 : input Mpo
 * \param O2 : input Mpo
 * \param Vket : input \f$\left|\Psi_{ket}\right>\f$
 * \param Qtarget : The quantum number of the product of \f$O_1\cdot O_2\f$. For abelian symmetries simply O1.Qtarget()+O2.Qtarget()
 */
template<typename Symmetry, typename MpoScalar, typename Scalar>
Scalar avg (const Mps<Symmetry,Scalar> &Vbra, 
            const Mpo<Symmetry,MpoScalar> &O1,
            const Mpo<Symmetry,MpoScalar> &O2, 
            const Mps<Symmetry,Scalar> &Vket,
            typename Symmetry::qType Qtarget = Symmetry::qvacuum())
{
	if constexpr (Symmetry::NON_ABELIAN )
	{
		Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Bnext;
		Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > B;
		
		B.setTarget(qarray3<Symmetry::Nq>{Vket.Qtarget(), Vbra.Qtarget(), Qtarget});
		for (size_t l=O1.length()-1; l!=-1; --l)
		{
			contract_R(B, 
			           Vbra.A_at(l), O1.W_at(l), O2.W_at(l), Vket.A_at(l), 
			           O1.locBasis(l), O1.opBasis(l), O2.opBasis(l),
			           O1.auxBasis(l+1), O2.auxBasis(l+1), O1.auxBasis(l), O2.auxBasis(l), 
			           Bnext);
			B.clear();
			B = Bnext;
//			cout << "after l=" << l << ", B.dim=" << B.dim << endl << endl;;
			Bnext.clear();
		}
		
		if (B.dim == 1)
		{
			double res = B.block[0][0][0].trace();
			if (Qtarget == Symmetry::qvacuum())
			{
#ifdef PRINT_SU2_FACTORS
				cout << termcolor::bold << termcolor::red << "Global SU2 factor in avg(Bra,O1,O2,Ket) from DmrgLinearAlgebra: " << termcolor::reset
					 << "√" << Symmetry::coeff_dot(O1.Qtarget()) << " • √" << Symmetry::coeff_dot(O1.Qtarget()) << endl;
#endif

				res *= sqrt(Symmetry::coeff_dot(O1.Qtarget())*Symmetry::coeff_dot(O2.Qtarget())); // scalar product coeff for SU(2)
			}
			return res;
		}
		else
		{
			lout << endl;
			lout << "Warning: Result of contraction in <φ|O1*O2|ψ> has " << B.dim << " blocks, returning 0!" << endl;
			lout << "MPS in question: " << Vket.info() << endl;
			lout << "MPO1 in question: " << O1.info() << endl;
			lout << "MPO2 in question: " << O2.info() << endl;
			lout << endl;
			return 0;
		}
	}
	else
	{
		Multipede<4,Symmetry,Matrix<Scalar,Dynamic,Dynamic> > B;
		Multipede<4,Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Bnext;
		
		B.setVacuum();
		for (size_t l=0; l<O2.length(); ++l)
		{
			contract_L(B, Vbra.A_at(l), O1.W_at(l), O2.W_at(l), Vket.A_at(l), O2.locBasis(l), O1.opBasis(l), O2.opBasis(l), Bnext);
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
			lout << "Warning: Result of contraction in <φ|O1*O2|ψ> has " << B.dim << " blocks, returning 0!" << endl;
			lout << "MPS in question: " << Vket.info() << endl;
			lout << "MPO1 in question: " << O1.info() << endl;
			lout << "MPO2 in question: " << O2.info() << endl;
			return 0;
		}
	}
}

/**
 * Apply an Mpo to an Mps \f$\left|\Psi_{out}\right> = H \left|\Psi_{in}\right>\f$ by using the zip-up algorithm (Stoudenmire, White 2010).
 * \param H : input Hamiltonian
 * \param Vin : input \f$\left|\Psi_{in}\right>\f$
 * \param Vout : output \f$\left|\Psi_{out}\right>\f$
 * \param VERBOSITY : verbosity level
 */
template<typename Symmetry, typename MpoScalar, typename Scalar>
void HxV (const Mpo<Symmetry,MpoScalar> &H, const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout, 
          DMRG::VERBOSITY::OPTION VERBOSITY=DMRG::VERBOSITY::HALFSWEEPWISE)
{
	Stopwatch<> Chronos;
	
	MpsCompressor<Symmetry,Scalar,MpoScalar> Compadre(VERBOSITY);
	Compadre.prodCompress(H, H, Vin, Vout, Vin.Qtarget(), Vin.calc_Dmax());
	
	if (VERBOSITY != DMRG::VERBOSITY::SILENT)
	{
		lout << Compadre.info() << endl;
		lout << Chronos.info("HxV") << endl;
		lout << "Vout: " << Vout.info() << endl << endl;
	}
}

template<typename Symmetry, typename MpoScalar, typename Scalar>
void HxV (const Mpo<Symmetry,MpoScalar> &H, Mps<Symmetry,Scalar> &Vinout, 
          DMRG::VERBOSITY::OPTION VERBOSITY=DMRG::VERBOSITY::HALFSWEEPWISE)
{
	Mps<Symmetry,Scalar> Vtmp;
	HxV(H,Vinout,Vtmp,VERBOSITY);
	Vinout = Vtmp;
}

/**
 * Performs an orthogonal polynomial iteration step \f$\left|\Psi_{out}\right> = \cdot H \left|\Psi_{in,1}\right> - 
 * B \left|\Psi_{in,2}\right>\f$ as needed in the polynomial recursion relation \f$P_n = (C_n x - A_n) P_{n-1} - B_n P_{n-2}\f$.
 * \warning The Hamiltonian is assumed to be rescaled by \p C_n and \p A_n already.
 * \param H : input Hamiltonian
 * \param Vin1 : input Mps \f$\left|T_{n-1}\right>\f$
 * \param polyB : the coefficient before the subtracted vector
 * \param Vin2 : input Mps \f$\left|T_{n-2}\right>\f$
 * \param Vout : output Mps \f$\left|T_{n}\right>\f$
 * \param VERBOSITY : verbosity level
 */
template<typename Symmetry, typename MpoScalar, typename Scalar>
void polyIter (const Mpo<Symmetry,MpoScalar> &H, const Mps<Symmetry,Scalar> &Vin1, double polyB, 
               const Mps<Symmetry,Scalar> &Vin2, Mps<Symmetry,Scalar> &Vout, 
               DMRG::VERBOSITY::OPTION VERBOSITY=DMRG::VERBOSITY::HALFSWEEPWISE)
{
	Stopwatch<> Chronos;
	MpsCompressor<Symmetry,Scalar,MpoScalar> Compadre(VERBOSITY);
	Compadre.polyCompress(H,Vin1,polyB,Vin2, Vout, Vin1.calc_Dmax());
	
	if (VERBOSITY != DMRG::VERBOSITY::SILENT)
	{
		lout << Compadre.info() << endl;
		lout << termcolor::bold << Chronos.info(make_string("polyIter B=",polyB)) << termcolor::reset << endl;
		lout << "Vout: " << Vout.info() << endl << endl;
	}
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
void addScale (const OtherScalar alpha, const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout, 
               DMRG::VERBOSITY::OPTION VERBOSITY=DMRG::VERBOSITY::SILENT)
{
	Stopwatch<> Chronos;
	MpsCompressor<Symmetry,Scalar,OtherScalar> Compadre(VERBOSITY);
	size_t Dstart = Vout.calc_Dmax();
	Mps<Symmetry,Scalar> Vtmp = Vout;
	Vtmp.addScale(alpha,Vin,false);
//	Compadre.stateCompress(Vtmp, Vout, Dstart, 1e-3, 100, 1, DMRG::COMPRESSION::RANDOM);
	Compadre.stateCompress(Vtmp, Vout, Dstart);
	
	if (VERBOSITY != DMRG::VERBOSITY::SILENT)
	{
		lout << Compadre.info() << endl;
		lout << Chronos.info("V+V") << endl;
		lout << "Vout: " << Vout.info() << endl << endl;
	}
}

template<typename Symmetry, typename MpoScalar, typename Scalar>
void OxV (const Mpo<Symmetry,MpoScalar> &O, const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout, DMRG::BROOM::OPTION TOOL=DMRG::BROOM::SVD)
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
	
	contract_C0(O.locBasis(0), O.opBasis(0), O.W_at(0), Vin.A[0], C);
	Vout.set_A_from_C(0,C,TOOL);
	
	for (size_t l=1; l<Vin.length(); ++l)
	{
		Stopwatch<> Chronos;
		contract_C(O.locBasis(l), O.opBasis(l), Vout.A[l-1], O.W_at(l), Vin.A[l], C, Cnext);
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
void OxV (const Mpo<Symmetry,MpoScalar> &O, Mps<Symmetry,Scalar> &Vinout, DMRG::BROOM::OPTION TOOL)
{
	Mps<Symmetry,Scalar> Vtmp;
	OxV(O,Vinout,Vtmp,TOOL);
	Vinout = Vtmp;
}

template<typename Symmetry, typename MpoScalar, typename Scalar>
void OxV_exact (const Mpo<Symmetry,MpoScalar> &O, const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout)
{
	size_t L = Vin.length();
	Vout = Mps<Symmetry,Scalar>(L, Vin.locBasis(), O.Qtarget(), O.volume(), Vin.calc_Nqmax());
	
	for (size_t l=0; l<L; ++l)
	{
		// cout << "l=" << l << endl;
		// auto tensorBase_l = Vin.inBasis(l).combine(O.inBasis(l));
		// cout << tensorBase_l << endl << tensorBase_l.printHistory() << endl;
		// auto tensorBase_r = Vin.outBasis(l).combine(O.outBasis(l));
		// cout << tensorBase_r << endl << tensorBase_r.printHistory() << endl;
		
		contract_AW(Vin.A_at(l), Vin.locBasis(l), O.W_at(l), O.opBasis(l),
		            Vin.inBasis(l) , O.inBasis(l) ,
		            Vin.outBasis(l), O.outBasis(l),
		            Vout.A_at(l));
	}
	
	Vout.update_inbase();
	Vout.update_outbase();
	Vout.calc_Qlimits();
	Vout.sweep(0, DMRG::BROOM::QR);
}

template<typename Symmetry, typename MpoScalar, typename Scalar>
void OxV_exact (const Mpo<Symmetry,MpoScalar> &O, Mps<Symmetry,Scalar> &Vinout)
{
	Mps<Symmetry,Scalar> Vtmp;
	OxV_exact(O,Vinout,Vtmp);
	Vinout = Vtmp;
}

#endif
