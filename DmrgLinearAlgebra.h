#ifndef STRAWBERRY_DMRGEXTERNAL_WITH_Q
#define STRAWBERRY_DMRGEXTERNAL_WITH_Q

#ifndef OXV_EXACT_INIT_M
#define OXV_EXACT_INIT_M 100
#endif

#include "Stopwatch.h" // from HELPERS

#include "Mps.h"
#include "Mpo.h"
#include "solvers/MpsCompressor.h"
#include "ParamHandler.h"

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

/**
 * Calculates the scalar product \f$\left<\Psi_{bra}|\Psi_{ket}\right>\f$ for a heterogenic MPS.
 * \param Vbra : input \f$\left<\Psi_{bra}\right|\f$
 * \param Vket : input \f$\left|\Psi_{ket}\right>\f$
 * \param Ncellshift : Shift \p Vbra by this many unit cells (negative=left-shift, positive=right-shift).
 */
template<typename Symmetry, typename Scalar>
Scalar dot_hetero (const Mps<Symmetry,Scalar> &Vbra, const Mps<Symmetry,Scalar> &Vket, int Ncellshift=0)
{
	if (Ncellshift==0)
	{
		return Vbra.dot(Vket);
	}
	else
	{
		auto Vbral = Vbra;
		auto Vketl = Vket;
		
		if (Ncellshift < 0) // shift Vbra to the left = elongate Vbra on the right, elongate Vket on the left
		{
			Vbral.elongate_hetero(0,abs(Ncellshift));
			Vketl.elongate_hetero(abs(Ncellshift),0);
		}
		else
		{
			Vbral.elongate_hetero(abs(Ncellshift),0);
			Vketl.elongate_hetero(0,abs(Ncellshift));
		}
//		Vbral.shift_hetero(Ncellshift);
		
		return Vbral.dot(Vketl);
	}
}

/**Swaps two Mps.*/
template<typename Symmetry, typename Scalar> 
void swap (Mps<Symmetry,Scalar> &V1, Mps<Symmetry,Scalar> &V2)
{
	V1.swap(V2);
}

template<typename Symmetry, typename MpoScalar, typename Scalar>
Array<Scalar,Dynamic,1> matrix_element (int iL, 
                                        int iR,
                                        const Mps<Symmetry,Scalar> &Vbra, 
                                        const Mpo<Symmetry,MpoScalar> &O, 
                                        const Mps<Symmetry,Scalar> &Vket, 
                                        size_t power_of_O = 1)
{
	assert(iL<O.length() and iR<O.length() and iL<iR);
	
	Array<Scalar,Dynamic,1> res(Vket.Qmultitarget().size());
	
	for (size_t i=0; i<Vket.Qmultitarget().size(); ++i)
	{
		Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Bnext;
		Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > B;
		Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Id;
		
		vector<qarray3<Symmetry::Nq> > Qt;
		Qt.push_back(qarray3<Symmetry::Nq>{Vket.Qmultitarget()[i], Vbra.Qmultitarget()[i], O.Qtarget()});
		B.setTarget(Qt);
		Id.setVacuum();
		
		for (int l=iR; l>=iL; --l)
		{
			contract_R(B, Vbra.A_at(l), O.get_W_power(power_of_O)[l], Vket.A_at(l), O.locBasis(l), O.get_qOp_power(power_of_O)[l], Bnext);
			B.clear();
			B = Bnext;
			Bnext.clear();
//			cout << "l=" << l << ", i=" << i << ", B.dim=" << B.dim << endl;
		}
		
//		if (B.dim == 1)
//		{
//			res[i] = B.block[0][0][0].trace();
//		}
//		else
//		{
//			res[i] = 0;
//		}
		res[i] = contract_LR(Id,B);
	}
	
//	Vbra.graph("Vbra");
//	Vket.graph("Vket");
//	cout << "dot_green=" << res.transpose() << endl;
	
	return res;
}

template<typename Symmetry, typename Scalar>
Array<Scalar,Dynamic,1> dot_green (const Mps<Symmetry,Scalar> &V1, const Mps<Symmetry,Scalar>&V2)
{
	assert(V1.length() == V2.length() and V1.locBasis() == V2.locBasis());
	assert(V1.Qmultitarget().size() == V2.Qmultitarget().size());
	
	return matrix_element(0, V1.length()-1, V1, Mpo<Symmetry,double>::Identity(V1.locBasis()), V2);
}

//template<typename Symmetry, typename Scalar>
//complex<double> dot (const Mps<Symmetry,Scalar> &V1, const Mps<Symmetry,Scalar>&V2)
//{
//	assert(V1.length() == V2.length() and V1.locBasis() == V2.locBasis());
//	assert(V1.Qmultitarget().size() == 2 and V2.Qmultitarget().size() == 2);
//	
//	VectorXd res = matrix_element(0, V1.length()-1, V1, Mpo<Symmetry,double>::Identity(V1.locBasis()), V2);
//	
//	return res(0) + 1.i * res(1);
//}

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
            size_t power_of_O = 1ul,  
            DMRG::DIRECTION::OPTION DIR = DMRG::DIRECTION::RIGHT,
            DMRG::VERBOSITY::OPTION CHOSEN_VERBOSITY = DMRG::VERBOSITY::SILENT)
{
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Bnext;
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > B;
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Id;
	
	Scalar out=0.;
	// Note DMRG::DIRECTION::RIGHT now adapted for infinite boundary conditions
	if (DIR == DMRG::DIRECTION::RIGHT)
	{
		vector<qarray3<Symmetry::Nq> > Qt;
		for (size_t i=0; i<Vket.Qmultitarget().size(); ++i)
		{
			Qt.push_back(qarray3<Symmetry::Nq>{Vket.Qmultitarget()[i], Vbra.Qmultitarget()[i], O.Qtarget()});
		}
		Id.setTarget(Qt);
		
		B.setVacuum();
		for (size_t l=0; l<O.length(); ++l)
		{
			Stopwatch<> Timer;
			contract_L(B, Vbra.A_at(l), O.get_W_power(power_of_O)[l], Vket.A_at(l), O.locBasis(l), O.get_qOp_power(power_of_O)[l], Bnext);
			B.clear();
			B = Bnext;
			Bnext.clear();
			if (CHOSEN_VERBOSITY>=2)
			{
				lout << "avg <Ψ|O|Ψ> L→R, l=" << l << ", mem=" << round(B.memory(GB),3) << "GB, " << Timer.info("time") << endl;
			}
		}
		// cout << B.print(true) << endl;
//		return B.block[0][0][0](0,0);
		out = contract_LR(B,Id);
	}
	else
	{
//		B.setTarget(qarray3<Symmetry::Nq>{Vket.Qtarget(), Vbra.Qtarget(), O.Qtarget()});
		vector<qarray3<Symmetry::Nq> > Qt;
		for (size_t i=0; i<Vket.Qmultitarget().size(); ++i)
		{
			Qt.push_back(qarray3<Symmetry::Nq>{Vket.Qmultitarget()[i], Vbra.Qmultitarget()[i], O.Qtarget()});
		}
		B.setTarget(Qt);
		Id.setVacuum();
//		B.setIdentity(1,1,Vket.outBasis(Vket.length()-1));
		
//		for (int l=O.length()-1; l>=0; --l)
		for (size_t l=O.length()-1; l!=-1; --l)
		{
			Stopwatch<> Timer;
			contract_R(B, Vbra.A_at(l), O.get_W_power(power_of_O)[l], Vket.A_at(l), O.locBasis(l), O.get_qOp_power(power_of_O)[l], Bnext);
			B.clear();
			B = Bnext;
			Bnext.clear();
			if (CHOSEN_VERBOSITY>=2)
			{
				lout << "avg <Ψ|O|Ψ> R→L, l=" << l << ", mem=" << round(B.memory(GB),3) << "GB, " << Timer.info("time")  << endl;
			}
		}
		out = contract_LR(Id,B);
	}
	return out;
	
// 	if (B.dim == 1)
// 	{
// 		return B.block[0][0][0].trace();
// 	}
// 	else
// 	{
// /*		lout << "Warning: Result of contraction in <φ|O|ψ> has several blocks, returning 0!" << endl;*/
// /*		lout << "MPS in question: " << Vket.info() << endl;*/
// /*		lout << "MPO in question: " << O.info() << endl;*/
// /*		lout << "dim=" << B.dim << endl;*/
// /*		lout << "B=" << B.print(true) << endl;*/
// 		return 0;
// 	}
	
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

template<typename Symmetry, typename MpoScalar, typename Scalar>
Scalar avg_hetero (const Mps<Symmetry,Scalar> &Vbra, 
                   const Mpo<Symmetry,MpoScalar> &O, 
                   const Mps<Symmetry,Scalar> &Vket, 
                   bool USE_BOUNDARY = false, 
                   size_t usePower=1ul,
                   const qarray<Symmetry::Nq> &Qmid = Symmetry::qvacuum())
{
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Bnext;
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > B;
	
	if (USE_BOUNDARY)
	{
		B = Vket.get_boundaryTensor(DMRG::DIRECTION::LEFT, usePower);
		assert(O.Qtarget() == Symmetry::qvacuum() and "Can only do avg_hetero with vacuum targets. Try OxV_exact followed by dot instead.");
	}
	else
	{
		//B.setIdentity(O.auxrows(0), 1, Vket.inBasis(0));
		B.setIdentity(O.auxBasis(0).M(), 1, Vket.inBasis(0));
	}
	
	for (size_t l=0; l<O.length(); ++l)
	{
//		if (USE_SQUARE == true)
//		{
//			contract_L(B, Vbra.A_at(l), O.Wsq_at(l), Vket.A_at(l), O.locBasis(l), O.opBasisSq(l), Bnext);
//		}
//		else
//		{
//			contract_L(B, Vbra.A_at(l), O.W_at(l), Vket.A_at(l), O.locBasis(l), O.opBasis(l), Bnext);
//		}
		
		contract_L(B, Vbra.A_at(l), O.get_W_power(usePower)[l], Vket.A_at(l), O.locBasis(l), O.get_qOp_power(usePower)[l], Bnext);
		
		B.clear();
		B = Bnext;
		Bnext.clear();
		
//		cout << "l=" << l << ", B.dim=" << B.dim << endl;
//		if (l==0)
//		{
//			cout << "B=" << endl << B.print(true) << endl << endl;
//			
//			Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > R;
//			R.setIdentity(1,1,Vket.outBasis(0));
//			cout << "R.setIdentity(1,1,Vket.outBasis(0))=" << endl << R.print(true) << endl << endl;
//		}
	}
	
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > BR;
	if (USE_BOUNDARY)
	{
		BR = Vket.get_boundaryTensor(DMRG::DIRECTION::RIGHT, usePower);
	}
	else
	{
		//BR.setIdentity(O.auxcols(O.length()-1), 1, Vket.outBasis((O.length()-1)));
		BR.setIdentity(O.auxBasis(O.length()).M(), 1, Vket.outBasis((O.length()-1)), Qmid);
	}
	
//	cout << "B=" << endl;
//	cout << B.print() << endl;
//	cout << "BR=" << endl;
//	cout << BR.print() << endl;
	
	return contract_LR(B,BR);
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
            typename Symmetry::qType Qtarget = Symmetry::qvacuum(),
            size_t usePower1=1,
            size_t usePower2=1,
            bool WARN=true,
            DMRG::VERBOSITY::OPTION CHOSEN_VERBOSITY = DMRG::VERBOSITY::SILENT)
{
	if constexpr (Symmetry::NON_ABELIAN)
	{
		Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Bnext;
		Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > B;
		Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Id;
		
		vector<qarray3<Symmetry::Nq> > Qt;
		for (size_t i=0; i<Vket.Qmultitarget().size(); ++i)
		{
			Qt.push_back(qarray3<Symmetry::Nq>{Vket.Qmultitarget()[i], Vbra.Qmultitarget()[i], Qtarget});
		}
		B.setTarget(Qt);
		Id.setVacuum();
		
		for (size_t l=O1.length()-1; l!=-1; --l)
		{
			Stopwatch<> Timer;
			contract_R(B, 
			           Vbra.A_at(l), O1.get_W_power(usePower1)[l], O2.get_W_power(usePower2)[l], Vket.A_at(l), 
			           O1.locBasis(l), O1.get_qOp_power(usePower1)[l], O2.get_qOp_power(usePower2)[l],
			           Bnext);
			B.clear();
			B = Bnext;
			Bnext.clear();
			if (CHOSEN_VERBOSITY>=2)
			{
				lout << "avg <Ψ|O·O|Ψ> L→R, l=" << l << ", mem=" << round(B.memory(GB),3) << "GB, " << Timer.info("time") << endl;
			}
		}
		return contract_LR(Id,B);
		
		if (B.dim == 1)
		{
			return B.block[0][0][0].trace();
		}
		else
		{
			if (WARN)
			{
				lout << endl;
				lout << "Warning: Result of contraction in <φ|O1·O2|ψ> has " << B.dim << " blocks, returning 0!" << endl;
				lout << "MPS in question: " << Vket.info() << endl;
				lout << "MPO1 in question: " << O1.info() << endl;
				lout << "MPO2 in question: " << O2.info() << endl;
				lout << endl;
			}
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
			if (WARN)
			{
				lout << "Warning: Result of contraction in <φ|O1·O2|ψ> has " << B.dim << " blocks, returning 0!" << endl;
				lout << "MPS in question: " << Vket.info() << endl;
				lout << "MPO1 in question: " << O1.info() << endl;
				lout << "MPO2 in question: " << O2.info() << endl;
			}
			return 0;
		}
	}
}

template<typename Symmetry, typename MpoScalar, typename Scalar>
Scalar avg (const Mps<Symmetry,Scalar> &Vbra, 
            const vector<Mpo<Symmetry,MpoScalar>> &O, 
            const Mps<Symmetry,Scalar> &Vket, 
            size_t usePower = 1ul,  
            DMRG::DIRECTION::OPTION DIR = DMRG::DIRECTION::LEFT,
            DMRG::VERBOSITY::OPTION CHOSEN_VERBOSITY = DMRG::VERBOSITY::SILENT)
{
	Scalar out = 0;
	for (int i=0; i<O.size(); ++i)
	{
		out += avg(Vbra, O[i], Vket, usePower, DIR, CHOSEN_VERBOSITY);
	}
	return out;
}

template<typename Symmetry, typename MpoScalar, typename Scalar>
Scalar avg (const Mps<Symmetry,Scalar> &Vbra, 
            const vector<Mpo<Symmetry,MpoScalar>> &O1,
            const vector<Mpo<Symmetry,MpoScalar>> &O2,
            const Mps<Symmetry,Scalar> &Vket, 
            typename Symmetry::qType Qtarget = Symmetry::qvacuum(),
            size_t usePower1 = 1ul,
            size_t usePower2 = 1ul,
            bool WARN=true,
            DMRG::VERBOSITY::OPTION CHOSEN_VERBOSITY = DMRG::VERBOSITY::SILENT)
{
	Scalar out = 0;
	for (int i=0; i<O1.size(); ++i)
	for (int j=0; j<O2.size(); ++j)
	{
		out += avg(Vbra, O1[i], O2[j], Vket, Qtarget, usePower1, usePower2, WARN, CHOSEN_VERBOSITY);
	}
	return out;
}

/**
 * Apply an Mpo to an Mps \f$\left|\Psi_{out}\right> = H \left|\Psi_{in}\right>\f$ by using the zip-up algorithm (Stoudenmire, White 2010).
 * \param H : input Hamiltonian
 * \param Vin : input \f$\left|\Psi_{in}\right>\f$
 * \param Vout : output \f$\left|\Psi_{out}\right>\f$
 * \param VERBOSE : print info if \p true
 */
template<typename Symmetry, typename MpoScalar, typename Scalar>
void HxV (const Mpo<Symmetry,MpoScalar> &H, const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout, 
          bool VERBOSE=true, double tol_compr=1e-4, int Mincr=100, int Mlimit=10000, int max_halfsweeps=100)
{
	Stopwatch<> Chronos;
	
	if (Vin.calc_Mmax() <= 4)
	{
		OxV_exact(H, Vin, Vout, 2., (VERBOSE)?DMRG::VERBOSITY::HALFSWEEPWISE:DMRG::VERBOSITY::SILENT, 200, 1, -1, DMRG::BROOM::QR);
	}
	else
	{
		MpsCompressor<Symmetry,Scalar,MpoScalar> Compadre((VERBOSE)?
	                                                  DMRG::VERBOSITY::HALFSWEEPWISE
	                                                  :DMRG::VERBOSITY::SILENT);
		Compadre.prodCompress(H, H, Vin, Vout, Vin.Qtarget(), Vin.calc_Mmax(), Mincr, Mlimit, tol_compr, max_halfsweeps);
		//Compadre.prodCompress(H, H, Vin, Vout, Vin.Qtarget(), Vin.calc_Mmax(), Mincr, Mlimit, tol_compr, max_halfsweeps, 1, 0, "backup", false);
	}
	
////double tol_compr = (Vin.calc_Nqavg() <= 4.)? 1.:1e-7;
//	double tol_compr = 1e-7;
//	OxV_exact(H, Vin, Vout, tol_compr, (VERBOSE)?DMRG::VERBOSITY::HALFSWEEPWISE:DMRG::VERBOSITY::SILENT);
	
	if (VERBOSE)
	{
//		lout << Compadre.info() << endl;
		lout << Chronos.info("HxV") << endl;
		lout << "Vout: " << Vout.info() << endl << endl;
	}
}

template<typename Symmetry, typename MpoScalar, typename Scalar>
void HxV (const Mpo<Symmetry,MpoScalar> &H, Mps<Symmetry,Scalar> &Vinout, bool VERBOSE = true, double tol_compr=1e-4, int Mincr=100, int Mlimit=10000, int max_halfsweeps=100)
{
	Mps<Symmetry,Scalar> Vtmp;
	HxV(H,Vinout,Vtmp,VERBOSE,tol_compr,Mincr,Mlimit,max_halfsweeps);
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
 * \param VERBOSE : print info if \p true
 */
template<typename Symmetry, typename MpoScalar, typename Scalar>
void polyIter (const Mpo<Symmetry,MpoScalar> &H, const Mps<Symmetry,Scalar> &Vin1, double polyB, 
               const Mps<Symmetry,Scalar> &Vin2, Mps<Symmetry,Scalar> &Vout, 
               bool VERBOSE = true)
{
	Stopwatch<> Chronos;
	
	MpsCompressor<Symmetry,Scalar,MpoScalar> Compadre((VERBOSE)?
	                                                  DMRG::VERBOSITY::HALFSWEEPWISE
	                                                  :DMRG::VERBOSITY::SILENT);
	Compadre.polyCompress(H,Vin1,polyB,Vin2, Vout, Vin1.calc_Mmax());
	
	if (VERBOSE)
	{
		lout << Compadre.info() << endl;
		lout << termcolor::bold << Chronos.info(make_string("polyIter B=",polyB)) << termcolor::reset << endl;
		lout << "Vout: " << Vout.info() << endl;
	}
}

template<typename Symmetry, typename Scalar, typename OtherScalar>
void addScale (const OtherScalar alpha, const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout, 
               DMRG::VERBOSITY::OPTION VERBOSITY=DMRG::VERBOSITY::SILENT)
{
	Stopwatch<> Chronos;
	MpsCompressor<Symmetry,Scalar,OtherScalar> Compadre(VERBOSITY);
	size_t Mstart = Vout.calc_Mmax();
	vector<Mps<Symmetry,Scalar> > V(2);
	vector<double> c(2);
	V[0] = Vout;
	V[1] = Vin;
	c[0] = 1.;
	c[1] = alpha;
	Compadre.lincomboCompress(V, c, Vout, Vout.calc_Mmax());
	
	if (VERBOSITY != DMRG::VERBOSITY::SILENT)
	{
		lout << Compadre.info() << endl;
		lout << Chronos.info("V+V") << endl;
		lout << "Vout: " << Vout.info() << endl << endl;
	}
}

template<typename Symmetry, typename MpoScalar, typename Scalar>
void OxV (const Mpo<Symmetry,MpoScalar> &H, const Mpo<Symmetry,MpoScalar> &Hdag, const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout, bool VERBOSE=true, double tol_compr=1e-4, int Mincr=100, int Mlimit=10000, int max_halfsweeps=100, int min_halfsweeps=1, bool MEASURE_DISTANCE=true)
{
	Stopwatch<> Chronos;
	
	if (Vin.calc_Mmax() <= 4)
	{
		OxV_exact(H, Vin, Vout, 2., (VERBOSE)?DMRG::VERBOSITY::HALFSWEEPWISE:DMRG::VERBOSITY::SILENT, 200, 1, -1, DMRG::BROOM::QR);
	}
	else
	{
		MpsCompressor<Symmetry,Scalar,MpoScalar> Compadre((VERBOSE)?
	                                                  DMRG::VERBOSITY::HALFSWEEPWISE
	                                                  :DMRG::VERBOSITY::SILENT);
		Compadre.prodCompress(H, Hdag, Vin, Vout, Vin.Qtarget(), Vin.calc_Mmax(), Mincr, Mlimit, tol_compr, max_halfsweeps, min_halfsweeps, 0, "backup", MEASURE_DISTANCE);
	}
	
////double tol_compr = (Vin.calc_Nqavg() <= 4.)? 1.:1e-7;
//	double tol_compr = 1e-7;
//	OxV_exact(H, Vin, Vout, tol_compr, (VERBOSE)?DMRG::VERBOSITY::HALFSWEEPWISE:DMRG::VERBOSITY::SILENT);
	
	if (VERBOSE)
	{
//		lout << Compadre.info() << endl;
		lout << Chronos.info("OxV") << endl;
		lout << "Vout: " << Vout.info() << endl << endl;
	}
}

template<typename Symmetry, typename MpoScalar, typename Scalar>
void OxV (const Mpo<Symmetry,MpoScalar> &H, const Mpo<Symmetry,MpoScalar> &Hdag, Mps<Symmetry,Scalar> &Vinout, bool VERBOSE=true, double tol_compr=1e-4, int Mincr=100, int Mlimit=10000, int max_halfsweeps=100, int min_halfsweeps=1, bool MEASURE_DISTANCE=true)
{
	Mps<Symmetry,Scalar> Vtmp;
	OxV(H,Hdag,Vinout,Vtmp,VERBOSE,tol_compr,Mincr,Mlimit,max_halfsweeps,min_halfsweeps,MEASURE_DISTANCE);
	Vinout = Vtmp;
}

//template<typename Symmetry, typename MpoScalar, typename Scalar>
//void OxV (const Mpo<Symmetry,MpoScalar> &O, const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout, DMRG::BROOM::OPTION TOOL=DMRG::BROOM::SVD)
//{
//	vector<Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > C;
//	vector<Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > Cnext;
//	
//	if (TOOL == DMRG::BROOM::QR)
//	{
//		assert(O.Qtarget() == Symmetry::qvacuum() and 
//		       "Need a qnumber-conserving operator in OxV for QR option!");
//		Vout = Vin;
//	}
//	else
//	{
//		Vout.outerResize(O, Vin.Qtarget()+O.Qtarget());
//	}
//	
//	contract_C0(O.locBasis(0), O.opBasis(0), O.W_at(0), Vin.A[0], C);
//	Vout.set_A_from_C(0,C,TOOL);
//	
//	for (size_t l=1; l<Vin.length(); ++l)
//	{
//		Stopwatch<> Chronos;
//		contract_C(O.locBasis(l), O.opBasis(l), Vout.A[l-1], O.W_at(l), Vin.A[l], C, Cnext);
////		lout << "l=" << l << ", " << Chronos.info("contract_C") << endl;
//		
//		for (size_t s1=0; s1<O.locBasis(l).size(); ++s1)
//		{
//			C[s1].clear();
//			C[s1] = Cnext[s1];
//			Cnext[s1].clear();
//		}
//		
//		Vout.set_A_from_C(l,C,TOOL);
////		lout << "l=" << l << ", " << Chronos.info("set_A_from_C") << endl;
//	}
//	
//	Vout.mend();
//	Vout.pivot = Vout.length()-1;
//	
////	std::array<Tripod<Nq,Matrix<Scalar,Dynamic,Dynamic> >,D> C;
////	std::array<Tripod<Nq,Matrix<Scalar,Dynamic,Dynamic> >,D> Cnext;
////	
////	if (TOOL == DMRG::BROOM::QR)
////	{
////		assert(O.Qtarget() == qvacuum<Nq>() and 
////		       "Need a qnumber-conserving operator in OxV for QR option!");
////		Vout = Vin;
////	}
////	else
////	{
////		Vout.outerResize(O, Vin.Qtarget()+O.Qtarget());
////	}
////	
////	contract_C0(O.locBasis(0), O.W_at(0), Vin.A[0], C);
////	Vout.set_A_from_C(0,C,TOOL);
////	
////	for (size_t l=1; l<Vin.length(); ++l)
////	{
////		contract_C(O.locBasis(l), Vout.A[l-1], O.W_at(l), Vin.A[l], C, Cnext);
////		
////		for (size_t s1=0; s1<D; ++s1)
////		{
////			C[s1].clear();
////			C[s1] = Cnext[s1];
////			Cnext[s1].clear();
////		}
////		
////		Vout.set_A_from_C(l,C,TOOL);
////	}
////	
////	Vout.mend();
////	Vout.pivot = Vout.length()-1;
//}

//template<typename Symmetry, typename MpoScalar, typename Scalar>
//void OxV (const Mpo<Symmetry,MpoScalar> &O, Mps<Symmetry,Scalar> &Vinout, DMRG::BROOM::OPTION TOOL)
//{
//	Mps<Symmetry,Scalar> Vtmp;
//	OxV(O,Vinout,Vtmp,TOOL);
//	Vinout = Vtmp;
//}

/**
 * Performs an exact MPO-MPS product.
 * \param O : input MPO
 * \param Vin : input MPS
 * \param Vout : output MPS
 * \param tol_compr : if \f$tol_compr < 1\f$, compresses the result with this tolerance
 * \param VERBOSITY : verbosity level, which is also passed on to the MpsCompressor class
*/
// Note: Changes to the function parameters here affect the forward declaration in Mps.h
template<typename Symmetry, typename MpoScalar, typename Scalar>
void OxV_exact (const Mpo<Symmetry,MpoScalar> &O, const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout, 
                double tol_compr = 1e-7, DMRG::VERBOSITY::OPTION VERBOSITY = DMRG::VERBOSITY::HALFSWEEPWISE,
                int max_halfsweeps = 200, int min_halfsweeps = 1, int Minit=-1,
                DMRG::BROOM::OPTION BROOMOPTION=DMRG::BROOM::QR)
{
	size_t L = Vin.length();
	auto Qt = Symmetry::reduceSilent(Vin.Qtarget(), O.Qtarget());
	bool TRIVIAL_BOUNDARIES = false;
	if (Vin.Boundaries.IS_TRIVIAL()) {TRIVIAL_BOUNDARIES = true;}
	
//	for (int i=0; i<Qt.size(); ++i)
//	{
//		cout << "i=" << i << ", Qt[i]=" << Qt[i] << endl;
//	}
	
	Vout = Mps<Symmetry,Scalar>(L, Vin.locBasis(), Qt[Qt.size()-1], O.volume(), 100ul, TRIVIAL_BOUNDARIES);
	Vout.set_Qmultitarget(Qt);
	Vout.min_Nsv = Vin.min_Nsv;
	Vout.N_phys = Vin.N_phys;
	
	if (Vin.Boundaries.IS_TRIVIAL())
	{
		Vout.set_open_bc();
	}
	else
	{
		Vout.Boundaries = Vin.Boundaries;
	}
	
	// FORCE_QTOT to create only one final block; 
	// otherwise crashes when using the result for further calculations (e.g. ground-state sweeping).
	// Irrelevant for infinite boundary conditions.
	for (size_t l=0; l<L; ++l)
	{
		bool FORCE_QTOT = (l!=L-1 or TRIVIAL_BOUNDARIES==false)? false:true;
		contract_AW(Vin.A_at(l), Vin.locBasis(l), O.W_at(l), O.opBasis(l),
		            Vin.inBasis(l) , O.inBasis(l),
		            Vin.outBasis(l), O.outBasis(l),
		            Vout.A_at(l),
		            FORCE_QTOT, Vout.Qtarget());
	}
	
	Vout.update_inbase();
	Vout.update_outbase();
	Vout.calc_Qlimits(); // Must be called here, depends on Qtot!
	
	string input_info, exact_info, swept_info, compressed_info, Compressor_info;
	
	if (VERBOSITY > DMRG::VERBOSITY::SILENT)
	{
//		lout << endl;
//		lout << termcolor::bold << "OxV_exact" << termcolor::reset << endl;
		input_info = Vin.info();
		exact_info = Vout.info();
//		lout << "input:\t" << Vin.info() << endl;
//		lout << "exact:\t" << Vout.info() << endl;
	}
	
	if (tol_compr < 1.)
	{
		MpsCompressor<Symmetry,Scalar,MpoScalar> Compadre(VERBOSITY);
		Mps<Symmetry,Scalar> Vtmp;
		Compadre.stateCompress(Vout, Vtmp, (Minit==-1)?Vin.calc_Mmax():Minit, 100, 10000, tol_compr, max_halfsweeps, min_halfsweeps);
		Vtmp.max_Nsv = Vtmp.calc_Mmax();
		
//		lout << "Vtmp.calc_Mmax()=" << Vtmp.calc_Mmax() << endl;
		if (Vtmp.calc_Mmax() == 0)
		{
			lout << termcolor::red << "Warning: OxV compression failed, returning exact result!" << termcolor::reset << endl;
			Vout.sweep(0,BROOMOPTION);
		}
		else
		{
			Vout = Vtmp;
			// shouldn't matter, but just to be sure:
			Vout.update_inbase();
			Vout.update_outbase();
			Vout.calc_Qlimits(); // Careful: Depends on Qtot!
			
			if (VERBOSITY > DMRG::VERBOSITY::SILENT)
			{
				compressed_info = Vout.info();
				Compressor_info = Compadre.info();
	//			lout << "compressed:\t" << Vout.info() << endl;
	//			lout << "\t" << Compadre.info() << endl;
			}
		}
	}
	else
	{
//		cout << Vout.info() << endl;
//		cout << "dot=" << dot(Vout,Vout) << endl;
		Vout.sweep(0,BROOMOPTION);
		
		if (VERBOSITY > DMRG::VERBOSITY::SILENT)
		{
//			lout << "swept:\t" << Vout.info() << endl;
			swept_info = Vout.info();
		}
	}
	
	if (VERBOSITY > DMRG::VERBOSITY::SILENT)
	{
		#pragma omp critical
		{
			lout << endl;
			lout << termcolor::bold << "OxV_exact" << termcolor::reset << endl;
			lout << "input:\t" << input_info << endl;
			lout << "oper.:\t" << O.info() << endl;
			lout << "exact:\t" << exact_info << endl;
			if (tol_compr < 1.)
			{
				lout << "compr.:\t" << compressed_info << endl;
				lout << "\t" << Compressor_info << endl;
			}
			else
			{
				lout << "swept:\t" << swept_info << endl;
			}
			lout << endl;
		}
	}
	
	if (Vout.calc_Nqavg() <= 1.5 and Vout.min_Nsv == 0 and 
	    Symmetry::IS_TRIVIAL == false and 
	    Vout.Boundaries.IS_TRIVIAL() == true)
	{
		Vout.min_Nsv = 1;
		lout << termcolor::blue << "Warning: Setting min_Nsv=1 to deal with small Hilbert space after OxV_exact!" << termcolor::reset << endl;
	}
}

template<typename Symmetry, typename MpoScalar, typename Scalar>
void OxV_exact (const Mpo<Symmetry,MpoScalar> &O, Mps<Symmetry,Scalar> &Vinout, 
                double tol_compr = 1e-7, DMRG::VERBOSITY::OPTION VERBOSITY = DMRG::VERBOSITY::HALFSWEEPWISE,
                int max_halfsweeps = 200, int min_halfsweeps = 1, int Minit=-1,
                DMRG::BROOM::OPTION BROOMOPTION=DMRG::BROOM::QR)
{
	Mps<Symmetry,Scalar> Vtmp;
	OxV_exact(O,Vinout,Vtmp,tol_compr,VERBOSITY,max_halfsweeps,min_halfsweeps,Minit,BROOMOPTION);
	Vinout = Vtmp;
}

template<typename Hamiltonian, typename Scalar>
Hamiltonian sum (const Hamiltonian &H1, const Hamiltonian &H2, DMRG::VERBOSITY::OPTION VERBOSITY = DMRG::VERBOSITY::SILENT)
{
	MpoTerms<typename Hamiltonian::Symmetry,Scalar> Terms1 = H1;
	Terms1.set_verbosity(VERBOSITY);
	MpoTerms<typename Hamiltonian::Symmetry,Scalar> Terms2 = H2;
	Terms2.set_verbosity(VERBOSITY);
	MpoTerms<typename Hamiltonian::Symmetry,Scalar> Sum_asTerms = Hamiltonian::sum(Terms1,Terms2);
	Mpo<typename Hamiltonian::Symmetry,Scalar> Sum_asMpo(Sum_asTerms);
	vector<Param> params;
	Hamiltonian Hres(Sum_asMpo,params);
	return Hres;
}

template<typename Hamiltonian, typename Scalar>
Hamiltonian prod (const Hamiltonian &H1, const Hamiltonian &H2, const qarray<Hamiltonian::Symmetry::Nq> &Qtot=Hamiltonian::Symmetry::qvacuum(), DMRG::VERBOSITY::OPTION VERBOSITY = DMRG::VERBOSITY::SILENT)
{
	MpoTerms<typename Hamiltonian::Symmetry,Scalar> Terms1 = H1;
	Terms1.set_verbosity(VERBOSITY);
	MpoTerms<typename Hamiltonian::Symmetry,Scalar> Terms2 = H2;
	Terms2.set_verbosity(VERBOSITY);
	MpoTerms<typename Hamiltonian::Symmetry,Scalar> Prod_asTerms = Hamiltonian::prod(Terms1,Terms2,Qtot);
	Mpo<typename Hamiltonian::Symmetry,Scalar> Prod_asMpo(Prod_asTerms);
	vector<Param> params; // needs a better solution: params contains info on Ly and states projected out of the basis
	Hamiltonian Hres(Prod_asMpo,params);
	return Hres;
}

template<typename Symmetry, typename Scalar=double>
Mpo<Symmetry,Scalar> sum (const Mpo<Symmetry,Scalar> &H1, const Mpo<Symmetry,Scalar> &H2, DMRG::VERBOSITY::OPTION VERBOSITY = DMRG::VERBOSITY::SILENT)
{
	Mpo<Symmetry,Scalar> res;
	if      ( H1.HAS_W() and !H2.HAS_W()) res = H1;
	else if (!H1.HAS_W() and  H2.HAS_W()) res = H2;
	else
	{
		MpoTerms<Symmetry,Scalar> Terms1 = H1;
		Terms1.set_verbosity(VERBOSITY);
		Terms1.calc(1);
		MpoTerms<Symmetry,Scalar> Terms2 = H2;
		Terms2.set_verbosity(VERBOSITY);
		Terms2.calc(1);
		MpoTerms<Symmetry,Scalar> Sum_asTerms = MpoTerms<Symmetry,Scalar>::sum(Terms1,Terms2);
		res = Mpo<Symmetry,Scalar>(Sum_asTerms);
	}
	return res;
}

template<typename Symmetry, typename Scalar=double>
Mpo<Symmetry,Scalar> diff (const Mpo<Symmetry,Scalar> &H1, const Mpo<Symmetry,Scalar> &H2, DMRG::VERBOSITY::OPTION VERBOSITY = DMRG::VERBOSITY::SILENT)
{
	
	Mpo<Symmetry,Scalar> res;
	if      ( H1.HAS_W() and !H2.HAS_W()) res = H1;
	else if (!H1.HAS_W() and  H2.HAS_W())
	{
		MpoTerms<Symmetry,Scalar> minusH2 = H2;
		minusH2.scale(-1.);
		res = Mpo<Symmetry,Scalar>(minusH2);
	}
	else
	{
		MpoTerms<Symmetry,Scalar> Terms1 = H1;
		Terms1.set_verbosity(VERBOSITY);
		Terms1.calc(1);
		auto minusH2 = H2;
		minusH2.scale(-1.);
		MpoTerms<Symmetry,Scalar> Terms2 = minusH2;
		Terms2.set_verbosity(VERBOSITY);
		Terms2.calc(1);
		MpoTerms<Symmetry,Scalar> Diff_asTerms = MpoTerms<Symmetry,Scalar>::sum(Terms1,Terms2);
		res = Mpo<Symmetry,Scalar>(Diff_asTerms);
	}
	return res;
}

template<typename Symmetry, typename Scalar=double>
Mpo<Symmetry,Scalar> prod (const Mpo<Symmetry,Scalar> &H1, const Mpo<Symmetry,Scalar> &H2, const qarray<Symmetry::Nq> &Qtot=Symmetry::qvacuum(), DMRG::VERBOSITY::OPTION VERBOSITY = DMRG::VERBOSITY::SILENT)
{
	MpoTerms<Symmetry,Scalar> Terms1 = H1;
	Terms1.set_verbosity(VERBOSITY);
	MpoTerms<Symmetry,Scalar> Terms2 = H2;
	Terms2.set_verbosity(VERBOSITY);
	MpoTerms<Symmetry,Scalar> Prod_asTerms = MpoTerms<Symmetry,Scalar>::prod(Terms1,Terms2,Qtot);
	Mpo<Symmetry,Scalar> Prod_asMpo(Prod_asTerms);
	return Prod_asMpo;
}

template<typename Symmetry, typename MpsScalar=double, typename MpoScalar=double>
void OvecxV(const std::vector<Mpo<Symmetry,MpoScalar>>& Os, Mps<Symmetry,MpsScalar>& v_in, std::size_t Minit_in=0ul, std::size_t Mincr=100ul, std::size_t Mlimit=10000ul, double tol_compr=1e-5, std::size_t max_halfsweeps=24ul, std::size_t min_halfsweeps=1ul)
{
    assert(Os.size() > 0);
    Mps<Symmetry,MpsScalar> v_other = v_in;
    std::size_t Minit = Minit_in;
    if(Minit_in == 0ul)
    {
        Minit = v_in.calc_Mmax();
    }
    assert(Os[0].IS_UNITARY() and "Unitarity is needed for this method");
    MpsCompressor<Symmetry,MpsScalar,MpoScalar> Compadre(DMRG::VERBOSITY::HALFSWEEPWISE);
    Compadre.prodCompress(Os[0], Os[0], v_in, v_other, Symmetry::qvacuum(), Minit, Mincr, Mlimit, tol_compr, max_halfsweeps, min_halfsweeps);
    for (std::size_t k=1; k<Os.size(); ++k)
    {
        Mps<Symmetry,MpsScalar> &v_old = (k % 2 == 1 ? v_other : v_in);
        Mps<Symmetry,MpsScalar> &v_new = (k % 2 == 1 ? v_in : v_other);
        
        if(Minit_in == 0ul)
        {
            Minit = v_old.calc_Mmax();
        }
        assert(Os[k].IS_UNITARY() and "Unitarity is needed for this method");
        MpsCompressor<Symmetry,MpsScalar,MpoScalar> Compadre(DMRG::VERBOSITY::HALFSWEEPWISE);
        Compadre.prodCompress(Os[k], Os[k], v_old, v_new, Symmetry::qvacuum(), Minit, Mincr, Mlimit, tol_compr, max_halfsweeps, min_halfsweeps);
    }
    if(Os.size() % 2 == 1)
    {
        v_in = v_other;
    }
}

#endif
