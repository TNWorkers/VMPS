#ifndef UMPS_COMPRESSOR_H_
#define UMPS_COMPRESSOR_H_

/// \cond
#include "termcolor.hpp" //from https://github.com/ikalnytskyi/termcolor
/// \endcond

#include "VumpsTransferMatrixAA.h"

#include "pivot/DmrgPivotOverlap1.h"

/**
 * Compressor for uMPS. Needed to obtain various operations containing uMPSs and MPOs with a variational approach.
 * \describe_Symmetry
 * \describe_Scalar
 * \describe_MpoScalar
 * \note Until now, only support for compressing a state.
 */
template<typename Symmetry, typename Scalar, typename MpoScalar=double>
class UmpsCompressor
{
public:
	
	UmpsCompressor (DMRG::VERBOSITY::OPTION VERBOSITY=DMRG::VERBOSITY::SILENT)
	:CHOSEN_VERBOSITY(VERBOSITY)
	{};

	//---info stuff---
	///\{
	/**\describe_info*/
	string info() const;
	
	// string t_info() const;
	
	/**\describe_memory*/
	double memory (MEMUNIT memunit=GB) const;
	///\}

	//---compression schemes---
	///\{
	/**
	 * Compresses a given uMps \f$V_{out} \approx V_{in}\f$. If convergence is not reached after 2 half-sweeps, 
	 * the bond dimension of \p Vout is increased and it is set to random.
	 * \param[in] Vin : input state to be compressed
	 * \param[out] Vout : compressed output state
	 * \param[in] Dinit : matrix size cutoff per site and subspace for \p Vout
	 * \param[in] Qinit : symmetry block cutoff per site for \p Vout
	 * \param[in] tol : tolerance for the variational error
	 * \param[in] max_iterations : maximal amount of iterations; break if exceeded
	 * \param[in] min_iterations : minimal amount of iterations
	 */
	void stateCompress (const Umps<Symmetry,Scalar> &Vin, Umps<Symmetry,Scalar> &Vout, 
						size_t Dinit_input, size_t Qinit_input, double tol_input, size_t max_iterations=100ul, size_t min_iterations=10ul);

private:
	
	DMRG::VERBOSITY::OPTION CHOSEN_VERBOSITY;

	// for |Vout> ≈ |Vin>
	vector<PivotOverlap1<Symmetry,Scalar> > Env;

	void optimize_parallel (const Umps<Symmetry,Scalar> &Vin, Umps<Symmetry,Scalar> &Vout);

	void build_cellEnv (const Umps<Symmetry,Scalar> &Vbra, const Umps<Symmetry,Scalar> &Vket);
	void build_LR (const Umps<Symmetry,Scalar> &Vbra, const Umps<Symmetry,Scalar> &Vket);
	void calc_error(const Umps<Symmetry,Scalar> &Vout);


	/**Safely calculates \f$l-1 mod L\f$ without overflow for \p size_t.*/
	inline size_t minus1modL (size_t l) const {return (l==0)? N_sites-1 : (l-1);}

	size_t N_sites;
	size_t N_iterations;
	size_t Dinit, Qinit;
	double err_var, tol;
	complex<double> lambdaL;
	double t_fixedL;
	double t_fixedR;
};

template<typename Symmetry, typename Scalar, typename MpoScalar>
string UmpsCompressor<Symmetry,Scalar,MpoScalar>::
info() const
{
	stringstream ss;
	ss << "UmpsCompressor: ";
	ss << "Dinit=" << Dinit << ", ";
	// ss << "Mmax=" << Mmax;
	// if (Mmax != Mmax_new)
	// {
	// 	ss << "→" << Mmax_new << ", ";
	// }
	// else
	// {
	// 	ss << " (not changed), ";
	// }
	
	ss << "|AL*C-AC/λ|^2=";
	if (err_var <= tol) {ss << termcolor::colorize << termcolor::green;}
	else               {ss << termcolor::colorize << termcolor::red;}
	ss << err_var << termcolor::reset << ", ";
	ss << "iterations=" << N_iterations << ", ";
	ss << "mem=" << round(memory(GB),3) << "GB";
	return ss.str();
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
double UmpsCompressor<Symmetry,Scalar,MpoScalar>::
memory (MEMUNIT memunit) const
{
	double res = 0.;
	for (size_t l=0; l<Env.size(); ++l)
	{
		res += Env[l].L.memory(memunit);
		res += Env[l].R.memory(memunit);
	}
	for (size_t l=0; l<Env.size(); ++l)
	{
		res += Env[l].L.memory(memunit);
		res += Env[l].R.memory(memunit);
	}
	return res;
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void UmpsCompressor<Symmetry,Scalar,MpoScalar>::
stateCompress (const Umps<Symmetry,Scalar> &Vin, Umps<Symmetry,Scalar> &Vout, 
               size_t Dinit_input, size_t Qinit_input, double tol_input, size_t max_iterations, size_t min_iterations)
{
	Dinit = Dinit_input;
	Qinit = Qinit_input;
	tol = tol_input;
	N_sites = Vin.length();
	
	//set initial state to random
	Vout = Umps<Symmetry,Scalar>(Vin.locBasis(), Vin.Qtarget(), Vin.length(), Dinit, Qinit);
	Vout.setRandom();
	cout << Vout.test_ortho() << endl;
	size_t loc=0;
	err_var = 1.;

	//set initial cell environments to random
	Env.clear();
	Env.resize(N_sites);
	Env[0].L.setRandom(Vout.inBasis(0), Vin.inBasis(0));
	Env[N_sites-1].R.setRandom(Vin.outBasis(N_sites-1), Vout.outBasis(N_sites-1));
	for (size_t l=0; l<N_sites; ++l)
	{
		Env[l].qloc = Vin.locBasis(l);
	}
	
	//apply variational optimization of the overlap
	while ((err_var > tol and N_iterations < max_iterations) or N_iterations < min_iterations or N_iterations%2 != 0)
	{
		optimize_parallel(Vin,Vout);
	}
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void UmpsCompressor<Symmetry,Scalar,MpoScalar>::
build_cellEnv (const Umps<Symmetry,Scalar> &Vin, const Umps<Symmetry,Scalar> &Vout)
{
	// Make environment for the unit cell
	build_LR(Vin, Vout);

	// Make environment for each site of the unit cell
	for (size_t l=1; l<N_sites; ++l)
	{
		contract_L(Env[l-1].L, 
		           Vout.A[GAUGE::L][l-1], Vin.A[GAUGE::L][l-1], 
		           Env[l-1].qloc, Env[l].L);
	}
	
	for (int l=N_sites-2; l>=0; --l)
	{
		contract_R(Env[l+1].R, 
		           Vout.A[GAUGE::R][l+1], Vin.A[GAUGE::R][l+1], 
		           Env[l+1].qloc, Env[l].R);
	}

}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void UmpsCompressor<Symmetry,Scalar,MpoScalar>::
build_LR (const Umps<Symmetry,Scalar> &Vin, const Umps<Symmetry,Scalar> &Vout)
{
	vector<vector<qarray<Symmetry::Nq> > > qloc_complete(N_sites);
	for (size_t loc=0; loc<N_sites; loc++) { qloc_complete[loc] = Env[loc].qloc; }
	TransferMatrixAA<Symmetry,Scalar> TL(GAUGE::R, Vout.A[GAUGE::L], Vin.A[GAUGE::L], qloc_complete);
	TransferMatrixAA<Symmetry,Scalar> TR(GAUGE::L, Vout.A[GAUGE::R], Vin.A[GAUGE::R], qloc_complete);
	Scalar eigval_dump=0.;
	Scalar eigval_used=0.;
	PivotVector<Symmetry,Scalar> xL(Env[0].L);
	PivotVector<Symmetry,Scalar> xR(Env[N_sites-1].R);
	Stopwatch<> FixedR;
	ArnoldiSolver<TransferMatrixAA<Symmetry,Scalar>,PivotVector<Symmetry,Scalar> > John(TR,xR,eigval_dump);
	t_fixedR = FixedR.time();
	Stopwatch<> FixedL;
	ArnoldiSolver<TransferMatrixAA<Symmetry,Scalar>,PivotVector<Symmetry,Scalar> > Lucy(TL,xL,eigval_used);
	t_fixedL = FixedL.time();
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "right fixed point (t=" << round(t_fixedR,2) << "): " << John.info() << endl;
		lout << "left  fixed point (t=" << round(t_fixedL,2) << "): " << Lucy.info() << endl;
	}
	// cout << "Fixed points:" << endl;
	// cout << xL.data[0].print(false) << endl;
	// cout << xR.data[0].print(false) << endl << endl;

	Env[0].L = xL.data[0];
	Env[N_sites-1].R = xR.data[0];
	lambdaL = eigval_used;
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void UmpsCompressor<Symmetry,Scalar,MpoScalar>::
optimize_parallel (const Umps<Symmetry,Scalar> &Vin, Umps<Symmetry,Scalar> &Vout)
{
	Stopwatch<> IterationTimer;
	//calculate fixed points L and R of the mixed transfer matrix (Vin.A-Vout.A)
	build_cellEnv(Vin, Vout);
	for (size_t loc=0; loc<N_sites; ++loc)
	{
		//update AC
		for(size_t s=0; s<Env[loc].qloc.size(); s++)
		{
			Vout.A[GAUGE::C][loc][s] = Env[loc].L * Vin.A[GAUGE::C][loc][s] * Env[loc].R;
		}
		//update C
		Vout.C[loc] = Env[loc].L * Vin.C[loc] * Env[loc].R;
	}
	Vout.normalize_C();
	//calc new AL and AR from AC and C
	for (size_t loc=0; loc<N_sites; ++loc)
	{
		(err_var>0.01)? Vout.svdDecompose(loc) : Vout.polarDecompose(loc);
	}
	// cout << Vout.test_ortho() << endl;
	
	calc_error(Vout);

	++N_iterations;

	// print stuff
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		size_t standard_precision = cout.precision();
		Vout.calc_entropy();
		lout << "S=" << Vout.entropy().transpose() << endl;
		lout << info() << endl;
		lout << Vout.info() << endl;
		lout << IterationTimer.info("full parallel iteration") << endl;
		lout << endl;
	}
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void UmpsCompressor<Symmetry,Scalar,MpoScalar>::
calc_error (const Umps<Symmetry,Scalar> &Vout)
{
	err_var=0.;
	for (size_t loc=0; loc<N_sites; loc++)
	for (size_t s=0; s<Env[loc].qloc.size(); s++)
	{
		err_var += (Vout.A[GAUGE::L][loc][s] * Vout.C[loc] - ( (1./lambdaL) * Vout.A[GAUGE::C][loc][s] )).norm().sum();
		// err_var += (Vout.A[GAUGE::L][loc][s] * Vout.C[loc] - Vout.A[GAUGE::C][loc][s]).norm().sum();
	}
}
#endif //UMPS_COMPRESSOR_H_
