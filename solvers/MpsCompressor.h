#ifndef STRAWBERRY_MPSCOMPRESSOR_WITH_Q
#define STRAWBERRY_MPSCOMPRESSOR_WITH_Q

#ifndef DMRG_POLYCOMPRESS_TOL
#define DMRG_POLYCOMPRESS_TOL 1e-4
#endif

#ifndef DMRG_POLYCOMPRESS_MIN
#define DMRG_POLYCOMPRESS_MIN 1
#endif

#ifndef DMRG_POLYCOMPRESS_MAX
#define DMRG_POLYCOMPRESS_MAX 16
#endif

#include "termcolor.hpp" //from https://github.com/ikalnytskyi/termcolor

#include "tensors/Biped.h"
#include "tensors/Multipede.h"
#include "LanczosSolver.h" // from ALGS
#include "tensors/DmrgContractions.h"
#include "pivot/DmrgPivotMatrix1.h"
#include "pivot/DmrgPivotMatrix2.h"
#include "Stopwatch.h" // from TOOLS
#include "pivot/DmrgPivotOverlap1.h"
#include "pivot/DmrgPivotOverlap2.h"

/**
 * Compressor for MPS. Needed to obtain various operations containing MPSs and MPOs with a variational approach.
 * \describe_Symmetry
 * \describe_Scalar
 * \describe_MpoScalar
 */
template<typename Symmetry, typename Scalar, typename MpoScalar=double>
class MpsCompressor
{
	typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;

public:
	
	MpsCompressor (DMRG::VERBOSITY::OPTION VERBOSITY=DMRG::VERBOSITY::SILENT)
	:CHOSEN_VERBOSITY(VERBOSITY)
	{};
	
	//---info stuff---
	///\{
	/**\describe_info*/
	string info() const;
	
	string t_info() const;
	
	/**\describe_memory*/
	double memory (MEMUNIT memunit=GB) const;
	///\}
	
	//---compression schemes---
	///\{
	/**
	 * Compresses a given Mps \f$V_{out} \approx V_{in}\f$. If convergence is not reached after 2 half-sweeps, 
	 * the bond dimension of \p Vout is increased and it is set to random.
	 * \param[in] Vin : input state to be compressed
	 * \param[out] Vout : compressed output state
	 * \param[in] Dcutoff_input : matrix size cutoff per site and subspace for \p Vout
	 * \param[in] tol : tolerance for the square norm of the difference: \f$\left|V_{out}-V_{in}\right|^2<tol\f$
	 * \param[in] max_halfsweeps : maximal amount of half-sweeps; break if exceeded
	 * \param[in] min_halfsweeps : minimal amount of half-sweeps
	 */
	void stateCompress (const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout, 
	                    size_t Dcutoff_input, double tol=1e-5, size_t max_halfsweeps=40, size_t min_halfsweeps=1);
	
	/**
	 * Compresses a matrix-vector product \f$\left|V_{out}\right> \approx H \left|V_{in}\right>\f$. 
	 * Needs to calculate \f$\left<V_{in}\right|H^{\dagger}H\left|V_{in}\right>\f$. 
	 * Works optimally with OpenMP and (at least) 2 threads. If convergence is not reached after 2 half-sweeps, 
	 * the bond dimension of \p Vout is increased and it is set to random.
	 * \param[in] H : Operator
	 * \param[in] Hdag : Adjoint operator
	 * \param[in] Vin : input state
	 * \param[out] Vout : compressed output state
	 * \param[in] Qtot_input : Resulting quantum number for Vout
	 * \param[in] Dcutoff_input : matrix size cutoff per site and subspace for \p Vout, good guess: Vin.calc_Dmax()
	 * \param[in] tol : tolerance for the square norm of the difference: \f$\left|V_{out} - H \cdot V_{in}\right|^2<tol\f$
	 * \param[in] max_halfsweeps : maximal amount of half-sweeps
	 * \param[in] min_halfsweeps : minimal amount of half-sweeps
	 */
	template<typename MpOperator>
	void prodCompress (const MpOperator &H, const MpOperator &Hdag, const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout, 
	                   qarray<Symmetry::Nq> Qtot_input,
	                   size_t Dcutoff_input, double tol=1e-8, size_t max_halfsweeps=42, size_t min_halfsweeps=1);
	
	/**
	 * Compresses an orthogonal iteration step \f$V_{out} \approx (C_n H - A_n) \cdot V_{in1} - B_n V_{in2}\f$. 
	 * Needs to calculate \f$\left<V_{in1}\right|H^2\left|V_{in1}\right>\f$, \f$\left<V_{in2}\right|H\left|V_{in1}\right>\f$ and \f$\big<V_{in2}\big|V_{in2}\big>\f$. 
	 * Works optimally with OpenMP and (at least) 3 threads, as the last overlap is cheap to do in the mixed-canonical representation. 
	 * If convergence is not reached after 4 half-sweeps, the bond dimension of \p Vout is increased and it is set to random.
	 * \warning The Hamiltonian has to be rescaled by \p C_n and \p A_n already.
	 * \param[in] H : Hamiltonian (an Mps with Mpo::Qtarget() = Symmetry::qvacuum()) rescaled by 2
	 * \param[in] Vin1 : input state to be multiplied
	 * \param[in] polyB : the coefficient before the subtracted vector
	 * \param[in] Vin2 : input state to be subtracted
	 * \param[out] Vout : compressed output state
	 * \param[in] Dcutoff_input : matrix size cutoff per site and subspace for \p Vout
	 * \param[in] tol : tolerance for the square norm of the difference: \f$\left|V_{out} - 2H \cdot V_{in1} - V_{in2}\right|^2<tol\f$
	 * 	                \warning Too small a value for \p tol will lead to bad convergence. Try something of the order of 1e-3 to 1e-4.
	 * \param[in] max_halfsweeps : maximal amount of half-sweeps
	 * \param[in] min_halfsweeps : minimal amount of half-sweeps
	 */
	template<typename MpOperator>
	void polyCompress (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin1, double polyB, const Mps<Symmetry,Scalar> &Vin2, Mps<Symmetry,Scalar> &Vout, 
	                   size_t Dcutoff_input, double tol=DMRG_POLYCOMPRESS_TOL, 
	                   size_t max_halfsweeps=DMRG_POLYCOMPRESS_MAX, size_t min_halfsweeps=DMRG_POLYCOMPRESS_MIN);
	///\}
	
private:
	
	DMRG::VERBOSITY::OPTION CHOSEN_VERBOSITY;
	
	// for |Vout> ≈ |Vin>
		vector<PivotOverlap1<Symmetry,Scalar> > Env;
		
		void prepSweep (const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout);
		
		void stateOptimize1 (const Mps<Symmetry,Scalar> &Vin, const Mps<Symmetry,Scalar> &Vout, PivotVector<Symmetry,Scalar> &Aout);
		
		void stateOptimize1 (const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout);
		
		void stateOptimize2 (const Mps<Symmetry,Scalar> &Vin, const Mps<Symmetry,Scalar> &Vout, 
		                         PivotVector<Symmetry,Scalar> &ApairOut);
		
		void stateOptimize2 (const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout);
		
		void build_L (size_t loc, const Mps<Symmetry,Scalar> &Vbra, const Mps<Symmetry,Scalar> &Vket, bool RANDOMIZE=false);
		
		void build_R (size_t loc, const Mps<Symmetry,Scalar> &Vbra, const Mps<Symmetry,Scalar> &Vket, bool RANDOMIZE=false);
		
	// for |Vout> ≈ H*|Vin>
		vector<PivotMatrix1<Symmetry,Scalar,MpoScalar> > Heff;
		
		template<typename MpOperator>
		void prepSweep (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout);
		
		template<typename MpOperator>
		void prodOptimize1 (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin, const Mps<Symmetry,Scalar> &Vout, 
		                    PivotVector<Symmetry,Scalar> &Aout);
		
		template<typename MpOperator>
		void prodOptimize1 (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout);
		
		template<typename MpOperator>
		void prodOptimize2 (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout);
		
		template<typename MpOperator>
		void prodOptimize2 (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin, const Mps<Symmetry,Scalar> &Vout, 
		                        PivotVector<Symmetry,Scalar> &ApairOut);
		
		template<typename MpOperator>
		void build_LW (size_t loc, const Mps<Symmetry,Scalar> &Vbra, const MpOperator &H, const Mps<Symmetry,Scalar> &Vket, bool RANDOMIZE=false);
		
		template<typename MpOperator>
		void build_RW (size_t loc, const Mps<Symmetry,Scalar> &Vbra, const MpOperator &H, const Mps<Symmetry,Scalar> &Vket, bool RANDOMIZE=false);
		
	// for |Vout> ≈ H*|Vin1> - polyB*|Vin2>
		template<typename MpOperator>
		void prepSweep (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin1, const Mps<Symmetry,Scalar> &Vin2, Mps<Symmetry,Scalar> &Vout, 
		                bool RANDOMIZE = false);
		
		template<typename MpOperator>
		void build_LRW (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin1, const Mps<Symmetry,Scalar> &Vin2, Mps<Symmetry,Scalar> &Vout);
	
	inline size_t loc1() const {return (CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? pivot : pivot-1;};
	inline size_t loc2() const {return (CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? pivot+1 : pivot;};
	
	void sweep_to_edge (const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout, bool BUILD_LR);
	
	template<typename MpOperator>
	void sweep_to_edge (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin1, const Mps<Symmetry,Scalar> &Vin2, 
	                    Mps<Symmetry,Scalar> &Vout, bool BUILD_LR, bool BUILD_LWRW);
	
	size_t N_sites;
	size_t N_sweepsteps, N_halfsweeps;
	size_t Dcutoff, Dcutoff_new;
	size_t Mmax, Mmax_new;
	double sqdist, tol;
	
	int pivot;
	DMRG::DIRECTION::OPTION CURRENT_DIRECTION;
	
	double t_opt = 0; // optimization
	double t_AA = 0; // contract_AA
	double t_sweep = 0; // sweepStep, sweepStep2
	double t_LR = 0; // build L, R, LW, RW
	double t_ohead = 0; // precalc_blockStructure
	double t_tot = 0; // full time step
};

template<typename Symmetry, typename Scalar, typename MpoScalar>
string MpsCompressor<Symmetry,Scalar,MpoScalar>::
info() const
{
	stringstream ss;
	ss << "MpsCompressor: ";
	ss << "Dcutoff=" << Dcutoff;
	if (Dcutoff != Dcutoff_new)
	{
		ss << "→" << Dcutoff_new << ", ";
	}
	else
	{
		ss << " (not resized), ";
	}
	ss << "Mmax=" << Mmax;
	if (Mmax != Mmax_new)
	{
		ss << "→" << Mmax_new << ", ";
	}
	else
	{
		ss << " (not changed), ";
	}
	
	ss << "|Vlhs-Vrhs|^2=";
	if (sqdist <= tol) {ss << termcolor::green;}
	else               {ss << termcolor::red;}
	ss << termcolor::reset << sqdist << ", ";
	ss << "halfsweeps=" << N_halfsweeps << ", ";
	ss << "mem=" << round(memory(GB),3) << "GB";
	return ss.str();
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
string MpsCompressor<Symmetry,Scalar,MpoScalar>::
t_info() const
{
	stringstream ss;
	ss << "t[s]=" << termcolor::bold << t_tot << termcolor::reset
	   << ", opt=" << round(t_opt/t_tot*100.,0) << "%"
	   << ", sweep=" << round(t_sweep/t_tot*100.) << "%"
	   << ", LR="    << round(t_LR/t_tot*100.) << "%";
	if (t_ohead > 0.)
	{
		ss << ", ohead=" << round(t_ohead/t_tot*100.) << "%";
	}
	if (t_AA > 0.)
	{
		ss << ", AA="    << round(t_AA/t_tot*100.) << "%";
	}
	return ss.str();
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
double MpsCompressor<Symmetry,Scalar,MpoScalar>::
memory (MEMUNIT memunit) const
{
	double res = 0.;
	for (size_t l=0; l<Env.size(); ++l)
	{
		res += Env[l].L.memory(memunit);
		res += Env[l].R.memory(memunit);
	}
	for (size_t l=0; l<Heff.size(); ++l)
	{
		res += Heff[l].L.memory(memunit);
		res += Heff[l].R.memory(memunit);
	}
	return res;
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
sweep_to_edge (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin1, const Mps<Symmetry,Scalar> &Vin2, 
               Mps<Symmetry,Scalar> &Vout, bool BUILD_LR, bool BUILD_LWRW)
{
	assert(pivot == 0 or pivot==1 or pivot==N_sites-2 or pivot==N_sites-1);
	
	if (pivot==1)
	{
		Vout.sweep(0,DMRG::BROOM::QR);
		pivot = 0;
		if (BUILD_LWRW)
		{
			build_RW(0,Vout,H,Vin1);
		}
		if (BUILD_LR)
		{
			build_R (0,Vout,  Vin2);
		}
	}
	else if (pivot==N_sites-2)
	{
		Vout.sweep(N_sites-1,DMRG::BROOM::QR);
		pivot = N_sites-1;
		if (BUILD_LWRW)
		{
			build_LW(N_sites-1,Vout,H,Vin1);
		}
		if (BUILD_LR)
		{
			build_L (N_sites-1,Vout,  Vin2);
		}
	}
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
sweep_to_edge (const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout, bool BUILD_LR)
{
	assert(pivot == 0 or pivot==1 or pivot==N_sites-2 or pivot==N_sites-1);
	
	if (pivot==1)
	{
		Vout.sweep(0,DMRG::BROOM::QR);
		pivot = 0;
		if (BUILD_LR)
		{
			build_R(0,Vout,Vin);
		}
	}
	else if (pivot==N_sites-2)
	{
		Vout.sweep(N_sites-1,DMRG::BROOM::QR);
		pivot = N_sites-1;
		if (BUILD_LR)
		{
			build_L(N_sites-1,Vout,Vin);
		}
	}
}

//---------------------------compression of |Psi>---------------------------
// |Vout> ≈ |Vin>, M(Vout) < M(Vin)
// convention in program: <Vout|Vin>

template<typename Symmetry, typename Scalar, typename MpoScalar>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
stateCompress (const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout, 
               size_t Dcutoff_input, double tol_input, size_t max_halfsweeps, size_t min_halfsweeps)
{
	Stopwatch<> Chronos;
	N_sites = Vin.length();
	tol = tol_input;
	double sqnormVin = isReal(dot(Vin,Vin));
	N_halfsweeps = 0;
	N_sweepsteps = 0;
	Dcutoff = Dcutoff_new = Dcutoff_input;
	
	// set L&R edges
	Env.clear();
	Env.resize(N_sites);
	Env[N_sites-1].R.setTarget(Vin.Qtot);
	Env[0].L.setVacuum();
	for (size_t l=0; l<N_sites; ++l)
	{
		Env[l].qloc = Vin.locBasis(l);
	}
	
	// set initial guess
	Vout = Vin;
	Vout.innerResize(Dcutoff);
	Vout.max_Nsv = Dcutoff;
	
	Mmax = Vout.calc_Mmax();
	prepSweep(Vin,Vout);
	sqdist = 1.;
	size_t halfSweepRange = N_sites;
	
	if (CHOSEN_VERBOSITY>=2)
	{
		lout << Chronos.info("preparation") << endl;
	}
	
	// must achieve sqdist > tol or break off after max_halfsweeps, do at least min_halfsweeps
	while ((sqdist > tol and N_halfsweeps < max_halfsweeps) or N_halfsweeps < min_halfsweeps or N_halfsweeps%2 != 0)
	{
		t_opt = 0;
		t_AA = 0;
		t_sweep = 0;
		t_LR = 0;
		t_ohead = 0;
		t_tot = 0;
		Stopwatch<> FullSweepTimer;
		
		// A 2-site sweep is necessary! Move pivot back to edge.
		if (N_halfsweeps%4 == 0 and N_halfsweeps > 1)
		{
			sweep_to_edge(Vin,Vout,true); // BUILD_LR = true
		}
		
		for (size_t j=1; j<=halfSweepRange; ++j)
		{
			turnaround(pivot, N_sites, CURRENT_DIRECTION);
			if (N_halfsweeps%4 == 0 and N_halfsweeps > 1)
			{
				stateOptimize2(Vin,Vout);
			}
			else
			{
				stateOptimize1(Vin,Vout);
			}
			++N_sweepsteps;
		}
		halfSweepRange = N_sites-1;
		++N_halfsweeps;
		
//		cout << "sqnormVin=" << sqnormVin << ", Vout.squaredNorm()=" << Vout.squaredNorm() << endl;
		sqdist = abs(sqnormVin-Vout.squaredNorm());
		assert(!std::isnan(sqdist));
		
		if (CHOSEN_VERBOSITY>=2)
		{
			lout << " distance^2=";
			if (sqdist <= tol) {lout << termcolor::green;}
			else               {lout << termcolor::red;}
			lout << sqdist << termcolor::reset << ", ";
			t_tot = FullSweepTimer.time();
			lout << t_info() << endl;
		}
		
		if (N_halfsweeps%4 == 0 and 
		    N_halfsweeps > 1 and 
		    N_halfsweeps != max_halfsweeps and 
		    sqdist > tol)
		{
			Vout.max_Nsv += 1;
			Dcutoff_new = Vout.max_Nsv;
			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
			{
				lout << "resize: " << Vout.max_Nsv-1 << "→" << Vout.max_Nsv << endl;
			}
		}
		
		Mmax_new = Vout.calc_Mmax();
	}
	
	// last sweep
	sweep_to_edge(Vin,Vout,false);
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
prepSweep (const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout)
{
	assert(Vout.pivot == 0 or Vout.pivot == N_sites-1 or Vout.pivot == -1);
	Vout.setRandom();
	
	if (Vout.pivot == N_sites-1 or
	    Vout.pivot == -1)
	{
		for (size_t l=N_sites-1; l>0; --l)
		{
			Vout.sweepStep(DMRG::DIRECTION::LEFT, l, DMRG::BROOM::QR, NULL,true);
			build_R(l-1,Vout,Vin,true); //true randomize Env[l].R
		}
		CURRENT_DIRECTION = DMRG::DIRECTION::RIGHT;
	}
	else if (Vout.pivot == 0)
	{
		for (size_t l=0; l<N_sites-1; ++l)
		{
			Vout.sweepStep(DMRG::DIRECTION::RIGHT, l, DMRG::BROOM::QR, NULL,true);
			build_L(l+1,Vout,Vin,true); //true randomize Env[l].L
		}
		CURRENT_DIRECTION = DMRG::DIRECTION::LEFT;
	}
	pivot = Vout.pivot;
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
stateOptimize1 (const Mps<Symmetry,Scalar> &Vin, const Mps<Symmetry,Scalar> &Vout, PivotVector<Symmetry,Scalar> &Aout)
{
	PivotVector<Symmetry,Scalar> Ain(Vin.A[pivot]);
	Stopwatch<> OptTimer;
	LRxV(Env[pivot], Ain, Aout);
	t_opt += OptTimer.time();
	
	for (size_t s=0; s<Aout.size(); ++s)
	{
		Aout[s] = Aout[s].cleaned();
	}
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
stateOptimize1 (const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout)
{
	PivotVector<Symmetry,Scalar> Aout;
	stateOptimize1(Vin,Vout,Aout);
	Vout.A[pivot] = Aout.data;
	// safeguard against sudden norm loss:
	if (Vout.squaredNorm() < 1e-7)
	{
		if (CHOSEN_VERBOSITY > 0)
		{
			lout << termcolor::bold << termcolor::red << "WARNING: small norm encountered at pivot=" << pivot << "!" << termcolor::reset << endl;
		}
		Vout /= sqrt(Vout.squaredNorm());
	}
	
	Stopwatch<> SweepTimer;
	Vout.sweepStep(CURRENT_DIRECTION, pivot, DMRG::BROOM::SVD);
	t_sweep += SweepTimer.time();
	pivot = Vout.get_pivot();
	(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(pivot,Vout,Vin) : build_R(pivot,Vout,Vin);
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
stateOptimize2 (const Mps<Symmetry,Scalar> &Vin, const Mps<Symmetry,Scalar> &Vout, PivotVector<Symmetry,Scalar> &ApairOut)
{
	Stopwatch<> AAtimer;
	PivotVector<Symmetry,Scalar> ApairIn(Vin.A[loc1()], Vin.locBasis(loc1()), 
	                                     Vin.A[loc2()], Vin.locBasis(loc2()),
	                                     Vin.QoutTop[loc1()], Vin.QoutBot[loc1()]);
	
	ApairOut = PivotVector<Symmetry,Scalar>(Vout.A[loc1()], Vout.locBasis(loc1()), 
	                                        Vout.A[loc2()], Vout.locBasis(loc2()),
	                                        Vout.QoutTop[loc1()], Vout.QoutBot[loc1()],
	                                        true);  // dry run: do not multiply matrices, just set blocks
	t_AA += AAtimer.time();
	
	Stopwatch<> OptTimer;
	PivotOverlap2<Symmetry,Scalar> Env2(Env[loc1()].L, Env[loc2()].R, Vin.locBasis(loc1()), Vin.locBasis(loc2()));
	LRxV(Env2, ApairIn, ApairOut);
	t_opt += OptTimer.time();
	
	for (size_t s=0; s<ApairOut.data.size(); ++s)
	{
		ApairOut.data[s] = ApairOut.data[s].cleaned();
	}
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
stateOptimize2 (const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout)
{
	PivotVector<Symmetry,Scalar> Apair;
	stateOptimize2(Vin,Vout,Apair);
	
	Stopwatch<> SweepTimer;
	Vout.sweepStep2(CURRENT_DIRECTION, loc1(), Apair.data);
	t_sweep += SweepTimer.time();
	pivot = Vout.get_pivot();
	(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(pivot,Vout,Vin) : build_R(pivot,Vout,Vin);
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
build_L (size_t loc, const Mps<Symmetry,Scalar> &Vbra, const Mps<Symmetry,Scalar> &Vket, bool RANDOMIZE)
{
	Stopwatch<> LRtimer;
	contract_L(Env[loc-1].L, Vbra.A[loc-1], Vket.A[loc-1], Vket.locBasis(loc-1), Env[loc].L, RANDOMIZE);
	t_LR += LRtimer.time();
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
build_R (size_t loc, const Mps<Symmetry,Scalar> &Vbra, const Mps<Symmetry,Scalar> &Vket, bool RANDOMIZE)
{
	Stopwatch<> LRtimer;
	contract_R(Env[loc+1].R, Vbra.A[loc+1], Vket.A[loc+1], Vket.locBasis(loc+1), Env[loc].R, RANDOMIZE);
	t_LR += LRtimer.time();
}

//---------------------------compression of H*|Psi>---------------------------
// |Vout> ≈ H|Vin>
// convention in program: <Vout|H|Vin>

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
prodCompress (const MpOperator &H, const MpOperator &Hdag, const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout, 
              qarray<Symmetry::Nq> Qtot, size_t Dcutoff_input, double tol_input, size_t max_halfsweeps, size_t min_halfsweeps)
{
//	assert(H.HAS_TWO_SITE_DATA() and "You need to call H.precalc_TwoSiteData() before prodCompress!");
	N_sites = Vin.length();
	Stopwatch<> Chronos;
	tol = tol_input;
	N_halfsweeps = 0;
	N_sweepsteps = 0;
	Dcutoff = Dcutoff_new = Dcutoff_input;
	
	if (H.Qtarget() == Symmetry::qvacuum())
	{
		Vout = Vin;
	}
	else
	{
		Vout = Mps<Symmetry,Scalar>(H, Dcutoff, Qtot, max(Vin.calc_Nqmax(), DMRG::CONTROL::DEFAULT::Qinit));
	}
	
	// prepare edges of LW & RW
	Heff.clear();
	Heff.resize(N_sites);
	Heff[0].L.setVacuum();
	Heff[N_sites-1].R.setTarget(qarray3<Symmetry::Nq>{Vin.Qtarget(), Vout.Qtarget(), H.Qtarget()});
	for (size_t l=0; l<N_sites; ++l)
	{
		Heff[l].W = H.W[l];
	}
	
	Vout.max_Nsv = Dcutoff;
	Vout.min_Nsv = Vin.min_Nsv;
	Mmax = Vout.calc_Mmax();
	double avgHsqVin;
	
	#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
	#pragma omp parallel sections
	#endif
	{
		#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
		#pragma omp section
		#endif
		{
			if (H.IS_UNITARY())
			{
				avgHsqVin = Vin.squaredNorm();
			}
			else
			{
				if (H.IS_HERMITIAN())
				{
					avgHsqVin = (H.check_SQUARE()==true)? isReal(avg(Vin,H,Vin,true)) : isReal(avg(Vin,H,H,Vin));
				}
				else
				{
					avgHsqVin = isReal(avg(Vin,Hdag,H,Vin));
				}
			}
		}
		#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
		#pragma omp section
		#endif
		{
			prepSweep(H,Vin,Vout);
		}
	}
	sqdist = 1.;
	size_t halfSweepRange = N_sites;
	
	if (CHOSEN_VERBOSITY>=2)
	{
		lout << Chronos.info("preparation") << endl;
	}
	
	// must achieve sqdist > tol or break off after max_halfsweeps, do at least min_halfsweeps
	while ((sqdist > tol and N_halfsweeps < max_halfsweeps) or N_halfsweeps < min_halfsweeps)
	{
		t_opt = 0;
		t_AA = 0;
		t_sweep = 0;
		t_LR = 0;
		t_ohead = 0;
		t_tot = 0;
		Stopwatch<> FullSweepTimer;
		
		// A 2-site sweep is necessary! Move pivot back to edge.
		if (N_halfsweeps%4 == 0 and N_halfsweeps > 1)
		{
			sweep_to_edge(H,Vin,Vin,Vout,false,true); // build_LWRW = true
		}
		
		// optimization
		for (size_t j=1; j<=halfSweepRange; ++j)
		{
			turnaround(pivot, N_sites, CURRENT_DIRECTION);
			if (N_halfsweeps%4 == 0 and N_halfsweeps > 1)
			{
				prodOptimize2(H,Vin,Vout);
			}
			else
			{
				prodOptimize1(H,Vin,Vout);
			}
			++N_sweepsteps;
		}
		halfSweepRange = N_sites-1;
		++N_halfsweeps;
		
		Scalar factor_cgc = 1.;//pow(Symmetry::degeneracy(H.Qtarget()),1);
		cout << "avgHsqVin=" << avgHsqVin << endl;
		cout << "Vout.squaredNorm()=" << Vout.squaredNorm() << endl;
		sqdist = abs(avgHsqVin - factor_cgc * factor_cgc * Vout.squaredNorm());
		assert(!std::isnan(sqdist));
		
		if (CHOSEN_VERBOSITY>=2)
		{
			lout << " distance^2=";
			if (sqdist <= tol) {lout << termcolor::green;}
			else               {lout << termcolor::red;}
			lout << sqdist << termcolor::reset << ", ";
			t_tot = FullSweepTimer.time();
			lout << t_info() << endl;
		}
		
		bool RESIZED = false;
		if (N_halfsweeps%4 == 0 and 
		    N_halfsweeps > 1 and 
		    N_halfsweeps != max_halfsweeps and 
		    sqdist > tol)
		{
			Vout.max_Nsv += 1;
			Dcutoff_new = Vout.max_Nsv;
			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
			{
				lout << "resize: " << Vout.max_Nsv-1 << "→" << Vout.max_Nsv << endl;
			}
		}
		
		Mmax_new = Vout.calc_Mmax();
	}
	
	// move pivot to edge at the end
//	if      (pivot==1)         {Vout.sweep(0,DMRG::BROOM::QR);}
//	else if (pivot==N_sites-2) {Vout.sweep(N_sites-1,DMRG::BROOM::QR);}
	Scalar factor_cgc = 1.;//Symmetry::degeneracy(H.Qtarget());
	Vout *= factor_cgc;
	sweep_to_edge(H,Vin,Vin,Vout,false,false);
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
prepSweep (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout)
{
	assert(Vout.pivot == 0 or Vout.pivot == N_sites-1 or Vout.pivot == -1);
	Vout.setRandom();
	
	if (Vout.pivot == N_sites-1 or Vout.pivot == -1)
	{
		for (size_t l=N_sites-1; l>0; --l)
		{
			Vout.sweepStep(DMRG::DIRECTION::LEFT, l, DMRG::BROOM::QR, NULL,true);
			build_RW(l-1,Vout,H,Vin,true);
		}
		CURRENT_DIRECTION = DMRG::DIRECTION::RIGHT;
	}
	else if (Vout.pivot == 0)
	{
		for (size_t l=0; l<N_sites-1; ++l)
		{
			Vout.sweepStep(DMRG::DIRECTION::RIGHT, l, DMRG::BROOM::QR, NULL,true);
			build_LW(l+1,Vout,H,Vin,true);
		}
		CURRENT_DIRECTION = DMRG::DIRECTION::LEFT;
	}
	
	pivot = Vout.pivot;
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
prodOptimize1 (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin, const Mps<Symmetry,Scalar> &Vout, PivotVector<Symmetry,Scalar> &Aout)
{
	Stopwatch<> OheadTimer;
	precalc_blockStructure (Heff[pivot].L, Vout.A[pivot], Heff[pivot].W, Vin.A[pivot], Heff[pivot].R, 
	                        H.locBasis(pivot), H.opBasis(pivot), 
	                        Heff[pivot].qlhs, Heff[pivot].qrhs, Heff[pivot].factor_cgcs);
	t_ohead += OheadTimer.time();
	
	PivotVector<Symmetry,Scalar> Ain(Vin.A[pivot]);
	Aout = PivotVector<Symmetry,Scalar>(Vout.A[pivot]);
	Stopwatch<> OptTimer;
	OxV(Heff[pivot], Ain, Aout);
	t_opt += OptTimer.time();
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
prodOptimize1 (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout)
{
	Stopwatch<> Chronos;
	
	PivotVector<Symmetry,Scalar> Aout;
	prodOptimize1(H,Vin,Vout,Aout);
	Vout.A[pivot] = Aout.data;
	
	// safeguard against sudden norm loss:
	if (Vout.squaredNorm() < 1e-7)
	{
		if (CHOSEN_VERBOSITY > 0)
		{
			lout << termcolor::bold << termcolor::red << "WARNING: small norm encountered at pivot=" << pivot << "!" << termcolor::reset << endl;
		}
		Vout /= sqrt(Vout.squaredNorm());
	}
	
	Stopwatch<> SweepTimer;
	Vout.sweepStep(CURRENT_DIRECTION, pivot, DMRG::BROOM::SVD);
	t_sweep += SweepTimer.time();
	pivot = Vout.get_pivot();
	(CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? build_LW(pivot,Vout,H,Vin) : build_RW(pivot,Vout,H,Vin);
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
prodOptimize2 (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin, const Mps<Symmetry,Scalar> &Vout, PivotVector<Symmetry,Scalar> &ApairOut)
{
	Stopwatch<> AAtimer;
	PivotVector<Symmetry,Scalar> ApairIn(Vin.A[loc1()], Vin.locBasis(loc1()), 
	                                     Vin.A[loc2()], Vin.locBasis(loc2()),
	                                     Vin.QoutTop[loc1()], Vin.QoutBot[loc1()]);
	
	ApairOut = PivotVector<Symmetry,Scalar>(Vout.A[loc1()], Vout.locBasis(loc1()), 
	                                        Vout.A[loc2()], Vout.locBasis(loc2()),
	                                        Vout.QoutTop[loc1()], Vout.QoutBot[loc1()], 
	                                        true); // dry run: do not multiply matrices, just set blocks
	t_AA += AAtimer.time();
	
	PivotMatrix2<Symmetry,Scalar,MpoScalar> Heff2(Heff[loc1()].L, Heff[loc2()].R, 
	                                              H.W[loc1()], H.W[loc2()], 
	                                              H.locBasis(loc1()), H.locBasis(loc2()), 
	                                              H.opBasis (loc1()), H.opBasis (loc2()));
	
	Stopwatch<> OheadTimer;
	precalc_blockStructure (Heff[loc1()].L, ApairOut.data, Heff2.W12, Heff2.W34, ApairIn.data, Heff[loc2()].R, 
	                        H.locBasis(loc1()), H.locBasis(loc2()), H.opBasis(loc1()), H.opBasis(loc2()), 
	                        Heff2.qlhs, Heff2.qrhs, Heff2.factor_cgcs);
	cout << "loc1()=" << loc1() << endl;
	cout << "Heff2.qlhs.size()=" << Heff2.qlhs.size() << endl;
	cout << "Heff2.qrhs.size()=" << Heff2.qrhs.size() << endl;
	t_ohead += OheadTimer.time();
	Stopwatch<> OptTimer;
	OxV(Heff2, ApairIn, ApairOut);
	t_opt += OptTimer.time();
	
	for (size_t s=0; s<ApairOut.data.size(); ++s)
	{
		ApairOut.data[s] = ApairOut.data[s].cleaned();
	}
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
prodOptimize2 (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout)
{
	Stopwatch<> Chronos;
	
	PivotVector<Symmetry,Scalar> Apair;
	prodOptimize2(H,Vin,Vout,Apair);
	Stopwatch<> SweepTimer;
	Vout.sweepStep2(CURRENT_DIRECTION, loc1(), Apair.data);
	t_sweep += SweepTimer.time();
	pivot = Vout.get_pivot();
	
	(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_LW(pivot,Vout,H,Vin) : build_RW(pivot,Vout,H,Vin);
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
build_LW (size_t loc, const Mps<Symmetry,Scalar> &Vbra, const MpOperator &H, const Mps<Symmetry,Scalar> &Vket, bool RANDOMIZE)
{
	Stopwatch<> LRtimer;
	contract_L(Heff[loc-1].L, Vbra.A[loc-1], H.W[loc-1], Vket.A[loc-1], H.locBasis(loc-1), H.opBasis(loc-1), Heff[loc].L, RANDOMIZE);
	t_LR += LRtimer.time();
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
build_RW (size_t loc, const Mps<Symmetry,Scalar> &Vbra, const MpOperator &H, const Mps<Symmetry,Scalar> &Vket, bool RANDOMIZE)
{
	Stopwatch<> LRtimer;
	contract_R(Heff[loc+1].R, Vbra.A[loc+1], H.W[loc+1], Vket.A[loc+1], H.locBasis(loc+1), H.opBasis(loc+1), Heff[loc].R, RANDOMIZE);
	t_LR += LRtimer.time();
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
polyCompress (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin1, double polyB, const Mps<Symmetry,Scalar> &Vin2, Mps<Symmetry,Scalar> &Vout, 
              size_t Dcutoff_input, double tol_input, size_t max_halfsweeps, size_t min_halfsweeps)
{
//	assert(H.HAS_TWO_SITE_DATA() and "You need to call H.precalc_TwoSiteData() before polyCompress!");
	N_sites = Vin1.length();
	tol = tol_input;
	Stopwatch<> Chronos;
	N_halfsweeps = 0;
	N_sweepsteps = 0;
	Dcutoff = Dcutoff_input;
	Dcutoff_new = Dcutoff_input;
	
	Vout = Vin1;
//	Vout = Mps<Symmetry,Scalar>(H, Dcutoff, Vin1.Qtarget(), max(Vin1.calc_Nqmax(), DMRG::CONTROL::DEFAULT::Qinit));
//	Vout.setRandom();
	if (CHOSEN_VERBOSITY>=2)
	{
		lout << "Vin: " << Vout.info() << endl;
	}
	
	// prepare edges of LW & RW
	Heff.clear();
	Heff.resize(N_sites);
	Heff[0].L.setVacuum();
	Heff[N_sites-1].R.setTarget(qarray3<Symmetry::Nq>{Vin1.Qtarget(), Vout.Qtarget(), Symmetry::qvacuum()});
	for (size_t l=0; l<N_sites; ++l)
	{
		Heff[l].W = H.W[l];
	}
	
	// set L&R edges
	Env.clear();
	Env.resize(N_sites);
	Env[N_sites-1].R.setTarget(Vin2.Qtarget());
	Env[0].L.setVacuum();
	for (size_t l=0; l<N_sites; ++l)
	{
		Env[l].qloc = H.locBasis(l);
	}
	
	double avgHsqV1, sqnormV2, overlapV12;
	sqnormV2 = Vin2.squaredNorm();
	
	#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
	#pragma omp parallel sections
	#endif
	{
		#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
		#pragma omp section
		#endif
		{
			avgHsqV1 = (H.check_SQUARE()==true)? isReal(avg(Vin1,H,Vin1,true)) : isReal(avg(Vin1,H,H,Vin1));
		}
		#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
		#pragma omp section
		#endif
		{
			overlapV12 = isReal(avg(Vin2,H,Vin1));
		}
		#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
		#pragma omp section
		#endif
		{
			prepSweep(H,Vin1,Vin2,Vout);
		}
	}
	sqdist = 1.;
	size_t halfSweepRange = N_sites;
	
	if (CHOSEN_VERBOSITY>=2)
	{
		lout << Chronos.info("preparation") << endl;
	}
	
	Mmax = Vout.calc_Mmax();
	Vout.max_Nsv = Dcutoff;
	Vout.min_Nsv = Vin1.min_Nsv;
	// In order to avoid block loss for small Hilbert spaces:
	if (Vout.calc_Nqmax() <= 4)
	{
		Vout.min_Nsv = 1;
	}
	
	// must achieve sqdist > tol or break off after max_halfsweeps, do at least min_halfsweeps
	while ((sqdist > tol and N_halfsweeps < max_halfsweeps) or N_halfsweeps < min_halfsweeps)
	{
		t_opt = 0;
		t_AA = 0;
		t_sweep = 0;
		t_LR = 0;
		t_ohead = 0;
		t_tot = 0;
		Stopwatch<> FullSweepTimer;
		
		// A 2-site sweep is necessary! Move pivot back to edge.
		if (N_halfsweeps%4 == 0 and N_halfsweeps > 1)
		{
			sweep_to_edge(H,Vin1,Vin2,Vout,true,true); // build_LR = true, build LWRW = true
		}
		
		for (size_t j=1; j<=halfSweepRange; ++j)
		{
			turnaround(pivot, N_sites, CURRENT_DIRECTION);
			Stopwatch<> Chronos;
			
			if (N_halfsweeps%4 == 0 and N_halfsweeps > 1)
			{
				PivotVector<Symmetry,Scalar> Apair1;
				prodOptimize2(H,Vin1,Vout,Apair1);
				
				PivotVector<Symmetry,Scalar> Apair2;
				stateOptimize2(Vin2,Vout,Apair2);
				
				for (size_t s=0; s<Apair1.size(); ++s)
				{
					Apair1[s].addScale(-polyB, Apair2[s]);
					Apair1[s] = Apair1[s].cleaned();
				}
				
				Stopwatch<> SweepTimer;
				Vout.sweepStep2(CURRENT_DIRECTION, loc1(), Apair1.data);
				t_sweep += SweepTimer.time();
				pivot = Vout.get_pivot();
			}
			else
			{
				PivotVector<Symmetry,Scalar> A1;
				prodOptimize1(H,Vin1,Vout,A1);
				
				PivotVector<Symmetry,Scalar> A2;
				stateOptimize1(Vin2,Vout,A2);
				
				for (size_t s=0; s<A1.size(); ++s)
				{
					A1[s].addScale(-polyB, A2[s]);
					A1[s] = A1[s].cleaned();
				}
				Vout.A[pivot] = A1.data;
				
				Stopwatch<> SweepTimer;
				Vout.sweepStep(CURRENT_DIRECTION, pivot, DMRG::BROOM::SVD);
				t_sweep += SweepTimer.time();
				pivot = Vout.get_pivot();
			}
			
			build_LRW(H,Vin1,Vin2,Vout);
			++N_sweepsteps;
		}
		halfSweepRange = N_sites-1;
		++N_halfsweeps;
		
		cout << "avgHsqV1=" << avgHsqV1 
		     << ", Vout.squaredNorm()=" << Vout.squaredNorm() 
		     << ", polyB*polyB*sqnormV2=" << polyB*polyB*sqnormV2 
		     << ", 2.*polyB*overlapV12=" << 2.*polyB*overlapV12 
		     << endl;
		sqdist = abs(avgHsqV1 - Vout.squaredNorm() + polyB*polyB*sqnormV2 - 2.*polyB*overlapV12);
		assert(!std::isnan(sqdist));
		
		if (CHOSEN_VERBOSITY>=2)
		{
			lout << " distance^2=";
			if (sqdist <= tol) {lout << termcolor::green;}
			else               {lout << termcolor::red;}
			lout << sqdist << termcolor::reset << ", ";
			t_tot = FullSweepTimer.time();
			lout << t_info() << endl;
		}
		
		bool RESIZED = false;
		if (N_halfsweeps%4 == 0 and 
		    N_halfsweeps > 1 and 
		    N_halfsweeps != max_halfsweeps and
		    sqdist > tol)
		{
			Vout.max_Nsv += 1;
			Dcutoff_new = Vout.max_Nsv;
			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
			{
				lout << "resize: " << Vout.max_Nsv-1 << "→" << Vout.max_Nsv << endl;
			}
		}
		
		if (N_halfsweeps%8 == 0 and
		    N_halfsweeps > 1 and 
		    N_halfsweeps != max_halfsweeps and
		    sqdist > tol)
		{
			lout << "Warning: Could not reach tolerance, restarting from random!" << endl;
			prepSweep(H,Vin1,Vin2,Vout,true);
		}
		
		Mmax_new = Vout.calc_Mmax();
	}
	
	// last sweep
	sweep_to_edge(H,Vin1,Vin2,Vout,false,false);
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
prepSweep (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin1, const Mps<Symmetry,Scalar> &Vin2, Mps<Symmetry,Scalar> &Vout, bool RANDOMIZE)
{
	assert(Vout.pivot == 0 or Vout.pivot == N_sites-1 or Vout.pivot == -1);
	if (RANDOMIZE) {Vout.setRandom();}
	
	if (Vout.pivot == N_sites-1)
	{
		for (size_t l=N_sites-1; l>0; --l)
		{
			Vout.sweepStep(DMRG::DIRECTION::LEFT, l, DMRG::BROOM::QR, NULL,RANDOMIZE);
			#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
			#pragma omp parallel sections
			#endif
			{
				#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
				#pragma omp section
				#endif
				{
					build_RW(l-1,Vout,H,Vin1,RANDOMIZE);
				}
				#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
				#pragma omp section
				#endif
				{
					build_R(l-1,Vout,Vin2,RANDOMIZE);
				}
			}
		}
		CURRENT_DIRECTION = DMRG::DIRECTION::RIGHT;
	}
	else if (Vout.pivot == 0 or Vout.pivot == -1)
	{
		for (size_t l=0; l<N_sites-1; ++l)
		{
			Vout.sweepStep(DMRG::DIRECTION::RIGHT, l, DMRG::BROOM::QR, NULL,RANDOMIZE);
			#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
			#pragma omp parallel sections
			#endif
			{
				#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
				#pragma omp section
				#endif
				{
					build_LW(l+1,Vout,H,Vin1,RANDOMIZE);
				}
				#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
				#pragma omp section
				#endif
				{
					build_L(l+1,Vout,Vin2,RANDOMIZE);
				}
			}
		}
		CURRENT_DIRECTION = DMRG::DIRECTION::LEFT;
	}
	pivot = Vout.pivot;
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
build_LRW (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin1, const Mps<Symmetry,Scalar> &Vin2, Mps<Symmetry,Scalar> &Vout)
{
	#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
	#pragma omp parallel sections
	#endif
	{
		#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
		#pragma omp section
		#endif
		{
			(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_LW(pivot,Vout,H,Vin1) : build_RW(pivot,Vout,H,Vin1);
		}
		#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
		#pragma omp section
		#endif
		{
			(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L (pivot,Vout,Vin2)   : build_R(pivot,Vout,Vin2);
		}
	}
}

#endif
