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
	                   size_t Dcutoff_input, double tol=1e-9, size_t max_halfsweeps=40, size_t min_halfsweeps=1);
	
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
		
		void stateOptimize1 (const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout, bool SWEEP=true);
		
		void stateOptimize2 (const Mps<Symmetry,Scalar> &Vin, const Mps<Symmetry,Scalar> &Vout, 
		                         PivotVector<Symmetry,Scalar> &ApairOut);
		
		void stateOptimize2 (const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout);
		
		void build_L (size_t loc, const Mps<Symmetry,Scalar> &Vbra, const Mps<Symmetry,Scalar> &Vket);
		
		void build_R (size_t loc, const Mps<Symmetry,Scalar> &Vbra, const Mps<Symmetry,Scalar> &Vket);
		
	// for |Vout> ≈ H*|Vin>
		vector<PivotMatrix1<Symmetry,Scalar,MpoScalar> > Heff;
		
		template<typename MpOperator>
		void prepSweep (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout);
		
		template<typename MpOperator>
		void prodOptimize1 (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout, bool SWEEP=true);
		
		template<typename MpOperator>
		void prodOptimize2 (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout);
		
		template<typename MpOperator>
		void prodOptimize2 (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin, const Mps<Symmetry,Scalar> &Vout, 
		                        PivotVector<Symmetry,Scalar> &ApairOut);
		
		template<typename MpOperator>
		void build_LW (size_t loc, const Mps<Symmetry,Scalar> &Vbra, const MpOperator &H, const Mps<Symmetry,Scalar> &Vket);
		
		template<typename MpOperator>
		void build_RW (size_t loc, const Mps<Symmetry,Scalar> &Vbra, const MpOperator &H, const Mps<Symmetry,Scalar> &Vket);
		
	// for |Vout> ≈ H*|Vin1> - polyB*|Vin2>
		template<typename MpOperator>
		void prepSweep (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin1, const Mps<Symmetry,Scalar> &Vin2, Mps<Symmetry,Scalar> &Vout);
		
		template<typename MpOperator>
		void polySweepStep (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin1, const Mps<Symmetry,Scalar> &Vin2, Mps<Symmetry,Scalar> &Vout);
	
	inline size_t loc1() const {return (CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? pivot : pivot-1;};
	inline size_t loc2() const {return (CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? pivot+1 : pivot;};
	
	size_t N_sites;
	size_t N_sweepsteps, N_halfsweeps;
	size_t Dcutoff, Dcutoff_new;
	size_t Mmax, Mmax_new;
	double sqdist;
	
	int pivot;
	DMRG::DIRECTION::OPTION CURRENT_DIRECTION;
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
	
	ss << "|Vlhs-Vrhs|^2=" << sqdist << ", ";
	ss << "halfsweeps=" << N_halfsweeps << ", ";
	ss << "mem=" << round(memory(GB),3) << "GB";
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

//---------------------------compression of |Psi>---------------------------
// |Vout> ≈ |Vin>, M(Vout) < M(Vin)
// convention in program: <Vout|Vin>

template<typename Symmetry, typename Scalar, typename MpoScalar>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
stateCompress (const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout, size_t Dcutoff_input, double tol, size_t max_halfsweeps, size_t min_halfsweeps)
{
	Stopwatch<> Chronos;
	N_sites = Vin.length();
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
	while ((sqdist > tol and N_halfsweeps < max_halfsweeps) or N_halfsweeps < min_halfsweeps)
	{
		Stopwatch<> Aion;
		
		// A 2-site sweep is necessary! Move pivot back to edge.
		if (N_halfsweeps%4 == 0 and N_halfsweeps > 0)
		{
			if (pivot==1)
			{
				Vout.sweep(0,DMRG::BROOM::QR);
				pivot=0;
				build_R(0,Vout,Vin);
			}
			else if (pivot==N_sites-2)
			{
				Vout.sweep(N_sites-1,DMRG::BROOM::QR);
				pivot = N_sites-1;
				build_L(N_sites-1,Vout,Vin);
			}
		}
		
		for (size_t j=1; j<=halfSweepRange; ++j)
		{
			turnaround(pivot, N_sites, CURRENT_DIRECTION);
			if (N_halfsweeps%4 == 0 and N_halfsweeps > 0)
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
		
		sqdist = abs(sqnormVin-Vout.squaredNorm());
		assert(!std::isnan(sqdist));
		
		if (CHOSEN_VERBOSITY>=2)
		{
			lout << Aion.info("half-sweep") << "\tdistance^2=" << sqdist << endl;
		}
		
		bool RESIZED = false;
		if (N_halfsweeps%4 == 0 and 
		    N_halfsweeps > 0 and 
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
	if      (pivot==1)         {Vout.sweep(0,DMRG::BROOM::QR);}
	else if (pivot==N_sites-2) {Vout.sweep(N_sites-1,DMRG::BROOM::QR);}
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
prepSweep (const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout)
{
	assert(Vout.pivot == 0 or Vout.pivot == N_sites-1 or Vout.pivot == -1);
	
	if (Vout.pivot == N_sites-1 or
	    Vout.pivot == -1)
	{
		for (size_t l=N_sites-1; l>0; --l)
		{
			Vout.setRandom(l);
			Vout.sweepStep(DMRG::DIRECTION::LEFT, l, DMRG::BROOM::QR, NULL,true);
			build_R(l-1,Vout,Vin);
		}
		CURRENT_DIRECTION = DMRG::DIRECTION::RIGHT;
	}
	else if (Vout.pivot == 0)
	{
		for (size_t l=0; l<N_sites-1; ++l)
		{
			Vout.setRandom(l);
			Vout.sweepStep(DMRG::DIRECTION::RIGHT, l, DMRG::BROOM::QR, NULL,true);
			build_L(l+1,Vout,Vin);
		}
		CURRENT_DIRECTION = DMRG::DIRECTION::LEFT;
	}
	pivot = Vout.pivot;
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
stateOptimize1 (const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout, bool SWEEP)
{
	PivotVector<Symmetry,Scalar> A(Vin.A[pivot]);
	LRxV(Env[pivot], A);
	for (size_t s=0; s<Vin.locBasis(pivot).size(); ++s)
	{
		Vout.A[pivot][s] = A.data[s].cleaned();
	}
	
	if (SWEEP)
	{
		Vout.sweepStep(CURRENT_DIRECTION, pivot, DMRG::BROOM::QR);
		pivot = Vout.get_pivot();
		(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(pivot,Vout,Vin) : build_R(pivot,Vout,Vin);
	}
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
stateOptimize2 (const Mps<Symmetry,Scalar> &Vin, const Mps<Symmetry,Scalar> &Vout, PivotVector<Symmetry,Scalar> &ApairOut)
{
	PivotVector<Symmetry,Scalar> ApairIn(Vin.A[loc1()], Vin.locBasis(loc1()), 
	                                     Vin.A[loc2()], Vin.locBasis(loc2()),
	                                     Vin.QoutTop[loc1()], Vin.QoutBot[loc1()]);
	
	ApairOut = PivotVector<Symmetry,Scalar>(Vout.A[loc1()], Vout.locBasis(loc1()), 
	                                        Vout.A[loc2()], Vout.locBasis(loc2()),
	                                        Vout.QoutTop[loc1()], Vout.QoutBot[loc1()],
	                                        true);  // dry run: do not multiply matrices, just set blocks
	
	PivotOverlap2<Symmetry,Scalar> Env2(Env[loc1()].L, Env[loc2()].R, Vin.locBasis(loc1()), Vin.locBasis(loc2()));
	LRxV(Env2, ApairIn, ApairOut);
	
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
	
	Vout.sweepStep2(CURRENT_DIRECTION, loc1(), Apair.data);
	pivot = Vout.get_pivot();
	(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(pivot,Vout,Vin) : build_R(pivot,Vout,Vin);
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
build_L (size_t loc, const Mps<Symmetry,Scalar> &Vbra, const Mps<Symmetry,Scalar> &Vket)
{
	contract_L(Env[loc-1].L, Vbra.A[loc-1], Vket.A[loc-1], Vket.locBasis(loc-1), Env[loc].L);
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
build_R (size_t loc, const Mps<Symmetry,Scalar> &Vbra, const Mps<Symmetry,Scalar> &Vket)
{
	contract_R(Env[loc+1].R, Vbra.A[loc+1], Vket.A[loc+1], Vket.locBasis(loc+1), Env[loc].R);
}

//---------------------------compression of H*|Psi>---------------------------
// |Vout> ≈ H|Vin>
// convention in program: <Vout|H|Vin>

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
prodCompress (const MpOperator &H, const MpOperator &Hdag, const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout, qarray<Symmetry::Nq> Qtot, size_t Dcutoff_input, double tol, size_t max_halfsweeps, size_t min_halfsweeps)
{
	N_sites = Vin.length();
	Stopwatch<> Chronos;
	N_halfsweeps = 0;
	N_sweepsteps = 0;
	Dcutoff = Dcutoff_new = Dcutoff_input;
	
	Vout = Mps<Symmetry,Scalar>(H, Dcutoff, Qtot, max(Vin.calc_Nqmax(), DMRG::CONTROL::DEFAULT::Qinit));
	
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
//				avgHsqVin = isReal(avg(Vin,HdagH,Vin));
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
		Stopwatch<> Aion;
		
		// A 2-site sweep is necessary! Move pivot back to edge.
		if (N_halfsweeps%4 == 0 and N_halfsweeps > 0)
		{
			if (pivot==1)
			{
				Vout.sweep(0,DMRG::BROOM::QR);
				pivot=0;
				build_RW(0,Vout,H,Vin);
			}
			else if (pivot==N_sites-2)
			{
				Vout.sweep(N_sites-1,DMRG::BROOM::QR);
				pivot = N_sites-1;
				build_LW(N_sites-1,Vout,H,Vin);
			}
		}
		
		// optimization
		for (size_t j=1; j<=halfSweepRange; ++j)
		{
			turnaround(pivot, N_sites, CURRENT_DIRECTION);
			if (N_halfsweeps%4 == 0 and N_halfsweeps > 0)
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
		
//		cout << "\tavgHsqVin=" << avgHsqVin << ", Vout.squaredNorm()=" << Vout.squaredNorm() << endl;
		sqdist = abs(avgHsqVin-Vout.squaredNorm());
		assert(!std::isnan(sqdist));
		
		if (CHOSEN_VERBOSITY>=2)
		{
			lout << Aion.info("half-sweep") << "\tdistance^2=" << sqdist << endl;
		}
		
		bool RESIZED = false;
		if (N_halfsweeps%4 == 0 and 
		    N_halfsweeps > 0 and 
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
	if      (pivot==1)         {Vout.sweep(0,DMRG::BROOM::QR);}
	else if (pivot==N_sites-2) {Vout.sweep(N_sites-1,DMRG::BROOM::QR);}
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
prepSweep (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout)
{
	assert(Vout.pivot == 0 or Vout.pivot == N_sites-1 or Vout.pivot == -1);
	
	if (Vout.pivot == N_sites-1 or Vout.pivot == -1)
	{
		for (size_t l=N_sites-1; l>0; --l)
		{
			Vout.setRandom(l);
			Stopwatch<> Chronos;
			Vout.sweepStep(DMRG::DIRECTION::LEFT, l, DMRG::BROOM::QR, NULL,true);
			build_RW(l-1,Vout,H,Vin);
		}
		CURRENT_DIRECTION = DMRG::DIRECTION::RIGHT;
	}
	else if (Vout.pivot == 0)
	{
		for (size_t l=0; l<N_sites-1; ++l)
		{
			Vout.setRandom(l);
			Stopwatch<> Chronos;
			Vout.sweepStep(DMRG::DIRECTION::RIGHT, l, DMRG::BROOM::QR, NULL,true);
			build_LW(l+1,Vout,H,Vin);
		}
		CURRENT_DIRECTION = DMRG::DIRECTION::LEFT;
	}
	pivot = Vout.pivot;
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
prodOptimize1 (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin, Mps<Symmetry,Scalar> &Vout, bool SWEEP)
{
	Stopwatch<> Chronos;
	
	precalc_blockStructure (Heff[pivot].L, Vout.A[pivot], Heff[pivot].W, Vin.A[pivot], Heff[pivot].R, 
	                        H.locBasis(pivot), H.opBasis(pivot), 
	                        Heff[pivot].qlhs, Heff[pivot].qrhs, Heff[pivot].factor_cgcs);
	
	PivotVector<Symmetry,Scalar> Ain(Vin.A[pivot]);
	PivotVector<Symmetry,Scalar> Aout(Vout.A[pivot]);
	OxV(Heff[pivot], Ain, Aout);
	Vout.A[pivot] = Aout.data;
	
	if (SWEEP)
	{
		Vout.sweepStep(CURRENT_DIRECTION, pivot, DMRG::BROOM::QR);
		pivot = Vout.get_pivot();
		(CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? build_LW(pivot,Vout,H,Vin) : build_RW(pivot,Vout,H,Vin);
	}
	
	if (CHOSEN_VERBOSITY == DMRG::VERBOSITY::STEPWISE)
	{
		lout << "optimization, loc=" << Chronos.info(pivot) << endl;
	}
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
prodOptimize2 (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin, const Mps<Symmetry,Scalar> &Vout, PivotVector<Symmetry,Scalar> &ApairOut)
{
	PivotVector<Symmetry,Scalar> ApairIn(Vin.A[loc1()], Vin.locBasis(loc1()), 
	                                     Vin.A[loc2()], Vin.locBasis(loc2()),
	                                     Vin.QoutTop[loc1()], Vin.QoutBot[loc1()]);
	
	ApairOut = PivotVector<Symmetry,Scalar>(Vout.A[loc1()], Vout.locBasis(loc1()), 
	                                        Vout.A[loc2()], Vout.locBasis(loc2()),
	                                        Vout.QoutTop[loc1()], Vout.QoutBot[loc1()], 
	                                        true); // dry run: do not multiply matrices, just set blocks
	
	PivotMatrix2<Symmetry,Scalar,MpoScalar> Heff2(Heff[loc1()].L, Heff[loc2()].R, 
	                                              H.W[loc1()], H.W[loc2()], 
	                                              H.locBasis(loc1()), H.locBasis(loc2()), 
	                                              H.opBasis (loc1()), H.opBasis (loc2()));
	
	precalc_blockStructure (Heff[loc1()].L, ApairOut.data, Heff2.W12, Heff2.W34, ApairIn.data, Heff[loc2()].R, 
	                        H.locBasis(loc1()), H.locBasis(loc2()), H.opBasis(loc1()), H.opBasis(loc2()), 
	                        Heff2.qlhs, Heff2.qrhs, Heff2.factor_cgcs);
	OxV(Heff2, ApairIn, ApairOut);
	
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
	Vout.sweepStep2(CURRENT_DIRECTION, loc1(), Apair.data);
	
	(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_LW(loc2(),Vout,H,Vin) : build_RW(loc1(),Vout,H,Vin);
	
	if (CHOSEN_VERBOSITY == DMRG::VERBOSITY::STEPWISE)
	{
		lout << "optimization & sweep step, 2-site, loc=" << Chronos.info(pivot) << endl;
	}
	
	pivot = Vout.get_pivot();
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
build_LW (size_t loc, const Mps<Symmetry,Scalar> &Vbra, const MpOperator &H, const Mps<Symmetry,Scalar> &Vket)
{
	contract_L(Heff[loc-1].L, Vbra.A[loc-1], H.W[loc-1], Vket.A[loc-1], H.locBasis(loc-1), H.opBasis(loc-1), Heff[loc].L);
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
build_RW (size_t loc, const Mps<Symmetry,Scalar> &Vbra, const MpOperator &H, const Mps<Symmetry,Scalar> &Vket)
{
	contract_R(Heff[loc+1].R, Vbra.A[loc+1], H.W[loc+1], Vket.A[loc+1], H.locBasis(loc+1), H.opBasis(loc+1), Heff[loc].R);
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
polyCompress (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin1, double polyB, const Mps<Symmetry,Scalar> &Vin2, Mps<Symmetry,Scalar> &Vout, size_t Dcutoff_input, double tol, size_t max_halfsweeps, size_t min_halfsweeps)
{
	N_sites = Vin1.length();
	Stopwatch<> Chronos;
	N_halfsweeps = 0;
	N_sweepsteps = 0;
	Dcutoff = Dcutoff_input;
	Dcutoff_new = Dcutoff_input;
	
//	Vout = Vin1;
	Vout = Mps<Symmetry,Scalar>(H, Dcutoff, Vin1.Qtarget(), max(Vin1.calc_Nqmax(), DMRG::CONTROL::DEFAULT::Qinit));
	Vout.setRandom();
	if (CHOSEN_VERBOSITY>=2)
	{
		lout << "Vin: " << Vout.info() << endl;
		Vout.graph("it");
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
	Env[N_sites-1].R.setTarget(Vin2.Qtot);
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
	
	// must achieve sqdist > tol or break off after max_halfsweeps, do at least min_halfsweeps
	while ((sqdist > tol and N_halfsweeps < max_halfsweeps) or N_halfsweeps < min_halfsweeps)
	{
		Stopwatch<> Aion;
		
		// A 2-site sweep is necessary! Move pivot back to edge.
		if (N_halfsweeps%4 == 0 and N_halfsweeps > 0)
		{
			if (pivot==1)
			{
				Vout.sweep(0,DMRG::BROOM::QR);
				pivot = 0;
				build_RW(0,Vout,H,Vin1);
				build_R (0,Vout,  Vin2);
			}
			else if (pivot==N_sites-2)
			{
				Vout.sweep(N_sites-1,DMRG::BROOM::QR);
				pivot = N_sites-1;
				build_LW(N_sites-1,Vout,H,Vin1);
				build_L (N_sites-1,Vout,  Vin2);
			}
		}
		
		for (size_t j=1; j<=halfSweepRange; ++j)
		{
			turnaround(pivot, N_sites, CURRENT_DIRECTION);
			Stopwatch<> Chronos;
			
			if (N_halfsweeps%4 == 0 and N_halfsweeps > 0)
			{
				PivotVector<Symmetry,Scalar> ApairV1;
				prodOptimize2(H,Vin1,Vout,ApairV1);
				
				PivotVector<Symmetry,Scalar> ApairV2;
				stateOptimize2(Vin2,Vout,ApairV2);
				
				for (size_t s=0; s<ApairV1.data.size(); ++s)
				{
					ApairV1.data[s].addScale(-polyB, ApairV2.data[s]);
					ApairV1.data[s] = ApairV1.data[s].cleaned();
				}
				
				Vout.sweepStep2(CURRENT_DIRECTION, loc1(), ApairV1.data);
				pivot = Vout.get_pivot();
			}
			else
			{
				stateOptimize1(Vin2,Vout,false);
				auto Atmp = Vout.A[pivot];
				
				prodOptimize1(H,Vin1,Vout,false);
				
				for (size_t s=0; s<H.locBasis(pivot).size(); ++s)
				{
					Vout.A[pivot][s].addScale(-polyB, Atmp[s]);
				}
				
				Vout.sweepStep(CURRENT_DIRECTION, pivot, DMRG::BROOM::QR);
				pivot = Vout.get_pivot();
			}
			
			polySweepStep(H,Vin1,Vin2,Vout);
			++N_sweepsteps;
		}
		halfSweepRange = N_sites-1;
		++N_halfsweeps;
		
		sqdist = abs(avgHsqV1 - Vout.squaredNorm() + polyB*polyB*sqnormV2 - 2.*polyB*overlapV12);
		assert(!std::isnan(sqdist));
		
		if (CHOSEN_VERBOSITY>=2)
		{
			lout << Aion.info("half-sweep") << "\tdistance^2=" << sqdist << endl;
		}
		
		bool RESIZED = false;
		if (N_halfsweeps%4 == 0 and 
		    N_halfsweeps > 0 and 
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
	if      (pivot==1)         {Vout.sweep(0,DMRG::BROOM::QR);}
	else if (pivot==N_sites-2) {Vout.sweep(N_sites-1,DMRG::BROOM::QR);}
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
prepSweep (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin1, const Mps<Symmetry,Scalar> &Vin2, Mps<Symmetry,Scalar> &Vout)
{
	assert(Vout.pivot == 0 or Vout.pivot == N_sites-1 or Vout.pivot == -1);
	
	if (Vout.pivot == N_sites-1)
	{
		for (size_t l=N_sites-1; l>0; --l)
		{
//			Vout.setRandom(l);
			Vout.sweepStep(DMRG::DIRECTION::LEFT, l, DMRG::BROOM::QR, NULL,true);
			#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
			#pragma omp parallel sections
			#endif
			{
				#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
				#pragma omp section
				#endif
				{
					build_RW(l-1,Vout,H,Vin1);
				}
				#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
				#pragma omp section
				#endif
				{
					build_R (l-1,Vout,Vin2);
				}
			}
		}
//		Vout.leftSweepStep(0, DMRG::BROOM::QR); // last sweep to get rid of large numbers
		CURRENT_DIRECTION = DMRG::DIRECTION::RIGHT;
	}
	else if (Vout.pivot == 0 or Vout.pivot == -1)
	{
		for (size_t l=0; l<N_sites-1; ++l)
		{
//			Vout.setRandom(l);
			Vout.sweepStep(DMRG::DIRECTION::RIGHT, l, DMRG::BROOM::QR, NULL,true);
			#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
			#pragma omp parallel sections
			#endif
			{
				#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
				#pragma omp section
				#endif
				{
					build_LW(l+1,Vout,H,Vin1);
				}
				#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
				#pragma omp section
				#endif
				{
					build_L (l+1,Vout,Vin2);
				}
			}
		}
//		Vout.rightSweepStep(N_sites-1, DMRG::BROOM::QR); // last sweep to get rid of large numbers
		CURRENT_DIRECTION = DMRG::DIRECTION::LEFT;
	}
	pivot = Vout.pivot;
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsCompressor<Symmetry,Scalar,MpoScalar>::
polySweepStep (const MpOperator &H, const Mps<Symmetry,Scalar> &Vin1, const Mps<Symmetry,Scalar> &Vin2, Mps<Symmetry,Scalar> &Vout)
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
