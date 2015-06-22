#ifndef STRAWBERRY_MPSCOMPRESSOR_WITH_Q
#define STRAWBERRY_MPSCOMPRESSOR_WITH_Q

#define COMPRESSOR_MAX_HALFSWEEPS 100

#include "Biped.h"
#include "Multipede.h"
#include "LanczosSolver.h" // for isReal
#include "DmrgContractionsQ.h"
#include "DmrgPivotStuffQ.h"
#include "LanczosMower.h"

/**Compressor of Matrix Product States with conserved quantum numbers.
\describe_Nq
\describe_Scalar
\describe_MpoScalar*/
template<size_t Nq, typename Scalar, typename MpoScalar=double>
class MpsQCompressor
{
typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;

public:
	
	//---constructor---
	MpsQCompressor(DMRG::VERBOSITY::OPTION VERBOSITY=DMRG::VERBOSITY::SILENT)
	:CHOSEN_VERBOSITY(VERBOSITY)
	{};
	
	//---compression schemes---
	///\{
	/**Compresses a given MpsQ \f$V_{out} \approx V_{in}\f$. If convergence is not reached after 2 half-sweeps, the bond dimension of \p Vout is increased and it is set to random.
	\param[in] Vin : input state to be compressed
	\param[out] Vout : compressed output state
	\param[in] Dcutoff_input : matrix size cutoff per site and subspace for \p Vout
	\param[in] tol : tolerance for the square norm of the difference: \f$\left|V_{out}-V_{in}\right|^2<tol\f$
	\param[in] max_halfsweeps : maximal amount of half-sweeps; break if exceeded
	\param[in] min_halfsweeps : minimal amount of half-sweeps
	\param[in] START : choice of initial guess: 
		- DMRG::COMPRESSION::RANDOM : use a random state with the cutoff given by \p Dcutoff_input
		- DMRG::COMPRESSION::RHS : makes no sense here, results in the same as above
		- DMRG::COMPRESSION::BRUTAL_SVD : cut \p Vin down to \p Dcutoff_input and use as initial guess
		- DMRG::COMPRESSION::RHS_SVD : makes no sense here, results in the same as above
	*/
	void varCompress (const MpsQ<Nq,Scalar> &Vin, MpsQ<Nq,Scalar> &Vout, 
	                  size_t Dcutoff_input, double tol=1e-6, size_t max_halfsweeps=100, size_t min_halfsweeps=1, 
	                  DMRG::COMPRESSION::INIT START = DMRG::COMPRESSION::BRUTAL_SVD);
	
	/**Compresses a matrix-vector product \f$\left|V_{out}\right> \approx H \left|V_{in}\right>\f$. Needs to calculate \f$\left<V_{in}\right|H^2\left|V_{in}\right>\f$. Works optimally with OpenMP and (at least) 2 threads. If convergence is not reached after 2 half-sweeps, the bond dimension of \p Vout is increased and it is set to random.
	\param[in] H : Hamiltonian (an MpsQ with MpoQ::Qtarget() = qvacuum<Nq>())
	\param[in] Vin : input state
	\param[out] Vout : compressed output state
	\param[in] Dcutoff_input : matrix size cutoff per site and subspace for \p Vout, good guess: Vin.calc_Dmax()
	\param[in] tol : tolerance for the square norm of the difference: \f$\left|V_{out} - H \cdot V_{in}\right|^2<tol\f$
	\param[in] max_halfsweeps : maximal amount of half-sweeps
	\param[in] min_halfsweeps : minimal amount of half-sweeps
	\param[in] START : choice of initial guess: 
		- DMRG::COMPRESSION::RANDOM : use a random state with the cutoff given by \p Dcutoff_input
		- DMRG::COMPRESSION::RHS : use \p Vin, \p Dcutoff_input is ignored
		- DMRG::COMPRESSION::BRUTAL_SVD : perform the multiplication using OxV, cutting the result according to \p Dcutoff_input
		- DMRG::COMPRESSION::RHS_SVD : perform the multiplication using OxV, cutting the result according to the subspaces of \p Vin
	*/
	template<typename MpOperator>
	void varCompress (const MpOperator &H, const MpsQ<Nq,Scalar> &Vin, MpsQ<Nq,Scalar> &Vout, 
	                  size_t Dcutoff_input, double tol=1e-6, size_t max_halfsweeps=100, size_t min_halfsweeps=1, 
	                  DMRG::COMPRESSION::INIT START = DMRG::COMPRESSION::RANDOM);
	
	/**Compresses a Chebyshev iteration step \f$V_{out} \approx 2H \cdot V_{in1} - V_{in2}\f$. Needs to calculate \f$\left<V_{in1}\right|H^2\left|V_{in1}\right>\f$, \f$\left<V_{in2}\right|H\left|V_{in1}\right>\f$ and \f$\big<V_{in2}\big|V_{in2}\big>\f$. Works optimally with OpenMP and (at least) 3 threads, as the last overlap is cheap to do in the mixed-canonical representation. If convergence is not reached after 2 half-sweeps, the bond dimension of \p Vout is increased and it is set to random.
	\warning The Hamiltonian has to be rescaled by 2 already.
	\param[in] H : Hamiltonian (an MpsQ with MpoQ::Qtarget() = qvacuum<Nq>()) rescaled by 2
	\param[in] Vin1 : input state to be multiplied
	\param[in] Vin2 : input state to be subtracted
	\param[out] Vout : compressed output state
	\param[in] Dcutoff_input : matrix size cutoff per site and subspace for \p Vout
	\param[in] tol : tolerance for the square norm of the difference: \f$\left|V_{out} - 2H \cdot V_{in1} - V_{in2}\right|^2<tol\f$
		\warning Too small a value for \p tol will lead to bad convergence. Try something of the order of 1e-3 to 1e-4.
	\param[in] max_halfsweeps : maximal amount of half-sweeps
	\param[in] min_halfsweeps : minimal amount of half-sweeps
	\param[in] START : choice of initial guess: 
		- DMRG::COMPRESSION::RANDOM : use a random state with the cutoff given by \p Dcutoff_input
		- DMRG::COMPRESSION::RHS : use \p Vin1 (previous Chebyshev iteration vector), \p Dcutoff_input is ignored
		- DMRG::COMPRESSION::BRUTAL_SVD : not implemented
		- DMRG::COMPRESSION::RHS_SVD : not implemented
	*/
	template<typename MpOperator>
	void chebCompress (const MpOperator &H, const MpsQ<Nq,Scalar> &Vin1, const MpsQ<Nq,Scalar> &Vin2, MpsQ<Nq,Scalar> &Vout, 
	                   size_t Dcutoff_input, double tol=1e-4, size_t max_halfsweeps=16, size_t min_halfsweeps=1, 
	                   DMRG::COMPRESSION::INIT START = DMRG::COMPRESSION::RHS);
	///\}
	
	//---info stuff---
	///\{
	/**\describe_info*/
	string info() const;
	/**\describe_memory*/
	double memory (MEMUNIT memunit=GB) const;
	/**\describe_overhead*/
	double overhead (MEMUNIT memunit=GB) const;
	///\}
	
private:
	
	// for |Vout> ≈ |Vin>
	vector<Biped<Nq,MatrixType> > L;
	vector<Biped<Nq,MatrixType> > R;
	void prepSweep (const MpsQ<Nq,Scalar> &Vin, MpsQ<Nq,Scalar> &Vout, bool RANDOMIZE=true);
	void optimizationStep (const MpsQ<Nq,Scalar> &Vin, MpsQ<Nq,Scalar> &Vout);
//	void optimizationStep (const MpsQ<Nq,Scalar> &Vin, std::array<Biped<Nq,Matrix<Scalar,Dynamic,Dynamic> >,D> &Aout);
	void sweepStep (const MpsQ<Nq,Scalar> &Vin, MpsQ<Nq,Scalar> &Vout);
	void build_L (size_t loc, const MpsQ<Nq,Scalar> &Vbra, const MpsQ<Nq,Scalar> &Vket);
	void build_R (size_t loc, const MpsQ<Nq,Scalar> &Vbra, const MpsQ<Nq,Scalar> &Vket);
	
	DMRG::VERBOSITY::OPTION CHOSEN_VERBOSITY;
	
	// for |Vout> ≈ H*|Vin>
	vector<PivotMatrixQ<Nq,Scalar,MpoScalar> > Heff;
	template<typename MpOperator>
	void prepSweep (const MpOperator &H, const MpsQ<Nq,Scalar> &Vin, MpsQ<Nq,Scalar> &Vout, DMRG::BROOM::OPTION TOOL = DMRG::BROOM::QR, bool RANDOMIZE=true);
	template<typename MpOperator>
	void optimizationStep (const MpOperator &H, const MpsQ<Nq,Scalar> &Vin, MpsQ<Nq,Scalar> &Vout);
	template<typename MpOperator>
	void sweepStep (const MpOperator &H, const MpsQ<Nq,Scalar> &Vin, MpsQ<Nq,Scalar> &Vout);
	template<typename MpOperator>
	void build_LW (size_t loc, const MpsQ<Nq,Scalar> &Vbra, const MpOperator &H, const MpsQ<Nq,Scalar> &Vket);
	template<typename MpOperator>
	void build_RW (size_t loc, const MpsQ<Nq,Scalar> &Vbra, const MpOperator &H, const MpsQ<Nq,Scalar> &Vket);
	
	// for |Vout> ≈ H*|Vin1> - |Vin2>
	template<typename MpOperator>
	void prepSweep (const MpOperator &H, const MpsQ<Nq,Scalar> &Vin1, const MpsQ<Nq,Scalar> &Vin2, MpsQ<Nq,Scalar> &Vout, bool RANDOMIZE=true);
	template<typename MpOperator>
	void sweepStep (const MpOperator &H, const MpsQ<Nq,Scalar> &Vin1, const MpsQ<Nq,Scalar> &Vin2, MpsQ<Nq,Scalar> &Vout);
	
	// mowing
	template<typename MpOperator> void mowSweeps (const MpOperator &H, MpsQ<Nq,Scalar> &Vout);
	void energyTruncationStep (MpsQ<Nq,Scalar> &Vbra, size_t dimK=10);
	ArrayXd mowedWeight;
	
	size_t N_sites;
	size_t N_sweepsteps, N_halfsweeps;
	size_t Dcutoff, Dcutoff_new;
	size_t Mmax, Mmax_new;
	double sqdist;
	
	int pivot;
	DMRG::DIRECTION::OPTION CURRENT_DIRECTION;
};

template<size_t Nq, typename Scalar, typename MpoScalar>
string MpsQCompressor<Nq,Scalar,MpoScalar>::
info() const
{
	stringstream ss;
	ss << "MpsQCompressor: ";
	ss << "Dcutoff=" << Dcutoff;
	if (Dcutoff != Dcutoff_new)
	{
		ss << "→" << Dcutoff_new << ", ";
		ss << "Mmax=" << Mmax << "→" << Mmax_new << ", ";
	}
	else
	{
		ss << " (not resized), ";
	}
	ss << "|Vlhs-Vrhs|^2=" << sqdist << ", ";
	ss << "halfsweeps=" << N_halfsweeps << ", ";
	ss << "mem=" << round(memory(GB),3) << "GB, overhead=" << round(overhead(MB),3) << "MB";
	return ss.str();
}

template<size_t Nq, typename Scalar, typename MpoScalar>
double MpsQCompressor<Nq,Scalar,MpoScalar>::
memory (MEMUNIT memunit) const
{
	double res = 0.;
	for (size_t l=0; l<L.size(); ++l)  {res += L[l].memory(memunit);}
	for (size_t l=0; l<R.size(); ++l)  {res += R[l].memory(memunit);}
	for (size_t l=0; l<Heff.size(); ++l)
	{
		res += Heff[l].L.memory(memunit);
		res += Heff[l].R.memory(memunit);
	}
	return res;
}

template<size_t Nq, typename Scalar, typename MpoScalar>
double MpsQCompressor<Nq,Scalar,MpoScalar>::
overhead (MEMUNIT memunit) const
{
	double res = 0.;
	for (size_t l=0; l<L.size(); ++l)  {res += L[l].overhead(memunit);}
	for (size_t l=0; l<R.size(); ++l)  {res += R[l].overhead(memunit);}
	for (size_t l=0; l<Heff.size(); ++l)
	{
		res += Heff[l].L.overhead(memunit);
		res += Heff[l].R.overhead(memunit);
		res += 2. * calc_memory<size_t>(Heff[l].qlhs.size(),memunit);
		res += 4. * calc_memory<size_t>(Heff[l].qrhs.size(),memunit);
	}
	return res;
}

//---------------------------compression of |Psi>---------------------------
// |Vout> ≈ |Vin>, M(Vout) < M(Vin)
// convention in program: <Vout|Vin>

template<size_t Nq, typename Scalar, typename MpoScalar>
void MpsQCompressor<Nq,Scalar,MpoScalar>::
varCompress (const MpsQ<Nq,Scalar> &Vin, MpsQ<Nq,Scalar> &Vout, size_t Dcutoff_input, double tol, size_t max_halfsweeps, size_t min_halfsweeps, DMRG::COMPRESSION::INIT START)
{
	Stopwatch Chronos;
	N_sites = Vin.length();
	double sqnormVin = isReal(dot(Vin,Vin));
	N_halfsweeps = 0;
	N_sweepsteps = 0;
	Dcutoff = Dcutoff_input;
	Dcutoff_new = Dcutoff_input;
	
	// set L&R edges
	L.resize(N_sites);
	R.resize(N_sites);
	R[N_sites-1].setTarget(Vin.Qtot);
	L[0].setVacuum();
	bool RANDOMIZE = false;
	
	// set initial guess
	if (START == DMRG::COMPRESSION::RANDOM or
	    START == DMRG::COMPRESSION::RHS)
	{
		Vout = Vin;
		Vout.dynamicResize(DMRG::RESIZE::DECR, Dcutoff);
		if (START == DMRG::COMPRESSION::RANDOM)
		{
			RANDOMIZE = true;
			//Vout.setRandom();
		}
	}
	else if (START == DMRG::COMPRESSION::BRUTAL_SVD or
	         START == DMRG::COMPRESSION::RHS_SVD)
	{
		Vout = Vin;
		Vout.N_sv = Dcutoff;
		if (Vout.pivot == -1)
		{
			Vout.skim(DMRG::DIRECTION::LEFT, DMRG::BROOM::BRUTAL_SVD);
		}
		else
		{
			Vout.skim(DMRG::BROOM::BRUTAL_SVD);
		}
	}
	
	Mmax = Vout.calc_Mmax();
	prepSweep(Vin,Vout,RANDOMIZE);
	sqdist = 1.;
	size_t halfSweepRange = N_sites;
	
	if (CHOSEN_VERBOSITY>=2)
	{
		lout << Chronos.info("preparation") << endl;
	}
	
	// must achieve sqdist > tol or break off after max_halfsweeps, do at least min_halfsweeps half-sweeps
	while ((sqdist > tol and N_halfsweeps < max_halfsweeps) or N_halfsweeps < min_halfsweeps)
	{
		Stopwatch Aion;
		for (size_t j=1; j<=halfSweepRange; ++j)
		{
			bring_her_about(pivot, N_sites, CURRENT_DIRECTION);
			optimizationStep(Vin,Vout);
			if (j != halfSweepRange)
			{
				sweepStep(Vin,Vout);
			}
			++N_sweepsteps;
		}
		halfSweepRange = N_sites-1;
		++N_halfsweeps;
		
		sqdist = abs(sqnormVin-Vout.squaredNorm());
		assert(!std::isnan(sqdist));
		// test with:
		//MpsQ<Nq,Scalar> Vtmp = Vbig;
		//Vtmp -= Vsmall;
		//abs(dot(Vtmp,Vtmp))
		
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
			size_t Dcutoff_old = Vout.calc_Dmax();
			size_t Mmax_old = Vout.calc_Mmax();
			
			Dcutoff_new = Dcutoff_old+1;
			Vout.dynamicResize(DMRG::RESIZE::CONSERV_INCR, Dcutoff_new);
			Vout.setRandom();
			Mmax_new = Vout.calc_Mmax();
			
			if (CHOSEN_VERBOSITY>=2)
			{
				lout << "resize: " << Dcutoff_old << "→" << Dcutoff_new << ", M=" << Mmax_old << "→" << Mmax_new << endl;
			}
			
			Vout.pivot = -1;
			prepSweep(Vin,Vout);
			pivot = Vout.pivot;
			halfSweepRange = N_sites;
			RESIZED = true;
		}
		
		if ((sqdist > tol and
		    RESIZED == false and
		    N_halfsweeps < max_halfsweeps) or
		    N_halfsweeps < min_halfsweeps)
		{
			sweepStep(Vin,Vout);
		}
	}
}

template<size_t Nq, typename Scalar, typename MpoScalar>
void MpsQCompressor<Nq,Scalar,MpoScalar>::
prepSweep (const MpsQ<Nq,Scalar> &Vin, MpsQ<Nq,Scalar> &Vout, bool RANDOMIZE)
{
	assert(Vout.pivot == 0 or Vout.pivot == N_sites-1 or Vout.pivot == -1);
	
	if (Vout.pivot == N_sites-1 or
	    Vout.pivot == -1)
	{
		for (size_t l=N_sites-1; l>0; --l)
		{
			if (RANDOMIZE == true)
			{
				Vout.setRandom(l);
				if (l>0) {Vout.setRandom(l-1);}
			}
			Vout.sweepStep(DMRG::DIRECTION::LEFT, l, DMRG::BROOM::QR);
			build_R(l-1,Vout,Vin);
		}
		CURRENT_DIRECTION = DMRG::DIRECTION::RIGHT;
	}
	else if (Vout.pivot == 0)
	{
		for (size_t l=0; l<N_sites-1; ++l)
		{
			if (RANDOMIZE == true)
			{
				Vout.setRandom(l);
				if (l<N_sites-1) {Vout.setRandom(l+1);}
			}
			Vout.sweepStep(DMRG::DIRECTION::RIGHT, l, DMRG::BROOM::QR);
			build_L(l+1,Vout,Vin);
		}
		CURRENT_DIRECTION = DMRG::DIRECTION::LEFT;
	}
	pivot = Vout.pivot;
}

template<size_t Nq, typename Scalar, typename MpoScalar>
void MpsQCompressor<Nq,Scalar,MpoScalar>::
sweepStep (const MpsQ<Nq,Scalar> &Vin, MpsQ<Nq,Scalar> &Vout)
{
	Vout.sweepStep(CURRENT_DIRECTION, pivot, DMRG::BROOM::QR);
	(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(++pivot,Vout,Vin) : build_R(--pivot,Vout,Vin);
}

template<size_t Nq, typename Scalar, typename MpoScalar>
void MpsQCompressor<Nq,Scalar,MpoScalar>::
optimizationStep (const MpsQ<Nq,Scalar> &Vin, MpsQ<Nq,Scalar> &Vout)
{
	for (size_t s=0; s<Vin.locBasis(pivot).size(); ++s)
	{
		Vout.A[pivot][s] = L[pivot] * Vin.A[pivot][s] * R[pivot];
	}
}

template<size_t Nq, typename Scalar, typename MpoScalar>
void MpsQCompressor<Nq,Scalar,MpoScalar>::
build_L (size_t loc, const MpsQ<Nq,Scalar> &Vbra, const MpsQ<Nq,Scalar> &Vket)
{
	L[loc] = Vbra.A[loc-1][0].adjoint() * L[loc-1] * Vket.A[loc-1][0];
	for (size_t s=1; s<Vbra.locBasis(loc-1).size(); ++s)
	{
		L[loc] += Vbra.A[loc-1][s].adjoint() * L[loc-1] * Vket.A[loc-1][s];
	}
}

template<size_t Nq, typename Scalar, typename MpoScalar>
void MpsQCompressor<Nq,Scalar,MpoScalar>::
build_R (size_t loc, const MpsQ<Nq,Scalar> &Vbra, const MpsQ<Nq,Scalar> &Vket)
{
	R[loc] = Vket.A[loc+1][0] * R[loc+1] * Vbra.A[loc+1][0].adjoint();
	for (size_t s=1; s<Vbra.locBasis(loc+1).size(); ++s)
	{
		R[loc] += Vket.A[loc+1][s] * R[loc+1] * Vbra.A[loc+1][s].adjoint();
	}
}

//---------------------------compression of H*|Psi>---------------------------
// |Vout> ≈ H|Vin>
// convention in program: <Vout|H|Vin>

template<size_t Nq, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsQCompressor<Nq,Scalar,MpoScalar>::
varCompress (const MpOperator &H, const MpsQ<Nq,Scalar> &Vin, MpsQ<Nq,Scalar> &Vout, size_t Dcutoff_input, double tol, size_t max_halfsweeps, size_t min_halfsweeps, DMRG::COMPRESSION::INIT START)
{
	N_sites = Vin.length();
	Stopwatch Chronos;
	N_halfsweeps = 0;
	N_sweepsteps = 0;
	Dcutoff = Dcutoff_input;
	Dcutoff_new = Dcutoff_input;
	
	if (START == DMRG::COMPRESSION::RHS)
	{
		Vout = Vin;
	}
	else if (START == DMRG::COMPRESSION::RANDOM)
	{
		Vout = Vin;
		Vout.dynamicResize(DMRG::RESIZE::DECR, Dcutoff);
		Vout.setRandom();
	}
	else if (START == DMRG::COMPRESSION::RHS_SVD)
	{
		OxV(H,Vin,Vout,DMRG::BROOM::QR);
	}
	else if (START == DMRG::COMPRESSION::BRUTAL_SVD)
	{
		size_t tmp = Vout.N_sv;
		Vout.N_sv = Dcutoff;
		OxV(H,Vin,Vout,DMRG::BROOM::BRUTAL_SVD);
		Vout.N_sv = tmp;
	}
	
	// prepare edges of LW & RW
	Heff.clear();
	Heff.resize(N_sites);
	Heff[0].L.setVacuum();
	Heff[N_sites-1].R.setTarget(qarray3<Nq>{Vin.Qtarget(), Vout.Qtarget(), qvacuum<Nq>()});
	
	Mmax = Vout.calc_Mmax();
	double sqnormVin;
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
				sqnormVin = Vin.squaredNorm();
			}
			else
			{
				sqnormVin = (H.check_SQUARE()==true)? isReal(avg(Vin,H,Vin,true)) : isReal(avg(Vin,H,H,Vin));
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
	
	// must achieve sqdist > tol or break off after max_halfsweeps, do at least 2 half-sweeps
	while ((sqdist > tol and N_halfsweeps < max_halfsweeps) or N_halfsweeps < min_halfsweeps)
	{
		Stopwatch Aion;
		for (size_t j=1; j<=halfSweepRange; ++j)
		{
			bring_her_about(pivot, N_sites, CURRENT_DIRECTION);
			optimizationStep(H,Vin,Vout);
			if (j != halfSweepRange) {sweepStep(H,Vin,Vout);}
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
			size_t Dcutoff_old = Vout.calc_Dmax();
			size_t Mmax_old = Vout.calc_Mmax();
			
			Dcutoff_new = Dcutoff_old+1;
			auto Vtmp = Vout;
			Vout.skim(DMRG::BROOM::SVD);
			Vout.dynamicResize(DMRG::RESIZE::CONSERV_INCR, Dcutoff_new);
			Vout.setRandom();
			
			Mmax_new = Vout.calc_Mmax();
			
			if (CHOSEN_VERBOSITY>=2)
			{
				lout << "resize: " << Dcutoff_old << "→" << Dcutoff_new << ", M=" << Mmax_old << "→" << Mmax_new << endl;
			}
			
			Vout.pivot = -1;
			prepSweep(H,Vin,Vout);
			pivot = Vout.pivot;
			halfSweepRange = N_sites;
			RESIZED = true;
		}
		
		if ((sqdist > tol and 
		    RESIZED == false and 
		    N_halfsweeps < max_halfsweeps) or
		    N_halfsweeps < min_halfsweeps)
		{
			sweepStep(H,Vin,Vout);
		}
	}
	
	// mowing
	if (Vout.N_mow > 0)
	{
		mowSweeps(H,Vout);
	}
}

template<size_t Nq, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsQCompressor<Nq,Scalar,MpoScalar>::
prepSweep (const MpOperator &H, const MpsQ<Nq,Scalar> &Vin, MpsQ<Nq,Scalar> &Vout, DMRG::BROOM::OPTION TOOL, bool RANDOMIZE)
{
	assert(Vout.pivot == 0 or Vout.pivot == N_sites-1 or Vout.pivot == -1);
	
	if (Vout.pivot == N_sites-1 or Vout.pivot == -1)
	{
		for (size_t l=N_sites-1; l>0; --l)
		{
			if (RANDOMIZE == true)
			{
				Vout.setRandom(l);
				if (l>0) {Vout.setRandom(l-1);}
			}
			Stopwatch Chronos;
			Vout.sweepStep(DMRG::DIRECTION::LEFT, l, TOOL);
			build_RW(l-1,Vout,H,Vin);
		}
		CURRENT_DIRECTION = DMRG::DIRECTION::RIGHT;
	}
	else if (Vout.pivot == 0)
	{
		for (size_t l=0; l<N_sites-1; ++l)
		{
			if (RANDOMIZE == true)
			{
				Vout.setRandom(l);
				if (l<N_sites-1) {Vout.setRandom(l+1);}
			}
			Stopwatch Chronos;
			Vout.sweepStep(DMRG::DIRECTION::RIGHT, l, TOOL);
			build_LW(l+1,Vout,H,Vin);
		}
		CURRENT_DIRECTION = DMRG::DIRECTION::LEFT;
	}
	pivot = Vout.pivot;
}

template<size_t Nq, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsQCompressor<Nq,Scalar,MpoScalar>::
sweepStep (const MpOperator &H, const MpsQ<Nq,Scalar> &Vin, MpsQ<Nq,Scalar> &Vout)
{
	Vout.sweepStep(CURRENT_DIRECTION, pivot, DMRG::BROOM::QR);
	(CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? build_LW(++pivot,Vout,H,Vin) : build_RW(--pivot,Vout,H,Vin);
}

template<size_t Nq, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsQCompressor<Nq,Scalar,MpoScalar>::
optimizationStep (const MpOperator &H, const MpsQ<Nq,Scalar> &Vin, MpsQ<Nq,Scalar> &Vout)
{
	Stopwatch Chronos;
	
	for (size_t s=0; s<Vin.locBasis(pivot).size(); ++s)
	{
		Vout.A[pivot][s].setZero();
	}
	
//	if (Heff[pivot].qlhs.size() == 0) // Doesn't work with chebCompress, why? Or does it?
	{
		Heff[pivot].W = H.W[pivot];
		precalc_blockStructure (Heff[pivot].L, Vout.A[pivot], Heff[pivot].W, Vin.A[pivot], Heff[pivot].R, 
		                        H.locBasis(pivot), Heff[pivot].qlhs, Heff[pivot].qrhs);
	}
	
	// why doesn't this work?
//	PivotVectorQ<Nq,Scalar> Vtmp;
//	Vtmp.A = Vin.A[pivot];
//	HxV(Heff[pivot],Vtmp);
//	Vout.A[pivot] = Vtmp.A;
	
	#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
	#pragma omp parallel for
	#endif
	for (size_t q=0; q<Heff[pivot].qlhs.size(); ++q)
	{
		size_t s1 = Heff[pivot].qlhs[q][0];
		size_t q1 = Heff[pivot].qlhs[q][1];
		
		for (auto irhs=Heff[pivot].qrhs[q].begin(); irhs!=Heff[pivot].qrhs[q].end(); ++irhs)
		{
			size_t s2 = (*irhs)[0];
			size_t q2 = (*irhs)[1];
			size_t qL = (*irhs)[2];
			size_t qR = (*irhs)[3];
			
			for (int k=0; k<H.W[pivot][s1][s2].outerSize(); ++k)
			for (typename SparseMatrix<MpoScalar>::InnerIterator iW(H.W[pivot][s1][s2],k); iW; ++iW)
			{
				if (Heff[pivot].L.block[qL][iW.row()][0].rows() != 0 and 
				    Heff[pivot].R.block[qR][iW.col()][0].rows() != 0)
				{
					Vout.A[pivot][s1].block[q1].noalias() += iW.value() * 
					                                         (Heff[pivot].L.block[qL][iW.row()][0] * 
					                                          Vin.A[pivot][s2].block[q2] * 
					                                          Heff[pivot].R.block[qR][iW.col()][0]);
				}
			}
		}
	}
	
	if (CHOSEN_VERBOSITY == DMRG::VERBOSITY::STEPWISE)
	{
		lout << "optimization loc=" << Chronos.info(pivot) << endl;
	}
}

template<size_t Nq, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsQCompressor<Nq,Scalar,MpoScalar>::
build_LW (size_t loc, const MpsQ<Nq,Scalar> &Vbra, const MpOperator &H, const MpsQ<Nq,Scalar> &Vket)
{
	contract_L(Heff[loc-1].L, Vbra.A[loc-1], H.W[loc-1], Vket.A[loc-1], H.locBasis(loc-1), Heff[loc].L);
}

template<size_t Nq, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsQCompressor<Nq,Scalar,MpoScalar>::
build_RW (size_t loc, const MpsQ<Nq,Scalar> &Vbra, const MpOperator &H, const MpsQ<Nq,Scalar> &Vket)
{
	contract_R(Heff[loc+1].R, Vbra.A[loc+1], H.W[loc+1], Vket.A[loc+1], H.locBasis(loc+1), Heff[loc].R);
}

template<size_t Nq, typename Scalar, typename MpoScalar>
void MpsQCompressor<Nq,Scalar,MpoScalar>::
energyTruncationStep (MpsQ<Nq,Scalar> &V, size_t dimK)
{
	if (Heff[pivot].qlhs.size() == 0)
	{
		precalc_blockStructure (Heff[pivot].L, V.A[pivot], Heff[pivot].W, V.A[pivot], Heff[pivot].R, 
		                        V.locBasis(pivot), Heff[pivot].qlhs, Heff[pivot].qrhs);
	}
	
	Heff[pivot].dim = 0;
	for (size_t s=0; s<V.locBasis(pivot).size(); ++s)
	for (size_t q=0; q<V.A[pivot][s].dim; ++q)
	{
		Heff[pivot].dim += V.A[pivot][s].block[q].rows() * V.A[pivot][s].block[q].cols();
	}
	
	PivotVectorQ<Nq,Scalar> Psi;
	Psi.A = V.A[pivot];
	
	LanczosMower<PivotMatrixQ<Nq,Scalar,MpoScalar>,PivotVectorQ<Nq,Scalar>,Scalar> Lutz(min(dimK,Heff[pivot].dim));
	Lutz.mow(Heff[pivot],Psi,2.);
	mowedWeight(pivot) = Lutz.get_mowedWeight();
	
	V.A[pivot] = Psi.A;
}

template<size_t Nq, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsQCompressor<Nq,Scalar,MpoScalar>::
chebCompress (const MpOperator &H, const MpsQ<Nq,Scalar> &Vin1, const MpsQ<Nq,Scalar> &Vin2, MpsQ<Nq,Scalar> &Vout, size_t Dcutoff_input, double tol, size_t max_halfsweeps, size_t min_halfsweeps, DMRG::COMPRESSION::INIT START)
{
	N_sites = Vin1.length();
	Stopwatch Chronos;
	N_halfsweeps = 0;
	N_sweepsteps = 0;
	Dcutoff = Dcutoff_input;
	Dcutoff_new = Dcutoff_input;
	
	if (START == DMRG::COMPRESSION::RHS)
	{
		Vout = Vin1;
	}
	else if (START == DMRG::COMPRESSION::RANDOM)
	{
		Vout = Vin1;
		Vout.dynamicResize(DMRG::RESIZE::DECR, Dcutoff);
		Vout.setRandom();
	}
//	else if (START == DMRG::COMPRESSION::RHS_SVD)
//	{
//		OxV(H,Vin,Vout,DMRG::BROOM::QR);
//	}
//	else if (START == DMRG::COMPRESSION::BRUTAL_SVD)
//	{
//		size_t tmp = Vout.N_sv;
//		Vout.N_sv = Dcutoff;
//		OxV(H,Vin,Vout,DMRG::BROOM::BRUTAL_SVD);
//		Vout.N_sv = tmp;
//	}
	
	// prepare edges of LW & RW
	Heff.clear();
	Heff.resize(N_sites);
	Heff[0].L.setVacuum();
	Heff[N_sites-1].R.setTarget(qarray3<Nq>{Vin1.Qtarget(), Vout.Qtarget(), qvacuum<Nq>()});
	
	// set L&R edges
	L.resize(N_sites);
	R.resize(N_sites);
	R[N_sites-1].setTarget(Vin2.Qtot);
	L[0].setVacuum();
	
	Mmax = Vout.calc_Mmax();
	double sqnormV1, sqnormV2, overlapV12;
	sqnormV2 = Vin2.squaredNorm();
	#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
	#pragma omp parallel sections
	#endif
	{
		#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
		#pragma omp section
		#endif
		{
			sqnormV1 = (H.check_SQUARE()==true)? isReal(avg(Vin1,H,Vin1,true)) : isReal(avg(Vin1,H,H,Vin1));
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
	
	// must achieve sqdist > tol or break off after max_halfsweeps, do at least 2 half-sweeps
	while ((sqdist > tol and N_halfsweeps < max_halfsweeps) or N_halfsweeps < min_halfsweeps)
	{
		Stopwatch Aion;
		for (size_t j=1; j<=halfSweepRange; ++j)
		{
			bring_her_about(pivot, N_sites, CURRENT_DIRECTION);
			Stopwatch Chronos;
			optimizationStep(Vin2,Vout);
			auto Atmp = Vout.A[pivot];
			optimizationStep(H,Vin1,Vout);
			for (size_t s=0; s<H.locBasis(pivot).size(); ++s)
			for (size_t q=0; q<Atmp[s].dim; ++q)
			{
				qarray2<Nq> quple = {Atmp[s].in[q], Atmp[s].out[q]};
				auto it = Vout.A[pivot][s].dict.find(quple);
				Vout.A[pivot][s].block[it->second] -= Atmp[s].block[q];
			}
			if (j != halfSweepRange)
			{
				sweepStep(H,Vin1,Vin2,Vout);
			}
			++N_sweepsteps;
		}
		halfSweepRange = N_sites-1;
		++N_halfsweeps;
		
		sqdist = abs(sqnormV1+sqnormV2-Vout.squaredNorm()-2.*overlapV12);
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
			Stopwatch ChronosResize;
			size_t Dcutoff_old = Vout.calc_Dmax();
			size_t Mmax_old = Vout.calc_Mmax();
			
			Dcutoff_new = Dcutoff_old+1;
			Vout.dynamicResize(DMRG::RESIZE::CONSERV_INCR, Dcutoff_new);
			Vout.setRandom();
			Mmax_new = Vout.calc_Mmax();
			
			Vout.pivot = -1;
			prepSweep(H,Vin1,Vin2,Vout);
			pivot = Vout.pivot;
			halfSweepRange = N_sites;
			RESIZED = true;
			
			if (CHOSEN_VERBOSITY>=2)
			{
				lout << "resize: " << Dcutoff_old << "→" << Dcutoff_new << ", M=" << Mmax_old << "→" << Mmax_new << "\t" << ChronosResize.info() << endl;
			}
		}
		
		if ((sqdist > tol and 
		    RESIZED == false and
		    N_halfsweeps < max_halfsweeps) or 
		    N_halfsweeps < min_halfsweeps)
		{
			sweepStep(H,Vin1,Vin2,Vout);
		}
	}
	
	// mowing
	if (Vout.N_mow > 0)
	{
		mowSweeps(H,Vout);
	}
}

template<size_t Nq, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsQCompressor<Nq,Scalar,MpoScalar>::
mowSweeps (const MpOperator &H, MpsQ<Nq,Scalar> &Vout)
{
//	mowedWeight.resize(N_sites);
//	
//	Heff.clear();
//	Heff.resize(N_sites);
//	Heff[0].L.setVacuum();
//	Heff[N_sites-1].R.setTarget(qarray3<Nq>{Vout.Qtarget(), Vout.Qtarget(), qvacuum<Nq>()});
//	for (size_t l=0; l<N_sites; ++l) {Heff[l].W = H.W[l];}
//	
//	// preparation
//	Stopwatch Aion;
//	
//	if (Vout.pivot == N_sites-1 or Vout.pivot == -1)
//	{
//		for (size_t l=N_sites-1; l>0; --l)
//		{
//			Vout.sweepStep(DMRG::DIRECTION::LEFT, l, DMRG::BROOM::QR);
//			build_RW(l-1,Vout,H,Vout);
//		}
//		CURRENT_DIRECTION = DMRG::DIRECTION::RIGHT;
//	}
//	else if (Vout.pivot == 0)
//	{
//		for (size_t l=0; l<N_sites-1; ++l)
//		{
//			Vout.sweepStep(DMRG::DIRECTION::RIGHT, l, DMRG::BROOM::QR);
//			build_LW(l+1,Vout,H,Vout);
//		}
//		CURRENT_DIRECTION = DMRG::DIRECTION::LEFT;
//	}
//	pivot = Vout.pivot;
//	
//	if (CHOSEN_VERBOSITY<2)
//	{
//		lout << Aion.info("mowing preparation") << endl;
//	}
//	
//	size_t halfSweepRange = N_sites;
//	
//	for (size_t j=1; j<=Vout.N_mow; ++j)
//	{
//		mowedWeight.setZero();
//		Stopwatch Aion;
//		for (size_t l=1; l<=halfSweepRange; ++l)
//		{
//			bring_her_about(pivot, N_sites, CURRENT_DIRECTION);
//			energyTruncationStep(Vout);
//			if (l != halfSweepRange)
//			{
//				Vout.sweepStep(CURRENT_DIRECTION, pivot, DMRG::BROOM::QR);
//				(CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? build_LW(++pivot,Vout,H,Vout) : build_RW(--pivot,Vout,H,Vout);
//			}
//		}
//		halfSweepRange = N_sites-1;
//		
//		if (CHOSEN_VERBOSITY<2)
//		{
//			lout << Aion.info("half-mowsweep") << "\tmowed_weight=" << mowedWeight.sum() << endl;
//		}
//		
//		if (j != Vout.N_mow)
//		{
//			Vout.sweepStep(CURRENT_DIRECTION, pivot, DMRG::BROOM::QR);
//			(CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? build_LW(++pivot,Vout,H,Vout) : build_RW(--pivot,Vout,H,Vout);
//		}
//	}
}

template<size_t Nq, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsQCompressor<Nq,Scalar,MpoScalar>::
prepSweep (const MpOperator &H, const MpsQ<Nq,Scalar> &Vin1, const MpsQ<Nq,Scalar> &Vin2, MpsQ<Nq,Scalar> &Vout, bool RANDOMIZE)
{
	assert(Vout.pivot == 0 or Vout.pivot == N_sites-1 or Vout.pivot == -1);
	
	if (Vout.pivot == N_sites-1)
	{
		for (size_t l=N_sites-1; l>0; --l)
		{
			if (RANDOMIZE == true)
			{
				Vout.setRandom(l);
				if (l>0) {Vout.setRandom(l-1);}
			}
			Vout.sweepStep(DMRG::DIRECTION::LEFT, l, DMRG::BROOM::QR);
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
			if (RANDOMIZE == true)
			{
				Vout.setRandom(l);
				if (l<N_sites-1) {Vout.setRandom(l+1);}
			}
			Vout.sweepStep(DMRG::DIRECTION::RIGHT, l, DMRG::BROOM::QR);
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

template<size_t Nq, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsQCompressor<Nq,Scalar,MpoScalar>::
sweepStep (const MpOperator &H, const MpsQ<Nq,Scalar> &Vin1, const MpsQ<Nq,Scalar> &Vin2, MpsQ<Nq,Scalar> &Vout)
{
	Vout.sweepStep(CURRENT_DIRECTION, pivot, DMRG::BROOM::QR);
	(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? ++pivot : --pivot;
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
			(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L (pivot,Vout,Vin2)     : build_R(pivot,Vout,Vin2);
		}
	}
}

#endif
