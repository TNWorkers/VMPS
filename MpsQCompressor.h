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

#include "Biped.h"
#include "Multipede.h"
#include "LanczosSolver.h"
#include "DmrgContractionsQ.h"
#include "DmrgPivotStuffQ.h"
#include "DmrgPivotStuff2Q.h"
#include "LanczosMower.h"
#include "Stopwatch.h"

/**Compressor of Matrix Product States with conserved quantum numbers.
\describe_Symmetry
\describe_Scalar
\describe_MpoScalar*/
template<typename Symmetry, typename Scalar, typename MpoScalar=double>
class MpsQCompressor
{
typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;

public:
	
	//---constructor---
	MpsQCompressor (DMRG::VERBOSITY::OPTION VERBOSITY=DMRG::VERBOSITY::SILENT)
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
	void varCompress (const MpsQ<Symmetry,Scalar> &Vin, MpsQ<Symmetry,Scalar> &Vout, 
	                  size_t Dcutoff_input, double tol=1e-5, size_t max_halfsweeps=40, size_t min_halfsweeps=1, 
	                  DMRG::COMPRESSION::INIT START = DMRG::COMPRESSION::BRUTAL_SVD);
	
	/**Compresses a matrix-vector product \f$\left|V_{out}\right> \approx H \left|V_{in}\right>\f$. Needs to calculate \f$\left<V_{in}\right|H^2\left|V_{in}\right>\f$. Works optimally with OpenMP and (at least) 2 threads. If convergence is not reached after 2 half-sweeps, the bond dimension of \p Vout is increased and it is set to random.
	\param[in] H : Hamiltonian (an MpsQ with MpoQ::Qtarget() = Symmetry::qvacuum())
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
	void varCompress (const MpOperator &H, const MpsQ<Symmetry,Scalar> &Vin, MpsQ<Symmetry,Scalar> &Vout, 
	                  size_t Dcutoff_input, double tol=1e-5, size_t max_halfsweeps=40, size_t min_halfsweeps=1, 
	                  DMRG::COMPRESSION::INIT START = DMRG::COMPRESSION::RANDOM);
	
	/**Compresses an orthogonal iteration step \f$V_{out} \approx (C_n H - A_n) \cdot V_{in1} - B_n V_{in2}\f$. Needs to calculate \f$\left<V_{in1}\right|H^2\left|V_{in1}\right>\f$, \f$\left<V_{in2}\right|H\left|V_{in1}\right>\f$ and \f$\big<V_{in2}\big|V_{in2}\big>\f$. Works optimally with OpenMP and (at least) 3 threads, as the last overlap is cheap to do in the mixed-canonical representation. If convergence is not reached after 4 half-sweeps, the bond dimension of \p Vout is increased and it is set to random.
	\warning The Hamiltonian has to be rescaled by \p C_n and \p A_n already.
	\param[in] H : Hamiltonian (an MpsQ with MpoQ::Qtarget() = Symmetry::qvacuum()) rescaled by 2
	\param[in] Vin1 : input state to be multiplied
	\param[in] polyB : the coefficient before the subtracted vector
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
	void polyCompress (const MpOperator &H, const MpsQ<Symmetry,Scalar> &Vin1, double polyB, const MpsQ<Symmetry,Scalar> &Vin2, MpsQ<Symmetry,Scalar> &Vout, 
	                   size_t Dcutoff_input, double tol=DMRG_POLYCOMPRESS_TOL, size_t max_halfsweeps=DMRG_POLYCOMPRESS_MAX, size_t min_halfsweeps=DMRG_POLYCOMPRESS_MIN, 
	                   DMRG::COMPRESSION::INIT START = DMRG::COMPRESSION::RHS);
	
//	template<typename MpOperator>
//	void sumCompress (const vector<MpsQ<Symmetry,Scalar> > &Vin, const vector<double> &factor, MpsQ<Symmetry,Scalar> &Vout, 
//	                  size_t Dcutoff_input, double tol=DMRG_POLYCOMPRESS_TOL, size_t max_halfsweeps=DMRG_POLYCOMPRESS_MAX, size_t min_halfsweeps=DMRG_POLYCOMPRESS_MIN, 
//	                  DMRG::COMPRESSION::INIT START = DMRG::COMPRESSION::RHS);
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
	vector<Biped<Symmetry,MatrixType> > L;
	vector<Biped<Symmetry,MatrixType> > R;
	void prepSweep (const MpsQ<Symmetry,Scalar> &Vin, MpsQ<Symmetry,Scalar> &Vout, bool RANDOMIZE=true);
	void optimizationStep (const MpsQ<Symmetry,Scalar> &Vin, MpsQ<Symmetry,Scalar> &Vout);
	void optimizationStep2 (const MpsQ<Symmetry,Scalar> &Vin, MpsQ<Symmetry,Scalar> &Vout);
	void sweepStep (const MpsQ<Symmetry,Scalar> &Vin, MpsQ<Symmetry,Scalar> &Vout);
	void build_L (size_t loc, const MpsQ<Symmetry,Scalar> &Vbra, const MpsQ<Symmetry,Scalar> &Vket);
	void build_R (size_t loc, const MpsQ<Symmetry,Scalar> &Vbra, const MpsQ<Symmetry,Scalar> &Vket);
	
	DMRG::VERBOSITY::OPTION CHOSEN_VERBOSITY;
	
	// for |Vout> ≈ H*|Vin>
	vector<PivotMatrixQ<Symmetry,Scalar,MpoScalar> > Heff;
	template<typename MpOperator>
	void prepSweep (const MpOperator &H, const MpsQ<Symmetry,Scalar> &Vin, MpsQ<Symmetry,Scalar> &Vout, bool RANDOMIZE=true);
	template<typename MpOperator>
	void optimizationStep (const MpOperator &H, const MpsQ<Symmetry,Scalar> &Vin, MpsQ<Symmetry,Scalar> &Vout);
	template<typename MpOperator>
	void optimizationStep2 (const MpOperator &H, const MpsQ<Symmetry,Scalar> &Vin, MpsQ<Symmetry,Scalar> &Vout);
	template<typename MpOperator>
	void sweepStep (const MpOperator &H, const MpsQ<Symmetry,Scalar> &Vin, MpsQ<Symmetry,Scalar> &Vout);
	template<typename MpOperator>
	void build_LW (size_t loc, const MpsQ<Symmetry,Scalar> &Vbra, const MpOperator &H, const MpsQ<Symmetry,Scalar> &Vket);
	template<typename MpOperator>
	void build_RW (size_t loc, const MpsQ<Symmetry,Scalar> &Vbra, const MpOperator &H, const MpsQ<Symmetry,Scalar> &Vket);
	
	// for |Vout> ≈ H*|Vin1> - |Vin2>
	template<typename MpOperator>
	void prepSweep (const MpOperator &H, const MpsQ<Symmetry,Scalar> &Vin1, const MpsQ<Symmetry,Scalar> &Vin2, MpsQ<Symmetry,Scalar> &Vout, bool RANDOMIZE=true);
	template<typename MpOperator>
	void sweepStep (const MpOperator &H, const MpsQ<Symmetry,Scalar> &Vin1, const MpsQ<Symmetry,Scalar> &Vin2, MpsQ<Symmetry,Scalar> &Vout);
	
	// mowing
	template<typename MpOperator> void mowSweeps (const MpOperator &H, MpsQ<Symmetry,Scalar> &Vout);
	void energyTruncationStep (MpsQ<Symmetry,Scalar> &Vbra, size_t dimK=10);
	ArrayXd mowedWeight;
	
	size_t N_sites;
	size_t N_sweepsteps, N_halfsweeps;
	size_t Dcutoff, Dcutoff_new;
	size_t Mmax, Mmax_new;
	double sqdist;
	
	int pivot;
	DMRG::DIRECTION::OPTION CURRENT_DIRECTION;
};

template<typename Symmetry, typename Scalar, typename MpoScalar>
string MpsQCompressor<Symmetry,Scalar,MpoScalar>::
info() const
{
	stringstream ss;
	ss << "MpsQCompressor: ";
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
	ss << "mem=" << round(memory(GB),3) << "GB, overhead=" << round(overhead(MB),3) << "MB";
	return ss.str();
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
double MpsQCompressor<Symmetry,Scalar,MpoScalar>::
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

template<typename Symmetry, typename Scalar, typename MpoScalar>
double MpsQCompressor<Symmetry,Scalar,MpoScalar>::
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

template<typename Symmetry, typename Scalar, typename MpoScalar>
void MpsQCompressor<Symmetry,Scalar,MpoScalar>::
varCompress (const MpsQ<Symmetry,Scalar> &Vin, MpsQ<Symmetry,Scalar> &Vout, size_t Dcutoff_input, double tol, size_t max_halfsweeps, size_t min_halfsweeps, DMRG::COMPRESSION::INIT START)
{
	Stopwatch<> Chronos;
	N_sites = Vin.length();
	double sqnormVin = isReal(dot(Vin,Vin));
	N_halfsweeps = 0;
	N_sweepsteps = 0;
	Dcutoff = Dcutoff_new = Dcutoff_input;
	
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
	
	Vout.N_sv = Dcutoff;
	Mmax = Vout.calc_Mmax();
	prepSweep(Vin,Vout,RANDOMIZE);
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
		for (size_t j=1; j<=halfSweepRange; ++j)
		{
			bring_her_about(pivot, N_sites, CURRENT_DIRECTION);
//			optimizationStep(Vin,Vout);
//			if (j != halfSweepRange)
//			{
//				sweepStep(Vin,Vout);
//			}
			if (N_halfsweeps%4 == 0 and N_halfsweeps > 0)
			{
				optimizationStep2(Vin,Vout);
			}
			else
			{
				optimizationStep(Vin,Vout);
				Vout.sweepStep(CURRENT_DIRECTION, pivot, DMRG::BROOM::QR);
				pivot = Vout.get_pivot();
			}
			sweepStep(Vin,Vout);
			++N_sweepsteps;
		}
		halfSweepRange = N_sites-1;
		++N_halfsweeps;
		
		sqdist = abs(sqnormVin-Vout.squaredNorm());
		assert(!std::isnan(sqdist));
		// test with:
		//MpsQ<Symmetry,Scalar> Vtmp = Vbig;
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
//			size_t Dcutoff_old = Vout.calc_Dmax();
//			size_t Mmax_old = Vout.calc_Mmax();
//			
//			Dcutoff_new = Dcutoff_old+1;
//			Vout.dynamicResize(DMRG::RESIZE::CONSERV_INCR, Dcutoff_new);
//			Vout.setRandom();
//			Mmax_new = Vout.calc_Mmax();
//			
//			if (CHOSEN_VERBOSITY>=2)
//			{
//				lout << "resize: " << Dcutoff_old << "→" << Dcutoff_new << ", M=" << Mmax_old << "→" << Mmax_new << endl;
//			}
//			
//			Vout.pivot = -1;
//			prepSweep(Vin,Vout);
//			pivot = Vout.pivot;
//			halfSweepRange = N_sites;
//			RESIZED = true;
			
			Vout.N_sv += 1;
			Dcutoff_new = Vout.N_sv;
			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
			{
				lout << "resize: " << Vout.N_sv-1 << "→" << Vout.N_sv << endl;
			}
		}
		
		Mmax_new = Vout.calc_Mmax();
		
//		if ((sqdist > tol and
//		    RESIZED == false and
//		    N_halfsweeps < max_halfsweeps) or
//		    N_halfsweeps < min_halfsweeps)
//		{
//			sweepStep(Vin,Vout);
//		}
	}
	
	// last sweep
	if      (pivot==1)         {Vout.sweep(0,DMRG::BROOM::QR);}
	else if (pivot==N_sites-2) {Vout.sweep(N_sites-1,DMRG::BROOM::QR);}
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void MpsQCompressor<Symmetry,Scalar,MpoScalar>::
prepSweep (const MpsQ<Symmetry,Scalar> &Vin, MpsQ<Symmetry,Scalar> &Vout, bool RANDOMIZE)
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
//				if (l>0) {Vout.setRandom(l-1);}
			}
			Vout.sweepStep(DMRG::DIRECTION::LEFT, l, DMRG::BROOM::QR, NULL,true);
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
//				if (l<N_sites-1) {Vout.setRandom(l+1);}
			}
			Vout.sweepStep(DMRG::DIRECTION::RIGHT, l, DMRG::BROOM::QR, NULL,true);
			build_L(l+1,Vout,Vin);
		}
		CURRENT_DIRECTION = DMRG::DIRECTION::LEFT;
	}
	pivot = Vout.pivot;
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void MpsQCompressor<Symmetry,Scalar,MpoScalar>::
sweepStep (const MpsQ<Symmetry,Scalar> &Vin, MpsQ<Symmetry,Scalar> &Vout)
{
//	Vout.sweepStep(CURRENT_DIRECTION, pivot, DMRG::BROOM::QR);
//	(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(++pivot,Vout,Vin) : build_R(--pivot,Vout,Vin);
	(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(pivot,Vout,Vin) : build_R(pivot,Vout,Vin);
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void MpsQCompressor<Symmetry,Scalar,MpoScalar>::
optimizationStep (const MpsQ<Symmetry,Scalar> &Vin, MpsQ<Symmetry,Scalar> &Vout)
{
	for (size_t s=0; s<Vin.locBasis(pivot).size(); ++s)
	{
		Vout.A[pivot][s] = L[pivot] * Vin.A[pivot][s] * R[pivot];
	}
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void MpsQCompressor<Symmetry,Scalar,MpoScalar>::
optimizationStep2 (const MpsQ<Symmetry,Scalar> &Vin, MpsQ<Symmetry,Scalar> &Vout)
{
	size_t loc1 = (CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? pivot : pivot-1;
	size_t loc2 = (CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? pivot+1 : pivot;
	
	vector<vector<Biped<Symmetry,MatrixType> > > Apair;
	Apair.resize(Vin.locBasis(loc1).size());
	for (size_t s1=0; s1<Vin.locBasis(loc1).size(); ++s1)
	{
		Apair[s1].resize(Vin.locBasis(loc2).size());
	}
	
	for (size_t s1=0; s1<Vin.locBasis(loc1).size(); ++s1)
	for (size_t s3=0; s3<Vin.locBasis(loc2).size(); ++s3)
	{
		Apair[s1][s3] = L[loc1] * Vin.A[loc1][s1] * Vin.A[loc2][s3] * R[loc2];
	}
	
	Vout.sweepStep2(CURRENT_DIRECTION, min(loc1,loc2), Apair);
	
	pivot = Vout.get_pivot();
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void MpsQCompressor<Symmetry,Scalar,MpoScalar>::
build_L (size_t loc, const MpsQ<Symmetry,Scalar> &Vbra, const MpsQ<Symmetry,Scalar> &Vket)
{
	L[loc] = Vbra.A[loc-1][0].adjoint() * L[loc-1] * Vket.A[loc-1][0];
	for (size_t s=1; s<Vbra.locBasis(loc-1).size(); ++s)
	{
		L[loc] += Vbra.A[loc-1][s].adjoint() * L[loc-1] * Vket.A[loc-1][s];
	}
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void MpsQCompressor<Symmetry,Scalar,MpoScalar>::
build_R (size_t loc, const MpsQ<Symmetry,Scalar> &Vbra, const MpsQ<Symmetry,Scalar> &Vket)
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

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsQCompressor<Symmetry,Scalar,MpoScalar>::
varCompress (const MpOperator &H, const MpsQ<Symmetry,Scalar> &Vin, MpsQ<Symmetry,Scalar> &Vout, size_t Dcutoff_input, double tol, size_t max_halfsweeps, size_t min_halfsweeps, DMRG::COMPRESSION::INIT START)
{
	N_sites = Vin.length();
	Stopwatch<> Chronos;
	N_halfsweeps = 0;
	N_sweepsteps = 0;
	Dcutoff = Dcutoff_new = Dcutoff_input;
	bool RANDOMIZE = false;
	
	if (START == DMRG::COMPRESSION::RHS)
	{
		Vout = Vin;
	}
	else if (START == DMRG::COMPRESSION::RANDOM)
	{
		Vout = Vin;
		Vout.dynamicResize(DMRG::RESIZE::DECR, Dcutoff);
		if (START == DMRG::COMPRESSION::RANDOM)
		{
			RANDOMIZE = true;
		}
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
	Heff[N_sites-1].R.setTarget(qarray3<Symmetry::Nq>{Vin.Qtarget(), Vout.Qtarget(), qvacuum<Symmetry::Nq>()});
	
	Vout.N_sv = Dcutoff;
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
			prepSweep(H,Vin,Vout,RANDOMIZE);
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
		for (size_t j=1; j<=halfSweepRange; ++j)
		{
			bring_her_about(pivot, N_sites, CURRENT_DIRECTION);
			if (N_halfsweeps%4 == 0 and N_halfsweeps > 0)
			{
//				lout << "switching to two-site algorithm for this half-sweep..." << endl;
				optimizationStep2(H,Vin,Vout);
			}
			else
			{
				optimizationStep(H,Vin,Vout);
				Vout.sweepStep(CURRENT_DIRECTION, pivot, DMRG::BROOM::QR);
				pivot = Vout.get_pivot();
			}
			sweepStep(H,Vin,Vout);
//			cout << Vout.test_ortho() << endl;
//			if (j != halfSweepRange) {sweepStep(H,Vin,Vout);}
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
//			Stopwatch<> ChronosResize;
//			size_t Dcutoff_old = Vout.calc_Dmax();
//			size_t Mmax_old = Vout.calc_Mmax();
//			
//			Dcutoff_new = Dcutoff_old+1;
//			Vout.dynamicResize(DMRG::RESIZE::CONSERV_INCR, Dcutoff_new);
//			Vout.setRandom();
//			Mmax_new = Vout.calc_Mmax();
//			
//			Vout.pivot = -1;
//			prepSweep(H,Vin,Vout);
//			pivot = Vout.pivot;
//			halfSweepRange = N_sites;
//			RESIZED = true;
//			
//			if (CHOSEN_VERBOSITY>=2)
//			{
//				lout << "resize: " << Dcutoff_old << "→" << Dcutoff_new << ", M=" << Mmax_old << "→" << Mmax_new << "\t" << ChronosResize.info() << endl;
//			}
			
			Vout.N_sv += 1;
			Dcutoff_new = Vout.N_sv;
			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
			{
				lout << "resize: " << Vout.N_sv-1 << "→" << Vout.N_sv << endl;
			}
		}
		
		Mmax_new = Vout.calc_Mmax();
		
//		if ((sqdist > tol and 
//		    RESIZED == false and 
//		    N_halfsweeps < max_halfsweeps) or
//		    N_halfsweeps < min_halfsweeps)
//		{
//			sweepStep(H,Vin,Vout);
//		}
	}
	
	// mowing
	if (Vout.N_mow > 0)
	{
		mowSweeps(H,Vout);
	}
	
	// last sweep
	if      (pivot==1)         {Vout.sweep(0,DMRG::BROOM::QR);}
	else if (pivot==N_sites-2) {Vout.sweep(N_sites-1,DMRG::BROOM::QR);}
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsQCompressor<Symmetry,Scalar,MpoScalar>::
prepSweep (const MpOperator &H, const MpsQ<Symmetry,Scalar> &Vin, MpsQ<Symmetry,Scalar> &Vout, bool RANDOMIZE)
{
	assert(Vout.pivot == 0 or Vout.pivot == N_sites-1 or Vout.pivot == -1);
	
	if (Vout.pivot == N_sites-1 or Vout.pivot == -1)
	{
		for (size_t l=N_sites-1; l>0; --l)
		{
			if (RANDOMIZE == true)
			{
				Vout.setRandom(l);
//				if (l>0) {Vout.setRandom(l-1);}
			}
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
			if (RANDOMIZE == true)
			{
				Vout.setRandom(l);
//				if (l<N_sites-1) {Vout.setRandom(l+1);}
			}
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
void MpsQCompressor<Symmetry,Scalar,MpoScalar>::
sweepStep (const MpOperator &H, const MpsQ<Symmetry,Scalar> &Vin, MpsQ<Symmetry,Scalar> &Vout)
{
//	Vout.sweepStep(CURRENT_DIRECTION, pivot, DMRG::BROOM::QR);
//	(CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? build_LW(++pivot,Vout,H,Vin) : build_RW(--pivot,Vout,H,Vin);
	(CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? build_LW(pivot,Vout,H,Vin) : build_RW(pivot,Vout,H,Vin);
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsQCompressor<Symmetry,Scalar,MpoScalar>::
optimizationStep (const MpOperator &H, const MpsQ<Symmetry,Scalar> &Vin, MpsQ<Symmetry,Scalar> &Vout)
{
	Stopwatch<> Chronos;
	
	for (size_t s=0; s<Vin.locBasis(pivot).size(); ++s)
	{
		Vout.A[pivot][s].setZero();
	}
	
//	if (Heff[pivot].qlhs.size() == 0) // Doesn't work with chebCompress, why? Or does it?
	{
		Heff[pivot].W = H.W[pivot];
		precalc_blockStructure (Heff[pivot].L, Vout.A[pivot], Heff[pivot].W, Vin.A[pivot], Heff[pivot].R, 
		                        H.locBasis(pivot), H.opBasis(pivot), Heff[pivot].qlhs, Heff[pivot].qrhs, Heff[pivot].factor_cgcs);
	}
	
	// why doesn't this work?
//	PivotVectorQ<Symmetry,Scalar> Vtmp;
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
		for (size_t p=0; p<Heff[pivot].qrhs[q].size(); ++p)
		// for (auto irhs=Heff[pivot].qrhs[q].begin(); irhs!=Heff[pivot].qrhs[q].end(); ++irhs)
		{
			size_t s2 = Heff[pivot].qrhs[q][p][0];
			size_t q2 = Heff[pivot].qrhs[q][p][1];
			size_t qL = Heff[pivot].qrhs[q][p][2];
			size_t qR = Heff[pivot].qrhs[q][p][3];
			size_t k = Heff[pivot].qrhs[q][p][4];
			for (int r=0; r<H.W[pivot][s1][s2][k].outerSize(); ++r)
			for (typename SparseMatrix<MpoScalar>::InnerIterator iW(H.W[pivot][s1][s2][k],r); iW; ++iW)
			{
				if (Heff[pivot].L.block[qL][iW.row()][0].rows() != 0 and 
				    Heff[pivot].R.block[qR][iW.col()][0].rows() != 0)
				{
					Vout.A[pivot][s1].block[q1].noalias() += Heff[pivot].factor_cgcs[q][p] * iW.value() * 
					                                         (Heff[pivot].L.block[qL][iW.row()][0] * 
					                                          Vin.A[pivot][s2].block[q2] * 
					                                          Heff[pivot].R.block[qR][iW.col()][0]);
				}
			}
		}
	}
	
	if (CHOSEN_VERBOSITY == DMRG::VERBOSITY::STEPWISE)
	{
		lout << "optimization, loc=" << Chronos.info(pivot) << endl;
	}
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsQCompressor<Symmetry,Scalar,MpoScalar>::
optimizationStep2 (const MpOperator &H, const MpsQ<Symmetry,Scalar> &Vin, MpsQ<Symmetry,Scalar> &Vout)
{
	Stopwatch<> Chronos;
	
	size_t loc1 = (CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? pivot : pivot-1;
	size_t loc2 = (CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? pivot+1 : pivot;
	
	Heff[loc1].W = H.W[loc1];
	Heff[loc2].W = H.W[loc2];
	
	vector<vector<Biped<Symmetry,MatrixType> > > Apair;
	Apair.resize(Vin.locBasis(loc1).size());
	for (size_t s1=0; s1<Vin.locBasis(loc1).size(); ++s1)
	{
		Apair[s1].resize(Vin.locBasis(loc2).size());
	}
	
	for (size_t s1=0; s1<Vin.locBasis(loc1).size(); ++s1)
	for (size_t s2=0; s2<Vin.locBasis(loc1).size(); ++s2)
	for (size_t k1=0; k1<H.opBasis(loc1).size(); ++k1)
	{
		if(Heff[loc1].W[s1][s2][k1].size() == 0) { continue; }
		for (size_t qL=0; qL<Heff[loc1].L.dim; ++qL)
		{
			vector<tuple<qarray3<Symmetry::Nq>,size_t,size_t> > ix12;
			bool FOUND_MATCH12 = AWA(Heff[loc1].L.in(qL), Heff[loc1].L.out(qL), Heff[loc1].L.mid(qL), s1, s2, Vin.locBasis(loc1),
									 k1, H.opBasis(loc1), Vout.A[loc1], Vin.A[loc1], ix12);
			// bool FOUND_MATCH = AWA(Lold.in(qL), Lold.out(qL), Lold.mid(qL), s1, s2, qloc, k, qOp, Abra, Aket, ix);

			if (FOUND_MATCH12)
			{
				for(size_t n=0; n<ix12.size(); n++ )
				{
					qarray3<Symmetry::Nq> quple12 = get<0>(ix12[n]);
					swap(quple12[0], quple12[1]);
					size_t qA12 = get<2>(ix12[n]);
					for (size_t s3=0; s3<Vin.locBasis(loc2).size(); ++s3)
					for (size_t s4=0; s4<Vin.locBasis(loc2).size(); ++s4)
					for (size_t k2=0; k2<H.opBasis(loc2).size(); ++k2)
					{
						if(Heff[loc2].W[s3][s4][k2].size() == 0) { continue; }
						vector<tuple<qarray3<Symmetry::Nq>,size_t,size_t> > ix34;
						bool FOUND_MATCH34 = AWA(quple12[0], quple12[1], quple12[2], s3, s4, Vin.locBasis(loc2),
												 k2, H.opBasis(loc2), Vout.A[loc2], Vin.A[loc2], ix34);
						if (FOUND_MATCH34)
						{
							for(size_t m=0; m<ix34.size(); m++)
							{
								qarray3<Symmetry::Nq> quple34 = get<0>(ix34[m]);
								size_t qA34 = get<2>(ix34[m]);
								auto qR = Heff[loc2].R.dict.find(quple34);
					
								if (qR != Heff[loc2].R.dict.end())
								{
									if (Heff[loc1].L.mid(qL) + Vin.locBasis(loc1)[s1] - Vin.locBasis(loc1)[s2] == 
										Heff[loc2].R.mid(qR->second) - Vin.locBasis(loc2)[s3] + Vin.locBasis(loc2)[s4])
									{
										for (int r12=0; r12<Heff[loc1].W[s1][s2][k1].outerSize(); ++r12)
										for (typename SparseMatrix<MpoScalar>::InnerIterator iW12(Heff[loc1].W[s1][s2][k1],r12); iW12; ++iW12)
										for (int r34=0; r34<Heff[loc2].W[s3][s4][k2].outerSize(); ++r34)
										for (typename SparseMatrix<MpoScalar>::InnerIterator iW34(Heff[loc2].W[s3][s4][k2],r34); iW34; ++iW34)
										{
											MatrixType Mtmp;
											MpoScalar Wfactor = iW12.value() * iW34.value();
								
											if (Heff[loc1].L.block[qL][iW12.row()][0].rows() != 0 and
												Heff[loc2].R.block[qR->second][iW34.col()][0].rows() !=0 and
												iW12.col() == iW34.row())
											{
//									Mtmp = Wfactor * 
//									       (Heff[loc1].L.block[qL][iW12.row()][0] * 
//									       Vin.A[loc1][s2].block[qA12] * 
//									       Vin.A[loc2][s4].block[qA34] * 
//									       Heff[loc2].R.block[qR->second][iW34.col()][0]);
												optimal_multiply(Wfactor, 
																 Heff[loc1].L.block[qL][iW12.row()][0],
																 Vin.A[loc1][s2].block[qA12],
																 Vin.A[loc2][s4].block[qA34],
																 Heff[loc2].R.block[qR->second][iW34.col()][0],
																 Mtmp);
											}
								
											if (Mtmp.rows() != 0)
											{
												qarray2<Symmetry::Nq> qupleApair = {Heff[loc1].L.in(qL), Heff[loc2].R.out(qR->second)};
												auto qApair = Apair[s1][s3].dict.find(qupleApair);
									
												if (qApair != Apair[s1][s3].dict.end())
												{
													Apair[s1][s3].block[qApair->second] += Mtmp;
												}
												else
												{
													Apair[s1][s3].push_back(qupleApair, Mtmp);
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	
	Vout.sweepStep2(CURRENT_DIRECTION, min(loc1,loc2), Apair);
	
	if (CHOSEN_VERBOSITY == DMRG::VERBOSITY::STEPWISE)
	{
		lout << "optimization & sweep step, 2-site, loc=" << Chronos.info(pivot) << endl;
	}
	
	pivot = Vout.get_pivot();
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsQCompressor<Symmetry,Scalar,MpoScalar>::
build_LW (size_t loc, const MpsQ<Symmetry,Scalar> &Vbra, const MpOperator &H, const MpsQ<Symmetry,Scalar> &Vket)
{
	contract_L(Heff[loc-1].L, Vbra.A[loc-1], H.W[loc-1], Vket.A[loc-1], H.locBasis(loc-1), H.opBasis(loc-1), Heff[loc].L);
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsQCompressor<Symmetry,Scalar,MpoScalar>::
build_RW (size_t loc, const MpsQ<Symmetry,Scalar> &Vbra, const MpOperator &H, const MpsQ<Symmetry,Scalar> &Vket)
{
	contract_R(Heff[loc+1].R, Vbra.A[loc+1], H.W[loc+1], Vket.A[loc+1], H.locBasis(loc+1), H.opBasis(loc+1), Heff[loc].R);
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
void MpsQCompressor<Symmetry,Scalar,MpoScalar>::
energyTruncationStep (MpsQ<Symmetry,Scalar> &V, size_t dimK)
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
	
	PivotVectorQ<Symmetry,Scalar> Psi;
	Psi.A = V.A[pivot];
	
	LanczosMower<PivotMatrixQ<Symmetry,Scalar,MpoScalar>,PivotVectorQ<Symmetry,Scalar>,Scalar> Lutz(min(dimK,Heff[pivot].dim));
	Lutz.mow(Heff[pivot],Psi,2.);
	mowedWeight(pivot) = Lutz.get_mowedWeight();
	
	V.A[pivot] = Psi.A;
}

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsQCompressor<Symmetry,Scalar,MpoScalar>::
polyCompress (const MpOperator &H, const MpsQ<Symmetry,Scalar> &Vin1, double polyB, const MpsQ<Symmetry,Scalar> &Vin2, MpsQ<Symmetry,Scalar> &Vout, size_t Dcutoff_input, double tol, size_t max_halfsweeps, size_t min_halfsweeps, DMRG::COMPRESSION::INIT START)
{
	N_sites = Vin1.length();
	Stopwatch<> Chronos;
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
	Heff[N_sites-1].R.setTarget(qarray3<Symmetry::Nq>{Vin1.Qtarget(), Vout.Qtarget(), Symmetry::qvacuum()});
	
	// set L&R edges
	L.resize(N_sites);
	R.resize(N_sites);
	R[N_sites-1].setTarget(Vin2.Qtot);
	L[0].setVacuum();
	
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
	
	Mmax = Vout.calc_Mmax();
	Vout.N_sv = Dcutoff;
	
	// must achieve sqdist > tol or break off after max_halfsweeps, do at least min_halfsweeps
	while ((sqdist > tol and N_halfsweeps < max_halfsweeps) or N_halfsweeps < min_halfsweeps)
	{
		Stopwatch<> Aion;
		for (size_t j=1; j<=halfSweepRange; ++j)
		{
			bring_her_about(pivot, N_sites, CURRENT_DIRECTION);
			Stopwatch<> Chronos;
			
//			optimizationStep(Vin2,Vout);
//			auto Atmp = Vout.A[pivot];
//			optimizationStep(H,Vin1,Vout);
//			for (size_t s=0; s<H.locBasis(pivot).size(); ++s)
//			for (size_t q=0; q<Atmp[s].dim; ++q)
//			{
//				qarray2<Symmetry::Nq> quple = {Atmp[s].in[q], Atmp[s].out[q]};
//				auto it = Vout.A[pivot][s].dict.find(quple);
//				Vout.A[pivot][s].block[it->second] -= polyB * Atmp[s].block[q];
//			}
			
			if (N_halfsweeps%4 == 0 and N_halfsweeps > 0)
			{
				size_t loc1 = (CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? pivot : pivot-1;
				size_t loc2 = (CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? pivot+1 : pivot;
				
				Heff[loc1].W = H.W[loc1];
				Heff[loc2].W = H.W[loc2];
				
				// H*Vin1
				vector<vector<Biped<Symmetry,MatrixType> > > Apair;
				HxV(Heff[loc1],Heff[loc2], Vin1.A[loc1],Vin1.A[loc2], Vout.A[loc1],Vout.A[loc2], Vin1.locBasis(loc1),Vin1.locBasis(loc2), Apair);
				
				// Vin2
				vector<vector<Biped<Symmetry,MatrixType> > > ApairVV;
				ApairVV.resize(Vin1.locBasis(loc1).size());
				for (size_t s1=0; s1<Vin1.locBasis(loc1).size(); ++s1)
				{
					ApairVV[s1].resize(Vin1.locBasis(loc2).size());
				}
				
				for (size_t s1=0; s1<Vin2.locBasis(loc1).size(); ++s1)
				for (size_t s3=0; s3<Vin2.locBasis(loc2).size(); ++s3)
				{
					ApairVV[s1][s3] = L[loc1] * Vin2.A[loc1][s1] * Vin2.A[loc2][s3] * R[loc2];
				}
				
				// H*Vin1-polyB*Vin2
				for (size_t s1=0; s1<H.locBasis(loc1).size(); ++s1)
				for (size_t s3=0; s3<H.locBasis(loc2).size(); ++s3)
				for (size_t q=0; q<Apair[s1][s3].dim; ++q)
				{
					qarray2<Symmetry::Nq> quple = {Apair[s1][s3].in[q], Apair[s1][s3].out[q]};
					auto it = ApairVV[s1][s3].dict.find(quple);
					
					MatrixType Mtmp = Apair[s1][s3].block[q];
					size_t rows = max(Apair[s1][s3].block[q].rows(), Apair[s1][s3].block[q].rows());
					size_t cols = max(Apair[s1][s3].block[q].cols(), Apair[s1][s3].block[q].cols());
					
					Apair[s1][s3].block[q].resize(rows,cols);
					Apair[s1][s3].block[q].setZero();
					Apair[s1][s3].block[q].topLeftCorner(Mtmp.rows(),Mtmp.cols()) = Mtmp;
					Apair[s1][s3].block[q].topLeftCorner(Apair[s1][s3].block[q].rows(),Apair[s1][s3].block[q].cols()) -= polyB * ApairVV[s1][s3].block[it->second];
				}
				
				Vout.sweepStep2(CURRENT_DIRECTION, min(loc1,loc2), Apair);
				pivot = Vout.get_pivot();
			}
			else
			{
				optimizationStep(Vin2,Vout);
				auto Atmp = Vout.A[pivot];
				
				optimizationStep(H,Vin1,Vout);
				
				for (size_t s=0; s<H.locBasis(pivot).size(); ++s)
				for (size_t q=0; q<Atmp[s].dim; ++q)
				{
					qarray2<Symmetry::Nq> quple = {Atmp[s].in[q], Atmp[s].out[q]};
					auto it = Vout.A[pivot][s].dict.find(quple);
					Vout.A[pivot][s].block[it->second] -= polyB * Atmp[s].block[q];
				}
				
				Vout.sweepStep(CURRENT_DIRECTION, pivot, DMRG::BROOM::QR);
				pivot = Vout.get_pivot();
			}
			
			sweepStep(H,Vin1,Vin2,Vout);
//			if (j != halfSweepRange)
//			{
//				sweepStep(H,Vin1,Vin2,Vout);
//			}
			++N_sweepsteps;
		}
		halfSweepRange = N_sites-1;
		++N_halfsweeps;
		
		sqdist = abs(sqnormV1 - Vout.squaredNorm() + polyB*polyB*sqnormV2 - 2.*polyB*overlapV12);
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
//			Stopwatch<> ChronosResize;
//			size_t Dcutoff_old = Vout.calc_Dmax();
//			size_t Mmax_old = Vout.calc_Mmax();
//			
//			Dcutoff_new = Dcutoff_old+1;
//			Vout.dynamicResize(DMRG::RESIZE::CONSERV_INCR, Dcutoff_new);
//			Vout.setRandom();
//			Mmax_new = Vout.calc_Mmax();
//			
//			Vout.pivot = -1;
//			prepSweep(H,Vin1,Vin2,Vout);
//			pivot = Vout.pivot;
//			halfSweepRange = N_sites;
//			RESIZED = true;
//			
//			if (CHOSEN_VERBOSITY>=2)
//			{
//				lout << "resize: " << Dcutoff_old << "→" << Dcutoff_new << ", M=" << Mmax_old << "→" << Mmax_new << "\t" << ChronosResize.info() << endl;
//			}
			
			Vout.N_sv += 1;
			Dcutoff_new = Vout.N_sv;
			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
			{
				lout << "resize: " << Vout.N_sv-1 << "→" << Vout.N_sv << endl;
			}
		}
		
		Mmax_new = Vout.calc_Mmax();
		
//		if ((sqdist > tol and 
//		    RESIZED == false and
//		    N_halfsweeps < max_halfsweeps) or 
//		    N_halfsweeps < min_halfsweeps)
//		{
//			sweepStep(H,Vin1,Vin2,Vout);
//		}
	}
	
	// last sweep
	if      (pivot==1)         {Vout.sweep(0,DMRG::BROOM::QR);}
	else if (pivot==N_sites-2) {Vout.sweep(N_sites-1,DMRG::BROOM::QR);}
}

//template<typename Symmetry, typename Scalar, typename MpoScalar>
//void MpsQCompressor<Symmetry,Scalar,MpoScalar>::
//sumCompress (const vector<MpsQ<Symmetry,Scalar> > &Vin, const vector<double> &factor, MpsQ<Symmetry,Scalar> &Vout, 
//size_t Dcutoff_input, double tol, size_t max_halfsweeps, size_t min_halfsweeps, DMRG::COMPRESSION::INIT START)
//{
//	N_sites = Vin1.length();
//	Stopwatch<> Chronos;
//	N_halfsweeps = 0;
//	N_sweepsteps = 0;
//	Dcutoff = Dcutoff_input;
//	Dcutoff_new = Dcutoff_input;
//	
//	if (START == DMRG::COMPRESSION::RHS)
//	{
//		Vout = Vin[0];
//	}
//	else if (START == DMRG::COMPRESSION::RANDOM)
//	{
//		Vout = Vin[0];
//		Vout.dynamicResize(DMRG::RESIZE::DECR, Dcutoff);
//		Vout.setRandom();
//	}
//	
//	// set L&R edges
//	L.resize(N_sites);
//	R.resize(N_sites);
//	R[N_sites-1].setTarget(Vin2.Qtot);
//	L[0].setVacuum();
//	
//	double sqnormV1, sqnormV2, overlapV12;
//	sqnormV2 = Vin2.squaredNorm();
//	#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
//	#pragma omp parallel sections
//	#endif
//	{
//		#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
//		#pragma omp section
//		#endif
//		{
//			sqnormV1 = (H.check_SQUARE()==true)? isReal(avg(Vin1,H,Vin1,true)) : isReal(avg(Vin1,H,H,Vin1));
//		}
//		#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
//		#pragma omp section
//		#endif
//		{
//			overlapV12 = isReal(avg(Vin2,H,Vin1));
//		}
//		#ifndef MPSQCOMPRESSOR_DONT_USE_OPENMP
//		#pragma omp section
//		#endif
//		{
//			prepSweep(H,Vin1,Vin2,Vout);
//		}
//	}
//	sqdist = 1.;
//	size_t halfSweepRange = N_sites;
//	
//	if (CHOSEN_VERBOSITY>=2)
//	{
//		lout << Chronos.info("preparation") << endl;
//	}
//	
//	Mmax = Vout.calc_Mmax();
//	Vout.N_sv = Dcutoff;
//	
//	// must achieve sqdist > tol or break off after max_halfsweeps, do at least min_halfsweeps
//	while ((sqdist > tol and N_halfsweeps < max_halfsweeps) or N_halfsweeps < min_halfsweeps)
//	{
//		Stopwatch<> Aion;
//		for (size_t j=1; j<=halfSweepRange; ++j)
//		{
//			bring_her_about(pivot, N_sites, CURRENT_DIRECTION);
//			Stopwatch<> Chronos;
//			
//			if (N_halfsweeps%4 == 0 and N_halfsweeps > 0)
//			{
//				size_t loc1 = (CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? pivot : pivot-1;
//				size_t loc2 = (CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? pivot+1 : pivot;
//				
//				Heff[loc1].W = H.W[loc1];
//				Heff[loc2].W = H.W[loc2];
//				
//				// H*Vin1
//				vector<vector<Biped<Symmetry,MatrixType> > > Apair;
//				HxV(Heff[loc1],Heff[loc2], Vin1.A[loc1],Vin1.A[loc2], Vout.A[loc1],Vout.A[loc2], Vin1.locBasis(loc1),Vin1.locBasis(loc2), Apair);
//				
//				// Vin2
//				vector<vector<Biped<Symmetry,MatrixType> > > ApairVV;
//				ApairVV.resize(Vin1.locBasis(loc1).size());
//				for (size_t s1=0; s1<Vin1.locBasis(loc1).size(); ++s1)
//				{
//					ApairVV[s1].resize(Vin1.locBasis(loc2).size());
//				}
//				
//				for (size_t s1=0; s1<Vin2.locBasis(loc1).size(); ++s1)
//				for (size_t s3=0; s3<Vin2.locBasis(loc2).size(); ++s3)
//				{
//					ApairVV[s1][s3] = L[loc1] * Vin2.A[loc1][s1] * Vin2.A[loc2][s3] * R[loc2];
//				}
//				
//				// H*Vin1-B*Vin2
//				for (size_t s1=0; s1<H.locBasis(loc1).size(); ++s1)
//				for (size_t s3=0; s3<H.locBasis(loc2).size(); ++s3)
//				for (size_t q=0; q<Apair[s1][s3].dim; ++q)
//				{
//					qarray2<Symmetry::Nq> quple = {Apair[s1][s3].in[q], Apair[s1][s3].out[q]};
//					auto it = ApairVV[s1][s3].dict.find(quple);
//					
//					MatrixType Mtmp = Apair[s1][s3].block[q];
//					size_t rows = max(Apair[s1][s3].block[q].rows(), Apair[s1][s3].block[q].rows());
//					size_t cols = max(Apair[s1][s3].block[q].cols(), Apair[s1][s3].block[q].cols());
//					
//					Apair[s1][s3].block[q].resize(rows,cols);
//					Apair[s1][s3].block[q].setZero();
//					Apair[s1][s3].block[q].topLeftCorner(Mtmp.rows(),Mtmp.cols()) = Mtmp;
//					Apair[s1][s3].block[q].topLeftCorner(Apair[s1][s3].block[q].rows(),Apair[s1][s3].block[q].cols()) -= polyB * ApairVV[s1][s3].block[it->second];
//				}
//				
//				Vout.sweepStep2(CURRENT_DIRECTION, min(loc1,loc2), Apair);
//				pivot = Vout.get_pivot();
//			}
//			else
//			{
//				optimizationStep(Vin2,Vout);
//				auto Atmp = Vout.A[pivot];
//				
//				optimizationStep(H,Vin1,Vout);
//				
//				for (size_t s=0; s<H.locBasis(pivot).size(); ++s)
//				for (size_t q=0; q<Atmp[s].dim; ++q)
//				{
//					qarray2<Symmetry::Nq> quple = {Atmp[s].in[q], Atmp[s].out[q]};
//					auto it = Vout.A[pivot][s].dict.find(quple);
//					Vout.A[pivot][s].block[it->second] -= polyB * Atmp[s].block[q];
//				}
//				
//				Vout.sweepStep(CURRENT_DIRECTION, pivot, DMRG::BROOM::QR);
//				pivot = Vout.get_pivot();
//			}
//			
//			sweepStep(H,Vin1,Vin2,Vout);
//			++N_sweepsteps;
//		}
//		halfSweepRange = N_sites-1;
//		++N_halfsweeps;
//		
//		sqdist = abs(sqnormV1 - Vout.squaredNorm() + polyB*polyB*sqnormV2 - 2.*polyB*overlapV12);
//		assert(!std::isnan(sqdist));
//		
//		if (CHOSEN_VERBOSITY>=2)
//		{
//			lout << Aion.info("half-sweep") << "\tdistance^2=" << sqdist << endl;
//		}
//		
//		bool RESIZED = false;
//		if (N_halfsweeps%4 == 0 and 
//		    N_halfsweeps > 0 and 
//		    N_halfsweeps != max_halfsweeps and
//		    sqdist > tol)
//		{
//			Vout.N_sv += 1;
//			Dcutoff_new = Vout.N_sv;
//			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
//			{
//				lout << "resize: " << Vout.N_sv-1 << "→" << Vout.N_sv << endl;
//			}
//		}
//		
//		Mmax_new = Vout.calc_Mmax();
//	}
//	
//	// last sweep
//	if      (pivot==1)         {Vout.sweep(0,DMRG::BROOM::QR);}
//	else if (pivot==N_sites-2) {Vout.sweep(N_sites-1,DMRG::BROOM::QR);}
//}

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsQCompressor<Symmetry,Scalar,MpoScalar>::
mowSweeps (const MpOperator &H, MpsQ<Symmetry,Scalar> &Vout)
{
//	mowedWeight.resize(N_sites);
//	
//	Heff.clear();
//	Heff.resize(N_sites);
//	Heff[0].L.setVacuum();
//	Heff[N_sites-1].R.setTarget(qarray3<Symmetry::Nq>{Vout.Qtarget(), Vout.Qtarget(), qvacuum<Symmetry::Nq>()});
//	for (size_t l=0; l<N_sites; ++l) {Heff[l].W = H.W[l];}
//	
//	// preparation
//	Stopwatch<> Aion;
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
//		Stopwatch<> Aion;
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

template<typename Symmetry, typename Scalar, typename MpoScalar>
template<typename MpOperator>
void MpsQCompressor<Symmetry,Scalar,MpoScalar>::
prepSweep (const MpOperator &H, const MpsQ<Symmetry,Scalar> &Vin1, const MpsQ<Symmetry,Scalar> &Vin2, MpsQ<Symmetry,Scalar> &Vout, bool RANDOMIZE)
{
	assert(Vout.pivot == 0 or Vout.pivot == N_sites-1 or Vout.pivot == -1);
	
	if (Vout.pivot == N_sites-1)
	{
		for (size_t l=N_sites-1; l>0; --l)
		{
			if (RANDOMIZE == true)
			{
				Vout.setRandom(l);
//				if (l>0) {Vout.setRandom(l-1);}
			}
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
			if (RANDOMIZE == true)
			{
				Vout.setRandom(l);
//				if (l<N_sites-1) {Vout.setRandom(l+1);}
			}
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
void MpsQCompressor<Symmetry,Scalar,MpoScalar>::
sweepStep (const MpOperator &H, const MpsQ<Symmetry,Scalar> &Vin1, const MpsQ<Symmetry,Scalar> &Vin2, MpsQ<Symmetry,Scalar> &Vout)
{
//	Vout.sweepStep(CURRENT_DIRECTION, pivot, DMRG::BROOM::QR);
//	(CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? ++pivot : --pivot;
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
