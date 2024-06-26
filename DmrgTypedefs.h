#ifndef DMRG_TYPEDEFS
#define DMRG_TYPEDEFS

/// \cond
#include <array>
#include <vector>
#include <functional>
#include <complex>
/// \endcond

#include "symmetry/qarray.h"

//include "tensors/SiteOperatorQ.h"
//include "tensors/SiteOperator.h"

//include "LanczosSolver.h" // from ALGS
#include "LanczosTypedefs.h"

#ifndef IS_REAL_FUNCTION
#define IS_REAL_FUNCTION
inline double isReal (double x) {return x;}
inline double isReal (std::complex<double> x) {return x.real();}
#endif

#ifndef CONJ_IF_COMPLEX
#define CONJ_IF_COMPLEX
inline double conjIfcomplex (double x) {return x;}
inline std::complex<double> conjIfcomplex (std::complex<double> x) {return conj(x);}
#endif

#define EIGEN_INITIALIZE_MATRICES_BY_ZERO

#ifndef SPIN_INDEX_ENUM
#define SPIN_INDEX_ENUM
	enum SPIN_INDEX
	{
		UP=false, /**<spin up*/
		DN=true, /**<spin down*/
		NOSPIN=2, /**<no spin (for consistency, also useful for iterations)*/
		UPDN=3 /**<both up and down (for consistency)*/
	};
	
	SPIN_INDEX operator! (const SPIN_INDEX sigma)
	{
		assert(sigma==UP or sigma==DN);
		return (sigma==UP) ? DN : UP;
	}
	//string spin_index_strings[] = {"UP","DN","NO","UPDN"};
	
	std::ostream& operator<< (std::ostream& s, SPIN_INDEX sigma)
	{
		if      (sigma==UP)     {s << "↑";}
		else if (sigma==DN)     {s << "↓";}
		else if (sigma==NOSPIN) {s << "↯";}
		else if (sigma==UPDN)   {s << "⇅";}
		return s;
	}
#endif

enum SPINOP_LABEL {SX, SY, iSY, SZ, SP, SM, QZ, QP, QM, QPZ, QMZ};

std::ostream& operator<< (std::ostream& s, SPINOP_LABEL Sa)
{
	if      (Sa==SX)  {s << "Sx";}
	else if (Sa==SY)  {s << "Sy";}
	else if (Sa==iSY) {s << "iSy";}
	else if (Sa==SZ)  {s << "Sz";}
	else if (Sa==SP)  {s << "S+";}
	else if (Sa==SM)  {s << "S-";}
	else if (Sa==QZ)  {s << "Qz";}
	else if (Sa==QP)  {s << "Q+";}
	else if (Sa==QM)  {s << "Q-";}
	else if (Sa==QPZ)  {s << "Q+z";}
	else if (Sa==QMZ)  {s << "Q-z";}
	return s;
}

enum STRING {NOSTRING, STRINGX, STRINGY, STRINGZ};

std::ostream& operator<< (std::ostream& s, STRING STR)
{
	if      (STR==NOSTRING) {s << "NOSTRING";}
	else if (STR==STRINGX)  {s << "STRINGX";}
	else if (STR==STRINGY)  {s << "STRINGY";}
	else if (STR==STRINGZ)  {s << "STRINGZ";}
	return s;
}

SPINOP_LABEL STRING_TO_SPINOP (STRING STR)
{
	if      (STR==STRINGX) {return SX;}
	else if (STR==STRINGY) {return iSY;}
	return SZ;
}

enum MODEL_FAMILY {HEISENBERG, HUBBARD, KONDO, SPINLESS};

std::ostream& operator<< (std::ostream& s, MODEL_FAMILY mf)
{
	if      (mf==HEISENBERG) {s << "HEISENBERG";}
	else if (mf==HUBBARD)    {s << "HUBBARD";}
	else if (mf==KONDO)      {s << "KONDO";}
	else if (mf==SPINLESS)   {s << "SPINLESS";}
	return s;
}

#ifndef KIND_ENUM
#define KIND_ENUM
namespace Sym{
	enum KIND {S,Salt,T,N,M,Nup,Ndn,Nparity,K};
}
#endif

std::ostream& operator<< (std::ostream& s, Sym::KIND l)
{
	if      (l==Sym::KIND::S)       {s << "S";}
	if      (l==Sym::KIND::Salt)    {s << "Salt";}
	else if (l==Sym::KIND::T)       {s << "T";}
	else if (l==Sym::KIND::N)       {s << "N";}
	else if (l==Sym::KIND::M)       {s << "M";}
	else if (l==Sym::KIND::Nup)     {s << "N↑";}
	else if (l==Sym::KIND::Ndn)     {s << "N↓";}
	else if (l==Sym::KIND::Nparity) {s << "P";}
	else if (l==Sym::KIND::K)       {s << "K";}
	return s;
}

enum PARITY {EVEN=0, ODD=1};

enum SUB_LATTICE {A=1,B=-1};

enum KONDO_SUBSYSTEM {IMP, SUB, IMPSUB};

std::ostream& operator<< (std::ostream& s, SUB_LATTICE sublat)
{
	if      (sublat==A)  {s << "A";}
	else if (sublat==B)  {s << "B";}
	return s;
}

SUB_LATTICE flip_sublattice(SUB_LATTICE sublat)
{
	return (sublat==A)? B:A;
}

void sublattice_check (const ArrayXXd &hopping, const vector<SUB_LATTICE> &G)
{
	for (size_t i=0; i<hopping.rows(); ++i)
	for (size_t j=0; j<hopping.cols(); ++j)
	{
		if (hopping(i,j)!=0.)
		{
			if (G[i]==G[j]) lout << "sublattice error: G[" << i << "]=" << G[i] << ", G[" << j << "]=" << G[j] << endl;
			assert(G[i]!=G[j] and "Sublattice check");
		}
	}
	lout << "Sublattice check passed!" << endl;
}

enum BC
{
	// PERIODIC=true,
	OPEN=true,
	INFINITE=false,
};

std::ostream& operator<< (std::ostream& s, BC boundary)
{
	// if      (boundary==BC::PERIODIC) {s << "periodic (finite system)";}
	if      (boundary==BC::OPEN)     {s << "open";}
	else if (boundary==BC::INFINITE) {s << "infinite";}
	return s;
}

/// \cond
#include <Eigen/Dense>
#include <Eigen/SparseCore>
/// \endcond

#ifndef EIGEN_DEFAULT_SPARSE_INDEX_TYPE
#define EIGEN_DEFAULT_SPARSE_INDEX_TYPE int
#endif
using namespace Eigen;
typedef SparseMatrix<double,ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> SparseMatrixXd;
typedef SparseMatrix<std::complex<double>,ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> SparseMatrixXcd;

template<typename Operator, typename Scalar>
struct PushType
{
	std::vector<std::tuple<std::size_t, std::vector<Operator>, Scalar>> data;
	
	template<typename OtherOperator>
	void push_back(const std::tuple<std::size_t, std::vector<OtherOperator>, Scalar> & elem)
	{
		if( std::abs(std::get<2>(elem) ) != 0 )
		{
			std::vector<Operator> plainOps;
			for (auto & op: std::get<1>(elem)) {plainOps.push_back(op.template plain<typename OtherOperator::Scalar>());}
			std::tuple<std::size_t, std::vector<Operator>, Scalar> plainElem;
			std::get<0>(plainElem) = std::get<0>(elem);
			std::get<1>(plainElem) = plainOps;
			std::get<2>(plainElem) = std::get<2>(elem);
			data.push_back(plainElem);
		}
	}
	
	void push_back(const std::tuple<std::size_t, std::vector<Operator>, Scalar> & elem) {if( std::abs(std::get<2>(elem) ) != 0 ) {data.push_back(elem);}}
	
	std::tuple<std::size_t, std::vector<Operator>, Scalar> operator[] ( std::size_t i ) const {return data[i];}
	std::tuple<std::size_t, std::vector<Operator>, Scalar>& operator[] ( std::size_t i ) {return data[i];}
	
	std::size_t size() const {return data.size();}
	
	template<typename OtherOperator, typename OtherScalar> PushType<OtherOperator,OtherScalar> cast()
	{
		PushType<OtherOperator,OtherScalar> out;
		for (size_t i=0; i<size(); i++)
		{
			std::vector<OtherOperator> otherOps(std::get<1>(data[i]).size());
			for (size_t j=0; j<std::get<1>(data[i]).size(); j++) {otherOps[j] = std::get<1>(data[i]).at(j).template cast<typename OtherOperator::Scalar>();}
			// OtherScalar otherCoupling = static_cast<OtherScalar>(std::get<2>(data[i]));
			out.push_back(make_tuple(std::get<0>(data[i]), otherOps, std::get<2>(data[i])));
		}
		return out;
	}
};

namespace VMPS
{
	struct DIRECTION
	{
		enum OPTION
		{
			LEFT, /**<sweep to the left, obviously*/
			RIGHT /**<sweep to the right, duh*/
		};
	};
	
	std::ostream& operator<< (std::ostream& s, VMPS::DIRECTION::OPTION DIR)
	{
		if (DIR==VMPS::DIRECTION::LEFT) {s << "LEFT";}
		else                            {s << "RIGHT";}
		return s;
	}
}
/**Namespace imitation for various enums.*/
struct DMRG
{
	/**Direction of the sweep.*/
	struct DIRECTION
	{
		enum OPTION
		{
			LEFT, /**<sweep to the left, obviously*/
			RIGHT /**<sweep to the right, duh*/
		};
	};
	/**Choice of tool to perfom the sweeping.*/
	struct BROOM
	{
		enum OPTION
		{
			QR, /**<Uses the QR decomposition. It is the fastest of all options, but preserves the matrix sizes, so that neither truncation nor growth can occur. Since the singular values are not computed, no entropy can be calculated and is therefore set to \p nan.*/
			SVD, /**<Uses the singular value decomposition. All singular values below eps_truncWeight are cut off.
			\warning Needs Lapack. Define DONT_USE_LAPACK_SVD to use Eigen's SVD routine. It is 3-90 times slower, but probably accurate and stable as hell.*/
			BRUTAL_SVD, /**<Uses the singular value decomposition, but imposes a limit on the matrix sizes. Only the first DmrgJanitor::N_sv singular values per symmetry block are kept. This option can be useful to generate starting points for variational procedures. \warning Needs Lapack (see above).*/
			RDM, /**<Uses diagonalization of the reduced density matrix. Adds a noise term multiplied by DmrgJanitor::eps_noise. All eigenvalues smaller than DmrgJanitor::eps_rdm are cut off. (Note that the eigenvalues are the singular values squared, so DmrgJanitor::eps_noise must be chosen accordingly).*/
			RICH_SVD, /**<Uses an enrichment scheme, a.k.a.\ the "Strictly Single-Site (SSS) algorithm" according to Hubig et al.\ (2015). The matrices are enlarged with non-local information times DmrgJanitor::eps_rsvd, then the usual SVD is performed (using DmrgJanitor::eps_truncWeight as a cutoff parameter). Note that DmrgJanitor::eps_rsvd is a much less sensitive parameter than DmrgJanitor::eps_rdm and can be even kept equal to 1 (but this will result in larger than optimal matrix sizes). \warning Needs Lapack (see above).*/
			QR_NULL /**Computes the null space only using full QR decomposition.*/
		};
	};
	
	/**Choice of verbosity for DmrgSolver and MpsCompressor.*/
	struct VERBOSITY
	{
		enum OPTION
		{
			SILENT=0, /**<level 0, prints no info*/
			ON_EXIT=1, /**<level 1, prints summary upon finishing*/
			HALFSWEEPWISE=2, /**<level 2, prints info every half-sweep or so*/
			STEPWISE=3, /**<level 3, prints as much info as possible*/
		};
	};
	
	/**Choice of initial guess for variational compression in MpsCompressor.*/
	struct COMPRESSION
	{
		enum INIT
		{
			RANDOM, /**<use random initial guess*/
			RHS, /**<use something from the right-hand side MPS as initial guess*/
			BRUTAL_SVD, /**<perform SVD to get the initial guess, cut given by Dcutoff*/
			RHS_SVD /**<perform SVD to get the initial guess, cut given by the right-hand side MPS*/
		};
	};
	
	/**Choice of how to resize in MpsCompressor.*/
	struct RESIZE
	{
		enum OPTION
		{
			CONSERV_INCR, /**<conservative increase: add a row and column of zeros for each subspace block, the bond dimension increases by \p Nqmax*/
			DECR /**<non-conservative decrease: cut all blocks according to a given \p Dmax*/
		};
	};
	
	/**Choice of how to check convergence of a state.*/
	struct CONVTEST
	{
		enum OPTION
		{
			VAR_2SITE, /**<Most efficient algorithm: the two-site variance as proposed in Hubig, Haegeman, Schollwöck (PRB 97, 2018), arXiv:1711.01104.*/
			VAR_HSQ, /**<Full variance of the energy: \f$\langle H^2\rangle-\langle H\rangle^2\f$.*/
			NORM_TEST, /**< Overlap to state from previous calculation. Not a very good measure.*/
			VAR_FULL /**< This computes the norm of the full resolvent: \f$\left\Vert H |\Psi\rangle - E |\Psi\rangle \right\Vert \f$.*/
		};
	};
	
	/**Iteration to perform in the solver.*/
	struct ITERATION
	{
		enum OPTION
		{
			ZERO_SITE=0, /**<Zero site algorithm (center matrix formalism)*/
			ONE_SITE=1,  /**<One site algorithm (Ac is optimized)*/
			TWO_SITE=2   /**<Two site algorithm (Very surprising)*/
		};
	};
	
	/**Default configuration values for various solvers.*/
	struct CONTROL
	{
		struct DEFAULT
		{
			//GLOB DEFAULTS
			constexpr static size_t min_halfsweeps = 6;
			constexpr static size_t max_halfsweeps = 20;
			constexpr static double tol_eigval = 1e-6;
			constexpr static double tol_state = 1e-5;
			constexpr static size_t Minit = 1;
			constexpr static size_t Mlimit = 1000;
			constexpr static int Qinit = 1; // Qinit=-1 resizes with all possible blocks
			constexpr static size_t savePeriod = 0;
			constexpr static char saveName[] = "MpsBackup";
			constexpr static DMRG::CONVTEST::OPTION CONVTEST = DMRG::CONVTEST::VAR_2SITE;
			constexpr static bool CALC_S_ON_EXIT = false;
			constexpr static bool CALC_ERR_ON_EXIT = true;
			constexpr static DMRG::DIRECTION::OPTION INITDIR = DMRG::DIRECTION::RIGHT;
			constexpr static double falphamin = 0.1;
			constexpr static double falphamax = 2.;
			
			#ifndef DMRG_CONTROL_DEFAULT_MIN_NSV
			#define DMRG_CONTROL_DEFAULT_MIN_NSV 0
			#endif
			
			//DYN DEFAULTS
			static double max_alpha_rsvd             (size_t i) {return (i<11)? 1e2:0;}
			static double min_alpha_rsvd             (size_t i) {return (i<11)? 1e-11:0;}
			static double eps_svd                    (size_t i) {return 1e-14;}
			static double eps_truncWeight            (size_t i) {return 0.;}
			static size_t Mincr_abs                  (size_t i) {return 20;} // increase M by at least 20
			static double Mincr_rel                  (size_t i) {return 1.1;} // increase M by at least 10%
			static size_t Mincr_per                  (size_t i) {return 2;} // increase M every 2 half-sweeps
			static size_t min_Nsv                    (size_t i) {return DMRG_CONTROL_DEFAULT_MIN_NSV;}
			static int    max_Nrich                  (size_t i) {return -1;} // -1 = use all
			static void   doSomething                (size_t i) {return;}
			static DMRG::ITERATION::OPTION iteration (size_t i) {return DMRG::ITERATION::ONE_SITE;}
			
			//LANCZOS DEFAULTS
			constexpr static ::LANCZOS::REORTHO::OPTION REORTHO = LANCZOS::REORTHO::FULL;
			constexpr static double tol_eigval_Lanczos = 1e-8;
			constexpr static double tol_state_Lanczos  = 1e-5;
			constexpr static size_t dimK               = 500ul;
		};
		
		struct GLOB
		{
			size_t min_halfsweeps           = CONTROL::DEFAULT::min_halfsweeps;
			size_t max_halfsweeps           = CONTROL::DEFAULT::max_halfsweeps;
			double tol_eigval               = CONTROL::DEFAULT::tol_eigval;
			double tol_state                = CONTROL::DEFAULT::tol_state;
			size_t Minit                    = CONTROL::DEFAULT::Minit;
			size_t Mlimit                   = CONTROL::DEFAULT::Mlimit;
			size_t Qinit                    = CONTROL::DEFAULT::Qinit;
			size_t savePeriod               = CONTROL::DEFAULT::savePeriod;
			std::string saveName            = std::string(CONTROL::DEFAULT::saveName);
			DMRG::CONVTEST::OPTION CONVTEST = CONTROL::DEFAULT::CONVTEST;
			bool CALC_S_ON_EXIT             = CONTROL::DEFAULT::CALC_S_ON_EXIT;
			bool CALC_ERR_ON_EXIT           = CONTROL::DEFAULT::CALC_ERR_ON_EXIT;
			DMRG::DIRECTION::OPTION INITDIR = CONTROL::DEFAULT::INITDIR;
			double falphamin                = CONTROL::DEFAULT::falphamin;
			double falphamax                = CONTROL::DEFAULT::falphamax;
		};
		
		struct DYN
		{
			function<double(size_t)> max_alpha_rsvd             = CONTROL::DEFAULT::max_alpha_rsvd;
			function<double(size_t)> min_alpha_rsvd             = CONTROL::DEFAULT::min_alpha_rsvd;
			function<double(size_t)> eps_svd                    = CONTROL::DEFAULT::eps_svd;
			function<double(size_t)> eps_truncWeight            = CONTROL::DEFAULT::eps_truncWeight;
			function<size_t(size_t)> Mincr_abs                  = CONTROL::DEFAULT::Mincr_abs;
			function<double(size_t)> Mincr_rel                  = CONTROL::DEFAULT::Mincr_rel;
			function<size_t(size_t)> Mincr_per                  = CONTROL::DEFAULT::Mincr_per;
			function<size_t(size_t)> min_Nsv                    = CONTROL::DEFAULT::min_Nsv;
			function<int(size_t)> max_Nrich                     = CONTROL::DEFAULT::max_Nrich;
			function<void(size_t)> doSomething                  = CONTROL::DEFAULT::doSomething;
			function<DMRG::ITERATION::OPTION(size_t)> iteration = CONTROL::DEFAULT::iteration;
		};
		
		struct LOC
		{
			::LANCZOS::REORTHO::OPTION REORTHO   = CONTROL::DEFAULT::REORTHO;
			double tol_eigval                    = CONTROL::DEFAULT::tol_eigval_Lanczos;
			double tol_state                     = CONTROL::DEFAULT::tol_state_Lanczos;
			size_t dimK                          = CONTROL::DEFAULT::dimK;
		};
	};
};

std::ostream& operator<< (std::ostream& s, DMRG::VERBOSITY::OPTION VERB)
{
	if      (VERB==DMRG::VERBOSITY::SILENT)        {s << "SILENT";}
	else if (VERB==DMRG::VERBOSITY::ON_EXIT)       {s << "ON_EXIT";}
	else if (VERB==DMRG::VERBOSITY::HALFSWEEPWISE) {s << "HALFSWEEPWISE";}
	else if (VERB==DMRG::VERBOSITY::STEPWISE)      {s << "STEPWSIE";}
	return s;
}

std::ostream& operator<< (std::ostream& s, DMRG::CONVTEST::OPTION TEST)
{
	if      (TEST==DMRG::CONVTEST::VAR_2SITE) {s << "2-site variance";}
	else if (TEST==DMRG::CONVTEST::VAR_HSQ)   {s << "|<H²>-<H>²|/L";}
	else if (TEST==DMRG::CONVTEST::NORM_TEST) {s << "norm comparison";}
	else if (TEST==DMRG::CONVTEST::VAR_FULL)  {s << "‖H|Ψ>-E|Ψ>‖/L";}
	return s;
}

inline std::istream & operator>>(std::istream & str, DMRG::VERBOSITY::OPTION &VERB)
{
	size_t verb = 0;
	if (str >> verb) { VERB = static_cast<DMRG::VERBOSITY::OPTION>(verb);}
	return str;
}

std::ostream& operator<< (std::ostream& s, DMRG::DIRECTION::OPTION DIR)
{
	if (DIR==DMRG::DIRECTION::LEFT) {s << "LEFT";}
	else                            {s << "RIGHT";}
	return s;
}

std::ostream& operator<< (std::ostream& s, DMRG::ITERATION::OPTION ITER)
{
	if (ITER==DMRG::ITERATION::ZERO_SITE)     {s << "0-site";}
	else if (ITER==DMRG::ITERATION::ONE_SITE) {s << "1-site";}
	else if (ITER==DMRG::ITERATION::TWO_SITE) {s << "2-site";}
	return s;
}

struct refEnergy
{
	double value = std::nan("0");
	string source = "unknown";
	string method = "unknown";
};

std::ostream& operator<< (std::ostream& s, refEnergy r)
{
	s << r.value << " (" << r.source << ")" << " [" << r.method << "]";
	return s;
}

template<typename Symmetry, typename Scalar>
struct TwoSiteData
{
	TwoSiteData(){};
	
	TwoSiteData (std::array<size_t,6> s, qarray2<Symmetry::Nq> qm, std::array<size_t,2> k, qarray<Symmetry::Nq> qOp_, Scalar cgc9_)
	:s1(s[0]), s2(s[1]), s3(s[2]), s4(s[3]), s1s3(s[4]), s2s4(s[5]), qmerge13(qm[0]), qmerge24(qm[1]), k12(k[0]), k34(k[1]), qOp(qOp_), cgc9(cgc9_)
	{};
	
	size_t s1, s2, s3, s4;
	size_t s1s3, s2s4;
	qarray<Symmetry::Nq> qmerge13, qmerge24;
	size_t k12, k34;
	qarray<Symmetry::Nq> qOp;
	Scalar cgc9 = 1.;
};

namespace PROP
{
	const bool HERMITIAN = true;
	const bool NON_HERMITIAN = false;
	const bool UNITARY = true;
	const bool NON_UNITARY = false;
	const bool FERMIONIC = true;
	const bool NON_FERMIONIC = false;
	const bool BOSONIC = false;
	const bool COMPRESS = true;
	const bool DONT_COMPRESS = false;
}

#endif
