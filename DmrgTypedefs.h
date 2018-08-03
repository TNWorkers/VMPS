#ifndef DMRG_TYPEDEFS
#define DMRG_TYPEDEFS

#include <array>
#include <vector>
#include <functional>

#include "tensors/SiteOperatorQ.h"
#include "tensors/SiteOperator.h"

#include "LanczosSolver.h" // from ALGS

#ifndef IS_REAL_FUNCTION
#define IS_REAL_FUNCTION
inline double isReal (double x) {return x;}
inline double isReal (std::complex<double> x) {return x.real();}
#endif

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

enum SPINOP_LABEL {SX, SY, iSY, SZ, SP, SM};

std::ostream& operator<< (std::ostream& s, SPINOP_LABEL Sa)
{
	if      (Sa==SX)  {s << "Sx";}
	else if (Sa==SY)  {s << "Sy";}
	else if (Sa==iSY) {s << "iSy";}
	else if (Sa==SZ)  {s << "Sz";}
	else if (Sa==SP)  {s << "S+";}
	else if (Sa==SM)  {s << "S-";}
	return s;
}

#ifndef KIND_ENUM
#define KIND_ENUM
namespace Sym{
	enum KIND {S,T,N,M,Nup,Ndn};
}
#endif

std::ostream& operator<< (std::ostream& s, Sym::KIND l)
{
	if      (l==Sym::KIND::S)   {s << "S";}
	else if (l==Sym::KIND::T)   {s << "T";}
	else if (l==Sym::KIND::N)   {s << "N";}
	else if (l==Sym::KIND::M)   {s << "M";}
	else if (l==Sym::KIND::Nup) {s << "N↑";}
	else if (l==Sym::KIND::Ndn) {s << "N↓";}
	return s;
}

enum PARITY {EVEN=0, ODD=1};

enum SUB_LATTICE {A=0,B=1};

std::ostream& operator<< (std::ostream& s, SUB_LATTICE sublat)
{
	if      (sublat==A)  {s << "A";}
	else if (sublat==B)  {s << "B";}
	return s;
}

//enum BC_CHOICE
//{
//	RING, /**<Periodic boundary conditions implemented via an MPO with transfer between the first and the last site.*/
//	HAIRSLIDE, /**<Periodic boundary conditions implemented via chain folding.*/
//	CYLINDER, /**<Periodic boundary conditions in y-direction by using the full Hilbert space.*/
//	FLADDER, /**<2-leg ladder flattened to chain with nnn-hopping.*/
//	CHAIN /**Chain with open boundary conditions for consistency.*/
//};

//std::ostream& operator<< (std::ostream& s, BC_CHOICE CHOICE)
//{
//	if      (CHOICE==RING)      {s << "RING";}
//	else if (CHOICE==CYLINDER)  {s << "CYLINDER";}
//	else if (CHOICE==HAIRSLIDE) {s << "HAIRSLIDE";}
//	else if (CHOICE==FLADDER)   {s << "FLADDER";}
//	else if (CHOICE==CHAIN)     {s << "CHAIN";}
//	return s;
//}

//template<BC_CHOICE CHOICE> struct BC;

//template<>
//struct BC<HAIRSLIDE>
//{
//	BC (size_t Lx_input)
//	:Lx(Lx_input/2), Ly(2), CHOICE(HAIRSLIDE)
//	{
//		assert(Lx_input%2==0 and "L must be even for rings because of folding!");
//	}
//	
//	BC_CHOICE CHOICE;
//	size_t Lx;
//	size_t Ly;
//};

//template<>
//struct BC<RING>
//{
//	BC (size_t Lx_input)
//	:Lx(Lx_input), Ly(1), CHOICE(RING)
//	{}
//	
//	BC_CHOICE CHOICE;
//	size_t Lx;
//	size_t Ly;
//};

//template<>
//struct BC<CYLINDER>
//{
//	BC (size_t Lx_input, size_t Ly_input)
//	:Lx(Lx_input), Ly(Ly_input), CHOICE(CYLINDER)
//	{}
//	
//	BC_CHOICE CHOICE;
//	size_t Lx;
//	size_t Ly;
//};

//template<>
//struct BC<FLADDER>
//{
//	BC (size_t Lx_input)
//	:Lx(2*Lx_input), Ly(1), CHOICE(FLADDER)
//	{}
//	
//	BC_CHOICE CHOICE;
//	size_t Lx;
//	size_t Ly;
//};

//template<>
//struct BC<CHAIN>
//{
//	BC (size_t Lx_input)
//	:Lx(Lx_input), Ly(1), CHOICE(CHAIN)
//	{}
//	
//	BC_CHOICE CHOICE;
//	size_t Lx;
//	size_t Ly;
//};

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#ifndef EIGEN_DEFAULT_SPARSE_INDEX_TYPE
#define EIGEN_DEFAULT_SPARSE_INDEX_TYPE int
#endif
using namespace Eigen;
typedef SparseMatrix<double,ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> SparseMatrixXd;
typedef SparseMatrix<std::complex<double>,ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> SparseMatrixXcd;

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
			SVD, /**<Uses the singular value decomposition. All singular values smaller than DmrgJanitor::eps_svd are cut off.
			\warning Needs Lapack. Define DONT_USE_LAPACK_SVD to use Eigen's SVD routine. It is 3-90 times slower, but probably accurate and stable as hell.*/
			BRUTAL_SVD, /**<Uses the singular value decomposition, but imposes a limit on the matrix sizes. Only the first DmrgJanitor::N_sv singular values per symmetry block are kept. This option can be useful to generate starting points for variational procedures. \warning Needs Lapack (see above).*/
			RDM, /**<Uses diagonalization of the reduced density matrix. Adds a noise term multiplied by DmrgJanitor::eps_noise. All eigenvalues smaller than DmrgJanitor::eps_rdm are cut off. (Note that the eigenvalues are the singular values squared, so DmrgJanitor::eps_noise must be chosen accordingly).*/
			RICH_SVD, /**<Uses an enrichment scheme, a.k.a.\ the "Strictly Single-Site (SSS) algorithm" according to Hubig et al.\ (2015). The matrices are enlarged with non-local information times DmrgJanitor::eps_rsvd, then the usual SVD is performed (using DmrgJanitor::eps_svd as a cutoff parameter). Note that DmrgJanitor::eps_rsvd is a much less sensitive parameter than DmrgJanitor::eps_rdm and can be even kept equal to 1 (but this will result in larger than optimal matrix sizes). \warning Needs Lapack (see above).*/
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
	
	/**Default configuration values for various solvers.*/
	struct CONTROL
	{
		struct DEFAULT
		{
			//GLOB DEFAULTS
			constexpr static size_t min_halfsweeps = 1;
			constexpr static size_t max_halfsweeps = 20;
			constexpr static double tol_eigval = 1e-6;
			constexpr static double tol_state = 1e-5;
			constexpr static size_t Dinit = 5;
			constexpr static size_t Dlimit = 500;
			constexpr static size_t Qinit = 10;
			constexpr static size_t savePeriod = 0;
			constexpr static DMRG::CONVTEST::OPTION CONVTEST = DMRG::CONVTEST::VAR_2SITE;
			constexpr static bool CALC_S_ON_EXIT = true;
			
			//DYN DEFAULTS
			static double max_alpha_rsvd (size_t i) {return (i<=10)? 1e2:0;}
			static double min_alpha_rsvd (size_t i) {return (i<=10)? 1e-11:0;}
			static double eps_svd        (size_t i) {return 1e-7;}
			static size_t Dincr_abs      (size_t i) {return 4;} // increase D by at least 2
			static double Dincr_rel      (size_t i) {return 1.1;} // increase D by at least 10%
			static size_t Dincr_per      (size_t i) {return 2;} // increase D every 2 half-sweeps
			static size_t min_Nsv        (size_t i) {return 0;}
			static int    max_Nrich      (size_t i) {return -1;} // -1 = infinity
			static void   doSomething    (size_t i) {return;} // -1 = infinity
			
			//LANCZOS DEFAULTS
			constexpr static ::LANCZOS::REORTHO::OPTION REORTHO = LANCZOS::REORTHO::FULL;
			constexpr static ::LANCZOS::CONVTEST::OPTION LANCZOS_CONVTEST = LANCZOS::CONVTEST::COEFFWISE;
			constexpr static double eps_eigval = 1e-7;
			constexpr static double eps_coeff = 1e-4;
			constexpr static size_t dimK = 40ul;
		};
		
		struct GLOB
		{
			size_t min_halfsweeps           = CONTROL::DEFAULT::min_halfsweeps;
			size_t max_halfsweeps           = CONTROL::DEFAULT::max_halfsweeps;
			double tol_eigval               = CONTROL::DEFAULT::tol_eigval;
			double tol_state                = CONTROL::DEFAULT::tol_state;
			size_t Dinit                    = CONTROL::DEFAULT::Dinit;
			size_t Dlimit                   = CONTROL::DEFAULT::Dlimit;
			size_t Qinit                    = CONTROL::DEFAULT::Qinit;
			size_t savePeriod               = CONTROL::DEFAULT::savePeriod;
			DMRG::CONVTEST::OPTION CONVTEST = CONTROL::DEFAULT::CONVTEST;
			bool CALC_S_ON_EXIT             = CONTROL::DEFAULT::CALC_S_ON_EXIT;
		};
		
		struct DYN
		{
			function<double(size_t)> max_alpha_rsvd = CONTROL::DEFAULT::max_alpha_rsvd;
			function<double(size_t)> min_alpha_rsvd = CONTROL::DEFAULT::min_alpha_rsvd;
			function<double(size_t)> eps_svd        = CONTROL::DEFAULT::eps_svd;
			function<size_t(size_t)> Dincr_abs      = CONTROL::DEFAULT::Dincr_abs;
			function<double(size_t)> Dincr_rel      = CONTROL::DEFAULT::Dincr_rel;
			function<size_t(size_t)> Dincr_per      = CONTROL::DEFAULT::Dincr_per;
			function<size_t(size_t)> min_Nsv        = CONTROL::DEFAULT::min_Nsv;
			function<int(size_t)> max_Nrich         = CONTROL::DEFAULT::max_Nrich;
			function<void(size_t)> doSomething      = CONTROL::DEFAULT::doSomething;
		};
		
		struct LANCZOS
		{
			::LANCZOS::REORTHO::OPTION REORTHO   = CONTROL::DEFAULT::REORTHO;
			::LANCZOS::CONVTEST::OPTION CONVTEST = CONTROL::DEFAULT::LANCZOS_CONVTEST;
			double eps_eigval                    = CONTROL::DEFAULT::eps_eigval;
			double eps_coeff                     = CONTROL::DEFAULT::eps_coeff;
			size_t dimK                          = CONTROL::DEFAULT::dimK;
		};
	};
};

std::ostream& operator<< (std::ostream& s, DMRG::VERBOSITY::OPTION VERB)
{
	if (VERB==DMRG::VERBOSITY::SILENT) {s << "SILENT";}
	else if (VERB==DMRG::VERBOSITY::ON_EXIT) {s << "ON_EXIT";}
	else if (VERB==DMRG::VERBOSITY::HALFSWEEPWISE) {s << "HALFSWEEPWISE";}
	else if (VERB==DMRG::VERBOSITY::STEPWISE) {s << "STEPWSIE";}
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

#endif
