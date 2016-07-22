#ifndef DMRG_TYPEDEFS
#define DMRG_TYPEDEFS

#include <array>

#ifndef IS_REAL_FUNCTION
#define IS_REAL_FUNCTION
inline double isReal (double x) {return x;}
inline double isReal (complex<double> x) {return x.real();}
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

enum PARITY {EVEN=0, ODD=1};

enum BC_CHOICE
{
	RING, /**<Periodic boundary conditions implemented via an MPO with transfer between the first and the last site.*/
	HAIRSLIDE, /**<Periodic boundary conditions implemented via chain folding.*/
	CYLINDER, /**<Periodic boundary conditions in y-direction by using the full Hilbert space.*/
	FLADDER, /**<2-leg ladder flattened to chain with nnn-hopping.*/
	CHAIN /**Chain with open boundary conditions for consistency.*/
};

std::ostream& operator<< (std::ostream& s, BC_CHOICE CHOICE)
{
	if      (CHOICE==RING)      {s << "RING";}
	else if (CHOICE==CYLINDER)  {s << "CYLINDER";}
	else if (CHOICE==HAIRSLIDE) {s << "HAIRSLIDE";}
	else if (CHOICE==FLADDER)   {s << "FLADDER";}
	else if (CHOICE==CHAIN)     {s << "CHAIN";}
	return s;
}

template<BC_CHOICE CHOICE> struct BC;

template<>
struct BC<HAIRSLIDE>
{
	BC (size_t Lx_input)
	:Lx(Lx_input/2), Ly(2), CHOICE(HAIRSLIDE)
	{
		assert(Lx_input%2==0 and "L must be even for rings because of folding!");
	}
	
	BC_CHOICE CHOICE;
	size_t Lx;
	size_t Ly;
};

template<>
struct BC<RING>
{
	BC (size_t Lx_input)
	:Lx(Lx_input), Ly(1), CHOICE(RING)
	{}
	
	BC_CHOICE CHOICE;
	size_t Lx;
	size_t Ly;
};

template<>
struct BC<CYLINDER>
{
	BC (size_t Lx_input, size_t Ly_input)
	:Lx(Lx_input), Ly(Ly_input), CHOICE(CYLINDER)
	{}
	
	BC_CHOICE CHOICE;
	size_t Lx;
	size_t Ly;
};

template<>
struct BC<FLADDER>
{
	BC (size_t Lx_input)
	:Lx(2*Lx_input), Ly(1), CHOICE(FLADDER)
	{}
	
	BC_CHOICE CHOICE;
	size_t Lx;
	size_t Ly;
};

template<>
struct BC<CHAIN>
{
	BC (size_t Lx_input)
	:Lx(Lx_input), Ly(1), CHOICE(CHAIN)
	{}
	
	BC_CHOICE CHOICE;
	size_t Lx;
	size_t Ly;
};

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#ifndef EIGEN_DEFAULT_SPARSE_INDEX_TYPE
#define EIGEN_DEFAULT_SPARSE_INDEX_TYPE int
#endif
using namespace Eigen;
typedef SparseMatrix<double,ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> SparseMatrixXd;
typedef SparseMatrix<complex<double>,ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> SparseMatrixXcd;

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
			RICH_SVD /**<Uses an enrichment scheme, a.k.a.\ the "Strictly Single-Site (SSS) algorithm" according to Hubig et al.\ (2015). The matrices are enlarged with non-local information times DmrgJanitor::eps_rsvd, then the usual SVD is performed (using DmrgJanitor::eps_svd as a cutoff parameter). Note that DmrgJanitor::eps_rsvd is a much less sensitive parameter than DmrgJanitor::eps_rdm and can be even kept equal to 1 (but this will result in larger than optimal matrix sizes). \warning Needs Lapack (see above).*/
		};
	};
	
	/**Choice of verbosity for DmrgSolver, DmrgSolverQ, MpsCompressor and MpsCompressorQ.*/
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
	
	/**Choice of initial guess for variational compression in MpsCompressor and MpsCompressorQ.*/
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
	
	/**Choice of how to resize in MpsCompressor and MpsCompressorQ.*/
	struct RESIZE
	{
		enum OPTION
		{
			CONSERV_INCR, /**<conservative increase: add a row and column of zeros for each subspace block, the bond dimension increases by \p Nqmax*/
			DECR /**<non-conservative decrease: cut all blocks according to a given \p Dmax*/
		};
	};
};

std::ostream& operator<< (std::ostream& s, DMRG::DIRECTION::OPTION DIR)
{
	if (DIR==DMRG::DIRECTION::LEFT) {s << "LEFT";}
	else                            {s << "RIGHT";}
	return s;
}

template<typename Scalar> using LocalTerms = vector<tuple<Scalar,SparseMatrix<Scalar,ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> > >;
template<typename Scalar> using TightTerms = vector<tuple<Scalar,SparseMatrix<Scalar,ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE>,
                                                                 SparseMatrix<Scalar,ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> > >;
template<typename Scalar> using NextnTerms = vector<tuple<double,SparseMatrix<Scalar,ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE>,
                                                                 SparseMatrix<Scalar,ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE>,
                                                                 SparseMatrix<Scalar,ColMajor,EIGEN_DEFAULT_SPARSE_INDEX_TYPE> > >;

template<typename Scalar>
struct HamiltonianTerms
{
	/**local terms of Hamiltonian, format: coupling, operator*/
	LocalTerms<Scalar> local;
	
	/**nearest-neighbour terms of Hamiltonian, format: coupling, operator 1, operator 2*/
	TightTerms<Scalar> tight;
	
	/**next-nearest-neighbour terms of Hamiltonian, format: coupling, operator 1, operator 2, transfer operator*/
	NextnTerms<Scalar> nextn;
	
	inline size_t auxdim() {return 2+tight.size()+2*nextn.size();}
};

typedef HamiltonianTerms<double>           HamiltonianTermsXd;
typedef HamiltonianTerms<complex<double> > HamiltonianTermsXcd;

#endif
