#ifndef VANILLA_VUMPSSOLVER
#define VANILLA_VUMPSSOLVER

/// \cond
//#include "unsupported/Eigen/IterativeSolvers"
#include "termcolor.hpp"
/// \endcond

#include "LanczosSolver.h" // from ALGS
#include "GMResSolver.h" // from ALGS

#include "VUMPS/Umps.h"
#include "VUMPS/VumpsPivotMatrices.h"
#include "pivot/DmrgPivotMatrix0.h"
#include "pivot/DmrgPivotMatrix2.h"
#include "VUMPS/VumpsMpoTransferMatrix.h"
#include "VUMPS/VumpsTransferMatrix.h"

/**
 * Solver that calculates the ground state of a UMPS. Analogue of the DmrgSolver class.
 * \ingroup VUMPS
 * \describe_Symmetry
 * \describe_Scalar
 */
template<typename Symmetry, typename MpHamiltonian, typename Scalar=double>
class VumpsSolver
{
	typedef Matrix<Scalar,Dynamic,Dynamic> MatrixType;
	typedef Matrix<complex<Scalar>,Dynamic,Dynamic> ComplexMatrixType;
	typedef Matrix<Scalar,Dynamic,1>       VectorType;
	typedef boost::multi_array<Scalar,4> TwoSiteHamiltonian;
	
public:
	
	VumpsSolver (DMRG::VERBOSITY::OPTION VERBOSITY=DMRG::VERBOSITY::SILENT)
	:CHOSEN_VERBOSITY(VERBOSITY)
	{};
	
	/**Resets the verbosity level.*/
	inline void set_verbosity (DMRG::VERBOSITY::OPTION VERBOSITY) {CHOSEN_VERBOSITY = VERBOSITY;};

	/**Returnx the verbosity level.*/
	inline DMRG::VERBOSITY::OPTION get_verbosity () {return CHOSEN_VERBOSITY;};

	//call this function if you want to set the parameters for the solver by yourself
	void userSetGlobParam    () { USER_SET_GLOBPARAM     = true; }
	void userSetDynParam     () { USER_SET_DYNPARAM      = true; }
	void userSetLanczosParam () { USER_SET_LANCZOSPARAM  = true; }

	VUMPS::CONTROL::GLOB GlobParam;
	VUMPS::CONTROL::DYN  DynParam;
	VUMPS::CONTROL::LANCZOS LanczosParam;

	///\{
	/**\describe_info*/
	string info() const;
	
	/**\describe_info*/
	string eigeninfo() const;
	
	/**\describe_memory*/
	double memory (MEMUNIT memunit=GB) const;
	
	inline size_t iterations() {return N_iterations;};
	
	/**
	 * Setup a logfile of the iterations.
	 * \param N_log_input : save the log every \p N_log_input half-sweeps
	 * \param file_e_input : file for the ground-state energy in the format [min(eL,eR), eL, eR]
	 * \param file_err_eigval_input : file for the energy error
	 * \param file_err_var_input : file for the variational error
	 * \param file_err_state_input : file for the global state error
	 */
	void set_log (int N_log_input, string file_e_input, string file_err_eigval_input, string file_err_var_input, string file_err_state_input);
	///\}
	
	/**
	 * Calculates the highest or lowest eigenstate with an MPO (algorithm 6).
	 */
	void edgeState (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout, 
	                qarray<Symmetry::Nq> Qtot, LANCZOS::EDGE::OPTION EDGE=LANCZOS::EDGE::GROUND, bool USE_STATE=false);
	
	const double& errVar()    {return err_var;}
	const double& errState()  {return err_state;}
	const double& errEigval() {return err_eigval;}
	
	bool FORCE_DO_SOMETHING = false;
	
	Mps<Symmetry,Scalar> create_Mps (size_t Ncells, const MpHamiltonian &H, const Eigenstate<Umps<Symmetry,Scalar> > &V, bool ADD_ODD_SITE=false);
	
private:
	
	///\{
	/**Prepares the class, setting up the environments. Used with an explicit 2-site Hamiltonian.*/
	void prepare_h2site (const TwoSiteHamiltonian &h2site, const vector<qarray<Symmetry::Nq> > &qloc_input, 
	                     Eigenstate<Umps<Symmetry,Scalar> > &Vout, 
	                     qarray<Symmetry::Nq> Qtot_input,
	                     size_t M, size_t Nqmax, bool USE_STATE=false);
	
	/**
	 * Performs an iteration with 1-site unit cell. Used with an explicit 2-site Hamiltonian.
	 * \warning : This function is not implemented for SU(2).
	 */
	void iteration_h2site (Eigenstate<Umps<Symmetry,Scalar> > &Vout);
	///\}
	
	///\{
	/**Prepares the class setting up the environments. Used with an Mpo.*/
	void prepare (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout, qarray<Symmetry::Nq> Qtot, bool USE_STATE=false);
	
	/**Performs an iteration with an n-site unit cell (in parallel, algorithm 3). Used with an MPO.*/
	void iteration_parallel (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout);
	
	/**Performs an iteration with an n-site unit cell (sequentially, algorithm 4). Used with an MPO.*/
	void iteration_sequential (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout);
	///\}
	
	///\{
	/**
	 * Performs an IDMRG iteration with a 2-site unit cell.
	 * \warning : This function is not implemented for SU(2).
	 */
	void iteration_idmrg (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout);
	
	/**Prepares the class, setting up the environments for IDMRG.*/
	void prepare_idmrg (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout, qarray<Symmetry::Nq> Qtot, bool USE_STATE=false);
	
	/**old energy for comparison in IDMRG.*/
	double Eold = std::nan("1");
	///\}
	
	///\{
	/**Builds the environment of a unit cell.*/
	void build_LR (const vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > &AL,
	               const vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > &AR,
	               const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &C,
	               const vector<vector<vector<vector<SparseMatrix<Scalar> > > > > &W, 
	               const vector<vector<qarray<Symmetry::Nq> > > &qloc, 
	               const vector<vector<qarray<Symmetry::Nq> > > &qOp,
	               Tripod<Symmetry,MatrixType> &L,
	               Tripod<Symmetry,MatrixType> &R);
	
	/**Builds environments for each site of the unit cell.*/
	void build_cellEnv (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout);
	///\}
	
	///\{
	/**
	 * This function adds orthogonal information to the UMPS in the unit cell at site loc and enlarge therewith the bond dimension and the number of symmetry blocks.
	 * For information see appendix B in Zauner-Stauber et al. 2018.
	 */
	void expand_basis (size_t loc, size_t DeltaD, const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout, 
	                   VUMPS::TWOSITE_A::OPTION option = VUMPS::TWOSITE_A::ALxAC);
	/**
	 * This function adds orthogonal information to the UMPS and enlarge therewith the bond dimension and the number of symmetry blocks.
	 * For information see appendix B in Zauner-Stauber et al. 2018.
	 * Just calls expand_basis() for all positions in the unit cell.
	 */
	void expand_basis (size_t DeltaD, const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout, 
	                   VUMPS::TWOSITE_A::OPTION option = VUMPS::TWOSITE_A::ALxAC);
	///\}
	
	///\{
	/**
	 * Calculates the two-site B-tensor (from double tangent space). It is relevant for orthogonal information in the Umps as well as for the global state error.
	 * \param loc : Calculate the two-site B-tensor at sites loc and loc+1
	 * \param H : Mpo
	 * \param Vout : Umps
	 * \param option : how to calculate the two-site A-tensor (ALxAC, ARxAC or ALxCxAR)
	 * \param B2 : The two-site B-tensor as the return value. 
	 * \param NL : The left nullspace, which is calculated during the routine and returned for later use. (Needed when performing the enrichment)
	 * \param NR : You can guess what this is, probably.
	 */
	void calc_B2 (size_t loc, const MpHamiltonian &H, const Eigenstate<Umps<Symmetry,Scalar> > &Vout, VUMPS::TWOSITE_A::OPTION option,
				  Biped<Symmetry,MatrixType>& B2, vector<Biped<Symmetry,MatrixType> > &NL, vector<Biped<Symmetry,MatrixType> > &NR) const;
	/**
	 * A wrapper, if you want to discard the nullspaces when calculating B2.
	 */
	void calc_B2 (size_t loc, const MpHamiltonian &H, const Eigenstate<Umps<Symmetry,Scalar> > &Vout, VUMPS::TWOSITE_A::OPTION option, Biped<Symmetry,MatrixType>& B2) const;
	///\}
	
	/**Cleans up after the iteration process.*/
	void cleanup (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout);
	
	/**chain length*/
	size_t N_sites;

	bool USER_SET_GLOBPARAM    = false;
	bool USER_SET_DYNPARAM     = false;
	bool USER_SET_LANCZOSPARAM = false;
	
	/**keeping track of iterations*/
	size_t N_iterations, N_iterations_without_expansion;
	
	/**errors*/
	double err_eigval, err_eigval_old, err_var, err_var_old, err_state=std::nan("1"), err_state_old=std::nan("1");
	
	/**environment for the 2-site Hamiltonian version*/
	vector<PivumpsMatrix1<Symmetry,Scalar,Scalar> > Heff;
	
	/**environment of \p AL and \p AR for the Mpo version*/
	vector<PivotMatrix1<Symmetry,Scalar,Scalar> > HeffA;
	
	/**environment of \p C for the Mpo version*/
	vector<PivotMatrix1<Symmetry,Scalar,Scalar> > HeffC;
	
	/**local base*/
	vector<qarray<Symmetry::Nq> > qloc;
	
	/**stored 2-site Hamiltonian*/
	TwoSiteHamiltonian h2site;
	
	/**bond dimension per subspace, bond dimension per site, Mpo bond dimension*/
	size_t D, M, dW;
	
	/**left and right error (eq. 18) and old errors from previous half-sweep*/
	double eL, eR, eoldR, eoldL;
	
	/**Solves the linear system (eq. C25ab) using GMRES.
	 * \param gauge : L or R
	 * \param ab : fixed index \p a (rows) or \p b (cols)
	 * \param A : contracted A-tensor of the cell
	 * \param Y_LR : |Y_Ra), (Y_La| for eq. C25ab
	 * \param LReigen : (L| or |R)
	 * \param W : Mpo tensor for the transfer matrix
	 * \param qloc : local basis
	 * \param qOp : operator basis
	 * \param LRdotY : (Y_La|R), (L|Y_Ra) for eq. C25ab
	 * \param LRguess : the starting guess for the linear solver
	 * \param LRres : resulting (H_L| or |H_R)
	 */
	void solve_linear (VMPS::DIRECTION::OPTION gauge, 
	                   size_t ab, 
	                   const vector<vector<Biped<Symmetry,MatrixType> > > &A, 
	                   const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &Y_LR, 
	                   const Biped<Symmetry,MatrixType> &LReigen, 
	                   const vector<vector<vector<vector<SparseMatrix<Scalar> > > > > &W, 
	                   const vector<vector<qarray<Symmetry::Nq> > > &qloc, 
	                   const vector<vector<qarray<Symmetry::Nq> > > &qOp,
	                   Scalar LRdotY, 
	                   const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &LRguess, 
	                   Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &LRres);
	
	/**Solves the linear system (eq. 15) using GMRES.
	 * \param DIR : L or R
	 * \param A : contracted A-tensor of the cell
	 * \param hLR : (h_L|, |h_R) for eq. 15
	 * \param LReigen : (L| or |R) 
	 * \param qloc : local basis
	 * \param hLRdotLR : (h_L|R), (L|h_R) for eq. 15
	 * \param LRres : resulting (H_L| or |H_R)
	 */
	void solve_linear (VMPS::DIRECTION::OPTION DIR, 
	                   const vector<vector<Biped<Symmetry,MatrixType> > > &A, 
	                   const Biped<Symmetry,MatrixType> &hLR, 
	                   const Biped<Symmetry,MatrixType> &LReigen, 
	                   const vector<vector<qarray<Symmetry::Nq> > > &qloc, 
	                   Scalar hLRdotLR, 
	                   Biped<Symmetry,MatrixType> &LRres);
	
	/**control of verbosity and algorithms*/
	DMRG::VERBOSITY::OPTION CHOSEN_VERBOSITY;
	
	/**Sets the Lanczos tolerances adaptively, depending on the current errors.*/
	void set_LanczosTolerances (double &tolLanczosEigval, double &tolLanczosState);
	
	/**Calculates the errors.*/
	void calc_errors (const MpHamiltonian &H, const Eigenstate<Umps<Symmetry,Scalar> > &Vout, 
	                  bool CALC_ERR_STATE=true, VUMPS::TWOSITE_A::OPTION option = VUMPS::TWOSITE_A::ALxAC);
	
	/**saved \f$Y_{L_{0}}\f$, see eq. (C26), (C27)*/
	Tripod<Symmetry,MatrixType> YLlast;
	
	/**saved \f$Y_{R_{dW-1}}\f$, see eq. (C26), (C27)*/
	Tripod<Symmetry,MatrixType> YRfrst;
	
	/**Tests if \f$C \cdot C^{\dagger}\f$ and \f$C^{\dagger} \cdot C\f$ are the eigenvectors of the transfer matrices.*/
	string test_LReigen (const Eigenstate<Umps<Symmetry,Scalar> > &Vout) const;
	
	///\{
	/**Save log every \p N_log optimization steps.*/
	size_t N_log = 0;
	
	/**log filenames*/
	string file_e, file_err_eigval, file_err_var, file_err_state;
	
	/**log data*/
	vector<double> eL_mem, eR_mem, err_eigval_mem, err_var_mem, err_state_mem;
	
	/**
	 * Function to write out the logfiles.
	 * \param FORCE : if \p true, forced write without checking any conditions
	 */
	void write_log (bool FORCE = false);
	///\}
};

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
string VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
info() const
{
	stringstream ss;
	ss << "VumpsSolver: ";
	ss << "L=" << N_sites << ", ";
	ss << eigeninfo();
	return ss.str();
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
string VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
eigeninfo() const
{
	stringstream ss;
	ss << "iterations=" << N_iterations << ", ";
	ss << "e0=" << setprecision(13) << min(eL,eR) << ", ";
	ss << "err_eigval=" << setprecision(13) << err_eigval << ", err_var=" << err_var << ", ";
	if (!isnan(err_state))
	{
		ss << "err_state=" << err_state << ", ";
	}
	ss << "mem=" << round(memory(GB),3) << "GB";
	return ss.str();
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
double VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
memory (MEMUNIT memunit) const
{
	double res = 0.;
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		res += Heff[l].L.memory(memunit);
//		res += Heff[l].R.memory(memunit);
//		for (size_t s1=0; s1<Heff[l].W.size(); ++s1)
//		for (size_t s2=0; s2<Heff[l].W[s1].size(); ++s2)
//		for (size_t k=0; k<Heff[l].W[s1][s2].size(); ++k)
//		{
//			res += calc_memory(Heff[l].W[s1][s2][k],memunit);
//		}
//	}
	return res;
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
set_log (int N_log_input, string file_e_input, string file_err_eigval_input, string file_err_var_input, string file_err_state_input)
{
	N_log           = N_log_input;
	file_e          = file_e_input;
	file_err_eigval = file_err_eigval_input;
	file_err_var    = file_err_var_input;
	file_err_state  = file_err_state_input;
	eL_mem.clear();
	eR_mem.clear();
	err_eigval_mem.clear();
	err_var_mem.clear();
	err_state_mem.clear();
};

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
write_log (bool FORCE)
{
	// save data
	if (N_log>0 or FORCE==true)
	{
		eL_mem.push_back(eL);
		eR_mem.push_back(eR);
		err_eigval_mem.push_back(err_eigval);
		err_var_mem.push_back(err_var);
		err_state_mem.push_back(err_state);
	}
	
	if ((N_log>0 and N_iterations%N_log==0) or FORCE==true)
	{
		// write out energy
		ofstream Filer(file_e);
		for (int i=0; i<eL_mem.size(); ++i)
		{
			Filer << i << "\t" << setprecision(13) << min(eL_mem[i],eR_mem[i]) << "\t" << eL_mem[i] << "\t" << eR_mem[i] << endl;
		}
		Filer.close();
		
		// write out energy error
		Filer.open(file_err_eigval);
		for (int i=0; i<err_eigval_mem.size(); ++i)
		{
			Filer << i << "\t" << setprecision(13) << err_eigval_mem[i] << endl;
		}
		Filer.close();
		
		// write out variational error
		Filer.open(file_err_var);
		for (int i=0; i<err_var_mem.size(); ++i)
		{
			Filer << i << "\t" << setprecision(13) << err_var_mem[i] << endl;
		}
		Filer.close();

		// write out global state error
		Filer.open(file_err_state);
		for (int i=0; i<err_state_mem.size(); ++i)
		{
			Filer << i << "\t" << setprecision(13) << err_state_mem[i] << endl;
		}
		Filer.close();

	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
set_LanczosTolerances (double &tolLanczosEigval, double &tolLanczosState)
{
	// Set less accuracy for the first iterations
	// tolLanczosEigval = max(max(1.e-2*err_eigval,1.e-13),1.e-13); // 1e-7
	// tolLanczosState  = max(max(1.e-2*err_var,   1.e-13),1.e-13); // 1e-4
	tolLanczosEigval = max(max(1.e-2*err_eigval,LanczosParam.eps_eigval),LanczosParam.eps_eigval); // 1e-7
	tolLanczosState  = max(max(1.e-2*err_var,   LanczosParam.eps_coeff) ,LanczosParam.eps_coeff); // 1e-4
	
	if (std::isnan(tolLanczosEigval))
	{
		tolLanczosEigval = 1e-7;
	}
	
	if (std::isnan(tolLanczosState))
	{
		tolLanczosState = 1e-4;
	}
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
	{
		lout << "current Lanczos tolerances: " << tolLanczosEigval << ", " << tolLanczosState << endl;
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
calc_errors (const MpHamiltonian &H, const Eigenstate<Umps<Symmetry,Scalar> > &Vout, bool CALC_ERR_STATE, VUMPS::TWOSITE_A::OPTION option)
{
	std::array<VectorXd,2> epsLRsq;
	std::array<GAUGE::OPTION,2> gs = {GAUGE::L, GAUGE::R};
	for (const auto &g:gs)
	{
		epsLRsq[g].resize(N_sites);
		for (size_t l=0; l<N_sites; ++l)
		{
			epsLRsq[g](l) = Vout.state.calc_epsLRsq(g,l);
		}
	}
	err_var_old = err_var;
	err_var = max(sqrt(epsLRsq[GAUGE::L].sum()), sqrt(epsLRsq[GAUGE::R].sum()));
	
	err_eigval_old = err_eigval;
	err_eigval = max(abs(eoldR-eR), abs(eoldL-eL));
	eoldR = eR;
	eoldL = eL;
	
	if (CALC_ERR_STATE)
	{
		//set the global state error to the largest norm of NAAN (=B2) in the unit cell.
		err_state_old = err_state;
		err_state = 0.;
		vector<double> norm_NAAN(N_sites);
		#pragma omp parallel for
		for (size_t l=0; l<N_sites; ++l)
		{
			vector<Biped<Symmetry,MatrixType> > NL;
			vector<Biped<Symmetry,MatrixType> > NR;
			Biped<Symmetry,MatrixType> NAAN;
			calc_B2(l, H, Vout, option, NAAN, NL, NR);
			norm_NAAN[l] = sqrt(NAAN.squaredNorm().sum());
		}
		
		for (size_t l=0; l<N_sites; ++l)
		{
			if (norm_NAAN[l] > err_state) {err_state = norm_NAAN[l];}
		}
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
prepare_h2site (const TwoSiteHamiltonian &h2site_input, const vector<qarray<Symmetry::Nq> > &qloc_input, Eigenstate<Umps<Symmetry,Scalar> > &Vout, 
                qarray<Symmetry::Nq> Qtot, size_t M_input, size_t Nqmax, bool USE_STATE)
{
	Stopwatch<> PrepTimer;

	// general
	N_sites = 1;
	N_iterations = 0;
	D = h2site_input.shape()[0]; // local dimension
	M = M_input; // bond dimension
	
	// effective and 2-site Hamiltonian
	Heff.resize(N_sites);
	h2site.resize(boost::extents[D][D][D][D]);
	h2site = h2site_input;
	
	// resize Vout

	if (!USE_STATE)
	{
		Vout.state = Umps<Symmetry,Scalar>(qloc_input, Qtot, N_sites, M, Nqmax);
		Vout.state.setRandom();
		for (size_t l=0; l<N_sites; ++l)
		{
			Vout.state.svdDecompose(l);
		}
	}
	Vout.state.calc_entropy((CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)? true : false);
	
	// initial energy & error
	eoldL = std::nan("");
	eoldR = std::nan("");
	err_eigval = 1.;
	err_var    = 1.;
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << PrepTimer.info("prepare") << endl; 
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
prepare (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout, qarray<Symmetry::Nq> Qtot, bool USE_STATE)
{
	N_sites = H.length();
	N_iterations = 0;
	N_iterations_without_expansion = 0;
	
	Stopwatch<> PrepTimer;
	
	// effective Hamiltonian
	D = H.locBasis(0).size();
	assert(H.auxrows(0) == H.auxcols(N_sites-1) and "You've inserted a strange MPO not consistent with the unit cell");
	dW = H.auxrows(0);
	
	// resize Vout
	if (!USE_STATE)
	{
		Vout.state = Umps<Symmetry,Scalar>(H, Qtot, N_sites, GlobParam.Dinit, GlobParam.Qinit);
		Vout.state.graph("init");
		Vout.state.max_Nsv = GlobParam.Dlimit;
		// Vout.state.min_Nsv = DynParam.min_Nsv(0);
		Vout.state.setRandom();
		for (size_t l=0; l<N_sites; ++l)
		{
			Vout.state.svdDecompose(l);
		}
	}
	Vout.state.calc_entropy((CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)? true : false);
	
	// initial energy
	eoldL = std::nan("");
	eoldR = std::nan("");
	err_eigval = 1.;
	err_var    = 1.;
	
	HeffA.clear();
	HeffA.resize(N_sites);
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << PrepTimer.info("• initial decomposition") << endl;
		lout <<                "• initial state        : " << Vout.state.info() << endl;
		int i_expansion_switchoff=0;
		for (int i=0; i<GlobParam.max_iterations; ++i)
		{
			if (DynParam.max_deltaD(i) == 0.) {i_expansion_switchoff = i; break;}
		}
		lout << "• expansion turned off after " << termcolor::underline << i_expansion_switchoff << termcolor::reset << " iterations" << endl;
		lout << "• initial bond dim. increase by " << termcolor::underline << static_cast<int>((DynParam.Dincr_rel(0)-1.)*100.) 
		     << "%" << termcolor::reset << " and at least by " << termcolor::underline << DynParam.Dincr_abs(0) << termcolor::reset
		     << " every " << termcolor::underline << DynParam.Dincr_per(0) << termcolor::reset << " iterations" << endl;
		lout << "• keep at least " << termcolor::underline << Vout.state.min_Nsv << termcolor::reset << " singular values per block" << endl;
		lout << "• make between " << termcolor::underline << GlobParam.min_iterations << termcolor::reset << " and " 
		     << termcolor::underline << GlobParam.max_iterations << termcolor::reset << " iterations" << endl;
		bool USE_PARALLEL=false, USE_SEQUENTIAL=false, USE_DYNAMIC=false;
		for (int i=0; i<GlobParam.max_iterations; ++i)
		{
			if (DynParam.iteration(i) == UMPS_ALG::PARALLEL)   {USE_PARALLEL=true;}
			if (DynParam.iteration(i) == UMPS_ALG::SEQUENTIAL) {USE_SEQUENTIAL=true;}
			if (DynParam.iteration(i) == UMPS_ALG::DYNAMIC) {USE_DYNAMIC=true;}
		}
		if (USE_DYNAMIC and N_sites == 1)
		{
			lout << "• use the parallel algorithm" << endl;
		}
		else if (USE_DYNAMIC and N_sites > 1)
		{
			lout << "• use the sequential algorithm" << endl;
		}
		else if (USE_PARALLEL and USE_SEQUENTIAL)
		{
			lout << "• use a combination of sequential and parallel algorithm" << endl;
		}
		else if (USE_PARALLEL)
		{
			lout << "• use the parallel algorithm" << endl;
		}
		else if (USE_SEQUENTIAL)
		{
			lout << "• use the sequential algorithm" << endl;
		}
		lout << "• eigenvalue tolerance : " << termcolor::underline << GlobParam.tol_eigval << termcolor::reset << endl;
		lout << "• variational tolerance: " << termcolor::underline << GlobParam.tol_var << termcolor::reset << endl;
		lout << "• state tolerance: " << termcolor::underline << GlobParam.tol_state << termcolor::reset << endl;
		lout << endl;
//		Vout.state.graph("init");
	}

}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
prepare_idmrg (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout, qarray<Symmetry::Nq> Qtot, bool USE_STATE)
{
	Stopwatch<> PrepTimer;
	
	// general
	N_sites = 1;
	N_iterations = 0;
	assert(H.auxrows(0) == H.auxcols(N_sites-1) and "You insert a strange MPO not consistent with the unit cell");
	dW = H.auxrows(0);
	
	// resize Vout
	if (!USE_STATE)
	{
		Vout.state = Umps<Symmetry,Scalar>(H.locBasis(0), Qtot, N_sites, GlobParam.Dinit, GlobParam.Qinit);
		Vout.state.max_Nsv = GlobParam.Dlimit;
		Vout.state.setRandom();
		for (size_t l=0; l<N_sites; ++l)
		{
			Vout.state.svdDecompose(l);
		}
	}
	Vout.state.calc_entropy((CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)? true : false);
	
	HeffA.resize(1);
	Qbasis<Symmetry> inbase;
	inbase.pullData(Vout.state.A[GAUGE::C][0],0);
	Qbasis<Symmetry> outbase;
	outbase.pullData(Vout.state.A[GAUGE::C][0],1);
	HeffA[0].L.setIdentity(dW, 1, inbase);
	HeffA[0].R.setIdentity(dW, 1, outbase);
	
	// initial energy & error
	eoldL = std::nan("");
	eoldR = std::nan("");
	err_eigval = 1.;
	err_var    = 1.;
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << PrepTimer.info("prepare") << endl; 
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
build_LR (const vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > &AL,
          const vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > &AR,
          const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &C,
          const vector<vector<vector<vector<SparseMatrix<Scalar> > > > > &W, 
          const vector<vector<qarray<Symmetry::Nq> > > &qloc, 
          const vector<vector<qarray<Symmetry::Nq> > > &qOp,
          Tripod<Symmetry,MatrixType> &L,
          Tripod<Symmetry,MatrixType> &R)
{
	Stopwatch<> GMresTimer;
	
	auto Lguess = L;
	auto Rguess = R;
	L.clear();
	R.clear();
	
	// |R) and (L|
	Biped<Symmetry,MatrixType> Reigen = C[N_sites-1].contract(C[N_sites-1].adjoint());
	Biped<Symmetry,MatrixType> Leigen = C[N_sites-1].adjoint().contract(C[N_sites-1]);
	
	// |YRa) and (YLa|
	vector<Tripod<Symmetry,MatrixType> > YL(dW);
	vector<Tripod<Symmetry,MatrixType> > YR(dW);
	
	// |Ra) and (La|
	Qbasis<Symmetry> inbase;
	inbase.pullData(AL[0],0);
	Qbasis<Symmetry> outbase;
	outbase.pullData(AL[0],0);
	
	Tripod<Symmetry,MatrixType> IdL; IdL.setIdentity(dW, 1, inbase);
	Tripod<Symmetry,MatrixType> IdR; IdR.setIdentity(dW, 1, outbase);
	L.insert(dW-1, IdL);
	R.insert(0,    IdR);
	
	#pragma omp parallel sections
	{
		// Eq. C19
		#pragma omp section
		{
			for (int b=dW-2; b>=0; --b)
			{
				YL[b] = make_YL(b, L, AL, W, PROP::HAMILTONIAN, AL, qloc, qOp);
				
				if (b > 0)
				{
					L.insert(b,YL[b]);
				}
				else
				{
					Tripod<Symmetry,MatrixType> Ltmp;
					Tripod<Symmetry,MatrixType> Ltmp_guess; Ltmp_guess.insert(b,Lguess);
					solve_linear(VMPS::DIRECTION::LEFT, b, AL, YL[b], Reigen, W, qloc, qOp, contract_LR(b,YL[b],Reigen), Ltmp_guess, Ltmp);
					L.insert(b,Ltmp);
					
					if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE and b == 0)
					{
						#pragma omp critical
						{
							cout << "<L[0]|R>=" << contract_LR(0,Ltmp,Reigen) << endl;
						}
					}
				}
			}
		}
		
		// Eq. C20
		#pragma omp section
		{
			for (int a=1; a<dW; ++a)
			{
				YR[a] = make_YR(a, R, AR, W, PROP::HAMILTONIAN, AR, qloc, qOp);
				
				if (a < dW-1)
				{
					R.insert(a,YR[a]);
				}
				else
				{
					Tripod<Symmetry,MatrixType> Rtmp;
					Tripod<Symmetry,MatrixType> Rtmp_guess; Rtmp_guess.insert(a,Rguess);
					solve_linear(VMPS::DIRECTION::RIGHT, a, AR, YR[a], Leigen, W, qloc, qOp, contract_LR(a,Leigen,YR[a]), Rtmp_guess, Rtmp);
					R.insert(a,Rtmp);
					
					if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE and a == dW-1)
					{
						#pragma omp critical
						{
							cout << "<L|R[dW-1]>=" << contract_LR(dW-1,Leigen,Rtmp) << endl;
						}
					}
				}
			}
		}
	}
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "linear systems" << GMresTimer.info() << endl;
	}
	
	YLlast = YL[0];
	YRfrst = YR[dW-1];
	
//	Tripod<Symmetry,MatrixType> Lcheck;
//	Tripod<Symmetry,MatrixType> Ltmp1=L;
//	
//	Tripod<Symmetry,MatrixType> Ltmp2;
//	for(int l=0; l<N_sites; l++)
//	{
//		contract_L(Ltmp1, AL[l], W[l], true, AL[l], qloc[l], qOp[l], Ltmp2, false, make_pair(FULL,0));
//		Ltmp1.clear();
//		Ltmp1 = Ltmp2;
//	}
//	Lcheck = Ltmp2;
//	
//	Tripod<Symmetry,MatrixType> Rcheck;
//	Tripod<Symmetry,MatrixType> Rtmp1=R;
//	
//	Tripod<Symmetry,MatrixType> Rtmp2;
//	for(int l=N_sites-1; l>=0; l--)
//	{
//		contract_R(Rtmp1, AR[l], W[l], true, AR[l], qloc[l], qOp[l], Rtmp2, false, make_pair(FULL,0));
//		Rtmp1.clear();
//		Rtmp1 = Rtmp2;
//	}
//	Rcheck = Rtmp2;
//	
//	cout << termcolor::magenta << "CHECK=" << L.compare(Lcheck) << "\t" << R.compare(Rcheck) << termcolor::reset << endl;
//	cout << (L-Lcheck).print(true,13) << endl;
//	cout << (R-Rcheck).print(true,13) << endl;
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
build_cellEnv (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout)
{
	// With a unit cell, Heff is a vector for each site
//	HeffA.clear();
//	HeffA.resize(N_sites);
	HeffC.clear();
	HeffC.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		HeffA[l].W = H.W[l];
		HeffC[l].W = H.W[l];
	}
	
	// Make environment for the unit cell
	build_LR (Vout.state.A[GAUGE::L], Vout.state.A[GAUGE::R], Vout.state.C, 
	          H.W, H.qloc, H.qOp, 
	          HeffA[0].L, HeffA[N_sites-1].R);
	
	// Make environment for each site of the unit cell
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			for (size_t l=1; l<N_sites; ++l)
			{
				contract_L(HeffA[l-1].L, 
				           Vout.state.A[GAUGE::L][l-1], H.W[l-1], PROP::HAMILTONIAN, Vout.state.A[GAUGE::L][l-1], 
				           H.locBasis(l-1), H.opBasis(l-1), 
				           HeffA[l].L);
			}
		}
		#pragma omp section
		{
			for (int l=N_sites-2; l>=0; --l)
			{
				contract_R(HeffA[l+1].R, 
				           Vout.state.A[GAUGE::R][l+1], H.W[l+1], PROP::HAMILTONIAN, Vout.state.A[GAUGE::R][l+1], 
				           H.locBasis(l+1), H.opBasis(l+1), 
				           HeffA[l].R);
			}
		}
	}
	
	for (size_t l=0; l<N_sites; ++l)
	{
		HeffC[l].L = HeffA[(l+1)%N_sites].L;
		HeffC[l].R = HeffA[l].R;
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
iteration_parallel (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout)
{
	Stopwatch<> IterationTimer;
	double tolLanczosEigval, tolLanczosState;
	set_LanczosTolerances(tolLanczosEigval,tolLanczosState);
	
	double t_exp   = 0.;
	double t_trunc = 0.;
//	cout << "N_iterations_without_expansion=" << N_iterations_without_expansion
//	     << ", max_iter_without_expansion=" << GlobParam.max_iter_without_expansion
//	     << ", min_iter_without_expansion=" << GlobParam.min_iter_without_expansion
//	     << boolalpha
//	     << ", err_var cond: " << (err_var < GlobParam.tol_var)
//	     << ", min cond: " << (N_iterations_without_expansion > GlobParam.min_iter_without_expansion)
//	     << ", max cond: " << (N_iterations_without_expansion > GlobParam.max_iter_without_expansion) << endl;
	
	// If: a) err_var has converged and minimal iteration number exceeded
	//     b) maximal iteration number exceeded
	if ((err_var < GlobParam.tol_var and N_iterations_without_expansion > GlobParam.min_iter_without_expansion) or
	    N_iterations_without_expansion > GlobParam.max_iter_without_expansion
	   )
	{
		Stopwatch<> ExpansionTimer;
		size_t deltaD = min(max(static_cast<size_t>(DynParam.Dincr_rel(N_iterations) * Vout.state.max_Nsv-Vout.state.max_Nsv), DynParam.Dincr_abs(N_iterations)),
		                    DynParam.max_deltaD(N_iterations));
		//make sure to perform at least one measurement before expanding the basis
		FORCE_DO_SOMETHING = true;
		DynParam.doSomething(N_iterations);
		FORCE_DO_SOMETHING = false;
		// if (Vout.state.calc_Dmax()+deltaD >= GlobParam.Dlimit) {deltaD = 0ul;}
		cout << "performing expansion with " << VUMPS::TWOSITE_A::ALxCxAR << endl;
		expand_basis(deltaD, H, Vout, VUMPS::TWOSITE_A::ALxCxAR);//);
		t_exp = ExpansionTimer.time();
		N_iterations_without_expansion = 0;
	}
	
	if (Vout.state.calc_fullMmax() <= GlobParam.fullMmaxBreakoff) // Don't need an extra iteration if fullMmaxBreakoff exceeded
	{
		if (err_var < GlobParam.tol_var and (N_iterations+1)%DynParam.Dincr_per(N_iterations) == 0 )
		{
			Stopwatch<> TruncationTimer;
			Vout.state.truncate();
			t_trunc = TruncationTimer.time();
		}
		
		Stopwatch<> EnvironmentTimer;
		build_cellEnv(H,Vout);
		double t_env = EnvironmentTimer.time();
		
		Stopwatch<> OptimizationTimer;
		// See Algorithm 4
		#pragma omp parallel for
		for (size_t l=0; l<N_sites; ++l)
		{
			precalc_blockStructure (HeffA[l].L, Vout.state.A[GAUGE::C][l], HeffA[l].W, Vout.state.A[GAUGE::C][l], HeffA[l].R, 
				                    H.locBasis(l), H.opBasis(l), HeffA[l].qlhs, HeffA[l].qrhs, HeffA[l].factor_cgcs);
			
			Eigenstate<PivotVector<Symmetry,Scalar> > gAC;
			Eigenstate<PivotVector<Symmetry,Scalar> > gC;
			
			// Solve for AC
			gAC.state = PivotVector<Symmetry,Scalar>(Vout.state.A[GAUGE::C][l]);
			
			Stopwatch<> LanczosTimer;
			LanczosSolver<PivotMatrix1<Symmetry,Scalar,Scalar>,PivotVector<Symmetry,Scalar>,Scalar> 
			Lutz(LanczosParam.REORTHO);
			Lutz.set_dimK(min(LanczosParam.dimK, dim(gAC.state)));
			Lutz.edgeState(HeffA[l], gAC, LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
			{
				#pragma omp critical
				{
					lout << "l=" << l << ", AC" << ", time" << LanczosTimer.info() << ", " << Lutz.info() << endl;
				}
			}
			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
			{
				#pragma omp critical
				{
					lout << "e0(AC)=" << setprecision(13) << gAC.energy << ", ratio=" << gAC.energy/Vout.energy << endl;
				}
			}
			
			// Solve for C
			gC.state = PivotVector<Symmetry,Scalar>(Vout.state.C[l]);
			
			LanczosSolver<PivotMatrix0<Symmetry,Scalar,Scalar>,PivotVector<Symmetry,Scalar>,Scalar> 
			Lucy(LanczosParam.REORTHO);
			Lucy.set_dimK(min(LanczosParam.dimK, dim(gC.state)));
			Lucy.edgeState(PivotMatrix0(HeffC[l]), gC, LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
			
			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
			{
				#pragma omp critical
				{
					lout << "l=" << l << ", C" << ", time" << LanczosTimer.info() << ", " << Lucy.info() << endl;
				}
			}
			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
			{
				#pragma omp critical
				{
					lout << "e0(C)=" << setprecision(13) << gC.energy << ", ratio=" << gC.energy/Vout.energy << endl;
				}
			}
			
			Vout.state.A[GAUGE::C][l] = gAC.state.data;
			Vout.state.C[l] = gC.state.data[0];
		}
		double t_opt = OptimizationTimer.time();
		
		Stopwatch<> SweepTimer;
		for (size_t l=0; l<N_sites; ++l)
		{
			(err_var>0.01)? Vout.state.svdDecompose(l) : Vout.state.polarDecompose(l);
		}
		Vout.state.calc_entropy((CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)? true : false);
		
	//	// Calculate energies
	//	Biped<Symmetry,ComplexMatrixType> Reigen_ = calc_LReigen(VMPS::DIRECTION::RIGHT, Vout.state.A[GAUGE::L], Vout.state.A[GAUGE::L], 
	//	                                                         Vout.state.outBasis(N_sites-1), Vout.state.outBasis(N_sites-1), Vout.state.qloc).state;
	//	Biped<Symmetry,ComplexMatrixType> Leigen_ = calc_LReigen(VMPS::DIRECTION::LEFT, Vout.state.A[GAUGE::R], Vout.state.A[GAUGE::R],
	//	                                                         Vout.state.inBasis(0), Vout.state.inBasis(0), Vout.state.qloc).state;
	//	complex<double> eL_ = contract_LR(0,    YLlast.template cast<ComplexMatrixType>(), Reigen_) / static_cast<double>(H.volume());
	//	complex<double> eR_ = contract_LR(dW-1, Leigen_, YRfrst.template cast<ComplexMatrixType>()) / static_cast<double>(H.volume());
	////	complex<double> eL_ = contract_LR(YLlast.template cast<ComplexMatrixType>(), Reigen_) / static_cast<double>(H.volume());
	////	complex<double> eR_ = contract_LR(Leigen_, YRfrst.template cast<ComplexMatrixType>()) / static_cast<double>(H.volume());
	//	cout << termcolor::blue << "eL_=" << eL_ << ", eR_=" << eR_ << termcolor::reset << endl;
		
		Biped<Symmetry,MatrixType> Reigen, Leigen;
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				Reigen = Vout.state.C[N_sites-1].contract(Vout.state.C[N_sites-1].adjoint());
				eL = contract_LR(0, YLlast, Reigen) / H.volume();
			}
			#pragma omp section
			{
				Leigen = Vout.state.C[N_sites-1].adjoint().contract(Vout.state.C[N_sites-1]);
				eR = contract_LR(dW-1, Leigen, YRfrst) / H.volume();
			}
		}
		Vout.energy = min(eL,eR);
		
		double t_sweep = SweepTimer.time();
		
		Stopwatch<> ErrorTimer;
		double t_err = 0;
		cout << "err_state_rel=" << abs(err_state_old-err_state)/err_state << endl;
		if (abs(err_state_old-err_state)/err_state > 1e-3 or N_iterations_without_expansion<=1 or N_iterations<=6)
		{
			calc_errors(H, Vout, true);
			t_err = ErrorTimer.time();
			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
			{
				lout << ErrorTimer.info("error calculation") << endl;
			}
		}
		else
		{
			calc_errors(H, Vout, false);
			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
			{
				lout << "State error seems converged and will be not recalculated until the next expansion!" << endl;
			}
		}
		
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
		{
			lout << Vout.state.test_ortho() << endl;
			lout << termcolor::blue << "eL=" << eL << ", eR=" << eR << termcolor::reset << endl;
			lout << test_LReigen(Vout) << endl;
		}
		
		++N_iterations;
		++N_iterations_without_expansion;
		
		double t_tot = IterationTimer.time();
		// print stuff
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			size_t standard_precision = cout.precision();
			lout << termcolor::bold << eigeninfo() << termcolor::reset << endl;
			
			lout << Vout.state.info() << endl;
			lout << IterationTimer.info("full parallel iteration") 
				 << " (environment=" << round(t_env/t_tot*100.,0)  << "%" 
				 << ", optimization=" << round(t_opt/t_tot*100.,0)  << "%" 
				 << ", sweep=" << round(t_sweep/t_tot*100.,0) << "%" 
				 << ", error=" << round(t_err/t_tot*100.,0) << "%";
			if (t_exp != 0.)  {lout << ", basis expansion="  << round(t_exp/t_tot*100.,0)   << "%";}
			if (t_trunc != 0) {lout << ", basis truncation=" << round(t_trunc/t_tot*100.,0) << "%";}
			lout << ")"<< endl;
			lout << endl;
		}
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
iteration_sequential (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout)
{
	Stopwatch<> IterationTimer;
	
	double tolLanczosEigval, tolLanczosState;
	set_LanczosTolerances(tolLanczosEigval,tolLanczosState);
	
	// See Algorithm 3
	for (size_t l=0; l<N_sites; ++l)
	{
		if (err_var < GlobParam.tol_var and N_iterations%DynParam.Dincr_per(N_iterations) == 0 )
		{
			size_t deltaD = min(max(static_cast<size_t>(DynParam.Dincr_rel(N_iterations) * Vout.state.max_Nsv-Vout.state.max_Nsv), DynParam.Dincr_abs(N_iterations)),
			                    DynParam.max_deltaD(N_iterations));
			if (Vout.state.calc_Dmax()+deltaD >= GlobParam.Dlimit) {deltaD = 0ul;}
			expand_basis(l,deltaD,H,Vout,VUMPS::TWOSITE_A::ALxAC);
		}
		
		build_cellEnv(H,Vout);
		
		precalc_blockStructure (HeffA[l].L, Vout.state.A[GAUGE::C][l], HeffA[l].W, Vout.state.A[GAUGE::C][l], HeffA[l].R, 
		                        H.locBasis(l), H.opBasis(l), HeffA[l].qlhs, HeffA[l].qrhs, HeffA[l].factor_cgcs);
		
		Eigenstate<PivotVector<Symmetry,Scalar> > gAC;
		Eigenstate<PivotVector<Symmetry,Scalar> > gCR;
		Eigenstate<PivotVector<Symmetry,Scalar> > gCL;
		
		// Solve for AC
		gAC.state = PivotVector<Symmetry,Scalar>(Vout.state.A[GAUGE::C][l]);
		
		Stopwatch<> LanczosTimer;
		LanczosSolver<PivotMatrix1<Symmetry,Scalar,Scalar>,PivotVector<Symmetry,Scalar>,Scalar> 
		Lutz(LanczosParam.REORTHO);
		Lutz.set_dimK(min(LanczosParam.dimK, dim(gAC.state)));
		Lutz.edgeState(HeffA[l], gAC, LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			lout << "l=" << l << ", AC" << ", time" << LanczosTimer.info() << ", " << Lutz.info() << endl;
		}
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
		{
			lout << "e0(AC)=" << setprecision(13) << gAC.energy << ", ratio=" << gAC.energy/Vout.energy << endl;
		}
		
		// Solve for CR
		gCR.state = PivotVector<Symmetry,Scalar>(Vout.state.C[l]);
		
		LanczosSolver<PivotMatrix0<Symmetry,Scalar,Scalar>,PivotVector<Symmetry,Scalar>,Scalar> 
		Lucy(LanczosParam.REORTHO);
		Lucy.set_dimK(min(LanczosParam.dimK, dim(gCR.state)));
		Lucy.edgeState(PivotMatrix0(HeffC[l]), gCR, LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
		//ensure phase convention: first element is positive
		if (gCR.state.data[0].block[0](0,0) < 0.) { gCR.state.data[0] = (-1.) * gCR.state.data[0]; }
		
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			lout << "l=" << l << ", CR" << ", time" << LanczosTimer.info() << ", " << Lucy.info() << endl;
		}
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
		{
			lout << "e0(C)=" << setprecision(13) << gCR.energy << ", ratio=" << gCR.energy/Vout.energy << endl;
		}
		
		// Solve for CL
		size_t lC = Vout.state.minus1modL(l);
		gCL.state = PivotVector<Symmetry,Scalar>(Vout.state.C[lC]);
		
		LanczosSolver<PivotMatrix0<Symmetry,Scalar,Scalar>,PivotVector<Symmetry,Scalar>,Scalar> 
		Luca(LanczosParam.REORTHO);
		Luca.set_dimK(min(LanczosParam.dimK, dim(gCL.state)));
		Luca.edgeState(PivotMatrix0(HeffC[lC]), gCL, LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
		//ensure phase convention: first element is positive
		if (gCL.state.data[0].block[0](0,0) < 0.) { gCL.state.data[0] = (-1.) * gCL.state.data[0]; }
		
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			lout << "l=" << l << ", CL" << ", time" << LanczosTimer.info() << ", " << Luca.info() << endl;
		}
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
		{
			lout << "e0(C)=" << setprecision(13) << gCL.energy << ", ratio=" << gCL.energy/Vout.energy << endl;
		}
		
		Vout.state.A[GAUGE::C][l] = gAC.state.data;
		Vout.state.C[lC]          = gCL.state.data[0]; // C(l-1 mod L) = CL
		Vout.state.C[l]           = gCR.state.data[0]; // C(l)         = CR
		
		(err_var>0.01)? Vout.state.svdDecompose(l,GAUGE::R) : Vout.state.polarDecompose(l,GAUGE::R); // AR from AC, CL
		(err_var>0.01)? Vout.state.svdDecompose(l,GAUGE::L) : Vout.state.polarDecompose(l,GAUGE::L); // AL from AC, CR
	}
	
	Vout.state.calc_entropy((CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)? true : false);
	
	// Calculate energies
	Biped<Symmetry,MatrixType> Reigen = Vout.state.C[N_sites-1].contract(Vout.state.C[N_sites-1].adjoint());
	Biped<Symmetry,MatrixType> Leigen = Vout.state.C[N_sites-1].adjoint().contract(Vout.state.C[N_sites-1]);
	eL = contract_LR(0, YLlast, Reigen) / H.volume();
	eR = contract_LR(dW-1, Leigen, YRfrst) / H.volume();
	
	Vout.energy = min(eL,eR);
	
	Stopwatch<> ErrorTimer;
	double t_err = 0;
	if (abs(err_state_old-err_state)/err_state_old > 0.001 or N_iterations_without_expansion<=1 or N_iterations<=6)
	{
		calc_errors(H, Vout, true);
		t_err = ErrorTimer.time();
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			lout << ErrorTimer.info("error calculation") << endl;
		}
	}
	else
	{
		calc_errors(H, Vout, false);
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			lout << "State error seems converged and will be not recalculated until the next expansion!" << endl;
		}
	}
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
	{
		lout << Vout.state.test_ortho() << endl;
		lout << termcolor::blue << "eL=" << eL << ", eR=" << eR << termcolor::reset << endl;
		lout << test_LReigen(Vout) << endl;
	}
		
	++N_iterations;
	
	// print stuff
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		size_t standard_precision = cout.precision();
		lout << "S=" << Vout.state.entropy().transpose() << endl;
		lout << termcolor::bold << eigeninfo() << termcolor::reset << endl;
		lout << IterationTimer.info("full sequential iteration") << endl;
		lout << endl;
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
iteration_h2site (Eigenstate<Umps<Symmetry,Scalar> > &Vout)
{
	Stopwatch<> IterationTimer;
	
	// |R) and (L|
	Biped<Symmetry,MatrixType> Reigen = Vout.state.C[N_sites-1].contract(Vout.state.C[N_sites-1].adjoint());
	Biped<Symmetry,MatrixType> Leigen = Vout.state.C[N_sites-1].adjoint().contract(Vout.state.C[N_sites-1]);
	
	// |h_R) and (h_L|
	Biped<Symmetry,MatrixType> hR = make_hR(h2site, Vout.state.A[GAUGE::R][0], Vout.state.locBasis(0));
	Biped<Symmetry,MatrixType> hL = make_hL(h2site, Vout.state.A[GAUGE::L][0], Vout.state.locBasis(0));
	
	// energies
	eL = (Leigen.contract(hR)).trace();
	eR = (hL.contract(Reigen)).trace();
	
	// |H_R) and (H_L|
	Biped<Symmetry,MatrixType> HL, HR;
	
	// Solve the linear systems in eq. 14
	Stopwatch<> GMresTimer;
	solve_linear(VMPS::DIRECTION::LEFT,  Vout.state.A[GAUGE::L], hL, Reigen, Vout.state.locBasis(), eR, HL);
	solve_linear(VMPS::DIRECTION::RIGHT, Vout.state.A[GAUGE::R], hR, Leigen, Vout.state.locBasis(), eL, HR);
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "linear systems" << GMresTimer.info() << endl;
	}
	
	// Doesn't work like that!! boost::multi_array is shit!
//	Heff[0] = PivumpsMatrix1<Symmetry,Scalar,Scalar>(HL, HR, h2site, Vout.state.A[GAUGE::L][0], Vout.state.A[GAUGE::R][0]);
	
	Heff[0].h.resize(boost::extents[D][D][D][D]);
	Heff[0].h = h2site;
	Heff[0].L = HL;
	Heff[0].R = HR;
	Heff[0].AL = Vout.state.A[GAUGE::L][0];
	Heff[0].AR = Vout.state.A[GAUGE::R][0];
	Heff[0].qloc = Vout.state.locBasis(0);
	
	double tolLanczosEigval, tolLanczosState;
	set_LanczosTolerances(tolLanczosEigval,tolLanczosState);
	
	// Solve for AC (eq. 11)
	Eigenstate<PivotVector<Symmetry,Scalar> > gAC;
	gAC.state = PivotVector<Symmetry,Scalar>(Vout.state.A[GAUGE::C][0]);
	
	Stopwatch<> LanczosTimer;
	LanczosSolver<PivumpsMatrix1<Symmetry,Scalar,Scalar>,PivotVector<Symmetry,Scalar>,Scalar> Lutz1(LanczosParam.REORTHO);
	Lutz1.set_dimK(min(LanczosParam.dimK, dim(gAC.state)));
	Lutz1.edgeState(Heff[0],gAC, LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "time" << LanczosTimer.info() << ", " << Lutz1.info() << endl;
	}
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
	{
		lout << "e0(AC)=" << setprecision(13) << gAC.energy << endl;
	}
	
	// Solve for C (eq. 16)
	Eigenstate<PivotVector<Symmetry,Scalar> > gC;
	gC.state = PivotVector<Symmetry,Scalar>(Vout.state.C[0]);
	
	LanczosSolver<PivumpsMatrix0<Symmetry,Scalar,Scalar>,PivotVector<Symmetry,Scalar>,Scalar> Lutz0(LanczosParam.REORTHO);
	Lutz0.set_dimK(min(LanczosParam.dimK, dim(gC.state)));
	Lutz0.edgeState(PivumpsMatrix0(Heff[0]),gC, LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "time" << LanczosTimer.info() << ", " << Lutz0.info() << endl;
	}
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
	{
		lout << "e0(C)=" << setprecision(13) << gC.energy << endl;
	}
	
	// Calculate AL and AR from AC, C
	Vout.state.A[GAUGE::C][0] = gAC.state.data;
	Vout.state.C[0]           = gC.state.data[0];
	(err_var>0.01)? Vout.state.svdDecompose(0) : Vout.state.polarDecompose(0);
	
	Vout.state.calc_entropy((CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)? true : false);
	
	// calc_errors(Vout);
	Vout.energy = min(eL,eR);
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
	{
		lout << Vout.state.test_ortho() << endl;
		lout << termcolor::blue << "eL=" << eL << ", eR=" << eR << termcolor::reset << endl;
		lout << test_LReigen(Vout) << endl;
	}
	
	++N_iterations;
	
	// Print stuff
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		size_t standard_precision = cout.precision();
		lout << "S=" << Vout.state.entropy().transpose() << endl;
		lout << eigeninfo() << endl;
		lout << IterationTimer.info("full iteration") << endl;
		lout << endl;
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
iteration_idmrg (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout)
{
	assert(H.length() == 2);
	Stopwatch<> IterationTimer;
	
	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > Atmp;
	contract_AA (Vout.state.A[GAUGE::C][0], H.locBasis(0), 
	             Vout.state.A[GAUGE::C][0], H.locBasis(1), 
	             Vout.state.Qtop(0), Vout.state.Qbot(0),
	             Atmp);
	for (size_t s=0; s<Atmp.size(); ++s)
	{
		Atmp[s] = Atmp[s].cleaned();
	}
	
	if (HeffA[0].W.size() == 0)
	{
		contract_WW<Symmetry,Scalar> (H.W_at(0), H.locBasis(0), H.opBasis(0), 
		                              H.W_at(1), H.locBasis(1), H.opBasis(1),
		                              HeffA[0].W, HeffA[0].qloc, HeffA[0].qOp);
	}
	
	Eigenstate<PivotVector<Symmetry,Scalar> > g;
	g.state = PivotVector<Symmetry,Scalar>(Atmp);
	
//	if (HeffA[0].qlhs.size() == 0)
	{
		HeffA[0].qlhs.clear();
		HeffA[0].qrhs.clear();
		HeffA[0].factor_cgcs.clear();
		precalc_blockStructure (HeffA[0].L, Atmp, HeffA[0].W, Atmp, HeffA[0].R, 
		                        HeffA[0].qloc, HeffA[0].qOp, 
		                        HeffA[0].qlhs, HeffA[0].qrhs, HeffA[0].factor_cgcs);
	}
	
	Stopwatch<> LanczosTimer;
	LanczosSolver<PivotMatrix1<Symmetry,Scalar,Scalar>,PivotVector<Symmetry,Scalar>,Scalar> 
	Lutz(LanczosParam.REORTHO);
	Lutz.set_dimK(min(LanczosParam.dimK, dim(g.state)));
	Lutz.edgeState(HeffA[0], g, LANCZOS::EDGE::GROUND, DMRG::CONTROL::DEFAULT::tol_eigval_Lanczos, DMRG::CONTROL::DEFAULT::tol_state_Lanczos, false);
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "time" << LanczosTimer.info() << ", " << Lutz.info() << endl;
	}
	
	auto Cref = Vout.state.C[0];
	
	// Mps<Symmetry,Scalar> Vtmp(2, H.locBasis(), Symmetry::qvacuum(), 2, Vout.state.Nqmax);
	// Vtmp.min_Nsv = M;
	// Vtmp.max_Nsv = M;
	// Vtmp.A[0] = Vout.state.A[GAUGE::C][0];
	// Vtmp.A[1] = Vout.state.A[GAUGE::C][0];
	// Vtmp.QinTop[0] = Vout.state.Qtop(0);
	// Vtmp.QinBot[0] = Vout.state.Qbot(0);
	// Vtmp.QoutTop[0] = Vout.state.Qtop(0);
	// Vtmp.QoutBot[0] = Vout.state.Qbot(0);
	// Vtmp.QinTop[1] = Vout.state.Qtop(1);
	// Vtmp.QinBot[1] = Vout.state.Qbot(1);
	// Vtmp.QoutTop[1] = Vout.state.Qtop(1);
	// Vtmp.QoutBot[1] = Vout.state.Qbot(1);
	// Vtmp.sweepStep2(DMRG::DIRECTION::RIGHT, 0, g.state.data, 
	//                 Vout.state.A[GAUGE::L][0], Vout.state.A[GAUGE::R][0], Vout.state.C[0],
	//                 true);
	
	double truncDump, Sdump;
	
	split_AA(DMRG::DIRECTION::RIGHT, g.state.data,
			 Vout.state.locBasis(0), Vout.state.A[GAUGE::L][0],
			 Vout.state.locBasis(0), Vout.state.A[GAUGE::R][0],
			 Vout.state.Qtop(0), Vout.state.Qbot(0),
			 Vout.state.C[0], true, truncDump, Sdump,
			 Vout.state.eps_svd,Vout.state.min_Nsv,Vout.state.max_Nsv);
	Vout.state.C[0] = 1./sqrt((Vout.state.C[0].contract(Vout.state.C[0].adjoint())).trace()) * Vout.state.C[0];
	
	Vout.state.calc_entropy((CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)? true : false);
	
	for (size_t s=0; s<H.locBasis(0).size(); ++s)
	{
		Vout.state.A[GAUGE::C][0][s] = Vout.state.A[GAUGE::L][0][s] *Vout.state.C[0];
//		Vout.state.A[GAUGE::C][0][s] = Vout.state.C[0].contract(Vout.state.A[GAUGE::R][0][s]);
	}
	
	++N_iterations;
	
	Vout.energy = 0.5*g.energy-Eold;
	err_eigval = abs(Vout.energy-eL); // the energy density is the contribution of the new site
	err_state = (Cref-Vout.state.C[0]).norm();
	err_var = err_state;
	Eold = 0.5*g.energy;
	eL = Vout.energy;
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
	{
		lout << Vout.state.test_ortho() << endl;
	}
	
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > HeffLtmp, HeffRtmp;
	contract_L(HeffA[0].L, Vout.state.A[GAUGE::L][0], H.W[0], PROP::HAMILTONIAN, Vout.state.A[GAUGE::L][0], H.locBasis(0), H.opBasis(0), HeffLtmp);
	HeffA[0].L = HeffLtmp;
	contract_R(HeffA[0].R, Vout.state.A[GAUGE::R][0], H.W[1], PROP::HAMILTONIAN, Vout.state.A[GAUGE::R][0], H.locBasis(1), H.opBasis(1), HeffRtmp);
	HeffA[0].R = HeffRtmp;
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		size_t standard_precision = cout.precision();
		lout << "S=" << Vout.state.entropy().transpose() << endl;
		lout << termcolor::bold << eigeninfo() << termcolor::reset << endl;
		lout << IterationTimer.info("full iteration") << endl;
		lout << endl;
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
string VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
test_LReigen (const Eigenstate<Umps<Symmetry,Scalar> > &Vout) const
{
	TransferMatrix<Symmetry,Scalar> TR(VMPS::DIRECTION::RIGHT, Vout.state.A[GAUGE::R], Vout.state.A[GAUGE::R], Vout.state.qloc);
	TransferMatrix<Symmetry,Scalar> TL(VMPS::DIRECTION::LEFT,  Vout.state.A[GAUGE::L], Vout.state.A[GAUGE::L], Vout.state.qloc);
	
	Biped<Symmetry,MatrixType> Reigen = Vout.state.C[N_sites-1].contract(Vout.state.C[N_sites-1].adjoint());
	Biped<Symmetry,MatrixType> Leigen = Vout.state.C[N_sites-1].adjoint().contract(Vout.state.C[N_sites-1]);
	
	TransferVector<Symmetry,Scalar> PsiR(Reigen);
	TransferVector<Symmetry,Scalar> PsiL(Leigen);
	
	HxV(TL,PsiR);
	HxV(TR,PsiL);
	
	stringstream ss;
	ss << "ReigenTest=" << (Reigen-PsiR.data).norm() << ", LeigenTest=" << (Leigen-PsiL.data).norm() << endl;
	return ss.str();
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
edgeState (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout, qarray<Symmetry::Nq> Qtot, LANCZOS::EDGE::OPTION EDGE, bool USE_STATE)
{
	if (CHOSEN_VERBOSITY>=2)
	{
		lout << endl << termcolor::colorize << termcolor::bold
		 << "———————————————————————————————————————————————VUMPS algorithm—————————————————————————————————————————————————————————"
		 <<  termcolor::reset << endl;
	}
	
	if (!USER_SET_GLOBPARAM) {GlobParam = H.get_VumpsGlobParam();}
	if (!USER_SET_DYNPARAM)  {DynParam  = H.get_VumpsDynParam();}
	
	// tol_eigval = tol_eigval_input;
	// tol_var = tol_var_input;
	
	if (DynParam.iteration(0) == UMPS_ALG::IDMRG)
	{
		prepare_idmrg(H, Vout, Qtot, USE_STATE);
	}
	else if (DynParam.iteration(0) == UMPS_ALG::H2SITE)
	{
		assert(H.length() == 2 and "Need L=2 for H2SITE!");
		for (size_t i=0; i<GlobParam.max_iterations; i++) 
		{
			assert(DynParam.iteration(i) == UMPS_ALG::H2SITE and "iteration H2SITE can not be mixed with other iterations");
		}
		prepare_h2site(H.H2site(0,true), H.locBasis(0), Vout, Qtot, GlobParam.Dinit, GlobParam.Qinit, USE_STATE);
	}
	else
	{
		prepare(H, Vout, Qtot, USE_STATE);
	}
	
	Stopwatch<> GlobalTimer;
	
	while (((err_eigval >= GlobParam.tol_eigval or err_state >= GlobParam.tol_state) and N_iterations < GlobParam.max_iterations) or 
	       N_iterations < GlobParam.min_iterations)
	{
		if (DynParam.iteration(N_iterations) == UMPS_ALG::PARALLEL)
		{
			iteration_parallel(H,Vout);
		}
		else if (DynParam.iteration(N_iterations) == UMPS_ALG::SEQUENTIAL)
		{
			iteration_sequential(H,Vout);
		}
		else if (DynParam.iteration(N_iterations) == UMPS_ALG::IDMRG)
		{
			iteration_idmrg(H,Vout);
		}
		else if (DynParam.iteration(N_iterations) == UMPS_ALG::H2SITE)
		{
			iteration_h2site(Vout);
		}
		else // dynamical choice: L=1 parallel, L>1 sequential
		{
			if (N_sites == 1) { iteration_parallel(H,Vout); }
			else              { iteration_sequential(H,Vout); }
		}
		
		DynParam.doSomething(N_iterations);
		
		write_log();
		#ifdef USE_HDF5_STORAGE
		if (GlobParam.savePeriod != 0 and N_iterations%GlobParam.savePeriod == 0)
		{
			string filename = make_string(GlobParam.saveName,"_fullMmax=",Vout.state.calc_fullMmax());
			lout << termcolor::green << "saving state to: " << filename << termcolor::reset << endl;
			Vout.state.save(filename,H.info());
		}
		#endif
		
		if (Vout.state.calc_fullMmax() > GlobParam.fullMmaxBreakoff)
		{
			lout << "Terminating because the bond dimension " << Vout.state.calc_fullMmax() 
			     << " exceeds " << GlobParam.fullMmaxBreakoff << "!" << endl;
			break;
		}
	}
	write_log(true); // force log on exit
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::ON_EXIT)
	{
		lout << GlobalTimer.info("total runtime") << endl;
		size_t standard_precision = cout.precision();
		lout << termcolor::bold << setprecision(14)
		     << "iterations=" << N_iterations
		     << ", e0=" << Vout.energy 
		     << ", err_eigval=" << err_eigval 
		     << ", err_var=" << err_var
		     << ", err_state=" << err_state
		     << setprecision(standard_precision) << termcolor::reset
		     << endl;
		lout << Vout.state.info() << endl;
		lout << endl;
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
solve_linear (VMPS::DIRECTION::OPTION DIR, 
              size_t ab, 
              const vector<vector<Biped<Symmetry,MatrixType> > > &A, 
              const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &Y_LR, 
              const Biped<Symmetry,MatrixType> &LReigen, 
              const vector<vector<vector<vector<SparseMatrix<Scalar> > > > > &W, 
              const vector<vector<qarray<Symmetry::Nq> > > &qloc, 
              const vector<vector<qarray<Symmetry::Nq> > > &qOp,
              Scalar LRdotY,
              const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &LRguess,  
              Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &LRres)
{
	MpoTransferMatrix<Symmetry,Scalar> T(DIR, A, A, LReigen, W, qloc, qOp, ab);
	MpoTransferVector<Symmetry,Scalar> bvec(Y_LR, ab, LRdotY); // right-hand site vector |Y_LR)-e*1
	
	// Solve linear system
	GMResSolver<MpoTransferMatrix<Symmetry,Scalar>,MpoTransferVector<Symmetry,Scalar> > Gimli;
	
	Gimli.set_dimK(min(100ul,dim(bvec)));
	MpoTransferVector<Symmetry,Scalar> LRres_tmp;
	Gimli.solve_linear(T, bvec, LRres_tmp, 1e-14, true);
	LRres = LRres_tmp.data;
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << DIR << ": " << Gimli.info() << endl;
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
solve_linear (VMPS::DIRECTION::OPTION DIR, 
              const vector<vector<Biped<Symmetry,MatrixType> > > &A, 
              const Biped<Symmetry,MatrixType> &hLR, 
              const Biped<Symmetry,MatrixType> &LReigen, 
              const vector<vector<qarray<Symmetry::Nq> > > &qloc, 
              Scalar hLRdotLR, 
              Biped<Symmetry,MatrixType> &LRres)
{
	TransferMatrix<Symmetry,Scalar> T(DIR,A,A,qloc,true);
	T.LReigen = LReigen;
	TransferVector<Symmetry,Scalar> bvec(hLR);
	
	for (size_t q=0; q<bvec.data.dim; ++q)
	{
		bvec.data.block[q] -= hLRdotLR * Matrix<Scalar,Dynamic,Dynamic>::Identity(bvec.data.block[q].rows(),
		                                                                          bvec.data.block[q].cols());
	}
	
	// Solve linear system
	GMResSolver<TransferMatrix<Symmetry,Scalar>,TransferVector<Symmetry,Scalar> > Gimli;
	
	Gimli.set_dimK(min(100ul,dim(bvec)));
	TransferVector<Symmetry,Scalar> LRres_tmp;
	Gimli.solve_linear(T,bvec,LRres_tmp);
	LRres = LRres_tmp.data;
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << DIR << ": " << Gimli.info() << endl;
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
expand_basis (size_t loc, size_t DeltaD, const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout, VUMPS::TWOSITE_A::OPTION option)
{
	//early return if one actually wants to do nothing.
	if (DeltaD == 0ul) {return;}
	
	vector<Biped<Symmetry,MatrixType> > NL;
	vector<Biped<Symmetry,MatrixType> > NR;
	Biped<Symmetry,MatrixType> NAAN;
	
	//calculate two-site B-Tensor (double tangent space) and obtain simultaneously NL and NR (nullspaces)
	calc_B2(loc, H, Vout, option, NAAN, NL, NR);
	
	// SVD-decompose NAAN
	Biped<Symmetry,MatrixType> U, Vdag;
	for (size_t q=0; q<NAAN.dim; ++q)
	{
		#ifdef DONT_USE_BDCSVD
		JacobiSVD<MatrixType> Jack; // standard SVD
		#else
		BDCSVD<MatrixType> Jack; // "Divide and conquer" SVD (only available in Eigen)
		#endif
		
		Jack.compute(NAAN.block[q], ComputeThinU|ComputeThinV);
		
		size_t Nret = (Jack.singularValues().array() > Vout.state.eps_svd).count();
		Nret = min(DeltaD, Nret);
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			lout << "q=" << NAAN.in[q] << ": Nret=" << Nret << ", ";
		}
		if (Nret > 0)
		{
			U.push_back(NAAN.in[q], NAAN.out[q], Jack.matrixU().leftCols(Nret));
			Vdag.push_back(NAAN.in[q], NAAN.out[q], Jack.matrixV().adjoint().topRows(Nret));
		}
	}
	lout << endl << endl;
	
	//calc P
	vector<Biped<Symmetry,MatrixType> > P(Vout.state.locBasis(loc).size());
	for (size_t s=0; s<Vout.state.locBasis(loc).size(); ++s)
	{
		P[s] = NL[s] * U;
	}
	Vout.state.enrich(loc, GAUGE::L, P);
	
	//Update the left environment if AL is involved in calculating the two site A-tensor, because we need correct environments for the effective two-site Hamiltonian.
	if (option == VUMPS::TWOSITE_A::ALxAC or option == VUMPS::TWOSITE_A::ALxCxAR)
	{
		contract_L(HeffA[loc].L, 
		           Vout.state.A[GAUGE::L][loc], H.W[loc], PROP::HAMILTONIAN, Vout.state.A[GAUGE::L][loc], 
		           H.locBasis(loc), H.opBasis(loc), 
		           HeffA[(loc+1)%N_sites].L);
	}
	
	P.clear();
	P.resize(Vout.state.locBasis((loc+1)%N_sites).size());
	for (size_t s=0; s<Vout.state.locBasis((loc+1)%N_sites).size(); ++s)
	{
		P[s] = Vdag * NR[s];
	}
	Vout.state.enrich(loc, GAUGE::R, P);
	
	//Update the right environment if AR is involved in calculating the two site A-tensor, because we need correct environments for the effective two-site Hamiltonian.
	//Note: maybe we only have to update the right environment if loc=0, since we need the updated environment only when loc=N_sites-1 and consequentially loc+1=0.
	if (option == VUMPS::TWOSITE_A::ACxAR or option == VUMPS::TWOSITE_A::ALxCxAR)
	{
		contract_R(HeffA[(loc+1)%N_sites].R,
		           Vout.state.A[GAUGE::R][(loc+1)%N_sites], H.W[(loc+1)%N_sites], PROP::HAMILTONIAN, Vout.state.A[GAUGE::R][(loc+1)%N_sites], 
		           H.locBasis((loc+1)%N_sites), H.opBasis((loc+1)%N_sites), 
		           HeffA[loc].R);
	}
	
	Vout.state.update_outbase(loc,GAUGE::L);
	
	//update C with zeros
	Vout.state.updateC(loc);
	
	//update AC with zeros at sites loc and loc+1
	Vout.state.updateAC(loc,GAUGE::L);
	Vout.state.updateAC((loc+1)%N_sites,GAUGE::L);
	
	// sort
	Vout.state.C[loc] = Vout.state.C[loc].sorted();
	Vout.state.sort_A(loc, GAUGE::L, true); //true means sort all gauges, the parameter GAUGE::L has no impact here.
	Vout.state.sort_A((loc+1)%N_sites, GAUGE::L, true); //true means sort all gauges, the parameter GAUGE::L has no impact here.
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
expand_basis (size_t DeltaD, const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout, VUMPS::TWOSITE_A::OPTION option)
{
	for (size_t loc=0; loc<N_sites; loc++)
	{
		expand_basis(loc, DeltaD, H, Vout, option);
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
calc_B2 (size_t loc, const MpHamiltonian &H, const Eigenstate<Umps<Symmetry,Scalar> > &Vout, VUMPS::TWOSITE_A::OPTION option,
         Biped<Symmetry,MatrixType>& B2, vector<Biped<Symmetry,MatrixType> > &NL, vector<Biped<Symmetry,MatrixType> > &NR) const
{
	Vout.state.calc_N(DMRG::DIRECTION::RIGHT, loc,             NL);
	Vout.state.calc_N(DMRG::DIRECTION::LEFT,  (loc+1)%N_sites, NR);
	
	PivotMatrix2<Symmetry,Scalar> H2(HeffA[loc].L, HeffA[(loc+1)%N_sites].R, HeffA[loc].W, HeffA[(loc+1)%N_sites].W, 
	                                 H.locBasis(loc), H.locBasis((loc+1)%N_sites), H.opBasis(loc), H.opBasis((loc+1)%N_sites));
	
	vector<Biped<Symmetry,MatrixType> > AL;
	vector<Biped<Symmetry,MatrixType> > AR;
	
	if (option == VUMPS::TWOSITE_A::ALxAC)
	{
		AL = Vout.state.A[GAUGE::L][loc];
		AR = Vout.state.A[GAUGE::C][(loc+1)%N_sites];
	}
	else if (option == VUMPS::TWOSITE_A::ACxAR)
	{
		assert(1!=1 and "The option ACxAR causes bugs. Fix them first to use it.");
		AL = Vout.state.A[GAUGE::C][loc];
		AR = Vout.state.A[GAUGE::R][(loc+1)%N_sites];
	}
	else if (option == VUMPS::TWOSITE_A::ALxCxAR)
	{
		AL.resize(Vout.state.A[GAUGE::L][loc].size());
		//Set AL to A[GAUGE::L]*C
		for (size_t s=0; s<Vout.state.A[GAUGE::L][loc].size(); ++s)
		{
			AL[s] = Vout.state.A[GAUGE::L][loc][s] * Vout.state.C[loc];
		}
		AR = Vout.state.A[GAUGE::R][(loc+1)%N_sites];
	}
	else
	{
		assert(1!=1 and "You inserted an invalid value for enum VUMPS::TWOSITEA::OPTION in calc_B2 from VumpsSolver.");
	}
	
	PivotVector<Symmetry,Scalar> A2C(AL, H.locBasis(loc), 
	                                 AR, H.locBasis((loc+1)%N_sites), 
	                                 Vout.state.Qtop(loc), Vout.state.Qbot((loc+1)%N_sites));
	
	precalc_blockStructure (HeffA[loc].L, A2C.data, HeffA[loc].W, HeffA[(loc+1)%N_sites].W, A2C.data, HeffA[(loc+1)%N_sites].R, 
	                        H.locBasis(loc), H.locBasis((loc+1)%N_sites), H.opBasis(loc), H.opBasis((loc+1)%N_sites), 
	                        H2.qlhs, H2.qrhs, H2.factor_cgcs);
	HxV(H2,A2C);
	
	split_AA(DMRG::DIRECTION::RIGHT, A2C.data, H.locBasis(loc), AL, H.locBasis((loc+1)%N_sites), AR,
	         Vout.state.Qtop(loc), Vout.state.Qbot(loc),
	         Vout.state.eps_svd,Vout.state.min_Nsv,Vout.state.max_Nsv);
	
	Qbasis<Symmetry> NRbasis; NRbasis.pullData(NR,1);
	Qbasis<Symmetry> NLbasis; NLbasis.pullData(NL,0);
	Qbasis<Symmetry> ARbasis; ARbasis.pullData(AR,1);
	Qbasis<Symmetry> ALbasis; ALbasis.pullData(AL,0);
	
	// calculate B2 = NAAN
	Biped<Symmetry,MatrixType> IdL; IdL.setIdentity(NLbasis, ALbasis);
	Biped<Symmetry,MatrixType> IdR; IdR.setIdentity(ARbasis, NRbasis);
	
	Biped<Symmetry,MatrixType> TL, TR;
	contract_L(IdL, NL, AL, H.locBasis(loc),             TL);
	contract_R(IdR, NR, AR, H.locBasis((loc+1)%N_sites), TR);
	
	B2 = TL.contract(TR);
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
calc_B2 (size_t loc, const MpHamiltonian &H, const Eigenstate<Umps<Symmetry,Scalar> > &Vout, VUMPS::TWOSITE_A::OPTION option, Biped<Symmetry,MatrixType>& B2) const
{
	vector<Biped<Symmetry,MatrixType> > NL_dump, NR_dump;
	calc_B2(loc,H,Vout,option,B2,NL_dump,NR_dump);
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
Mps<Symmetry,Scalar> VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
create_Mps (size_t Ncells, const MpHamiltonian &H, const Eigenstate<Umps<Symmetry,Scalar> > &V, bool ADD_ODD_SITE)
{
	size_t add = (ADD_ODD_SITE)? 1:0;
	size_t Lhetero = Ncells * V.state.length() + add;
	
	vector<vector<Biped<Symmetry,MatrixType> > > As(Lhetero);
	As[0] = V.state.A[GAUGE::C][0];
	for (size_t l=1; l<Lhetero; ++l)
	{
		As[l] = V.state.A[GAUGE::R][l%N_sites];
	}
//		As[Lhetero-1] = V.state.A[GAUGE::C][(Lhetero-1)%N_sites];
//		for (int l=Lhetero-2; l>=0; --l)
//		{
//			As[l] = V.state.A[GAUGE::L][l%N_sites];
//		}
	
	vector<vector<qarray<Symmetry::Nq> > > qloc(Lhetero);
	for (size_t l=0; l<Lhetero; ++l)
	{
		qloc[l] = V.state.locBasis(l%N_sites);
	}
	
	Mps<Symmetry,Scalar> Mout(Lhetero, As, qloc, V.state.Qtarget(), Lhetero);
	
	Mout.BoundaryL = HeffA[0].L;
	Mout.BoundaryR = HeffA[(Lhetero-1)%N_sites].R;
	
	return Mout;
};

//*******************************************************************************************************************************************************************************
//This  function is expand_basis for the whole unit cell, but with updating AC only at the end.
//This could be slightly more efficient than the variant used above, so if you want you can try this function.
//It is basically the same code as in expand_basis (size_t loc, size_t DeltaD, const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout, VUMPS::TWOSITE_A::OPTION option)
//but without updating AC. This is here done after the loop over the unit cell.
//*******************************************************************************************************************************************************************************

// template<typename Symmetry, typename MpHamiltonian, typename Scalar>
// void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
// expand_basis (size_t DeltaD, const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout, VUMPS::TWOSITE_A::OPTION option)
// {
// 	if (DeltaD == 0) {return;} //early exit if the dimension to increase is zero.
	
// 	for(size_t loc=0; loc<N_sites; loc++)
// 	{
// 		vector<Biped<Symmetry,MatrixType> > NL;
// 		vector<Biped<Symmetry,MatrixType> > NR;
// 		Biped<Symmetry,MatrixType> NAAN;
// 		//calculate two-site B-Tensor (double tangent space) and obtain simultaneously NL and NR (nullspaces)
// 		calc_B2(loc, H, Vout, option, NAAN, NL, NR);

// 		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
// 		{
// 			lout << "l=" << loc << ", norm(NAAN)=" << sqrt(NAAN.squaredNorm().sum())  << endl;
// 		}
		
// 		// SVD-decompose NAAN
// 		Biped<Symmetry,MatrixType> U, Vdag;
// 		for (size_t q=0; q<NAAN.dim; ++q)
// 		{
//             #ifdef DONT_USE_BDCSVD
// 			JacobiSVD<MatrixType> Jack; // standard SVD
//             #else
// 			BDCSVD<MatrixType> Jack; // "Divide and conquer" SVD (only available in Eigen)
//             #endif

// 			Jack.compute(NAAN.block[q], ComputeThinU|ComputeThinV);
		
// 			size_t Nret = (Jack.singularValues().array() > Vout.state.eps_svd).count();
// 			Nret = min(DeltaD, Nret);
// 			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
// 			{
// 				lout << "q=" << NAAN.in[q] << ", Nret=" << Nret << endl;
// 			}
// 			if(Nret > 0)
// 			{
// 				U.push_back(NAAN.in[q], NAAN.out[q], Jack.matrixU().leftCols(Nret));
// 				Vdag.push_back(NAAN.in[q], NAAN.out[q], Jack.matrixV().adjoint().topRows(Nret));
// 			}
// 		}
		
// 		//calc P
// 		vector<Biped<Symmetry,MatrixType> > P(Vout.state.locBasis(loc).size());
// 		for (size_t s=0; s<Vout.state.locBasis(loc).size(); ++s)
// 		{
// 			P[s] = NL[s] * U;
// 		}

// 		Vout.state.enrich(loc, GAUGE::L, P);

// 		//Update the left environment if AL is involved in calculating the two site A-tensor, because we need correct environments for the effective two-site Hamiltonian.
// 		if (option == VUMPS::TWOSITE_A::ALxAC or option == VUMPS::TWOSITE_A::ALxCxAR)
// 		{
// 			contract_L(HeffA[loc].L, 
// 					   Vout.state.A[GAUGE::L][loc], H.W[loc], PROP::HAMILTONIAN, Vout.state.A[GAUGE::L][loc], 
// 					   H.locBasis(loc), H.opBasis(loc), 
// 					   HeffA[(loc+1)%N_sites].L);
// 		}
		
// 		P.clear();
// 		P.resize(Vout.state.locBasis((loc+1)%N_sites).size());
// 		for (size_t s=0; s<Vout.state.locBasis((loc+1)%N_sites).size(); ++s)
// 		{
// 			P[s] = Vdag * NR[s];
// 		}
		
// 		Vout.state.enrich(loc, GAUGE::R, P);

// 		//Update the right environment if AR is involved in calculating the two site A-tensor, because we need correct environments for the effective two-site Hamiltonian.
// 		//Note: maybe we only have to update the right environment if loc=0, since we need the updated environment only when loc=N_sites-1 and consequentially loc+1=0.
// 		if (option == VUMPS::TWOSITE_A::ACxAR or option == VUMPS::TWOSITE_A::ALxCxAR)
// 		{
// 			contract_R(HeffA[(loc+1)%N_sites].R,
// 					   Vout.state.A[GAUGE::R][(loc+1)%N_sites], H.W[(loc+1)%N_sites], PROP::HAMILTONIAN, Vout.state.A[GAUGE::R][(loc+1)%N_sites], 
// 					   H.locBasis((loc+1)%N_sites), H.opBasis((loc+1)%N_sites), 
// 					   HeffA[loc].R);
// 		}
		
// 		Vout.state.update_outbase(loc,GAUGE::L);

// 		//update C with zeros
// 		Vout.state.updateC(loc);
		
// 		// sort
// 		Vout.state.sort_A(loc, GAUGE::L);
// 		Vout.state.sort_A((loc+1)%N_sites, GAUGE::R);
// 		Vout.state.C[loc] = Vout.state.C[loc].sorted();
// 	}
	
// 	// update AC with zeros and sort
// 	for (size_t loc=0; loc<N_sites; loc++)
// 	{
// 		Vout.state.updateAC(loc,GAUGE::L);
// 		Vout.state.sort_A(loc, GAUGE::L, true); //true means sort all gauges, the parameter GAUGE::L has no impact here.
// 		Vout.state.C[loc] = Vout.state.C[loc].sorted();
// 	}
	
// 	Vout.state.update_inbase();
// 	Vout.state.update_outbase();
// }

#endif
