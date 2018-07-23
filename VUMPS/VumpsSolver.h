#ifndef VANILLA_VUMPSSOLVER
#define VANILLA_VUMPSSOLVER

#include "unsupported/Eigen/IterativeSolvers"
#include "termcolor.hpp"

#include "Mpo.h"
#include "VUMPS/Umps.h"
#include "VUMPS/VumpsPivotMatrices.h"
#include "pivot/DmrgPivotMatrix0.h"
#include "pivot/DmrgPivotMatrix1.h"
#include "pivot/DmrgPivotMatrix2.h"
#include "tensors/DmrgIndexGymnastics.h"
#include "DmrgLinearAlgebra.h"
#include "LanczosSolver.h" // from LANCZOS
#include "VUMPS/VumpsContractions.h"
#include "GMResSolver.h" // from LANCZOS
#include "VUMPS/VumpsTransferMatrix.h"
#include "VUMPS/VumpsTransferMatrixAA.h"

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
	
	/**Resets a custom algorithm.*/
	inline void set_algorithm (UMPS_ALG::OPTION ALGORITHM) {CHOSEN_ALGORITHM = ALGORITHM; USER_SET_ALGORITHM = true;};
	
	///\{
	/**\describe_info*/
	string info() const;
	
	/**\describe_info*/
	string eigeninfo() const;
	
	/**\describe_memory*/
	double memory (MEMUNIT memunit=GB) const;
	
	/**Setup a logfile of the iterations.
	* \param N_log_input : save the log every \p N_log_input half-sweeps
	* \param file_e_input : file for the ground-state energy in the format [min(eL,eR), eL, eR]
	* \param file_err_eigval_input : file for the energy error
	* \param file_err_var_input : file for the variational error
	*/
	void set_log (int N_log_input, string file_e_input, string file_err_eigval_input, string file_err_var_input);
	///\}
	
	/**Calculates the highest or lowest eigenstate with an MPO (algorithm 6). Works also for a 2- and 4-site unit cell. Simply create an MPO on 2 or 4 sites.*/
	void edgeState (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout, 
	                qarray<Symmetry::Nq> Qtot_input, 
	                double tol_eigval_input=1e-7, double tol_var_input=1e-6, 
	                size_t M=10, size_t Nqmax=4, 
	                size_t max_iterations=50, size_t min_iterations=6);
	
private:
	
	///\{
	/**Prepares the class, setting up the environments. Used with an explicit 2-site Hamiltonian.*/
	void prepare_h2site (const TwoSiteHamiltonian &h2site, const vector<qarray<Symmetry::Nq> > &qloc_input, 
	                     Eigenstate<Umps<Symmetry,Scalar> > &Vout, 
	                     qarray<Symmetry::Nq> Qtot_input,
	                     size_t M, size_t Nqmax);
	
	/**Performs an iteration with 1-site unit cell. Used with an explicit 2-site Hamiltonian.*/
	void iteration_h2site (Eigenstate<Umps<Symmetry,Scalar> > &Vout);
	///\}
	
	///\{
	/**Prepares the class setting up the environments. Used with an Mpo.*/
	void prepare (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout, qarray<Symmetry::Nq> Qtot, size_t M, size_t Nqmax);
	
	/**Performs an iteration with an n-site unit cell (in parallel, algorithm 3). Used with an MPO.*/
	void iteration_parallel (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout);
	
	/**Performs an iteration with an n-site unit cell (sequentially, algorithm 4). Used with an MPO.*/
	void iteration_sequential (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout);
	///\}
	
	///\{
	/**Performs an IDMRG iteration with a 1-site unit cell.*/
	void iteration_idmrg (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout);
	
	/**Prepares the class, setting up the environments for IDMRG.*/
	void prepare_idmrg (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout, qarray<Symmetry::Nq> Qtot, size_t M_input, size_t Nqmax);
	
	/**old energy for comparison in IDMRG.*/
	double Eold = std::nan("1");
	///\}
	
	///\{
	/**Builds the environment of a unit cell.*/
	void build_LR (const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &AL,
	               const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &AR,
	               const Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &C,
	               const vector<vector<vector<SparseMatrix<Scalar> > > > &W, 
	               const vector<qarray<Symmetry::Nq> > &qloc, 
	               const vector<qarray<Symmetry::Nq> > &qOp,
	               Tripod<Symmetry,MatrixType> &L,
	               Tripod<Symmetry,MatrixType> &R);
	
	/**Builds environments for each site of the unit cell.*/
	void build_cellEnv (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout);
	///\}
	
	/**Cleans up after the iteration process.*/
	void cleanup (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout);
	
	/**chain length*/
	size_t N_sites;
	
	/**tolerances*/
	double tol_eigval, tol_var;
	
	/**keeping track of iterations*/
	size_t N_iterations;
	
	/**errors*/
	double err_eigval, err_var, err_state=std::nan("1");
	
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
	* \param LRres : resulting (H_L| or |H_R)
	*/
	void solve_linear (GAUGE::OPTION gauge, 
	                   size_t ab, 
	                   const vector<Biped<Symmetry,MatrixType> > &A, 
	                   const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &Y_LR, 
	                   const Biped<Symmetry,MatrixType> &LReigen, 
	                   const vector<vector<vector<SparseMatrix<Scalar> > > > &W, 
	                   const vector<qarray<Symmetry::Nq> > &qloc, 
	                   const vector<qarray<Symmetry::Nq> > &qOp,
	                   Scalar LRdotY, 
	                   Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &LRres);
	
	/**Solves the linear system (eq. 15) using GMRES.
	* \param gauge : L or R
	* \param A : contracted A-tensor of the cell
	* \param hLR : (h_L|, |h_R) for eq. 15
	* \param LReigen : (L| or |R) 
	* \param qloc : local basis
	* \param hLRdotLR : (h_L|R), (L|h_R) for eq. 15
	* \param LRres : resulting (H_L| or |H_R)*/
	void solve_linear (GAUGE::OPTION gauge, 
	                   const vector<Biped<Symmetry,MatrixType> > &A, 
	                   const Biped<Symmetry,MatrixType> &hLR, 
	                   const Biped<Symmetry,MatrixType> &LReigen, 
	                   const vector<qarray<Symmetry::Nq> > &qloc, 
	                   Scalar hLRdotLR, 
	                   Biped<Symmetry,MatrixType> &LRres);
	
	/**control of verbosity and algorithms*/
	DMRG::VERBOSITY::OPTION CHOSEN_VERBOSITY;
	UMPS_ALG::OPTION CHOSEN_ALGORITHM = UMPS_ALG::DYNAMIC;
	bool USER_SET_ALGORITHM = false;
	
	/**Sets the Lanczos tolerances adaptively, depending on the current errors.*/
	void set_LanczosTolerances (double &tolLanczosEigval, double &tolLanczosState);
	
	/**Calculates the errors and sets the right sign for \p C.*/
	void calc_errors (Eigenstate<Umps<Symmetry,Scalar> > &Vout);
	
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
	string file_e, file_err_eigval, file_err_var;
	
	/**log data*/
	vector<double> eL_mem, eR_mem, err_eigval_mem, err_var_mem;
	
	/**Function to write out the logfiles.
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
set_log (int N_log_input, string file_e_input, string file_err_eigval_input, string file_err_var_input)
{
	N_log           = N_log_input;
	file_e          = file_e_input;
	file_err_eigval = file_err_eigval_input;
	file_err_var    = file_err_var_input;
	eL_mem.clear();
	eR_mem.clear();
	err_eigval_mem.clear();
	err_var_mem.clear();
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
	}
	
	if (N_log>0 and N_iterations%N_log==0 or FORCE==true)
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
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
set_LanczosTolerances (double &tolLanczosEigval, double &tolLanczosState)
{
	// Set less accuracy for the first iterations
	tolLanczosEigval = max(max(1e-2*err_eigval,1e-13),1e-13); // 1e-7
	tolLanczosState  = max(max(1e-2*err_var,   1e-10),1e-13); // 1e-4
	
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
calc_errors (Eigenstate<Umps<Symmetry,Scalar> > &Vout)
{
	std::array<VectorXd,2> epsLRsq;
	std::array<GAUGE::OPTION,2> gs = {GAUGE::L, GAUGE::R};
	
	for (const auto &g:gs)
	{
		epsLRsq[g].resize(N_sites);
		
		for (size_t l=0; l<N_sites; ++l)
		{
			epsLRsq[g](l) = Vout.state.calc_epsLRsq(g,l);
			
			bool GAUGE_FLIP = false;
			// If wrong phase, flip sign of C and recalculate:
			if (epsLRsq[g](l) > 1.)
			{
				size_t lC = (g==GAUGE::R)? Vout.state.minus1modL(l):l;
				Vout.state.C[lC] = -1. * Vout.state.C[lC];
				GAUGE_FLIP = true;
			}
			if (GAUGE_FLIP)
			{
				epsLRsq[g](l) = Vout.state.calc_epsLRsq(g,l);
			}
		}
	}
	
	err_var = max(sqrt(epsLRsq[GAUGE::L].sum()), 
	              sqrt(epsLRsq[GAUGE::R].sum()));
	err_eigval = max(abs(eoldR-eR), 
	                 abs(eoldL-eL));
	eoldR = eR;
	eoldL = eL;
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
prepare_h2site (const TwoSiteHamiltonian &h2site_input, const vector<qarray<Symmetry::Nq> > &qloc_input, Eigenstate<Umps<Symmetry,Scalar> > &Vout, 
                qarray<Symmetry::Nq> Qtot, size_t M_input, size_t Nqmax)
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
	Vout.state = Umps<Symmetry,Scalar>(qloc_input, Qtot, N_sites, M, Nqmax);
	Vout.state.N_sv = M;
	Vout.state.setRandom();
	for (size_t l=0; l<N_sites; ++l)
	{
		Vout.state.svdDecompose(l);
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
prepare (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout, qarray<Symmetry::Nq> Qtot, size_t M_input, size_t Nqmax)
{
	N_sites = H.length();
	N_iterations = 0;
	
	Stopwatch<> PrepTimer;
	
	// effective Hamiltonian
	D = H.locBasis(0).size();
	M = M_input;
	dW = H.auxdim();
	
	// resize Vout
	Vout.state = Umps<Symmetry,Scalar>(H.locBasis(0), Qtot, N_sites, M, Nqmax);
	Vout.state.N_sv = M;
	Vout.state.setRandom();
	for (size_t l=0; l<N_sites; ++l)
	{
		Vout.state.svdDecompose(l);
	}
	Vout.state.calc_entropy((CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)? true : false);
	
	// initial energy
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
prepare_idmrg (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout, qarray<Symmetry::Nq> Qtot, size_t M_input, size_t Nqmax)
{
	Stopwatch<> PrepTimer;
	
	// general
	N_sites = 1;
	N_iterations = 0;
	M = M_input; // bond dimension
	dW = H.auxdim();
	
	// resize Vout
	Vout.state = Umps<Symmetry,Scalar>(H.locBasis(0), Qtot, N_sites, M, Nqmax);
	Vout.state.N_sv = M;
	Vout.state.setRandom();
	for (size_t l=0; l<N_sites; ++l)
	{
		Vout.state.svdDecompose(l);
	}
	Vout.state.calc_entropy((CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)? true : false);
	
	HeffA.resize(1);
	Qbasis<Symmetry> inbase;
	inbase.pullData(Vout.state.A[GAUGE::C][0],0);
	Qbasis<Symmetry> outbase;
	outbase.pullData(Vout.state.A[GAUGE::C][0],1);
	HeffA[0].L.setIdentity(M, dW, 1, inbase);
	HeffA[0].R.setIdentity(M, dW, 1, outbase);
	
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
build_LR (const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &AL,
          const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &AR,
          const Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &C,
          const vector<vector<vector<SparseMatrix<Scalar> > > > &W, 
          const vector<qarray<Symmetry::Nq> > &qlocCell, 
          const vector<qarray<Symmetry::Nq> > &qOpCell,
          Tripod<Symmetry,MatrixType> &L,
          Tripod<Symmetry,MatrixType> &R)
{
	Stopwatch<> GMresTimer;
	
	L.clear();
	R.clear();
	
	// |R) and (L|
	Biped<Symmetry,MatrixType> Reigen = C.contract(C.adjoint());
	Biped<Symmetry,MatrixType> Leigen = C.adjoint().contract(C);
	
	// |YRa) and (YLa|
	vector<Tripod<Symmetry,MatrixType> > YL(dW);
	vector<Tripod<Symmetry,MatrixType> > YR(dW);
	
	// |Ra) and (La|
	Qbasis<Symmetry> inbase;
	inbase.pullData(AL,0);
	Qbasis<Symmetry> outbase;
	outbase.pullData(AL,1);
	
	Tripod<Symmetry,MatrixType> IdL; IdL.setIdentity(M, dW, 1, inbase);
	Tripod<Symmetry,MatrixType> IdR; IdR.setIdentity(M, dW, 1, outbase);
	L.insert(dW-1, IdL);
	R.insert(0,    IdR);
	
	auto Wsum = [&W, &qlocCell, &qOpCell] (size_t a, size_t b)
	{
		double res = 0;
		for (size_t s1=0; s1<qlocCell.size(); ++s1)
		for (size_t s2=0; s2<qlocCell.size(); ++s2)
		for (size_t k=0; k<qOpCell.size(); ++k)
		{
			for (int r=0; r<W[s1][s2][k].outerSize(); ++r)
			for (typename SparseMatrix<Scalar>::InnerIterator iW(W[s1][s2][k],r); iW; ++iW)
			{
				if (iW.row() == a and iW.col() == b)
				{
					res += abs(iW.value());
				}
			}
		}
		return res;
	};
	
//	#pragma omp parallel sections
	{
		// Eq. C19
//		#pragma omp section
		{
			for (int b=dW-2; b>=0; --b)
			{
				YL[b] = make_YL(b, L, AL, W, AL, qlocCell, qOpCell);
				
				if (Wsum(b,b) == 0.)
				{
					L.insert(b,YL[b]);
				}
				else
				{
					Tripod<Symmetry,MatrixType> Ltmp;
					solve_linear(GAUGE::L, b, AL, YL[b], Reigen, W, qlocCell, qOpCell, contract_LR(b,YL[b],Reigen), Ltmp);
					L.insert(b,Ltmp);
					
					if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE and b == 0)
					{
						cout << "<L[0]|R>=" << contract_LR(0,Ltmp,Reigen) << endl;
					}
				}
			}
		}
		
		// Eq. C20
//		#pragma omp section
		{
			for (int a=1; a<dW; ++a)
			{
				YR[a] = make_YR(a, R, AR, W, AR, qlocCell, qOpCell);
				
				if (Wsum(a,a) == 0.)
				{
					R.insert(a,YR[a]);
				}
				else
				{
					Tripod<Symmetry,MatrixType> Rtmp;
					solve_linear(GAUGE::R, a, AR, YR[a], Leigen, W, qlocCell, qOpCell, contract_LR(a,Leigen,YR[a]), Rtmp);
					R.insert(a,Rtmp);
					
					if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE and a == dW-1)
					{
						cout << "<L|R[dW-1]>=" << contract_LR(dW-1,Leigen,Rtmp) << endl;
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
//	contract_L(L, AL, W, AL, qlocCell, qOpCell, Lcheck);
//	Tripod<Symmetry,MatrixType> Rcheck;
//	contract_R(R, AR, W, AR, qlocCell, qOpCell, Rcheck);
//	
//	cout << termcolor::magenta << "CHECK=" << L.compare(Lcheck) << ", " << R.compare(Rcheck) << termcolor::reset << endl;
//	cout << (R-Rcheck).print(true,13) << endl;
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
build_cellEnv (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout)
{
	// Pre-contract A-tensors and W-tensors
	vector<vector<vector<SparseMatrix<Scalar> > > > Wcell = H.W_at(0);
	vector<qarray<Symmetry::Nq> > qOpCell                 = H.opBasis(0);
	
	for (size_t g=0; g<2; ++g)
	{
		Vout.state.Acell[g] = Vout.state.A[g][0];
	}
	Vout.state.qlocCell = H.locBasis(0);
	
	for (size_t l=0; l<N_sites-1; ++l)
	{
		for (size_t g=0; g<2; ++g)
		{
			vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > Atmp;
			contract_AA (Vout.state.Acell[g], Vout.state.qlocCell, 
			             Vout.state.A[g][l+1], H.locBasis(l+1), 
			             Vout.state.Qtop(l), Vout.state.Qbot(l),
			             Atmp);
			for (size_t s=0; s<Atmp.size(); ++s)
			{
				Atmp[s] = Atmp[s].cleaned();
			}
			Vout.state.Acell[g] = Atmp;
		}
		
		vector<vector<vector<SparseMatrix<Scalar> > > > Wtmp;
		vector<qarray<Symmetry::Nq> > qlocTmp;
		vector<qarray<Symmetry::Nq> > qOpTmp;
		
		contract_WW<Symmetry,Scalar> (Wcell, Vout.state.qlocCell, qOpCell, 
		                              H.W_at(l+1), H.locBasis(l+1), H.opBasis(l+1),
		                              Wtmp, qlocTmp, qOpTmp);
		
		Wcell = Wtmp;
		Vout.state.qlocCell = qlocTmp;
		qOpCell = qOpTmp;
	}
	
	// With a unit cell, Heff is a vector for each site
	HeffA.clear();
	HeffA.resize(N_sites);
	HeffC.clear();
	HeffC.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		HeffA[l].W = H.W[l];
		HeffC[l].W = H.W[l];
	}
	
	// Make environment for the unit cell
	build_LR (Vout.state.Acell[GAUGE::L], Vout.state.Acell[GAUGE::R], Vout.state.C[N_sites-1], 
	          Wcell, Vout.state.qlocCell, qOpCell, 
	          HeffA[0].L, HeffA[N_sites-1].R);
	
	// Make environment for each site of the unit cell
	for (size_t l=1; l<N_sites; ++l)
	{
		contract_L (HeffA[l-1].L, Vout.state.A[GAUGE::L][l-1], H.W[l-1], Vout.state.A[GAUGE::L][l-1], H.locBasis(l-1), H.opBasis(l-1), HeffA[l].L);
	}
	
	for (int l=N_sites-2; l>=0; --l)
	{
		contract_R (HeffA[l+1].R, Vout.state.A[GAUGE::R][l+1], H.W[l+1], Vout.state.A[GAUGE::R][l+1], H.locBasis(l+1), H.opBasis(l+1), HeffA[l].R);
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
	
	build_cellEnv(H,Vout);
	
	// See Algorithm 4
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
		Lutz(LANCZOS::REORTHO::FULL, LANCZOS::CONVTEST::SQ_TEST);
		Lutz.set_dimK(min(30ul, dim(gAC.state)));
		Lutz.edgeState(HeffA[l], gAC, LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			lout << "l=" << l << ", AC" << ", time" << LanczosTimer.info() << ", " << Lutz.info() << endl;
		}
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
		{
			lout << "e0(AC)=" << setprecision(13) << gAC.energy << ", ratio=" << gAC.energy/Vout.energy << endl;
		}
		
		// Solve for C
		gC.state = PivotVector<Symmetry,Scalar>(Vout.state.C[l]);
		
		LanczosSolver<PivotMatrix0<Symmetry,Scalar,Scalar>,PivotVector<Symmetry,Scalar>,Scalar> 
		Lucy(LANCZOS::REORTHO::FULL, LANCZOS::CONVTEST::SQ_TEST);
		Lucy.set_dimK(min(30ul, dim(gC.state)));
		Lucy.edgeState(PivotMatrix0(HeffC[l]), gC, LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
		
//		qarray2<Symmetry::Nq> vac = {Symmetry::qvacuum(), Symmetry::qvacuum()};
//		auto it = gC.state.data[0].dict.find(vac);
//		double maxnorm = 0.;
//		double maxq;
//		for (size_t q=0; q<gC.state.data[0].size(); ++q)
//		{
//			double norm = gC.state.data[0].block[q].norm();
//			if (norm > maxnorm) {maxnorm = norm; maxq = q;}
//		}
//		gC.state.data[0].block[it->second].swap(gC.state.data[0].block[maxq]);
//		cout << "swapped: " << it->second << " and " << maxq << endl;
		
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			lout << "l=" << l << ", C" << ", time" << LanczosTimer.info() << ", " << Lucy.info() << endl;
		}
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
		{
			lout << "e0(C)=" << setprecision(13) << gC.energy << ", ratio=" << gC.energy/Vout.energy << endl;
		}
		
		Vout.state.A[GAUGE::C][l] = gAC.state.data;
		Vout.state.C[l] = gC.state.data[0];
	}
	
	for (size_t l=0; l<N_sites; ++l)
	{
		(err_var>0.01)? Vout.state.svdDecompose(l) : Vout.state.polarDecompose(l);
	}
	Vout.state.calc_entropy((CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)? true : false);
	
	// Calculate energies
//	Biped<Symmetry,ComplexMatrixType> Reigen = calc_LReigen(GAUGE::R, Vout.state.Acell[GAUGE::R], Vout.state.Acell[GAUGE::R], 
//	                                                        Vout.state.C[N_sites-1], Vout.state.qlocCell).state;
//	Biped<Symmetry,ComplexMatrixType> Leigen = calc_LReigen(GAUGE::L, Vout.state.Acell[GAUGE::L], Vout.state.Acell[GAUGE::L],
//	                                                        Vout.state.C[N_sites-1], Vout.state.qlocCell).state;
//	eL = abs(contract_LR(0,    YLlast.template cast<ComplexMatrixType>(), Reigen)) / H.volume();
//	eR = abs(contract_LR(dW-1, Leigen, YRfrst.template cast<ComplexMatrixType>())) / H.volume();
//	cout << termcolor::blue << "eL=" << eL << ", eR=" << eR << termcolor::reset << endl;
	
	Biped<Symmetry,MatrixType> Reigen = Vout.state.C[N_sites-1].contract(Vout.state.C[N_sites-1].adjoint());
	Biped<Symmetry,MatrixType> Leigen = Vout.state.C[N_sites-1].adjoint().contract(Vout.state.C[N_sites-1]);
	eL = contract_LR(0, YLlast, Reigen) / H.volume();
	eR = contract_LR(dW-1, Leigen, YRfrst) / H.volume();
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
	{
		lout << Vout.state.test_ortho() << endl;
		lout << termcolor::blue << "eL=" << eL << ", eR=" << eR << termcolor::reset << endl;
		lout << test_LReigen(Vout) << endl;
	}
	
	calc_errors(Vout);
	Vout.energy = min(eL,eR);
	
	++N_iterations;
	
	// print stuff
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
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
iteration_sequential (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout)
{
	Stopwatch<> IterationTimer;
	
	double tolLanczosEigval, tolLanczosState;
	set_LanczosTolerances(tolLanczosEigval,tolLanczosState);
	
	// See Algorithm 3
	for (size_t l=0; l<N_sites; ++l)
	{
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
		Lutz(LANCZOS::REORTHO::FULL, LANCZOS::CONVTEST::SQ_TEST);
		Lutz.set_dimK(min(30ul, dim(gAC.state)));
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
		Lucy(LANCZOS::REORTHO::FULL, LANCZOS::CONVTEST::SQ_TEST);
		Lucy.set_dimK(min(30ul, dim(gCR.state)));
		Lucy.edgeState(PivotMatrix0(HeffC[l]), gCR, LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
		
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
		Luca(LANCZOS::REORTHO::FULL, LANCZOS::CONVTEST::SQ_TEST);
		Luca.set_dimK(min(30ul, dim(gCL.state)));
		Luca.edgeState(PivotMatrix0(HeffC[lC]), gCL, LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, true);
		
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
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
	{
		lout << Vout.state.test_ortho() << endl;
		lout << termcolor::blue << "eL=" << eL << ", eR=" << eR << termcolor::reset << endl;
		lout << test_LReigen(Vout) << endl;
	}
	
	calc_errors(Vout);
	Vout.energy = min(eL,eR);
	
	++N_iterations;
	
	// print stuff
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
	solve_linear (GAUGE::L, Vout.state.A[GAUGE::L][0], hL, Reigen, Vout.state.locBasis(0), eR, HL);
	solve_linear (GAUGE::R, Vout.state.A[GAUGE::R][0], hR, Leigen, Vout.state.locBasis(0), eL, HR);
	
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
	LanczosSolver<PivumpsMatrix1<Symmetry,Scalar,Scalar>,PivotVector<Symmetry,Scalar>,Scalar> Lutz1(LANCZOS::REORTHO::FULL);
	Lutz1.set_dimK(min(30ul, dim(gAC.state)));
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
	
	LanczosSolver<PivumpsMatrix0<Symmetry,Scalar,Scalar>,PivotVector<Symmetry,Scalar>,Scalar> Lutz0(LANCZOS::REORTHO::FULL);
	Lutz0.set_dimK(min(30ul, dim(gC.state)));
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
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
	{
		lout << Vout.state.test_ortho() << endl;
		lout << termcolor::blue << "eL=" << eL << ", eR=" << eR << termcolor::reset << endl;
		lout << test_LReigen(Vout) << endl;
	}
	
	calc_errors(Vout);
	Vout.energy = min(eL,eR);
	
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
	Lutz(LANCZOS::REORTHO::FULL, LANCZOS::CONVTEST::SQ_TEST);
	Lutz.set_dimK(min(30ul, dim(g.state)));
	Lutz.edgeState(HeffA[0], g, LANCZOS::EDGE::GROUND, DMRG::CONTROL::DEFAULT::eps_eigval, DMRG::CONTROL::DEFAULT::eps_coeff, false);
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "time" << LanczosTimer.info() << ", " << Lutz.info() << endl;
	}
	
	auto Cref = Vout.state.C[0];
	
	Mps<Symmetry,Scalar> Vtmp(2, H.locBasis(), Symmetry::qvacuum(), 2, Vout.state.Nqmax);
	Vtmp.min_Nsv = M;
	Vtmp.max_Nsv = M;
	Vtmp.sweepStep2(DMRG::DIRECTION::RIGHT, 0, g.state.data, 
	                Vout.state.A[GAUGE::L][0], Vout.state.A[GAUGE::R][0], Vout.state.C[0],
	                true);
	
	Vout.state.C[0] = 1./sqrt((Vout.state.C[0].contract(Vout.state.C[0].adjoint())).trace()) * Vout.state.C[0];
	
	Vout.state.calc_entropy((CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)? true : false);
	
	for (size_t s=0; s<H.locBasis(0).size(); ++s)
	{
		Vout.state.A[GAUGE::C][0][s] = Vout.state.A[GAUGE::L][0][s].contract(Vout.state.C[0]);
//		Vout.state.A[GAUGE::C][0][s] = Vout.state.C[0].contract(Vout.state.A[GAUGE::R][0][s]);
	}
	
	++N_iterations;
	
	Vout.energy = 0.5*g.energy-Eold;
	err_eigval = abs(Vout.energy-eL); // the energy density is the contribution of the new site
	err_state = (Cref-Vout.state.C[0]).norm().sum();
	err_var = err_state;
	Eold = 0.5*g.energy;
	eL = Vout.energy;
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
	{
		lout << Vout.state.test_ortho() << endl;
	}
	
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > HeffLtmp, HeffRtmp;
	contract_L(HeffA[0].L, Vout.state.A[GAUGE::L][0], H.W[0], Vout.state.A[GAUGE::L][0], H.locBasis(0), H.opBasis(0), HeffLtmp);
	HeffA[0].L = HeffLtmp;
	contract_R(HeffA[0].R, Vout.state.A[GAUGE::R][0], H.W[1], Vout.state.A[GAUGE::R][0], H.locBasis(1), H.opBasis(1), HeffRtmp);
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
	TransferMatrixAA TR(GAUGE::R, Vout.state.Acell[GAUGE::R], Vout.state.Acell[GAUGE::R], Vout.state.qlocCell);
	TransferMatrixAA TL(GAUGE::L, Vout.state.Acell[GAUGE::L], Vout.state.Acell[GAUGE::L], Vout.state.qlocCell);
	
	Biped<Symmetry,MatrixType> Reigen = Vout.state.C[N_sites-1].contract(Vout.state.C[N_sites-1].adjoint());
	Biped<Symmetry,MatrixType> Leigen = Vout.state.C[N_sites-1].adjoint().contract(Vout.state.C[N_sites-1]);
	
	PivotVector<Symmetry,Scalar> PsiR(Reigen);
	PivotVector<Symmetry,Scalar> PsiL(Leigen);
	
	HxV(TL,PsiR);
	HxV(TR,PsiL);
	
	stringstream ss;
	ss << "ReigenTest=" << (Reigen-PsiR.data[0]).norm().sum() << ", LeigenTest=" << (Leigen-PsiL.data[0]).norm().sum() << endl;
	return ss.str();
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
edgeState (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout, qarray<Symmetry::Nq> Qtot, 
           double tol_eigval_input, double tol_var_input, size_t M, size_t Nqmax, 
           size_t max_iterations, size_t min_iterations)
{
	if (CHOSEN_VERBOSITY>=2)
	{
		lout << endl << termcolor::colorize << termcolor::bold
		 << "——————————————————————————————————————————VUMPS algorithm " << CHOSEN_ALGORITHM << "——————————————————————————————————————————"
		 <<  termcolor::reset << endl;
	}
	
	tol_eigval = tol_eigval_input;
	tol_var = tol_var_input;
	
	if (USER_SET_ALGORITHM and CHOSEN_ALGORITHM == UMPS_ALG::IDMRG)
	{
		prepare_idmrg(H, Vout, Qtot, M, Nqmax);
	}
	else if (USER_SET_ALGORITHM and CHOSEN_ALGORITHM == UMPS_ALG::H2SITE)
	{
		assert(H.length() == 2 and "Need L=2 for H2SITE!");
		prepare_h2site(H.H2site(0,true), H.locBasis(0), Vout, Qtot, M, Nqmax);
	}
	else
	{
		prepare(H, Vout, Qtot, M, Nqmax);
	}
	
	Stopwatch<> GlobalTimer;
	
	while (((err_eigval >= tol_eigval or err_var >= tol_var) and N_iterations < max_iterations) or N_iterations < min_iterations)
	{
		if (USER_SET_ALGORITHM) // custom choice of algorithm
		{
			if (CHOSEN_ALGORITHM == UMPS_ALG::PARALLEL)
			{
				iteration_parallel(H,Vout);
			}
			else if (CHOSEN_ALGORITHM == UMPS_ALG::SEQUENTIAL)
			{
				iteration_sequential(H,Vout);
			}
			else if (CHOSEN_ALGORITHM == UMPS_ALG::H2SITE)
			{
				iteration_h2site(Vout);
			}
			else if (CHOSEN_ALGORITHM == UMPS_ALG::IDMRG)
			{
				iteration_idmrg(H,Vout);
			}
		}
		else // dynamical choice: L=1 parallel, L>1 sequential
		{
			if (N_sites == 1)
			{
				iteration_parallel(H,Vout);
			}
			else
			{
				iteration_sequential(H,Vout);
			}
		}
		
		write_log();
	}
	write_log(true); // force log on exit
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << GlobalTimer.info("total runtime") << endl;
	}
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::ON_EXIT)
	{
		size_t standard_precision = cout.precision();
		lout << termcolor::bold
		     << "iterations=" << N_iterations
		     << ", e0=" << setprecision(13) << Vout.energy 
		     << ", err_eigval=" << err_eigval 
		     << ", err_var=" << err_var 
		     << setprecision(standard_precision)
		     << termcolor::reset
		     << endl;
		lout << Vout.state.info() << endl;
		lout << endl;
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
solve_linear (GAUGE::OPTION gauge, 
              size_t ab, 
              const vector<Biped<Symmetry,MatrixType> > &A, 
              const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &Y_LR, 
              const Biped<Symmetry,MatrixType> &LReigen, 
              const vector<vector<vector<SparseMatrix<Scalar> > > > &W, 
              const vector<qarray<Symmetry::Nq> > &qloc, 
              const vector<qarray<Symmetry::Nq> > &qOp,
              Scalar LRdotY, 
              Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &LRres)
{
	TransferMatrix<Symmetry,Scalar> T(gauge, A, A, LReigen, W, qloc, qOp, ab);
	TransferVector<Symmetry,Scalar> bvec(Y_LR, ab, LRdotY); // right-hand site vector |Y_LR)-e*1
	
	// Solve linear system
	GMResSolver<TransferMatrix<Symmetry,Scalar>,TransferVector<Symmetry,Scalar> > Gimli;
	
//	size_t dimK = GMRes_dimK[gauge];
//	if      (GMRes_Niter[gauge] > 1 and dimK < 200) {dimK += 10;}
//	else if (GMRes_Niter[gauge] == 1 and dimK > 10) {dimK -= 10;}
//	Gimli.set_dimK(min(dimK,dim(bvec)));
//	Gimli.set_dimK(max(dimK,10ul));
	
	Gimli.set_dimK(min(100ul,dim(bvec)));
	TransferVector<Symmetry,Scalar> LRres_tmp;
	Gimli.solve_linear(T, bvec, LRres_tmp);
	LRres = LRres_tmp.data;
	
//	GMRes_Niter[gauge] = Gimli.get_Niter();
//	GMRes_dimK [gauge] = Gimli.get_dimK();
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << gauge << ": " << Gimli.info() << endl;
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
solve_linear (GAUGE::OPTION gauge, 
              const vector<Biped<Symmetry,MatrixType> > &A, 
              const Biped<Symmetry,MatrixType> &hLR, 
              const Biped<Symmetry,MatrixType> &LReigen, 
              const vector<qarray<Symmetry::Nq> > &qloc, 
              Scalar hLRdotLR, 
              Biped<Symmetry,MatrixType> &LRres)
{
	TransferMatrixAA<Symmetry,Scalar> T(gauge,A,A,qloc,true);
	T.LReigen = LReigen;
	PivotVector<Symmetry,Scalar> bvec(hLR);
	
	for (size_t s=0; s<bvec.data.size(); ++s)
	for (size_t q=0; q<bvec.data[s].dim; ++q)
	{
		bvec.data[s].block[q] -= hLRdotLR * Matrix<Scalar,Dynamic,Dynamic>::Identity(bvec.data[s].block[q].rows(),
		                                                                             bvec.data[s].block[q].cols());
	}
	
	// Solve linear system
	GMResSolver<TransferMatrixAA<Symmetry,Scalar>,PivotVector<Symmetry,Scalar> > Gimli;
	
//	size_t dimK = GMRes_dimK[gauge];
//	if      (GMRes_Niter[gauge] > 1 and dimK < 100) {dimK += 10;}
//	else if (GMRes_Niter[gauge] == 1 and dimK > 10) {dimK -= 10;}
//	Gimli.set_dimK(min(dimK,dim(bvec)));
//	Gimli.set_dimK(max(dimK,10ul));
	
	Gimli.set_dimK(min(100ul,dim(bvec)));
	PivotVector<Symmetry,Scalar> LRres_tmp;
	Gimli.solve_linear(T,bvec,LRres_tmp);
	LRres = LRres_tmp.data[0];
	
//	GMRes_Niter[gauge] = Gimli.get_Niter();
//	GMRes_dimK[gauge]  = Gimli.get_dimK();
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << gauge << ": " << Gimli.info() << endl;
	}
}

#endif
