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
	typedef Matrix<Scalar,Dynamic,1>       VectorType;
	typedef boost::multi_array<Scalar,4> TwoSiteHamiltonian;
	
public:
	
	VumpsSolver (DMRG::VERBOSITY::OPTION VERBOSITY=DMRG::VERBOSITY::SILENT)
	:CHOSEN_VERBOSITY(VERBOSITY)
	{};
	
	///\{
	/**\describe_info*/
	string info() const;
	
	/**\describe_info*/
	string eigeninfo() const;
	
	/**\describe_memory*/
	double memory   (MEMUNIT memunit=GB) const;
	
	/**Setup a logfile of the iterations.
	* \param N_log_input : save the log every \p N_log half-sweeps
	* \param file_e_input : file for the ground-state energy in the format [min(eL,eR), eL, eR]
	* \param file_err_eigval_input : file for the energy error
	* \param file_err_var_input : file for the variatonal error
	*/
	void set_log (int N_log_input, string file_e_input, string file_err_eigval_input, string file_err_var_input)
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
	///\}
	
	/**Calculates the highest or lowest eigenstate with an explicit 2-site Hamiltonian (algorithm 2). No unit cell is implemented here.*/
//	void edgeState (const TwoSiteHamiltonian &h2site, const vector<qarray<Symmetry::Nq> > &qloc_input, 
//	                Eigenstate<Umps<Symmetry,Scalar> > &Vout, 
//	                double tol_eigval_input=1e-7, double tol_var_input=1e-6, 
//	                size_t Dlimit=500, 
//	                size_t max_iterations=50, size_t min_iterations=6);
//	
	/**Calculates the highest or lowest eigenstate with an MPO (algorithm 6). Works also for a 2- and 4-site unit cell. Simply create an MPO on 2 or 4 sites.*/
	void edgeState (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout, 
	                double tol_eigval_input=1e-7, double tol_var_input=1e-6, 
	                size_t M=10, size_t Nqmax=4, 
	                size_t max_iterations=50, size_t min_iterations=6);
	
private:
	
	/**Resets the verbosity level.*/
	inline void set_verbosity (DMRG::VERBOSITY::OPTION VERBOSITY) {CHOSEN_VERBOSITY = VERBOSITY;};
	
	///\{
//	/**Prepares the class, setting up the environments. Used with an explicit 2-site Hamiltonian.*/
//	void prepare (const TwoSiteHamiltonian &h2site, const vector<qarray<Symmetry::Nq> > &qloc_input,
//	              Eigenstate<Umps<Symmetry,Scalar> > &Vout, size_t M);
//	
//	/**Performs a half-sweep with 1-site unit cell. Used with an explicit 2-site Hamiltonian.*/
//	void iteration1 (Eigenstate<Umps<Symmetry,Scalar> > &Vout);
	///\}
	
	///\{
	/**Prepares the class setting up the environments. Used with an MPO.*/
	void prepare (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout, size_t M, size_t Nqmax);
	
	void build_LR (const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &AL,
	               const vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > &AR,
	               const Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &C,
	               const vector<vector<vector<SparseMatrix<Scalar> > > > &W, 
	               const vector<qarray<Symmetry::Nq> > &qloc, 
	               const vector<qarray<Symmetry::Nq> > &qOp,
	               Tripod<Symmetry,MatrixType> &L,
	               Tripod<Symmetry,MatrixType> &R);
	
//	/**Performs a half-sweep with an n-site unit cell (sequentially, algorithm 4). Used with an MPO.*/
	void iteration_parallel (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout);
	///\}
	
	/**Clean up after the iteration process.*/
	void cleanup (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout);
	
	size_t N_sites;
	double tol_eigval, tol_var;
	size_t N_iterations;
	double err_eigval, err_var, err_state=std::nan("1");
	
	vector<PivumpsMatrix1<Symmetry,Scalar,Scalar> > Heff; // environment
	vector<qarray<Symmetry::Nq> > qloc;
//	TwoSiteHamiltonian h2site; // stored 2-site Hamiltonian
	size_t D, M, dW; // bond dimension per subspace, bond dimension per site, MPO bond dimension
	
	double eL, eR, eoldR, eoldL; // left and right error (eq. 18) and old errors from previous half-sweep
	
	/**Solves the linear system (eq. 15 or eq. C25ab) using GMRES.
	* \param gauge : L or R
	* \param A : A, Apair or Aquadruple
	* \param Y_LR : (h_L|, |h_R) for eq. 15 or |Y_Ra), (Y_La| for eq. C25ab
	* \param LReigen : (L| or |R)
	* \param W : MPO tensor for the transfer matrix
	* \param qloc :
	* \param qOp :
	* \param LRdotY : (h_L|R), (L|h_R) for eq. 15 or (Y_La|R), (L|Y_Ra) for eq. C25ab
	* \param LRres : resulting (H_L| or |H_R)
	*/
//	template<typename Atype, typename Wtype>
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
	
	DMRG::VERBOSITY::OPTION CHOSEN_VERBOSITY;
	
	/**Sets the Lanczos tolerances adaptively, depending on the current errors.*/
	void set_LanczosTolerances (double &tolLanczosEigval, double &tolLanczosState);
	
	/**Creates the left and right transfer matrices (eq. A7) explicitly. This is only for testing purposes, as a 4-legged tensor this is very inefficient.*/
//	void make_explicitT (const Umps<Symmetry,Scalar> &Vbra, const Umps<Symmetry,Scalar> &Vket, MatrixType &TL, MatrixType &TR);
	
	/**Explicitly calculates the left eigenvector of the transfer matrix \f$T_L\f$. This is only for testing purposes and very inefficient.*/
//	MatrixXd eigenvectorL (const MatrixType &TL);
	
	/**Explicitly calculates the right eigenvector of the transfer matrix \f$T_R\f$. This is only for testing purposes and very inefficient.*/
//	MatrixXd eigenvectorR (const MatrixType &TR);
	
	Tripod<Symmetry,MatrixType> YLlast;
	Tripod<Symmetry,MatrixType> YRfrst;
	
	///\{
	/**log stuff*/
	size_t N_log=0;
	string file_e, file_err_eigval, file_err_var;
	vector<double> eL_mem, eR_mem, err_eigval_mem, err_var_mem;
	void write_log (bool FORCE=false);
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

//template<typename Symmetry, typename MpHamiltonian, typename Scalar>
//void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
//prepare (const TwoSiteHamiltonian &h2site_input, const vector<qarray<Symmetry::Nq> > &qloc_input, Eigenstate<Umps<Symmetry,Scalar> > &Vout, size_t M_input)
//{
//	Stopwatch<> PrepTimer;
//	
//	// general
//	N_sites = 1;
//	N_iterations = 0;
//	D = h2site_input.shape()[0]; // local dimension
//	M = M_input; // bond dimension
//	
//	// effective and 2-site Hamiltonian
//	Heff.resize(N_sites);
//	h2site.resize(boost::extents[D][D][D][D]);
//	h2site = h2site_input;
//	
//	// resize Vout
//	Vout.state = Umps<Symmetry,Scalar>(qloc_input, N_sites, M);
//	Vout.state.N_sv = M;
//	Vout.state.setRandom();
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		Vout.state.svdDecompose(l);
//	}
//	
//	// initial energy & error
//	eoldL = std::nan("");
//	eoldR = std::nan("");
//	err_eigval = 1.;
//	err_var    = 1.;
//}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
set_LanczosTolerances (double &tolLanczosEigval, double &tolLanczosState)
{
	// Set less accuracy for the first iterations
	tolLanczosEigval = max(max(1e-2*err_eigval,1e-13),1e-13); // 1e-7
	tolLanczosState  = max(max(1e-2*err_var,   1e-8), 1e-13); // 1e-4
	
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

//template<typename Symmetry, typename MpHamiltonian, typename Scalar>
//void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
//iteration1 (Eigenstate<Umps<Symmetry,Scalar> > &Vout)
//{
//	Stopwatch<> IterationTimer;
//	
//	// |R) and (L|
//	MatrixType Reigen = Vout.state.C[N_sites-1].block[0] * Vout.state.C[N_sites-1].block[0].adjoint();
//	MatrixType Leigen = Vout.state.C[N_sites-1].block[0].adjoint() * Vout.state.C[N_sites-1].block[0];
//	
//	// |h_R) and (h_L|
//	MatrixType hR = make_hR(h2site, Vout.state.A[GAUGE::R][0], Vout.state.locBasis(0));
//	MatrixType hL = make_hL(h2site, Vout.state.A[GAUGE::L][0], Vout.state.locBasis(0));
//	
//	// energies
//	eL = (Leigen * hR).trace();
//	eR = (hL * Reigen).trace();
//	
//	// |H_R) and (H_L|
//	MatrixType HR(M,M), HL(M,M);
//	
//	// Solve the linear systems in eq. 14
//	Stopwatch<> GMresTimer;
//	vector<Scalar> Wdummy; // This dummy also clarifies the template parameter of solve_linear for the compiler
//	solve_linear(GAUGE::L, Vout.state.A[GAUGE::L][0], hL, Reigen, Wdummy, eR, HL);
//	solve_linear(GAUGE::R, Vout.state.A[GAUGE::R][0], hR, Leigen, Wdummy, eL, HR);
//	
//	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
//	{
//		lout << "linear systems" << GMresTimer.info() << endl;
//	}
//	
//	// Doesn't work like that!! boost::multi_array is shit!
////	Heff[0] = PivumpsMatrix1<Symmetry,Scalar,Scalar>(HL, HR, h2site, Vout.state.A[GAUGE::L][0], Vout.state.A[GAUGE::R][0]);
//	
//	Heff[0].h.resize(boost::extents[D][D][D][D]);
//	Heff[0].h = h2site;
//	Heff[0].L = HL;
//	Heff[0].R = HR;
//	Heff[0].AL = Vout.state.A[GAUGE::L][0];
//	Heff[0].AR = Vout.state.A[GAUGE::R][0];
//	
//	double tolLanczosEigval, tolLanczosState;
//	set_LanczosTolerances(tolLanczosEigval,tolLanczosState);
//	
//	// Solve for AC (eq. 11)
//	Eigenstate<PivotVector<Symmetry,Scalar> > gAC;
//	gAC.state = PivotVector<Symmetry,Scalar>(Vout.state.A[GAUGE::C][0]);
//	
//	Stopwatch<> LanczosTimer;
//	LanczosSolver<PivumpsMatrix1<Symmetry,Scalar,Scalar>,PivotVector<Symmetry,Scalar>,Scalar> Lutz1(LANCZOS::REORTHO::FULL);
//	Lutz1.set_dimK(min(30ul, dim(gAC.state)));
//	Lutz1.edgeState(Heff[0],gAC, LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
//	
//	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
//	{
//		lout << "time" << LanczosTimer.info() << ", " << Lutz1.info() << endl;
//	}
//	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
//	{
//		lout << "e0(AC)=" << setprecision(13) << gAC.energy << endl;
//	}
//	
//	// Solve for C (eq. 16)
//	Eigenstate<PivotVector<Symmetry,Scalar> > gC;
//	gC.state = PivotVector<Symmetry,Scalar>(Vout.state.C[0]);
//	
//	LanczosSolver<PivumpsMatrix0<Symmetry,Scalar,Scalar>,PivotVector<Symmetry,Scalar>,Scalar> Lutz0(LANCZOS::REORTHO::FULL);
//	Lutz0.set_dimK(min(30ul, dim(gC.state)));
//	Lutz0.edgeState(PivumpsMatrix0(Heff[0]),gC, LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
//	
//	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
//	{
//		lout << "time" << LanczosTimer.info() << ", " << Lutz0.info() << endl;
//	}
//	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
//	{
//		lout << "e0(C)=" << setprecision(13) << gC.energy << endl;
//	}
//	
//	// Calculate AL and AR from AC, C
//	Vout.state.A[GAUGE::C][0] = gAC.state.data;
//	Vout.state.C[0]           = gC.state.data[0];
//	(err_var>0.1)? Vout.state.svdDecompose(0) : Vout.state.polarDecompose(0);
//	
//	// Calculate errors
//	double epsLsq, epsRsq;
//	Vout.state.calc_epsLRsq(0,epsLsq,epsRsq);
//	err_var = max(sqrt(epsLsq),sqrt(epsRsq));
//	
//	err_eigval = max(abs(eoldR-eR), abs(eoldL-eL));
//	eoldR = eR;
//	eoldL = eL;
//	Vout.energy = min(eL,eR);
//	
//	++N_iterations;
//	
//	// Print stuff
//	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
//	{
//		size_t standard_precision = cout.precision();
//		lout << "S=" << Vout.state.entropy(0) << endl;
//		lout << eigeninfo() << endl;
//		lout << IterationTimer.info("full iteration") << endl;
//		lout << endl;
//	}
//}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
prepare (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout, size_t M_input, size_t Nqmax)
{
//	assert(H.length()<=2 or H.length()==4); // only 2- and 4-site unit cells are implemented
	
	N_sites = H.length();
	N_iterations = 0;
	
	Stopwatch<> PrepTimer;
	
	// effective Hamiltonian
	D = H.locBasis(0).size();
	M = M_input;
	dW = H.auxdim();
	
	// resize Vout
	Vout.state = Umps<Symmetry,Scalar>(H.locBasis(0), N_sites, M, Nqmax);
	Vout.state.N_sv = M;
	Vout.state.setRandom();
	for (size_t l=0; l<N_sites; ++l)
	{
		Vout.state.svdDecompose(l);
	}
	
	// initial energy
	eoldL = std::nan("");
	eoldR = std::nan("");
	
	err_eigval = 1.;
	err_var    = 1.;
}

//template<typename Symmetry, typename MpHamiltonian, typename Scalar>
//void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
//make_explicitT (const Umps<Symmetry,Scalar> &Vbra, const Umps<Symmetry,Scalar> &Vket, MatrixType &TL, MatrixType &TR)
//{
////	vector<vector<MatrixType> > TL(H.auxdim());
////	vector<vector<MatrixType> > TR(H.auxdim());
////	
////	for (size_t a=0; a<H.auxdim(); ++a)
////	{
////		TL[a].resize(H.auxdim());
////		TR[a].resize(H.auxdim());
////	}
////	
////	for (size_t a=0; a<H.auxdim(); ++a)
////	for (size_t b=0; b<H.auxdim(); ++b)
////	{
////		TL[a][b].resize(M*M,M*M); TL[a][b].setZero();
////		TR[a][b].resize(M*M,M*M); TR[a][b].setZero();
////	}
////	
////	for (size_t s1=0; s1<D; ++s1)
////	for (size_t s2=0; s2<D; ++s2)
////	for (int k12=0; k12<H.W[0][s1][s2].outerSize(); ++k12)
////	for (typename SparseMatrix<Scalar>::InnerIterator iW(H.W[0][s1][s2],k12); iW; ++iW)
////	for (size_t i=0; i<M; ++i)
////	for (size_t j=0; j<M; ++j)
////	for (size_t k=0; k<M; ++k)
////	for (size_t l=0; l<M; ++l)
////	{
////		size_t a = iW.row();
////		size_t b = iW.col();
////		
////		size_t r = i + M*l; // note: rows of A & cols of A† (= rows of A*) become new rows of T
////		size_t c = j + M*k; // note: cols of A & rows of A† (= cols of A*) become new cols of T
////		
////		TL[a][b](r,c) += iW.value() * Vout.state.A[GAUGE::L][0][s2].block[0](i,j) * Vout.state.A[GAUGE::L][0][s1].block[0].adjoint()(k,l);
////		TR[a][b](r,c) += iW.value() * Vout.state.A[GAUGE::R][0][s2].block[0](i,j) * Vout.state.A[GAUGE::R][0][s1].block[0].adjoint()(k,l);
////	}
//	
//	TL.resize(M*M,M*M); TL.setZero();
//	TR.resize(M*M,M*M); TR.setZero();
////	for (size_t s=0; s<D; ++s)
////	{
////		// only for real:
////		TL += kroneckerProduct(Vout.state.A[GAUGE::L][0][s].block[0], Vout.state.A[GAUGE::L][0][s].block[0]); 
////		TR += kroneckerProduct(Vout.state.A[GAUGE::R][0][s].block[0], Vout.state.A[GAUGE::R][0][s].block[0]);
////	}
//	for (size_t s=0; s<D; ++s)
//	for (size_t i=0; i<M; ++i)
//	for (size_t j=0; j<M; ++j)
//	for (size_t k=0; k<M; ++k)
//	for (size_t l=0; l<M; ++l)
//	{
//		size_t r = i + M*l; // Note: rows of A & cols of A† (= rows of A*) become new rows of T
//		size_t c = j + M*k; // Note: cols of A & rows of A† (= cols of A*) become new cols of T
//		TL(r,c) += Vket.A[GAUGE::L][0][s].block[0](i,j) * Vbra.A[GAUGE::L][0][s].block[0].adjoint()(k,l);
//		TR(r,c) += Vket.A[GAUGE::R][0][s].block[0](i,j) * Vbra.A[GAUGE::R][0][s].block[0].adjoint()(k,l);
//	}
//}

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
					res += iW.value();
				}
			}
		}
		return res;
	};
	
	// Eq. C19
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
			solve_linear(GAUGE::L, b, AL, YL[b], Reigen, W, qlocCell, qOpCell, contract_LR(YL[b],Reigen), Ltmp);
			L.insert(b,Ltmp);
			
			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE and b == 0)
			{
				cout << "<L[0]|R>=" << contract_LR(Ltmp,Reigen) << endl;
			}
		}
	}
	
	// Eq. C20
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
			solve_linear(GAUGE::R, a, AR, YR[a], Leigen, W, qlocCell, qOpCell, contract_LR(Leigen,YR[a]), Rtmp);
			R.insert(a,Rtmp);
			
			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE and a == dW-1)
			{
				cout << "<L|R[dW-1]>=" << contract_LR(Leigen,Rtmp) << endl;
			}
		}
	}
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "linear systems" << GMresTimer.info() << endl;
	}
	
	YLlast = YL[0];
	YRfrst = YR[dW-1];
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
iteration_parallel (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout)
{
	Stopwatch<> IterationTimer;
	
	// Pre-contract A-tensors and W-tensors
	vector<vector<vector<SparseMatrix<Scalar> > > > Wcell = H.W_at(0);
	vector<qarray<Symmetry::Nq> > qlocCell                = H.locBasis(0);
	vector<qarray<Symmetry::Nq> > qOpCell                 = H.opBasis(0);
	
	for (size_t g=0; g<2; ++g)
	{
		Vout.state.Acell[g] = Vout.state.A[g][0];
	}
	
	for (size_t l=0; l<N_sites-1; ++l)
	{
		for (size_t g=0; g<2; ++g)
		{
			vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > Atmp;
			contract_AA (Vout.state.Acell[g], qlocCell, 
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
		
		contract_WW<Symmetry,Scalar> (Wcell, qlocCell, qOpCell, 
		                              H.W_at(l+1), H.locBasis(l+1), H.opBasis(l+1),
		                              Wtmp, qlocTmp, qOpTmp);
		
		Wcell = Wtmp;
		qlocCell = qlocTmp;
		qOpCell = qOpTmp;
	}
	
	// With a unit cell, Heff is a vector for each site
	vector<PivotMatrix1<Symmetry,Scalar,Scalar> > HeffA(N_sites);
	vector<PivotMatrix1<Symmetry,Scalar,Scalar> > HeffAC(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		HeffA[l].W  = H.W[l];
		HeffAC[l].W = H.W[l];
	}
	
	// Make environment for the unit cell
	build_LR (Vout.state.Acell[GAUGE::L], Vout.state.Acell[GAUGE::R], Vout.state.C[N_sites-1], 
	          Wcell, qlocCell, qOpCell, 
	          HeffA[0].L, HeffA[N_sites-1].R);

	// for(size_t q=0; q<HeffA[N_sites-1].R.dim; q++)
	// {
	// 	// if(HeffA[N_sites-1].R.mid(q) == Symmetry::qvacuum())
	// 	// {
	// 		for (size_t a=0; a<HeffA[N_sites-1].R.block[q].shape()[0]; ++a)
	// 		{
	// 			if(HeffA[N_sites-1].R.block[q][a][0].size() != 0)
	// 			{
	// 				HeffA[N_sites-1].R.block[q][a][0] = HeffA[N_sites-1].R.block[q][a][0]
	// 					* sqrt(Symmetry::coeff_dot(HeffA[N_sites-1].R.in(q)))
	// 					* sqrt(Symmetry::coeff_dot(HeffA[N_sites-1].R.out(q)))
	// 					* sqrt(Symmetry::coeff_dot(HeffA[N_sites-1].R.mid(q)));
	// 			}
	// 		}
	// 	// }
	// 	// else {cout << termcolor::red << "Non diagonal blocks in the right cell environment" << termcolor::reset << endl;}
	// }

	// for(size_t q=0; q<HeffA[0].L.dim; q++)
	// {
	// 	// if(HeffA[0].L.mid(q) == Symmetry::qvacuum())
	// 	// {
	// 		for (size_t a=0; a<HeffA[0].L.block[q].shape()[0]; ++a)
	// 		{
	// 			if(HeffA[0].L.block[q][a][0].size() != 0)
	// 			{
	// 				HeffA[0].L.block[q][a][0] = HeffA[0].L.block[q][a][0]
	// 					* sqrt(Symmetry::coeff_dot(HeffA[0].L.in(q)))
	// 					* sqrt(Symmetry::coeff_dot(HeffA[0].L.out(q)))
	// 					* sqrt(Symmetry::coeff_dot(HeffA[0].L.mid(q)));
	// 			}
	// 		}
	// 	// }
	// 	// else {cout << termcolor::red << "Non diagonal blocks in the right cell environment" << termcolor::reset << endl;}
	// }

	// Make environment for each site of the unit cell
	for (size_t l=1; l<N_sites; ++l)
	{
		contract_L(HeffA[l-1].L, Vout.state.A[GAUGE::L][l-1], H.W[l-1], Vout.state.A[GAUGE::L][l-1], H.locBasis(l-1), H.opBasis(l-1), HeffA[l].L);
	}
	
	for (int l=N_sites-2; l>=0; --l)
	{
		contract_R(HeffA[l+1].R, Vout.state.A[GAUGE::R][l+1], H.W[l+1], Vout.state.A[GAUGE::R][l+1], H.locBasis(l+1), H.opBasis(l+1), HeffA[l].R);
	}
	
	for (size_t l=0; l<N_sites; ++l)
	{
		HeffAC[l].L = HeffA[(l+1)%N_sites].L;
		HeffAC[l].R = HeffA[l].R;
	}
	
	vector<Eigenstate<PivotVector<Symmetry,Scalar> > > gAC(N_sites);
	vector<Eigenstate<PivotVector<Symmetry,Scalar> > > gC(N_sites);
	
	double tolLanczosEigval, tolLanczosState;
	set_LanczosTolerances(tolLanczosEigval,tolLanczosState);
	
	// local optimization
	for (size_t l=0; l<N_sites; ++l)
	{
		precalc_blockStructure (HeffA[l].L, Vout.state.A[GAUGE::C][l], HeffA[l].W, Vout.state.A[GAUGE::C][l], HeffA[l].R, 
		                        H.locBasis(l), H.opBasis(l), HeffA[l].qlhs, HeffA[l].qrhs, HeffA[l].factor_cgcs);
		
		// Solve for AC
		gAC[l].state = PivotVector<Symmetry,Scalar>(Vout.state.A[GAUGE::C][l]);
		
		Stopwatch<> LanczosTimer;
		LanczosSolver<PivotMatrix1<Symmetry,Scalar,Scalar>,PivotVector<Symmetry,Scalar>,Scalar> Lutz(LANCZOS::REORTHO::FULL,LANCZOS::CONVTEST::SQ_TEST);
		Lutz.set_dimK(min(30ul, dim(gAC[l].state)));
		Lutz.edgeState(HeffA[l],gAC[l], LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			lout << "l=" << l << ", AC" << ", time" << LanczosTimer.info() << ", " << Lutz.info() << endl;
		}
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
		{
			lout << "e0(AC)=" << setprecision(13) << gAC[l].energy << endl;
		}
		
		// Solve for C
		gC[l].state = PivotVector<Symmetry,Scalar>(Vout.state.C[l]);
		
		LanczosSolver<PivotMatrix0<Symmetry,Scalar,Scalar>,PivotVector<Symmetry,Scalar>,Scalar> Lucy(LANCZOS::REORTHO::FULL,LANCZOS::CONVTEST::SQ_TEST);
		Lucy.set_dimK(min(30ul, dim(gC[l].state)));
		Lucy.edgeState(PivotMatrix0(HeffAC[l]),gC[l], LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			lout << "l=" << l << ", C" << ", time" << LanczosTimer.info() << ", " << Lucy.info() << endl;
		}
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
		{
			lout << "e0(C)=" << setprecision(13) << gC[l].energy << endl;
		}
	}
	
	// Calculate AL, AR from AC, C
	for (size_t l=0; l<N_sites; ++l)
	{
		Vout.state.A[GAUGE::C][l] = gAC[l].state.data;
		Vout.state.C[l]           = gC[l].state.data[0];
	}
	
	for (size_t l=0; l<N_sites; ++l)
	{
		(err_var>0.1)? Vout.state.svdDecompose(l) : Vout.state.polarDecompose(l);
	}
	
//	TransferMatrixAA<Symmetry,Scalar> TL(GAUGE::L, Vout.state.Acell[GAUGE::L], Vout.state.Acell[GAUGE::L], qlocCell);
//	TransferMatrixAA<Symmetry,Scalar> TR(GAUGE::R, Vout.state.Acell[GAUGE::R], Vout.state.Acell[GAUGE::R], qlocCell);
//	
//	PivotVector<Symmetry,complex<double> > ReigenTmp(gC[N_sites-1].state.data[0].template cast<MatrixXcd>());
//	PivotVector<Symmetry,complex<double> > LeigenTmp(gC[N_sites-1].state.data[0].template cast<MatrixXcd>());
//	
//	ArnoldiSolver<TransferMatrixAA<Symmetry,double>,PivotVector<Symmetry,complex<double> > > ArnieL;
//	ArnoldiSolver<TransferMatrixAA<Symmetry,double>,PivotVector<Symmetry,complex<double> > > ArnieR;
//	
//	ArnieL.set_dimK(30ul);
//	ArnieR.set_dimK(30ul);
//	
//	complex<double> lambdaR, lambdaL;
//	
//	ArnieL.calc_dominant(TL, ReigenTmp ,lambdaR);
//	lout << ArnieL.info() << endl;
//	
//	ArnieR.calc_dominant(TR, LeigenTmp ,lambdaL);
//	lout << ArnieR.info() << endl;
//	
//	Biped<Symmetry,MatrixXcd> Reigen = ReigenTmp.data[0];
//	Biped<Symmetry,MatrixXcd> Leigen = LeigenTmp.data[0];
//	
//	cout << "lambdaL=" << lambdaL << ", lambdaR=" << lambdaR << endl;
//	eL = isReal(contract_LR(YLlast.template cast<MatrixXcd>(), Reigen)) / H.volume();
//	eR = isReal(contract_LR(Leigen, YRfrst.template cast<MatrixXcd>())) / H.volume();
	
	// Calculate energies
	Biped<Symmetry,MatrixType> Reigen = Vout.state.C[N_sites-1].contract(Vout.state.C[N_sites-1].adjoint());
	Biped<Symmetry,MatrixType> Leigen = Vout.state.C[N_sites-1].adjoint().contract(Vout.state.C[N_sites-1]);
	eL = isReal(contract_LR(YLlast, Reigen)) / H.volume();
	eR = isReal(contract_LR(Leigen, YRfrst)) / H.volume();
	
	// Calculate errors
	MatrixXd epsLRsq(N_sites,2);
	for (size_t l=0; l<N_sites; ++l)
	{
		Vout.state.calc_epsLRsq(l,epsLRsq(l,0),epsLRsq(l,1));
	}
	err_var = sqrt(epsLRsq.sum());
	
	err_eigval = max(abs(eoldR-eR), abs(eoldL-eL));
	eoldR = eR;
	eoldL = eL;
	Vout.energy = min(eL,eR);
	
	if (N_sites == 1 and CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
	{
		lout << "eL=" << eL << ", eR=" << eR << endl;
		lout << "ratio test AC: " << abs(2.-gAC[0].energy/Vout.energy) << ", C: " << abs(1.-gC[0].energy/Vout.energy) << endl;
	}
	
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

//template<typename Symmetry, typename MpHamiltonian, typename Scalar>
//void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
//edgeState (const TwoSiteHamiltonian &h2site, const vector<qarray<Symmetry::Nq> > &qloc, Eigenstate<Umps<Symmetry,Scalar> > &Vout, double tol_eigval_input, double tol_var_input, size_t M, size_t max_iterations, size_t min_iterations)
//{
//	tol_eigval = tol_eigval_input;
//	tol_var = tol_var_input;
//	
//	prepare(h2site, qloc, Vout, M);
//	
//	Stopwatch<> GlobalTimer;
//	
//	while (((err_eigval >= tol_eigval or err_var >= tol_var) and N_iterations < max_iterations) or N_iterations < min_iterations)
//	{
//		iteration1(Vout);
//		write_log();
//	}
//	write_log(true); // force log on exit
//	
//	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
//	{
//		lout << GlobalTimer.info("total runtime") << endl;
//	}
//	
//	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::ON_EXIT)
//	{
//		size_t standard_precision = cout.precision();
//		lout << "emin=" << setprecision(13) << Vout.energy << setprecision(standard_precision) << endl;
//		lout << Vout.state.info() << endl;
//		lout << endl;
//	}
//}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
edgeState (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout, 
           double tol_eigval_input, double tol_var_input, size_t M, size_t Nqmax, 
           size_t max_iterations, size_t min_iterations)
{
	tol_eigval = tol_eigval_input;
	tol_var = tol_var_input;
	
	prepare(H, Vout, M, Nqmax);
	
	Stopwatch<> GlobalTimer;
	
	while (((err_eigval >= tol_eigval or err_var >= tol_var) and N_iterations < max_iterations) or N_iterations < min_iterations)
	{
		iteration_parallel(H,Vout);
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
		     << "e0=" << setprecision(13) << Vout.energy 
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
	// Transfer operator
	TransferMatrix<Symmetry,Scalar> T(gauge, A, A, LReigen, W, qloc, qOp, ab);
	
	// Right-hand site vector |Y_LR)-e*1
	TransferVector<Symmetry,Scalar> bvec(Y_LR, ab, LRdotY);
	
	// Solve linear system
	GMResSolver<TransferMatrix<Symmetry,Scalar>,TransferVector<Symmetry,Scalar> > Gimli;
	Gimli.set_dimK(min(30ul,dim(bvec)));
	TransferVector<Symmetry,Scalar> LRres_tmp;
	Gimli.solve_linear(T, bvec, LRres_tmp);
	LRres = LRres_tmp.data;
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << Gimli.info() << endl;
	}
}

//template<typename Symmetry, typename MpHamiltonian, typename Scalar>
//MatrixXd VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
//eigenvectorL (const MatrixType &TL)
//{
//	EigenSolver<MatrixType> Lutz(TL);
//	int max_index;
//	Lutz.eigenvalues().cwiseAbs().maxCoeff(&max_index);
//	
//	MatrixType Mout(M,M);
//	
//	for (size_t i=0; i<M; ++i)
//	for (size_t j=0; j<M; ++j)
//	{
//		size_t r = i + M*j;
//		Mout(i,j) = Lutz.eigenvectors().col(max_index)(r).real();
//	}
//	
//	return Mout;
//}

//template<typename Symmetry, typename MpHamiltonian, typename Scalar>
//MatrixXd VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
//eigenvectorR (const MatrixType &TR)
//{
//	EigenSolver<MatrixType> Lutz(TR.adjoint());
//	int max_index;
//	Lutz.eigenvalues().cwiseAbs().maxCoeff(&max_index);
//	
//	MatrixType Mout(M,M);
//	
//	for (size_t i=0; i<M; ++i)
//	for (size_t j=0; j<M; ++j)
//	{
//		size_t r = i + M*j;
//		Mout(i,j) = Lutz.eigenvectors().col(max_index)(r).real();
//	}
//	
//	return Mout;
//}

#endif
