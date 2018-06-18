#ifndef VANILLA_VUMPSSOLVER
#define VANILLA_VUMPSSOLVER

#include "unsupported/Eigen/IterativeSolvers"

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
#include "symmetry/U0.h" // for qloc3dummy

/**Solver that calculates the ground state of a UMPS. Analogue of the DmrgSolver class.
\ingroup VUMPS
\describe_Symmetry
\describe_Scalar*/
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
	
	/**Performs a half-sweep with 1-site unit cell. Used with an MPO.*/
	void iteration1 (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout);
	
//	/**Performs a half-sweep with a 2-site unit cell (sequentially, algorithm 4). Used with an MPO.*/
//	void iteration2 (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout);
//	
//	/**Performs a half-sweep with a 4-site unit cell (sequentially, algorithm 4). Used with an MPO.*/
//	void iteration4 (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout);
	///\}
	
	/**Clean up after the iteration process.*/
	void cleanup (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout);
	
	size_t N_sites;
	double tol_eigval, tol_var;
	size_t N_iterations;
	double err_eigval, err_var, err_state=std::nan("1");
	
	vector<PivumpsMatrix1<Symmetry,Scalar,Scalar> > Heff; // environment
	vector<qarray<Symmetry::Nq> > qloc;
	TwoSiteHamiltonian h2site; // stored 2-site Hamiltonian
	size_t D, M, dW; // bond dimension per subspace, bond dimension per site, MPO bond dimension
	
	double eL, eR, eoldR, eoldL; // left and right error (eq. 18) and old errors from previous half-sweep
	
	/**Solves the linear system (eq. 15 or eq. C25ab) using GMRES.
	* \param gauge : L or R
	* \param A : A, Apair or Aquadruple
	* \param hLR : (h_L|, |h_R) for eq. 15 or |Y_Ra), (Y_La| for eq. C25ab
	* \param LReigen : (L| or |R)
	* \param Warray : MPO tensor for the transfer matrix
	* \param e : (h_L|R), (L|h_R) for eq. 15 or (Y_La|R), (L|Y_Ra) for eq. C25ab
	* \param Hres : resulting (H_L| or |H_R)
	*/
//	template<typename Atype, typename Wtype>
	void solve_linear (GAUGE::OPTION gauge, 
	                   size_t ab, 
	                   const vector<Biped<Symmetry,MatrixType> > &A, 
	                   const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &hLR, 
	                   const Biped<Symmetry,MatrixType> &LReigen, 
	                   const MpHamiltonian &H, 
	                   double e, 
	                   Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &Hres);
	
	DMRG::VERBOSITY::OPTION CHOSEN_VERBOSITY;
	
	/**Sets the Lanczos tolerances adaptively, depending on the current errors.*/
	void set_LanczosTolerances (double &tolLanczosEigval, double &tolLanczosState);
	
	/**Creates the left and right transfer matrices (eq. A7) explicitly. This is only for testing purposes, as a 4-legged tensor this is very inefficient.*/
	void make_explicitT (const Umps<Symmetry,Scalar> &Vbra, const Umps<Symmetry,Scalar> &Vket, MatrixType &TL, MatrixType &TR);
	
	/**Explicitly calculates the left eigenvector of the transfer matrix \f$T_L\f$. This is only for testing purposes and very inefficient.*/
	MatrixXd eigenvectorL (const MatrixType &TL);
	
	/**Explicitly calculates the right eigenvector of the transfer matrix \f$T_R\f$. This is only for testing purposes and very inefficient.*/
	MatrixXd eigenvectorR (const MatrixType &TR);
	
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
	tolLanczosEigval = max(max(1e-2*err_eigval,1e-7),1e-12); // 1e-7
	tolLanczosState  = max(max(1e-2*err_var,   1e-4),1e-12); // 1e-4
	
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
	assert(H.length()<=2 or H.length()==4); // only 2- and 4-site unit cells are implemented
	
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

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
make_explicitT (const Umps<Symmetry,Scalar> &Vbra, const Umps<Symmetry,Scalar> &Vket, MatrixType &TL, MatrixType &TR)
{
//	vector<vector<MatrixType> > TL(H.auxdim());
//	vector<vector<MatrixType> > TR(H.auxdim());
//	
//	for (size_t a=0; a<H.auxdim(); ++a)
//	{
//		TL[a].resize(H.auxdim());
//		TR[a].resize(H.auxdim());
//	}
//	
//	for (size_t a=0; a<H.auxdim(); ++a)
//	for (size_t b=0; b<H.auxdim(); ++b)
//	{
//		TL[a][b].resize(M*M,M*M); TL[a][b].setZero();
//		TR[a][b].resize(M*M,M*M); TR[a][b].setZero();
//	}
//	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	for (int k12=0; k12<H.W[0][s1][s2].outerSize(); ++k12)
//	for (typename SparseMatrix<Scalar>::InnerIterator iW(H.W[0][s1][s2],k12); iW; ++iW)
//	for (size_t i=0; i<M; ++i)
//	for (size_t j=0; j<M; ++j)
//	for (size_t k=0; k<M; ++k)
//	for (size_t l=0; l<M; ++l)
//	{
//		size_t a = iW.row();
//		size_t b = iW.col();
//		
//		size_t r = i + M*l; // note: rows of A & cols of A† (= rows of A*) become new rows of T
//		size_t c = j + M*k; // note: cols of A & rows of A† (= cols of A*) become new cols of T
//		
//		TL[a][b](r,c) += iW.value() * Vout.state.A[GAUGE::L][0][s2].block[0](i,j) * Vout.state.A[GAUGE::L][0][s1].block[0].adjoint()(k,l);
//		TR[a][b](r,c) += iW.value() * Vout.state.A[GAUGE::R][0][s2].block[0](i,j) * Vout.state.A[GAUGE::R][0][s1].block[0].adjoint()(k,l);
//	}
	
	TL.resize(M*M,M*M); TL.setZero();
	TR.resize(M*M,M*M); TR.setZero();
//	for (size_t s=0; s<D; ++s)
//	{
//		// only for real:
//		TL += kroneckerProduct(Vout.state.A[GAUGE::L][0][s].block[0], Vout.state.A[GAUGE::L][0][s].block[0]); 
//		TR += kroneckerProduct(Vout.state.A[GAUGE::R][0][s].block[0], Vout.state.A[GAUGE::R][0][s].block[0]);
//	}
	for (size_t s=0; s<D; ++s)
	for (size_t i=0; i<M; ++i)
	for (size_t j=0; j<M; ++j)
	for (size_t k=0; k<M; ++k)
	for (size_t l=0; l<M; ++l)
	{
		size_t r = i + M*l; // Note: rows of A & cols of A† (= rows of A*) become new rows of T
		size_t c = j + M*k; // Note: cols of A & rows of A† (= cols of A*) become new cols of T
		TL(r,c) += Vket.A[GAUGE::L][0][s].block[0](i,j) * Vbra.A[GAUGE::L][0][s].block[0].adjoint()(k,l);
		TR(r,c) += Vket.A[GAUGE::R][0][s].block[0](i,j) * Vbra.A[GAUGE::R][0][s].block[0].adjoint()(k,l);
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
iteration1 (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout)
{
	Stopwatch<> IterationTimer;
	
	// |R) and (L|
	Biped<Symmetry,MatrixType> Reigen = Vout.state.C[0].contract(Vout.state.C[0].adjoint());
	Biped<Symmetry,MatrixType> Leigen = Vout.state.C[0].adjoint().contract(Vout.state.C[0]);
	
//	MatrixType TL, TR;
//	make_explicitT(Vout.state,Vout.state,TL,TR);
//	MatrixType Leigen = eigenvectorL(TL);
//	MatrixType Reigen = eigenvectorR(TR);
//	cout << "Leigen: " << (Leigen-Vout.state.C[0].block[0].adjoint() * Vout.state.C[0].block[0]).norm() << endl;
//	cout << "Reigen: " << (Reigen-Vout.state.C[0].block[0] * Vout.state.C[0].block[0].adjoint()).norm() << endl;
	
	// |YRa) and (YLa|
	vector<Tripod<Symmetry,MatrixType> > YL(dW);
	vector<Tripod<Symmetry,MatrixType> > YR(dW);
	
//	for (size_t a=0; a<dW; ++a)
//	{
//		YL[a].resize(M,M); YL[a].setZero();
//		YR[a].resize(M,M); YR[a].setZero();
//	}
	
	// |Ra) and (La|
//	boost::multi_array<MatrixType,LEGLIMIT> L(boost::extents[dW][1]);
//	boost::multi_array<MatrixType,LEGLIMIT> R(boost::extents[dW][1]);
	Tripod<Symmetry,MatrixType> L;
	Tripod<Symmetry,MatrixType> R;
	
	Stopwatch<> GMresTimer;
//	L[dW-1][0].resize(M,M);
//	L[dW-1][0].setIdentity();
//	R[0][0].resize(M,M);
//	R[0][0].setIdentity();
	Tripod<Symmetry,MatrixType> Id; Id.setIdentity(M,M,dW,1);
	L.insert(dW-1,dW,Id);
	R.insert(0,dW,Id);
	
	// Eq. C19
	for (int b=dW-2; b>=0; --b)
	{
//		YL[b] = make_YL(b, H.W[0], L, Vout.state.A[GAUGE::L][0], H.locBasis(0));
		YL[b] = make_YL(b, L, Vout.state.A[GAUGE::L][0], H.W[0], Vout.state.A[GAUGE::L][0], H.locBasis(0), H.opBasis(0));
		
//		vector<Scalar> Wval(D);
//		for (size_t s=0; s<D; ++s)
//		{
//			Wval[s] = H.W[0][s][s][0].coeff(b,b);
//		}
		
		double Wsum = 0;
		for (size_t s=0; s<H.locBasis(0).size(); ++s)
		for (size_t k=0; k<H.opBasis(0).size(); ++k)
		{
			for (int r=0; r<H.W[0][s][s][k].outerSize(); ++r)
			for (typename SparseMatrix<Scalar>::InnerIterator iW(H.W[0][s][s][k],r); iW; ++iW)
			{
				if (iW.row() == b and iW.col() == b)
				{
					Wsum += iW.value();
				}
			}
		}
		
//		if (accumulate(Wval.begin(),Wval.end(),0) == 0.)
		if (Wsum == 0.)
		{
//			L[b][0] = YL[b];
			L.insert(b,dW,YL[b]);
		}
		else
		{
//			solve_linear(GAUGE::L, Vout.state.A[GAUGE::L][0], YL[b], Reigen, Wval, (YL[b]*Reigen).trace(), L[b][0]);
			Tripod<Symmetry,MatrixType> Ltmp;
			solve_linear(GAUGE::L, b, Vout.state.A[GAUGE::L][0], YL[b], Reigen, H, contract_LR(YL[b], Reigen), Ltmp);
			L.insert(b,dW,Ltmp);
		}
	}
	
	// Eq. C20
	for (int a=1; a<dW; ++a)
	{
//		YR[a] = make_YR(a, H.W[0], R, Vout.state.A[GAUGE::R][0], H.locBasis(0));
		YR[a] = make_YR(a, R, Vout.state.A[GAUGE::R][0], H.W[0], Vout.state.A[GAUGE::R][0], H.locBasis(0), H.opBasis(0));
		
//		vector<Scalar> Wval(D);
//		for (size_t s=0; s<D; ++s)
//		{
//			Wval[s] = H.W[0][s][s][0].coeff(a,a);
//		}
		
		double Wsum = 0;
		for (size_t s=0; s<H.locBasis(0).size(); ++s)
		for (size_t k=0; k<H.opBasis(0).size(); ++k)
		{
			for (int r=0; r<H.W[0][s][s][k].outerSize(); ++r)
			for (typename SparseMatrix<Scalar>::InnerIterator iW(H.W[0][s][s][k],r); iW; ++iW)
			{
				if (iW.row() == a and iW.col() == a)
				{
					Wsum += iW.value();
				}
			}
		}
		
//		if (accumulate(Wval.begin(),Wval.end(),0) == 0.)
		if (Wsum == 0.)
		{
//			R[a][0] = YR[a];
			R.insert(a,dW,YR[a]);
		}
		else
		{
//			solve_linear(GAUGE::R, Vout.state.A[GAUGE::R][0], YR[a], Leigen, Wval, (Leigen*YR[a]).trace(), R[a][0]);
			Tripod<Symmetry,MatrixType> Rtmp;
			solve_linear(GAUGE::R, a, Vout.state.A[GAUGE::R][0], YR[a], Leigen, H, contract_LR(Leigen, YR[a]), Rtmp);
			R.insert(a,dW,Rtmp);
		}
	}
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "linear systems" << GMresTimer.info() << endl;
	}
	
	double tolLanczosEigval, tolLanczosState;
	set_LanczosTolerances(tolLanczosEigval,tolLanczosState);
	
	PivotMatrix1<Symmetry,Scalar,Scalar> HeffA;
	HeffA.W = H.W[0];
	HeffA.qloc = H.locBasis(0);
//	HeffA.L.push_back(qloc3dummy,L);
//	HeffA.R.push_back(qloc3dummy,R);
	HeffA.L = L;
	HeffA.R = R;
	precalc_blockStructure (HeffA.L, Vout.state.A[GAUGE::C][0], HeffA.W, Vout.state.A[GAUGE::C][0], HeffA.R, 
	                        H.locBasis(0), H.opBasis(0), HeffA.qlhs, HeffA.qrhs, HeffA.factor_cgcs);
	
	// Solve for AC
	Eigenstate<PivotVector<Symmetry,Scalar> > gAC;
	gAC.state = PivotVector<Symmetry,Scalar>(Vout.state.A[GAUGE::C][0]);
	
	Stopwatch<> LanczosTimer;
	LanczosSolver<PivotMatrix1<Symmetry,Scalar,Scalar>,PivotVector<Symmetry,Scalar>,Scalar> Lutz(LANCZOS::REORTHO::FULL);
	Lutz.set_dimK(min(30ul, dim(gAC.state)));
	Lutz.edgeState(HeffA,gAC, LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "time" << LanczosTimer.info() << ", " << Lutz.info() << endl;
	}
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
	{
		lout << "e0(AC)=" << setprecision(13) << gAC.energy << endl;
	}
	
	// Solve for C
	Eigenstate<PivotVector<Symmetry,Scalar> > gC;
	gC.state = PivotVector<Symmetry,Scalar>(Vout.state.C[0]);
	
	LanczosSolver<PivotMatrix0<Symmetry,Scalar,Scalar>,PivotVector<Symmetry,Scalar>,Scalar> Lucy(LANCZOS::REORTHO::FULL);
	Lucy.set_dimK(min(30ul, dim(gC.state)));
	Lucy.edgeState(PivotMatrix0(HeffA),gC, LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << "time" << LanczosTimer.info() << ", " << Lucy.info() << endl;
	}
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
	{
		lout << "e0(C)=" << setprecision(13) << gC.energy << endl;
	}
	
	// Calculate energies
	eL = contract_LR(YL[0], Reigen);
	eR = contract_LR(Leigen, YR[dW-1]);
	cout << "eL=" << eL << ", eR=" << eR << endl;
	
	// Calculate AL, AR from AC, C
	Vout.state.A[GAUGE::C][0] = gAC.state.data;
	Vout.state.C[0]           = gC.state.data[0];
	(err_var>0.1)? Vout.state.svdDecompose(0) : Vout.state.polarDecompose(0);
	
	// Calcualte errors
	double epsLsq, epsRsq;
	Vout.state.calc_epsLRsq(0,epsLsq,epsRsq);
	err_var = max(sqrt(epsLsq),sqrt(epsRsq));
	
	err_eigval = max(abs(eoldR-eR), abs(eoldL-eL));
	eoldR = eR;
	eoldL = eL;
	Vout.energy = min(eL,eR);
	
	++N_iterations;
	
	// Print stuff
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		size_t standard_precision = cout.precision();
		lout << "S=" << Vout.state.entropy(0) << endl;
		lout << eigeninfo() << endl;
		lout << IterationTimer.info("full iteration") << endl;
		lout << endl;
	}
//	
//////	if (N_iterations%4 == 0)
////	if (err_var < 1e-4)
////	{
////		PivotMatrix2<Symmetry,Scalar,Scalar> Heff2;
////		Heff2.W12 = H.W[0];
////		Heff2.W34 = H.W[0];
////		Heff2.L = HeffA.L;
////		Heff2.R = HeffA.R;
////		Heff2.dim = D*D*M*M;
////		Heff2.qloc12 = H.locBasis(0);
////		Heff2.qloc34 = H.locBasis(0);
////	//	contract_R(HeffA.R, Vout.state.A[GAUGE::R][0], H.W[0], Vout.state.A[GAUGE::R][0], H.locBasis(0), Heff2.R);
////		
////		vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > AA;
////		contract_AA(Vout.state.A[GAUGE::C][0], H.locBasis(0), Vout.state.A[GAUGE::R][0], H.locBasis(0), AA);
////		
////		PivotVector2<Symmetry,Scalar> Apair;
////		Apair.A = AA;
////		
////		HxV(Heff2,Apair);
////		
////		MatrixType B2(Vout.state.N[GAUGE::L][0][0].block[0].cols(), Vout.state.N[GAUGE::R][0][0].block[0].rows());
////		B2.setZero();
////		for (size_t s1=0; s1<D; ++s1)
////		for (size_t s2=0; s2<D; ++s2)
////		{
////			B2 += Vout.state.N[GAUGE::L][0][s1].block[0].adjoint() * 
////			      Apair.A[s1][s2].block[0] * 
////			      Vout.state.N[GAUGE::R][0][s2].block[0].adjoint();
////		}
////		BDCSVD<MatrixType> Jack(B2,ComputeThinU|ComputeThinV);
////		
////		err_state = B2.norm();
////		
//////		cout << "B2=" << B2.norm() << endl;
//////		
//////		MatrixType B2alt(Vout.state.N[GAUGE::L][0][0].block[0].cols(), Vout.state.N[GAUGE::R][0][0].block[0].rows());
//////		B2alt.setZero();
//////		
//////		for (size_t s1=0; s1<D; ++s1)
//////		for (size_t s2=0; s2<D; ++s2)
//////		for (size_t s3=0; s3<D; ++s3)
//////		for (size_t s4=0; s4<D; ++s4)
//////		{
//////			if (H.H2site(0,0,false)[s1][s2][s3][s4] != 0.)
//////			{
//////				B2alt += H.H2site(0,0,false)[s1][s2][s3][s4] * 
//////				         Vout.state.N[GAUGE::L][0][s1].block[0].adjoint() * 
//////				         Apair.A[s2][s4].block[0] * 
//////				         Vout.state.N[GAUGE::R][0][s3].block[0].adjoint();
//////			}
//////		}
//////		
//////		cout << "B2alt=" << B2alt.norm() << endl;
////		
////		//		double eps_svd = 1e-4;
//////		size_t Nret = (Jack.singularValues().array() > eps_svd).count();
//////		Nret = min(Nret,500-D);
//////		cout << "Nret=" << Nret << endl;
////		
//////		if (Nret > 0)
//////		{
//////			for (size_t s=0; s<D; ++s)
//////			{
//////				Vout.state.A[GAUGE::L][0][s].block[0].conservativeResize(M+Nret,M+Nret);
//////				Vout.state.A[GAUGE::L][0][s].block[0].block(0,D, M,Nret) = Vout.state.N[GAUGE::L][0][s].block[0] * 
//////				                                                           Jack.matrixU().leftCols(Nret);
//////				Vout.state.A[GAUGE::L][0][s].block[0].bottomRows(Nret).setZero();
//////				
//////				Vout.state.A[GAUGE::R][0][s].block[0].conservativeResize(M+Nret,M+Nret);
//////				Vout.state.A[GAUGE::R][0][s].block[0].block(D,0, Nret,M) = Jack.matrixV().adjoint().topRows(Nret) * 
//////				                                                           Vout.state.N[GAUGE::R][0][s].block[0];
//////				Vout.state.A[GAUGE::R][0][s].block[0].rightCols(Nret).setZero();
//////				
//////				Vout.state.A[GAUGE::C][0][s].block[0].conservativeResize(M+Nret,M+Nret);
//////				Vout.state.A[GAUGE::C][0][s].block[0].bottomRows(Nret).setZero();
//////				Vout.state.A[GAUGE::C][0][s].block[0].rightCols(Nret).setZero();
//////			}
//////			
//////			Vout.state.C[0].block[0].conservativeResize(M+Nret,M+Nret);
//////			Vout.state.C[0].block[0].bottomRows(Nret).setZero();
//////			Vout.state.C[0].block[0].rightCols(Nret).setZero();
//////			
//////			M += Nret;
//////			cout << "new M=" << M << endl;
//////		}
////	}
////	else
////	{
////		err_state = std::nan("1");
////	}
}

//template<typename Symmetry, typename MpHamiltonian, typename Scalar>
//void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
//iteration2 (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout)
//{
//	Stopwatch<> IterationTimer;
//	
//	// Pre-contract two A-tensors to a pair
//	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > ApairL;
//	vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > ApairR;
//	contract_AA(Vout.state.A[GAUGE::L][0], H.locBasis(0), Vout.state.A[GAUGE::L][1], H.locBasis(1), ApairL);
//	contract_AA(Vout.state.A[GAUGE::R][0], H.locBasis(0), Vout.state.A[GAUGE::R][1], H.locBasis(1), ApairR);
//	
//	// Pre-contract the MPO to a 4-tensor, but leave either the row of the first W-tensor or the column of the last W-tensor as a free index
//	typedef tuple<size_t,size_t,size_t,size_t,size_t,Scalar> Wtuple;
//	std::map<int,vector<Wtuple> > WWWWbyRow;
//	std::map<int,vector<Wtuple> > WWWWbyCol;
//	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	for (int k12=0; k12<H.W[0][s1][s2][0].outerSize(); ++k12)
//	for (typename SparseMatrix<Scalar>::InnerIterator iW12(H.W[0][s1][s2][0],k12); iW12; ++iW12)
//	for (size_t s3=0; s3<D; ++s3)
//	for (size_t s4=0; s4<D; ++s4)
//	for (int k34=0; k34<H.W[1][s3][s4][0].outerSize(); ++k34)
//	for (typename SparseMatrix<Scalar>::InnerIterator iW34(H.W[1][s3][s4][0],k34); iW34; ++iW34)
//	{
//		if (iW12.col()==iW34.row())
//		{
//			if (abs(iW12.value())>1e-15 and abs(iW34.value())>1e-15 and iW12.row()>iW34.col())
//			{
//				auto val = iW12.value() * iW34.value();
//				Wtuple r = make_tuple(iW34.col(),s1,s2,s3,s4,val);
//				Wtuple c = make_tuple(iW12.row(),s1,s2,s3,s4,val);
//				WWWWbyRow[iW12.row()].push_back(r);
//				WWWWbyCol[iW34.col()].push_back(c);
//			}
//		}
//	}
//	
//	// |R) and (L|
//	MatrixType Reigen = Vout.state.C[N_sites-1].block[0] * Vout.state.C[N_sites-1].block[0].adjoint();
//	MatrixType Leigen = Vout.state.C[N_sites-1].block[0].adjoint() * Vout.state.C[N_sites-1].block[0];
//	
//	// |YRa) and (YLa|
//	vector<MatrixType> YL(dW);
//	vector<MatrixType> YR(dW);
//	
//	for (size_t a=0; a<dW; ++a)
//	{
//		YL[a].resize(M,M); YL[a].setZero();
//		YR[a].resize(M,M); YR[a].setZero();
//	}
//	
//	// |Ra) and (La|
//	boost::multi_array<MatrixType,LEGLIMIT> L(boost::extents[dW][1]);
//	boost::multi_array<MatrixType,LEGLIMIT> R(boost::extents[dW][1]);
//	
//	L[dW-1][0].resize(M,M);
//	L[dW-1][0].setIdentity();
//	
//	Stopwatch<> GMresTimer;
//	
////	#pragma omp parallel sections
//	{
//		// Eq. C19
////		#pragma omp section
//		{
//			for (int b=dW-2; b>=0; --b)
//			{
//				YL[b] = make_YL(WWWWbyCol[b], L, ApairL, H.locBasis(0));
//				
//				boost::multi_array<Scalar,4> Warray = make_Warray4<MpHamiltonian,Scalar>(b,H);
//				
//				if (sum<MpHamiltonian,Scalar>(Warray) == 0.)
//				{
//					L[b][0] = YL[b];
//				}
//				else
//				{
//					double e = (YL[b]*Reigen).trace();
//					solve_linear(GAUGE::L, ApairL, YL[b], Reigen, Warray, e, L[b][0]);
//				}
//			}
//		}
//		// Eq. C20
////		#pragma omp section
//		{
//			R[0][0].resize(M,M);
//			R[0][0].setIdentity();
//			
//			for (int a=1; a<dW; ++a)
//			{
//				YR[a] = make_YR(WWWWbyRow[a], R, ApairR, H.locBasis(0));
//				
//				boost::multi_array<Scalar,4> Warray = make_Warray4<MpHamiltonian,Scalar>(a,H);
//				
//				if (sum<MpHamiltonian,Scalar>(Warray) == 0.)
//				{
//					R[a][0] = YR[a];
//				}
//				else
//				{
//					double e = (Leigen*YR[a]).trace();
//					solve_linear(GAUGE::R, ApairR, YR[a], Leigen, Warray, e, R[a][0]);
//				}
//			}
//		}
//	}
//	
//	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
//	{
//		lout << "linear systems" << GMresTimer.info() << endl;
//	}
//	
//	// With a unit cell, Heff is a vector for each site
//	vector<PivotMatrix1<Symmetry,Scalar,Scalar> > HeffA(N_sites);
//	vector<PivotMatrix1<Symmetry,Scalar,Scalar> > HeffAC(N_sites);
//	
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		HeffA[l].W  = H.W[l];
//		HeffAC[l].W = H.W[l];
//	}
//	
//	HeffA[0].L.push_back(qloc3dummy,L);
//	HeffA[N_sites-1].R.push_back(qloc3dummy,R);
//	
//	// Make environment for each site of the unit cell
//	for (size_t l=1; l<N_sites; ++l)
//	{
//		contract_L(HeffA[l-1].L, Vout.state.A[GAUGE::L][l-1], H.W[l-1], Vout.state.A[GAUGE::L][l-1], H.locBasis(l-1), H.opBasis(l-1), HeffA[l].L);
//	}
//	
//	for (int l=N_sites-2; l>=0; --l)
//	{
//		contract_R(HeffA[l+1].R, Vout.state.A[GAUGE::R][l+1], H.W[l+1], Vout.state.A[GAUGE::R][l+1], H.locBasis(l+1), H.opBasis(l+1), HeffA[l].R);
//	}
//	
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		HeffAC[l].L = HeffA[(l+1)%N_sites].L;
//		HeffAC[l].R = HeffA[l].R;
//	}
//	
//	vector<Eigenstate<PivotVector<Symmetry,Scalar> > > gAC(N_sites);
//	vector<Eigenstate<PivotVector<Symmetry,Scalar> > > gC(N_sites);
//	
//	double tolLanczosEigval, tolLanczosState;
//	set_LanczosTolerances(tolLanczosEigval,tolLanczosState);
//	
//	// local optimization
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		precalc_blockStructure (HeffA[l].L, Vout.state.A[GAUGE::C][l], HeffA[l].W, Vout.state.A[GAUGE::C][l], HeffA[l].R, 
//		                        H.locBasis(l), H.opBasis(l), HeffA[l].qlhs, HeffA[l].qrhs, HeffA[l].factor_cgcs);
//		
//		// Solve for AC
//		gAC[l].state = PivotVector<Symmetry,Scalar>(Vout.state.A[GAUGE::C][l]);
//		
//		Stopwatch<> LanczosTimer;
//		LanczosSolver<PivotMatrix1<Symmetry,Scalar,Scalar>,PivotVector<Symmetry,Scalar>,Scalar> Lutz(LANCZOS::REORTHO::FULL,LANCZOS::CONVTEST::SQ_TEST);
//		Lutz.set_dimK(min(30ul, dim(gAC[l].state)));
//		Lutz.edgeState(HeffA[l],gAC[l], LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
//		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
//		{
//			lout << "l=" << l << ", time" << LanczosTimer.info() << ", " << Lutz.info() << endl;
//		}
//		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
//		{
//			lout << "e0(AC)=" << setprecision(13) << gAC[l].energy << endl;
//		}
//		
//		// Solve for C
//		gC[l].state = PivotVector<Symmetry,Scalar>(Vout.state.C[l]);
//		
//		LanczosSolver<PivotMatrix0<Symmetry,Scalar,Scalar>,PivotVector<Symmetry,Scalar>,Scalar> Lucy(LANCZOS::REORTHO::FULL,LANCZOS::CONVTEST::SQ_TEST);
//		Lucy.set_dimK(min(30ul, dim(gC[l].state)));
//		Lucy.edgeState(PivotMatrix0(HeffAC[l]),gC[l], LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
//		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
//		{
//			lout << "l=" << l << ", time" << LanczosTimer.info() << ", " << Lucy.info() << endl;
//		}
//		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
//		{
//			lout << "e0(C)=" << setprecision(13) << gC[l].energy << endl;
//		}
//	}
//	
//	// Calculate AL, AR from AC, C
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		Vout.state.A[GAUGE::C][l] = gAC[l].state.data;
//		Vout.state.C[l]           = gC[l].state.data[0];
//	}
//	
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		(err_var>0.1)? Vout.state.svdDecompose(l) : Vout.state.polarDecompose(l);
//	}
//	
//	// Calculate energies
//	eL = (YL[0]*Reigen).trace() / H.volume();
//	eR = (Leigen*YR[dW-1]).trace() / H.volume();
//	
//	// Calculate errors
//	MatrixXd epsLRsq(N_sites,2);
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		Vout.state.calc_epsLRsq(l,epsLRsq(l,0),epsLRsq(l,1));
//	}
//	err_var = sqrt(epsLRsq.sum());
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
//		lout << "S=" << Vout.state.entropy(0) << ", " << Vout.state.entropy(1) << endl;
//		lout << eigeninfo() << endl;
//		lout << IterationTimer.info("full iteration") << endl;
//		lout << endl;
//	}
//}

//template<typename Symmetry, typename MpHamiltonian, typename Scalar>
//void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
//iteration4 (const MpHamiltonian &H, Eigenstate<Umps<Symmetry,Scalar> > &Vout)
//{
//	Stopwatch<> IterationTimer;
//	
//	// Pre-contract four A-tensors to a quadruple
//	boost::multi_array<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> >,4> AquadrupleL;
//	AquadrupleL.resize(boost::extents[D][D][D][D]);
//	boost::multi_array<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> >,4> AquadrupleR;
//	AquadrupleR.resize(boost::extents[D][D][D][D]);
//	contract_AAAA(Vout.state.A[GAUGE::L][0], H.locBasis(0), 
//	              Vout.state.A[GAUGE::L][1], H.locBasis(1), 
//	              Vout.state.A[GAUGE::L][2], H.locBasis(2), 
//	              Vout.state.A[GAUGE::L][3], H.locBasis(3), 
//	              AquadrupleL);
//	contract_AAAA(Vout.state.A[GAUGE::R][0], H.locBasis(0), 
//	              Vout.state.A[GAUGE::R][1], H.locBasis(1), 
//	              Vout.state.A[GAUGE::R][2], H.locBasis(2), 
//	              Vout.state.A[GAUGE::R][3], H.locBasis(3), 
//	              AquadrupleR);
//	
//	// Pre-contract the MPO to an 8-tensor, but leave either the row of the first W-tensor or the column of the last W-tensor as a free index
//	typedef tuple<size_t,size_t,size_t,size_t,size_t,size_t,size_t,size_t,size_t,Scalar> Wtuple;
//	std::map<int,vector<Wtuple> > WWWWoffDiagRow;
//	std::map<int,vector<Wtuple> > WWWWoffDiagCol;
//	std::map<int,vector<Wtuple> > WWWWdiag;
//	
//	for (size_t s1=0; s1<D; ++s1)
//	for (size_t s2=0; s2<D; ++s2)
//	for (int k12=0; k12<H.W[0][s1][s2][0].outerSize(); ++k12)
//	for (typename SparseMatrix<Scalar>::InnerIterator iW12(H.W[0][s1][s2][0],k12); iW12; ++iW12)
//	for (size_t s3=0; s3<D; ++s3)
//	for (size_t s4=0; s4<D; ++s4)
//	for (int k34=0; k34<H.W[1][s3][s4][0].outerSize(); ++k34)
//	for (typename SparseMatrix<Scalar>::InnerIterator iW34(H.W[1][s3][s4][0],k34); iW34; ++iW34)
//	for (size_t s5=0; s5<D; ++s5)
//	for (size_t s6=0; s6<D; ++s6)
//	for (int k56=0; k56<H.W[2][s5][s6][0].outerSize(); ++k56)
//	for (typename SparseMatrix<Scalar>::InnerIterator iW56(H.W[2][s5][s6][0],k56); iW56; ++iW56)
//	for (size_t s7=0; s7<D; ++s7)
//	for (size_t s8=0; s8<D; ++s8)
//	for (int k78=0; k78<H.W[3][s7][s8][0].outerSize(); ++k78)
//	for (typename SparseMatrix<Scalar>::InnerIterator iW78(H.W[3][s7][s8][0],k78); iW78; ++iW78)
//	{
//		if (iW12.col()==iW34.row() and 
//		    iW34.col()==iW56.row() and 
//		    iW56.col()==iW78.row())
//		{
//			if (    abs(iW12.value())>1e-15 and abs(iW34.value())>1e-15 
//			    and abs(iW56.value())>1e-15 and abs(iW78.value())>1e-15
//			    and iW12.row()>iW78.col())
//			{
//				auto val = iW12.value() * iW34.value() * iW56.value() * iW78.value();
//				Wtuple r = make_tuple(iW78.col(),s1,s2,s3,s4,s5,s6,s7,s8,val);
//				Wtuple c = make_tuple(iW12.row(),s1,s2,s3,s4,s5,s6,s7,s8,val);
//				WWWWoffDiagRow[iW12.row()].push_back(r);
//				WWWWoffDiagCol[iW78.col()].push_back(c);
//			}
//		}
//	}
//	
//	// |R) and (L|
//	MatrixType Reigen = Vout.state.C[N_sites-1].block[0] * Vout.state.C[N_sites-1].block[0].adjoint();
//	MatrixType Leigen = Vout.state.C[N_sites-1].block[0].adjoint() * Vout.state.C[N_sites-1].block[0];
//	
//	// |YRa) and (YLa|
//	vector<MatrixType> YL(dW);
//	vector<MatrixType> YR(dW);
//	
//	for (size_t a=0; a<dW; ++a)
//	{
//		YL[a].resize(M,M); YL[a].setZero();
//		YR[a].resize(M,M); YR[a].setZero();
//	}
//	
//	// |Ra) and (La|
//	boost::multi_array<MatrixType,LEGLIMIT> L(boost::extents[dW][1]);
//	boost::multi_array<MatrixType,LEGLIMIT> R(boost::extents[dW][1]);
//	
//	L[dW-1][0].resize(M,M);
//	L[dW-1][0].setIdentity();
//	R[0][0].resize(M,M);
//	R[0][0].setIdentity();
//	
//	Stopwatch<> GMresTimer;
//	
////	#pragma omp parallel sections
//	{
//		// Eq. C19
////		#pragma omp section
//		{
//			for (int b=dW-2; b>=0; --b)
//			{
//				YL[b] = make_YL(WWWWoffDiagCol[b], L, AquadrupleL, H.locBasis(0));
//				
//				boost::multi_array<Scalar,8> Warray = make_Warray8<MpHamiltonian,Scalar>(b,H);
//				
//				if (sum<MpHamiltonian,Scalar>(Warray) == 0.)
//				{
//					L[b][0] = YL[b];
//				}
//				else
//				{
//					double e = (YL[b]*Reigen).trace();
//					solve_linear(GAUGE::L, AquadrupleL, YL[b], Reigen, Warray, e, L[b][0]);
//				}
//			}
//		}
//		// Eq. C20
////		#pragma omp section
//		{
//			for (int a=1; a<dW; ++a)
//			{
//				YR[a] = make_YR(WWWWoffDiagRow[a], R, AquadrupleR, H.locBasis(0));
//				
//				boost::multi_array<Scalar,8> Warray = make_Warray8<MpHamiltonian,Scalar>(a,H);
//				
//				if (sum<MpHamiltonian,Scalar>(Warray) == 0.)
//				{
//					R[a][0] = YR[a];
//				}
//				else
//				{
//					double e = (Leigen*YR[a]).trace();
//					solve_linear(GAUGE::R, AquadrupleR, YR[a], Leigen, Warray, e, R[a][0]);
//				}
//			}
//		}
//	}
//	
//	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
//	{
//		lout << "linear systems" << GMresTimer.info() << endl;
//	}
//	
//	vector<Eigenstate<PivotVector<Symmetry,Scalar> > > gAC(N_sites);
//	vector<Eigenstate<PivotVector<Symmetry,Scalar> > > gC(N_sites);
//	
//	double tolLanczosEigval, tolLanczosState;
//	set_LanczosTolerances(tolLanczosEigval,tolLanczosState);
//	
//	// With a unit cell, Heff is a vector for each site
//	vector<PivotMatrix1<Symmetry,Scalar,Scalar> > HeffA(N_sites);
//	vector<PivotMatrix1<Symmetry,Scalar,Scalar> > HeffAC(N_sites);
//	
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		HeffA[l].W  = H.W[l];
//		HeffAC[l].W = H.W[l];
//	}
//	
//	HeffA[0].L.push_back(qloc3dummy,L);
//	HeffA[N_sites-1].R.push_back(qloc3dummy,R);
//	
//	// Make environment for each site of the unit cell
//	for (size_t l=1; l<N_sites; ++l)
//	{
//		contract_L(HeffA[l-1].L, Vout.state.A[GAUGE::L][l-1], H.W[l-1], Vout.state.A[GAUGE::L][l-1], H.locBasis(l-1), H.opBasis(l-1), HeffA[l].L);
//	}
//	
//	for (int l=N_sites-2; l>=0; --l)
//	{
//		contract_R(HeffA[l+1].R, Vout.state.A[GAUGE::R][l+1], H.W[l+1], Vout.state.A[GAUGE::R][l+1], H.locBasis(l+1), H.opBasis(l+1), HeffA[l].R);
//	}
//	
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		HeffAC[l].L = HeffA[(l+1)%N_sites].L;
//		HeffAC[l].R = HeffA[l].R;
//	}
//	
//	// local optimization
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		precalc_blockStructure (HeffA[l].L, Vout.state.A[GAUGE::C][l], HeffA[l].W, Vout.state.A[GAUGE::C][l], HeffA[l].R, 
//		                        H.locBasis(l), H.opBasis(l), HeffA[l].qlhs, HeffA[l].qrhs, HeffA[l].factor_cgcs);
//		
//		// Solve for AC
//		gAC[l].state = PivotVector<Symmetry,Scalar>(Vout.state.A[GAUGE::C][l]);
//		
//		Stopwatch<> LanczosTimer;
//		LanczosSolver<PivotMatrix1<Symmetry,Scalar,Scalar>,PivotVector<Symmetry,Scalar>,Scalar> Lutz(LANCZOS::REORTHO::FULL,LANCZOS::CONVTEST::SQ_TEST);
//		Lutz.set_dimK(min(30ul, dim(gAC[l].state)));
//		Lutz.edgeState(HeffA[l],gAC[l], LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
//		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
//		{
//			lout << "l=" << l << ", time" << LanczosTimer.info() << ", " << Lutz.info() << endl;
//		}
//		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
//		{
//			lout << "e0(AC)=" << setprecision(13) << gAC[l].energy << endl;
//		}
//		
//		// Solve for C
//		gC[l].state = PivotVector<Symmetry,Scalar>(Vout.state.C[l]);
//		
//		LanczosSolver<PivotMatrix0<Symmetry,Scalar,Scalar>,PivotVector<Symmetry,Scalar>,Scalar> Lucy(LANCZOS::REORTHO::FULL,LANCZOS::CONVTEST::SQ_TEST);
//		Lucy.set_dimK(min(30ul, dim(gC[l].state)));
//		Lucy.edgeState(PivotMatrix0(HeffAC[l]),gC[l], LANCZOS::EDGE::GROUND, tolLanczosEigval,tolLanczosState, false);
//		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
//		{
//			lout << "l=" << l << ", time" << LanczosTimer.info() << ", " << Lucy.info() << endl;
//		}
//		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
//		{
//			lout << "e0(C)=" << setprecision(13) << gC[l].energy << endl;
//		}
//	}
//	
//	// Calculate AL, AR from AC, C
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		Vout.state.A[GAUGE::C][l] = gAC[l].state.data;
//		Vout.state.C[l]           = gC[l].state.data[0];
//	}
//	
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		(err_var>0.1)? Vout.state.svdDecompose(l) : Vout.state.polarDecompose(l);
//	}
//	
//	// Calculate energies
//	eL = (YL[0]*Reigen).trace() / H.volume();
//	eR = (Leigen*YR[dW-1]).trace() / H.volume();
//	
//	// Calculate errors
//	MatrixXd epsLRsq(N_sites,2);
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		Vout.state.calc_epsLRsq(l,epsLRsq(l,0),epsLRsq(l,1));
//	}
//	err_var = sqrt(epsLRsq.sum());
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
//		lout << "S=" << Vout.state.entropy(0) << ", " 
//		             << Vout.state.entropy(1) << ", " 
//		             << Vout.state.entropy(2) << ", " 
//		             << Vout.state.entropy(3) << endl;
//		lout << eigeninfo() << endl;
//		lout << IterationTimer.info("full iteration") << endl;
//		lout << endl;
//	}
//}

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
		if      (N_sites==1) {iteration1(H,Vout);}
//		else if (N_sites==2) {iteration2(H,Vout);}
//		else if (N_sites==4) {iteration4(H,Vout);}
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
		lout << "emin=" << setprecision(13) << Vout.energy << setprecision(standard_precision) << endl;
		lout << Vout.state.info() << endl;
		lout << endl;
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
solve_linear (GAUGE::OPTION gauge, 
              size_t ab, 
              const vector<Biped<Symmetry,MatrixType> > &A, 
              const Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &hLR, 
              const Biped<Symmetry,MatrixType> &LReigen, 
              const MpHamiltonian &H, 
              double e, 
              Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > &Hres)
{
	// local dimension for the whole unit cell
//	vector<size_t> Dvec(N_sites);
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		Dvec[l] = D;
//	}
	
	// Transfer operator
//	TransferMatrix<Symmetry,Scalar> T(gauge, A, A, LReigen, Warray, Dvec);
	TransferMatrix<Symmetry,Scalar> T(gauge, A, A, LReigen, H.W[0], H.locBasis(0), H.opBasis(0), ab);
	
	// Right-hand site vector |hLR)-e*1
	TransferVector<Symmetry,Scalar> bvec;
	bvec.data = hLR;
	bvec.ab = ab;
	for (size_t q=0; q<bvec.data.dim; ++q)
	{
		bvec.data.block[q][ab][0] -= e * MatrixType::Identity(bvec.data.block[q][ab][0].rows(),bvec.data.block[q][ab][0].cols());
	}
//	bvec.A -= e * MatrixType::Identity(bvec.A.rows(),bvec.A.cols());
	
	// Solve linear system
	GMResSolver<TransferMatrix<Symmetry,Scalar>,TransferVector<Symmetry,Scalar> > Gimli;
	Gimli.set_dimK(min(10ul,M*M));
	TransferVector<Symmetry,Scalar> Hres_tmp;
	Gimli.solve_linear(T,bvec,Hres_tmp);
	Hres = Hres_tmp.data;
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
	{
		lout << Gimli.info() << endl;
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
MatrixXd VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
eigenvectorL (const MatrixType &TL)
{
	EigenSolver<MatrixType> Lutz(TL);
	int max_index;
	Lutz.eigenvalues().cwiseAbs().maxCoeff(&max_index);
	
	MatrixType Mout(M,M);
	
	for (size_t i=0; i<M; ++i)
	for (size_t j=0; j<M; ++j)
	{
		size_t r = i + M*j;
		Mout(i,j) = Lutz.eigenvectors().col(max_index)(r).real();
	}
	
	return Mout;
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
MatrixXd VumpsSolver<Symmetry,MpHamiltonian,Scalar>::
eigenvectorR (const MatrixType &TR)
{
	EigenSolver<MatrixType> Lutz(TR.adjoint());
	int max_index;
	Lutz.eigenvalues().cwiseAbs().maxCoeff(&max_index);
	
	MatrixType Mout(M,M);
	
	for (size_t i=0; i<M; ++i)
	for (size_t j=0; j<M; ++j)
	{
		size_t r = i + M*j;
		Mout(i,j) = Lutz.eigenvectors().col(max_index)(r).real();
	}
	
	return Mout;
}

#endif
