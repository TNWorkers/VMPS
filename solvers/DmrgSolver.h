#ifndef STRAWBERRY_DMRGSOLVER_WITH_Q
#define STRAWBERRY_DMRGSOLVER_WITH_Q

/// \cond
#include "termcolor.hpp" //from https://github.com/ikalnytskyi/termcolor
/// \endcond

#ifdef USE_HDF5_STORAGE
	#include <HDF5Interface.h> // from TOOLS
#endif

#include "LanczosSolver.h" // from ALGS
#include "Stopwatch.h" // from TOOLS
#include "TerminalPlot.h"

#include "Mps.h"
#include "DmrgLinearAlgebra.h" // for avg()
#include "pivot/DmrgPivotMatrix0.h"

//include "solvers/MpsCompressor.h"
//include "tensors/DmrgContractions.h"
//include "Mpo.h"

struct SweepStatus
{
	int pivot = -1;
	DMRG::DIRECTION::OPTION CURRENT_DIRECTION;
	size_t N_sweepsteps = 0;
	size_t N_halfsweeps = 0;
};

template<typename Symmetry, typename MpHamiltonian, typename Scalar = double>
class DmrgSolver
{
	static constexpr size_t Nq = Symmetry::Nq;
	typedef typename Symmetry::qType qType;
public:
	
	DmrgSolver (DMRG::VERBOSITY::OPTION VERBOSITY=DMRG::VERBOSITY::SILENT)
	:CHOSEN_VERBOSITY(VERBOSITY)
	{};
	
	string info() const;
	string eigeninfo() const;
	double memory (MEMUNIT memunit=GB) const;
	
	void edgeState (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, 
	                qarray<Nq> Qtot_input, LANCZOS::EDGE::OPTION EDGE = LANCZOS::EDGE::GROUND, bool USE_STATE=false);

	//call this function if you want to set the parameters for the solver by yourself
	void userSetGlobParam    () { USER_SET_GLOBPARAM     = true; }
	void userSetDynParam     () { USER_SET_DYNPARAM      = true; }
	void userSetLanczosParam () { USER_SET_LANCZOSPARAM  = true; }

	DMRG::CONTROL::GLOB GlobParam;
	DMRG::CONTROL::DYN  DynParam;
	DMRG::CONTROL::LANCZOS LanczosParam;

	inline void set_verbosity (DMRG::VERBOSITY::OPTION VERBOSITY) {CHOSEN_VERBOSITY = VERBOSITY;};
	inline DMRG::VERBOSITY::OPTION get_verbosity () const {return CHOSEN_VERBOSITY;};
	
	void prepare (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, 
	              qarray<Nq> Qtot_input, bool USE_STATE=false);
	
	void halfsweep (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, 
	                LANCZOS::EDGE::OPTION EDGE = LANCZOS::EDGE::GROUND);
	
	void cleanup (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, 
	              LANCZOS::EDGE::OPTION EDGE = LANCZOS::EDGE::GROUND);

	///\{
	/**
	 * Performs an 0-site iteration, 
	 * e.g. solves the effective eigenvalue problem of the 0-site effective Hamiltonian and updates therewith the center-matrix C.
	 * \warning This iteration is not extensively tested and there is no guarantee that the sweep protocol is correct.
	 * \warning No subspace expansion scheme is included. -> bad convergence.
	 */
	void iteration_zero (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, LANCZOS::EDGE::OPTION EDGE,
	                     double &time_lanczos, double &time_sweep, double &time_LR, double &time_overhead);
	/**
	 * Performs an 1-site iteration, 
	 * e.g. solves the effective eigenvalue problem of the 1-site effective Hamiltonian and updates therewith the A-tensors directly.
	 * \note Standard iteration. Best tested and suited with an expansion scheme to enlarge the bond dimension.
	 */
	void iteration_one  (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, LANCZOS::EDGE::OPTION EDGE,
	                     double &time_lanczos, double &time_sweep, double &time_LR, double &time_overhead);
	/**
	 * Performs an 2-site iteration, 
	 * e.g. solves the effective eigenvalue problem of the 2-site effective Hamiltonian and updates therewith the two-site wavefunction.
	 * \note Bond dimension gets enlarged automatically. The truncated weight is a good convergence measure here and can also be used for extrapolations.
	 * \note This algorithm is quite slow for challenging problems.
	 */
	void iteration_two  (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, LANCZOS::EDGE::OPTION EDGE,
	                     double &time_lanczos, double &time_sweep, double &time_LR, double &time_overhead);
	///\}

	/**Returns the current error of the eigenvalue while the sweep process.*/
	inline double get_errEigval() const {return err_eigval;};
	
	/**Returns the current error of the state while the sweep process.*/
	inline double get_errState() const {return err_state;};
	
	/**Returns the current pivot site of the sweep process.*/
	inline double get_pivot() const {return SweepStat.pivot;};
	
	/**Returns the current direction of the sweep process.*/
	inline double get_direction() const {return SweepStat.CURRENT_DIRECTION;};
	
	void push_back (const Mps<Symmetry,Scalar> &Psi0_input) {Psi0.push_back(Psi0_input);};
	
	#ifdef USE_HDF5_STORAGE
	/**Save the current SweepStatus to <filename>.h5.*/
	void save (string filename) const;
	
	/**Load the a SweepStatus from <filename>.h5.*/
	void load (string filename);
	#endif
	
	/**Energy penalty for projected-out states.*/
	double Epenalty = 1e4;
	
	inline void set_SweepStatus (const SweepStatus &SweepStat_input)
	{
		SweepStat = SweepStat_input;
	}
	
private:
	
	size_t N_sites, N_phys;
	size_t Dmax, Mmax, Nqmax;
	double totalTruncWeight;
	size_t Mmax_old;
	double err_eigval, err_state;
	
	vector<PivotMatrix1<Symmetry,Scalar,Scalar> > Heff; // Scalar = MpoScalar for ground state
	
	double Eold;
	
	double DeltaEopt;
	double max_alpha_rsvd, min_alpha_rsvd;
	
	bool USER_SET_GLOBPARAM    = false;
	bool USER_SET_DYNPARAM     = false;
	bool USER_SET_LANCZOSPARAM = false;
	
	SweepStatus SweepStat;
	
	inline size_t loc1() const {return (SweepStat.CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? SweepStat.pivot : SweepStat.pivot-1;};
	inline size_t loc2() const {return (SweepStat.CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? SweepStat.pivot+1 : SweepStat.pivot;};
	
	void LanczosStep (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, LANCZOS::EDGE::OPTION EDGE);
	void sweep_to_edge (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, bool MAKE_ENVIRONMENT);
	
	/**Constructs the left transfer matrix at chain site \p loc (left environment of \p loc).*/
	void build_L (const MpHamiltonian &H, const Eigenstate<Mps<Symmetry,Scalar> > &Vout, size_t loc);
	
	/**Constructs the right transfer matrix at chain site \p loc (right environment of \p loc).*/
	void build_R (const MpHamiltonian &H, const Eigenstate<Mps<Symmetry,Scalar> > &Vout, size_t loc);
	
	void build_PL (const MpHamiltonian &H, const Eigenstate<Mps<Symmetry,Scalar> > &Vout, size_t loc);
	void build_PR (const MpHamiltonian &H, const Eigenstate<Mps<Symmetry,Scalar> > &Vout, size_t loc);
	
	void adapt_alpha_rsvd (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, LANCZOS::EDGE::OPTION EDGE);
	
	/**Projected-out states to find the edge of the spectrum.*/
	vector<Mps<Symmetry,Scalar> > Psi0;
	double E0;
	
	DMRG::VERBOSITY::OPTION CHOSEN_VERBOSITY;
};

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
string DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
info() const
{
	stringstream ss;
	ss << "DmrgSolver: ";
	ss << "L=" << N_sites << ", ";
	ss << "Mmax=" << Mmax << ", Dmax=" << Dmax << ", " << "Nqmax=" << Nqmax << ", ";
	ss << "trunc_weight=" << totalTruncWeight << ", ";
	ss << eigeninfo();
	return ss.str();
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
string DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
eigeninfo() const
{
	stringstream ss;
	ss << termcolor::colorize << termcolor::underline << "half-sweeps=" << SweepStat.N_halfsweeps;
	ss << termcolor::reset;
	ss << ", next algorithm=" << DynParam.iteration(SweepStat.N_halfsweeps);
	ss << ", ";
	ss << "err_eigval=" << err_eigval << ", err_state=" << err_state << ", ";
	ss << "mem=" << round(memory(GB),3) << "GB";
	return ss.str();
}

#ifdef USE_HDF5_STORAGE
template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
save (string filename) const
{
	filename+=".h5";
	HDF5Interface target(filename, WRITE);
	target.save_scalar(SweepStat.pivot,"pivot");
	target.save_scalar(SweepStat.N_halfsweeps,"N_halfsweeps");
	target.save_scalar(SweepStat.N_sweepsteps,"N_sweepsteps");
	target.save_scalar(static_cast<int>(SweepStat.CURRENT_DIRECTION),"direction");
	target.save_scalar(Dmax,"D");
	target.save_scalar(Mmax,"M");
	target.save_scalar(err_eigval,"errorE");
	target.save_scalar(err_state,"errorS");
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
load (string filename)
{
	filename+=".h5";
	HDF5Interface source(filename, READ);
	source.load_scalar(SweepStat.pivot,"pivot");
	source.load_scalar(SweepStat.N_halfsweeps,"N_halfsweeps");
	source.load_scalar(SweepStat.N_sweepsteps,"N_sweepsteps");
	int DIR_IN_INT;
	source.load_scalar(DIR_IN_INT,"direction");
	SweepStat.CURRENT_DIRECTION = static_cast<DMRG::DIRECTION::OPTION>(DIR_IN_INT);
	source.load_scalar(err_eigval,"errorE");
	source.load_scalar(err_state,"errorS");
}
#endif

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
double DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
memory (MEMUNIT memunit) const
{
	double res = 0.;
	for (size_t l=0; l<N_sites; ++l)
	{
		res += Heff[l].L.memory(memunit);
		res += Heff[l].R.memory(memunit);
		for (size_t s1=0; s1<Heff[l].W.size(); ++s1)
		for (size_t s2=0; s2<Heff[l].W[s1].size(); ++s2)
		for (size_t k=0; k<Heff[l].W[s1][s2].size(); ++k)
		{
			res += Heff[l].W[s1][s2][k].memory(memunit);
		}
	}
	return res;
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
prepare (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, qarray<Nq> Qtot_input, bool USE_STATE)
{
	if (CHOSEN_VERBOSITY>=2)
	{
		lout << endl << termcolor::colorize << termcolor::bold
		 << "————————————————————————————————————————————DMRG Algorithm————————————————————————————————————————————"
		 <<  termcolor::reset << endl;
	}
	
	N_sites = H.length();
	N_phys  = H.volume();
	
//	if (!USER_SET_GLOBPARAM) { GlobParam = H.get_DmrgGlobParam(); }
//	if (!USER_SET_DYNPARAM)  { DynParam  = H.get_DmrgDynParam(); }
	
	// set edges
	Heff.clear();
	Heff.resize(N_sites);
	if (Vout.state.Boundaries.IS_TRIVIAL())
	{
		Heff[0].L.setVacuum();
		Heff[N_sites-1].R.setTarget(qarray3<Nq>{Qtot_input, Qtot_input, Symmetry::qvacuum()});
	}
	else
	{
		Heff[0].L         = Vout.state.get_boundaryTensor(DMRG::DIRECTION::LEFT);
		Heff[N_sites-1].R = Vout.state.get_boundaryTensor(DMRG::DIRECTION::RIGHT);
	}
	
	Stopwatch<> PrepTimer;
	if (!USE_STATE)
	{
		// resize Vout
		auto Boundaries_tmp = Vout.state.Boundaries; // save to temporary, otherwise reset in the following constructor
		Vout.state = Mps<Symmetry,Scalar>(H, GlobParam.Minit, Qtot_input, GlobParam.Qinit);
		Vout.state.max_Nsv = GlobParam.Minit;
		Vout.state.min_Nsv = DynParam.min_Nsv(0);
		Vout.state.max_Nrich = DynParam.max_Nrich(0);
		Vout.state.Boundaries = Boundaries_tmp;
//		Vout.state.graph("init");
		// If boundaries not pre-set, set them to trivial now:
		if (Vout.state.Boundaries.IS_TRIVIAL()) Vout.state.Boundaries.set_open_bc(Qtot_input);
		
		Vout.state.update_inbase();
		Vout.state.update_outbase();
		Vout.state.calc_Qlimits();
		
		Vout.state.setRandom();
		
		// adjust corners for IBC
		if (H.GOT_SEMIOPEN_LEFT or !H.get_boundary_condition())
		{
			for (size_t s=0; s<Vout.state.qloc[N_sites-1].size(); ++s)
			for (size_t q=0; q<Vout.state.A[N_sites-1][s].dim; ++q)
			{
				for (size_t r=0; r<Heff[N_sites-1].R.dim; ++r)
				for (size_t a=0; a<Heff[N_sites-1].R.block[r].shape()[0]; ++a)
				{
					if (Heff[N_sites-1].R.block[r][a][0].size() != 0)
					{
						if (Vout.state.A[N_sites-1][s].out[q] == Heff[N_sites-1].R.in(r))
						{
							Vout.state.A[N_sites-1][s].block[q].resize(Vout.state.A[N_sites-1][s].block[q].rows(),
							                                           Heff[N_sites-1].R.block[r][a][0].rows());
							Vout.state.A[N_sites-1][s].block[q].setRandom();
						}
					}
				}
			}
			Vout.state.update_inbase();
			Vout.state.update_outbase();
		}
		if (H.GOT_SEMIOPEN_RIGHT or !H.get_boundary_condition())
		{
			for (size_t s=0; s<Vout.state.qloc[0].size(); ++s)
			for (size_t q=0; q<Vout.state.A[0][s].dim; ++q)
			{
				for (size_t r=0; r<Heff[0].L.dim; ++r)
				for (size_t a=0; a<Heff[0].L.block[r].shape()[0]; ++a)
				{
					if (Heff[0].L.block[r][a][0].size() != 0)
					{
						if (Vout.state.A[0][s].in[q] == Heff[0].L.out(r))
						{
							Vout.state.A[0][s].block[q].resize(Heff[0].L.block[r][a][0].cols(),
							                                   Vout.state.A[0][s].block[q].cols());
							Vout.state.A[0][s].block[q].setRandom();
						}
					}
				}
			}
			Vout.state.update_inbase();
			Vout.state.update_outbase();
		}
		
		// leads to segfault with local basis:
//		if (!H.GOT_OPEN_BC)
//		{
//			for (int l=0; l<N_sites-1; ++l)
//			{
//				Vout.state.A[l] = Vout.state.Boundaries.A[0][l%Vout.state.Boundaries.length()];
//			}
//			Vout.state.A[N_sites-1] = Vout.state.Boundaries.A[2][(N_sites-1)%Vout.state.Boundaries.length()];
//			
//			Vout.state.update_inbase();
//			Vout.state.update_outbase();
//		}
		
		Mmax_old = GlobParam.Minit;
	}
	else
	{
		Vout.state.max_Nsv = Vout.state.calc_Mmax();
		Mmax_old = Vout.state.max_Nsv;
	}
	
//	Vout.state.graph("ginit");
	
	//if the SweepStatus is default initialized (pivot==-1), one initial sweep from right-to-left and N_halfsweeps = N_sweepsteps = 0,
	//otherwise prepare for continuing at the given SweepStatus.
	if (SweepStat.pivot == -1)
	{
		SweepStat.N_sweepsteps = SweepStat.N_halfsweeps = 0;
		if (GlobParam.INITDIR == DMRG::DIRECTION::RIGHT)
		{
			for (int l=N_sites-1; l>0; --l)
			{
				if (!USE_STATE) {Vout.state.setRandom(l);} // Avoids overflow for large chains. Don't set random for loaded states.
				Vout.state.sweepStep(DMRG::DIRECTION::LEFT, l, DMRG::BROOM::QR);
				build_R(H,Vout,l-1);
			}
			Vout.state.sweepStep(DMRG::DIRECTION::LEFT, 0, DMRG::BROOM::QR, NULL, true); // removes large numbers from first matrix
			SweepStat.CURRENT_DIRECTION = DMRG::DIRECTION::RIGHT;
			SweepStat.pivot = 0;
		}
		else
		{
			for (size_t l=0; l<N_sites-1; ++l)
			{
				if (!USE_STATE) {Vout.state.setRandom(l);} // Avoids overflow for large chains. Don't set random for loaded states.
				Vout.state.sweepStep(DMRG::DIRECTION::RIGHT, l, DMRG::BROOM::QR);
				build_L(H,Vout,l+1);
			}
			Vout.state.sweepStep(DMRG::DIRECTION::RIGHT, N_sites-1, DMRG::BROOM::QR, NULL, true); // removes large numbers from first matrix
			SweepStat.CURRENT_DIRECTION = DMRG::DIRECTION::LEFT;
			SweepStat.pivot = N_sites-1;
		}
	}
	else
	{
		if (SweepStat.CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)
		{
			for (size_t l=N_sites-1; l>0; --l)
			{
				Vout.state.sweepStep(DMRG::DIRECTION::LEFT, l, DMRG::BROOM::QR);
				build_R(H,Vout,l-1);
			}
			Vout.state.sweepStep(DMRG::DIRECTION::LEFT, 0, DMRG::BROOM::QR); // removes large numbers from first matrix
			
			for (size_t l=0; l<SweepStat.pivot; ++l)
			{
				Vout.state.sweepStep(DMRG::DIRECTION::RIGHT, l, DMRG::BROOM::QR);
				build_L(H,Vout,l+1);
			}
		}
		else if (SweepStat.CURRENT_DIRECTION == DMRG::DIRECTION::LEFT)
		{
			for (size_t l=0; l<N_sites-1; ++l)
			{
				Vout.state.sweepStep(DMRG::DIRECTION::RIGHT, l, DMRG::BROOM::QR);
				build_L(H,Vout,l+1);
			}
			Vout.state.sweepStep(DMRG::DIRECTION::RIGHT, N_sites-1, DMRG::BROOM::QR); // removes large numbers from first matrix
			
			for (size_t l=N_sites-1; l>SweepStat.pivot; --l)
			{
				Vout.state.sweepStep(DMRG::DIRECTION::LEFT, l, DMRG::BROOM::QR);
				build_R(H,Vout,l-1);
			}
		}
	}
	
	// resize environments for projected-out states
	if (Psi0.size() > 0)
	{
		for (size_t l=0; l<N_sites; ++l)
		{
			Heff[l].Epenalty = Epenalty;
			Heff[l].PL.resize(Psi0.size());
			Heff[l].PR.resize(Psi0.size());
		}
		E0 = isReal(avg(Psi0[0], H, Psi0[0]));
	}
	
	// build environments for projected-out states
	// convention: Psi0 ist ket, current Psi is bra
	for (size_t n=0; n<Psi0.size(); ++n)
	{
		Heff[0].PL[n].setVacuum();
		for (size_t l=1; l<N_sites; ++l) build_PL(H,Vout,l);
		
		Heff[N_sites-1].PR[n].setTarget(Vout.state.Qtot);
		for (int l=N_sites-2; l>=0; --l) build_PR(H,Vout,l);
	}
	
	// initial energy
	if (SweepStat.pivot == 0)
	{
		Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Rtmp;
		contract_R(Heff[0].R, Vout.state.A[0], H.W[0], Vout.state.A[0], H.locBasis(0), H.opBasis(0), Rtmp);
		if (Rtmp.dim == 0)
		{
			Eold = 0;
		}
		else
		{
			assert(Rtmp.dim == 1 and
			Rtmp.block[0][0][0].rows() == 1 and
			Rtmp.block[0][0][0].cols() == 1 and
			"Result of contraction <ψ|H|ψ> in DmrgSolver::prepare is not a scalar!");
			Eold = isReal(Rtmp.block[0][0][0](0,0));
		}
	}
	else if (SweepStat.pivot == N_sites-1)
	{
		Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Ltmp;
		contract_L(Heff[N_sites-1].L, Vout.state.A[N_sites-1], H.W[N_sites-1], Vout.state.A[N_sites-1], H.locBasis(N_sites-1), H.opBasis(N_sites-1), Ltmp);
		if (Ltmp.dim == 0)
		{
			Eold = 0;
		}
		else
		{
			assert(Ltmp.dim == 1 and
			Ltmp.block[0][0][0].rows() == 1 and
			Ltmp.block[0][0][0].cols() == 1 and
			"Result of contraction <ψ|H|ψ> in DmrgSolver::prepare is not a scalar!");
			Eold = isReal(Ltmp.block[0][0][0](0,0));
		}
	}
	else
	{
		Eold = std::real(avg(Vout.state,H,Vout.state));
	}
	Vout.energy = Eold;
	
	// initial cutoffs
	Vout.state.eps_svd    = DynParam.eps_svd(0);
	Vout.state.alpha_rsvd = DynParam.max_alpha_rsvd(0);
	
	if (CHOSEN_VERBOSITY>=2)
	{
		lout << PrepTimer.info("• initial state & sweep") << endl;
		size_t standard_precision = cout.precision();
		lout << std::setprecision(15) << "• initial energy        : E₀=" << Eold << std::setprecision(standard_precision) << endl;
		lout <<                          "• initial state         : " << Vout.state.info() << endl;
		lout <<                          "• initial fluctuation strength  : α_rsvd=";
		cout << termcolor::underline;
		lout << Vout.state.alpha_rsvd;
		cout << termcolor::reset;
		lout << endl;
		
		lout << "• initial singular value cutoff : ε_svd=";
		cout << termcolor::underline;
		lout << Vout.state.eps_svd;
		cout << termcolor::reset;
		lout << endl;
		
		int i_alpha_switchoff=0;
		for (int i=0; i<GlobParam.max_halfsweeps; ++i)
		{
			if (DynParam.max_alpha_rsvd(i) == 0.) {i_alpha_switchoff = i; break;}
		}
		lout << "• fluctuations turned off after ";
		cout << termcolor::underline;
		lout << i_alpha_switchoff;
		cout << termcolor::reset;
		lout << " half-sweeps" << endl;
		if (Vout.state.max_Nrich == -1)
		{
			lout <<            "• fluctuations use ";
			cout << termcolor::underline;
			lout << "all";
			cout << termcolor::reset;
			lout << " additional states" << endl;
		}
		else
		{
			lout << "• fluctuations use ";
			cout << termcolor::underline;
			lout << Vout.state.max_Nrich;
			cout << termcolor::reset;
			lout << " additional states" << endl;
		}
		lout << "• initial bond dim. increase by ";
		cout << termcolor::underline;
		lout << static_cast<int>((DynParam.Mincr_rel(0)-1.)*100.) << "%";
		cout << termcolor::reset;
		lout << " and at least by ";
		cout << termcolor::underline;
		lout << DynParam.Mincr_abs(0);
		cout << termcolor::reset;
		lout << " every ";
		cout << termcolor::underline;
		lout << DynParam.Mincr_per(0);
		cout << termcolor::reset;
		lout << " half-sweeps" << endl;
		
		lout << "• keep at least ";
		cout << termcolor::underline;
		lout << Vout.state.min_Nsv;
		cout << termcolor::reset;
		lout << " singular values per block" << endl;
		
		lout << "• make between ";
		cout << termcolor::underline;
		lout << GlobParam.min_halfsweeps;
		cout << termcolor::reset;
		lout << " and ";
		cout << termcolor::underline;
		lout << GlobParam.max_halfsweeps;
		cout << termcolor::reset;
		lout << " half-sweep iterations" << endl;
		lout << "• eigenvalue tolerance: ";
		cout << termcolor::underline;
		lout << GlobParam.tol_eigval;
		cout << termcolor::reset;
		lout << endl;
		
		lout << "• state tolerance: ";
		cout << termcolor::underline;
		lout << GlobParam.tol_state;
		cout << termcolor::reset;
		lout << " using ";
		cout << termcolor::underline;
		lout << GlobParam.CONVTEST;
		cout << termcolor::reset;
		lout << endl;
		
		lout << "• initial algorithm: ";
		cout << termcolor::underline;
		lout << DynParam.iteration(0);
		cout << termcolor::reset;
		lout << ", initial direction: ";
		cout << termcolor::underline;
		lout << GlobParam.INITDIR;
		cout << termcolor::reset;
		lout << endl;
		
		lout << "• calculate entropy on exit: " << boolalpha;
		cout << termcolor::underline;
		lout << GlobParam.CALC_S_ON_EXIT;
		cout << termcolor::reset << endl;
		lout << endl;
		
//		Vout.state.graph("init");
	}
	
	err_eigval = 1.;
	err_state  = 1.;
	
//	Vout.state.graph("init");
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
halfsweep (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, LANCZOS::EDGE::OPTION EDGE)
{
	Stopwatch<> HalfsweepTimer;
	
	// save state for reference
	Mps<Symmetry,Scalar> Vref;
	if (GlobParam.CONVTEST == DMRG::CONVTEST::NORM_TEST)
	{
		Vref = Vout.state;
	}
	
	size_t halfsweepRange = (SweepStat.N_halfsweeps==0)? N_sites : N_sites-1; // one extra step on 1st iteration
	
	double t_Lanczos = 0;
	double t_sweep = 0;
	double t_LR = 0;
	double t_overhead = 0;
	double t_err = 0;
	
	// If the next sweep is a 2-site or a 0-site sweep, move pivot back to edge. not sure if this is necessary for a 0-site sweep...
	if (DynParam.iteration(SweepStat.N_halfsweeps) == DMRG::ITERATION::TWO_SITE or
		DynParam.iteration(SweepStat.N_halfsweeps) == DMRG::ITERATION::ZERO_SITE)
	{
		sweep_to_edge(H,Vout,true); // build_LR = true
	}
	
	for (size_t j=1; j<=halfsweepRange; ++j)
	{
		turnaround(SweepStat.pivot, N_sites, SweepStat.CURRENT_DIRECTION);
		
		switch (DynParam.iteration(SweepStat.N_halfsweeps))
		{
		case DMRG::ITERATION::ZERO_SITE:
			iteration_zero(H, Vout, EDGE, t_Lanczos, t_sweep, t_LR, t_overhead); break;
		case DMRG::ITERATION::ONE_SITE:
			iteration_one (H, Vout, EDGE, t_Lanczos, t_sweep, t_LR, t_overhead); break;
		case DMRG::ITERATION::TWO_SITE:
			iteration_two (H, Vout, EDGE, t_Lanczos, t_sweep, t_LR, t_overhead); break;
		}
		++SweepStat.N_sweepsteps;
	}
	++SweepStat.N_halfsweeps;
	
	// calculate state error
	stringstream errorCalcInfo;
	err_eigval = abs(Eold-Vout.energy)/this->N_sites;
	if (GlobParam.CONVTEST == DMRG::CONVTEST::NORM_TEST and SweepStat.N_halfsweeps > GlobParam.min_halfsweeps)
	{
		Stopwatch<> ErrTimer;
		err_state = abs(1.-abs(dot(Vout.state,Vref)));
		t_err += ErrTimer.time();
	}
	else if (GlobParam.CONVTEST == DMRG::CONVTEST::VAR_HSQ and SweepStat.N_halfsweeps > GlobParam.min_halfsweeps)
	{
		Stopwatch<> HsqTimer;
		DMRG::DIRECTION::OPTION DIR = (SweepStat.N_halfsweeps%2==0) ? DMRG::DIRECTION::RIGHT : DMRG::DIRECTION::LEFT;
		
		double avgHsq = (H.check_power(2ul)==true)? isReal(avg(Vout.state,H,Vout.state,2,DIR)) : isReal(avg(Vout.state,H,H,Vout.state));
		err_state = abs(avgHsq-pow(Vout.energy,2))/this->N_sites;
		
		t_err += HsqTimer.time();
		if (CHOSEN_VERBOSITY>=2)
		{
			errorCalcInfo << HsqTimer.info("<H^2>") << endl;
		}
	}
	else if (GlobParam.CONVTEST == DMRG::CONVTEST::VAR_2SITE and SweepStat.N_halfsweeps > GlobParam.min_halfsweeps)
	{
		Stopwatch<> HsqTimer;
		double t_LR = 0;
		double t_Nsp = 0;
		double t_QR = 0;
		double t_GRALF = 0;
		
		sweep_to_edge(H,Vout,true);
		turnaround(SweepStat.pivot, N_sites, SweepStat.CURRENT_DIRECTION);
		
		err_state = 0.;
		
		vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > Nsaved(N_sites);
		DMRG::DIRECTION::OPTION DIR_N = SweepStat.CURRENT_DIRECTION;
		
		// one-site variance
		for (size_t l=0; l<N_sites; ++l)
		{
			// calculate the nullspace tensor F/G
			Stopwatch<> NspTimer;
			Vout.state.calc_N(SweepStat.CURRENT_DIRECTION, SweepStat.pivot, Nsaved[SweepStat.pivot]);
			t_Nsp += NspTimer.time();
			
			// contract Fig. 4 top from Hubig, Haegeman, Schollwöck (PRB 97, 2018), arXiv:1711.01104
			Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Err;
			Stopwatch<> GRALFtimer;
			contract_GRALF (Heff[SweepStat.pivot].L, Vout.state.A[SweepStat.pivot], H.W[SweepStat.pivot], 
			                Nsaved[SweepStat.pivot], Heff[SweepStat.pivot].R, 
			                H.locBasis(SweepStat.pivot), H.opBasis(SweepStat.pivot), Err, SweepStat.CURRENT_DIRECTION);
			t_GRALF += GRALFtimer.time();
			err_state += Err.squaredNorm().sum();
			
			// sweep to next site
			if ((l<N_sites-1 and DIR_N == DMRG::DIRECTION::RIGHT) or
			    (l>0         and DIR_N == DMRG::DIRECTION::LEFT))
			{
				Stopwatch<> QRtimer;
				Vout.state.sweepStep(SweepStat.CURRENT_DIRECTION, SweepStat.pivot, DMRG::BROOM::QR);
				t_QR += QRtimer.time();
				Stopwatch<> LRtimer;
				(SweepStat.CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(H,Vout,++SweepStat.pivot) : build_R(H,Vout,--SweepStat.pivot);
				(SweepStat.CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_PL(H,Vout,SweepStat.pivot)  : build_PR(H,Vout,SweepStat.pivot);
				t_LR += LRtimer.time();
			}
		}
		
		turnaround(SweepStat.pivot, N_sites, SweepStat.CURRENT_DIRECTION);
		
		// two-site variance
		for (size_t bond=0; bond<this->N_sites-1; ++bond)
		{
			size_t loc1 = (SweepStat.CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? SweepStat.pivot : SweepStat.pivot-1;
			size_t loc2 = (SweepStat.CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? SweepStat.pivot+1 : SweepStat.pivot;
			
			// calculate the nullspace tensor F/G with QR_NULL
			vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > N;
			if (DIR_N == DMRG::DIRECTION::LEFT)
			{
				N = Nsaved[loc2];
			}
			else
			{
				Stopwatch<> NspTimer;
				Vout.state.calc_N(DMRG::DIRECTION::LEFT, loc2, N);
				t_Nsp += NspTimer.time();
			}
			
			// pre-contract the right site
			Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Y;
			Stopwatch<> LRtimer1;
			contract_R(Heff[loc2].R, Vout.state.A[loc2], H.W[loc2], N, H.locBasis(loc2), H.opBasis(loc2), Y);
			t_LR += LRtimer1.time();
			
			// complete the contraction in Fig. 4 bottom from Hubig, Haegeman, Schollwöck (PRB 97, 2018), arXiv:1711.01104
			N.clear();
			if (DIR_N == DMRG::DIRECTION::RIGHT)
			{
				N = Nsaved[loc1];
			}
			else
			{
				Stopwatch<> NspTimer;
				Vout.state.calc_N(DMRG::DIRECTION::RIGHT, loc1, N);
				t_Nsp += NspTimer.time();
			}
			Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Err2;
			Stopwatch<> GRALFtimer;
			contract_GRALF (Heff[loc1].L, Vout.state.A[loc1], H.W[loc1], N, Y, 
			                H.locBasis(loc1), H.opBasis(loc1), Err2, DMRG::DIRECTION::RIGHT);
			t_GRALF += GRALFtimer.time();
			err_state += Err2.squaredNorm().sum();
			
			// sweep to next site
			Stopwatch<> QRtimer;
			Vout.state.sweepStep(SweepStat.CURRENT_DIRECTION, SweepStat.pivot, DMRG::BROOM::QR);
			t_QR += QRtimer.time();
			Stopwatch<> LRtimer2;
			(SweepStat.CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(H,Vout,++SweepStat.pivot) : build_R(H,Vout,--SweepStat.pivot);
			(SweepStat.CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_PL(H,Vout,SweepStat.pivot)  : build_PR(H,Vout,SweepStat.pivot);
			t_LR += LRtimer2.time();
		}
		
		// sweep back to the beginning (one site away from the edge)
		turnaround(SweepStat.pivot, N_sites, SweepStat.CURRENT_DIRECTION);
		Stopwatch<> QRtimer;
		Vout.state.sweepStep(SweepStat.CURRENT_DIRECTION, SweepStat.pivot, DMRG::BROOM::QR);
		t_QR += QRtimer.time();
		Stopwatch<> LRtimer;
		(SweepStat.CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(H,Vout,++SweepStat.pivot) : build_R(H,Vout,--SweepStat.pivot);
		(SweepStat.CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_PL(H,Vout,SweepStat.pivot)  : build_PR(H,Vout,SweepStat.pivot);
		t_LR += LRtimer.time();
		
		err_state /= this->N_sites;
		
		if (CHOSEN_VERBOSITY>=2)
		{
			double t_tot = HsqTimer.time();
			t_err += t_tot;
			errorCalcInfo << HsqTimer.info("2-site variance") 
						  << " ("
						  << "GRALF=" << round(t_GRALF/t_tot*100.,0) << "%" 
						  << ", LR=" << round(t_LR/t_tot*100.,0) << "%" 
						  << ", Nsp=" << round(t_Nsp/t_tot*100.,0) << "%" 
						  << ", QR=" << round(t_QR/t_tot*100.,0) << "%" 
						  << ")"
						  << endl;
		}
		
//		Mps<Symmetry,Scalar> HxPsi;
//		Mps<Symmetry,Scalar> Psi = Vout.state; Psi.sweep(0,DMRG::BROOM::QR);
//		HxV(H,Psi,HxPsi,DMRG::VERBOSITY::HALFSWEEPWISE);
//		Mps<Symmetry,Scalar> ExPsi = Vout.state;
//		ExPsi *= Vout.energy;
//		cout << HxPsi.validate("HxPsi") << ", " << HxPsi.info() << endl;
//		cout << ExPsi.validate("ExPsi") << ", " << ExPsi.info() << endl;
//		HxPsi -= ExPsi;
//		cout << HxPsi.validate("res") << ", " << HxPsi.info() << endl;
//		double err_exact_ = HxPsi.dot(HxPsi) / this->N_sites;
//		
//		Stopwatch<> HsqTimer_;
//		double PsixHxHxPsi = (H.check_power(2ul)==true)? isReal(avg(Vout.state,H,Vout.state,2)) : isReal(avg(Vout.state,H,H,Vout.state));
//		double PsixPsi = dot(Vout.state,Vout.state);
//		double PsixHxPsi = isReal(avg(Vout.state,H,Vout.state));
//		double err_exact = (PsixHxHxPsi + pow(Vout.energy,2)*PsixPsi - 2.*Vout.energy*PsixHxPsi) / this->N_sites;
//		cout << sqrt(PsixHxHxPsi) << ", " << Vout.energy << ", " << PsixHxPsi << ", " << PsixPsi << endl;
//		
//		cout << TCOLOR(RED) << "err_state=" << err_state << ", err_exact=" << err_exact 
//		<< ", " << err_exact_ 
//		<< ", diff=" << abs(err_state-err_exact)
//		<< ", ratio=" << err_state/err_exact
//		     << ", " << HsqTimer_.info("‖H|Ψ>-E|Ψ>‖") << TCOLOR(BLACK) << endl;
	}
	else if (GlobParam.CONVTEST == DMRG::CONVTEST::VAR_FULL and SweepStat.N_halfsweeps > GlobParam.min_halfsweeps)
	{
		Stopwatch<> HsqTimer;
		
		Mps<Symmetry,Scalar> HxPsi;
		Mps<Symmetry,Scalar> Psi = Vout.state;
		Psi.sweep(0,DMRG::BROOM::QR);
		HxV(H, Psi, HxPsi, false);
		
		Mps<Symmetry,Scalar> ExPsi = Vout.state;
		ExPsi *= Vout.energy;
		HxPsi -= ExPsi;
		
		err_state = std::real(HxPsi.dot(HxPsi)) / this->N_sites;
		double t_tot = HsqTimer.time();
		
		if (CHOSEN_VERBOSITY >= 2)
		{
			errorCalcInfo << HsqTimer.info("‖H|Ψ>-E|Ψ>‖/L") << endl;
		}
		
		t_err += t_tot;
	}
	
	Eold = Vout.energy;
	if (GlobParam.CONVTEST == DMRG::CONVTEST::NORM_TEST)
	{
		Vref = Vout.state;
	}
	
	// calculate stats
	Mmax = Vout.state.calc_Mmax();
	Dmax = Vout.state.calc_Dmax();
	Nqmax = Vout.state.calc_Nqmax();
	totalTruncWeight = Vout.state.truncWeight.sum();
	
	// print stuff
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << eigeninfo() << endl;
		size_t standard_precision = cout.precision();
		if (EDGE == LANCZOS::EDGE::GROUND)
		{
			lout << "E₀=" << setprecision(15) << Vout.energy << ", E₀/L=" << Vout.energy/N_phys << setprecision(standard_precision) << endl;
		}
		else
		{
			lout << "Eₘₐₓ=" << setprecision(15) << Vout.energy << ", Eₘₐₓ/L=" << Vout.energy/N_phys << setprecision(standard_precision) << endl;
		}
		lout << errorCalcInfo.str();
		lout << Vout.state.info() << endl;
		double t_halfsweep = HalfsweepTimer.time();
		lout << HalfsweepTimer.info("half-sweep") 
		     << " ("
		     << "Lanczos=" << round(t_Lanczos/t_halfsweep*100.,0) << "%"
		     << ", sweeps=" << round(t_sweep/t_halfsweep*100.,0) << "%"
		     << ", LR=" << round(t_LR/t_halfsweep*100.,0) << "%"
		     << ", overhead=" << round(t_overhead/t_halfsweep*100.,0) << "%"
		     << ", err=" << round(t_err/t_halfsweep*100.,0) << "%"
		     << ")"
		     << endl;
		
//		// check qmid:
//		for (size_t l=0; l<N_sites; ++l)
//		{
//			set<qarray<Nq> > qmids;
//			for (size_t q=0; q<Heff[l].L.dim; ++q)
//			{
//				qmids.insert(Heff[l].L.mid(q));
//			}
//			
//			cout << "l=" << l << endl;
//			for (const auto &qmid:qmids)
//			{
//				cout << qmid << " ";
//			}
//			cout << endl;
//		}
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
iteration_zero (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, LANCZOS::EDGE::OPTION EDGE,
                double &time_lanczos, double &time_sweep, double &time_LR, double &time_overhead)
{
	//*********************************************************LanczosStep******************************************************
	double Ei = Vout.energy;
	
	Stopwatch<> OheadTimer;
	Eigenstate<PivotVector<Symmetry,Scalar> > g;
	int old_pivot = SweepStat.pivot;
	(SweepStat.CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? Vout.state.rightSplitStep(SweepStat.pivot,g.state.data[0]):
		                                           Vout.state.leftSplitStep(SweepStat.pivot,g.state.data[0]);
	SweepStat.pivot = Vout.state.get_pivot();
	(SweepStat.CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(H,Vout,SweepStat.pivot) : build_R(H,Vout,SweepStat.pivot);
	
	PivotMatrix0<Symmetry,Scalar,Scalar> Heff0;
	(SweepStat.CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)?
	 Heff0 = PivotMatrix0<Symmetry,Scalar,Scalar>(Heff[old_pivot+1].L, Heff[old_pivot].R):
	 Heff0 = PivotMatrix0<Symmetry,Scalar,Scalar>(Heff[old_pivot].L, Heff[old_pivot-1].R);
	time_overhead += OheadTimer.time();
	
	Stopwatch<> LanczosTimer;
	LanczosSolver<PivotMatrix0<Symmetry,Scalar,Scalar>,PivotVector<Symmetry,Scalar>,Scalar> Lutz(LanczosParam.REORTHO);
	
	Lutz.set_efficiency(LANCZOS::EFFICIENCY::TIME);
	Lutz.set_dimK(min(LanczosParam.dimK, dim(g.state)));
	Lutz.edgeState(Heff0, g, EDGE, LanczosParam.tol_eigval, LanczosParam.tol_state, false);
	
	if (CHOSEN_VERBOSITY == DMRG::VERBOSITY::STEPWISE)
	{
		lout << "loc=" << SweepStat.pivot << "\t" << Lutz.info() << endl;
		lout << Vout.state.test_ortho() << ", " << g.energy << endl;
	}
	
	Vout.energy = g.energy;
	Vout.state.absorb(SweepStat.pivot, SweepStat.CURRENT_DIRECTION, g.state.data[0]);
	
	DeltaEopt = Ei-Vout.energy;
	time_lanczos += LanczosTimer.time();
	//**************************************************************************************************************************
	
	// Vout.state.min_Nsv = DynParam.min_Nsv(SweepStat.N_halfsweeps);
	// Vout.state.max_Nrich = DynParam.max_Nrich(SweepStat.N_halfsweeps);
	// Stopwatch<> SweepTimer;
	// Vout.state.sweepStep(SweepStat.CURRENT_DIRECTION, SweepStat.pivot, DMRG::BROOM::RICH_SVD, &Heff[SweepStat.pivot]);
	// time_sweep += SweepTimer.time();
		
	// Stopwatch<> LRtimer;
	// (SweepStat.CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(H,Vout,++SweepStat.pivot) : build_R(H,Vout,--SweepStat.pivot);
	// (SweepStat.CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_PL(H,Vout,SweepStat.pivot)  : build_PR(H,Vout,SweepStat.pivot);
	// time_LR += LRtimer.time();
	
	adapt_alpha_rsvd(H,Vout,EDGE);
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
iteration_one (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, LANCZOS::EDGE::OPTION EDGE,
               double &time_lanczos, double &time_sweep, double &time_LR, double &time_overhead)
{
	//*********************************************************LanczosStep******************************************************
	double Ei = Vout.energy;
	
	Stopwatch<> OheadTimer;
	Heff[SweepStat.pivot].W = H.W[SweepStat.pivot];
	
	// contract environment for excited states
	if (Psi0.size() > 0)
	{
		Heff[SweepStat.pivot].A0proj.resize(Psi0.size());
		for (int n=0; n<Psi0.size(); ++n)
		{
			Heff[SweepStat.pivot].A0proj[n].resize(Psi0[n].A[SweepStat.pivot].size());
		}
		
		for (int n=0; n<Psi0.size(); ++n)
		{
			PivotOverlap1 PO(Heff[SweepStat.pivot].PL[n], Heff[SweepStat.pivot].PR[n], Psi0[n].locBasis(SweepStat.pivot));
			PivotVector<Symmetry,Scalar> Ain = PivotVector<Symmetry,Scalar>(Psi0[n].A[SweepStat.pivot]);
			PivotVector<Symmetry,Scalar> Aout;
			LRxV(PO,Ain,Aout);
			Heff[SweepStat.pivot].A0proj[n] = Aout.data;
		}
	}
	
	precalc_blockStructure (Heff[SweepStat.pivot].L, 
	                        Vout.state.A[SweepStat.pivot], Heff[SweepStat.pivot].W, Vout.state.A[SweepStat.pivot], 
	                        Heff[SweepStat.pivot].R, 
	                        H.locBasis(SweepStat.pivot), H.opBasis(SweepStat.pivot), 
	                        Heff[SweepStat.pivot].qlhs, Heff[SweepStat.pivot].qrhs, Heff[SweepStat.pivot].factor_cgcs);
	Heff[SweepStat.pivot].qloc = H.locBasis(SweepStat.pivot);
	Heff[SweepStat.pivot].qOp = H.opBasis(SweepStat.pivot);
	
	Eigenstate<PivotVector<Symmetry,Scalar> > g;
	g.state = PivotVector<Symmetry,Scalar>(Vout.state.A[SweepStat.pivot]);
	g.state /= sqrt(dot(g.state,g.state));
//	cout << "dot PivotVector before=" << dot(g.state,g.state) << endl;
	time_overhead += OheadTimer.time();
	
	Stopwatch<> LanczosTimer;
	LanczosSolver<PivotMatrix1<Symmetry,Scalar,Scalar>,PivotVector<Symmetry,Scalar>,Scalar> Lutz(LanczosParam.REORTHO);
	
	Lutz.set_efficiency(LANCZOS::EFFICIENCY::TIME);
	Lutz.set_dimK(min(LanczosParam.dimK, dim(g.state)));
	Lutz.edgeState(Heff[SweepStat.pivot], g, EDGE, LanczosParam.tol_eigval, LanczosParam.tol_state, false);
	
	if (Psi0.size() > 0)
	{
		// STEPWISE: print always, HALFSWEEPWISE: print after every halfsweep
		if (CHOSEN_VERBOSITY == DMRG::VERBOSITY::STEPWISE or
		    CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE and 
				((loc1()==0 and SweepStat.CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT) or 
				 (loc2()==N_sites-1) and SweepStat.CURRENT_DIRECTION == DMRG::DIRECTION::LEFT)
		   )
		{
			for (int n=0; n<Psi0.size(); ++n)
			{
				Scalar overlap = 0;
				for (size_t s=0; s<Heff[SweepStat.pivot].A0proj[n].size(); ++s)
				{
					overlap += Heff[SweepStat.pivot].A0proj[n][s].adjoint().contract(g.state.data[s]).trace();
				}
				lout << "pivot=" << SweepStat.pivot << ", n=" << n << ", |overlap|=" << std::abs(overlap) << ", gap=" << g.energy-E0 << endl;
			}
		}
	}
	if (CHOSEN_VERBOSITY == DMRG::VERBOSITY::STEPWISE)
	{
		lout << "loc=" << SweepStat.pivot << "\t" << Lutz.info() << endl;
		lout << Vout.state.test_ortho() << ", " << g.energy << endl;
	}
	
	Vout.energy = g.energy;
	Vout.state.A[SweepStat.pivot] = g.state.data;
//	cout << "dot PivotVector after=" << dot(g.state,g.state) << endl;
//	cout << "loc=" << SweepStat.pivot << "\t" << Lutz.info() << endl;
	DeltaEopt = Ei-Vout.energy;
	// LanczosStep(H, Vout, EDGE);
	time_lanczos += LanczosTimer.time();
	//**************************************************************************************************************************
	
	Vout.state.min_Nsv = DynParam.min_Nsv(SweepStat.N_halfsweeps);
	Vout.state.max_Nrich = DynParam.max_Nrich(SweepStat.N_halfsweeps);
	Stopwatch<> SweepTimer;
	Vout.state.sweepStep(SweepStat.CURRENT_DIRECTION, SweepStat.pivot, DMRG::BROOM::RICH_SVD, &Heff[SweepStat.pivot]);
	time_sweep += SweepTimer.time();
	
	Stopwatch<> LRtimer;
	(SweepStat.CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(H,Vout,++SweepStat.pivot) : build_R(H,Vout,--SweepStat.pivot);
	(SweepStat.CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_PL(H,Vout,SweepStat.pivot)  : build_PR(H,Vout,SweepStat.pivot);
	time_LR += LRtimer.time();
	
	adapt_alpha_rsvd(H,Vout,EDGE);
	
	if (!Vout.state.Boundaries.IS_TRIVIAL())
	{
		double norm = std::real(Vout.state.dot(Vout.state));
		Vout.state /= sqrt(Vout.state.dot(Vout.state));
//		cout << Vout.state.test_ortho() << ", old_dot=" << norm << ", dot=" << Vout.state.dot(Vout.state) << endl;
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
iteration_two (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, LANCZOS::EDGE::OPTION EDGE,
               double &time_lanczos, double &time_sweep, double &time_LR, double &time_overhead)
{
	//*********************************************************LanczosStep******************************************************
	double Ei = Vout.energy;
	
	Stopwatch<> OheadTimer;
	Eigenstate<PivotVector<Symmetry,Scalar> > g;
	g.state = PivotVector<Symmetry,Scalar>(Vout.state.A[loc1()], Vout.state.locBasis(loc1()), 
	                                       Vout.state.A[loc2()], Vout.state.locBasis(loc2()),
	                                       Vout.state.QoutTop[loc1()], Vout.state.QoutBot[loc1()]);
	
	PivotMatrix2<Symmetry,Scalar,Scalar> Heff2(Heff[loc1()].L, Heff[loc2()].R, 
	                                           H.W[loc1()], H.W[loc2()], 
	                                           H.locBasis(loc1()), H.locBasis(loc2()), 
	                                           H.opBasis (loc1()), H.opBasis (loc2()));
	
	if (Psi0.size() > 0)
	{
		Heff2.Epenalty = Epenalty;
//		Heff2.PL.resize(Psi0.size());
//		Heff2.PR.resize(Psi0.size());
//		for (int n=0; n<Psi0.size(); ++n)
//		{
//			Heff2.PL[n] = Heff[loc1()].PL[n];
//			Heff2.PR[n] = Heff[loc2()].PR[n];
//		}
		
		// contract A0pair
		vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > A0pair(Psi0.size());
		for (int n=0; n<Psi0.size(); ++n)
		{
//			contract_AA(Psi0[n].A[loc1()], Psi0[n].locBasis(loc1()), 
//			            Psi0[n].A[loc2()], Psi0[n].locBasis(loc2()), 
//			            Psi0[n].QoutTop[loc1()], Psi0[n].QoutBot[loc1()], A0pair[n]);
			contract_AA2(Psi0[n].A[loc1()], Psi0[n].locBasis(loc1()), 
			             Psi0[n].A[loc2()], Psi0[n].locBasis(loc2()), 
			             A0pair[n]);
//			Qbasis<Symmetry> loc12; loc12.pullData(Psi0[n].locBasis(loc1()),true);
//			Qbasis<Symmetry> loc34; loc34.pullData(Psi0[n].locBasis(loc2()),true);
//			auto loc = loc12.combine(loc34);
//			extend_A(A0pair[n], loc.qloc());
		}
		
		Heff2.A0proj.resize(Psi0.size());
		for (int n=0; n<Psi0.size(); ++n) Heff2.A0proj[n].resize(A0pair[n].size());
		
		for (int n=0; n<Psi0.size(); ++n)
		{
			PivotOverlap2 PO(Heff[loc1()].PL[n], Heff[loc2()].PR[n], Psi0[n].locBasis(loc1()), Psi0[n].locBasis(loc2()));
			PivotVector<Symmetry,Scalar> Ain = PivotVector<Symmetry,Scalar>(A0pair[n]);
			PivotVector<Symmetry,Scalar> Aout;
			LRxV(PO,Ain,Aout);
//			for (int s=0; s<Aout.data.size(); ++s) Aout.data[s] = Aout.data[s].cleaned();
			Heff2.A0proj[n] = Aout.data;
		}
	}
	
	precalc_blockStructure (Heff2.L, g.state.data, Heff2.W12, Heff2.W34, g.state.data, Heff2.R, 
	                        H.locBasis(loc1()), H.locBasis(loc2()), H.opBasis(loc1()), H.opBasis(loc2()), 
	                        Heff2.qlhs, Heff2.qrhs, Heff2.factor_cgcs);
	time_overhead += OheadTimer.time();
	
	Stopwatch<> LanczosTimer;
	LanczosSolver<PivotMatrix2<Symmetry,Scalar,Scalar>,PivotVector<Symmetry,Scalar>,Scalar> Lutz(LanczosParam.REORTHO);
	
	Lutz.set_efficiency(LANCZOS::EFFICIENCY::TIME);
	Lutz.set_dimK(min(LanczosParam.dimK, dim(g.state)));
	Lutz.edgeState(Heff2, g, EDGE, LanczosParam.tol_eigval, LanczosParam.tol_state, false);
	time_lanczos += LanczosTimer.time();
	
	if (Psi0.size() > 0)
	{
		// STEPWISE: print always, HALFSWEEPWISE: print after every halfsweep
		if (CHOSEN_VERBOSITY == DMRG::VERBOSITY::STEPWISE or
		    CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE and 
				((loc1()==0 and SweepStat.CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT) or 
				 (loc2()==N_sites-1) and SweepStat.CURRENT_DIRECTION == DMRG::DIRECTION::LEFT)
		   )
		{
			for (int n=0; n<Psi0.size(); ++n)
			{
				Scalar overlap = 0;
				for (size_t s=0; s<Heff2.A0proj[n].size(); ++s)
				{
					overlap += Heff2.A0proj[n][s].adjoint().contract(g.state.data[s]).trace();
				}
				lout << "bond=" << loc1() << "-" << loc2() << ", n=" << n << ", |overlap|=" << std::abs(overlap) << ", gap=" << g.energy-E0 << endl;
			}
		}
	}
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
	{
		lout << "loc=" << SweepStat.pivot << "\t" << Lutz.info() << endl;
		lout << Vout.state.test_ortho() << ", " << g.energy << endl;
	}
	
	Vout.energy = g.energy;
	for (size_t s=0; s<g.state.data.size(); ++s)
	{
		g.state.data[s] = g.state.data[s].cleaned();
	}
	
	DeltaEopt = Ei-Vout.energy;
	//**************************************************************************************************************************
	
	Vout.state.min_Nsv = DynParam.min_Nsv(SweepStat.N_halfsweeps);
	Vout.state.max_Nrich = DynParam.max_Nrich(SweepStat.N_halfsweeps);
	Stopwatch<> SweepTimer;
	Vout.state.sweepStep2(SweepStat.CURRENT_DIRECTION, loc1(), g.state.data);
	time_sweep += SweepTimer.time();
	
	Stopwatch<> LRtimer;
	(SweepStat.CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(H,Vout,++SweepStat.pivot) : build_R(H,Vout,--SweepStat.pivot);
	(SweepStat.CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_PL(H,Vout,SweepStat.pivot)  : build_PR(H,Vout,SweepStat.pivot);
	time_LR += LRtimer.time();
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
sweep_to_edge (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, bool MAKE_ENVIRONMENT)
{
	assert(SweepStat.pivot == 0 or SweepStat.pivot==1 or SweepStat.pivot==N_sites-2 or SweepStat.pivot==N_sites-1);

	// assert(SweepStat.pivot==1 or SweepStat.pivot==N_sites-2);
	
	if (SweepStat.pivot==1)
	{
		Vout.state.sweepStep(DMRG::DIRECTION::LEFT, 1, DMRG::BROOM::QR);
		if (MAKE_ENVIRONMENT)
		{
			build_R(H,Vout,0);
			SweepStat.pivot = 0;
		}
	}
	else if (SweepStat.pivot==N_sites-2)
	{
		Vout.state.sweepStep(DMRG::DIRECTION::RIGHT, N_sites-2, DMRG::BROOM::QR);
		if (MAKE_ENVIRONMENT)
		{
			build_L(H,Vout,N_sites-1);
			SweepStat.pivot = N_sites-1;
		}
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
cleanup (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, LANCZOS::EDGE::OPTION EDGE)
{
	sweep_to_edge(H,Vout,false);
	
	if (GlobParam.CALC_S_ON_EXIT)
	{
		Vout.state.skim(DMRG::BROOM::SVD,NULL);
	}
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::ON_EXIT)
	{
		size_t standard_precision = cout.precision();
		string Eedge = (EDGE == LANCZOS::EDGE::GROUND)? "Emin" : "Emax";
		lout << termcolor::bold << Eedge << "=" << setprecision(15) << Vout.energy << ", "
			 << Eedge << "/L=" << Vout.energy/N_phys << setprecision(standard_precision) << termcolor::reset << endl;
		lout << eigeninfo() << endl;
		lout << Vout.state.info() << endl;
		
		if (GlobParam.CALC_S_ON_EXIT)
		{
			// size_t standard_precision = cout.precision();
			PlotParams p;
			p.label = "Entropy";
			if (Vout.state.entropy().rows() > 1)
			{
				TerminalPlot::plot(Vout.state.entropy(),p);
			}
			// lout << setprecision(2) << "S=" << Vout.state.entropy().transpose() << setprecision(standard_precision) << endl;
		}
	}
	
	if (N_sites>4 and GlobParam.CALC_S_ON_EXIT)
	{
		size_t l_start = N_sites%2 == 0 ? N_sites/2ul : (N_sites+1ul)/2ul;
		
		for (size_t l=l_start; l<=l_start+1; l++)
		{
			auto [qs,svs] = Vout.state.entanglementSpectrumLoc(l);
			ofstream Filer(make_string("sv_final_",l,".dat"));
			size_t index=0;
			for (size_t i=0; i<svs.size(); i++)
			{
				for (size_t deg=0; deg<Symmetry::degeneracy(qs[i]); deg++)
				{
					Filer << index << "\t"  << qs[i] << "\t" << svs[i] << endl;
					index++;
				}
			}
			Filer.close();
		}
	}
	
	if (Vout.state.calc_Nqavg() <= 1.5 and !Symmetry::IS_TRIVIAL and Vout.state.min_Nsv == 0)
	{
		Vout.state.min_Nsv = 1;
		lout << termcolor::blue << "DmrgSolver::cleanup notice: Setting min_Nsv=1 do deal with small Hilbert space!" << termcolor::reset << endl;
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
edgeState (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, qarray<Nq> Qtot_input, LANCZOS::EDGE::OPTION EDGE, bool USE_STATE)
{
	prepare(H, Vout, Qtot_input, USE_STATE);
	
	Stopwatch<> TotalTimer;
	
	bool MESSAGE_ALPHA = false;
	
	while (((err_eigval >= GlobParam.tol_eigval or err_state >= GlobParam.tol_state) and SweepStat.N_halfsweeps < GlobParam.max_halfsweeps) or
	       SweepStat.N_halfsweeps < GlobParam.min_halfsweeps)
	{
		// Set limits for alpha for the upcoming halfsweep
		min_alpha_rsvd = DynParam.min_alpha_rsvd(SweepStat.N_halfsweeps);
		max_alpha_rsvd = DynParam.max_alpha_rsvd(SweepStat.N_halfsweeps);
		
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE and 
		    max_alpha_rsvd == 0. and
		    MESSAGE_ALPHA == false)
		{
			lout << "α_rsvd is turned off now!" << endl << endl;
			MESSAGE_ALPHA = true;
		}
		
		// sweep
		halfsweep(H,Vout,EDGE); //SweepStat.N_halfsweeps gets incremented by 1!
//		Vout.state.graph(make_string("sweep",SweepStat.N_halfsweeps));
		
		// overwrite if alpha_rsvd was switched on
		if (DynParam.max_alpha_rsvd(SweepStat.N_halfsweeps-1) == 0. and DynParam.max_alpha_rsvd(SweepStat.N_halfsweeps) != 0.)
		{
			Vout.state.alpha_rsvd = DynParam.max_alpha_rsvd(SweepStat.N_halfsweeps);
			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
			{
				lout << endl << "α_rsvd switched on to: " << Vout.state.alpha_rsvd << " for halfsweep " << SweepStat.N_halfsweeps << endl << endl;
			}
		}
		
		size_t j = SweepStat.N_halfsweeps;
		
		DynParam.doSomething(j);
		
		// If truncated weight too large, increase upper limit per subspace by 10%, but at least by dimqlocAvg, overall never larger than Mlimit
		Vout.state.eps_svd = DynParam.eps_svd(j);
		if (j%DynParam.Mincr_per(j) == 0)
		//and (totalTruncWeight >= Vout.state.eps_svd or err_state > 10.*GlobParam.tol_state)
		{
			// increase by Mincr_abs for small Mmax and by Mincr_rel for large Mmax(e.g. add 10% of current Mmax)
			size_t max_Nsv_new = max(static_cast<size_t>(DynParam.Mincr_rel(j) * Vout.state.max_Nsv), 
			                                             Vout.state.max_Nsv + DynParam.Mincr_abs(j));
			// do not increase beyond Mlimit
			Vout.state.max_Nsv = min(max_Nsv_new, GlobParam.Mlimit);
		}
		
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			if (Vout.state.max_Nsv != Mmax_old)
			{
				lout << "Mmax=" << Mmax_old << "→" << Vout.state.max_Nsv << endl;
				Mmax_old = Vout.state.max_Nsv;
			}
			lout << endl;
		}
		
		#ifdef USE_HDF5_STORAGE
		if (GlobParam.savePeriod != 0 and j%GlobParam.savePeriod == 0)
		{
			lout << termcolor::green << "saving state to: " << GlobParam.saveName << termcolor::reset << endl;
			Vout.state.save(GlobParam.saveName,H.info());
		}
		#endif
	}
	
	#ifdef USE_HDF5_STORAGE
	if (GlobParam.savePeriod != 0)
	{
		string filename = make_string(GlobParam.saveName,"_fullMmax=",Vout.state.calc_fullMmax());
		lout << termcolor::green << "saving final state to: " << filename << termcolor::reset << endl;
		Vout.state.save(filename,H.info());
	}
	#endif
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::ON_EXIT)
	{
		lout << TotalTimer.info("total runtime") << endl;
	}
	cleanup(H,Vout,EDGE);
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
adapt_alpha_rsvd (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, LANCZOS::EDGE::OPTION EDGE)
{
	// adapt alpha
	if (Psi0.size() > 0)
	{
		Heff[SweepStat.pivot].A0proj.resize(Psi0.size());
		for (int n=0; n<Psi0.size(); ++n)
		{
			Heff[SweepStat.pivot].A0proj[n].resize(Psi0[n].A[SweepStat.pivot].size());
		}
		
		for (int n=0; n<Psi0.size(); ++n)
		for (int s=0; s<Psi0[n].A[SweepStat.pivot].size(); ++s)
		{
			PivotOverlap1 PO(Heff[SweepStat.pivot].PL[n], Heff[SweepStat.pivot].PR[n], Psi0[n].locBasis(SweepStat.pivot));
			PivotVector<Symmetry,Scalar> Ain = PivotVector<Symmetry,Scalar>(Psi0[n].A[SweepStat.pivot]);
			PivotVector<Symmetry,Scalar> Aout;
			LRxV(PO,Ain,Aout);
			Heff[SweepStat.pivot].A0proj[n] = Aout.data;
			
			// push new blocks of result as zero matrices into A
//			for (int q=0; q<Heff[SweepStat.pivot].A0proj[n][s].dim; ++q)
//			{
//				qarray2<Symmetry::Nq> cmp = {Heff[SweepStat.pivot].A0proj[n][s].in[q], Heff[SweepStat.pivot].A0proj[n][s].out[q]};
//				auto qA = Vout.state.A[SweepStat.pivot][s].dict.find(cmp);
//				if (qA == Vout.state.A[SweepStat.pivot][s].dict.end())
//				{
//					Matrix<Scalar,Dynamic,Dynamic> ZeroBlock(Heff[SweepStat.pivot].A0proj[n][s].block[q].rows(),
//					                                         Heff[SweepStat.pivot].A0proj[n][s].block[q].cols());
//					ZeroBlock.setZero();
//					Vout.state.A[SweepStat.pivot][s].push_back(cmp,ZeroBlock);
//				}
//			}
		}
	}
	
	PivotVector<Symmetry,Scalar> Vtmp1(Vout.state.A[SweepStat.pivot]);
	PivotVector<Symmetry,Scalar> Vtmp2;
	Heff[SweepStat.pivot].W = H.W[SweepStat.pivot];
	precalc_blockStructure (Heff[SweepStat.pivot].L, Vout.state.A[SweepStat.pivot], 
	                        Heff[SweepStat.pivot].W, Vout.state.A[SweepStat.pivot], Heff[SweepStat.pivot].R, 
	                        H.locBasis(SweepStat.pivot), H.opBasis(SweepStat.pivot), Heff[SweepStat.pivot].qlhs, Heff[SweepStat.pivot].qrhs,
	                        Heff[SweepStat.pivot].factor_cgcs);
	Heff[SweepStat.pivot].qloc = H.locBasis(SweepStat.pivot);
	Heff[SweepStat.pivot].qOp  = H.opBasis(SweepStat.pivot);
	HxV(Heff[SweepStat.pivot], Vtmp1, Vtmp2);
	
	double DeltaEtrunc = std::real(dot(Vtmp1,Vtmp2))-Vout.energy;
	
//	if (DeltaEtrunc < 0.3*DeltaEopt) {Vout.state.alpha_rsvd *= sqrt(10.);}
//	else                             {Vout.state.alpha_rsvd /= sqrt(10.);}
//	Vout.state.alpha_rsvd = min(Vout.state.alpha_rsvd, max_alpha_rsvd);
	
	double f;
	double epsilon = 1e-9;
	if (abs(DeltaEopt) < epsilon or abs(DeltaEtrunc) < epsilon)
	{
		if (abs(DeltaEtrunc) > epsilon) {f = 0.9;}
		else                            {f = 1.001;}
	}
	else
	{
		double r = abs(DeltaEtrunc) / abs(DeltaEopt);
		if ((DeltaEtrunc < 0. and EDGE == LANCZOS::EDGE::GROUND) or
		    (DeltaEtrunc > 0. and EDGE == LANCZOS::EDGE::ROOF)
		   )
		{
			f = 2.*(r+1.);
		}
		else if (r < 0.05)    {f = 1.2-r;}
		else if (r > 0.3)     {f = 1./(r+0.75);}
	}
	f = max(0.1,min(2.,f)); // limit between [0.1,2]
	Vout.state.alpha_rsvd *= f;
	// limit between [min_alpha_rsvd,max_alpha_rsvd]:
	// double alpha_min = min(DynParam.min_alpha_rsvd(SweepStat.N_halfsweeps), 
	//                        DynParam.max_alpha_rsvd(SweepStat.N_halfsweeps)); // for the accidental case alpha_min > alpha_max
	// Vout.state.alpha_rsvd = max(alpha_min, min(DynParam.max_alpha_rsvd(SweepStat.N_halfsweeps), Vout.state.alpha_rsvd));
	double alpha_min = min(min_alpha_rsvd, max_alpha_rsvd); // for the accidental case alpha_min > alpha_max
	Vout.state.alpha_rsvd = max(alpha_min, min(max_alpha_rsvd, Vout.state.alpha_rsvd));
	
//	cout << "ΔEopt=" << DeltaEopt << ", ΔEtrunc=" << DeltaEtrunc << ", f=" << f << ", alpha=" << Vout.state.alpha_rsvd << endl;
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE)
	{
		lout << "ΔEopt=" << DeltaEopt << ", ΔEtrunc=" << DeltaEtrunc << ", α=" << Vout.state.alpha_rsvd << endl;
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
LanczosStep (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, LANCZOS::EDGE::OPTION EDGE)
{
	double Ei = Vout.energy;
	
//	if (Heff[SweepStat.pivot].qloc.size() == 0)
	{
		Heff[SweepStat.pivot].W = H.W[SweepStat.pivot];
		precalc_blockStructure (Heff[SweepStat.pivot].L, Vout.state.A[SweepStat.pivot], 
		                        Heff[SweepStat.pivot].W, Vout.state.A[SweepStat.pivot], Heff[SweepStat.pivot].R, 
		                        H.locBasis(SweepStat.pivot), H.opBasis(SweepStat.pivot), Heff[SweepStat.pivot].qlhs, Heff[SweepStat.pivot].qrhs,
		                        Heff[SweepStat.pivot].factor_cgcs);
		Heff[SweepStat.pivot].qloc = H.locBasis(SweepStat.pivot);
		Heff[SweepStat.pivot].qOp = H.opBasis(SweepStat.pivot);
	}
	
	Eigenstate<PivotVector<Symmetry,Scalar> > g;
	g.state = PivotVector<Symmetry,Scalar>(Vout.state.A[SweepStat.pivot]);
	LanczosSolver<PivotMatrix1<Symmetry,Scalar,Scalar>,PivotVector<Symmetry,Scalar>,Scalar> Lutz(LanczosParam.REORTHO);
	
	Lutz.set_efficiency(LANCZOS::EFFICIENCY::TIME);
	Lutz.set_dimK(min(LanczosParam.dimK, dim(g.state)));
	Lutz.edgeState(Heff[SweepStat.pivot],g, EDGE, LanczosParam.tol_eigval, LanczosParam.tol_state, false);
	
	if (CHOSEN_VERBOSITY == DMRG::VERBOSITY::STEPWISE)
	{
		lout << "loc=" << SweepStat.pivot << "\t" << Lutz.info() << endl;
		lout << Vout.state.test_ortho() << ", " << g.energy << endl;
	}
	
	Vout.energy = g.energy;
	Vout.state.A[SweepStat.pivot] = g.state.data;
	DeltaEopt = Ei-Vout.energy;
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
inline void DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
build_L (const MpHamiltonian &H, const Eigenstate<Mps<Symmetry,Scalar> > &Vout, size_t loc)
{
	contract_L(Heff[loc-1].L, Vout.state.A[loc-1], H.W[loc-1], Vout.state.A[loc-1], H.locBasis(loc-1), H.opBasis(loc-1), Heff[loc].L);
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
inline void DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
build_R (const MpHamiltonian &H, const Eigenstate<Mps<Symmetry,Scalar> > &Vout, size_t loc)
{
	contract_R(Heff[loc+1].R, Vout.state.A[loc+1], H.W[loc+1], Vout.state.A[loc+1], H.locBasis(loc+1), H.opBasis(loc+1), Heff[loc].R);
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
inline void DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
build_PL (const MpHamiltonian &H, const Eigenstate<Mps<Symmetry,Scalar> > &Vout, size_t loc)
{
	for (size_t n=0; n<Psi0.size(); ++n)
	{
//		cout << "BUILDING LEFT ENVIRONMENT AT loc=" << loc << endl;
		contract_L(Heff[loc-1].PL[n], Vout.state.A[loc-1], Psi0[n].A[loc-1], H.locBasis(loc-1), Heff[loc].PL[n], false, true);
		Heff[loc].PL[n] = Heff[loc].PL[n].cleaned();
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
inline void DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
build_PR (const MpHamiltonian &H, const Eigenstate<Mps<Symmetry,Scalar> > &Vout, size_t loc)
{
	for (size_t n=0; n<Psi0.size(); ++n)
	{
//		cout << "BUILDING RIGHT ENVIRONMENT AT loc=" << loc << endl;
		contract_R(Heff[loc+1].PR[n], Vout.state.A[loc+1], Psi0[n].A[loc+1], H.locBasis(loc+1), Heff[loc].PR[n], false, true);
		Heff[loc].PR[n] = Heff[loc].PR[n].cleaned();
	}
}

#endif
