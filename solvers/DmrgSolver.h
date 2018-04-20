#ifndef STRAWBERRY_DMRGSOLVER_WITH_Q
#define STRAWBERRY_DMRGSOLVER_WITH_Q

#include "Mpo.h"
#include "Mps.h"
#include "pivot/DmrgPivotMatrix1.h"
#include "tensors/DmrgContractions.h"
#include "DmrgLinearAlgebra.h" // for avg()
#include "LanczosSolver.h" // from ALGS
#include "Stopwatch.h" // from TOOLS
#ifdef USE_HDF5_STORAGE
	#include <HDF5Interface.h> // from TOOLS
#endif
#include "solvers/MpsCompressor.h"

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
	double memory   (MEMUNIT memunit=GB) const;
	
	void edgeState (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, qarray<Nq> Qtot_input, 
	                LANCZOS::EDGE::OPTION EDGE = LANCZOS::EDGE::GROUND,
	                DMRG::CONVTEST::OPTION TEST = DMRG::CONVTEST::VAR_2SITE,
	                double tol_eigval_input=1e-7, double tol_state_input=1e-6, 
	                size_t Dinit=4, size_t Dlimit=500, int Qinit=50,
	                size_t max_halfsweeps=50, size_t min_halfsweeps=6, 
                    double max_alpha_rsvd_input=1e2, double eps_svd_input=1e-7, 
	                size_t savePeriod=0);
	
	inline void set_verbosity (DMRG::VERBOSITY::OPTION VERBOSITY) {CHOSEN_VERBOSITY = VERBOSITY;};
	
	void prepare (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, qarray<Nq> Qtot_input, bool useState=false, size_t Dinit=5, int Qinit=50,
	              double max_alpha_rsvd_input=1., double eps_svd_input=1e-7);
	void halfsweep (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, 
	                LANCZOS::EDGE::OPTION EDGE = LANCZOS::EDGE::GROUND, 
	                DMRG::CONVTEST::OPTION TEST = DMRG::CONVTEST::VAR_2SITE);
	void cleanup (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, 
	              LANCZOS::EDGE::OPTION EDGE = LANCZOS::EDGE::GROUND);
	
	/**Returns the current error of the eigenvalue while the sweep process.*/
	inline double get_errEigval() const {return err_eigval;};
	
	/**Returns the current error of the state while the sweep process.*/
	inline double get_errState() const {return err_state;};
	
	/**Returns the current pivot site of the sweep process.*/
	inline double get_pivot() const {return stat.pivot;};
	
	/**Returns the current direction of the sweep process.*/
	inline double get_direction() const {return stat.CURRENT_DIRECTION;};
	
	void push_back(const Mps<Symmetry,Scalar> &Psi0_input)
	{
		Psi0.push_back(Psi0_input);
	};
	
	#ifdef USE_HDF5_STORAGE
	/**Save the current SweepStatus to <filename>.h5.*/
	void save(string filename) const;
	/**Load the a SweepStatus from <filename>.h5.*/
	void load(string filename);
	#endif
	
private:
	
	size_t N_sites, N_phys;
	size_t Dmax, Mmax, Nqmax;
	double tol_eigval, tol_state;
	double totalTruncWeight;
	size_t Dmax_old;
	double err_eigval, err_state;
	
	vector<PivotMatrix1<Symmetry,Scalar,Scalar> > Heff; // Scalar = MpoScalar for ground state
	
	double Eold;
	
	double DeltaEopt;
	double max_alpha_rsvd;
	
	struct SweepStatus
	{
		int pivot=-1;
		DMRG::DIRECTION::OPTION CURRENT_DIRECTION;
		size_t N_sweepsteps, N_halfsweeps;
	};
	SweepStatus stat;
	
	void LanczosStep (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, LANCZOS::EDGE::OPTION EDGE);
	void sweepStep (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout);
	void sweep_to_edge (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, bool MAKE_ENVIRONMENT);
	
	/**Constructs the left transfer matrix at chain site \p loc (left environment of \p loc).*/
	void build_L (const MpHamiltonian &H, const Eigenstate<Mps<Symmetry,Scalar> > &Vout, size_t loc);
	
	/**Constructs the right transfer matrix at chain site \p loc (right environment of \p loc).*/
	void build_R (const MpHamiltonian &H, const Eigenstate<Mps<Symmetry,Scalar> > &Vout, size_t loc);
	
	void build_PL (const MpHamiltonian &H, const Eigenstate<Mps<Symmetry,Scalar> > &Vout, size_t loc);
	void build_PR (const MpHamiltonian &H, const Eigenstate<Mps<Symmetry,Scalar> > &Vout, size_t loc);
	
	/**Projected-out states to find the edge of the spectrum.*/
	vector<Mps<Symmetry,Scalar> > Psi0;
	
	/**Energy penalty for projected-out states.*/
	double Epenalty = 10.;
	
	DMRG::VERBOSITY::OPTION CHOSEN_VERBOSITY;
	DMRG::CONVTEST::OPTION CHOSEN_CONVTEST;
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
	ss << "half-sweeps=";
	if ((stat.N_sweepsteps-1)/(N_sites-1)>0)
	{
		ss << (stat.N_sweepsteps-1)/(N_sites-1);
		if ((stat.N_sweepsteps-1)%(N_sites-1)!=0) {ss << "+";}
	}
	if ((stat.N_sweepsteps-1)%(N_sites-1)!=0) {ss << (stat.N_sweepsteps-1)%(N_sites-1) << "/" << (N_sites-1);}
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
	target.save_scalar(stat.pivot,"pivot");
	target.save_scalar(stat.N_halfsweeps,"N_halfsweeps");
	target.save_scalar(stat.N_sweepsteps,"N_sweepsteps");
	target.save_scalar(static_cast<int>(stat.CURRENT_DIRECTION),"direction");
	target.save_scalar(Dmax,"D");
	target.save_scalar(err_eigval,"errorE");
	target.save_scalar(err_state,"errorS");
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
load (string filename)
{
	filename+=".h5";
	HDF5Interface source(filename, READ);
	source.load_scalar(stat.pivot,"pivot");
	source.load_scalar(stat.N_halfsweeps,"N_halfsweeps");
	source.load_scalar(stat.N_sweepsteps,"N_sweepsteps");
	int DIR_IN_INT;
	source.load_scalar(DIR_IN_INT,"direction");
	stat.CURRENT_DIRECTION = static_cast<DMRG::DIRECTION::OPTION>(DIR_IN_INT);
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
			res += calc_memory(Heff[l].W[s1][s2][k],memunit);
		}
	}
	return res;
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
prepare (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, qarray<Nq> Qtot_input, bool USE_STATE, size_t Dinit, int Qinit,
         double max_alpha_rsvd_input, double eps_svd_input)
{
	N_sites = H.length();
	N_phys  = H.volume();
	
	max_alpha_rsvd = max_alpha_rsvd_input;
	
	Stopwatch<> PrepTimer;
	if (!USE_STATE)
	{
		// resize Vout
		Vout.state = Mps<Symmetry,Scalar>(H, Dinit, Qtot_input, Qinit);
		Vout.state.N_sv = Dinit;
		Vout.state.setRandom();
	}
	Dmax_old = Dinit;
	
	// set edges
	Heff.clear();
	Heff.resize(N_sites);
	Heff[0].L.setVacuum();
	Heff[N_sites-1].R.setTarget(qarray3<Nq>{Qtot_input, Qtot_input, Symmetry::qvacuum()});

	//if the SweepStatus is default initialized (pivot==-1), one initial sweep from right-to-left and N_halfsweeps = N_sweepsteps = 0,
	//otherwise prepare for continuing at the given SweepStatus.
	if (stat.pivot == -1)
	{
		stat.N_sweepsteps = stat.N_halfsweeps = 0;
		for (size_t l=N_sites-1; l>0; --l)
		{
			Vout.state.sweepStep(DMRG::DIRECTION::LEFT, l, DMRG::BROOM::QR); // SVD correct here?
			build_R(H,Vout,l-1);
		}
		Vout.state.sweepStep(DMRG::DIRECTION::LEFT, 0, DMRG::BROOM::QR); // removes large numbers from first matrix
		stat.CURRENT_DIRECTION = DMRG::DIRECTION::RIGHT;
		stat.pivot = 0;
	}
	else
	{
		if (stat.CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)
		{
			for (size_t l=N_sites-1; l>0; --l)
			{
				Vout.state.sweepStep(DMRG::DIRECTION::LEFT, l, DMRG::BROOM::QR);
				build_R(H,Vout,l-1);
			}
			Vout.state.sweepStep(DMRG::DIRECTION::LEFT, 0, DMRG::BROOM::QR); // removes large numbers from first matrix
			for (size_t l=0; l<stat.pivot; ++l)
			{
				Vout.state.sweepStep(DMRG::DIRECTION::RIGHT, l, DMRG::BROOM::QR);
				build_L(H,Vout,l+1);
			}
		}
		else if (stat.CURRENT_DIRECTION == DMRG::DIRECTION::LEFT)
		{
			for (size_t l=0; l<N_sites-1; ++l)
			{
				Vout.state.sweepStep(DMRG::DIRECTION::RIGHT, l, DMRG::BROOM::QR);
				build_L(H,Vout,l+1);
			}
			Vout.state.sweepStep(DMRG::DIRECTION::RIGHT, 0, DMRG::BROOM::QR); // removes large numbers from first matrix
			for (size_t l=N_sites-1; l>stat.pivot; --l)
			{
				Vout.state.sweepStep(DMRG::DIRECTION::LEFT, l, DMRG::BROOM::QR);
				build_R(H,Vout,l-1);
			}
		}
	}
	
//	// initial sweep, left-to-right:
//	for (size_t l=0; l<N_sites-1; ++l)
//	{
//		cout << "l=" << l << endl;
//		Vout.state.sweepStep(DMRG::DIRECTION::RIGHT, l, DMRG::BROOM::QR);
//		build_L(H,Vout,l+1);
//	}
//	Vout.state.sweepStep(DMRG::DIRECTION::RIGHT, 0, DMRG::BROOM::QR); // removes large numbers from first matrix
//	stat.CURRENT_DIRECTION = DMRG::DIRECTION::LEFT;
//	stat.pivot = N_sites-1;
	
	// resize environments for projected-out states
	if (Psi0.size() > 0)
	{
		for (size_t l=0; l<N_sites; ++l)
		{
			Heff[l].Epenalty = Epenalty;
			Heff[l].PL.resize(Psi0.size());
			Heff[l].PR.resize(Psi0.size());
			Heff[l].A0.resize(Psi0.size());
			for (size_t n=0; n<Psi0.size(); ++n)
			{
				Heff[l].A0[n] = Psi0[n].A[l];
			}
		}
	}
	
	// build environments for projected-out states
	for (size_t n=0; n<Psi0.size(); ++n)
	{
		Heff[0].PL[n].setVacuum();
		for (size_t l=1; l<N_sites; ++l)
		{
			Heff[l].PL[n] = Vout.state.A[l-1][0].adjoint() * Heff[l-1].PL[n] * Psi0[n].A[l-1][0];
			for (size_t s=1; s<Vout.state.locBasis(l-1).size(); ++s)
			{
				Heff[l].PL[n] += Vout.state.A[l-1][s].adjoint() * Heff[l-1].PL[n] * Psi0[n].A[l-1][s];
			}
		}
		
		Heff[N_sites-1].PR[n].setTarget(Vout.state.Qtot);
		for (int l=N_sites-2; l>=0; --l)
		{
			Heff[l].PR[n] = Psi0[n].A[l+1][0] * Heff[l+1].PR[n] * Vout.state.A[l+1][0].adjoint();
			for (size_t s=1; s<Vout.state.locBasis(l+1).size(); ++s)
			{
				Heff[l].PR[n] += Psi0[n].A[l+1][s] * Heff[l+1].PR[n] * Vout.state.A[l+1][s].adjoint();
			}
		}
	}
	
	if (CHOSEN_VERBOSITY>=2) {lout << PrepTimer.info("initial state & sweep") << endl;}
	// initial energy
	if (stat.pivot == 0)
	{
//		Vout.state.graph("init");
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
	else
	{
		Eold = avg(Vout.state,H,Vout.state);
	}
	Vout.energy = Eold;
	if (CHOSEN_VERBOSITY>=2)
	{
		lout << "initial energy: E₀=" << Eold << endl;
		lout << Vout.state.info() << endl;
		lout << endl;
		Vout.state.graph("init");
	}
	
	// initial cutoffs
	Vout.state.eps_svd = eps_svd_input;
	Vout.state.alpha_rsvd = max_alpha_rsvd_input;
	
	err_eigval = 1.;
	err_state  = 1.;
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
halfsweep (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, LANCZOS::EDGE::OPTION EDGE, DMRG::CONVTEST::OPTION TEST)
{
	Stopwatch<> HalfsweepTimer;
	
	// save state for reference
	Mps<Symmetry,Scalar> Vref;
	if (TEST == DMRG::CONVTEST::NORM_TEST)
	{
		Vref = Vout.state;
	}
	
	size_t halfsweepRange = (stat.N_halfsweeps==0)? N_sites : N_sites-1; // one extra step on 1st iteration
	double t_Lanczos = 0;
	double t_sweep = 0;
	double t_LR = 0;
	double t_err=0;
	for (size_t j=1; j<=halfsweepRange; ++j)
	{
		turnaround(stat.pivot, N_sites, stat.CURRENT_DIRECTION);
		
		Stopwatch<> LanczosTimer;
		LanczosStep(H, Vout, EDGE);
		t_Lanczos += LanczosTimer.time();
		
		Stopwatch<> SweepTimer;
		Vout.state.sweepStep(stat.CURRENT_DIRECTION, stat.pivot, DMRG::BROOM::RICH_SVD, &Heff[stat.pivot]);
		t_sweep += SweepTimer.time();
		
		Stopwatch<> LRtimer;
		sweepStep(H,Vout);
		t_LR += LRtimer.time();
		
		++stat.N_sweepsteps;
	}
	++stat.N_halfsweeps;
	
	// calculate state error
	err_eigval = abs(Eold-Vout.energy)/this->N_sites;
	if (TEST == DMRG::CONVTEST::NORM_TEST)
	{
		err_state = abs(1.-abs(dot(Vout.state,Vref)));
	}
	else if (TEST == DMRG::CONVTEST::VAR_HSQ)
	{
		Stopwatch<> HsqTimer;
		DMRG::DIRECTION::OPTION DIR = (stat.N_halfsweeps%2==0) ? DMRG::DIRECTION::RIGHT : DMRG::DIRECTION::LEFT;
		
		double avgHsq = (H.check_SQUARE()==true)? isReal(avg(Vout.state,H,Vout.state,true,DIR)) : isReal(avg(Vout.state,H,H,Vout.state));
		err_state = abs(avgHsq-pow(Vout.energy,2))/this->N_sites;
		
		if (CHOSEN_VERBOSITY>=2)
		{
			lout << HsqTimer.info("<H^2>") << endl;
		}
	}
	else if (TEST == DMRG::CONVTEST::VAR_2SITE)
	{
		Stopwatch<> HsqTimer;
		double t_LR=0;
		double t_N=0;
		double t_QR=0;
		double t_GRALF=0;
		
		sweep_to_edge(H,Vout,true);
		err_state = 0.;
		
		vector<vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > > Nsaved(this->N_sites);
		DMRG::DIRECTION::OPTION DIR_N = stat.CURRENT_DIRECTION;
		
		// one-site variance
		for (size_t l=0; l<this->N_sites; ++l)
		{
			// calculate the nullspace tensor F/G with QR_NULL
//			vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > N;
//			Vout.state.sweepStep(stat.CURRENT_DIRECTION, stat.pivot, DMRG::BROOM::QR_NULL, NULL, false, &N);
			Stopwatch<> Ntimer;
			Vout.state.calc_N(stat.CURRENT_DIRECTION, stat.pivot, Nsaved[stat.pivot]);
			t_N += Ntimer.time();
			
			// contract Fig. 4 top from Hubig, Haegeman, Schollwöck (PRB 97, 2018), arXiv:1711.01104
			Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Err;
			Stopwatch<> GRALFtimer;
			contract_GRALF (Heff[stat.pivot].L, Vout.state.A[stat.pivot], Heff[stat.pivot].W, Nsaved[stat.pivot], Heff[stat.pivot].R, 
			                H.locBasis(stat.pivot), H.opBasis(stat.pivot), Err, stat.CURRENT_DIRECTION);
			t_GRALF += GRALFtimer.time();
			err_state += Err.squaredNorm().sum();
			
			// sweep to next site
			if (l<N_sites-1)
			{
				Stopwatch<> QRtimer;
				Vout.state.sweepStep(stat.CURRENT_DIRECTION, stat.pivot, DMRG::BROOM::QR);
				t_QR += QRtimer.time();
				Stopwatch<> LRtimer;
				(stat.CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(H,Vout,++stat.pivot) : build_R(H,Vout,--stat.pivot);
				t_LR += LRtimer.time();
			}
		}
		
		turnaround(stat.pivot, N_sites, stat.CURRENT_DIRECTION);
		
		// two-site variance
		for (size_t bond=0; bond<this->N_sites-1; ++bond)
		{
			size_t loc1 = (stat.CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? stat.pivot : stat.pivot-1;
			size_t loc2 = (stat.CURRENT_DIRECTION==DMRG::DIRECTION::RIGHT)? stat.pivot+1 : stat.pivot;
			
			// calculate the nullspace tensor F/G with QR_NULL
			vector<Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > > N;
			if (DIR_N == DMRG::DIRECTION::LEFT)
			{
				N = Nsaved[loc2];
			}
			else
			{
				Stopwatch<> Ntimer;
				Vout.state.calc_N(DMRG::DIRECTION::LEFT, loc2, N);
				t_N += Ntimer.time();
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
				Stopwatch<> Ntimer;
				Vout.state.calc_N(DMRG::DIRECTION::RIGHT, loc1, N);
				t_N += Ntimer.time();
			}
			Biped<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Err2;
			Stopwatch<> GRALFtimer;
			contract_GRALF (Heff[loc1].L, Vout.state.A[loc1], Heff[loc1].W, N, Y, 
			                H.locBasis(loc1), H.opBasis(loc1), Err2, DMRG::DIRECTION::RIGHT);
			t_GRALF += GRALFtimer.time();
			err_state += Err2.squaredNorm().sum();
			
			// sweep to next site
			Stopwatch<> QRtimer;
			Vout.state.sweepStep(stat.CURRENT_DIRECTION, stat.pivot, DMRG::BROOM::QR);
			t_QR += QRtimer.time();
			Stopwatch<> LRtimer2;
			(stat.CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(H,Vout,++stat.pivot) : build_R(H,Vout,--stat.pivot);
			t_LR += LRtimer2.time();
		}
		
		// sweep back to the beginning (one site away from the edge)
		turnaround(stat.pivot, N_sites, stat.CURRENT_DIRECTION);
		Stopwatch<> QRtimer;
		Vout.state.sweepStep(stat.CURRENT_DIRECTION, stat.pivot, DMRG::BROOM::QR);
		t_QR += QRtimer.time();
		Stopwatch<> LRtimer;
		(stat.CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(H,Vout,++stat.pivot) : build_R(H,Vout,--stat.pivot);
		t_LR += LRtimer.time();
		
		err_state /= this->N_sites;
		
		if (CHOSEN_VERBOSITY>=2)
		{
			double t_tot = HsqTimer.time();
			t_err += t_tot;
			lout << HsqTimer.info("2-site variance") 
			     << " ("
			     << "GRALF=" << round(t_GRALF/t_tot*100.,0) << "%" 
			     << ", LR=" << round(t_LR/t_tot*100.,0) << "%" 
			     << ", N=" << round(t_N/t_tot*100.,0) << "%" 
			     << ", QR=" << round(t_QR/t_tot*100.,0) << "%" 
			     << ")"
			     << endl;
		}
		
//		Mps<Symmetry,Scalar> HxPsi;
//		Mps<Symmetry,Scalar> Psi = Vout.state; Psi.sweep(0,DMRG::BROOM::QR);
////		if constexpr (Symmetry::NON_ABELIAN) {HxV(H,Psi,HxPsi,DMRG::VERBOSITY::HALFSWEEPWISE);}
////		else {HxPsi.eps_svd = 0.; OxV(H,Psi,HxPsi);}
//		HxV(H,Psi,HxPsi,DMRG::VERBOSITY::HALFSWEEPWISE);
//		
//		Mps<Symmetry,Scalar> ExPsi = Vout.state;
//		ExPsi *= Vout.energy;
//		HxPsi -= ExPsi;
//		
//		double err_exact = HxPsi.dot(HxPsi) / this->N_sites;
//		
//		cout << "err_state=" << err_state << ", err_exact=" << err_exact << ", diff=" << abs(err_state-err_exact) << endl;
	}
	else if (TEST == DMRG::CONVTEST::VAR_FULL)
	{
		Stopwatch<> HsqTimer;
		Mps<Symmetry,Scalar> HxPsi;
		Mps<Symmetry,Scalar> Psi = Vout.state; Psi.sweep(0,DMRG::BROOM::QR);
		if constexpr (Symmetry::NON_ABELIAN) {HxV(H,Psi,HxPsi,DMRG::VERBOSITY::HALFSWEEPWISE);}
		else                                 {HxPsi.eps_svd = 0.; OxV(H,Psi,HxPsi);}
		
		Mps<Symmetry,Scalar> ExPsi = Vout.state;
		ExPsi *= Vout.energy;
		HxPsi -= ExPsi;
		
		double err_state = HxPsi.dot(HxPsi) / this->N_sites;
		if (CHOSEN_VERBOSITY >= 2)
		{
			lout << HsqTimer.info("‖H|Ψ>-E|Ψ>‖") << endl;
		}
	}
	
	Eold = Vout.energy;
	if (TEST == DMRG::CONVTEST::NORM_TEST)
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
		size_t standard_precision = cout.precision();
		if (EDGE == LANCZOS::EDGE::GROUND)
		{
			lout << "E₀=" << setprecision(13) << Vout.energy << ", E₀/L=" << Vout.energy/N_phys << setprecision(standard_precision) << endl;
		}
		else
		{
			lout << "E₀=" << setprecision(13) << Vout.energy << ", E₀/L=" << Vout.energy/N_phys << setprecision(standard_precision) << endl;
		}
		lout << eigeninfo() << endl;
		lout << Vout.state.info() << endl;
		double t_halfsweep = HalfsweepTimer.time();
		lout << HalfsweepTimer.info("half-sweep") 
		     << " ("
		     << "Lanczos=" << round(t_Lanczos/t_halfsweep*100.,0) << "%"
		     << ", sweeps=" << round(t_sweep/t_halfsweep*100.,0) << "%"
		     << ", LR=" << round(t_LR/t_halfsweep*100.,0) << "%"
		     << ", err=" << round(t_err/t_halfsweep*100.,0) << "%"
		     << ")"
		     << endl;
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
sweep_to_edge (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, bool MAKE_ENVIRONMENT)
{
	assert(stat.pivot==1 or stat.pivot==N_sites-2);
	if (stat.pivot==1)
	{
		Vout.state.sweepStep(DMRG::DIRECTION::LEFT, 1, DMRG::BROOM::QR);
		if (MAKE_ENVIRONMENT)
		{
			build_R(H,Vout,0);
			stat.pivot = 0;
		}
	}
	else if (stat.pivot==N_sites-2)
	{
		Vout.state.sweepStep(DMRG::DIRECTION::RIGHT, N_sites-2, DMRG::BROOM::QR);
		if (MAKE_ENVIRONMENT)
		{
			build_L(H,Vout,N_sites-1);
			stat.pivot = N_sites-1;
		}
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
cleanup (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, LANCZOS::EDGE::OPTION EDGE)
{
	sweep_to_edge(H,Vout,false);
	
	Vout.state.set_defaultCutoffs();
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::ON_EXIT)
	{
		size_t standard_precision = cout.precision();
		string Eedge = (EDGE == LANCZOS::EDGE::GROUND)? "Emin" : "Emax";
		lout << Eedge << "=" << setprecision(13) << Vout.energy << ", " << Eedge << "/L=" << Vout.energy/N_phys << setprecision(standard_precision) << endl;
		lout << Vout.state.info() << endl;
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
edgeState (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout, qarray<Nq> Qtot_input, LANCZOS::EDGE::OPTION EDGE,
		   DMRG::CONVTEST::OPTION TEST, double tol_eigval_input, double tol_state_input, size_t Dinit, size_t Dlimit, int Qinit,
		   size_t max_halfsweeps, size_t min_halfsweeps, double max_alpha_rsvd_input, double eps_svd_input, size_t savePeriod)
{
	tol_eigval = tol_eigval_input;
	tol_state  = tol_state_input;
	
	prepare(H, Vout, Qtot_input, false, Dinit, Qinit, max_alpha_rsvd_input, eps_svd_input);
	
	Stopwatch<> Saturn;
	
	// lambda function to print tolerances
	auto print_alpha_eps = [this,&Vout] ()
	{
		if (CHOSEN_VERBOSITY>=2)
		{
			lout //<< "α_noise=" << Vout.state.alpha_noise << ", "
			     //<< "ε_rdm=" << Vout.state.eps_rdm << ", "
			     << "α_rsvd=" << Vout.state.alpha_rsvd << ", "
			     << "ε_svd=" << Vout.state.eps_svd 
			     << endl;
		}
	};
	
	print_alpha_eps(); lout << endl;
	
	// average local dimension for later bond dimension increase
	size_t dimqlocAvg = 0;
	for (size_t l=0; l<H.length(); ++l)
	{
		dimqlocAvg += H.locBasis(l).size();
	}
	dimqlocAvg /= H.length();
	
	while (((err_eigval >= tol_eigval or err_state >= tol_state) and stat.N_halfsweeps < max_halfsweeps) or 
	       stat.N_halfsweeps < min_halfsweeps)
	{
		// For non-abelian symmetries, the fluctuations are not working correctly, so that they have to be turned off to allow convergence.
		// 8 is probably a good value for all "easy" models... if the convergence is not good, enhance this value.
//		if constexpr (Symmetry::NON_ABELIAN)
//		{
//			if (stat.N_halfsweeps == 8)
//			{
//				Vout.state.alpha_rsvd = 0.;
//				if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
//				{
//					lout << "Set α_rsvd=0." << endl;
//				}
//			}
//		}
		
		// sweep
		halfsweep(H,Vout,EDGE,TEST);
		
		// If truncated weight too large, increase upper limit per subspace by 10%, but at least by dimqlocAvg, overall never larger than Dlimit
		if (stat.N_halfsweeps%2 == 0 and totalTruncWeight >= Vout.state.eps_svd)
		{
			// increase by dimqlocAvg (at least 2), but by no more than 10%
			size_t N_sv_new = max(static_cast<size_t>(1.1*Vout.state.N_sv), Vout.state.N_sv+max(dimqlocAvg,2ul));
			// do not increase beyond Dlimit
			Vout.state.N_sv = min(N_sv_new,Dlimit);
		}
		
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			if (Vout.state.N_sv != Dmax_old)
			{
				lout << "Dmax=" << Dmax_old << "→" << Vout.state.N_sv << endl;
				Dmax_old = Vout.state.N_sv;
			}
			lout << endl;
		}
		
		#ifdef USE_HDF5_STORAGE
		if (savePeriod != 0 and stat.N_halfsweeps%savePeriod == 0)
		{
			Vout.state.save("mpsBackup");
		}
		#endif
	}
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << Saturn.info("total runtime") << endl;
	}
	cleanup(H,Vout,EDGE);
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
void DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
sweepStep (const MpHamiltonian &H, Eigenstate<Mps<Symmetry,Scalar> > &Vout)
{
	// build environments
	(stat.CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_L(H,Vout,++stat.pivot) : build_R(H,Vout,--stat.pivot);
	(stat.CURRENT_DIRECTION == DMRG::DIRECTION::RIGHT)? build_PL(H,Vout,stat.pivot)  : build_PR(H,Vout,stat.pivot);
	
	// adapt alpha
	PivotVector<Symmetry,Scalar> Vtmp1(Vout.state.A[stat.pivot]);
	PivotVector<Symmetry,Scalar> Vtmp2;
	Heff[stat.pivot].W = H.W[stat.pivot];
	precalc_blockStructure (Heff[stat.pivot].L, Vout.state.A[stat.pivot], Heff[stat.pivot].W, Vout.state.A[stat.pivot], Heff[stat.pivot].R, 
	                        H.locBasis(stat.pivot), H.opBasis(stat.pivot), Heff[stat.pivot].qlhs, Heff[stat.pivot].qrhs,
	                        Heff[stat.pivot].factor_cgcs);
	Heff[stat.pivot].qloc = H.locBasis(stat.pivot);
	Heff[stat.pivot].qOp  = H.opBasis(stat.pivot);
	HxV(Heff[stat.pivot], Vtmp1, Vtmp2);
	
	double DeltaEtrunc = dot(Vtmp1,Vtmp2)-Vout.energy;
	
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
		if (DeltaEtrunc < 0.) {f = 2.*(r+1.);}
		else if (r < 0.05)    {f = 1.2-r;}
		else if (r > 0.3)     {f = 1./(r+0.75);}
	}
	f = max(0.1,min(2.,f)); // limit between [0.1,2]
	Vout.state.alpha_rsvd *= f;
	Vout.state.alpha_rsvd = max(1e-11,min(max_alpha_rsvd,Vout.state.alpha_rsvd)); // limit between [1e-11,max_alpha_rsvd]
	
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
	
//	if (Heff[stat.pivot].qloc.size() == 0)
	{
		Heff[stat.pivot].W = H.W[stat.pivot];
		precalc_blockStructure (Heff[stat.pivot].L, Vout.state.A[stat.pivot], Heff[stat.pivot].W, Vout.state.A[stat.pivot], Heff[stat.pivot].R, 
		                        H.locBasis(stat.pivot), H.opBasis(stat.pivot), Heff[stat.pivot].qlhs, Heff[stat.pivot].qrhs,
		                        Heff[stat.pivot].factor_cgcs);
		Heff[stat.pivot].qloc = H.locBasis(stat.pivot);
		Heff[stat.pivot].qOp = H.opBasis(stat.pivot);
	}
	
	Eigenstate<PivotVector<Symmetry,Scalar> > g;
	g.state = PivotVector<Symmetry,Scalar>(Vout.state.A[stat.pivot]);
	LanczosSolver<PivotMatrix1<Symmetry,Scalar,Scalar>,PivotVector<Symmetry,Scalar>,Scalar> Lutz(LANCZOS::REORTHO::FULL);
	
	Lutz.set_dimK(min(29ul,dim(g.state)));
	Lutz.edgeState(Heff[stat.pivot],g, EDGE, 1e-7,1e-4, false);
	
	if (CHOSEN_VERBOSITY == DMRG::VERBOSITY::STEPWISE)
	{
		lout << "loc=" << stat.pivot << "\t" << Lutz.info() << endl;
		lout << Vout.state.test_ortho() << ", " << g.energy << endl;
	}
	
	Vout.energy = g.energy;
	Vout.state.A[stat.pivot] = g.state.data;
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
		Heff[loc].PL[n] = Vout.state.A[loc-1][0].adjoint() * Heff[loc-1].PL[n] * Psi0[n].A[loc-1][0];
		
		for (size_t s=1; s<Vout.state.locBasis(loc-1).size(); ++s)
		{
			Heff[loc].PL[n] += Vout.state.A[loc-1][s].adjoint() * Heff[loc-1].PL[n] * Psi0[n].A[loc-1][s];
		}
	}
}

template<typename Symmetry, typename MpHamiltonian, typename Scalar>
inline void DmrgSolver<Symmetry,MpHamiltonian,Scalar>::
build_PR (const MpHamiltonian &H, const Eigenstate<Mps<Symmetry,Scalar> > &Vout, size_t loc)
{
	for (size_t n=0; n<Psi0.size(); ++n)
	{
		Heff[loc].PR[n] = Psi0[n].A[loc+1][0] * Heff[loc+1].PR[n] * Vout.state.A[loc+1][0].adjoint();
		
		for (size_t s=1; s<Vout.state.locBasis(loc+1).size(); ++s)
		{
			Heff[loc].PR[n] += Psi0[n].A[loc+1][s] * Heff[loc+1].PR[n] * Vout.state.A[loc+1][s].adjoint();
		}
	}
}

#endif
