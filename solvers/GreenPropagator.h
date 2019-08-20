#ifndef GREEN_PROPAGATOR
#define GREEN_PROPAGATOR

#include "SuperQuadrator.h"
#include "solvers/TDVPPropagator.h"
#include "IntervalIterator.h"

#include <unsupported/Eigen/FFT>

/**Range of k-values:
\param MPI_PPI : from -pi to +pi
\param ZERO_2PI : from 0 to +2*pi
*/
enum Q_RANGE {MPI_PPI, ZERO_2PI};

/**cutoff function for the time domain*/
struct GreenPropagatorCutoff
{
	static double tmax;
	static double gauss (double tval) {return exp(-pow(2.*tval/tmax,2));};
};
double GreenPropagatorCutoff::tmax = 20.;

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
class GreenPropagator
{
public:
	
	GreenPropagator(){};
	
	/**
	\param label_input : prefix for saved files (e.g. type of Green's function, Hamiltonian parameters)
	\param tmax_input : maximal propagation time
	\param Nt_input : amount of time steps; the optimal number seems to be such that the average timestep is below 0.1
	\param wmin_input : minimal frequency for Fourier transform
	\param wmax_input : maximal frequency for Fourier transform
	\param Nw_input : amount of frequency points
	\param tol_compr_input : compression tolerance during time propagation
	\param GAUSSINT : if \p true, compute Gaussian integration weights for the cutoff function
	\param Q_RANGE_CHOICE_input : choose the q-range (-π to π, 0 to 2π)
	*/
	GreenPropagator (string label_input, 
	                 double tmax_input, int Nt_input, double wmin_input, double wmax_input, int Nw_input=1000, double tol_compr_input=1e-4, 
	                 bool GAUSSINT=true, Q_RANGE Q_RANGE_CHOICE_input=MPI_PPI)
	:label(label_input), tmax(tmax_input), Nt(Nt_input), wmin(wmin_input), wmax(wmax_input), Nw(Nw_input), tol_compr(tol_compr_input), 
	 USE_GAUSSIAN_INTEGRATION(GAUSSINT), Q_RANGE_CHOICE(Q_RANGE_CHOICE_input)
	{
		GreenPropagatorCutoff::tmax = tmax;
	}
	
	/**
	Reads G(t,x) from file, so that G(ω,q) can be recalculated.
	\param label_input : prefix for saved files (e.g. type of Green's function, Hamiltonian parameters)
	\param Lcell_input : unit cell length
	\param tmax_input : maximal propagation time
	\param Gtx_input : input of complex G(ω,q)
	\param GAUSSINT : if \p true, compute Gaussian integration weights for the cutoff function
	\param Q_RANGE_CHOICE_input : choose the q-range (-π to π, 0 to 2π)
	*/
	GreenPropagator (string label_input, int Lcell_input, double tmax_input, const MatrixXcd &Gtx_input, 
	                 bool GAUSSINT=true, Q_RANGE Q_RANGE_CHOICE_input=MPI_PPI)
	:label(label_input), Lcell(Lcell_input), tmax(tmax_input), USE_GAUSSIAN_INTEGRATION(GAUSSINT), Q_RANGE_CHOICE(Q_RANGE_CHOICE_input)
	{
		Gtx = Gtx_input;
		
		Nt = Gtx.rows();
		Nq = Gtx.cols();
		Lhetero = Nq;
		make_xarrays(Lhetero,Lcell);
		
		for (int l=0; l<Nq; ++l)
		{
			if (xvals[l] == 0) Gloct = Gtx.col(l);
		}
		
		GreenPropagatorCutoff::tmax = tmax;
		calc_intweights();
	}
	
	/**
	Reads G(t,x) with unit cell resolution from file, so that G(ω,q) can be recalculated.
	\param label_input : prefix for saved files (e.g. type of Green's function)
	\param tmax_input : maximal propagation time
	\param Gtx_input : input of complex G(ω,q)
	\param GAUSSINT : if \p true, compute Gaussian integration weights for the cutoff function
	\param Q_RANGE_CHOICE_input : choose the q-range (-π to π, 0 to 2π)
	*/
	GreenPropagator (string label_input, double tmax_input, const vector<vector<MatrixXcd>> &Gtx_input, 
	                 bool GAUSSINT=true, Q_RANGE Q_RANGE_CHOICE_input=MPI_PPI)
	:label(label_input), tmax(tmax_input), USE_GAUSSIAN_INTEGRATION(GAUSSINT), Q_RANGE_CHOICE(Q_RANGE_CHOICE_input)
	{
		Lcell = Gtx_input.size();
		GtxCell.resize(Lcell); for (int i=0; i<Lcell; ++i) GtxCell[i].resize(Lcell);
		
		for (int i=0; i<Lcell; ++i)
		for (int j=0; j<Lcell; ++j)
		{
			GtxCell[i][j] = Gtx_input[i][j];
		}
		
		Nt = GtxCell[0][0].rows();
		Nqc = GtxCell[0][0].cols();
		Lhetero = Nqc*Lcell;
		Nq = Lhetero;
		make_xarrays(Lhetero,Lcell);
		
		GloctCell.resize(Lcell);
		for (int i=0; i<Lcell; ++i)
		for (int n=0; n<Nqc; ++n)
		{
			if (dcell[n*Lcell] == 0) GloctCell[i] = GtxCell[i][i].col(n);
		}
		
		GreenPropagatorCutoff::tmax = tmax;
		calc_intweights();
	}
	
	/**
	Computes the Green's function G(t,x).
	\param H_hetero : Hamiltonian of heterogenic section
	\param OxPhi : vector with all local excitations
	\param OxPhi0 : starting state to propagate where the local excitation is
	\param Eg : ground-state energy
	\param TIME_FORWARDS : For photoemission, set to \p false. For inverse photoemission, set to \p true.
	*/
	void compute (const Hamiltonian &H_hetero, const vector<Mps<Symmetry,complex<double>>> &OxPhi, Mps<Symmetry,complex<double>> &OxPhi0, 
	              double Eg, bool TIME_FORWARDS = true);
	
	/**
	Computes the Green's function G(t,x) using counterpropagations forward and backward in time on the unit cell.
	Optimal when run with 2*Lcell threads.
	\param H_hetero : Hamiltonian of heterogenic section
	\param OxPhi : vector with all local excitations
	\param Eg : ground-state energy
	\param TIME_FORWARDS : For photoemission, set to \p false. For inverse photoemission, set to \p true.
	*/
	void compute_cell (const Hamiltonian &H_hetero, const vector<Mps<Symmetry,complex<double>>> &OxPhi, double Eg, bool TIME_FORWARDS = true);
	
	/**
	Recalculates the t→ω Fourier transform for a different ω-range
	\param wmin_new : minimal frequency for Fourier transform
	\param wmax_new : maximal frequency for Fourier transform
	\param Nw_new : amount of frequency points
	*/
	void recalc_FTw (double wmin_new, double wmax_new, int Nw_new=1000);
	
	void recalc_FTwCell (double wmin_new, double wmax_new, int Nw_new=1000);
	
	/**
	Set a Hermitian operator to be measured in the time-propagated state for testing purposes.
	\param Measure_input : vector of operators, length must be \p Lhetero
	\param measure_interval_input : measure after that many timesteps (there is always measurement at t=0)
	\param measure_name_input : How to label the operator in the output file
	\param measure_subfolder_input : Into which subfolder to put the output file
	*/
	void set_measurement (const vector<Mpo<Symmetry,MpoScalar>> &Measure_input, int measure_interval_input=10, 
	                      string measure_name_input="M", string measure_subfolder_input=".")
	{
		Measure = Measure_input;
		measure_interval = measure_interval_input;
		measure_name = measure_name_input;
		measure_subfolder = measure_subfolder_input;
	}
	
	/**Saves the real and imaginary parts of the Green's function into plain text files.*/
	void save() const;
	
	/**Saves the real and imaginary parts of the Green's function resolved by unit cell index into plain text files.*/
	void save_cell() const;
	
	/**Diagonalizes the Green's function matrix for different bands and saves the result.*/
	void diagonalize_and_save_cell() const;
	
	/**Calculates and saves the selfenergy Σ(ω,q).
	\param eps : free dispersion
	\param SAVE_G0 : whether to save the free Green's function G₀(q,ω) as well
	\param eta : broadening for G₀(q,ω).
	*/
	void save_selfenergy (double (*eps)(double), bool SAVE_G0, double eta=0.1) const;
	
	/**Calculates and saves the selfenergy Σᵢ(ω,q) for subband i (given by the unit cell).
	\param i : band index
	\param eps : free dispersion
	\param SAVE_G0 : whether to save the free Green's function G₀(q,ω) as well
	\param eta : broadening for G₀(q,ω).
	*/
	void save_selfenergy_band (int i, double (*eps)(double), bool SAVE_G0=true, double eta=0.1) const;
	
	/**Integrates the QDOS up to a given chemical potential μ (or the Fermi energy, since T=0). 
	   Can be used to find the right μ which gives the chosen filling n.
	\param mu : chemical potential, upper integration limit
	*/
	double integrate_Glocw (double mu);
	
	double integrate_Glocw_band (int i, double mu);
	
	inline MatrixXcd get_Gtx() const {return Gtx;}
	
	inline vector<vector<MatrixXcd>> get_GtxCell() const {return GtxCell;}
	
	inline void set_verbosity (DMRG::VERBOSITY::OPTION VERBOSITY) {CHOSEN_VERBOSITY = VERBOSITY;};
	
	/**
	Fourier transform G(ω,x)→G(ω,q) when system is supposed to be translationally invariant despite a unit cell.
	*/
	void FT_allSites();
	
	vector<MatrixXcd> FTbands() const;
	
	/**For testing purposes, see calc_Green, calc_GreenCell.*/
	vector<Mps<Symmetry,complex<double>>> OxPhiFull;
	
private:
	
	string label;
	
	int Nt, Nw, Nq, Nqc;
	double tmax;
	double wmin, wmax;
	int Lhetero, Lcell;
	double tol_compr;
	
	double tol_Lanczos = 1e-6; // 1e-6 seems sufficient; increase to 1e-8 for higher accuracy
	double tol_DeltaS = 1e-3; // 1e-3 seems sufficient; switch to homogeneous 2-site for higher accuracy
	
	DMRG::VERBOSITY::OPTION CHOSEN_VERBOSITY = DMRG::VERBOSITY::HALFSWEEPWISE;
	
	string xinfo() const;
	string qinfo (bool BETWEEN_CELLS=false) const;
	string tinfo() const;
	string winfo() const; // w=ω
	string xt_info() const;
	string xtqw_info (bool BETWEEN_CELLS=false) const;
	
	SuperQuadrator<GAUSS_LEGENDRE> TimeIntegrator;
	
	ArrayXd tvals, weights, tsteps;
	bool USE_GAUSSIAN_INTEGRATION = true;
	Q_RANGE Q_RANGE_CHOICE = MPI_PPI;
	bool TIME_FORWARDS;
	
	vector<double> xvals; // site labels: relative distance from excitation centre in sites
	vector<int> xinds; // site indices from 0 to Lhetero-1
	vector<int> dcell; // relative distance from excitation centre in unit cells
	vector<int> icell; // site indices within unit cell
	
	MatrixXcd Gtx, Gtq, Gwq;
	VectorXcd Gloct, Glocw;
	
	vector<vector<MatrixXcd>> GtxCell, GtqCell, GwqCell;
	vector<VectorXcd> GloctCell, GlocwCell;
	
	void calc_Green (int tindex, complex<double> phase, const vector<Mps<Symmetry,complex<double>>> &OxPhi, const Mps<Symmetry,complex<double>> &Psi);
	void calc_GreenCell (int tindex, complex<double> phase, const vector<Mps<Symmetry,complex<double>>> &OxPhi, const vector<Mps<Symmetry,complex<double>>> &Psi);
	void calc_GreenCell (int tindex, complex<double> phase, const std::array<vector<Mps<Symmetry,complex<double>>>,2> &Psi);
	
	void calc_intweights();
	void make_xarrays (int Lhetero_input, int Lcell_input);
	
	void propagate (const Hamiltonian &H_hetero, const vector<Mps<Symmetry,complex<double>>> &OxPhi, Mps<Symmetry,complex<double>> &OxPhi0, 
	                double Eg, bool TIME_FORWARDS);
	void propagate_cell (const Hamiltonian &H_hetero, const vector<Mps<Symmetry,complex<double>>> &OxPhi, double Eg, bool TIME_FORWARDS=true);
	void counterpropagate_cell (const Hamiltonian &H_hetero, const vector<Mps<Symmetry,complex<double>>> &OxPhi, double Eg, bool TIME_FORWARDS=true);
	
	void FT_xq();
	void FTcell_xq();
	void FT_tw();
	void FTcell_tw();
	ArrayXcd FTloc_tw (const VectorXcd &Gloct, const ArrayXd &wvals);
	
	vector<Mpo<Symmetry,MpoScalar>> Measure;
	void measure_wavepacket (const Mps<Symmetry,complex<double>> &Psi, double tval, string info="");
	int measure_interval;
	string measure_name, measure_subfolder;
};

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
compute (const Hamiltonian &H_hetero, const vector<Mps<Symmetry,complex<double>>> &OxPhi,
         Mps<Symmetry,complex<double>> &OxPhi0, double Eg, bool TIME_FORWARDS)
{
	Lcell = OxPhi0.Boundaries.length();
	Lhetero = H_hetero.length();
	if (Q_RANGE_CHOICE == MPI_PPI) assert(Lhetero%2 == 0 and "Please use an even number of sites in the heterogenic region!");
	assert(Lhetero%Lcell == 0 and "The heterogenic region is not commensurable with the length of the unit cell!");
	
//	for (size_t l=0; l<Lcell; ++l)
//	{
//		assert(OxPhi[l].locality() == Lhetero/2+l);
//	}
	
	calc_intweights();
	
	Nq = Lhetero;
	Nqc = Lhetero/Lcell;
	if (Q_RANGE_CHOICE == MPI_PPI) assert(Nqc%2 == 0 and "Please use an even number of unit cells!");
	make_xarrays(Lhetero,Lcell);
	
	propagate(H_hetero, OxPhi, OxPhi0, Eg, TIME_FORWARDS);
	
	FT_xq();
	FT_tw();
	save();
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
propagate (const Hamiltonian &H_hetero, const vector<Mps<Symmetry,complex<double>>> &OxPhi, Mps<Symmetry,complex<double>> &OxPhi0, double Eg, bool TIME_FORWARDS)
{
	double tsign = (TIME_FORWARDS==true)? -1.:+1.;
	
	Mps<Symmetry,complex<double> > Psi = OxPhi0;
	Psi.eps_svd = tol_compr;
	Psi.max_Nsv = Psi.calc_Dmax();
	
	TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,Mps<Symmetry,complex<double>>> TDVP(H_hetero, Psi);
	EntropyObserver<Mps<Symmetry,complex<double>>> Sobs(Lhetero, Nt, CHOSEN_VERBOSITY, tol_DeltaS);
	vector<bool> TWO_SITE = Sobs.TWO_SITE(0,Psi);
	if (CHOSEN_VERBOSITY > DMRG::VERBOSITY::ON_EXIT) lout << endl;
	
	Gtx.resize(Nt,Lhetero); Gtx.setZero();
	Gloct.resize(Nt); Gloct.setZero();
	
	IntervalIterator t(0.,tmax,Nt);
	double tval = 0.;
	
	// 0.1. measure wavepacket at t=0
	measure_wavepacket(Psi,0);
	
	// 0.2. if no (open) integration weights, calculate G at t=0
	if (!USE_GAUSSIAN_INTEGRATION)
	{
		Stopwatch<> StepTimer;
		calc_Green(0, -1.i*exp(-1.i*tsign*Eg*tval), OxPhi, Psi);
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			lout << StepTimer.info("G(t,x) calculation") << endl;
			lout << endl;
		}
	}
	
	Stopwatch<> TpropTimer;
	for (t=t.begin(); t!=t.end(); ++t)
	{
		Stopwatch<> StepTimer;
		// 1. propagate
		TDVP.t_step_adaptive(H_hetero, Psi, 1.i*tsign*tsteps(t.index()), TWO_SITE, 1,tol_Lanczos);
//		TDVP.t_step(H_hetero, Psi, 1.i*tsign*tsteps(t.index()), 1,tol_Lanczos);
		tval = tsteps.head(t.index()+1).sum();
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE) lout << StepTimer.info("time step") << endl;
		
		if (Psi.get_truncWeight().sum() > 0.5*tol_compr)
		{
			Psi.max_Nsv = min(static_cast<size_t>(max(Psi.max_Nsv*1.1, Psi.max_Nsv+1.)),200ul);
			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
			{
				lout << termcolor::yellow << "Setting Psi.max_Nsv to " << Psi.max_Nsv << termcolor::reset << endl;
			}
		}
		
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			lout << TDVP.info() << endl;
			lout << Psi.info() << endl;
			lout << "propagated to t=" << tval << ", stepsize=" << tsteps(t.index()) << endl;
		}
		
		// 2. measure
		// 2.1. Green's function
		calc_Green(t.index(), -1.i*exp(-1.i*tsign*Eg*tval), OxPhi, Psi);
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE) lout << StepTimer.info("G(t,x) calculation") << endl;
		// 2.2. measure wavepacket at t
		if ((t.index()-1)%measure_interval == 0 and t.index() > 1) measure_wavepacket(Psi,tval);
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE) lout << StepTimer.info("wavepacket measurement") << endl;
		
		// 3. check entropy increase
		auto PsiTmp = Psi;
		PsiTmp.eps_svd = 1e-15;
		PsiTmp.skim(DMRG::BROOM::SVD);
		double r = (t.index()==0)? 1.:tsteps(t.index()-1)/tsteps(t.index());
		TWO_SITE = Sobs.TWO_SITE(t.index(), PsiTmp, r);
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE) lout << StepTimer.info("entropy calculation") << endl;
		
		// final info
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			lout << TpropTimer.info("total running time",false) << endl;
			lout << endl;
		}
	}
	
	// measure wavepacket at t=t_end
	measure_wavepacket(Psi,tval);
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
compute_cell (const Hamiltonian &H_hetero, const vector<Mps<Symmetry,complex<double>>> &OxPhi, double Eg, bool TIME_FORWARDS)
{
	Lcell = OxPhi[0].Boundaries.length();
	Lhetero = H_hetero.length();
	assert(Lhetero%Lcell == 0);
	
//	for (size_t l=0; l<Lcell; ++l)
//	{
//		assert(OxPhi[l].locality() == Lhetero/2+l);
//	}
	
	calc_intweights();
	
	Nq = Lhetero;
	Nqc = Lhetero/Lcell;
	make_xarrays(Nq,Lcell);
	
	GtxCell.resize(Lcell);
	for (int i=0; i<Lcell; ++i)
	{
		GtxCell[i].resize(Lcell);
	}
	for (int i=0; i<Lcell; ++i)
	for (int j=0; j<Lcell; ++j)
	{
		GtxCell[i][j].resize(Nt,Nqc);
	}
	
	GloctCell.resize(Lcell);
	for (int i=0; i<Lcell; ++i)
	{
		GloctCell[i].resize(Nt);
	}
	
//	propagate_cell(H_hetero, OxPhi, Eg, TIME_FORWARDS);
	counterpropagate_cell(H_hetero, OxPhi, Eg, TIME_FORWARDS);
	
	FTcell_xq();
	FTcell_tw();
	save_cell();
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
propagate_cell (const Hamiltonian &H_hetero, const vector<Mps<Symmetry,complex<double>>> &OxPhi, double Eg, bool TIME_FORWARDS)
{
	double tsign = (TIME_FORWARDS==true)? -1.:+1.;
	
	vector<Mps<Symmetry,complex<double>>> Psi(Lcell);
	for (int i=0; i<Lcell; ++i)
	{
		Psi[i] = OxPhi[i];
		Psi[i].eps_svd = tol_compr;
		Psi[i].max_Nsv = Psi[i].calc_Dmax();
	}
	
	vector<TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,Mps<Symmetry,complex<double>>>> TDVP(Lcell);
	for (int i=0; i<Lcell; ++i)
	{
		TDVP[i] = TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,Mps<Symmetry,complex<double>>>(H_hetero, Psi[i]);
	}
	vector<EntropyObserver<Mps<Symmetry,complex<double>>>> Sobs(Lcell);
	vector<vector<bool>> TWO_SITE(Lcell);
	for (int i=0; i<Lcell; ++i)
	{
		DMRG::VERBOSITY::OPTION VERB = (i==0)? CHOSEN_VERBOSITY:DMRG::VERBOSITY::SILENT;
		Sobs[i] = EntropyObserver<Mps<Symmetry,complex<double>>>(Lhetero, Nt, VERB, tol_DeltaS);
		TWO_SITE[i] = Sobs[i].TWO_SITE(0,Psi[i]);
	}
	if (CHOSEN_VERBOSITY > DMRG::VERBOSITY::ON_EXIT) lout << endl;
	
	IntervalIterator t(0.,tmax,Nt);
	double tval = 0.;
	
	// 0.1. measure wavepacket at t=0
	for (int i=0; i<Lcell; ++i)
	{
		measure_wavepacket(Psi[i],0,make_string("i=",i,"_"));
	}
	
	// 0.2. if no (open) integration weights, calculate G at t=0
	if (!USE_GAUSSIAN_INTEGRATION)
	{
		Stopwatch<> StepTimer;
		calc_GreenCell(0, -1.i*exp(-1.i*tsign*Eg*tval), OxPhi, Psi);
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			lout << StepTimer.info("G(t,x) calculation") << endl;
			lout << endl;
		}
	}
	
	Stopwatch<> TpropTimer;
	for (t=t.begin(); t!=t.end(); ++t)
	{
		Stopwatch<> StepTimer;
		
		// 1. propagate
		#pragma omp parallel for
		for (int i=0; i<Lcell; ++i)
		{
			TDVP[i].t_step_adaptive(H_hetero, Psi[i], 1.i*tsign*tsteps(t.index()), TWO_SITE[i], 1,tol_Lanczos);
			
			if (Psi[i].get_truncWeight().sum() > 0.5*tol_compr)
			{
				Psi[i].max_Nsv = min(static_cast<size_t>(max(Psi[i].max_Nsv*1.1, Psi[i].max_Nsv+1.)),200ul);
				if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE and i==0)
				{
					lout << termcolor::yellow << "Setting Psi.max_Nsv to " << Psi[i].max_Nsv << termcolor::reset << endl;
				}
			}
		}
		tval = tsteps.head(t.index()+1).sum();
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE) lout << StepTimer.info("time step") << endl;
		
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			lout << TDVP[0].info() << endl;
			lout << Psi[0].info() << endl;
			lout << "propagated to t=" << tval << ", stepsize=" << tsteps(t.index()) << endl;
		}
		
		// 2. measure
		// 2.1. Green's function
		calc_GreenCell(t.index(), -1.i*exp(-1.i*tsign*Eg*tval), OxPhi, Psi);
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE) lout << StepTimer.info("G(t,x) calculation") << endl;
		// 2.2. measure wavepacket at t
		#pragma omp parallel for
		for (int i=0; i<Lcell; ++i)
		{
			if ((t.index()-1)%measure_interval == 0 and t.index() > 1) measure_wavepacket(Psi[i],tval,make_string("i=",i,"_"));
		}
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE) lout << StepTimer.info("wavepacket measurement") << endl;
		
		// 3. check entropy increase
		#pragma omp parallel for
		for (int i=0; i<Lcell; ++i)
		{
			auto PsiTmp = Psi[i];
			PsiTmp.eps_svd = 1e-15;
			PsiTmp.skim(DMRG::BROOM::SVD);
			double r = (t.index()==0)? 1.:tsteps(t.index()-1)/tsteps(t.index());
			TWO_SITE[i] = Sobs[i].TWO_SITE(t.index(), PsiTmp, r);
			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE and i==0) lout << StepTimer.info("entropy calculation") << endl;
		}
		
		// final info
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			lout << TpropTimer.info("total running time",false) << endl;
			lout << endl;
		}
	}
	
	// measure wavepacket at t=t_end
	#pragma omp parallel for
	for (int i=0; i<Lcell; ++i)
	{
		measure_wavepacket(Psi[i],tval,make_string("i=",i,"_"));
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
counterpropagate_cell (const Hamiltonian &H_hetero, const vector<Mps<Symmetry,complex<double>>> &OxPhi, double Eg, bool TIME_FORWARDS)
{
	double tsign = (TIME_FORWARDS==true)? -1.:+1.;
	std::array<double,2> zfac;
	zfac[0] = +1.; // forw propagation
	zfac[1] = -1.; // back propagation
	
	std::array<vector<Mps<Symmetry,complex<double>>>,2> Psi;
	for (int z=0; z<2; ++z)
	{
		Psi[z].resize(Lcell);
		for (int i=0; i<Lcell; ++i)
		{
			Psi[z][i] = OxPhi[i];
			Psi[z][i].eps_svd = tol_compr;
			Psi[z][i].max_Nsv = Psi[z][i].calc_Dmax();
		}
	}
	
	std::array<vector<TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,Mps<Symmetry,complex<double>>>>,2> TDVP;
	for (int z=0; z<2; ++z)
	{
		TDVP[z].resize(Lcell);
		for (int i=0; i<Lcell; ++i)
		{
			TDVP[z][i] = TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,Mps<Symmetry,complex<double>>>(H_hetero, Psi[z][i]);
		}
	}
	std::array<vector<EntropyObserver<Mps<Symmetry,complex<double>>>>,2> Sobs;
	std::array<vector<vector<bool>>,2> TWO_SITE;
	for (int z=0; z<2; ++z)
	{
		Sobs[z].resize(Lcell);
		TWO_SITE[z].resize(Lcell);
		
		for (int i=0; i<Lcell; ++i)
		{
			DMRG::VERBOSITY::OPTION VERB = (i==0 and z==0)? CHOSEN_VERBOSITY:DMRG::VERBOSITY::SILENT;
			Sobs[z][i] = EntropyObserver<Mps<Symmetry,complex<double>>>(Lhetero, Nt, VERB, tol_DeltaS);
			TWO_SITE[z][i] = Sobs[z][i].TWO_SITE(0,Psi[z][i]);
		}
	}
	if (CHOSEN_VERBOSITY > DMRG::VERBOSITY::ON_EXIT) lout << endl;
	
	IntervalIterator t(0.,tmax,Nt);
	double tval = 0.;
	
	// 0.1. measure wavepacket at t=0
	if (Measure.size() != 0)
	{
		for (int i=0; i<Lcell; ++i)
		{
			measure_wavepacket(Psi[0][i],0,make_string("i=",i,"_"));
		}
	}
	
	// 0.2. if no (open) integration weights, calculate G at t=0
	if (!USE_GAUSSIAN_INTEGRATION)
	{
		Stopwatch<> StepTimer;
		calc_GreenCell(0, -1.i*exp(-1.i*tsign*Eg*tval), Psi);
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			lout << StepTimer.info("G(t,x) calculation") << endl;
			lout << endl;
		}
	}
	
	Stopwatch<> TpropTimer;
	for (t=t.begin(); t!=t.end(); ++t)
	{
		Stopwatch<> StepTimer;
		
		// 1. propagate
		#pragma omp parallel for collapse(2)
		for (int z=0; z<2; ++z)
		for (int i=0; i<Lcell; ++i)
		{
			//---------------------------------------------------------------------------------------------------------------
			TDVP[z][i].t_step_adaptive(H_hetero, Psi[z][i], 0.5*1.i*zfac[z]*tsign*tsteps(t.index()), TWO_SITE[z][i], 1,tol_Lanczos);
			//TDVP[z][i].t_step(H_hetero, Psi[z][i], 0.5*1.i*zfac[z]*tsign*tsteps(t.index()), 1,tol_Lanczos);
			//---------------------------------------------------------------------------------------------------------------
			
			if (Psi[z][i].get_truncWeight().sum() > 0.5*tol_compr)
			{
				Psi[z][i].max_Nsv = min(static_cast<size_t>(max(Psi[z][i].max_Nsv*1.1, Psi[z][i].max_Nsv+1.)),200ul);
				if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE and i==0 and z==0)
				{
					lout << termcolor::yellow << "Setting Psi.max_Nsv to " << Psi[z][i].max_Nsv << termcolor::reset << endl;
				}
			}
		}
		tval = tsteps.head(t.index()+1).sum();
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE) lout << StepTimer.info("time step") << endl;
		
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			lout << TDVP[0][0].info() << endl;
			lout << Psi[0][0].info() << endl;
			lout << "propagated to t=0.5*" << tval << "=" << 0.5*tval 
			     << ", stepsize=0.5*" << tsteps(t.index()) << "=" << 0.5*tsteps(t.index()) << endl;
		}
		
		// 2. measure
		// 2.1. Green's function
		//----------------------------------------------------------
		calc_GreenCell(t.index(), -1.i*exp(-1.i*tsign*Eg*tval), Psi);
		//----------------------------------------------------------
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE) lout << StepTimer.info("G(t,x) calculation") << endl;
		// 2.2. measure wavepacket at t
		if (Measure.size() != 0)
		{
			#pragma omp parallel for
			for (int i=0; i<Lcell; ++i)
			{
				if ((t.index()-1)%measure_interval == 0 and t.index() > 1) 
				{
//					lout << termcolor::yellow << "tval=" << tval << ", MEASURE" << termcolor::reset << endl;
					measure_wavepacket(Psi[0][i],0.5*tval,make_string("i=",i,"_"));
				}
			}
		}
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE) lout << StepTimer.info("wavepacket measurement") << endl;
		
		// 3. check entropy increase
		#pragma omp parallel for collapse(2)
		for (int z=0; z<2; ++z)
		for (int i=0; i<Lcell; ++i)
		{
			auto PsiTmp = Psi[z][i];
			PsiTmp.eps_svd = 1e-15;
			PsiTmp.skim(DMRG::BROOM::SVD);
			double r = (t.index()==0)? 1.:tsteps(t.index()-1)/tsteps(t.index());
//			// override: always 2-site near excitation centre:
//			vector<int> overrides = {Lhetero/2-1, Lhetero/2, Lhetero/2+1};
			TWO_SITE[z][i] = Sobs[z][i].TWO_SITE(t.index(), PsiTmp, r);
			
			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE and i==0 and z==0) lout << StepTimer.info("entropy calculation") << endl;
		}
		
		// final info
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			lout << TpropTimer.info("total running time",false) << endl;
			lout << endl;
		}
	}
	
	// measure wavepacket at t=t_end
	if (Measure.size() != 0)
	{
		#pragma omp parallel for
		for (int i=0; i<Lcell; ++i)
		{
			measure_wavepacket(Psi[0][i],0.5*tval,make_string("i=",i,"_"));
		}
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
calc_Green (int tindex, complex<double> phase, const vector<Mps<Symmetry,complex<double>>> &OxPhi, const Mps<Symmetry,complex<double>> &Psi)
{
//	MatrixXcd Gtx_(Gtx.rows(),Gtx.cols());
	
//	//variant: Don't use cell shift, OxPhiFull must be of length Lhetero
//	#pragma omp parallel for
//	for (size_t l=0; l<Lhetero; ++l)
//	{
//		Gtx(tindex,l) = phase * dot_hetero(OxPhiFull[l], Psi);
//		if (xvals[l] == 0) Gloct(tindex) = Gtx(tindex,l); // save local Green's function
//	}
	
	// variant: Use cell shift
	#pragma omp parallel for
	for (size_t l=0; l<Lhetero; ++l)
	{
		Gtx(tindex,l) = phase * dot_hetero(OxPhi[icell[l]], Psi, dcell[l]);
		if (xvals[l] == 0) Gloct(tindex) = Gtx(tindex,l); // save local Green's function
	}
	
//	for (size_t l=0; l<Lhetero; ++l)
//	{
//		cout << "l=" << l << ", " << Gtx(tindex,l) << ", " << Gtx_(tindex,l) << ", diff=" << abs(Gtx(tindex,l)-Gtx_(tindex,l)) << endl;
//	}
//	cout << termcolor::blue << "total diff = " << (Gtx.row(tindex)-Gtx_.row(tindex)).norm() << termcolor::reset << endl;
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
calc_GreenCell (int tindex, complex<double> phase, 
                const vector<Mps<Symmetry,complex<double>>> &OxPhi,
                const vector<Mps<Symmetry,complex<double>>> &Psi)
{
	// variant: Use cell shift
	#pragma omp parallel for collapse(3)
	for (size_t i=0; i<Lcell; ++i)
	for (size_t j=0; j<Lcell; ++j)
	for (size_t n=0; n<Nqc; ++n)
	{
		GtxCell[i][j](tindex,n) = phase * dot_hetero(OxPhi[i], Psi[j], dcell[n*Lcell]);
	}
	
//	//variant: Don't use cell shift, OxPhiFull must be of length Lhetero
//	#pragma omp parallel for collapse(3)
//	for (size_t i=0; i<Lcell; ++i)
//	for (size_t j=0; j<Lcell; ++j)
//	for (size_t n=0; n<Nqc; ++n)
//	{
//		GtxCell[i][j](tindex,n) = phase * dot_hetero(OxPhiFull[n*Lcell+i], Psi[j]);
//	}
	
	for (size_t i=0; i<Lcell; ++i)
	for (size_t n=0; n<Nqc; ++n)
	{
		if (dcell[n*Lcell] == 0) 
		{
			GloctCell[i](tindex) = GtxCell[i][i](tindex,n); // save local Green's function
		}
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
calc_GreenCell (int tindex, complex<double> phase, 
                const std::array<vector<Mps<Symmetry,complex<double>>>,2> &Psi)
{
	#pragma omp parallel for collapse(3)
	for (size_t i=0; i<Lcell; ++i)
	for (size_t j=0; j<Lcell; ++j)
	for (size_t n=0; n<Nqc; ++n)
	{
		GtxCell[i][j](tindex,n) = phase * dot_hetero(Psi[1][i], Psi[0][j], dcell[n*Lcell]);
	}
	
	for (size_t i=0; i<Lcell; ++i)
	for (size_t n=0; n<Nqc; ++n)
	{
		if (dcell[n*Lcell] == 0) 
		{
			GloctCell[i](tindex) = GtxCell[i][i](tindex,n); // save local Green's function
		}
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
measure_wavepacket (const Mps<Symmetry,complex<double>> &Psi, double tval, string info)
{
	if (Measure.size() != 0)
	{
//		cout << termcolor::yellow << "in measure: t=" << tval << ", info=" << info << termcolor::reset << endl;
		double norm = Psi.squaredNorm();
		ArrayXd res(Measure.size());
		
		#pragma omp parallel for
		for (size_t l=0; l<Measure.size(); ++l)
		{
			res(l) = isReal(avg_hetero(Psi, Measure[l], Psi)) / norm;
		}
		
		ofstream Filer(make_string(measure_subfolder,"/",label,"_Op=",measure_name,"_",info,xinfo(),"_t=",tval,".dat"));
		for (size_t l=0; l<Measure.size(); ++l)
		{
			Filer << xvals[l] << "\t" << res(l) << endl;
		}
		Filer.close();
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
make_xarrays (int Lhetero_input, int Lcell_input)
{
	xinds.resize(Lhetero_input);
	xvals.resize(Lhetero_input);
	
	iota(begin(xinds),end(xinds),0);
	
	int x0 = Lhetero_input/2; // first site of central unit cell
//	cout << "make_xarrays: x0=" << x0 << endl;
	
	for (int ix=0; ix<xinds.size(); ++ix)
	{
		xvals[ix] = static_cast<double>(xinds[ix]-x0);
	}
	
	for (int d=0; d<Lhetero_input/Lcell_input; ++d)
	for (int i=0; i<Lcell_input; ++i)
	{
		icell.push_back(i);
		dcell.push_back(d);
	}
	
	for (int i=0; i<Lhetero_input; ++i)
	{
		dcell[i] -= x0/Lcell_input;
	}
	
//	for (int i=0; i<Lhetero; ++i)
//	{
//		cout << "i=" << xinds[i] << ", x=" << xvals[i] << ", icell=" << icell[i] << ", dcell=" << dcell[i] << endl;
//	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
calc_intweights()
{
	Stopwatch<> Watch;
	TimeIntegrator = SuperQuadrator<GAUSS_LEGENDRE>(GreenPropagatorCutoff::gauss,0.,tmax,Nt);
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << Watch.info("integration weights") << endl;
	}
	
	if (USE_GAUSSIAN_INTEGRATION)
	{
		tvals   = TimeIntegrator.get_abscissa();
		weights = TimeIntegrator.get_weights();
		tsteps  = TimeIntegrator.get_steps();
		
//		ofstream Filer(make_string("weights_",tinfo(),".dat"));
//		for (int i=0; i<Nt; ++i)
//		{
//			Filer << tvals(i) << "\t" << weights(i) << endl;
//		}
//		Filer.close();
		
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
		{
			double erf2 = 0.995322265018952734162069256367252928610891797040060076738; // Mathematica♥
			double integral = 0.25*sqrt(M_PI)*erf2*tmax;
			lout << termcolor::blue 
				 << setprecision(14)
				 << "integration weight test: ∫w(t)dt=" << weights.sum() 
				 << ", analytical=" << integral 
				 << ", diff=" << abs(weights.sum()-integral) 
				 << termcolor::reset << endl;
		}
	}
	else
	{
		double dt = tmax/Nt;
		
		tvals.resize(Nt);
		for (int i=0; i<Nt; ++i) tvals(i) = i*dt;
		
		weights.resize(Nt); weights = 1.;
		tsteps.resize(Nt); tsteps = dt;
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
FT_xq()
{
	IntervalIterator w(wmin,wmax,Nw);
	Gtq.resize(Nt,Nq); Gtq.setZero();
	
	Stopwatch<> FourierWatch;
	
	// Use FFT to transform from x to q
	Eigen::FFT<double> fft;
	for (int it=0; it<tvals.rows(); ++it)
	{
		VectorXcd vtmp;
		fft.fwd(vtmp,Gtx.row(it));
		Gtq.row(it) = vtmp;
	}
	// phase shift factor because the origin site is in the middle
	for (int iq=0; iq<Nq; ++iq)
	{
		Gtq.col(iq) *= exp(-1.i*2.*M_PI/double(xvals.size())*xvals[0]*double(iq));
	}
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::ON_EXIT)
	{
		lout << FourierWatch.info("FFT x→q") << endl;
	}
	
//	// Explicit FT for testing
//	for (int it=0; it<Nt; ++it)
//	for (int iq=0; iq<Nq; ++iq)
//	{
//		double qval = 2.*M_PI/Nq * iq;
//		
//		for (int ix=0; ix<Nq; ++ix)
//		{
//			Gtq(it,iq) += Gtx(it,ix) * exp(-1.i*qval*xvals[ix]);
//		}
//	}
//	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
//	{
//		lout << FourierWatch.info("FT x→q") << endl;
//	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
FTcell_xq()
{
	IntervalIterator w(wmin,wmax,Nw);
	
	GtqCell.resize(Lcell);
	for (int i=0; i<Lcell; ++i) GtqCell[i].resize(Lcell);
	
	Stopwatch<> FourierWatch;
	
	for (int i=0; i<Lcell; ++i)
	for (int j=0; j<Lcell; ++j)
	{
		GtqCell[i][j].resize(Nt,Nqc); GtqCell[i][j].setZero();
		
		for (int iq=0; iq<Nqc; ++iq)
		{
			double qval = 2.*M_PI/Nqc * iq;
			
			for (int n=0; n<Nqc; ++n)
			{
				GtqCell[i][j].col(iq) += GtxCell[i][j].col(n) * exp(-1.i*qval*double(dcell[n*Lcell]));
			}
		}
	}
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << FourierWatch.info("FT intercell x→q") << endl;
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
FT_tw()
{
	IntervalIterator w(wmin,wmax,Nw);
	ArrayXd wvals = w.get_abscissa();
	Gwq.resize(Nw,Nq); Gwq.setZero();
	
	Stopwatch<> FourierWatch;
	
	// Use normal summation to transform from t to w
//	#pragma omp parallel for
	for (int iw=0; iw<wvals.rows(); ++iw)
	{
		double wval = wvals(iw);
		
		for (int it=0; it<tvals.rows(); ++it)
		{
			double tval = tvals(it);
			// If Gaussian integration is employed, the damping is already included in the weights
			double damping = (USE_GAUSSIAN_INTEGRATION)? 1.:exp(-pow(2.*tval/tmax,2));
			
			Gwq.row(iw) += weights(it) * damping * exp(+1.i*wval*tval) * Gtq.row(it); 
		}
	}
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::ON_EXIT)
	{
		lout << FourierWatch.info("FT t→ω") << endl;
	}
	
	if (Q_RANGE_CHOICE == MPI_PPI)
	{
		MatrixXcd Gwq_tmp = Gwq;
		Gwq_tmp.leftCols (Nq/2) = Gwq.rightCols(Nq/2);
		Gwq_tmp.rightCols(Nq/2) = Gwq.leftCols (Nq/2);
		Gwq = Gwq_tmp;
	}
	
	// repeat last q-point for better plotting
	Gwq.conservativeResize(Nw,Nq+1);
	Gwq.col(Nq) = Gwq.col(0);
	
	// Calculate FT of local Green's function
	Glocw = FTloc_tw(Gloct,wvals);
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
FTcell_tw()
{
	Nq = Lhetero;
	Nqc = Lhetero/Lcell;
	
	IntervalIterator w(wmin,wmax,Nw);
	ArrayXd wvals = w.get_abscissa();
	
	GwqCell.resize(Lcell); for (int i=0; i<Lcell; ++i) GwqCell[i].resize(Lcell);
	GlocwCell.resize(Lcell);
	
	Stopwatch<> FourierWatch;
	
	for (int i=0; i<Lcell; ++i)
	for (int j=0; j<Lcell; ++j)
	{
		GwqCell[i][j].resize(Nw,Nqc); GwqCell[i][j].setZero();
		
		// Use normal summation to transform from t to w
		#pragma omp parallel for
		for (int iw=0; iw<wvals.rows(); ++iw)
		{
			double wval = wvals(iw);
			
			for (int it=0; it<tvals.rows(); ++it)
			{
				double tval = tvals(it);
				// If Gaussian integration is employed, the damping is already included in the weights
				double damping = (USE_GAUSSIAN_INTEGRATION)? 1.:exp(-pow(2.*tval/tmax,2));
				
				GwqCell[i][j].row(iw) += weights(it) * damping * exp(+1.i*wval*tval) * GtqCell[i][j].row(it);
			}
		}
		
		if (Q_RANGE_CHOICE == MPI_PPI)
		{
			MatrixXcd Gwq_tmp = GwqCell[i][j];
			Gwq_tmp.leftCols (Nqc/2) = GwqCell[i][j].rightCols(Nqc/2);
			Gwq_tmp.rightCols(Nqc/2) = GwqCell[i][j].leftCols (Nqc/2);
			GwqCell[i][j] = Gwq_tmp;
		}
		
		// repeat last q-point for better plotting
		GwqCell[i][j].conservativeResize(Nw,Nqc+1);
		GwqCell[i][j].col(Nqc) = GwqCell[i][j].col(0);
		
		// Calculate local Green's function
		if (i==j)
		{
			GlocwCell[i] = FTloc_tw(GloctCell[i],wvals);
		}
	}
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::ON_EXIT)
	{
		lout << FourierWatch.info("FT intercell t→ω") << endl;
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
FT_allSites()
{
	Gtq.resize(Nt,Nq); Gtq.setZero();
	
	Stopwatch<> FourierWatch;
	
	for (int iq=0; iq<Nq; ++iq)
	{
		double qval = 2.*M_PI/Nq * iq;
		
		for (int n=0; n<Nqc; ++n)
		for (int i=0; i<Lcell; ++i)
		for (int j=0; j<Lcell; ++j)
		{
			Gtq.col(iq) += 1./Lcell * GtxCell[i][j].col(n) * exp(-1.i*qval*double(Lcell)*double(dcell[n*Lcell])) * exp(-1.i*qval*double(i-j));
		}
	}
	
	lout << FourierWatch.info("FT all sites x→q") << endl;
	
	Gloct.resize(Nt);
	Gloct = GloctCell[0];
	
	FT_tw();
	
	bool PRINT = (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::ON_EXIT)? true:false;
	
	saveMatrix(Gwq.real(), label+"_G=ωqRe_"+xtqw_info()+".dat", PRINT);
	saveMatrix(Gwq.imag(), label+"_G=ωqIm_"+xtqw_info()+".dat", PRINT);
//	saveMatrix_cpython(Gwq, label+"_G=ωq_"+xtqw_info()+".dat", PRINT);
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
recalc_FTw (double wmin_new, double wmax_new, int Nw_new)
{
	wmin = wmin_new;
	wmax = wmax_new;
	Nw = Nw_new;
	
	FT_xq();
	FT_tw();
	save();
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
recalc_FTwCell (double wmin_new, double wmax_new, int Nw_new)
{
	wmin = wmin_new;
	wmax = wmax_new;
	Nw = Nw_new;
	
	FTcell_xq();
	FTcell_tw();
	save_cell();
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
ArrayXcd GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
FTloc_tw (const VectorXcd &Gloct_in, const ArrayXd &wvals)
{
	assert(USE_GAUSSIAN_INTEGRATION);
	ArrayXcd Glocw_out(wvals.rows()); Glocw_out.setZero();
	
	Stopwatch<> FourierWatch;
	
	// Use normal summation to transform from t to w
	#pragma omp parallel for
	for (int iw=0; iw<wvals.rows(); ++iw)
	{
		double wval = wvals(iw);
		for (int it=0; it<tvals.rows(); ++it)
		{
			double tval = tvals(it);
			// If Gaussian integration is employed, the damping is already included in the weights
			double damping = (USE_GAUSSIAN_INTEGRATION)? 1.:exp(-pow(2.*tval/tmax,2));
			
			Glocw_out(iw) += weights(it) * damping * exp(+1.i*wval*tval) * Gloct_in(it);
		}
	}
	
	return Glocw_out;
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
double GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
integrate_Glocw (double mu)
{
	Quadrator<GAUSS_LEGENDRE> Q;
	int Nint = 200;
	
	ArrayXd wabscissa(Nint);
	for (int i=0; i<Nint; ++i) {wabscissa(i) = Q.abscissa(i,wmin,mu,Nint);}
	
	ArrayXd QDOS = -1.*M_1_PI * FTloc_tw(Gloct,wabscissa).imag();
	
	return (Q.get_weights(wmin,mu,Nint) * QDOS).sum();
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
double GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
integrate_Glocw_band (int i, double mu)
{
	Quadrator<GAUSS_LEGENDRE> Q;
	int Nint = 200;
	
	ArrayXd wabscissa(Nint);
	for (int i=0; i<Nint; ++i) {wabscissa(i) = Q.abscissa(i,wmin,mu,Nint);}
	
	ArrayXd QDOS = -1.*M_1_PI * FTloc_tw(GloctCell[i],wabscissa).imag();
	
	return (Q.get_weights(wmin,mu,Nint) * QDOS).sum();
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
string GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
xinfo() const
{
	stringstream ss;
	ss << "L=" << Lhetero;
	return ss.str();
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
string GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
qinfo (bool BETWEEN_CELLS) const
{
	stringstream ss;
	int qmin, qmax;
	if      (Q_RANGE_CHOICE==MPI_PPI)  {qmin=-1; qmax=1;}
	else if (Q_RANGE_CHOICE==ZERO_2PI) {qmin=0;  qmax=2;}
	ss << "qmin=" << qmin << "_qmax=" << qmax;
//	<< "_Nq=";
//	if (BETWEEN_CELLS) {ss << Nqc+1;}
//	else               {ss << Nq+1;}
	return ss.str();
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
string GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
tinfo() const
{
	stringstream ss;
	ss << "tmax=" << tmax;
	return ss.str();
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
string GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
winfo() const
{
	stringstream ss;
	ss << "wmin=" << wmin << "_wmax=" << wmax;
//	<< "_Nw=" << Nw;
	return ss.str();
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
string GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
xt_info() const
{
	return xinfo() + "_" + tinfo();
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
string GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
xtqw_info (bool BETWEEN_CELLS) const
{
	return xinfo() + "_" + tinfo() + "_" + qinfo(BETWEEN_CELLS) + "_" + winfo();
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
vector<MatrixXcd> GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
FTbands() const
{
	vector<MatrixXcd> GwqDiag(Lcell);
	
	for (int i=0; i<Lcell; ++i)
	{
		GwqDiag[i].resize(Nw,Nqc);
	}
	
	IntervalIterator w(wmin,wmax,Nw);
	ArrayXd wvals = w.get_abscissa();
	
	for (int iq=0; iq<Nqc; ++iq)
	for (int iw=0; iw<Nw; ++iw)
	{
		MatrixXcd Gtmp(Lcell,Lcell);
		
		double qval = 2.*M_PI/Nqc * iq;
		double wval = wvals(iw);
		
		for (int i=0; i<Lcell; ++i)
		for (int j=0; j<Lcell; ++j)
		{
			Gtmp(i,j) = GwqCell[i][j](iw,iq);
		}
		
		ComplexEigenSolver<MatrixXcd> Eugen;
		Eugen.compute(Gtmp);
		VectorXcd eig = Eugen.eigenvalues();
		
		// sort according to imaginary part:
		sort(eig.data(), eig.data()+eig.size(),
		[](const complex<double> &a, const complex<double> &b) -> bool
		{
			return a.imag() > b.imag(); 
		});
		
		if (Lcell>1)
		{
			double diff = abs(eig(0)-eig(1));
			if (diff < 1e-4)
			{
				cout << "q=" << qval << ", w=" << wval << ", eig=" << eig.transpose() << ", diff=" << diff << endl;
			}
		}
		
		for (int i=0; i<Lcell; ++i)
		{
			GwqDiag[i](iw,iq) = eig(i);
		}
	}
	
	return GwqDiag;
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
save() const
{
	bool PRINT = (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::ON_EXIT)? true:false;
	
	saveMatrix(Gtx.real(), label+"_G=txRe_"+xt_info()+".dat", PRINT);
	saveMatrix(Gtx.imag(), label+"_G=txIm_"+xt_info()+".dat", PRINT);
//	saveMatrix_cpython(Gtx, label+"_G=tx_"+xt_info()+".dat", PRINT);
//	
	saveMatrix(Gwq.real(), label+"_G=ωqRe_"+xtqw_info()+".dat", PRINT);
	saveMatrix(Gwq.imag(), label+"_G=ωqIm_"+xtqw_info()+".dat", PRINT);
//	saveMatrix_cpython(Gwq, label+"_G=ωq_"+xtqw_info()+".dat", PRINT);
	
	IntervalIterator w(wmin,wmax,Nw);
	ArrayXd wvals = w.get_abscissa();
	
	save_xy(wvals, -M_1_PI*Glocw.imag(), label+"_G=QDOS_"+xt_info()+".dat", PRINT);
	save_xy(tvals, Gloct.real(), Gloct.imag(), make_string(label,"_G=t0_",xt_info(),".dat"), PRINT);
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
save_cell() const
{
	bool PRINT = (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::ON_EXIT)? true:false;
	
	for (int i=0; i<Lcell; ++i)
	for (int j=0; j<Lcell; ++j)
	{
		saveMatrix(GtxCell[i][j].real(), make_string(label,"_G=txRe_i=",i,"_j=",j,"_",xt_info(),".dat"), PRINT);
		saveMatrix(GtxCell[i][j].imag(), make_string(label,"_G=txIm_i=",i,"_j=",j,"_",xt_info(),".dat"), PRINT);
//		saveMatrix_cpython(GtxCell[i][j], make_string(label,"_G=tx_i=",i,"_j=",j,"_",xt_info(),".dat"), PRINT);
		
		saveMatrix(GwqCell[i][j].real(), make_string(label,"_G=ωqRe_i=",i,"_j=",j,"_",xtqw_info(true),".dat"), PRINT);
		saveMatrix(GwqCell[i][j].imag(), make_string(label,"_G=ωqIm_i=",i,"_j=",j,"_",xtqw_info(true),".dat"), PRINT);
//		saveMatrix_cpython(GwqCell[i][j], make_string(label,"_G=ωq_i=",i,"_j=",j,"_",xtqw_info(true),".dat"), PRINT); // BETWEEN_CELLS=true
	}
	
	for (int i=0; i<Lcell; ++i)
	{
		IntervalIterator w(wmin,wmax,Nw);
		ArrayXd wvals = w.get_abscissa();
		
		save_xy(wvals, -M_1_PI*GlocwCell[i].imag(), make_string(label,"_G=QDOS_i=",i,"_",xt_info(),".dat"), PRINT);
		save_xy(tvals, GloctCell[i].real(), GloctCell[i].imag(), make_string(label,"_G=t0_i=",i,"_",xt_info(),".dat"), PRINT);
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
diagonalize_and_save_cell() const
{
	bool PRINT = (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::ON_EXIT)? true:false;
	
	auto FTdiag = FTbands();
	
	for (int i=0; i<Lcell; ++i)
	{
		saveMatrix(FTdiag[i].real(), make_string(label,"_G=ωqdiagRe_i=",i,"_",xtqw_info(true),".dat"), PRINT);
		saveMatrix(FTdiag[i].imag(), make_string(label,"_G=ωqdiagIm_i=",i,"_",xtqw_info(true),".dat"), PRINT);
//		saveMatrix_cpython(FTdiag[i], make_string(label,"_G=ωqdiag_i=",i,"_",xtqw_info(true),".dat"), PRINT);
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
save_selfenergy (double (*eps)(double), bool SAVE_G0, double eta) const
{
	IntervalIterator w(wmin,wmax,Nw);
	ArrayXd wvals = w.get_abscissa();
	IntervalIterator q;
	
	if (Q_RANGE_CHOICE == MPI_PPI)
	{
		q = IntervalIterator(-M_PI,M_PI,Nq);
	}
	else if (Q_RANGE_CHOICE == ZERO_2PI)
	{
		q = IntervalIterator(0,2.*M_PI,Nq);
	}
	ArrayXd qvals = q.get_abscissa();
	
	MatrixXcd Swq(Nw,Nq);
	MatrixXcd G0wq(Nw,Nq);
	
	if (Q_RANGE_CHOICE == MPI_PPI)
	{
		for (int iw=0; iw<wvals.rows(); ++iw)
		for (int iq=0; iq<Nq; ++iq)
		{
			Swq(iw,iq) = wvals(iw)-eps(qvals(iq))-pow(Gwq(iw,iq),-1); // Σ(ω,q) = ω-ε(q)-1/G(ω,q)
			G0wq(iw,iq) = pow(wvals(iw)-eps(qvals(iq))+1.i*eta,-1); // G₀(ω,q) = 1/(ω-ε(q)+iη)
		}
	}
	
	bool PRINT = (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::ON_EXIT)? true:false;
	
	saveMatrix(Swq.real(), label+"_G=ΣωqRe_"+xtqw_info()+".dat", PRINT);
	saveMatrix(Swq.imag(), label+"_G=ΣωqIm_"+xtqw_info()+".dat", PRINT);
//	saveMatrix_cpython(Swq, label+"_G=Σωq_"+xtqw_info()+".dat", PRINT);
	
	if (SAVE_G0)
	{
		saveMatrix(G0wq.real(), label+"_G=ωqfreeRe_"+xtqw_info()+".dat", PRINT);
		saveMatrix(G0wq.imag(), label+"_G=ωqfreeIm_"+xtqw_info()+".dat", PRINT);
//		saveMatrix_cpython(G0wq, label+"_G=ωqfree_"+xtqw_info()+".dat", PRINT);
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
save_selfenergy_band (int i, double (*eps)(double), bool SAVE_G0, double eta) const
{
	IntervalIterator w(wmin,wmax,Nw);
	ArrayXd wvals = w.get_abscissa();
	IntervalIterator q;
	
	if (Q_RANGE_CHOICE == MPI_PPI)
	{
		q = IntervalIterator(-M_PI,M_PI,Nqc);
	}
	else if (Q_RANGE_CHOICE == ZERO_2PI)
	{
		q = IntervalIterator(0,2*M_PI,Nqc);
	}
	ArrayXd qvals = q.get_abscissa();
	
	MatrixXcd Swq(Nw,Nqc);
	MatrixXcd Gfreewq(Nw,Nqc);
	
	if (Q_RANGE_CHOICE == MPI_PPI)
	{
		for (int iw=0; iw<wvals.rows(); ++iw)
		for (int iq=0; iq<Nqc; ++iq)
		{
			Swq(iw,iq) = wvals(iw)-eps(qvals(iq))-pow(GwqCell[i][i](iw,iq),-1); // Σ(ω,q) = ω-ε(q)-1/G(ω,q)
			Gfreewq(iw,iq) = pow(wvals(iw)-eps(qvals(iq))+1.i*eta,-1); // G₀(ω,q) = 1/(ω-ε(q)+iη)
		}
	}
	
	bool PRINT = (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::ON_EXIT)? true:false;
	
	saveMatrix(Swq.real(), make_string(label,"_G=Σ",i,i,"ωqRe_",xtqw_info(),".dat"), PRINT);
	saveMatrix(Swq.imag(), make_string(label,"_G=Σ",i,i,"ωqIm_",xtqw_info(),".dat"), PRINT);
//	saveMatrix_cpython(Swq, make_string(label,"_G=Σ",i,i,"ωq_",xtqw_info(),".dat"), PRINT);
	
	if (SAVE_G0)
	{
		saveMatrix(Gfreewq.real(), make_string(label,"_G=",i,i,"ωqfreeRe_",xtqw_info(),".dat"), PRINT);
		saveMatrix(Gfreewq.imag(), make_string(label,"_G=",i,i,"ωqfreeIm_",xtqw_info(),".dat"), PRINT);
//		saveMatrix_cpython(Gfreewq, make_string(label,"_G=",i,i,"ωqfree_",xtqw_info(),".dat"), PRINT);
	}
}

#endif
