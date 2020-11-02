#ifndef GREEN_PROPAGATOR
#define GREEN_PROPAGATOR

#ifdef GREENPROPAGATOR_USE_GAUSSIAN_QUADRATURE
#include "SuperQuadrator.h"
#else
#include "Quadrator.h"
#endif

#include "solvers/TDVPPropagator.h"
#include "solvers/EntropyObserver.h"
#include "IntervalIterator.h"
#include "ComplexInterpolGSL.h"

#ifdef GREENPROPAGATOR_USE_HDF5
#include "HDF5Interface.h"
#endif

#include <unsupported/Eigen/FFT>
#include "gsl_integration.h" // from TOOLS
#include "ooura_integration.h" // from TOOLS

/**Range of k-values:
\param MPI_PPI : from -pi to +pi
\param ZERO_2PI : from 0 to +2*pi
*/
enum Q_RANGE {MPI_PPI=0, ZERO_2PI=1};
enum GREEN_INTEGRATION {DIRECT=0, INTERP=1, OOURA=2};

std::ostream& operator<< (std::ostream& s, GREEN_INTEGRATION GI)
{
	if      (GI==DIRECT) {s << "DIRECT";}
	else if (GI==INTERP) {s << "INTERP";}
	else if (GI==OOURA ) {s << "OOURA" ;}
	return s;
}

/**cutoff function for the time domain*/
struct GreenPropagatorGlobal
{
	static double tmax;
	
	static double damping (double tval) {return lorentz(tval);};
	
	static double gauss (double tval) {return exp(-pow(2.*tval/tmax,2));};
	static double lorentz (double tval) {return exp(-4.*tval/tmax);};
};
double GreenPropagatorGlobal::tmax = 20.;

/**Continuous Fourier transform using GSL interpolation*/
VectorXcd FT_interpol (const VectorXd &tvals, const VectorXcd &fvals, bool USE_QAWO, double wmin, double wmax, int wpoints)
{
	VectorXcd res(wpoints);
	
	IntervalIterator w(wmin,wmax,wpoints);
	for (w=w.begin(); w!=w.end(); ++w)
	{
		double wval = *w;
		ComplexInterpol Interp(tvals);
		
		for (int it=0; it<tvals.rows(); ++it)
		{
			double tval = tvals(it);
			complex<double> Gval = (USE_QAWO)? fvals(it) : exp(+1.i*wval*tval) * fvals(it);
			Interp.insert(it,Gval);
		}
		Interp.set_splines();
		
		if (USE_QAWO)
		{
			gsl_function_pp FpRe([=](double t)->double{return Interp.quick_evaluateRe(t);});
			gsl_function_pp FpIm([=](double t)->double{return Interp.quick_evaluateIm(t);});
			gsl_function * FRe = static_cast<gsl_function*>(&FpRe);
			gsl_function * FIm = static_cast<gsl_function*>(&FpIm);
			
			res(w.index()) = fouriergrate(*FRe, *FIm, tvals(0),tvals(tvals.rows()-1), 1e-5,1e-5, wval);
		}
		else
		{
			res(w.index()) = Interp.integrate();
		}
		Interp.kill_splines();
	}
	return res;
}

struct GreenPropagatorLog
{
	GreenPropagatorLog(){};
	
	GreenPropagatorLog (int Nt, int L, string logfolder_input)
	:logfolder(logfolder_input)
	{
		Entropy.resize(Nt,L-1); Entropy.setZero();
		DeltaS.resize(Nt,L-1); Entropy.setZero();
		VarE.resize(Nt,L); VarE.setZero();
		TwoSite.resize(Nt,L-1); TwoSite.setZero();
		dimK2avg.resize(Nt,1); dimK2avg.setZero();
		dimK1avg.resize(Nt,1); dimK1avg.setZero();
		dimK0avg.resize(Nt,1); dimK0avg.setZero();
		dur_tstep.resize(Nt,1); dur_tstep.setZero();
		taxis.resize(Nt,1); taxis.setZero();
	};
	
	void save (string label) const
	{
		#ifdef GREENPROPAGATOR_USE_HDF5
		HDF5Interface target(label+".log.h5",WRITE);
		target.save_matrix(Entropy,"Entropy","");
		target.save_matrix(DeltaS,"DeltaS","");
		target.save_matrix(VarE,"VarE","");
		target.save_matrix(TwoSite,"TwoSite","");
		target.save_matrix(dimK2avg,"dimK2avg","");
		target.save_matrix(dimK1avg,"dimK1avg","");
		target.save_matrix(dimK0avg,"dimK0avg","");
		target.save_matrix(dur_tstep,"dur_tstep","");
		target.save_matrix(taxis,"taxis","");
		target.close();
		#endif
	};
	
	string logfolder = "./";
	MatrixXd Entropy, DeltaS, VarE, TwoSite, dimK2avg, dimK1avg, dimK0avg, dur_tstep, taxis;
};

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
class GreenPropagator
{
public:
	
	GreenPropagator(){};
	
	// BASIC CONSTRUCTOR
	/**
	\param label_input : prefix for saved files (e.g. type of Green's function, Hamiltonian parameters)
	\param tmax_input : maximal propagation time
	\param Nt_input : amount of time steps; the optimal number seems to be such that the average timestep is below 0.1
	\param wmin_input : minimal frequency for Fourier transform
	\param wmax_input : maximal frequency for Fourier transform
	\param Nw_input : amount of frequency points
	\param Q_RANGE_CHOICE_input : choose the q-range (-π to π, 0 to 2π)
	\param Nq_input : amount of momentum points (Note: the first point is repeated at the output)
	\param GAUSSINT : if \p true, compute Gaussian integration weights for the cutoff function
	*/
	GreenPropagator (string label_input, 
	                 double tmax_input, int Nt_input, 
	                 int Ns_input=1, 
	                 double wmin_input=-10., double wmax_input=10., int Nw_input=501,
	                 Q_RANGE Q_RANGE_CHOICE_input=MPI_PPI, int Nq_input=501,
	                 GREEN_INTEGRATION GREENINT=OOURA)
	:label(label_input), tmax(tmax_input), Nt(Nt_input),
	 Ns(Ns_input),
	 wmin(wmin_input), wmax(wmax_input), Nw(Nw_input),
	 Q_RANGE_CHOICE(Q_RANGE_CHOICE_input), Nq(Nq_input),
	 GREENINT_CHOICE(GREENINT)
	{
		#ifndef GREENPROPAGATOR_USE_GAUSSIAN_QUADRATURE
		assert(GREENINT!=DIRECT);
		#endif
		GreenPropagatorGlobal::tmax = tmax;
		set_qlims(Q_RANGE_CHOICE);
		calc_intweights();
	}
	
	// LOADING CONSTRUCTORS FOLLOW:
	
//	// LOAD: MATRIX, NO CELL
//	/**
//	Reads G(t,x) from a matrix, so that G(ω,q) can be recalculated.
//	\param label_input : prefix for saved files (e.g. type of Green's function, Hamiltonian parameters)
//	\param Lcell_input : unit cell length
//	\param Ncells_input : number of unit cells
//	\param tmax_input : maximal propagation time
//	\param Gtx_input : input of complex G(ω,q)
//	\param Q_RANGE_CHOICE_input : choose the q-range (-π to π, 0 to 2π)
//	\param GAUSSINT : if \p true, compute Gaussian integration weights for the cutoff function
//	*/
//	GreenPropagator (string label_input, int Lcell_input, int Ncells_input, double tmax_input, const MatrixXcd &Gtx_input, 
//	                 Q_RANGE Q_RANGE_CHOICE_input=MPI_PPI, int Nq_input=501, GREEN_INTEGRATION GREENINT=OOURA)
//	:label(label_input), Lcell(Lcell_input), Ncells(Ncells_input), tmax(tmax_input), GREENINT_CHOICE(GREENINT), Q_RANGE_CHOICE(Q_RANGE_CHOICE_input)
//	{
//		Gtx = Gtx_input;
//		
//		#ifdef GREENPROPAGATOR_USE_GAUSSIAN_QUADRATURE
//		Nt = (GREENINT==DIRECT)? Gtx.rows():Gtx.rows()-1;
//		#else
//		assert(GREENINT!=DIRECT);
//		Nt = Gtx.rows()-1;
//		#endif
//		Nq = Nq_input;
//		Lhetero = Lcell*Ncells;
//		
//		make_xarrays(Lhetero,Lcell,Ncells,Ns);
//		set_qlims(Q_RANGE_CHOICE);
//		
//		for (int l=0; l<Lhetero; ++l)
//		{
//			if (xvals[l] == 0) Gloct = Gtx.col(l);
//		}
//		
//		GreenPropagatorGlobal::tmax = tmax;
//		calc_intweights();
//	}
	
	// LOAD: MATRIX, CELL
	/**
	Reads G(t,x) with unit cell resolution from array input, so that G(ω,q) can be recalculated.
	\param label_input : prefix for saved files (e.g. type of Green's function)
	\param tmax_input : maximal propagation time
	\param Gtx_input : input of complex G(ω,q)
	\param Q_RANGE_CHOICE_input : choose the q-range (-π to π, 0 to 2π)
	\param Nq_input : amount of momentum points (Note: the first point is repeated at the output)
	\param GAUSSINT : if \p true, compute Gaussian integration weights for the cutoff function
	*/
	GreenPropagator (string label_input, int Lcell_input, int Ncells_input, int Ns_input, double tmax_input, const vector<vector<MatrixXcd>> &Gtx_input, 
	                 Q_RANGE Q_RANGE_CHOICE_input=MPI_PPI, int Nq_input=501, GREEN_INTEGRATION GREENINT=OOURA)
	:label(label_input), Lcell(Lcell_input), Ncells(Ncells_input), Ns(Ns_input), tmax(tmax_input), GREENINT_CHOICE(GREENINT), Q_RANGE_CHOICE(Q_RANGE_CHOICE_input)
	{
		GtxCell.resize(Lcell); for (int i=0; i<Lcell; ++i) GtxCell[i].resize(Lcell);
		
		for (int i=0; i<Lcell; ++i)
		for (int j=0; j<Lcell; ++j)
		{
			GtxCell[i][j] = Gtx_input[i][j];
		}
		
		#ifdef GREENPROPAGATOR_USE_GAUSSIAN_QUADRATURE
		Nt = (GREENINT==DIRECT)? GtxCell[0][0].rows():GtxCell[0][0].rows()-1;
		#else
		assert(GREENINT!=DIRECT);
		Nt = GtxCell[0][0].rows()-1;
		#endif
		Ncells = GtxCell[0][0].cols();
		Nq = Nq_input;
		Lhetero = Ncells*Lcell;
		
		make_xarrays(Lhetero,Lcell,Ncells,Ns);
		set_qlims(Q_RANGE_CHOICE);
		
		GloctCell.resize(Lcell);
		for (int i=0; i<Lcell; ++i)
		for (int n=0; n<Ncells; ++n)
		{
			if (dcell[n*Lcell] == 0) GloctCell[i] = GtxCell[i][i].col(n);
		}
		
		GreenPropagatorGlobal::tmax = tmax;
		calc_intweights();
		if (CHOSEN_VERBOSITY>DMRG::VERBOSITY::SILENT) lout << termcolor::green << "loading " << label_input << " successful!" << termcolor::reset << endl;
	}
	
//	// LOAD: HDF5, NO CELL
//	/**
//	Reads G(t,x) with unit cell resolution from file, so that G(ω,q) can be recalculated.
//	\param label_input : prefix for saved files (e.g. type of Green's function)
//	\param Lcell_input : unit cell length
//	\param tmax_input : maximal propagation time
//	\param files : input vector of files, a sum is performed over all data
//	\param Q_RANGE_CHOICE_input : choose the q-range (-π to π, 0 to 2π)
//	\param Nq_input : amount of momentum points (Note: the first point is repeated at the output)
//	\param GAUSSINT : if \p true, compute Gaussian integration weights for the cutoff function
//	*/
//	GreenPropagator (string label_input, int Lcell_input, double tmax_input, const vector<string> &files, 
//	                 Q_RANGE Q_RANGE_CHOICE_input=MPI_PPI, int Nq_input=501, GREEN_INTEGRATION GREENINT=OOURA)
//	:label(label_input), Lcell(Lcell_input), tmax(tmax_input), GREENINT_CHOICE(GREENINT), Q_RANGE_CHOICE(Q_RANGE_CHOICE_input)
//	{
//		#ifdef GREENPROPAGATOR_USE_HDF5
//		for (const auto &file:files)
//		{
//			MatrixXd MtmpRe, MtmpIm;
//			HDF5Interface Reader(file+".h5",READ);
//			Reader.load_matrix(MtmpRe,"txRe","G");
//			Reader.load_matrix(MtmpIm,"txIm","G");
//			Reader.close();
//			
//			if (Gtx.size() == 0)
//			{
//				Gtx = MtmpRe+1.i*MtmpIm;
//			}
//			else
//			{
//				Gtx += MtmpRe+1.i*MtmpIm;
//			}
//		}
//		#endif
//		
//		Ncells = 1;
//		#ifdef GREENPROPAGATOR_USE_GAUSSIAN_QUADRATURE
//		Nt = (GREENINT==DIRECT)? Gtx.rows():Gtx.rows()-1;
//		#else
//		assert(GREENINT!=DIRECT);
//		Nt = Gtx.rows()-1;
//		#endif
//		Nq = Nq_input;
//		Lhetero = Lcell;
//		
//		make_xarrays(Lhetero,Lcell,Ncells,Ns);
//		set_qlims(Q_RANGE_CHOICE);
//		
//		for (int l=0; l<Lhetero; ++l)
//		{
//			if (xvals[l] == 0) Gloct = Gtx.col(l);
//		}
//		
//		GreenPropagatorGlobal::tmax = tmax;
//		calc_intweights();
//	}
	
	// LOAD: HDF5, CELL
	/**
	Reads G(t,x) with unit cell resolution from file, so that G(ω,q) can be recalculated.
	\param label_input : prefix for saved files (e.g. type of Green's function)
	\param Lcell_input : unit cell length
	\param Ncells_input : number of unit cells
	\param tmax_input : maximal propagation time
	\param files : input vector of files, a sum is performed over all data
	\param Q_RANGE_CHOICE_input : choose the q-range (-π to π, 0 to 2π)
	\param Nq_input : amount of momentum points (Note: the first point is repeated at the output)
	\param GAUSSINT : if \p true, compute Gaussian integration weights for the cutoff function
	*/
	GreenPropagator (string label_input, int Lcell_input, int Ncells_input, int Ns_input, double tmax_input, const vector<string> &files, 
	                 Q_RANGE Q_RANGE_CHOICE_input=MPI_PPI, int Nq_input=501, GREEN_INTEGRATION GREENINT=OOURA)
	:label(label_input), Lcell(Lcell_input), Ncells(Ncells_input), Ns(Ns_input), tmax(tmax_input), GREENINT_CHOICE(GREENINT), Q_RANGE_CHOICE(Q_RANGE_CHOICE_input)
	{
		GtxCell.resize(Lcell); for (int i=0; i<Lcell; ++i) {GtxCell[i].resize(Lcell);}
		#ifdef GREENPROPAGATOR_USE_HDF5
		for (const auto &file:files)
		for (int i=0; i<Lcell; ++i)
		for (int j=0; j<Lcell; ++j)
		{
			MatrixXd MtmpRe, MtmpIm;
			HDF5Interface Reader(file+".h5",READ);
			Reader.load_matrix(MtmpRe,"txRe",make_string("G",i,j));
			Reader.load_matrix(MtmpIm,"txIm",make_string("G",i,j));
			Reader.close();
			
			if (GtxCell[i][j].size() == 0)
			{
				GtxCell[i][j] = MtmpRe+1.i*MtmpIm;
			}
			else
			{
				GtxCell[i][j] += MtmpRe+1.i*MtmpIm;
			}
		}
		#endif
		
		#ifdef GREENPROPAGATOR_USE_GAUSSIAN_QUADRATURE
		Nt = (GREENINT==DIRECT)? GtxCell[0][0].rows():GtxCell[0][0].rows()-1;
		#else
		assert(GREENINT!=DIRECT);
		Nt = GtxCell[0][0].rows()-1;
		#endif
		Ncells = GtxCell[0][0].cols();
		Nq = Nq_input;
		Lhetero = Ncells*Lcell;
		
		make_xarrays(Lhetero,Lcell,Ncells,Ns);
		set_qlims(Q_RANGE_CHOICE);
		
		GloctCell.resize(Lcell);
		for (int i=0; i<Lcell; ++i)
		for (int n=0; n<Ncells; ++n)
		{
			if (dcellFT[n*Lcell] == 0) GloctCell[i] = GtxCell[i][i].col(n);
		}
		
		GreenPropagatorGlobal::tmax = tmax;
		calc_intweights();
		if (CHOSEN_VERBOSITY>DMRG::VERBOSITY::SILENT) lout << termcolor::green << "loading " << label_input << " successful!" << termcolor::reset << endl;
	}
	
	/**
	Computes the Green's function G(t,x).
	\param H : Hamiltonian of heterogenic section
	\param OxPhi : vector with all local excitations
	\param OxPhi0 : starting state where the local excitation is located
	\param Eg : ground-state energy
	\param TIME_FORWARDS : For photoemission, set to \p false. For inverse photoemission, set to \p true.
	*/
//	void compute (const Hamiltonian &H, const vector<Mps<Symmetry,complex<double>>> &OxPhi, Mps<Symmetry,complex<double>> &OxPhi0, 
//	              double Eg, bool TIME_FORWARDS = true);
	
	/**
	Computes the Green's function G(t,x) using counterpropagations forward and backward in time on the unit cell.
	Optimal when run with 2*Lcell threads.
	\param H : Hamiltonian of heterogenic section
	\param OxPhi : vector with all local excitations
	\param Eg : ground-state energy
	\param TIME_FORWARDS : For photoemission, set to \p false. For inverse photoemission, set to \p true.
	\param COUNTERPROPAGATE : If \p true, use the more efficient propagations forwards and backwards in time (not for finite MPS)
	*/
	void compute_cell (const Hamiltonian &H, const vector<Mps<Symmetry,complex<double>>> &OxPhi, double Eg, 
	                   bool TIME_FORWARDS = true, bool COUNTERPROPAGATE = true);
	
//	void compute_thermal (const Hamiltonian &H, const vector<Mpo<Symmetry,MpoScalar>> &Odag, const Mps<Symmetry,complex<double>> &Phi, 
//	                      Mps<Symmetry,complex<double>> &OxPhi0, bool TIME_FORWARDS);
	void compute_thermal_cell (const Hamiltonian &H, const vector<Mpo<Symmetry,MpoScalar>> &Odag, const Mps<Symmetry,complex<double>> &Phi, 
	                           vector<Mps<Symmetry,complex<double>>> &OxPhi0, bool TIME_FORWARDS);
	
	/**
	Recalculates the t→ω Fourier transform for a different ω-range
	\param wmin_new : minimal frequency for Fourier transform
	\param wmax_new : maximal frequency for Fourier transform
	\param Nw_new : amount of frequency points
	*/
//	void recalc_FTw (double wmin_new, double wmax_new, int Nw_new=1000);
	
	void recalc_FTwCell (double wmin_new, double wmax_new, int Nw_new=1000, bool TRANSPOSE=false);
	
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
	void save (bool IGNORE_TX = false) const;
	
	/**Integrates the QDOS up to a given chemical potential μ (or the Fermi energy, since T=0). 
	   Can be used to find the right μ which gives the chosen filling n.
	\param mu : chemical potential, upper integration limit
	\param Nint : amount of Gauss-Legendre integration points
	*/
//	double integrate_Glocw (double mu, int Nint = 1000);
	
	/**Integrates the QDOS up to a given chemical potential μ for the unit cell. 
	   The total integral is normalized \f$\sum_{i \in cell} n_i\f$.
	\param mu : chemical potential, upper integration limit
	\param Nint : amount of Gauss-Legendre integration points
	*/
	double integrate_Glocw_cell (double mu, int Nint = 1000);
	
//	inline MatrixXcd get_Gtx() const {return Gtx;}
	
	inline vector<vector<MatrixXcd>> get_GtxCell() const {return GtxCell;}
	
	inline void set_verbosity (DMRG::VERBOSITY::OPTION VERBOSITY) {CHOSEN_VERBOSITY = VERBOSITY;};
	
	/**
	Fourier transform G(ω,x)→G(ω,q) when system is supposed to be translationally invariant despite a unit cell.
	\param TW_FIRST_XQ_SECOND: if true, transform first t->ω, then x->q (usually faster because Nx<<Nq)
	*/
	void FT_allSites (bool TW_FIRST_XQ_SECOND = true);
	
	ArrayXcd FTloc_tw (const VectorXcd &Gloct, const ArrayXd &wvals);
	
	inline void set_OxPhiFull (const vector<Mps<Symmetry,complex<double>>> &OxPhiFull_input) {OxPhiFull = OxPhiFull_input;}
	
	inline void set_Qmulti (int NQ_input) {NQ = NQ_input;}
	
	string xinfo() const;
	string qinfo() const;
	string tinfo() const;
	string winfo() const; // w=ω
	string xt_info() const;
	string xtqw_info() const;
	void print_starttext() const;
	
	ArrayXd ncell;
	double mu = std::nan("0");
	
	void set_tol_DeltaS (double x) {tol_DeltaS = x;};
	void set_lim_Nsv (size_t x)    {lim_Nsv    = x;};
	void set_tol_compr (double x)  {tol_compr  = x;};
	void set_h_ooura (double x)    {h_ooura  = x;};
	void set_log (string logfolder_input="./") {SAVE_LOG = true; logfolder = logfolder_input;}
	void set_dLphys (int x) {dLphys = x;};
	bool USE_QAWO = false;
	
private:
	
	string label;
	
	int Nt, Nw, Nq;
	int NQ = 0;
	double tmax, wmin, wmax, qmin, qmax;
	Q_RANGE Q_RANGE_CHOICE = MPI_PPI;
	int Lhetero, Lcell, Ncells, Ns;
	int dLphys = 1;
	
	double tol_compr = 1e-4; // compression tolerance during time propagation
	double tol_Lanczos = 1e-7; // 1e-6 seems sufficient; increase to 1e-8 for higher accuracy
	double tol_DeltaS = 1e-3; // 1e-3 seems good for DIRECT (initially small timesteps); 1e-2 seems good for INTERP (equidistant timesteps)
	size_t lim_Nsv = 500ul;
	double h_ooura = 0.001; // stepsize for Ooura integration if GREENINT_CHOICE == OOURA; smaller = more accurate & slower
	GREEN_INTEGRATION GREENINT_CHOICE = OOURA;
	bool SAVE_LOG = false;
	
	DMRG::VERBOSITY::OPTION CHOSEN_VERBOSITY = DMRG::VERBOSITY::HALFSWEEPWISE;
	
	#ifdef GREENPROPAGATOR_USE_GAUSSIAN_QUADRATURE
	SuperQuadrator<GAUSS_LEGENDRE> TimeIntegrator;
	#endif
	
	ArrayXd tvals, weights, tsteps;
	bool TIME_FORWARDS;
	
	vector<double> xvals; // site labels: relative distance from excitation centre in sites
	vector<int> xinds; // site indices from 0 to Lhetero-1
	vector<int> dcell; // relative distance from excitation centre in unit cells
	vector<int> icell; // site indices within unit cell
	vector<int> dcellFT;
	
	vector<Mps<Symmetry,complex<double>>> OxPhiFull;
	
	MatrixXcd Gtx, Gwx, Gtq, Gwq;
	VectorXcd Gloct, Glocw, G0q, G0x;
	
	// SU(2) Qmultitarget
	vector<MatrixXcd> GtxQmulti, GtqQmulti, GwqQmulti;
	vector<VectorXcd> GloctQmulti, GlocwQmulti, G0qQmulti;
	
	vector<vector<MatrixXcd>> GtxCell, GwxCell, GtqCell, GwqCell;
	vector<vector<VectorXcd>> G0qCell, G0xCell;
	vector<VectorXcd> GloctCell, GlocwCell;
	
	void calc_Green (const int &tindex, const complex<double> &phase, 
	                 const vector<Mps<Symmetry,complex<double>>> &OxPhi, const Mps<Symmetry,complex<double>> &Psi);
	void calc_GreenCell (const int &tindex, const complex<double> &phase, 
	                     const vector<Mps<Symmetry,complex<double>>> &OxPhi, const vector<Mps<Symmetry,complex<double>>> &Psi);
	void calc_GreenCell (const int &tindex, const complex<double> &phase, 
	                     const std::array<vector<Mps<Symmetry,complex<double>>>,2> &Psi);
	
	void calc_Green_thermal (const int &tindex, const vector<Mpo<Symmetry,MpoScalar>> &Ox, 
	                         const Mps<Symmetry,complex<double>> &Phi, const Mps<Symmetry,complex<double>> &Psi);
	void calc_GreenCell_thermal (const int &tindex, const vector<Mpo<Symmetry,MpoScalar>> &Ox, const vector<Mps<Symmetry,complex<double>>> &Psi);
	
	void calc_intweights();
	void make_xarrays (int Lhetero_input, int Lcell_input, int Ncells_input, int Ns=1, bool PRINT=false);
	void set_qlims (Q_RANGE CHOICE);
	
//	void propagate (const Hamiltonian &H, const vector<Mps<Symmetry,complex<double>>> &OxPhi, Mps<Symmetry,complex<double>> &OxPhi0, 
//	                double Eg, bool TIME_FORWARDS);
	void propagate_cell (const Hamiltonian &H, const vector<Mps<Symmetry,complex<double>>> &OxPhi, double Eg, bool TIME_FORWARDS=true);
	void counterpropagate_cell (const Hamiltonian &H, const vector<Mps<Symmetry,complex<double>>> &OxPhi, double Eg, bool TIME_FORWARDS=true);
	
	void propagate_thermal (const Hamiltonian &H, const vector<Mpo<Symmetry,MpoScalar>> &Ox, Mps<Symmetry,complex<double>> Phi, 
	                        Mps<Symmetry,complex<double>> &OxPhi0, bool TIME_FORWARDS);
	void propagate_thermal_cell (const Hamiltonian &H, const vector<Mpo<Symmetry,MpoScalar>> &Ox, Mps<Symmetry,complex<double>> Phi, 
	                             vector<Mps<Symmetry,complex<double>>> &OxPhi0, bool TIME_FORWARDS);
	
//	void FT_xq (const MatrixXcd &Gtx, MatrixXcd &Gtq);
	void FTcell_xq();
//	void FT_tw (const MatrixXcd &Gtq, MatrixXcd &Gwq, VectorXcd &G0q, VectorXcd &Glocw);
	void FTcell_tw();
	
	void FTcellww_xq (bool TRANSPOSE = false);
	void FTcellxx_tw();
//	void FTww_xq (const MatrixXcd &Gwx, const MatrixXcd &G0x, MatrixXcd &Gwq, VectorXcd &G0q);
//	void FTxx_tw (const MatrixXcd &Gtx, MatrixXcd &Gwx, VectorXcd &G0x, VectorXcd &Glocw);
	
	vector<Mpo<Symmetry,MpoScalar>> Measure;
	void measure_wavepacket (const Mps<Symmetry,complex<double>> &Psi, double tval, string info="");
	int measure_interval;
	string measure_name, measure_subfolder;
	
	GreenPropagatorLog log;
	string logfolder = "./";
	void save_log (int i, int tindex, double tval, const Mps<Symmetry,complex<double>> &Psi, 
	               const TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,Mps<Symmetry,complex<double>>> &TDVP, 
	               const EntropyObserver<Mps<Symmetry,complex<double>>> &Sobs, 
	               const vector<bool> &TWO_SITE);
};

//template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
//void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
//compute (const Hamiltonian &H, const vector<Mps<Symmetry,complex<double>>> &OxPhi,
//         Mps<Symmetry,complex<double>> &OxPhi0, double Eg, bool TIME_FORWARDS)
//{
//	print_starttext();
//	
//	Lcell = max(OxPhi0.Boundaries.length(),1ul);
//	Lhetero = H.length();
//	Ncells = Lhetero/Lcell;
////	cout << "Lcell=" << Lcell << ", Lhetero=" << Lhetero << ", Ncells=" << Ncells << endl;
//	
//	if (Q_RANGE_CHOICE == MPI_PPI) assert(Lhetero%2 == 0 and "Please use an even number of sites in the heterogenic region!");
//	assert(Lhetero%Lcell == 0 and "The heterogenic region is not commensurable with the length of the unit cell!");
//	
//	if (Q_RANGE_CHOICE == MPI_PPI) assert(Ncells%2 == 0 and "Please use an even number of unit cells!");
//	
//	make_xarrays(Lhetero,Lcell,Ncells,Ns);
////	calc_intweights();
//	
//	propagate(H, OxPhi, OxPhi0, Eg, TIME_FORWARDS);
//	
//	if (NQ == 0)
//	{
//		FT_xq(Gtx,Gtq);
//		FT_tw(Gtq,Gwq,G0q,Glocw);
//	}
//	else
//	{
//		for (int iQ=0; iQ<NQ; ++iQ)
//		{
//			GtqQmulti.resize(NQ);
//			FT_xq(GtxQmulti[iQ],GtqQmulti[iQ]);
//			GwqQmulti.resize(NQ);
//			G0qQmulti.resize(NQ);
//			GlocwQmulti.resize(NQ);
//			FT_tw(GtqQmulti[iQ],GwqQmulti[iQ],G0qQmulti[iQ],GlocwQmulti[iQ]);
//		}
//	}
//}

//template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
//void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
//propagate (const Hamiltonian &H, const vector<Mps<Symmetry,complex<double>>> &OxPhi, Mps<Symmetry,complex<double>> &OxPhi0, double Eg, bool TIME_FORWARDS)
//{
//	double tsign = (TIME_FORWARDS==true)? -1.:+1.;
//	
//	Mps<Symmetry,complex<double>> Psi = OxPhi0;
//	Psi.eps_svd = tol_compr;
//	Psi.max_Nsv = max(Psi.calc_Mmax(),500ul);
//	
//	TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,Mps<Symmetry,complex<double>>> TDVP(H, Psi);
//	EntropyObserver<Mps<Symmetry,complex<double>>> Sobs(Lhetero, Nt, CHOSEN_VERBOSITY, tol_DeltaS);
//	vector<bool> TWO_SITE = Sobs.TWO_SITE(0, Psi, 1.);
//	if (CHOSEN_VERBOSITY > DMRG::VERBOSITY::ON_EXIT) lout << endl;
//	
//	Gtx.resize(Nt,Lhetero); Gtx.setZero();
//	Gloct.resize(Nt); Gloct.setZero();
//	
//	if (NQ>0)
//	{
//		GtxQmulti.resize(NQ);
//		for (int i=0; i<NQ; ++i) {GtxQmulti[i].resize(Nt,Lhetero); GtxQmulti[i].setZero();}
//		GloctQmulti.resize(NQ);
//		for (int i=0; i<NQ; ++i) {GloctQmulti[i].resize(Nt); GloctQmulti[i].setZero();}
//	}
//	
//	IntervalIterator t(0.,tmax,Nt);
//	double tval = 0.;
//	
//	if (SAVE_LOG) log = GreenPropagatorLog(Nt,Lhetero,logfolder);
//	
//	// measure wavepacket at t=0
//	measure_wavepacket(Psi,0);
//	
//	Stopwatch<> TpropTimer;
//	for (t=t.begin(); t!=t.end(); ++t)
//	{
//		Stopwatch<> StepTimer;
//		// 1. propagate
//		//----------------------------------------------------------------------------------------
//		if (tsteps(t.index()) != 0.)
//		TDVP.t_step_adaptive(H, Psi, 1.i*tsign*tsteps(t.index()), TWO_SITE, 1,tol_Lanczos);
//		//----------------------------------------------------------------------------------------
//		tval = tsteps.head(t.index()+1).sum();
//		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE) lout << StepTimer.info("time step") << endl;
//		
//		if (Psi.get_truncWeight().sum() > 0.5*tol_compr)
//		{
//			Psi.max_Nsv = min(static_cast<size_t>(max(Psi.max_Nsv*1.1, Psi.max_Nsv+1.)),lim_Nsv);
//			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
//			{
//				lout << termcolor::yellow << "Setting Psi.max_Nsv to " << Psi.max_Nsv << termcolor::reset << endl;
//			}
//		}
//		
//		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
//		{
//			lout << TDVP.info() << endl;
//			lout << Psi.info() << endl;
//			lout << "propagated to: t=" << tval << ", stepsize=" << tsteps(t.index()) << ", step#" << t.index()+1 << "/" << Nt << endl;
//		}
//		
//		// 2. measure
//		// 2.1. Green's function
//		calc_Green(t.index(), -1.i*exp(-1.i*tsign*Eg*tval), OxPhi, Psi);
//		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE) lout << StepTimer.info("G(t,x) calculation") << endl;
//		
//		// 2.2. measure wavepacket at t
//		if (Measure.size() != 0)
//		{
//			if ((t.index()-1)%measure_interval == 0 and t.index() > 1) measure_wavepacket(Psi,tval);
//		}
//		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE) lout << StepTimer.info("wavepacket measurement") << endl;
//		
//		// 3. check entropy increase
//		auto PsiTmp = Psi; PsiTmp.entropy_skim();
//		double r = (t.index()==0)? 1.:tsteps(t.index()-1)/tsteps(t.index());
//		// If INTERP is used, G(t=0) is used in the first step and t.index()==1 is the first real propagation step.
//		// Force 2-site here algorithm for better accuracay:
//		vector<size_t> true_overrides;
//		if (GREENINT_CHOICE != DIRECT and t.index() == 1)
//		{
//			true_overrides.resize(Lhetero-1);
//			iota(true_overrides.begin(), true_overrides.end(), 0);
//		}
//		TWO_SITE = Sobs.TWO_SITE(t.index(), PsiTmp, r, true_overrides);
//		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE) lout << StepTimer.info("entropy calculation") << endl;
//		
//		save_log(0,t.index(),tval,PsiTmp,TDVP,Sobs,TWO_SITE);
//		
//		// final info
//		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
//		{
//			lout << TpropTimer.info("total running time",false) << endl;
//			lout << endl;
//		}
//	}
//	
//	// measure wavepacket at t=t_end
//	if (Measure.size() != 0)
//	{
//		measure_wavepacket(Psi,tval);
//	}
//}

//template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
//void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
//compute_thermal (const Hamiltonian &H, const vector<Mpo<Symmetry,MpoScalar>> &Odag, const Mps<Symmetry,complex<double>> &Phi, 
//                 Mps<Symmetry,complex<double>> &OxPhi0, bool TIME_FORWARDS)
//{
//	print_starttext();
//	assert(H.volume()/H.length()==1 or H.volume()/H.length()==2);
//	
//	Lcell = max(OxPhi0.Boundaries.length(),1ul);
//	dLphys = (H.volume()/H.length()==2)? 1:2;
//	Lhetero = (dLphys==1)? H.length():H.length()/2;
//	Ncells = Lhetero/Lcell;
////	cout << "Lcell=" << Lcell << ", Lhetero=" << Lhetero << ", Ncells=" << Ncells << ", dLphys=" << dLphys << endl;
//	
//	if (Q_RANGE_CHOICE == MPI_PPI) assert(Lhetero%2 == 0 and "Please use an even number of sites in the heterogenic region!");
//	assert(Lhetero%Lcell == 0 and "The heterogenic region is not commensurable with the length of the unit cell!");
//	
//	if (Q_RANGE_CHOICE == MPI_PPI) assert(Ncells%2 == 0 and "Please use an even number of unit cells!");
//	
//	make_xarrays(Lhetero,Lcell,Ncells,Ns);
////	calc_intweights();
//	
//	propagate_thermal(H, Odag, Phi, OxPhi0, TIME_FORWARDS);
//	
//	if (NQ == 0)
//	{
//		FT_xq(Gtx,Gtq);
//		FT_tw(Gtq,Gwq,G0q,Glocw);
//	}
//	else
//	{
//		for (int iQ=0; iQ<NQ; ++iQ)
//		{
//			GtqQmulti.resize(NQ);
//			FT_xq(GtxQmulti[iQ],GtqQmulti[iQ]);
//			GwqQmulti.resize(NQ);
//			G0qQmulti.resize(NQ);
//			GlocwQmulti.resize(NQ);
//			FT_tw(GtqQmulti[iQ],GwqQmulti[iQ],G0qQmulti[iQ],GlocwQmulti[iQ]);
//		}
//	}
//}

//template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
//void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
//propagate_thermal (const Hamiltonian &H, const vector<Mpo<Symmetry,MpoScalar>> &Odag, Mps<Symmetry,complex<double>> Phi, 
//                   Mps<Symmetry,complex<double>> &OxPhi0, bool TIME_FORWARDS)
//{
//	double tsign = (TIME_FORWARDS==true)? -1.:+1.;
//	
//	Mps<Symmetry,complex<double>> Psi = OxPhi0;
//	Psi.eps_svd = tol_compr;
//	Psi.max_Nsv = max(Psi.calc_Mmax(),min(500ul,lim_Nsv));
//	
//	TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,Mps<Symmetry,complex<double>>> TDVPket(H,Psi);
//	EntropyObserver<Mps<Symmetry,complex<double>>> SobsKet(H.length(), Nt, CHOSEN_VERBOSITY, tol_DeltaS);
//	vector<bool> TWO_SITE_KET = SobsKet.TWO_SITE(0, Psi, 1.);
//	
//	TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,Mps<Symmetry,complex<double>>> TDVPbra(H,Phi);
//	EntropyObserver<Mps<Symmetry,complex<double>>> SobsBra(H.length(), Nt, CHOSEN_VERBOSITY, tol_DeltaS);
//	vector<bool> TWO_SITE_BRA = SobsBra.TWO_SITE(0, Phi, 1.);
//	
//	if (CHOSEN_VERBOSITY > DMRG::VERBOSITY::ON_EXIT) lout << endl;
//	
//	Gtx.resize(Nt,Lhetero); Gtx.setZero();
//	Gloct.resize(Nt); Gloct.setZero();
//	
//	if (NQ>0)
//	{
//		GtxQmulti.resize(NQ);
//		for (int i=0; i<NQ; ++i) {GtxQmulti[i].resize(Nt,Lhetero); GtxQmulti[i].setZero();}
//		GloctQmulti.resize(NQ);
//		for (int i=0; i<NQ; ++i) {GloctQmulti[i].resize(Nt); GloctQmulti[i].setZero();}
//	}
//	
//	IntervalIterator t(0.,tmax,Nt);
//	double tval = 0.;
//	
//	if (SAVE_LOG) log = GreenPropagatorLog(Nt,Lhetero,logfolder);
//	
//	// measure wavepacket at t=0
//	measure_wavepacket(Psi,0);
//	
//	Stopwatch<> TpropTimer;
//	for (t=t.begin(); t!=t.end(); ++t)
//	{
//		Stopwatch<> StepTimer;
//		// 1. propagate
//		//----------------------------------------------------------------------------------------
//		if (tsteps(t.index()) != 0.)
//		{
//			#pragma omp parallel sections
//			{
//				#pragma omp section
//				{
//					TDVPket.t_step_adaptive(H, Psi, 1.i*tsign*tsteps(t.index()), TWO_SITE_KET, 1,tol_Lanczos);
//				}
//				#pragma omp section
//				{
//					// Note: time direction will flipped via taking the adjoint
//					TDVPbra.t_step_adaptive(H, Phi, 1.i*tsign*tsteps(t.index()), TWO_SITE_BRA, 1,tol_Lanczos);
//				}
//			}
//		}
//		//----------------------------------------------------------------------------------------
//		tval = tsteps.head(t.index()+1).sum();
//		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE) lout << StepTimer.info("time step") << endl;
//		
//		// 2.1 increase bond dimension of Psi
//		if (Psi.get_truncWeight().sum() > 0.5*tol_compr)
//		{
//			Psi.max_Nsv = min(static_cast<size_t>(max(Psi.max_Nsv*1.1, Psi.max_Nsv+1.)),lim_Nsv);
//			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
//			{
//				lout << termcolor::yellow << "Setting Psi.max_Nsv to " << Psi.max_Nsv << termcolor::reset << endl;
//			}
//		}
//		
//		// 2.2 increase bond dimension of Phi
//		if (Phi.get_truncWeight().sum() > 0.5*tol_compr)
//		{
//			Phi.max_Nsv = min(static_cast<size_t>(max(Phi.max_Nsv*1.1, Phi.max_Nsv+1.)),lim_Nsv);
//			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
//			{
//				lout << termcolor::yellow << "Setting Phi.max_Nsv to " << Phi.max_Nsv << termcolor::reset << endl;
//			}
//		}
//		
//		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
//		{
//			lout << "KET: " << TDVPket.info() << endl;
//			lout << "KET: " << Psi.info() << endl;
//			lout << "BRA: " << TDVPbra.info() << endl;
//			lout << "BRA: " << Phi.info() << endl;
//			lout << "propagated to: t=" << tval << ", stepsize=" << tsteps(t.index()) << ", step#" << t.index()+1 << "/" << Nt << endl;
//		}
//		
//		// 3. measure
//		// 3.1. Green's function
//		calc_Green_thermal(t.index(), Odag, Phi, Psi);
//		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE) lout << StepTimer.info("G(t,x) calculation") << endl;
//		
//		// 3.2. measure wavepacket at t
//		if (Measure.size() != 0)
//		{
//			if ((t.index()-1)%measure_interval == 0 and t.index() > 1) measure_wavepacket(Psi,tval);
//		}
//		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE) lout << StepTimer.info("wavepacket measurement") << endl;
//		
//		// 4.1. check entropy increase of Psi
//		auto PsiTmp = Psi; PsiTmp.entropy_skim();
//		double r = (t.index()==0)? 1.:tsteps(t.index()-1)/tsteps(t.index());
//		// If INTERP is used, G(t=0) is used in the first step and t.index()==1 is the first real propagation step.
//		// Force 2-site here algorithm for better accuracay:
//		vector<size_t> true_overrides_ket;
//		if (GREENINT_CHOICE != DIRECT and t.index() == 1)
//		{
//			true_overrides_ket.resize(H.length()-1);
//			iota(true_overrides_ket.begin(), true_overrides_ket.end(), 0);
//		}
//		TWO_SITE_KET = SobsKet.TWO_SITE(t.index(), PsiTmp, r, true_overrides_ket);
//		
//		// 4.1. check entropy increase of Phi
//		auto PhiTmp = Phi; PhiTmp.entropy_skim();
//		r = (t.index()==0)? 1.:tsteps(t.index()-1)/tsteps(t.index());
//		// If INTERP is used, G(t=0) is used in the first step and t.index()==1 is the first real propagation step.
//		// Force 2-site here algorithm for better accuracay:
//		vector<size_t> true_overrides_bra;
//		if (GREENINT_CHOICE != DIRECT and t.index() == 1)
//		{
//			true_overrides_bra.resize(H.length()-1);
//			iota(true_overrides_bra.begin(), true_overrides_bra.end(), 0);
//		}
//		TWO_SITE_BRA = SobsBra.TWO_SITE(t.index(), PhiTmp, r, true_overrides_bra);
//		
//		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE) lout << StepTimer.info("entropy calculation") << endl;
//		save_log(0,t.index(),tval,PsiTmp,TDVPket,SobsKet,TWO_SITE_KET);
//		
//		// final info
//		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
//		{
//			lout << TpropTimer.info("total running time",false) << endl;
//			lout << endl;
//		}
//	}
//	
//	// measure wavepacket at t=t_end
//	if (Measure.size() != 0)
//	{
//		measure_wavepacket(Psi,tval);
//	}
//}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
compute_cell (const Hamiltonian &H, const vector<Mps<Symmetry,complex<double>>> &OxPhi, double Eg, bool TIME_FORWARDS, bool COUNTERPROPAGATE)
{
	print_starttext();
	
	Lcell = OxPhi.size();
	Lhetero = H.length();
	Ncells = Lhetero/Lcell;
	
	assert(Lhetero%Lcell == 0);
	
//	calc_intweights();
	make_xarrays(Lhetero,Lcell,Ncells,Ns);
	
	GtxCell.resize(Lcell);
	for (int i=0; i<Lcell; ++i)
	{
		GtxCell[i].resize(Lcell);
	}
	for (int i=0; i<Lcell; ++i)
	for (int j=0; j<Lcell; ++j)
	{
		GtxCell[i][j].resize(Nt,Ncells);
		GtxCell[i][j].setZero();
	}
	
	GloctCell.resize(Lcell);
	for (int i=0; i<Lcell; ++i)
	{
		GloctCell[i].resize(Nt);
		GloctCell[i].setZero();
	}
	
	if (OxPhi[0].Boundaries.IS_TRIVIAL() or COUNTERPROPAGATE == false)
	{
		propagate_cell(H, OxPhi, Eg, TIME_FORWARDS);
	}
	else
	{
		#pragma omp critical
		{
			lout << termcolor::blue << label << ": using counterpropagation algorithm, best with 2*Lcell=" << 2*Lcell << " threads" << termcolor::reset << endl;
			#ifdef _OPENMP
			lout << termcolor::blue << "current OMP threads=" << omp_get_max_threads() << termcolor::reset << endl;
			#endif
		}
		counterpropagate_cell(H, OxPhi, Eg, TIME_FORWARDS);
	}
	
//	FTcell_xq();
//	FTcell_tw();
	FTcellxx_tw();
	FTcellww_xq((TIME_FORWARDS)?false:true);
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
propagate_cell (const Hamiltonian &H, const vector<Mps<Symmetry,complex<double>>> &OxPhi, double Eg, bool TIME_FORWARDS)
{
	double tsign = (TIME_FORWARDS==true)? -1.:+1.;
	
	vector<Mps<Symmetry,complex<double>>> Psi(Lcell);
	for (int i=0; i<Lcell; ++i)
	{
		Psi[i] = OxPhi[i];
		Psi[i].eps_svd = tol_compr;
		Psi[i].max_Nsv = max(Psi[i].calc_Mmax(),500ul);
	}
	
	vector<TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,Mps<Symmetry,complex<double>>>> TDVP(Lcell);
	for (int i=0; i<Lcell; ++i)
	{
		TDVP[i] = TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,Mps<Symmetry,complex<double>>>(H, Psi[i]);
	}
	
	vector<EntropyObserver<Mps<Symmetry,complex<double>>>> Sobs(Lcell);
	vector<vector<bool>> TWO_SITE(Lcell);
	for (int i=0; i<Lcell; ++i)
	{
		DMRG::VERBOSITY::OPTION VERB = (i==0)? CHOSEN_VERBOSITY:DMRG::VERBOSITY::SILENT;
		Sobs[i] = EntropyObserver<Mps<Symmetry,complex<double>>>(Lhetero, Nt, VERB, tol_DeltaS);
		TWO_SITE[i] = Sobs[i].TWO_SITE(0, Psi[i], 1.); // {}, {0ul,size_t(Lhetero)-2ul}
	}
	if (CHOSEN_VERBOSITY > DMRG::VERBOSITY::ON_EXIT) lout << endl;
	
	IntervalIterator t(0.,tmax,Nt);
	double tval = 0.;
	
	if (SAVE_LOG) log = GreenPropagatorLog(Nt,Lhetero,logfolder);
	
	// 0.1. measure wavepacket at t=0
	for (int i=0; i<Lcell; ++i)
	{
		measure_wavepacket(Psi[i],0,make_string("i=",i,"_"));
	}
	
	// 0.2. if no (open) integration weights, calculate G at t=0
	if (GREENINT_CHOICE != DIRECT)
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
			if (tsteps(t.index()) != 0.)
			{
				//-------------------------------------------------------------------------------------------------
				TDVP[i].t_step_adaptive(H, Psi[i], 1.i*tsign*tsteps(t.index()), TWO_SITE[i], 1,tol_Lanczos);
				//-------------------------------------------------------------------------------------------------
			}
			
			if (Psi[i].get_truncWeight().sum() > 0.5*tol_compr)
			{
				Psi[i].max_Nsv = min(static_cast<size_t>(max(Psi[i].max_Nsv*1.1, Psi[i].max_Nsv+1.)),lim_Nsv);
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
			lout << "propagated to: t=" << tval << ", stepsize=" << tsteps(t.index()) << ", step#" << t.index()+1 << "/" << Nt << endl;
		}
		
		// 2. measure
		// 2.1. Green's function
		calc_GreenCell(t.index(), -1.i*exp(-1.i*tsign*Eg*tval), OxPhi, Psi);
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE) lout << StepTimer.info("G(t,x) calculation") << endl;
		// 2.2. measure wavepacket at t
		if (Measure.size() != 0)
		{
			#pragma omp parallel for
			for (int i=0; i<Lcell; ++i)
			{
				if ((t.index()-1)%measure_interval == 0 and t.index() > 1) measure_wavepacket(Psi[i],tval,make_string("i=",i,"_"));
			}
			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE) lout << StepTimer.info("wavepacket measurement") << endl;
		}
		
		// 3. check entropy increase
		#pragma omp parallel for
		for (int i=0; i<Lcell; ++i)
		{
			auto PsiTmp = Psi[i]; PsiTmp.entropy_skim();
			double r = (t.index()==0)? 1.:tsteps(t.index()-1)/tsteps(t.index());
			TWO_SITE[i] = Sobs[i].TWO_SITE(t.index(), PsiTmp, r); // {}, {0ul,size_t(Lhetero)-2ul}
			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE and i==0) lout << StepTimer.info("entropy calculation") << endl;
			
			save_log(i,t.index(),tval,PsiTmp,TDVP[i],Sobs[i],TWO_SITE[i]);
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
			measure_wavepacket(Psi[i],tval,make_string("i=",i,"_"));
		}
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
counterpropagate_cell (const Hamiltonian &H, const vector<Mps<Symmetry,complex<double>>> &OxPhi, double Eg, bool TIME_FORWARDS)
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
			Psi[z][i].max_Nsv = max(Psi[z][i].calc_Mmax(),500ul);
		}
	}
	
	std::array<vector<TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,Mps<Symmetry,complex<double>>>>,2> TDVP;
	for (int z=0; z<2; ++z)
	{
		TDVP[z].resize(Lcell);
		for (int i=0; i<Lcell; ++i)
		{
			TDVP[z][i] = TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,Mps<Symmetry,complex<double>>>(H, Psi[z][i]);
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
			TWO_SITE[z][i] = Sobs[z][i].TWO_SITE(0, Psi[z][i]);
		}
	}
	if (CHOSEN_VERBOSITY > DMRG::VERBOSITY::ON_EXIT) lout << endl;
	
	IntervalIterator t(0.,tmax,Nt);
	double tval = 0.;
	
	if (SAVE_LOG) log = GreenPropagatorLog(Nt,Lhetero,logfolder);
	
	// 0.1. measure wavepacket at t=0
	if (Measure.size() != 0)
	{
		for (int i=0; i<Lcell; ++i)
		{
			measure_wavepacket(Psi[0][i],0,make_string("i=",i,"_"));
		}
	}
	
	// 0.2. if no (open) integration weights, calculate G at t=0
	if (GREENINT_CHOICE != DIRECT)
	{
		Stopwatch<> StepTimer;
//		lout << termcolor::blue << "phase=" << -1.i*exp(-1.i*tsign*Eg*tval) << termcolor::reset << endl;
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
			if (tsteps(t.index()) != 0.)
			{
				//---------------------------------------------------------------------------------------------------------------
				TDVP[z][i].t_step_adaptive(H, Psi[z][i], 0.5*1.i*zfac[z]*tsign*tsteps(t.index()), TWO_SITE[z][i], 1,tol_Lanczos);
	//			#pragma omp critical
	//			{
	//				cout << "z=" << z << ", i=" << i << ", norm=" << Psi[z][i].squaredNorm() << ", E=" << avg_hetero(Psi[z][i], H, Psi[z][i], true) << endl;
	//			}
				if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::STEPWISE and i==0 and z==0)
				{
					std::streamsize ss = std::cout.precision();
					lout << "δE=" << setprecision(4) << TDVP[z][i].get_deltaE().transpose() << setprecision(ss) << endl;
				}
			}
			//---------------------------------------------------------------------------------------------------------------
			
			if (Psi[z][i].get_truncWeight().sum() > 0.5*tol_compr)
			{
				Psi[z][i].max_Nsv = min(static_cast<size_t>(max(Psi[z][i].max_Nsv*1.1, Psi[z][i].max_Nsv+1.)),lim_Nsv);
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
			lout << "propagated to: t=0.5*" << tval << "=" << 0.5*tval 
			     << ", stepsize=0.5*" << tsteps(t.index()) << "=" << 0.5*tsteps(t.index())
			     << ", step#" << t.index()+1 << "/" << Nt
			     << endl;
		}
		
		// 2. measure
		// 2.1. Green's function
		//----------------------------------------------------------
//		lout << termcolor::blue << "phase=" << -1.i*exp(-1.i*tsign*Eg*tval) << termcolor::reset << endl;
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
			auto PsiTmp = Psi[z][i]; PsiTmp.entropy_skim();
			double r = (t.index()==0)? 1.:tsteps(t.index()-1)/tsteps(t.index());
//			// possible override: always 2-site near excitation centre:
//			vector<size_t> overrides = {0ul, size_t(Lhetero-2)};
			TWO_SITE[z][i] = Sobs[z][i].TWO_SITE(t.index(), PsiTmp, r);
			
			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE and i==0 and z==0) lout << StepTimer.info("entropy calculation") << endl;
			
			if (z==0) save_log(i,t.index(),0.5*tval,PsiTmp,TDVP[z][i],Sobs[z][i],TWO_SITE[z][i]);
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
compute_thermal_cell (const Hamiltonian &H, const vector<Mpo<Symmetry,MpoScalar>> &Odag, const Mps<Symmetry,complex<double>> &Phi, 
                      vector<Mps<Symmetry,complex<double>>> &OxPhi0, bool TIME_FORWARDS)
{
	print_starttext();
	assert(H.volume()/H.length()==1 or H.volume()/H.length()==2);
	
	Lcell = OxPhi0.size();
	dLphys = (H.volume()/H.length()==2)? 1:2;
	Lhetero = (dLphys==1)? H.length():H.length()/2;
	Ncells = Lhetero/Lcell;
//	cout << "Lcell=" << Lcell << ", Lhetero=" << Lhetero << ", Ncells=" << Ncells << ", dLphys=" << dLphys << endl;
	
	assert(Lhetero%Lcell == 0);
	
//	calc_intweights();
	make_xarrays(Lhetero,Lcell,Ncells,Ns);
	
	GtxCell.resize(Lcell);
	for (int i=0; i<Lcell; ++i)
	{
		GtxCell[i].resize(Lcell);
	}
	for (int i=0; i<Lcell; ++i)
	for (int j=0; j<Lcell; ++j)
	{
		GtxCell[i][j].resize(Nt,Ncells);
		GtxCell[i][j].setZero();
	}
	
	GloctCell.resize(Lcell);
	for (int i=0; i<Lcell; ++i)
	{
		GloctCell[i].resize(Nt);
		GloctCell[i].setZero();
	}
	
	propagate_thermal_cell(H, Odag, Phi, OxPhi0, TIME_FORWARDS);
	
//	FTcell_xq();
//	FTcell_tw();
	FTcellxx_tw();
	FTcellww_xq((TIME_FORWARDS)?false:true);
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
propagate_thermal_cell (const Hamiltonian &H, const vector<Mpo<Symmetry,MpoScalar>> &Odag, Mps<Symmetry,complex<double>> Phi, 
                        vector<Mps<Symmetry,complex<double>>> &OxPhi0, bool TIME_FORWARDS)
{
	double tsign = (TIME_FORWARDS==true)? -1.:+1.;
	
	vector<Mps<Symmetry,complex<double>>> Psi(Lcell+1);
	for (int i=0; i<Lcell+1; ++i)
	{
		Psi[i] = (i==Lcell)? Phi : OxPhi0[i]; // Phi = Psi[Lcell]
		Psi[i].eps_svd = tol_compr;
		Psi[i].max_Nsv = max(Psi[i].calc_Mmax(),min(500ul,lim_Nsv));
	}
	
	vector<TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,Mps<Symmetry,complex<double>>>> TDVP(Lcell+1);
	for (int i=0; i<Lcell+1; ++i)
	{
		TDVP[i] = TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,Mps<Symmetry,complex<double>>>(H,Psi[i]);
	}
//	vector<TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,Mps<Symmetry,complex<double>>>> TDVPa(Lcell+1);
//	for (int i=0; i<Lcell+1; ++i)
//	{
//		TDVPa[i] = TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,Mps<Symmetry,complex<double>>>(Ha,Psi[i]);
//	}
	vector<EntropyObserver<Mps<Symmetry,complex<double>>>> Sobs(Lcell+1);
	vector<vector<bool>> TWO_SITE(Lcell+1);
	for (int i=0; i<Lcell+1; ++i)
	{
		DMRG::VERBOSITY::OPTION VERB = (i==0)? CHOSEN_VERBOSITY:DMRG::VERBOSITY::SILENT;
		Sobs[i] = EntropyObserver<Mps<Symmetry,complex<double>>>(H.length(), Nt, VERB, tol_DeltaS);
		TWO_SITE[i] = Sobs[i].TWO_SITE(0, Psi[i], 1.); // {}, {0ul,size_t(Lhetero)-2ul}
	}
	if (CHOSEN_VERBOSITY > DMRG::VERBOSITY::ON_EXIT) lout << endl;
	
	IntervalIterator t(0.,tmax,Nt);
	double tval = 0.;
	
	if (SAVE_LOG) log = GreenPropagatorLog(Nt,Lhetero,logfolder);
	
	// 0.1. measure wavepacket at t=0
	for (int i=0; i<Lcell; ++i)
	{
		measure_wavepacket(Psi[i],0,make_string("i=",i,"_"));
	}
	
	// 0.2. if no (open) integration weights, calculate G at t=0
	if (GREENINT_CHOICE != DIRECT)
	{
		Stopwatch<> StepTimer;
		calc_GreenCell_thermal(0, Odag, Psi);
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
		for (int i=0; i<Lcell+1; ++i)
		{
			//-------------------------------------------------------------------------------------------------
			if (tsteps(t.index()) != 0.)
			{
				TDVP[i].t_step_adaptive(H, Psi[i], 1.i*tsign*tsteps(t.index()), TWO_SITE[i], 1,tol_Lanczos);
			}
			//-------------------------------------------------------------------------------------------------
			
			if (Psi[i].get_truncWeight().sum() > 0.5*tol_compr)
			{
				Psi[i].max_Nsv = min(static_cast<size_t>(max(Psi[i].max_Nsv*1.1, Psi[i].max_Nsv+1.)),lim_Nsv);
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
			lout << "KET: " << TDVP[0].info() << endl;
			lout << "KET: " << Psi[0].info() << endl;
			lout << "BRA: " << TDVP[Lcell].info() << endl;
			lout << "BRA: " << Psi[Lcell].info() << endl;
			lout << termcolor::blue 
			     << "propagated to: t=" << tval << ", stepsize=" << tsteps(t.index()) << ", step#" << t.index()+1 << "/" << Nt 
			     << termcolor::reset << endl;
		}
		
		// 2. measure
		// 2.1. Green's function
		calc_GreenCell_thermal(t.index(), Odag, Psi);
		if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE) lout << StepTimer.info("G(t,x) calculation") << endl;
		// 2.2. measure wavepacket at t
		if (Measure.size() != 0)
		{
			#pragma omp parallel for
			for (int i=0; i<Lcell; ++i)
			{
				if ((t.index()-1)%measure_interval == 0 and t.index() > 1) measure_wavepacket(Psi[i],tval,make_string("i=",i,"_"));
			}
			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE) lout << StepTimer.info("wavepacket measurement") << endl;
		}
		
		// 3. check entropy increase
		#pragma omp parallel for
		for (int i=0; i<Lcell+1; ++i)
		{
			auto PsiTmp = Psi[i]; PsiTmp.entropy_skim();
			double r = (t.index()==0)? 1.:tsteps(t.index()-1)/tsteps(t.index());
			TWO_SITE[i] = Sobs[i].TWO_SITE(t.index(), PsiTmp, r); // {}, {0ul,size_t(Lhetero)-2ul}
			if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE and i==0)
			lout << StepTimer.info("entropy calculation") << endl;
			
			save_log(i,t.index(),tval,PsiTmp,TDVP[i],Sobs[i],TWO_SITE[i]);
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
			measure_wavepacket(Psi[i],tval,make_string("i=",i,"_"));
		}
	}
}

//template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
//void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
//calc_Green (const int &tindex, const complex<double> &phase, const vector<Mps<Symmetry,complex<double>>> &OxPhi, const Mps<Symmetry,complex<double>> &Psi)
//{
////	MatrixXcd Gtx_(Gtx.rows(),Gtx.cols());
//	
//	//variant: Don't use cell shift, OxPhiFull must be of length Lhetero
////	if (Psi.Boundaries.IS_TRIVIAL())
//	if (OxPhiFull.size() > 0)
//	{
////		assert(OxPhiFull.size() == Lhetero and "Call set_OxPhiFull with this setup! OxPhi parameter will be ignored.");
//		if (NQ == 0)
//		{
//			#pragma omp parallel for
//			for (size_t l=0; l<Lhetero; ++l)
//			{
//				Gtx(tindex,l) = phase * dot(OxPhiFull[l], Psi);
//			}
//		}
//		// SU(2) Qmultitarget
//		else
//		{
//			#pragma omp parallel for
//			for (size_t l=0; l<Lhetero; ++l)
//			{
//				auto dotres = dot_green(OxPhiFull[l], Psi);
//				for (size_t iQ=0; iQ<GtxQmulti.size(); ++iQ)
//				{
//					GtxQmulti[iQ](tindex,l) = phase * dotres[iQ];
//				}
//			}
//		}
//	}
//	// variant: Use cell shift
//	else
//	{
////		cout << "use cell shift, Lhetero=" << Lhetero << ", tindex=" << tindex << endl;
//		#pragma omp parallel for
//		for (size_t l=0; l<Lhetero; ++l)
//		{
//			Gtx(tindex,l) = phase * dot_hetero(OxPhi[icell[l]], Psi, dcell[l]);
////			cout << "l=" << l << ", icell[l]=" << icell[l] << ", dcell[l]=" << dcell[l] << ", G=" << Gtx(tindex,l) << endl;
//		}
//	}
//	
//	for (size_t l=0; l<Lhetero; ++l)
//	{
//		if (xvals[l] == 0)
//		{
//			if (NQ == 0)
//			{
//				Gloct(tindex) = Gtx(tindex,l); // save local Green's function
//			}
//			else
//			{
//				for (size_t iQ=0; iQ<GtxQmulti.size(); ++iQ)
//				{
//					GloctQmulti[iQ](tindex) = GtxQmulti[iQ](tindex,l);
//				}
//			}
//		}
//	}
//	
////	for (size_t l=0; l<Lhetero; ++l)
////	{
////		cout << "l=" << l << ", " << Gtx(tindex,l) << ", " << Gtx_(tindex,l) << ", diff=" << abs(Gtx(tindex,l)-Gtx_(tindex,l)) << endl;
////	}
////	cout << termcolor::blue << "total diff = " << (Gtx.row(tindex)-Gtx_.row(tindex)).norm() << termcolor::reset << endl;
//}

//template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
//void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
//calc_Green_thermal (const int &tindex, const vector<Mpo<Symmetry,MpoScalar>> &Odag, 
//                    const Mps<Symmetry,complex<double>> &Phi, const Mps<Symmetry,complex<double>> &Psi)
//{
//	assert(Psi.Boundaries.IS_TRIVIAL());
//	
//	#pragma omp parallel for
//	for (size_t l=0; l<Lhetero; ++l)
//	{
////		Mps<Symmetry,complex<double>> OxPhi;
////		OxV_exact(Ox[l], Phi, OxPhi, 2., DMRG::VERBOSITY::SILENT);
////		Gtx(tindex,l) = -1.i*dot(OxPhi,Psi);
//		
//		Gtx(tindex,l) = -1.i * avg(Phi, Odag[l], Psi);
//		
////		#pragma omp critial
////		{
////			lout << "G1=" << conj(avg(Psi, Ox[l], Phi)) << ", G2=" << dot(OxPhi,Psi) << endl;
////		}
//	}
//	
//	for (size_t l=0; l<Lhetero; ++l)
//	{
//		if (xvals[l] == 0)
//		{
//			Gloct(tindex) = Gtx(tindex,l); // save local Green's function
//		}
//	}
//}

// used in propagate
template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
calc_GreenCell (const int &tindex,
                const complex<double> &phase, 
                const vector<Mps<Symmetry,complex<double>>> &OxPhi,
                const vector<Mps<Symmetry,complex<double>>> &Psi)
{
	//variant: Don't use cell shift, OxPhiFull must be of length Lhetero
//	if (Psi[0].Boundaries.IS_TRIVIAL())
	if (OxPhiFull.size() > 0)
	{
		#pragma omp parallel for collapse(3)
		for (size_t n=0; n<Ncells; ++n)
		for (size_t i=0; i<Lcell; ++i)
		for (size_t j=0; j<Lcell; ++j)
		{
			GtxCell[i][j](tindex,n) = phase * dot(OxPhiFull[n*Lcell+i], Psi[j]);
		}
	}
	// variant: Use cell shift
	else
	{
		#pragma omp parallel for collapse(3)
		for (size_t i=0; i<Lcell; ++i)
		for (size_t j=0; j<Lcell; ++j)
		for (size_t n=0; n<Ncells; ++n)
		{
			GtxCell[i][j](tindex,n) = phase * dot_hetero(OxPhi[i], Psi[j], dcell[n*Lcell]);
		}
//		#pragma omp parallel for collapse(3)
//		for (size_t n=0; n<Ncells; ++n)
//		for (size_t i=0; i<Lcell; ++i)
//		for (size_t j=0; j<Lcell; ++j)
//		{
//			GtxCell[i][j](tindex,n) = phase * dot_hetero(OxPhiFull[n*Lcell+i], Psi[j]);
//		}
	}
	
	for (size_t i=0; i<Lcell; ++i)
	for (size_t n=0; n<Ncells; ++n)
	{
		if (dcell[n*Lcell] == 0) 
		{
			GloctCell[i](tindex) = GtxCell[i][i](tindex,n); // save local Green's function
		}
	}
}

// used in counterpropagate
template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
calc_GreenCell (const int &tindex,
                const complex<double> &phase, 
                const std::array<vector<Mps<Symmetry,complex<double>>>,2> &Psi)
{
//	cout << "phase=" << phase << endl;
//	#pragma omp parallel for collapse(3)
	for (size_t i=0; i<Lcell; ++i)
	for (size_t j=0; j<Lcell; ++j)
	for (size_t n=0; n<Ncells; ++n)
	{
		GtxCell[i][j](tindex,n) = phase * dot_hetero(Psi[1][i], Psi[0][j], dcell[n*Lcell]);
//		#pragma omp critical
//		{
//			cout << "i=" << i << ", j=" << j << ", n=" << n << ", dcell=" << dcell[n*Lcell] << ", dot=" << dot_hetero(Psi[1][i], Psi[0][j], dcell[n*Lcell]) << ", G=" << GtxCell[i][j](tindex,n) << endl;
//		}
	}
	
	for (size_t i=0; i<Lcell; ++i)
	for (size_t n=0; n<Ncells; ++n)
	{
		if (dcell[n*Lcell] == 0) 
		{
			GloctCell[i](tindex) = GtxCell[i][i](tindex,n); // save local Green's function
		}
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
calc_GreenCell_thermal (const int &tindex, const vector<Mpo<Symmetry,MpoScalar>> &Odag, const vector<Mps<Symmetry,complex<double>>> &Psi)
{
	for (size_t i=0; i<Lcell+1; ++i) assert(Psi[i].Boundaries.IS_TRIVIAL());
	
	#pragma omp parallel for collapse(3)
	for (size_t n=0; n<Ncells; ++n)
	for (size_t i=0; i<Lcell; ++i)
	for (size_t j=0; j<Lcell; ++j)
	{
//		Mps<Symmetry,complex<double>> OxPhi;
//		OxV_exact(Ox[n*Lcell+i], Psi[Lcell], OxPhi, 2., DMRG::VERBOSITY::SILENT); // Phi = Psi[Lcell]
//		GtxCell[i][j](tindex,n) = -1.i*dot(OxPhi,Psi[j]);
		
		GtxCell[i][j](tindex,n) = -1.i * avg(Psi[Lcell], Odag[n*Lcell+i], Psi[j]); // Phi = Psi[Lcell]
		
//		#pragma omp critial
//		{
//			lout << "G1=" << conj(avg(Psi[j], Ox[n*Lcell+i], Psi[Lcell])) << ", G2=" << dot(OxPhi,Psi[j]) << endl;
//		}
	}
	
	for (size_t i=0; i<Lcell; ++i)
	for (size_t n=0; n<Ncells; ++n)
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
//		cout << termcolor::yellow << make_string(measure_subfolder,"/",label,"_Op=",measure_name,"_",info,xinfo(),"_t=",tval,".dat") << termcolor::reset << endl;
		for (size_t l=0; l<Measure.size(); ++l)
		{
			Filer << xvals[l] << "\t" << res(l) << endl;
		}
		Filer.close();
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
make_xarrays (int Lhetero_input, int Lcell_input, int Ncells_input, int Ns, bool PRINT)
{
	xinds.resize(Lhetero_input);
	xvals.resize(Lhetero_input);
	
	iota(begin(xinds),end(xinds),0);
	
	int x0 = Lhetero_input/2; // first site of central unit cell
	if (PRINT and CHOSEN_VERBOSITY > DMRG::VERBOSITY::SILENT) lout << "make_xarrays: x0=" << x0 << endl;
	
	for (int ix=0; ix<xinds.size(); ++ix)
	{
		xvals[ix] = static_cast<double>(xinds[ix]-x0);
	}
	
	for (int d=0; d<Ncells_input; ++d)
	for (int i=0; i<Lcell_input; ++i)
	{
		icell.push_back(i);
		dcell.push_back(d);
	}
	
	for (int d=0; d<Ncells_input*Ns; ++d)
	for (int i=0; i<Lcell_input/Ns; ++i)
	{
		dcellFT.push_back(d);
	}
	
	for (int i=0; i<Lhetero_input; ++i)
	{
		dcell[i] -= x0/Lcell_input;
		dcellFT[i] -= x0/Lcell_input*Ns;
	}
	
	if (PRINT and CHOSEN_VERBOSITY > DMRG::VERBOSITY::SILENT)
	for (int i=0; i<Lhetero; ++i)
	{
		lout << "i=" << xinds[i] << ", x=" << xvals[i] << ", icell=" << icell[i] << ", dcell=" << dcell[i] << ", dcellFT=" << dcellFT[i] << endl;
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
set_qlims (Q_RANGE CHOICE)
{
	if (CHOICE == MPI_PPI)
	{
		qmin = -M_PI;
		qmax = +M_PI;
	}
	else if (CHOICE == ZERO_2PI)
	{
		qmin = 0.;
		qmax = 2.*M_PI;
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
save_log (int i, int tindex, double tval, const Mps<Symmetry,complex<double>> &Psi, const TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,Mps<Symmetry,complex<double>>> &TDVP, const EntropyObserver<Mps<Symmetry,complex<double>>> &Sobs, const vector<bool> &TWO_SITE)
{
	if (SAVE_LOG)
	{
		log.Entropy.row(tindex) = Psi.entropy();
		log.VarE.row(tindex) = TDVP.get_deltaE();
		for (int l=0; l<TWO_SITE.size(); ++l) log.TwoSite(tindex,l) = int(TWO_SITE[l]);
		log.DeltaS = Sobs.get_DeltaSb();
		log.taxis(tindex,0) = tval;
		log.dur_tstep(tindex,0) = TDVP.get_t_tot();
		
		log.dimK2avg(tindex,0) = 0;
		for (int i=0; i<TDVP.get_dimK2_log().size(); ++i)
		{
			log.dimK2avg(tindex,0) += double(TDVP.get_dimK2_log()[i])/TDVP.get_dimK2_log().size();
		}
		
		log.dimK1avg(tindex,0) = 0;
		for (int i=0; i<TDVP.get_dimK1_log().size(); ++i)
		{
			log.dimK1avg(tindex,0) += double(TDVP.get_dimK1_log()[i])/TDVP.get_dimK1_log().size();
		}
		
		log.dimK0avg(tindex,0) = 0;
		for (int i=0; i<TDVP.get_dimK0_log().size(); ++i)
		{
			log.dimK0avg(tindex,0) += double(TDVP.get_dimK0_log()[i])/TDVP.get_dimK0_log().size();
		}
		
		log.save(make_string(label,"_i=",i,"_",xt_info()));
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
calc_intweights()
{
	Stopwatch<> Watch;
	
	#ifdef GREENPROPAGATOR_USE_GAUSSIAN_QUADRATURE
	if (GREENINT_CHOICE == DIRECT)
	{
		TimeIntegrator = SuperQuadrator<GAUSS_LEGENDRE>(GreenPropagatorGlobal::damping,0.,tmax,Nt);
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
			double erf2 = 0.995322265018952734162069256367252928610891797040060076738; // Mathematica♥♥♥
			double integral = 0.25*sqrt(M_PI)*erf2*tmax;
			double diff = abs(weights.sum()-integral);
			#pragma omp critical
			{
				if      (diff < 1e-5)                  cout << termcolor::green;
				else if (diff < 1e-1 and diff >= 1e-5) cout << termcolor::yellow;
				else                                   cout << termcolor::red;
				lout << setprecision(14)
					 << "integration weight test: ∫w(t)dt=" << weights.sum() 
					 << ", analytical=" << integral 
					 << ", diff=" << abs(weights.sum()-integral)
					 << setprecision(6);
				cout << termcolor::reset;
				lout << endl;
			}
		}
	}
	else
	#endif
	{
		double dt = tmax/Nt;
		Nt += 1;
		
		tvals.resize(Nt);
		for (int i=0; i<Nt; ++i) tvals(i) = i*dt;
		
		weights.resize(Nt); weights = 1.;
		tsteps.resize(Nt); tsteps = dt; tsteps(0) = 0;
	}
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		#pragma omp critical
		{
			lout << "tmax=" << tmax
				 << ", tpoints=" << Nt
				 << ", max(tstep)=" << tsteps.maxCoeff()
				 << ", tol_compr=" << tol_compr
				 << ", tol_Lanczos=" << tol_Lanczos
				 << ", tol_DeltaS=" << tol_DeltaS
				 << ", INT=" << GREENINT_CHOICE;
			lout << endl;
			lout << Watch.info("integration weights") << endl;
		}
	}
}

//template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
//void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
//FT_xq (const MatrixXcd &Gtx, MatrixXcd &Gtq)
//{
//	IntervalIterator q(qmin,qmax,Nq);
//	ArrayXd qvals = q.get_abscissa();
//	Gtq.resize(Nt,Nq); Gtq.setZero();
//	
//	Stopwatch<> FourierWatch;
//	
////	// Use FFT to transform from x to q
////	Eigen::FFT<double> fft;
////	for (int it=0; it<tvals.rows(); ++it)
////	{
////		VectorXcd vtmp;
////		fft.fwd(vtmp,Gtx.row(it));
////		Gtq.row(it) = vtmp;
////	}
////	// phase shift factor because the origin site is in the middle
////	for (int iq=0; iq<Nq; ++iq)
////	{
////		Gtq.col(iq) *= exp(-1.i*2.*M_PI/double(xvals.size())*xvals[0]*double(iq));
////	}
////	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::ON_EXIT)
////	{
////		lout << FourierWatch.info("FFT x→q") << endl;
////	}
//	
//	// Explicit FT
//	for (int iq=0; iq<Nq; ++iq)
//	for (int ix=0; ix<Gtx.cols(); ++ix)
//	{
//		Gtq.col(iq) += Gtx.col(ix) * exp(-1.i*qvals(iq)*xvals[ix]);
//	}
//	
//	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
//	{
//		lout << FourierWatch.info("FT x→q (const q)") << endl;
//	}
//}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
FTcell_xq()
{
	IntervalIterator q(qmin,qmax,Nq);
	ArrayXd qvals = q.get_abscissa();
	
	GtqCell.resize(Lcell);
	for (int i=0; i<Lcell; ++i) GtqCell[i].resize(Lcell);
	
	Stopwatch<> FourierWatch;
	
	for (int i=0; i<Lcell; ++i)
	for (int j=0; j<Lcell; ++j)
	{
		GtqCell[i][j].resize(Nt,Nq); GtqCell[i][j].setZero();
		
		for (int iq=0; iq<Nq; ++iq)
		for (int n=0; n<Ncells; ++n)
		{
			GtqCell[i][j].col(iq) += GtxCell[i][j].col(n) * exp(-1.i*qvals(iq)*double(dcell[n*Lcell]));
		}
	}
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << FourierWatch.info(label+" FT intercell x→q (const t)") << endl;
	}
}

// //-----old version before Ns-----
//template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
//void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
//FTcellww_xq (bool TRANSPOSE)
//{
//	IntervalIterator q(qmin,qmax,Nq);
//	ArrayXd qvals = q.get_abscissa();
//	
//	GwqCell.resize(Lcell);
//	for (int i=0; i<Lcell; ++i) GwqCell[i].resize(Lcell);
//	
//	G0qCell.resize(Lcell);
//	for (int i=0; i<Lcell; ++i) G0qCell[i].resize(Lcell);
//	
//	Stopwatch<> FourierWatch;
//	
//	for (int i=0; i<Lcell; ++i)
//	for (int j=0; j<Lcell; ++j)
//	{
//		GwqCell[i][j].resize(Nw,Nq); GwqCell[i][j].setZero();
//		G0qCell[i][j].resize(Nq); G0qCell[i][j].setZero();
//		
//		for (int iq=0; iq<Nq; ++iq)
//		for (int n=0; n<Ncells; ++n)
//		{
//			if (TRANSPOSE)
//			{
//				GwqCell[i][j].col(iq) += GwxCell[j][i].col(n) * exp(+1.i*qvals(iq)*double(dcell[n*Lcell]));
//				G0qCell[i][j](iq) += G0xCell[j][i](n) * exp(+1.i*qvals(iq)*double(dcell[n*Lcell]));
//			}
//			else
//			{
//				GwqCell[i][j].col(iq) += GwxCell[i][j].col(n) * exp(-1.i*qvals(iq)*double(dcell[n*Lcell]));
//				G0qCell[i][j](iq) += G0xCell[i][j](n) * exp(-1.i*qvals(iq)*double(dcell[n*Lcell]));
//			}
//		}
//	}
//	
//	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
//	{
//		lout << FourierWatch.info(label+" FT intercell x→q (const ω)") << endl;
//	}
//}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
FTcellww_xq (bool TRANSPOSE)
{
	IntervalIterator q(qmin,qmax,Nq);
	ArrayXd qvals = q.get_abscissa();
	
	int Ls = Lcell/Ns;
	
	GwqCell.resize(Ls);
	for (int i=0; i<Ls; ++i) GwqCell[i].resize(Ls);
	
	G0qCell.resize(Ls);
	for (int i=0; i<Ls; ++i) G0qCell[i].resize(Ls);
	
	Stopwatch<> FourierWatch;
	
	for (int i=0; i<Ls; ++i)
	for (int j=0; j<Ls; ++j)
	{
		GwqCell[i][j].resize(Nw,Nq); GwqCell[i][j].setZero();
		G0qCell[i][j].resize(Nq); G0qCell[i][j].setZero();
	}
	
	for (int i=0; i<Lcell; ++i)
	for (int j=0; j<Lcell; ++j)
	{
		
		for (int n=0; n<Ncells; ++n)
		{
//			cout << "i=" << i << ", j=" << j << ", n=" << n << ", d=" << dcell[n*Lcell] << ", i_=" << (n*Lcell+i)%Ls << ", j_=" << (n*Lcell+j)%Ls << ", n_=" << n*Lcell << ", dFT=" << dcellFT[n*Lcell]+(i-j)/Ls << endl;
			for (int iq=0; iq<Nq; ++iq)
			{
				if (TRANSPOSE)
				{
					GwqCell[(n*Lcell+i)%Ls][(n*Lcell+j)%Ls].col(iq) += GwxCell[j][i].col(n) * exp(+1.i*qvals(iq)*double(dcellFT[n*Lcell]+(j-i)/Ls));
					G0qCell[(n*Lcell+i)%Ls][(n*Lcell+j)%Ls](iq)     += G0xCell[j][i]    (n) * exp(+1.i*qvals(iq)*double(dcellFT[n*Lcell]+(j-i)/Ls));
				}
				else
				{
					GwqCell[(n*Lcell+i)%Ls][(n*Lcell+j)%Ls].col(iq) += GwxCell[i][j].col(n) * exp(-1.i*qvals(iq)*double(dcellFT[n*Lcell]+(i-j)/Ls));
					G0qCell[(n*Lcell+i)%Ls][(n*Lcell+j)%Ls](iq)     += G0xCell[i][j]    (n) * exp(-1.i*qvals(iq)*double(dcellFT[n*Lcell]+(i-j)/Ls));
				}
			}
		}
	}
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
	{
		lout << FourierWatch.info(label+" FT intercell x→q (const ω)") << endl;
	}
}

//template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
//void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
//FT_tw (const MatrixXcd &Gtq, MatrixXcd &Gwq, VectorXcd &G0q, VectorXcd &Glocw)
//{
//	IntervalIterator w(wmin,wmax,Nw);
//	ArrayXd wvals = w.get_abscissa();
//	Gwq.resize(Nw,Nq); Gwq.setZero();
//	G0q.resize(Nq); G0q.setZero();
//	
//	Stopwatch<> FourierWatch;
//	
//	if (GREENINT_CHOICE == DIRECT)
//	{
//		for (int iw=0; iw<wvals.rows(); ++iw)
//		{
//			double wval = wvals(iw);
//			
//			for (int it=0; it<tvals.rows(); ++it)
//			{
//				double tval = tvals(it);
//				// If Gaussian integration is employed, the damping is already included in the weights
//				Gwq.row(iw) += weights(it) * exp(+1.i*wval*tval) * Gtq.row(it); 
//			}
//		}
//	}
//	else if (GREENINT_CHOICE == INTERP)
//	{
////		for (int iw=0; iw<wvals.rows(); ++iw)
////		{
////			double wval = wvals(iw);
////			
////			for (int iq=0; iq<Nq; ++iq)
////			{
////				ComplexInterpol Gtq_interp(tvals);
////				for (int it=0; it<tvals.rows(); ++it)
////				{
////					double tval = tvals(it);
////					complex<double> Gval = exp(+1.i*wval*tval) * GreenPropagatorGlobal::damping(tval) * Gtq(it,iq);
////					Gtq_interp.insert(it,Gval);
////				}
////				Gwq(iw,iq) += Gtq_interp.integrate();
////				Gtq_interp.kill_splines();
////			}
////		}
//		#pragma omp parallel for
//		for (int iq=0; iq<Nq; ++iq)
//		{
//			VectorXcd fvals = Gtq.col(iq);
//			for (int it=0; it<tvals.rows(); ++it) fvals(it) *= GreenPropagatorGlobal::damping(tvals(it));
//			Gwq.col(iq) = ::FT_interpol(tvals,fvals,USE_QAWO,wmin,wmax,Nw);
//		}
//	}
//	else if (GREENINT_CHOICE == OOURA)
//	{
//		#pragma omp parallel for
//		for (int iq=0; iq<Nq; ++iq)
//		{
//			VectorXcd fvals = Gtq.col(iq);
//			for (int it=0; it<tvals.rows(); ++it) fvals(it) *= GreenPropagatorGlobal::damping(tvals(it));
//			Gwq.col(iq) = Ooura::FT(tvals,fvals,h_ooura,wmin,wmax,Nw);
//		}
//	}
//	
//	if (GREENINT_CHOICE == DIRECT)
//	{
//		for (int it=0; it<tvals.rows(); ++it)
//		{
//			double tval = tvals(it);
//			// If Gaussian integration is employed, the damping is already included in the weights
//			G0q += weights(it) * Gtq.row(it);
//		}
//	}
//	else if (GREENINT_CHOICE == INTERP or GREENINT_CHOICE == OOURA)
//	{
//		for (int iq=0; iq<Nq; ++iq)
//		{
//			ComplexInterpol Gtq_interp(tvals);
//			for (int it=0; it<tvals.rows(); ++it)
//			{
//				double tval = tvals(it);
//				complex<double> Gval = GreenPropagatorGlobal::damping(tval) * Gtq(it,iq);
//				Gtq_interp.insert(it,Gval);
//			}
//			G0q(iq) += Gtq_interp.integrate();
//			Gtq_interp.kill_splines();
//		}
//	}
//	
//	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::ON_EXIT)
//	{
//		lout << FourierWatch.info(label+" FT t→ω (const q)");
//		if (GREENINT_CHOICE == INTERP) lout << boolalpha << ", USE_QAWO=" << USE_QAWO;
//		lout << endl;
//	}
//	
//	// Calculate FT of local Green's function
//	Glocw = FTloc_tw(Gloct,wvals);
//}

//template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
//void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
//FTxx_tw (const MatrixXcd &Gtx, MatrixXcd &Gwx, VectorXcd &G0x, VectorXcd &Glocw)
//{
//	IntervalIterator w(wmin,wmax,Nw);
//	ArrayXd wvals = w.get_abscissa();
//	int Nx = Gtx.cols();
//	Gwx.resize(Nw,Nx); Gwx.setZero();
//	G0x.resize(Nq); G0x.setZero();
//	
//	Stopwatch<> FourierWatch;
//	
//	if (GREENINT_CHOICE == DIRECT)
//	{
//		for (int iw=0; iw<wvals.rows(); ++iw)
//		{
//			double wval = wvals(iw);
//			
//			for (int it=0; it<tvals.rows(); ++it)
//			{
//				double tval = tvals(it);
//				// If Gaussian integration is employed, the damping is already included in the weights
//				Gwx.row(iw) += weights(it) * exp(+1.i*wval*tval) * Gtx.row(it); 
//			}
//		}
//	}
//	else if (GREENINT_CHOICE == INTERP)
//	{
//		#pragma omp parallel for
//		for (int ix=0; ix<Nx; ++ix)
//		{
//			VectorXcd fvals = Gtx.col(ix);
//			for (int it=0; it<tvals.rows(); ++it) fvals(it) *= GreenPropagatorGlobal::damping(tvals(it));
//			Gwx.col(ix) = ::FT_interpol(tvals,fvals,USE_QAWO,wmin,wmax,Nw);
//		}
//	}
//	else if (GREENINT_CHOICE == OOURA)
//	{
//		#pragma omp parallel for
//		for (int ix=0; ix<Nx; ++ix)
//		{
//			VectorXcd fvals = Gtx.col(ix);
//			for (int it=0; it<tvals.rows(); ++it) fvals(it) *= GreenPropagatorGlobal::damping(tvals(it));
//			Gwx.col(ix) = Ooura::FT(tvals,fvals,h_ooura,wmin,wmax,Nw);
//		}
//	}
//	
//	if (GREENINT_CHOICE == DIRECT)
//	{
//		for (int it=0; it<tvals.rows(); ++it)
//		{
//			double tval = tvals(it);
//			// If Gaussian integration is employed, the damping is already included in the weights
//			G0x += weights(it) * Gtx.row(it);
//		}
//	}
//	else if (GREENINT_CHOICE == INTERP or GREENINT_CHOICE == OOURA)
//	{
//		for (int ix=0; ix<Nx; ++ix)
//		{
//			ComplexInterpol Gtx_interp(tvals);
//			for (int it=0; it<tvals.rows(); ++it)
//			{
//				double tval = tvals(it);
//				complex<double> Gval = GreenPropagatorGlobal::damping(tval) * Gtq(it,ix);
//				Gtx_interp.insert(it,Gval);
//			}
//			G0x(ix) += Gtx_interp.integrate();
//			Gtx_interp.kill_splines();
//		}
//	}
//	
//	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::ON_EXIT)
//	{
//		lout << FourierWatch.info(label+" FT t→ω (const x)");
//		if (GREENINT_CHOICE == INTERP) lout << boolalpha << ", USE_QAWO=" << USE_QAWO;
//		lout << endl;
//	}
//	
//	// Calculate FT of local Green's function
//	Glocw = FTloc_tw(Gloct,wvals);
//}

//template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
//void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
//FTww_xq (const MatrixXcd &Gwx, const MatrixXcd &G0x, MatrixXcd &Gwq, VectorXcd &G0q)
//{
//	IntervalIterator q(qmin,qmax,Nq);
//	int Nx = Gwx.cols();
//	ArrayXd qvals = q.get_abscissa();
//	Gwq.resize(Nw,Nq); Gwq.setZero();
//	G0q.resize(Nq); G0q.setZero();
//	
//	Stopwatch<> FourierWatch;
//	
//	// Explicit FT
//	for (int iq=0; iq<Nq; ++iq)
//	for (int ix=0; ix<Nx; ++ix)
//	{
//		Gtq.col(iq) += Gtx.col(ix) * exp(-1.i*qvals(iq)*xvals[ix]);
//		G0q(iq) += G0x(ix) * exp(-1.i*qvals(iq)*xvals[ix]);
//	}
//	
//	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::HALFSWEEPWISE)
//	{
//		lout << FourierWatch.info("FT x→q (const ω)") << endl;
//	}
//}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
FTcell_tw()
{
	IntervalIterator w(wmin,wmax,Nw);
	ArrayXd wvals = w.get_abscissa();
	
	GwqCell.resize(Lcell); for (int i=0; i<Lcell; ++i) GwqCell[i].resize(Lcell);
	G0qCell.resize(Lcell); for (int i=0; i<Lcell; ++i) G0qCell[i].resize(Lcell);
	GlocwCell.resize(Lcell);
	
	Stopwatch<> FourierWatch;
	
	for (int i=0; i<Lcell; ++i)
	for (int j=0; j<Lcell; ++j)
	{
		GwqCell[i][j].resize(Nw,Nq); GwqCell[i][j].setZero();
		
		if (GREENINT_CHOICE == DIRECT)
		{
			for (int iw=0; iw<wvals.rows(); ++iw)
			{
				double wval = wvals(iw);
				
				for (int it=0; it<tvals.rows(); ++it)
				{
					double tval = tvals(it);
					// If Gaussian integration is employed, the damping is already included in the weights
					GwqCell[i][j].row(iw) += weights(it) * exp(+1.i*wval*tval) * GtqCell[i][j].row(it);
				}
			}
		}
		else if (GREENINT_CHOICE == INTERP)
		{
			#pragma omp parallel for
			for (int iq=0; iq<Nq; ++iq)
			{
				VectorXcd fvals = GtqCell[i][j].col(iq);
				for (int it=0; it<tvals.rows(); ++it) fvals(it) *= GreenPropagatorGlobal::damping(tvals(it));
				GwqCell[i][j].col(iq) = ::FT_interpol(tvals,fvals,USE_QAWO,wmin,wmax,Nw);
			}
		}
		else if (GREENINT_CHOICE == OOURA)
		{
			#pragma omp parallel for
			for (int iq=0; iq<Nq; ++iq)
			{
				VectorXcd fvals = GtqCell[i][j].col(iq);
				for (int it=0; it<tvals.rows(); ++it) fvals(it) *= GreenPropagatorGlobal::damping(tvals(it));
				GwqCell[i][j].col(iq) = Ooura::FT(tvals,fvals,h_ooura,wmin,wmax,Nw);
			}
		}
		
		G0qCell[i][j].resize(Nq); G0qCell[i][j].setZero();
		
		if (GREENINT_CHOICE == DIRECT)
		{
			for (int it=0; it<tvals.rows(); ++it)
			{
				double tval = tvals(it);
				// If Gaussian integration is employed, the damping is already included in the weights
				G0qCell[i][j] += weights(it) * GtqCell[i][j].row(it);
			}
		}
		else if (GREENINT_CHOICE == INTERP or GREENINT_CHOICE == OOURA)
		{
			for (int iq=0; iq<Nq; ++iq)
			{
				ComplexInterpol Gtq_interp(tvals);
				for (int it=0; it<tvals.rows(); ++it)
				{
					double tval = tvals(it);
					complex<double> Gval = GreenPropagatorGlobal::damping(tval)* GtqCell[i][j](it,iq);
					Gtq_interp.insert(it,Gval);
				}
				G0qCell[i][j](iq) += Gtq_interp.integrate();
				Gtq_interp.kill_splines();
			}
		}
		
		// Calculate local Green's function
		if (i==j)
		{
			GlocwCell[i] = FTloc_tw(GloctCell[i],wvals);
		}
	}
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::ON_EXIT)
	{
		lout << FourierWatch.info(make_string(label+" FT intercell t→ω (const q), ωmin=",wmin,", ωmax=",wmax,", Nω=",Nw));
		if (GREENINT_CHOICE == INTERP) lout << boolalpha << ", USE_QAWO=" << USE_QAWO;
		lout << endl;
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
FTcellxx_tw()
{
	IntervalIterator w(wmin,wmax,Nw);
	ArrayXd wvals = w.get_abscissa();
	
	GwxCell.resize(Lcell); for (int i=0; i<Lcell; ++i) GwxCell[i].resize(Lcell);
	G0xCell.resize(Lcell); for (int i=0; i<Lcell; ++i) G0xCell[i].resize(Lcell);
	GlocwCell.resize(Lcell);
	int Nx = GtxCell[0][0].cols();
	
	Stopwatch<> FourierWatch;
	
	for (int i=0; i<Lcell; ++i)
	for (int j=0; j<Lcell; ++j)
	{
		GwxCell[i][j].resize(Nw,Nx); GwxCell[i][j].setZero();
		
		if (GREENINT_CHOICE == DIRECT)
		{
			for (int iw=0; iw<wvals.rows(); ++iw)
			{
				double wval = wvals(iw);
				
				for (int it=0; it<tvals.rows(); ++it)
				{
					double tval = tvals(it);
					// If Gaussian integration is employed, the damping is already included in the weights
					GwxCell[i][j].row(iw) += weights(it) * exp(+1.i*wval*tval) * GtxCell[i][j].row(it);
				}
			}
		}
		else if (GREENINT_CHOICE == INTERP)
		{
			#pragma omp parallel for
			for (int ix=0; ix<Nx; ++ix)
			{
				VectorXcd fvals = GtxCell[i][j].col(ix);
				for (int it=0; it<tvals.rows(); ++it) fvals(it) *= GreenPropagatorGlobal::damping(tvals(it));
				GwxCell[i][j].col(ix) = ::FT_interpol(tvals,fvals,USE_QAWO,wmin,wmax,Nw);
			}
		}
		else if (GREENINT_CHOICE == OOURA)
		{
			#pragma omp parallel for
			for (int ix=0; ix<Nx; ++ix)
			{
				VectorXcd fvals = GtxCell[i][j].col(ix);
				for (int it=0; it<tvals.rows(); ++it) fvals(it) *= GreenPropagatorGlobal::damping(tvals(it));
				GwxCell[i][j].col(ix) = Ooura::FT(tvals,fvals,h_ooura,wmin,wmax,Nw);
			}
		}
		
		G0xCell[i][j].resize(Nx); G0xCell[i][j].setZero();
		
		if (GREENINT_CHOICE == DIRECT)
		{
			for (int it=0; it<tvals.rows(); ++it)
			{
				double tval = tvals(it);
				// If Gaussian integration is employed, the damping is already included in the weights
				G0xCell[i][j] += weights(it) * GtxCell[i][j].row(it);
			}
		}
		else if (GREENINT_CHOICE == INTERP or GREENINT_CHOICE == OOURA)
		{
			for (int ix=0; ix<Nx; ++ix)
			{
				ComplexInterpol Gtx_interp(tvals);
				for (int it=0; it<tvals.rows(); ++it)
				{
					double tval = tvals(it);
					complex<double> Gval = GreenPropagatorGlobal::damping(tval)* GtxCell[i][j](it,ix);
					Gtx_interp.insert(it,Gval);
				}
				G0xCell[i][j](ix) += Gtx_interp.integrate();
				Gtx_interp.kill_splines();
			}
		}
		
		// Calculate local Green's function
		if (i==j)
		{
			GlocwCell[i] = FTloc_tw(GloctCell[i],wvals);
		}
	}
	
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::ON_EXIT)
	{
		lout << FourierWatch.info(make_string(label+" FT intercell t→ω (const x), ωmin=",wmin,", ωmax=",wmax,", Nω=",Nw));
		if (GREENINT_CHOICE == INTERP) lout << boolalpha << ", USE_QAWO=" << USE_QAWO;
		lout << endl;
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
ArrayXcd GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
FTloc_tw (const VectorXcd &Gloct_in, const ArrayXd &wvals)
{
	ArrayXcd Glocw_out(wvals.rows()); Glocw_out.setZero();
	
	Stopwatch<> FourierWatch;
	
	// Use normal summation to transform from t to w
	if (GREENINT_CHOICE == DIRECT)
	{
		for (int iw=0; iw<wvals.rows(); ++iw)
		{
			double wval = wvals(iw);
			
			for (int it=0; it<tvals.rows(); ++it)
			{
				double tval = tvals(it);
				// If Gaussian integration is employed, the damping is already included in the weights
				Glocw_out(iw) += weights(it) * exp(+1.i*wval*tval) * Gloct_in(it);
			}
		}
	}
	else if (GREENINT_CHOICE == INTERP)
	{
		VectorXcd fvals = Gloct_in;
		for (int it=0; it<tvals.rows(); ++it) fvals(it) *= GreenPropagatorGlobal::damping(tvals(it));
		Glocw_out = ::FT_interpol(tvals,fvals,USE_QAWO,wmin,wmax,Nw);
	}
	else if (GREENINT_CHOICE == OOURA)
	{
		VectorXcd fvals = Gloct_in;
		for (int it=0; it<tvals.rows(); ++it) fvals(it) *= GreenPropagatorGlobal::damping(tvals(it));
		Glocw_out = Ooura::FT(tvals,fvals,h_ooura,wmin,wmax,Nw);
	}
	
	return Glocw_out;
}

//template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
//void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
//FT_allSites (bool TW_FIRST_XQ_SECOND)
//{
//	IntervalIterator q(qmin,qmax,Nq);
//	ArrayXd qvals = q.get_abscissa();
//	
//	Stopwatch<> FourierWatch;
//	
//	if (TW_FIRST_XQ_SECOND)
//	{
//		Gwq.resize(Nw,Nq); Gwq.setZero();
//		G0q.resize(Nq); G0q.setZero();
//		
//		FTcellxx_tw();
//		
//		for (int iq=0; iq<Nq; ++iq)
//		{
//			for (int n=0; n<Ncells; ++n)
//			for (int i=0; i<Lcell; ++i)
//			for (int j=0; j<Lcell; ++j)
//			{
//				Gwq.col(iq) += 1./Lcell * GwxCell[i][j].col(n) * exp(-1.i*qvals(iq)*double(Lcell)*double(dcell[n*Lcell])) * exp(-1.i*qvals(iq)*double(i-j));
//				G0q(iq) += 1./Lcell * G0xCell[i][j](n) * exp(-1.i*qvals(iq)*double(Lcell)*double(dcell[n*Lcell])) * exp(-1.i*qvals(iq)*double(i-j));
//			}
//		}
//		
//		Glocw = GlocwCell[0];
//		
//		lout << FourierWatch.info(label+" FT all sites x→q (const ω)");
//		if (GREENINT_CHOICE == INTERP) lout << boolalpha << ", USE_QAWO=" << USE_QAWO;
//		lout << endl;
//	}
//	else
//	{
//		Gtq.resize(Nt,Nq); Gtq.setZero();
//		
//		for (int iq=0; iq<Nq; ++iq)
//		{
//			for (int n=0; n<Ncells; ++n)
//			for (int i=0; i<Lcell; ++i)
//			for (int j=0; j<Lcell; ++j)
//			{
//				Gtq.col(iq) += 1./Lcell * GtxCell[i][j].col(n) * exp(-1.i*qvals(iq)*double(Lcell)*double(dcell[n*Lcell])) * exp(-1.i*qvals(iq)*double(i-j));
//			}
//		}
//		
//		lout << FourierWatch.info(label+" FT all sites x→q (const t)") << endl;
//		
//		Gloct.resize(Nt);
//		Gloct = GloctCell[0];
//		
//		FT_tw(Gtq,Gwq,G0q,Glocw);
//	}
//}

//template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
//void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
//recalc_FTw (double wmin_new, double wmax_new, int Nw_new)
//{
//	wmin = wmin_new;
//	wmax = wmax_new;
//	Nw = Nw_new;
//	
//	if (NQ == 0)
//	{
////		FT_xq(Gtx,Gtq);
////		FT_tw(Gtq,Gwq,G0q,Glocw);
//		FTxx_tw(Gtx,Gwx,G0x,Glocw);
//		FTww_xq(Gwx,G0x,Gwq,G0q);
//	}
//	else
//	{
//		for (int iQ=0; iQ<NQ; ++iQ)
//		{
//			GtqQmulti.resize(NQ);
//			FT_xq(GtxQmulti[iQ],GtqQmulti[iQ]);
//			GwqQmulti.resize(NQ);
//			G0qQmulti.resize(NQ);
//			GlocwQmulti.resize(NQ);
//			FT_tw(GtqQmulti[iQ],GwqQmulti[iQ],G0qQmulti[iQ],GlocwQmulti[iQ]);
//		}
//	}
//}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
recalc_FTwCell (double wmin_new, double wmax_new, int Nw_new, bool TRANSPOSE)
{
	wmin = wmin_new;
	wmax = wmax_new;
	Nw = Nw_new;
	
//	FTcell_xq();
//	FTcell_tw();
	FTcellxx_tw();
	FTcellww_xq(TRANSPOSE);
}

//template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
//double GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
//integrate_Glocw (double mu, int Nint)
//{
//	Quadrator<GAUSS_LEGENDRE> Q;
//	
//	ArrayXd wabscissa(Nint);
//	for (int i=0; i<Nint; ++i) {wabscissa(i) = Q.abscissa(i,wmin,mu,Nint);}
//	
//	ArrayXd QDOS = -1.*M_1_PI * FTloc_tw(Gloct,wabscissa).imag();
//	
//	return (Q.get_weights(wmin,mu,Nint) * QDOS).sum();
//}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
double GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
integrate_Glocw_cell (double mu, int Nint)
{
	Quadrator<GAUSS_LEGENDRE> Q;
	
	ArrayXd wabscissa(Nint);
	for (int i=0; i<Nint; ++i) {wabscissa(i) = Q.abscissa(i,wmin,mu,Nint);}
	
	ArrayXd QDOS = -1.*M_1_PI * FTloc_tw(GloctCell[0],wabscissa).imag();
	for (int b=1; b<Lcell; ++b)
	{
		QDOS += -1.*M_1_PI * FTloc_tw(GloctCell[b],wabscissa).imag();
	}
	
	return (Q.get_weights(wmin,mu,Nint) * QDOS).sum();
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
string GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
xinfo() const
{
	stringstream ss;
	ss << "L=" << Lcell << "x" << Ncells;
	if (dLphys==2) ss << "_dLphys=" << dLphys;
	return ss.str();
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
string GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
qinfo() const
{
	stringstream ss;
	int qmin_pi, qmax_pi;
	if      (Q_RANGE_CHOICE==MPI_PPI)  {qmin_pi=-1; qmax_pi=1;}
	else if (Q_RANGE_CHOICE==ZERO_2PI) {qmin_pi=0;  qmax_pi=2;}
	ss << "qmin=" << qmin_pi << "_qmax=" << qmax_pi;
	return ss.str();
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
string GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
tinfo() const
{
	stringstream ss;
	ss << "tmax=" << tmax << "_INT=" << GREENINT_CHOICE;
	return ss.str();
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
string GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
winfo() const
{
	stringstream ss;
	ss << "wmin=" << wmin << "_wmax=" << wmax << "_Ns=" << Ns;
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
xtqw_info() const
{
	return xinfo() + "_" + tinfo() + "_" + qinfo() + "_" + winfo();
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
print_starttext() const
{
	if (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::ON_EXIT)
	{
		#pragma omp critical
		{
			lout << endl << termcolor::colorize << termcolor::bold
			     << "———————————————————————————————————"
			     << " GreenPropagator "
			     << label
			     << " ———————————————————————————————————"
			     <<  termcolor::reset << endl << endl;
		}
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
save (bool IGNORE_TX) const
{
	#pragma omp critical
	{
		bool PRINT = (CHOSEN_VERBOSITY >= DMRG::VERBOSITY::ON_EXIT)? true:false;
		
		IntervalIterator w(wmin,wmax,Nw);
		ArrayXd wvals = w.get_abscissa();
		
		#ifdef GREENPROPAGATOR_USE_HDF5
		lout << "tx-file: " << label+"_"+xt_info()+".h5" << endl;
		lout << "ωq-file: " << label+"_"+xtqw_info()+".h5" << endl;
		HDF5Interface target_tx(label+"_"+xt_info()+".h5",WRITE);
		HDF5Interface target_wq(label+"_"+xtqw_info()+".h5",WRITE);
		target_tx.create_group("G");
		target_wq.create_group("G");
		if (NQ>0)
		{
			for (int iQ=0; iQ<NQ; ++iQ)
			{
				stringstream ss; ss << ".Q" << iQ;
				target_tx.create_group("G"); target_tx.create_group("G"+ss.str());
				target_wq.create_group("G"); target_wq.create_group("G"+ss.str());
			}
		}
		//---
		target_tx.create_group("tinfo");
		target_tx.save_matrix(MatrixXd(tvals),"tvals","tinfo");
		target_tx.save_matrix(MatrixXd(weights),"weights","tinfo");
		target_tx.save_matrix(MatrixXd(tsteps),"tsteps","tinfo");
		target_tx.save_scalar(tmax,"tmax","tinfo");
		//---
		target_wq.create_group("tinfo");
		target_wq.save_matrix(MatrixXd(tvals),"t","");
		#endif
		
//		if (Gtx.size() > 0 or Gwq.size() > 0)
//		{
//			#ifdef GREENPROPAGATOR_USE_HDF5
//			if (PRINT) lout << label << " saving G[txRe], G[txIm], G[ωqRe], G[ωqIm]" << endl;
//			target_tx.save_matrix(MatrixXd(Gtx.real()),"txRe","G");
//			target_tx.save_matrix(MatrixXd(Gtx.imag()),"txIm","G");
//			target_wq.save_matrix(MatrixXd(Gwq.real()),"ωqRe","G");
//			target_wq.save_matrix(MatrixXd(Gwq.imag()),"ωqIm","G");
//			if (G0q.size() > 0)
//			{
//				if (PRINT) lout << label << " saving G[0qRe], G[0qIm]" << endl;
//				target_wq.save_matrix(MatrixXd(G0q.real()),"0qRe","G");
//				target_wq.save_matrix(MatrixXd(G0q.imag()),"0qIm","G");
//			}
//			#else
//			saveMatrix(Gtx.real(), label+"_G=txRe_"+xt_info()+".dat", PRINT);
//			saveMatrix(Gtx.imag(), label+"_G=txIm_"+xt_info()+".dat", PRINT);
//			saveMatrix(Gwq.real(), label+"_G=ωqRe_"+xtqw_info()+".dat", PRINT);
//			saveMatrix(Gwq.imag(), label+"_G=ωqIm_"+xtqw_info()+".dat", PRINT);
//			if (G0q.size() > 0)
//			{
//				saveMatrix(G0q.real(), make_string(label,"_G=0qRe_",xtqw_info(),".dat"), PRINT);
//				saveMatrix(G0q.imag(), make_string(label,"_G=0qIm_",xtqw_info(),".dat"), PRINT);
//			}
//			#endif
//		}
//		
//		if (GtxQmulti.size() > 0 or GwqQmulti.size() > 0)
//		{
//			for (int iQ=0; iQ<NQ; ++iQ)
//			{
//				stringstream ss; ss << ".Q" << iQ;
//				
//				#ifdef GREENPROPAGATOR_USE_HDF5
//				if (PRINT) lout << label << " saving G[txRe], G[txIm], G[ωqRe], G[ωqIm], iQ=" << iQ << endl;
//				target_tx.save_matrix(MatrixXd(GtxQmulti[iQ].real()),"txRe","G"+ss.str());
//				target_tx.save_matrix(MatrixXd(GtxQmulti[iQ].imag()),"txIm","G"+ss.str());
//				target_wq.save_matrix(MatrixXd(GwqQmulti[iQ].real()),"ωqRe","G"+ss.str());
//				target_wq.save_matrix(MatrixXd(GwqQmulti[iQ].imag()),"ωqIm","G"+ss.str());
//				if (G0qQmulti[iQ].size() > 0)
//				{
//					if (PRINT) lout << label << " saving G[0qRe], G[0qIm], iQ=" << iQ << endl;
//					target_wq.save_matrix(MatrixXd(G0qQmulti[iQ].real()),"0qRe","G"+ss.str());
//					target_wq.save_matrix(MatrixXd(G0qQmulti[iQ].imag()),"0qIm","G"+ss.str());
//				}
//				#else
//				saveMatrix(GtxQmulti[iQ].real(), label+"_G=txRe_"+xt_info()+ss.str()+".dat", PRINT);
//				saveMatrix(GtxQmulti[iQ].imag(), label+"_G=txIm_"+xt_info()+ss.str()+".dat", PRINT);
//				saveMatrix(GwqQmulti[iQ].real(), label+"_G=ωqRe_"+xtqw_info()+ss.str()+".dat", PRINT);
//				saveMatrix(GwqQmulti[iQ].imag(), label+"_G=ωqIm_"+xtqw_info()+ss.str()+".dat", PRINT);
//				if (G0qQmulti[iQ].size() > 0)
//				{
//					saveMatrix(G0qQmulti[iQ].real(), make_string(label,"_G=0qRe_",xtqw_info(),ss.str(),".dat"), PRINT);
//					saveMatrix(G0qQmulti[iQ].imag(), make_string(label,"_G=0qIm_",xtqw_info(),ss.str(),".dat"), PRINT);
//				}
//				#endif
//			}
//		}
//		
//		if (Glocw.size() > 0 or Gloct.size() > 0)
//		{
//			#ifdef GREENPROPAGATOR_USE_HDF5
//			if (PRINT) lout << label << " saving G[QDOS], G[t0Re], G[t0Im]" << endl;
//			target_wq.save_matrix(MatrixXd(-M_1_PI*Glocw.imag()),"QDOS","G");
//			target_tx.save_matrix(MatrixXd(Gloct.real()),"t0Re","G");
//			target_tx.save_matrix(MatrixXd(Gloct.imag()),"t0Im","G");
//			#else
//			save_xy(wvals, -M_1_PI*Glocw.imag(), label+"_G=QDOS_"+xt_info()+".dat", PRINT);
//			save_xy(tvals, Gloct.real(), Gloct.imag(), make_string(label,"_G=t0_",xt_info(),".dat"), PRINT);
//			#endif
//		}
//		
//		if (GlocwQmulti.size() > 0 or GloctQmulti.size() > 0)
//		{
//			for (int iQ=0; iQ<NQ; ++iQ)
//			{
//				stringstream ss; ss << ".Q" << iQ;
//				
//				#ifdef GREENPROPAGATOR_USE_HDF5
//				if (PRINT) lout << label << " saving G[QDOS], G[t0Re], G[t0Im], iQ=" << iQ << endl;
//				target_wq.save_matrix(MatrixXd(-M_1_PI*GlocwQmulti[iQ].imag()),"QDOS","G"+ss.str());
//				target_tx.save_matrix(MatrixXd(GloctQmulti[iQ].real()),"t0Re","G"+ss.str());
//				target_tx.save_matrix(MatrixXd(GloctQmulti[iQ].imag()),"t0Im","G"+ss.str());
//				#else
//				save_xy(wvals, -M_1_PI*GlocwQmulti[iQ].imag(), label+"_G=QDOS_"+xt_info()+ss.str()+".dat", PRINT);
//				save_xy(tvals, GloctQmulti[iQ].real(), GloctQmulti[iQ].imag(), make_string(label,"_G=t0_",xt_info(),ss.str(),".dat"), PRINT);
//				#endif
//			}
//		}
		
		if (GtxCell.size() > 0 and !IGNORE_TX)
		{
			for (int i=0; i<Lcell; ++i)
			for (int j=0; j<Lcell; ++j)
			{
				string Gstring = make_string("G",i,j);
				#ifdef GREENPROPAGATOR_USE_HDF5
				if (PRINT) lout << label << " saving " << make_string("G",i,j) << "[txRe], " << Gstring << "[txIm] " << endl;
				target_tx.create_group(make_string("G",i,j));
				target_tx.save_matrix(MatrixXd(GtxCell[i][j].real()),"txRe",Gstring);
				target_tx.save_matrix(MatrixXd(GtxCell[i][j].imag()),"txIm",Gstring);
				#else
				saveMatrix(GtxCell[i][j].real(), make_string(label,"_G=txRe_i=",i,"_j=",j,"_",xt_info(),".dat"), PRINT);
				saveMatrix(GtxCell[i][j].imag(), make_string(label,"_G=txIm_i=",i,"_j=",j,"_",xt_info(),".dat"), PRINT);
				#endif
			}
		}
		if (GwqCell.size() > 0)
		{
			for (int i=0; i<Lcell/Ns; ++i)
			for (int j=0; j<Lcell/Ns; ++j)
			{
				string Gstring = make_string("G",i,j);
				#ifdef GREENPROPAGATOR_USE_HDF5
				if (PRINT) lout << label << " saving " << make_string("G",i,j) << "[ωqRe], " << Gstring << "[ωqIm] " << endl;
				target_wq.create_group(make_string("G",i,j));
				target_wq.save_matrix(MatrixXd(GwqCell[i][j].real()),"ωqRe",Gstring);
				target_wq.save_matrix(MatrixXd(GwqCell[i][j].imag()),"ωqIm",Gstring);
				if (G0qCell.size() > 0)
				{
					if (PRINT) lout << label << " saving " << Gstring << "[0qRe], " << Gstring << "[0qIm]" << endl;
					target_wq.save_matrix(MatrixXd(G0qCell[i][j].real()),"0qRe",Gstring);
					target_wq.save_matrix(MatrixXd(G0qCell[i][j].imag()),"0qIm",Gstring);
				}
				#else
				saveMatrix(GwqCell[i][j].real(), make_string(label,"_G=ωqRe_i=",i,"_j=",j,"_",xtqw_info(),".dat"), PRINT);
				saveMatrix(GwqCell[i][j].imag(), make_string(label,"_G=ωqIm_i=",i,"_j=",j,"_",xtqw_info(),".dat"), PRINT);
				if (G0qCell.size() > 0)
				{
					saveMatrix(G0qCell[i][j].real(), make_string(label,"_G=0qRe_i=",i,"_j=",j,"_",xtqw_info(),".dat"), PRINT);
					saveMatrix(G0qCell[i][j].imag(), make_string(label,"_G=0qIm_i=",i,"_j=",j,"_",xtqw_info(),".dat"), PRINT);
				}
				#endif
			}
		}
		
		if (GlocwCell.size() > 0 or GloctCell.size() > 0)
		{
			for (int i=0; i<Lcell; ++i)
			{
				string Gstring = make_string("G",i);
				#ifdef GREENPROPAGATOR_USE_HDF5
				target_tx.create_group(make_string("G",i));
				target_tx.save_matrix(MatrixXd(GloctCell[i].real()),"t0Re",make_string("G",i));
				target_tx.save_matrix(MatrixXd(GloctCell[i].imag()),"t0Im",make_string("G",i));
				#else
				save_xy(tvals, GloctCell[i].real(), GloctCell[i].imag(), make_string(label,"_G=t0_i=",i,"_",xt_info(),".dat"), PRINT);
				#endif
			}
			for (int i=0; i<Lcell/Ns; ++i)
			{
				string Gstring = make_string("G",i);
				#ifdef GREENPROPAGATOR_USE_HDF5
				if (PRINT) lout << label << " saving " << Gstring << "[QDOS], " << Gstring << "[t0Re], " << Gstring << "[t0Im]" << endl;
				if (PRINT) lout << label << " saving " << Gstring << "[QDOS], " << Gstring << "[t0Re], " << Gstring << "[t0Im]" << endl;
				target_wq.create_group(make_string("G",i));
				target_wq.save_matrix(MatrixXd(-M_1_PI*GlocwCell[i].imag()),"QDOS",make_string("G",i));
				#else
				save_xy(wvals, -M_1_PI*GlocwCell[i].imag(), make_string(label,"_G=QDOS_i=",i,"_",xtqw_info(),".dat"), PRINT);
				#endif
			}
		}
		
		if (ncell.size() > 0)
		{
			if (PRINT) lout << label << " saving ncell" << endl;
			#ifdef GREENPROPAGATOR_USE_HDF5
			target_wq.save_matrix(MatrixXd(ncell),"ncell");
			#else
			saveMatrix(MatrixXd(ncell), "ncell", PRINT);
			#endif
		}
		
		if (!std::isnan(mu))
		{
			if (PRINT) lout << label << " saving μ" << endl;
			#ifdef GREENPROPAGATOR_USE_HDF5
			target_wq.save_scalar(mu,"μ");
			#else
//			saveMatrix(MatrixXd(mu), "μ", PRINT);
			#endif
		}
		
		#ifdef GREENPROPAGATOR_USE_HDF5
		target_tx.close();
		target_wq.close();
		#endif
	}
}

#endif
