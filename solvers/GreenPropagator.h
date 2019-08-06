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
	\param label_input : prefix for saved files (e.g. type of Green's function)
	\param tmax_input : maximal propagation time
	\param Nt_input : amount of time steps; the optimal number seems to be such that the average timestep is 0.1
	\param x0 : the site of the local excitation
	\param wmin_input : minimal frequency for Fourier transform
	\param wmax_input : maximal frequency for Fourier transform
	\param Nw_input : amount of frequency points
	\param tol_compr_input : compression tolerance during time propagation
	\param GAUSSIAN_input : if \p true, compute Gaussian integration weights for the cutoff function
	\param Q_RANGE : choose the q-range (-π to π, 0 to 2π)
	*/
	GreenPropagator (string label_input, 
	                 double tmax_input, int Nt_input, int x0_input, double wmin_input, double wmax_input, int Nw_input=1000, double tol_compr_input=1e-4, 
	                 bool GAUSSINT=true, Q_RANGE Q_RANGE_CHOICE_input=MPI_PPI)
	:label(label_input), tmax(tmax_input), Nt(Nt_input), x0(x0_input), wmin(wmin_input), wmax(wmax_input), Nw(Nw_input), tol_compr(tol_compr_input), 
	 USE_GAUSSIAN_INTEGRATION(GAUSSINT), Q_RANGE_CHOICE(Q_RANGE_CHOICE_input)
	{
		GreenPropagatorCutoff::tmax = tmax;
	}
	
	/**
	Reads G(t,x) from file, so that G(ω,q) can be recalculated.
	\param label_input : prefix for saved files (e.g. type of Green's function)
	\param tmax_input : maximal propagation time
	\param Nt_input : amount of time steps; the optimal number seems to be such that the average timestep is 0.1
	\param x0 : the site of the local excitation
	\param GtxRe : file with the real part of G(ω,q)
	\param GtxIm : file with the imaginary part of G(ω,q)
	\param GAUSSIAN_input : if \p true, compute Gaussian integration weights for the cutoff function
	\param Q_RANGE : choose the q-range (-π to π, 0 to 2π)
	*/
	GreenPropagator (string label_input,
	                 double tmax_input, int Nt_input, int x0_input, 
	                 const MatrixXd &GtxRe, const MatrixXd &GtxIm, 
	                 bool GAUSSINT=true, Q_RANGE Q_RANGE_CHOICE_input=MPI_PPI)
	:label(label_input), tmax(tmax_input), Nt(Nt_input), x0(x0_input), 
	 USE_GAUSSIAN_INTEGRATION(GAUSSINT), Q_RANGE_CHOICE(Q_RANGE_CHOICE_input)
	{
		Gtx = GtxRe + 1.i * GtxIm;
		Nt = Gtx.rows();
		Nq = Gtx.cols();
		Lhetero = Nq;
		make_xarrays(x0,Nq);
		for (int l=0; l<Nq; ++l)
		{
			if (xvals[l] == 0) Gloct = Gtx.col(l);
		}
		GreenPropagatorCutoff::tmax = tmax;
		calc_intweights();
	}
	
	/**
	Computes the Green's function G(t,x).
	\param H_hetero : Hamiltonian of heterogenic section
	\param Phi : infinite ground state with heterogenic section
	\param OxPhi : vector with all local excitations
	\param OxPhi0 : starting state to propagate where the local excitation is at x0
	\param Eg : ground-state energy
	*/
	void compute (const Hamiltonian &H_hetero, const vector<Mps<Symmetry,complex<double>>> &OxPhi, Mps<Symmetry,complex<double>> &OxPhi0, 
	              double Eg, bool TIME_FORWARDS = true);
	
	/**
	Recalculates the t→ω Fourier transform for a different ω-range
	\param wmin_new : minimal frequency for Fourier transform
	\param wmax_new : maximal frequency for Fourier transform
	\param Nw_new : amount of frequency points
	*/
	void recalc_FTw (double wmin_new, double wmax_new, int Nw_new=1000);
	
	/**
	Set a Hermitian operator to be measured in the time-propagated state for testing purposes.
	\param Measure_input : vector of operators, length must be \p Lhetero
	\param measure_interval_input : measure after that many timesteps (it is always measured at zero)
	*/
	void set_measurement (const vector<Mpo<Symmetry,MpoScalar>> &Measure_input, int measure_interval_input=10)
	{
		Measure = Measure_input;
		measure_interval = measure_interval_input;
	}
	
	/**Saves the real and imaginary parts of the Green's function into plain text files.*/
	void save() const;
	
	/**Calculates and saves the selfenergy Σ(ω,q).
	\param SAVE_G0 : Choose whether to save the free Green's function G₀(q,ω) as well.
	*/
	void save_selfenergy (bool SAVE_G0 = true) const;
	
	/**Integrates the QDOS up to a given chemical potential μ (or the Fermi energy, since T=0). 
	   Can be used to find the right μ which gives the chosen filling n.
	\param mu : chemical potential, upper integration limit
	*/
	double integrate_Glocw (double mu);
	
	inline MatrixXcd get_Gtx() const {return Gtx;}
	
private:
	
	string label;
	
	int Nt, Nw, Nq;
	double tmax;
	double wmin, wmax;
	int Lhetero;
	double Eg;
	double tol_compr;
	
	string xinfo() const;
	string qinfo() const;
	string tinfo() const;
	string winfo() const; // w is ω
	string xt_info() const;
	string xtqw_info() const;
	
	SuperQuadrator<GAUSS_LEGENDRE> TimeIntegrator;
	
	ArrayXd tvals, weights, tsteps;
	bool USE_GAUSSIAN_INTEGRATION = true;
	Q_RANGE Q_RANGE_CHOICE = MPI_PPI;
	bool TIME_FORWARDS;
	
	vector<double> xvals;
	vector<int> xinds;
	int x0;
	
	MatrixXcd Gtx, Gtq, Gwq, Gloct;
	ArrayXcd Glocw;
	
	void calc_intweights();
	void make_xarrays (int x0_input, int Lhetero_input);
	void propagate (const Hamiltonian &H_hetero, const vector<Mps<Symmetry,complex<double>>> &OxPhi, Mps<Symmetry,complex<double>> &OxPhi0, 
	                double Eg, bool TIME_FORWARDS);
	void counterpropagate (const Hamiltonian &H_hetero, const Mps<Symmetry,double> &Phi, const Mps<Symmetry,double> &OjxPhi);
	
	void FT_xq();
	void FT_tw();
	ArrayXcd FTloc_tw (const ArrayXd &wvals);
	
	vector<Mpo<Symmetry,MpoScalar>> Measure;
	void measure_wavepacket (const Mps<Symmetry,complex<double>> &Psi, double t);
	int measure_interval;
};

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
compute (const Hamiltonian &H_hetero, const vector<Mps<Symmetry,complex<double>>> &OxPhi, Mps<Symmetry,complex<double>> &OxPhi0, double Eg, bool TIME_FORWARDS)
{
	calc_intweights();
	
	Lhetero = H_hetero.length();
	Nq = Lhetero;
	make_xarrays(x0,Nq);
	
	propagate(H_hetero, OxPhi, OxPhi0, Eg, TIME_FORWARDS);
//	counterpropagate(H_hetero, Phi, OxPhi);
	
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
	
	TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,Mps<Symmetry,complex<double> >> TDVP(H_hetero, Psi);
	EntropyObserver<Mps<Symmetry,complex<double>>> Sobs(Lhetero, Nt, DMRG::VERBOSITY::HALFSWEEPWISE, 1e-2);
	vector<bool> TWO_SITE = Sobs.TWO_SITE(0,Psi);
	lout << endl;
	
	Gtx.resize(Nt,Lhetero); Gtx.setZero();
	Gloct.resize(Nt,Lhetero); Gloct.setZero();
	
	IntervalIterator t(0.,tmax,Nt);
	double tval = 0.;
	
	// 0.1. measure wavepacket at t=0
	measure_wavepacket(Psi,0);
	
	// 0.2. if no (open) integration weights, calculate G at t=0
	if (!USE_GAUSSIAN_INTEGRATION)
	{
		Stopwatch<> StepTimer;
		#pragma omp parallel for
		for (size_t l=0; l<Lhetero; ++l)
		{
			Gtx(0,l) = -1.i * exp(-1.i*tsign*Eg*tval) * dot_hetero(OxPhi[l], Psi);
			if (xvals[l] == 0) Gloct(0) = Gtx(0,l); // save local Green's function
		}
		lout << StepTimer.info("G(t,x) calculation") << endl;
		lout << endl;
	}
	
	Stopwatch<> TimePropagationTimer;
	for (t=t.begin(); t!=t.end(); ++t)
	{
		Stopwatch<> StepTimer;
		// 1. propagate
		TDVP.t_step_adaptive(H_hetero, Psi, 1.i*tsign*tsteps(t.index()), TWO_SITE, 1,1e-6);
//		TDVP.t_step(H_hetero, Psi, 1.i*tsign*tsteps(t.index()), 1,1e-8);
		tval += tsteps(t.index());
		lout << StepTimer.info("time step") << endl;
		
		if (Psi.get_truncWeight().sum() > 0.5*tol_compr)
		{
			Psi.max_Nsv = min(static_cast<size_t>(max(Psi.max_Nsv*1.1, Psi.max_Nsv+1.)),200ul);
			lout << termcolor::yellow << "Setting Psi.max_Nsv to " << Psi.max_Nsv << termcolor::reset << endl;
		}
		
		lout << TDVP.info() << endl;
		lout << Psi.info() << endl;
		lout << "propagated to t=" << tval << ", stepsize=" << tsteps(t.index()) << endl;
		
		// 2. measure
		// 2.1. Green's function
		#pragma omp parallel for
		for (size_t l=0; l<Lhetero; ++l)
		{
			Gtx(t.index(),l) = -1.i * exp(-1.i*tsign*Eg*tval) * dot_hetero(OxPhi[l], Psi);
			if (xvals[l] == 0) Gloct(t.index()) = Gtx(t.index(),l); // save local Green's function
		}
		lout << StepTimer.info("G(t,x) calculation") << endl;
		// 2.2. measure wavepacket at t
		if ((t.index()-1)%measure_interval == 0 and t.index() > 1) measure_wavepacket(Psi,tval);
		lout << StepTimer.info("wavepacket measurement") << endl;
		
		// 3. check entropy increase
		auto PsiTmp = Psi;
		PsiTmp.eps_svd = 1e-15;
		PsiTmp.skim(DMRG::BROOM::SVD);
		double r = (t.index()==0)? 1.:tsteps(t.index()-1)/tsteps(t.index());
		TWO_SITE = Sobs.TWO_SITE(t.index(), PsiTmp, r);
		lout << StepTimer.info("entropy calculation") << endl;
		
		lout << TimePropagationTimer.info("total running time",false) << endl;
		lout << endl;
	}
	
	// measure wavepacket at t=t_end
	measure_wavepacket(Psi,tval);
	
	saveMatrix(Gtx.real(), label+"_GtxRe_"+xt_info()+".dat");
	saveMatrix(Gtx.imag(), label+"_GtxIm_"+xt_info()+".dat");
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
measure_wavepacket (const Mps<Symmetry,complex<double>> &Psi, double t)
{
	if (Measure.size() != 0)
	{
		double norm = Psi.squaredNorm();
		ArrayXd res(Measure.size());
		
		#pragma omp parallel for
		for (size_t l=0; l<Measure.size(); ++l)
		{
			res(l) = isReal(avg_hetero(Psi, Measure[l], Psi)) / norm;
		}
		
		ofstream Filer(make_string(label,"_Mx_",xinfo(),"_t=",t,".dat"));
		for (size_t l=0; l<Measure.size(); ++l)
		{
			Filer << xvals[l] << "\t" << res(l) << endl;
		}
		Filer.close();
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
make_xarrays (int x0_input, int Lhetero_input)
{
	xinds.resize(Lhetero_input);
	xvals.resize(Lhetero_input);
	
	iota(begin(xinds),end(xinds),0);
	
	for (int ix=0; ix<xinds.size(); ++ix)
	{
		xvals[ix] = static_cast<double>(xinds[ix]-x0_input);
	}
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
calc_intweights()
{
	Stopwatch<> Watch;
	TimeIntegrator = SuperQuadrator<GAUSS_LEGENDRE>(GreenPropagatorCutoff::gauss,0.,tmax,Nt);
	lout << Watch.info("integration weights") << endl;
	
	if (USE_GAUSSIAN_INTEGRATION)
	{
		tvals   = TimeIntegrator.get_abscissa();
		weights = TimeIntegrator.get_weights();
		tsteps  = TimeIntegrator.get_steps();
		
		ofstream Filer(make_string("weights_",tinfo(),".dat"));
		for (int i=0; i<Nt; ++i)
		{
			Filer << tvals(i) << "\t" << weights(i) << endl;
		}
		Filer.close();
		
		double erf2 = 0.995322265018952734162069256367252928610891797040060076738; // Mathematica♥
		double integral = 0.25*sqrt(M_PI)*erf2*tmax;
		lout << termcolor::blue 
		     << setprecision(14)
		     << "integration weight test: ∫w(t)dt=" << weights.sum() 
		     << ", analytical=" << integral 
		     << ", diff=" << abs(weights.sum()-integral) 
		     << termcolor::reset << endl;
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


//template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
//void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
//counterpropagate (const Hamiltonian &H_hetero, const Mps<Symmetry,double> &Phi, const Mps<Symmetry,double> &OjxPhi)
//{
//	Eg = avg_hetero(Phi, H_hetero, Phi, true);
//	lout << "Eg=" << Eg << endl;
//	
//	Mps<Symmetry,complex<double> > PsiF = OjxPhi.template cast<complex<double> >();
//	Mps<Symmetry,complex<double> > PsiB = OjxPhi.template cast<complex<double> >();
//	PsiF.eps_svd = tol_compr;
//	PsiB.eps_svd = tol_compr;
//	PsiF.max_Nsv = PsiF.calc_Dmax();
//	PsiB.max_Nsv = PsiB.calc_Dmax();
//	
//	TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,Mps<Symmetry,complex<double> >> TDVP_F(H_hetero, PsiF);
//	TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,Mps<Symmetry,complex<double> >> TDVP_B(H_hetero, PsiB);
//	EntropyObserver<Mps<Symmetry,complex<double>>> SobsF(Lhetero, Nt, DMRG::VERBOSITY::HALFSWEEPWISE);
//	EntropyObserver<Mps<Symmetry,complex<double>>> SobsB(Lhetero, Nt, DMRG::VERBOSITY::HALFSWEEPWISE);
//	vector<bool> TWO_SITE_F = SobsF.TWO_SITE(0, PsiF);
//	vector<bool> TWO_SITE_B = SobsB.TWO_SITE(0, PsiB);
//	lout << endl;
//	
//	Gtx.resize(Nt,Lhetero); Gtx.setZero();
//	
//	IntervalIterator x(-0.5*(Lhetero-1), 0.5*(Lhetero-1), Lhetero);
//	IntervalIterator t(0.,tmax,Nt);
//	double tval = 0.;
////	auto Phic = Phi.template cast<complex<double> >();
////	
////	// If no (open) integration weights, measure at t=0
////	if (!USE_GAUSSIAN_INTEGRATION)
////	{
////		lout << "measure at t=0" << endl;
////		Stopwatch<> ContractionTimer;
////		#pragma omp parallel for
////		for (size_t l=0; l<Lhetero; ++l)
////		{
////			Gtx(0,l) = -1.i * exp(1.i*Eg*tval) * avg_hetero(Phic, Odagi[l], Psi);
////		}
////		lout << ContractionTimer.info("contractions") << endl;
////		lout << endl;
////	}
//	
//	if (Measure.size() != 0)
//	{
//		double norm = OjxPhi.squaredNorm();
//		for (x=x.begin(); x!=x.end(); ++x)
//		{
//			x << avg_hetero(OjxPhi, Measure[x.index()], OjxPhi) / norm;
//		}
//		x.save(make_string("Mx_",xqinfo(),"_t=0.dat"));
//	}
//	
//	Stopwatch<> TimePropagationTimer;
//	for (t=t.begin(); t!=t.end(); ++t)
//	{
//		// 1. propagate
//		#pragma omp parallel sections
//		{
//			#pragma omp section
//			{
//				TDVP_F.t_step_adaptive(H_hetero, PsiF, -1.i*0.5*tsteps(t.index()), TWO_SITE_F, 1,1e-6);
//			}
//			#pragma omp section
//			{
//				TDVP_B.t_step_adaptive(H_hetero, PsiB, +1.i*0.5*tsteps(t.index()), TWO_SITE_B, 1,1e-6);
//			}
//		}
//		tval += tsteps(t.index());
//		
//		if (PsiF.get_truncWeight().sum() > 0.5*tol_compr)
//		{
//			PsiF.max_Nsv = min(static_cast<size_t>(max(PsiF.max_Nsv*1.1, PsiF.max_Nsv+1.)),200ul);
//			lout << termcolor::yellow << "Setting Psi.max_Nsv to " << PsiF.max_Nsv << termcolor::reset << endl;
//		}
//		
//		if (PsiB.get_truncWeight().sum() > 0.5*tol_compr)
//		{
//			PsiB.max_Nsv = min(static_cast<size_t>(max(PsiB.max_Nsv*1.1, PsiB.max_Nsv+1.)),200ul);
//			lout << termcolor::yellow << "Setting Psi.max_Nsv to " << PsiB.max_Nsv << termcolor::reset << endl;
//		}
//		
//		lout << "===forwards===" << endl;
//		lout << TDVP_F.info() << endl;
//		lout << PsiF.info() << endl;
//		
//		lout << "===backwards===" << endl;
//		lout << TDVP_B.info() << endl;
//		lout << PsiB.info() << endl;
//		
//		lout << "propagated to t=" << tval << ", stepsize=" << tsteps(t.index()) << endl;
//		
//		// 2. measure
//		Stopwatch<> ContractionTimer;
//		// 2.1. Green's function
////		#pragma omp parallel for
//		for (x=x.begin(); x!=x.end(); ++x)
//		{
//			Gtx(t.index(),x.index()) = -1.i * exp(1.i*Eg*tval) * dot_hetero(PsiB,PsiF,*x);
//		}
//		// 2.2. a corr. function with the time-evolved state
//		if (Measure.size() != 0 and t.index()%measure_interval==0)
//		{
//			double norm = PsiF.squaredNorm();
//			for (x=x.begin(); x!=x.end(); ++x)
//			{
//				x << avg_hetero(PsiF, Measure[x.index()], PsiF).real() / norm;
//			}
//			x.save(make_string("Mx_",xqinfo(),"_t=",tval,".dat"));
//		}
//		lout << ContractionTimer.info("contractions") << endl;
//		
//		// 3. check entropy increase
//		double r = (t.index()==0)? 1.:tsteps(t.index()-1)/tsteps(t.index());
//		
//		auto PsiFtmp = PsiF;
//		PsiFtmp.eps_svd = 1e-15;
//		PsiFtmp.skim(DMRG::BROOM::SVD);
//		TWO_SITE_F = SobsF.TWO_SITE(t.index(), PsiFtmp, r);
//		
//		auto PsiBtmp = PsiB;
//		PsiBtmp.eps_svd = 1e-15;
//		PsiBtmp.skim(DMRG::BROOM::SVD);
//		TWO_SITE_B = SobsB.TWO_SITE(t.index(), PsiBtmp, r);
//		
//		lout << TimePropagationTimer.info("total running time",false) << endl;
//		lout << endl;
//	}
//	
//	saveMatrix(Gtx.real(),make_string("GtxRe_",xqt_info(),".dat"));
//	saveMatrix(Gtx.imag(),make_string("GtxIm_",xqt_info(),".dat"));
//}

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
	// phase shift factor because the origin site is at x0
	for (int iq=0; iq<Nq; ++iq)
	{
		Gtq.col(iq) *= exp(-1.i*2.*M_PI/double(xvals.size())*xvals[0]*double(iq));
	}
	lout << FourierWatch.info("FFT x→q") << endl;
	
//	// Explicit FT for testing
//	for (int it=0; it<Nt; ++it)
//	for (int iq=0; iq<Nq; ++iq)
//	{
//		double qval = 2.*M_PI/Nq * iq;
//		for (int ix=0; ix<Nq; ++ix)
//		{
//			Gtq(it,iq) += Gtx(it,ix) * exp(-1.i*qval*xvals[ix]);
//		}
//	}
//	lout << FourierWatch.info("FT x→q") << endl;
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
	#pragma omp parallel for
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
	lout << FourierWatch.info("FT t→ω") << endl;
	
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
	Nq += 1;
	
	// Calculate local Green's function
	Glocw = FTloc_tw(wvals);
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
recalc_FTw (double wmin_new, double wmax_new, int Nw_new)
{
	wmin = wmin_new;
	wmax = wmax_new;
	Nw = Nw_new;
	
//	twinfo = make_string(txinfo,"_wmin=",wmin,"_wmax=",wmax,"_Nw=",Nw);
	FT_xq();
	FT_tw();
	save();
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
ArrayXcd GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
FTloc_tw (const ArrayXd &wvals)
{
	assert(USE_GAUSSIAN_INTEGRATION);
	ArrayXcd res(wvals.rows()); res.setZero();
	
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
			
			res(iw) += weights(it) * damping * exp(+1.i*wval*tval) * Gloct(it);
		}
	}
	
	return res;
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
double GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
integrate_Glocw (double mu)
{
	Quadrator<GAUSS_LEGENDRE> Q;
	int Nint = 200;
	
	ArrayXd wabscissa(Nint);
	for (int i=0; i<Nint; ++i) {wabscissa(i) = Q.abscissa(i,wmin,mu,Nint);}
	
	ArrayXd QDOS = -1.*M_1_PI * FTloc_tw(wabscissa).imag();
	
	return (Q.get_weights(wmin,mu,Nint) * QDOS).sum();
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
string GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
xinfo() const
{
	stringstream ss;
	ss << "x0=" << x0 << "_L=" << Lhetero;
	return ss.str();
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
string GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
qinfo() const
{
	stringstream ss;
	int qmin, qmax;
	if      (Q_RANGE_CHOICE==MPI_PPI)  {qmin=-1; qmax=1;}
	else if (Q_RANGE_CHOICE==ZERO_2PI) {qmin=0;  qmax=2;}
	ss << "qmin=" << qmin << "_qmax=" << qmax << "_Nq=" << Nq;
	return ss.str();
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
string GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
tinfo() const
{
	stringstream ss;
	ss << "tmax=" << tmax << "_Nt=" << Nt;
	return ss.str();
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
string GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
winfo() const
{
	stringstream ss;
	ss << "wmin=" << wmin << "_wmax=" << wmax << "_Nw=" << Nw;
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
save() const
{
	saveMatrix(Gtx.real(), label+"_GtxRe_"+xt_info()+".dat", true); // PRINT = true
	saveMatrix(Gtx.imag(), label+"_GtxIm_"+xt_info()+".dat", true);
	
	saveMatrix(Gwq.real(), label+"_GωqRe_"+xtqw_info()+".dat", true);
	saveMatrix(Gwq.imag(), label+"_GωqIm_"+xtqw_info()+".dat", true);
	
	IntervalIterator w(wmin,wmax,Nw);
	ArrayXd wvals = w.get_abscissa();
	save_xy(wvals, -M_1_PI*Glocw.imag(), label+"_QDOS_"+xt_info()+".dat", true);
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
save_selfenergy (bool SAVE_G0) const
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
			Swq(iw,iq) = wvals(iw)+2.*cos(qvals(iq))-pow(Gwq(iw,iq),-1);
			G0wq(iw,iq)     = pow(wvals(iw)+2.*cos(qvals(iq))+1.i*1e-1,-1);
		}
	}
	
	saveMatrix(Swq.real(), label+"_ΣωqRe_"+xtqw_info()+".dat", true); // PRINT = true
	saveMatrix(Swq.imag(), label+"_ΣωqIm_"+xtqw_info()+".dat", true);
	
	if (SAVE_G0)
	{
		saveMatrix(G0wq.real(), label+"_G0ωqRe_"+xtqw_info()+".dat", true);
		saveMatrix(G0wq.imag(), label+"_G0ωqIm_"+xtqw_info()+".dat", true);
	}
}

#endif
