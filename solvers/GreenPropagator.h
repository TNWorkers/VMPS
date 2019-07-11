#ifndef GREEN_PROPAGATOR
#define GREEN_PROPAGATOR

#include "SuperQuadrator.h"
#include "solvers/TDVPPropagator.h"
#include "IntervalIterator.h"

#include <unsupported/Eigen/FFT>

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
	
	/**
	\param[in] tmax_input : maximal propagation time
	\param[in] Nt_input : amount of time steps; the optimal number seems to be such that the average timestep is 0.1
	\param[in] wmin_input : minimal frequency for Fourier transform
	\param[in] wmax_input : maximal frequency for Fourier transform
	\param[in] Nw_input : amount of frequency points
	\param[in] tol_compr_input : compression tolerance during time propagation
	\param[in] GAUSSIAN_input : if \p true, compute Gaussian integration weights for the cutoff function
	*/
	GreenPropagator (double tmax_input, int Nt_input, double wmin_input, double wmax_input, int Nw_input=1000, double tol_compr_input=1e-4, 
	                 bool GAUSSIAN_input=true)
	:tmax(tmax_input), Nt(Nt_input), wmin(wmin_input), wmax(wmax_input), Nw(Nw_input), tol_compr(tol_compr_input),
	 USE_GAUSSIAN_INTEGRATION(GAUSSIAN_input)
	{
		GreenPropagatorCutoff::tmax = tmax;
		tinfo  = make_string("tmax=",tmax,"_Nt=",Nt);
		twinfo = make_string(tinfo,"_wmin=",wmin,"_wmax=",wmax,"_Nw=",Nw);
	}
	
	/**
	\param[in] H_hetero : Hamiltonian of heterogenic section
	\param[in] Phi : infinite ground state with heterogenic section
	\param[in] Oj : excitation operator, must be at central site of heterogenic section (need only one because of translational invariance)
	\param[in] Odagi : adjoint operator for all sites of the heterogenic section
	*/
	void compute (const Hamiltonian &H_hetero, const Mps<Symmetry,double> &Phi, const Mpo<Symmetry,MpoScalar> &Oj, const vector<Mpo<Symmetry,MpoScalar>> &Odagi);
	
	/**
	Recalculates the t→ω Fourier transform for a different ω-range
	\param[in] wmin_new : minimal frequency for Fourier transform
	\param[in] wmax_new : maximal frequency for Fourier transform
	\param[in] Nw_new : amount of frequency points
	*/
	void recalc_FTw (double wmin_new, double wmax_new, int Nw_new=1000);
	
private:
	
	int Nt, Nw, Nq;
	double tmax;
	double wmin, wmax;
	int Lhetero;
	double Eg;
	double tol_compr;
	string tinfo, twinfo;
	
	SuperQuadrator<GAUSS_LEGENDRE> TimeIntegrator;
	
	ArrayXd tvals, weights, tsteps;
	bool USE_GAUSSIAN_INTEGRATION = true;
	
	TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,Mps<Symmetry,complex<double> >> TDVP;
	EntropyObserver<Mps<Symmetry,complex<double>>> Sobs;
	vector<bool> TWO_SITE;
	
	MatrixXcd Gtx, Gtq, Gwq;
	
	void calc_tinfo();
	void propagate (const Hamiltonian &H_hetero, const Mps<Symmetry,double> &Phi, const Mps<Symmetry,double> &OjxPhi, const vector<Mpo<Symmetry,MpoScalar> > &Odagi);
	void FT_xq();
	void FT_tw();
};

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
compute (const Hamiltonian &H_hetero, const Mps<Symmetry,double> &Phi, const Mpo<Symmetry,MpoScalar> &Oj, const vector<Mpo<Symmetry,MpoScalar>> &Odagi)
{
	calc_tinfo();
	
	Lhetero = H_hetero.length();
	Nq = Lhetero;
	
	Mps<Symmetry,double> OjxPhi;
	OxV_exact(Oj, Phi, OjxPhi, 2., DMRG::VERBOSITY::HALFSWEEPWISE);
	OjxPhi.sweep(0,DMRG::BROOM::QR);
	
	propagate(H_hetero, Phi, OjxPhi, Odagi);
	
	saveMatrix(Gtx.real(),"GtxRe_"+tinfo+".dat");
	saveMatrix(Gtx.imag(),"GtxIm_"+tinfo+".dat");
	
	FT_xq();
	FT_tw();
	saveMatrix(Gwq.real(),"GwqRe_"+twinfo+".dat");
	saveMatrix(Gwq.imag(),"GwqIm_"+twinfo+".dat");
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
calc_tinfo()
{
	TimeIntegrator = SuperQuadrator<GAUSS_LEGENDRE>(GreenPropagatorCutoff::gauss,0.,tmax,Nt);
	
	if (USE_GAUSSIAN_INTEGRATION)
	{
		tvals   = TimeIntegrator.get_abscissa();
		weights = TimeIntegrator.get_weights();
		tsteps  = TimeIntegrator.get_steps();
		
		ofstream Filer(make_string("weights_",tinfo,".dat"));
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

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
propagate (const Hamiltonian &H_hetero, const Mps<Symmetry,double> &Phi, const Mps<Symmetry,double> &OjxPhi, const vector<Mpo<Symmetry,MpoScalar>> &Odagi)
{
	Eg = avg_hetero(Phi, H_hetero, Phi, true);
	
	Mps<Symmetry,complex<double> > Psi = OjxPhi.template cast<complex<double> >();
	Psi.eps_svd = tol_compr;
	Psi.max_Nsv = Psi.calc_Dmax();
	
	TDVP = TDVPPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar,Mps<Symmetry,complex<double> >>(H_hetero, Psi);
	Sobs = EntropyObserver<Mps<Symmetry,complex<double>>>(Lhetero,Nt,DMRG::VERBOSITY::HALFSWEEPWISE);
	TWO_SITE = Sobs.TWO_SITE(0,Psi);
	lout << endl;
	
	Gtx.resize(Nt,Lhetero); Gtx.setZero();
	
	IntervalIterator x(-0.5*(Lhetero-1), 0.5*(Lhetero-1), Lhetero);
	IntervalIterator t(0.,tmax,Nt);
	double tval = 0.;
	double Eg = avg_hetero(Phi, H_hetero, Phi, true);
	auto Phic = Phi.template cast<complex<double> >();
	
	// If no (open) integration weights, measure at t=0
	if (!USE_GAUSSIAN_INTEGRATION)
	{
		lout << "measure at t=0" << endl;
		Stopwatch<> ContractionTimer;
		for (size_t l=0; l<Lhetero; ++l)
		{
			Gtx(0,l) = -1.i * exp(1.i*Eg*tval) * avg_hetero(Phic, Odagi[l], Psi);
		}
		lout << ContractionTimer.info("contractions") << endl;
		lout << endl;
	}
	
	Stopwatch<> TimePropagationTimer;
	for (t=t.begin(); t!=t.end(); ++t)
	{
		// 1. propagate
		TDVP.t_step_adaptive(H_hetero, Psi, -1.i*tsteps(t.index()), TWO_SITE, 1,1e-6);
		tval += tsteps(t.index());
		
		if (Psi.get_truncWeight().sum() > 0.5*tol_compr)
		{
			Psi.max_Nsv = min(static_cast<size_t>(max(Psi.max_Nsv*1.1, Psi.max_Nsv+1.)),200ul);
			lout << termcolor::yellow << "Setting Psi.max_Nsv to " << Psi.max_Nsv << termcolor::reset << endl;
		}
		
		lout << TDVP.info() << endl;
		lout << Psi.info() << endl;
		lout << "propagated to t=" << tval << ", stepsize=" << tsteps(t.index()) << endl;
		
		// 2. measure
		Stopwatch<> ContractionTimer;
		for (size_t l=0; l<Lhetero; ++l)
		{
			Gtx(t.index(),l) = -1.i * exp(1.i*Eg*tval) * avg_hetero(Phic, Odagi[l], Psi);
		}
		lout << ContractionTimer.info("contractions") << endl;
		
		// 3. determine entropy
		auto PsiTmp = Psi;
		PsiTmp.eps_svd = 1e-15;
		PsiTmp.skim(DMRG::BROOM::SVD);
		double r = (t.index()==0)? 1.:tsteps(t.index()-1)/tsteps(t.index());
		TWO_SITE = Sobs.TWO_SITE(t.index(), PsiTmp, r);
		
		lout << TimePropagationTimer.info("total running time",false) << endl;
		lout << endl;
	}
	
	saveMatrix(Gtx.real(),make_string("GtxRe_",tinfo,".dat"));
	saveMatrix(Gtx.imag(),make_string("GtxIm_",tinfo,".dat"));
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
FT_xq()
{
	IntervalIterator w(wmin,wmax,Nw);
	Gtq.resize(Nt,Lhetero); Gtq.setZero();
	
	Stopwatch<> FourierWatch;
	
	// Use FFT to transform from x to q
	Eigen::FFT<double> fft;
	for (int it=0; it<tvals.rows(); ++it)
	{
		VectorXcd vtmp;
		fft.fwd(vtmp,Gtx.row(it));
		Gtq.row(it) = vtmp;
		
		// phase shift factor because the origin site is in the middle
		for (int iq=0; iq<Nq; ++iq)
		{
			Gtq(it,iq) *= exp(1.i*M_PI*(Lhetero-1.)/double(Lhetero)*double(iq));
		}
	}
	lout << FourierWatch.info("FFT x→q") << endl;
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
			
			// If Gaussian integration is employed, the damping is already included in weights
			double damping = (USE_GAUSSIAN_INTEGRATION)? 1.:exp(-pow(2.*tval/tmax,2));
			
			Gwq.row(iw) += weights(it) * damping * exp(+1.i*wval*tval) * Gtq.row(it);
		}
	}
	lout << FourierWatch.info("FT t→ω") << endl;
	
	Gwq.conservativeResize(Nw,Nq+1);
	Gwq.col(Nq) = Gwq.col(0);
}

template<typename Hamiltonian, typename Symmetry, typename MpoScalar, typename TimeScalar>
void GreenPropagator<Hamiltonian,Symmetry,MpoScalar,TimeScalar>::
recalc_FTw (double wmin_new, double wmax_new, int Nw_new)
{
	wmin = wmin_new;
	wmax = wmax_new;
	Nw = Nw_new;
	twinfo = make_string(tinfo,"_wmin=",wmin,"_wmax=",wmax,"_Nw=",Nw);
	FT_tw();
}

#endif
