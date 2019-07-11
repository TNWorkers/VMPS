#define LANCZOS_MAX_ITERATIONS 1e2

#define USE_HDF5_STORAGE
#define DMRG_DONT_USE_OPENMP
#define MPSQCOMPRESSOR_DONT_USE_OPENMP

#include <iostream>
#include <fstream>
#include <complex>

#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

#include <gsl/gsl_sf_ellint.h>

#include "Logger.h"
Logger lout;
#include "ArgParser.h"

#include "util/LapackManager.h"

#include "StringStuff.h"
#include "Stopwatch.h"

#include <Eigen/Core>
using namespace Eigen;
#include <unsupported/Eigen/FFT>

//-------- Test of the ArnoldiSolver:
//size_t dim (const MatrixXcd &A) {return A.rows();}
//#include "LanczosWrappers.h"
//#include "HxV.h"

#include "VUMPS/VumpsSolver.h"
#include "VUMPS/VumpsLinearAlgebra.h"
#include "VUMPS/UmpsCompressor.h"
#include "models/Heisenberg.h"
#include "models/HeisenbergU1.h"
#include "models/HeisenbergSU2.h"
#include "DmrgLinearAlgebra.h"
#include "solvers/TDVPPropagator.h"
#include "solvers/EntropyObserver.h"

#include "IntervalIterator.h"
#include "Quadrator.h"
#define CHEBTRANS_DONT_USE_FFTWOMP
#include "SuperQuadrator.h"

size_t L, Ncells, Lhetero;
int M, Dtot, N;
double J;
double tmax, dt, tol_compr;
size_t Chi, max_iter, min_iter, Qinit, D;
double tol_eigval, tol_var, tol_state;

double w (double tval)
{
	return exp(-pow(2.*tval/tmax,2));
}

double one (double tval)
{
	return 1.;
}

int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",2);
	Ncells = args.get<size_t>("Ncells",30);
	Lhetero = L*Ncells+1;
	J = args.get<double>("J",1.);
	M = args.get<int>("M",0);
	Dtot = abs(M)+1;
	N = args.get<int>("N",L/2);
	D = args.get<size_t>("D",3ul);
	bool LOAD = static_cast<bool>(args.get<int>("LOAD",0));
	
	dt = args.get<double>("dt",0.1);
	tmax = args.get<double>("tmax",10.);
	int Nt = static_cast<int>(tmax/dt);
	tol_compr = 1e-4;
	
	Chi = args.get<size_t>("Chi",4);
	tol_eigval = args.get<double>("tol_eigval",1e-7);
	tol_var = args.get<double>("tol_var",1e-6);
	tol_state = args.get<double>("tol_state",1e-2);
	
	max_iter = args.get<size_t>("max_iter",100ul);
	min_iter = args.get<size_t>("min_iter",1ul);
	Qinit = args.get<size_t>("Qinit",6ul);
	
	VUMPS::CONTROL::GLOB GlobParams;
	GlobParams.min_iterations = min_iter;
	GlobParams.max_iterations = max_iter;
	GlobParams.Dinit  = Chi;
	GlobParams.Dlimit = Chi;
	GlobParams.Qinit = Qinit;
	GlobParams.tol_eigval = tol_eigval;
	GlobParams.tol_var = tol_var;
	GlobParams.tol_state = tol_state;
	GlobParams.max_iter_without_expansion = 20ul;
	
//	typedef VMPS::HeisenbergSU2 MODEL;
//	qarray<MODEL::Symmetry::Nq> Qc = {Dtot};
//	cout << "Dtot=" << Dtot << endl;
//	MODEL H(L,{{"J",J},{"OPEN_BC",false},{"D",D},{"CALC_SQUARE",false}});
//	lout << H.info() << endl;
//	H.transform_base(Qc);
//	
//	MODEL::uSolver uDMRG(DMRG::VERBOSITY::HALFSWEEPWISE);
//	Eigenstate<MODEL::StateUd> g;
//	uDMRG.userSetGlobParam();
//	uDMRG.userSetLanczosParam();
//	uDMRG.GlobParam = GlobParams;
//	uDMRG.edgeState(H, g, Qc);
	
	
	typedef VMPS::HeisenbergU1 MODEL;
	qarray<MODEL::Symmetry::Nq> Q = MODEL::Symmetry::qvacuum();
	
	Quadrator<GAUSS_LEGENDRE> Quad;
	SuperQuadrator<GAUSS_LEGENDRE> SuperQuad(w,0.,tmax,Nt);
	
	VectorXd tvals   = SuperQuad.get_abscissa();
	VectorXd weights = SuperQuad.get_weights();
	VectorXd tsteps  = SuperQuad.get_steps();
	
	ofstream Filer("weights.dat");
	for (int i=0; i<Nt; ++i)
	{
		lout << tvals(i) << "\t" << weights(i) << endl;
		Filer << tvals(i) << "\t" << weights(i) << endl;
	}
	Filer.close();
//	
//	ArrayXd tvals   = Quad.get_abscissa(0,tmax,Nt);
//	ArrayXd weights = Quad.get_weights (0,tmax,Nt);
//	ArrayXd tsteps  = Quad.get_steps   (0,tmax,Nt);
	
//	ArrayXd tvals(Nt);
//	for (int i=0; i<Nt; ++i)
//	{
//		tvals(i) = i*0.1;
//	}
//	ArrayXd weights(Nt); weights = 1.;
//	ArrayXd tsteps(Nt); tsteps = 0.1;
	
	MatrixXcd Gtx(Nt,Lhetero); Gtx.setZero();
	
	for (int i=0; i<Nt; ++i)
	{
		cout << "t=" << tvals(i) << ", step=" << tsteps(i) << ", w=" << weights(i) << endl;
	}
	
	IntervalIterator x(-0.5*(Lhetero-1), 0.5*(Lhetero-1), Lhetero);
	IntervalIterator t(0.,tmax,Nt);
	
	if (!LOAD)
	{
		MODEL H(L,{{"J",J},{"D",D},{"OPEN_BC",false}});
		lout << H.info() << endl;
	//	H.transform_base(Q);
		
		MODEL::uSolver uDMRG(DMRG::VERBOSITY::HALFSWEEPWISE);
		Eigenstate<MODEL::StateUd> g;
		uDMRG.userSetGlobParam();
		uDMRG.userSetLanczosParam();
		uDMRG.GlobParam = GlobParams;
		uDMRG.edgeState(H,g,Q);
		
		Mps<MODEL::Symmetry,double> Phi = uDMRG.create_Mps(Ncells, H, g, true);
		lout << Phi.info() << endl;
		
		MODEL::StateXd AxPhi;
		
		MODEL H_hetero(Lhetero,{{"J",1.},{"D",D},{"OPEN_BC",false}});
		Phi.graph("before");
		OxV_exact(H_hetero.Scomp(SP,Ncells), Phi, AxPhi, 2., DMRG::VERBOSITY::HALFSWEEPWISE);
	//	OxV_exact(H_hetero.S(Ncells), Phi, AxPhi, 2., DMRG::VERBOSITY::HALFSWEEPWISE);
		AxPhi.graph("after");
		AxPhi.sweep(0,DMRG::BROOM::QR);
		lout << AxPhi.info() << endl << endl;
		
		double Eg = avg_hetero(Phi, H_hetero, Phi, true);
		lout << setprecision(14) << "Eg=" << Eg << ", " << Eg/(Lhetero+2.) << ", " << g.energy << endl;
		
		for (x=x.begin(); x!=x.end(); ++x)
		{
			double res = avg_hetero(AxPhi, H_hetero.Scomp(SZ,x.index()), AxPhi);
			lout << "x=" << x.index() << ", <Sz>=" << res << endl;
			x << res;
		}
		lout << endl;
		x.save(make_string("Sz_L=",Lhetero,"_t=0.dat"));
		
		Mps<MODEL::Symmetry,complex<double> > Psi = AxPhi.cast<complex<double> >();
		Psi.eps_svd = tol_compr;
		Psi.max_Nsv = Psi.calc_Dmax();
		
		TDVPPropagator<MODEL,MODEL::Symmetry,double,complex<double>,MODEL::StateXcd> TDVP(H_hetero, Psi);
		EntropyObserver<MODEL::StateXcd> Sobs(Lhetero,Nt,DMRG::VERBOSITY::HALFSWEEPWISE);
		vector<bool> TWO_SITE = Sobs.TWO_SITE(0,Psi);
		
		double tval = 0.;
		
		Stopwatch<> TimePropagationTimer;
		for (t=t.begin(); t!=t.end(); ++t)
		{
			// propagate
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
			
			// measure
			Stopwatch<> ContractionTimer;
			for (size_t l=0; l<Lhetero; ++l)
			{
				Gtx(t.index(),l) = -1.i * exp(1.i*Eg*tval) * avg_hetero(Phi.cast<complex<double> >(), H_hetero.Scomp(SM,l), Psi);
			}
			lout << ContractionTimer.info("contractions") << endl;
			
//			if (t.index()%10 == 0 and t.index() > 0)
//			{
//				for (x=x.begin(); x!=x.end(); ++x)
//				{
//					double res = isReal(avg_hetero(Psi, H_hetero.Sz(x.index()), Psi));
//					lout << "x=" << x.index() << ", <Sz>=" << res << endl;
//					x << res;
//				}
//				x.save(make_string("Sz_L=",Lhetero,"_t=",tval,".dat"));
//			}
			
			
			// determine entropy
			auto PsiTmp = Psi;
			PsiTmp.eps_svd = 1e-15;
			PsiTmp.skim(DMRG::BROOM::SVD);
			double r = (t.index()==0)? 1.:tsteps(t.index()-1)/tsteps(t.index());
			TWO_SITE = Sobs.TWO_SITE(t.index(), PsiTmp, r);
			lout << endl;
		}
		lout << TimePropagationTimer.info("full time propagation") << endl;
		
		saveMatrix(Gtx.real(),"GtxRe.dat");
		saveMatrix(Gtx.imag(),"GtxIm.dat");
	}
	else
	{
		Gtx = loadMatrix("GtxRe.dat") + 1.i * loadMatrix("GtxIm.dat");
	}
	
//	VectorXcd v(Lhetero); v.setRandom();
//	VectorXcd u(Lhetero); u.setZero();
//	for (int k=0; k<Lhetero; ++k)
//	for (int j=0; j<Lhetero; ++j)
//	{
//		u(k) += exp(-1.i*2.*M_PI/Lhetero*double((j-0.5*(Lhetero-1))*k)) * v(j);
//	}
	
	// Fourier transform
	int Nw = 1e3;
	int Nq = Lhetero; //Lhetero+1;
	IntervalIterator w(0.,3.5,Nw);
	IntervalIterator q(0.,2.*M_PI,Nq);
	MatrixXcd Gwq(Nw,Nq); Gwq.setZero();
	
	ArrayXd wvals = w.get_abscissa();
//	ArrayXd qvals = q.get_abscissa();
	ArrayXd qvals(Lhetero);
	for (int k=0; k<Lhetero; ++k) qvals(k) = k*2.*M_PI/Lhetero;
	ArrayXd xvals = x.get_abscissa();
	
//	for (int iq=0; iq<qvals.rows(); ++iq)
//	for (int ix=0; ix<xvals.rows(); ++ix)
//	{
//		double xval = xvals(ix);
//		double qval = qvals(iq);
//		u(iq) += exp(-1.i*qval*xval) * v(ix);
//	}
//	
//	VectorXcd z(Lhetero);
	Eigen::FFT<double> fft;
//	fft.fwd(z,v);
	
//	for (int k=0; k<Lhetero; ++k)
//	{
//		z(k) *= exp(1.i*M_PI*double(Lhetero-1.)/double(Lhetero)*double(k));
//	}
//	
//	for (int i=0; i<Lhetero; ++i)
//	{
//		cout << u(i) << "\t" << z(i) << endl;
//	}
	
	Stopwatch<> FourierWatch;
	
	MatrixXcd Gtq(Nt,Nq); Gtq.setZero();
	
	for (int it=0; it<tvals.rows(); ++it)
	{
		VectorXcd v;
		fft.fwd(v,Gtx.row(it));
		Gtq.row(it) = v;
		
		for (int iq=0; iq<qvals.rows(); ++iq)
		{
			Gtq(it,iq) *= exp(1.i*M_PI*(Lhetero-1.)/double(Lhetero)*double(iq));
		}
	}
	lout << FourierWatch.info("FFT x->q") << endl;
	
	#pragma omp parallel for
	for (int iw=0; iw<wvals.rows(); ++iw)
	{
		double wval = wvals(iw);
		
		for (int it=0; it<tvals.rows(); ++it)
		{
			double tval = tvals(it);
			
			Gwq.row(iw) += weights(it) * 
//			              exp(-pow(2.*tval/tmax,2)) * 
			              exp(+1.i*wval*tval) * 
			              Gtq.row(it);
		}
	}
	lout << FourierWatch.info("FT t->w") << endl;
	
	Gwq.conservativeResize(Nw,Nq+1);
	Gwq.col(Nq) = Gwq.col(0);
	saveMatrix(Gwq.real(),"GwqRe.dat");
	saveMatrix(Gwq.imag(),"GwqIm.dat");
	
	MatrixXcd Gwq_(Nw,Nq); Gwq_.setZero();
//	for (w=w.begin(); w!=w.end(); ++w)
//	for (q=q.begin(); q!=q.end(); ++q)
	#pragma omp parallel for collapse(2)
	for (int iw=0; iw<wvals.rows(); ++iw)
	for (int iq=0; iq<qvals.rows(); ++iq)
	{
		double wval = wvals(iw);
		double qval = qvals(iq);
		
//		for (x=x.begin(); x!=x.end(); ++x)
//		for (t=t.begin(); t!=t.end(); ++t)
		for (int ix=0; ix<xvals.rows(); ++ix)
		for (int it=0; it<tvals.rows(); ++it)
		{
			double tval = tvals(it);
			double xval = xvals(ix);
			Gwq_(iw,iq) += weights(it) * 
//			              exp(-pow(2.*tval/tmax,2)) * 
			              exp(+1.i*wval*tval) * 
			              exp(-1.i*qval*xval) * 
			              Gtx(it,ix);
		}
	}
	lout << FourierWatch.info("FT") << endl;
	
	
	Gwq_.conservativeResize(Nw,Nq+1);
	Gwq_.col(Nq) = Gwq_.col(0);
	
	saveMatrix(Gwq_.real(),"GwqRe_.dat");
	saveMatrix(Gwq_.imag(),"GwqIm_.dat");
	saveMatrix(-M_1_PI*Gwq.imag(),"S.dat");
	
	
	
	
}
