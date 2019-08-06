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
#include "models/SpinlessFermionsU1.h"
#include "models/HubbardU1xU1.h"
#include "models/HubbardSU2xU1.h"

#include "IntervalIterator.h"
#include "Quadrator.h"
#define CHEBTRANS_DONT_USE_FFTWOMP
#include "SuperQuadrator.h"
#include "solvers/GreenPropagator.h"

size_t L, Ncells, Lhetero, x0;
int M, Dtot, N;
double J, V, U;
double tmax, dt, tol_compr;
int Nt;
size_t Chi, max_iter, min_iter, Qinit, D;
double tol_eigval, tol_var, tol_state;
bool GAUSSINT, RELOAD;
double wmin, wmax;

// set model here:
#define FERMIONS
//#define HEISENBERG

#ifdef HEISENBERG
typedef VMPS::HeisenbergSU2 MODEL;
#elif defined(FERMIONS)
//typedef VMPS::SpinlessFermionsU1 MODEL; // spinless
typedef VMPS::HubbardSU2xU1 MODEL; double spinfac=1.; // spinful
//typedef VMPS::HubbardU1xU1 MODEL; double spinfac=2.; // spinful
#endif

std::array<GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double>>,2> Green;
GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double>> Gfull;

#include "RootFinder.h" // from ALGS
double n_mu (double mu, void*)
{
	return spinfac * Gfull.integrate_Glocw(mu) - N/L;
}

int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",2);
	Ncells = args.get<size_t>("Ncells",31);
	cout << "Ncells=" << Ncells << endl;
	Lhetero = L*Ncells;
	x0 = Ncells;
	J = args.get<double>("J",1.);
	V = args.get<double>("V",4.);
	U = args.get<double>("U",6.);
	M = args.get<int>("M",0);
	Dtot = abs(M)+1;
	N = args.get<int>("N",L/2);
	D = args.get<size_t>("D",3ul);
	GAUSSINT = static_cast<bool>(args.get<int>("GAUSSINT",true));
	
	RELOAD = static_cast<bool>(args.get<int>("RELOAD",false));
	#ifdef HEISENBERG
	wmin = args.get<double>("wmin",-1.);
	wmax = args.get<double>("wmax",+6.);
	#elif defined(FERMIONS)
	wmin = args.get<double>("wmin",-5.);
	wmax = args.get<double>("wmax",+10.);
	#endif
	
	dt = args.get<double>("dt",0.1);
	tmax = args.get<double>("tmax",10.);
	Nt = static_cast<int>(tmax/dt);
	tol_compr = 1e-4;
	lout << "Nt=" << Nt << endl;
	
	Chi = args.get<size_t>("Chi",4);
	tol_eigval = args.get<double>("tol_eigval",1e-5);
	tol_var = args.get<double>("tol_var",1e-5);
	tol_state = args.get<double>("tol_state",1e-2);
	
	max_iter = args.get<size_t>("max_iter",50ul);
	min_iter = args.get<size_t>("min_iter",20ul);
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
	
	// reload:
	if (RELOAD)
	{
		cout << "tmax=" << ", Nt=" << Nt << ", Ncells=" << Ncells << ", wmin=" << wmin << ", wmax=" << wmax << endl;
		#ifdef HEISENBERG
		Q_RANGE QR = ZERO_2PI;
		#elif defined(FERMIONS)
		Q_RANGE QR = MPI_PPI;
		#endif
		Gfull = GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> >
		("A1P",tmax,Nt,x0,loadMatrix(make_string("PES_GtxRe_x0=",Ncells,"_L=",Lhetero,"_tmax=",tmax,"_Nt=",Nt,".dat"))+
		                  loadMatrix(make_string("IPE_GtxRe_x0=",Ncells,"_L=",Lhetero,"_tmax=",tmax,"_Nt=",Nt,".dat")),
		                  loadMatrix(make_string("PES_GtxIm_x0=",Ncells,"_L=",Lhetero,"_tmax=",tmax,"_Nt=",Nt,".dat"))+
		                  loadMatrix(make_string("IPE_GtxIm_x0=",Ncells,"_L=",Lhetero,"_tmax=",tmax,"_Nt=",Nt,".dat")),
		true,QR); // GAUSSINT=true
		Gfull.recalc_FTw(wmin,wmax,1000);
		
		#ifdef FERMIONS
		Gfull.save_selfenergy();
		IntervalIterator mu(wmin,wmax,101);
		for (mu=mu.begin(); mu!=mu.end(); ++mu)
		{
			double res = spinfac * Gfull.integrate_Glocw(*mu);
			mu << res;
		}
		mu.save("n(mu).dat");
		RootFinder R(n_mu,wmin,wmax);
		lout << "mu=" << R.root() << endl;
		#endif
	}
	else
	{
		#ifdef HEISENBERG
		qarray<MODEL::Symmetry::Nq> Q = MODEL::Symmetry::qvacuum();
		#elif defined(FERMIONS)
	//	qarray<MODEL::Symmetry::Nq> Q = {N};   // spinless
		qarray<MODEL::Symmetry::Nq> Q = MODEL::singlet(N); // spinful
		#endif
		
		#ifdef HEISENBERG
		MODEL H(L,{{"J",J},{"D",D},{"OPEN_BC",false},{"CALC_SQUARE",false}});
		#elif defined(FERMIONS)
	//	MODEL H(L,{{"V",V},{"OPEN_BC",false},{"CALC_SQUARE",false}}); // spinless
		MODEL H(L,{{"U",U},{"OPEN_BC",false},{"CALC_SQUARE",false}}); // spinful
		#endif
		lout << H.info() << endl;
		H.transform_base(Q,true); // PRINT=true
		
		MODEL::uSolver uDMRG(DMRG::VERBOSITY::HALFSWEEPWISE);
		Eigenstate<MODEL::StateUd> g;
		uDMRG.userSetGlobParam();
		uDMRG.GlobParam = GlobParams;
		uDMRG.edgeState(H,g,Q);
		g.state.save("Umps");
	//	g.state.load("Umps");
	//	lout << g.state.info() << endl;
	//	g.state.graph("Umps");
		
		#if defined(FERMIONS)
		for (int l=0; l<L; ++l)
		{
//			lout << "l=" << l 
//			     << " n=" << avg(g.state, H.n<UP>(l), g.state) 
//			     << ", "  << avg(g.state, H.n<DN>(l), g.state) 
//			     << ", d=" << avg(g.state, H.d(l), g.state) 
//			     << endl;
			lout << "l=" << l 
			     << " n=" << avg(g.state, H.n(l), g.state) 
			     << ", d=" << avg(g.state, H.d(l), g.state) 
			     << endl;
		}
		#endif
		
		#ifdef HEISENBERG
		MODEL H_hetero(Lhetero,{{"J",1.},{"D",D},{"OPEN_BC",false},{"CALC_SQUARE",false}});
		#elif defined(FERMIONS)
	//	MODEL H_hetero(Lhetero,{{"V",V},{"OPEN_BC",false},{"CALC_SQUARE",false}}); // spinless
		MODEL H_hetero(Lhetero,{{"U",U},{"OPEN_BC",false},{"CALC_SQUARE",false}}); // spinful
		#endif
		lout << H_hetero.info() << endl;
		H_hetero.transform_base(Q,false,L); // PRINT=false
		H_hetero.precalc_TwoSiteData(true); // FORCE=true
		
		// create vector of O
		std::array<vector<Mpo<MODEL::Symmetry,double>>,2> Obra;
		Obra[0].resize(Lhetero);
		Obra[1].resize(Lhetero);
		for (int l=0; l<Lhetero; ++l)
		{
			#ifdef HEISENBERG
//			Obra[l] = H_hetero.Scomp(SP,l);
			Obra[0][l]= H_hetero.S(l,0,sqrt(3.));
			#elif defined(FERMIONS)
			Obra[0][l] = H_hetero.c(l,0,1.);
			Obra[1][l] = H_hetero.cdag(l,0,1.);
			#endif
			Obra[0][l].transform_base(Q,false,L); // PRINT=false
			Obra[1][l].transform_base(Q,false,L);
		}
		#ifdef HEISENBERG
		Mpo<MODEL::Symmetry,double> Oket = H_hetero.S(x0);
		#elif defined(FERMIONS)
//		Mpo<MODEL::Symmetry,double> Oket = H_hetero.c(x0,0,1.);
		std::array<Mpo<MODEL::Symmetry,double>,2> Oket;
		Oket[0] = H_hetero.c(x0,0,1.);
		Oket[1] = H_hetero.cdag(x0,0,1.);
		#endif
		Oket[0].transform_base(Q,false,L); // PRINT=false
		Oket[1].transform_base(Q,false,L);
		
		// OxV for all sites
		Stopwatch<> OxVTimer;
		Mps<MODEL::Symmetry,double> Phi = uDMRG.create_Mps(Ncells, g, H, x0, false); // ground state as heterogenic MPS
		//---
		std::array<vector<Mps<MODEL::Symmetry,complex<double>>>,2> OxPhi;
		OxPhi[0].resize(Lhetero);
		OxPhi[1].resize(Lhetero);
		std::array<vector<Mps<MODEL::Symmetry,double>>,2> OxPhi_tmp;
		OxPhi_tmp[0] = uDMRG.create_Mps(Ncells, g, H, Obra[0][x0], Obra[0], false);
		OxPhi_tmp[1] = uDMRG.create_Mps(Ncells, g, H, Obra[1][x0], Obra[1], false);
		for (int l=0; l<Lhetero; ++l)
		{
			OxPhi[0][l] = OxPhi_tmp[0][l].template cast<complex<double>>();
			OxPhi[1][l] = OxPhi_tmp[1][l].template cast<complex<double>>();
		}
		std::array<Mps<MODEL::Symmetry,complex<double>>,2> OxPhi0;
		OxPhi0[0] = uDMRG.create_Mps(Ncells, g, H, Oket[0], Oket[0], false).template cast<complex<double>>();
		OxPhi0[1] = uDMRG.create_Mps(Ncells, g, H, Oket[1], Oket[1], false).template cast<complex<double>>();
		lout << "dot_hetero(OxPhi0[0],OxPhi0[0])=" << isReal(dot_hetero(OxPhi0[0],OxPhi0[0])) << endl;
		lout << "dot_hetero(OxPhi0[1],OxPhi0[1])=" << isReal(dot_hetero(OxPhi0[1],OxPhi0[1])) << endl;
		lout << OxVTimer.info("OxV for all sites") << endl;
		
//		// some testing
//		#ifdef FERMIONS
//		int DeltaCells = 4;
//		cout << "Ncells=" << Ncells << endl;
//		auto Op1 = H_hetero.nn(Ncells-DeltaCells*L,Ncells); Op1.transform_base(Q,false,L);
//		cout << "nn=" << avg_hetero(Phi, Op1, Phi) << endl;
//		//---
//		auto Op2 = H_hetero.n(Ncells); Op2.transform_base(Q,false,L);
//		MODEL::StateXd O2xPhi = uDMRG.create_Mps(Ncells, g, H, Op2, Op2, false);
//		cout << "dot(shift=-)" << DeltaCells << ": " << dot_hetero(O2xPhi,O2xPhi,-DeltaCells) << endl;
//		cout << "dot(shift=+)" << DeltaCells << ": " << dot_hetero(O2xPhi,O2xPhi,+DeltaCells) << endl;
//		#endif
		
		// GreenPropagator
		#ifdef HEISENBERG
		Green[0] = GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> >("SSF",tmax,Nt,x0,wmin,wmax,1000,1e-4,GAUSSINT,ZERO_2PI);
		Green[1] = GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> >("SSF",tmax,Nt,x0,wmin,wmax,1000,1e-4,GAUSSINT,ZERO_2PI);
		#elif defined(FERMIONS) // TIME_FORWARDS=true/false
		Green[0] = GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> >("PES",tmax,Nt,x0,wmin,wmax,1000,1e-4,GAUSSINT,MPI_PPI);
		Green[1] = GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> >("IPE",tmax,Nt,x0,wmin,wmax,1000,1e-4,GAUSSINT,MPI_PPI);
		#endif
		
		// set operator to measure
		vector<Mpo<MODEL::Symmetry,double>> Measure;
		#ifdef HEISENBERG
		Measure.resize(Lhetero-1);
		#elif defined(FERMIONS)
		Measure.resize(Lhetero);
		#endif
		for (int l=0; l<Measure.size(); ++l)
		{
			#ifdef HEISENBERG
//			Measure[l] = H_hetero.SzSz(l,l+1);
			Measure[l] = H_hetero.SdagS(l,l+1);
			#elif defined(FERMIONS)
			Measure[l] = H_hetero.n(l);
			#endif
			Measure[l].transform_base(Q,false,L); // PRINT=false
		}
		Green[0].set_measurement(Measure,10);
		Green[1].set_measurement(Measure,10);
		
		double Eg = avg_hetero(Phi, H_hetero, Phi, true);
		lout << setprecision(14) << "Eg=" << Eg << ", (Lhetero+1)*e0=" << g.energy*(Lhetero+1) << ", diff=" << abs(Eg-g.energy*(Lhetero+1)) << endl;
		
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				// PES
				auto H_hetero_copy = H_hetero;
				Green[0].compute(H_hetero_copy, OxPhi[0], OxPhi0[0], Eg, false); // TIME_FORWARDS=true
			}
			#pragma omp section
			{
				// IPE
				auto H_hetero_copy = H_hetero;
				Green[1].compute(H_hetero_copy, OxPhi[1], OxPhi0[1], Eg, true); // TIME_FORWARDS=true
			}
		}
		
		Gfull = GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> >
		("A1P",tmax,Nt,x0,Green[0].get_Gtx().real()+Green[1].get_Gtx().real(),
		                  Green[0].get_Gtx().imag()+Green[1].get_Gtx().imag(),
		true,MPI_PPI); // GAUSSINT=true
		Gfull.recalc_FTw(wmin,wmax,1000);
		
		#ifdef FERMIONS
		Gfull.save_selfenergy();
		IntervalIterator mu(wmin,wmax,101);
		for (mu=mu.begin(); mu!=mu.end(); ++mu)
		{
			double res = spinfac * Gfull.integrate_Glocw(*mu);
			mu << res;
		}
		mu.save("n(mu).dat");
		RootFinder R(n_mu,wmin,wmax);
		lout << "mu=" << R.root() << endl;
		#endif
	}
}
