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
//#include "VUMPS/UmpsCompressor.h"
//#include "models/Heisenberg.h"
//#include "models/HeisenbergU1.h"
#include "models/HeisenbergSU2.h"
#include "DmrgLinearAlgebra.h"
#include "solvers/TDVPPropagator.h"
#include "solvers/EntropyObserver.h"
//#include "models/SpinlessFermionsU1.h"
//#include "models/HubbardU1xU1.h"
#include "models/HubbardSU2xU1.h"

#include "IntervalIterator.h"
#include "Quadrator.h"
#define CHEBTRANS_DONT_USE_FFTWOMP
#include "SuperQuadrator.h"
#include "solvers/GreenPropagator.h"

size_t L, Ncells, Lhetero, x0;
int M, Dtot, N;
double J, V, U, tPrime;
double tmax, dt, tol_compr;
int Nt;
size_t Chi, max_iter, min_iter, Qinit, D;
double tol_eigval, tol_var, tol_state;
bool GAUSSINT, RELOAD, CELL;
double wmin, wmax;

// set model here:
//#define FERMIONS
#define HEISENBERG

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
	omp_set_nested(1);
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",2);
	Ncells = args.get<size_t>("Ncells",30);
	cout << "Ncells=" << Ncells << endl;
	Lhetero = L*Ncells;
	x0 = Lhetero/2;
	tPrime = args.get<double>("tPrime",0.);
	J = args.get<double>("J",1.);
	V = args.get<double>("V",4.);
	U = args.get<double>("U",0.);
	M = args.get<int>("M",0);
	Dtot = abs(M)+1;
	N = args.get<int>("N",L);
	D = args.get<size_t>("D",3ul);
	GAUSSINT = static_cast<bool>(args.get<int>("GAUSSINT",true));
	RELOAD = static_cast<bool>(args.get<int>("RELOAD",false));
	CELL = static_cast<bool>(args.get<int>("CELL",true));
	
	#ifdef HEISENBERG
	wmin = args.get<double>("wmin",-1.);
	wmax = args.get<double>("wmax",+6.);
	#elif defined(FERMIONS)
	wmin = args.get<double>("wmin",-10.);
	wmax = args.get<double>("wmax",+10.);
	#endif
	
	if (CELL)
	{
		dt = args.get<double>("dt",0.2);
	}
	else
	{
		dt = args.get<double>("dt",0.1);
	}
	tmax = args.get<double>("tmax",10.);
	Nt = static_cast<int>(tmax/dt);
	tol_compr =  args.get<double>("tol_compr",1e-4);
	lout << "Nt=" << Nt << endl;
	lout << "tol_compr=" << tol_compr << endl;
	
	Chi = args.get<size_t>("Chi",2);
	tol_eigval = args.get<double>("tol_eigval",1e-5);
	tol_var = args.get<double>("tol_var",1e-5);
	tol_state = args.get<double>("tol_state",1e-4);
	
	max_iter = args.get<size_t>("max_iter",150ul);
	min_iter = args.get<size_t>("min_iter",100ul);
	Qinit = args.get<size_t>("Qinit",6ul);
	
	VUMPS::CONTROL::GLOB GlobParams;
	GlobParams.min_iterations = min_iter;
	GlobParams.max_iterations = max_iter;
	GlobParams.Dinit = Chi;
	GlobParams.Dlimit = Chi;
	GlobParams.Qinit = Qinit;
	GlobParams.tol_eigval = tol_eigval;
	GlobParams.tol_var = tol_var;
	GlobParams.tol_state = tol_state;
	GlobParams.max_iter_without_expansion = 20ul;
	
	// reload:
	if (RELOAD)
	{
//		if (CELL)
//		{
//			vector<vector<MatrixXcd>> GinCell(L); for (int i=0; i<L; ++i) {GinCell[i].resize(L);}
//			for (int i=0; i<L; ++i) 
//			for (int j=0; j<L; ++j)
//			{
//				GinCell[i][j].resize(Nt,Ncells);
//				GinCell[i][j].setZero();
//			}
//			
//			for (int i=0; i<L; ++i)
//			for (int j=0; j<L; ++j)
//			{
//				GinCell[i][j] +=     readMatrix(make_string("PES_G=txRe_i=",i,"_j=",j,"_L=",Lhetero,"_tmax=",tmax,".dat"))+
//				                 1.i*readMatrix(make_string("PES_G=txIm_i=",i,"_j=",j,"_L=",Lhetero,"_tmax=",tmax,".dat"))+
//				                     readMatrix(make_string("IPE_G=txRe_i=",i,"_j=",j,"_L=",Lhetero,"_tmax=",tmax,".dat"))+
//				                 1.i*readMatrix(make_string("IPE_G=txIm_i=",i,"_j=",j,"_L=",Lhetero,"_tmax=",tmax,".dat"));
//			}
//			
//			Gfull = GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> >("A1P",tmax,GinCell,true,ZERO_2PI);
//			
//			Gfull.recalc_FTwCell(wmin,wmax,1000);
//			Gfull.FT_allSites();
//			Gfull.diagonalize_and_save_cell();
//		}
//		else
//		{
//			MatrixXcd Gin =      readMatrix(make_string("PES_G=txRe_L=",Lhetero,"_tmax=",tmax,".dat"))+
//				            +1.i*readMatrix(make_string("PES_G=txIm_L=",Lhetero,"_tmax=",tmax,".dat"))+
//				                 readMatrix(make_string("IPE_G=txRe_L=",Lhetero,"_tmax=",tmax,".dat"))+
//				            +1.i*readMatrix(make_string("IPE_G=txIm_L=",Lhetero,"_tmax=",tmax,".dat"));
//			
//			Gfull = GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> >("A1P",L,tmax,Gin,true,ZERO_2PI);
//			
//			Gfull.recalc_FTw(wmin,wmax,1000);
//		}
//		
//		Gfull.save();
//		
//		IntervalIterator mu(wmin,wmax,101);
//		for (mu=mu.begin(); mu!=mu.end(); ++mu)
//		{
//			double res = spinfac * Gfull.integrate_Glocw(*mu);
//			mu << res;
//		}
//		mu.save("n(μ).dat");
//		RootFinder R(n_mu,wmin,wmax);
//		lout << "μ=" << R.root() << endl;
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
		MODEL H(L,{{"U",U},{"tPrime",tPrime},{"OPEN_BC",false},{"CALC_SQUARE",false}}); // spinful
		#endif
		lout << H.info() << endl;
		H.transform_base(Q,true); // PRINT=true
		
		MODEL::uSolver uDMRG(DMRG::VERBOSITY::HALFSWEEPWISE);
		Eigenstate<MODEL::StateUd> g;
		uDMRG.userSetGlobParam();
		uDMRG.GlobParam = GlobParams;
		uDMRG.edgeState(H,g,Q);
//		g.state.save("Umps");
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
		MODEL H_hetero(Lhetero,{{"U",U},{"tPrime",tPrime},{"OPEN_BC",false},{"CALC_SQUARE",false}}); // spinful
		#endif
		lout << H_hetero.info() << endl;
		H_hetero.transform_base(Q,false,L); // PRINT=false
		H_hetero.precalc_TwoSiteData(true); // FORCE=true
		
		// O
		std::array<vector<Mpo<MODEL::Symmetry,double>>,2> O;
		O[0].resize(L);
		O[1].resize(L);
		for (int l=0; l<L; ++l)
		{
			#ifdef HEISENBERG
//			O[l] = H_hetero.Scomp(SP,l);
			O[0][l]= H_hetero.S(Lhetero/2+l,0,1.);
			#elif defined(FERMIONS)
			O[0][l] = H_hetero.c(Lhetero/2+l,0,1.);
			O[1][l] = H_hetero.cdag(Lhetero/2+l,0,1.);
			#endif
			O[0][l].transform_base(Q,false,L); // PRINT=false
			O[1][l].transform_base(Q,false,L);
			
			cout << "l=" << l << ", loc=" << Lhetero/2+l << endl;
		}
		// Ofull
		std::array<vector<Mpo<MODEL::Symmetry,double>>,2> Ofull;
		Ofull[0].resize(Lhetero);
		Ofull[1].resize(Lhetero);
		for (int l=0; l<Lhetero; ++l)
		{
			#ifdef HEISENBERG
//			Ofull[l] = H_hetero.Scomp(SP,l);
			Ofull[0][l]= H_hetero.S(l,0,1.);
			#elif defined(FERMIONS)
			Ofull[0][l] = H_hetero.c(l,0,1.);
			Ofull[1][l] = H_hetero.cdag(l,0,1.);
			#endif
			Ofull[0][l].transform_base(Q,false,L); // PRINT=false
			Ofull[1][l].transform_base(Q,false,L);
		}
		
		// OxV for all sites
		Stopwatch<> OxVTimer;
		Mps<MODEL::Symmetry,double> Phi = uDMRG.create_Mps(Ncells, g, H, x0, false); // ground state as heterogenic MPS
		// OxPhi
		std::array<vector<Mps<MODEL::Symmetry,complex<double>>>,2> OxPhi;
		OxPhi[0].resize(L);
		OxPhi[1].resize(L);
		std::array<vector<Mps<MODEL::Symmetry,double>>,2> OxPhi_tmp;
		OxPhi_tmp[0] = uDMRG.create_Mps(Ncells, g, H, O[0][0], O[0], false);
		OxPhi_tmp[1] = uDMRG.create_Mps(Ncells, g, H, O[1][0], O[1], false);
		for (int l=0; l<L; ++l)
		{
			OxPhi[0][l] = OxPhi_tmp[0][l].template cast<complex<double>>();
			OxPhi[1][l] = OxPhi_tmp[1][l].template cast<complex<double>>();
		}
		// OxPhiFull
		std::array<vector<Mps<MODEL::Symmetry,complex<double>>>,2> OxPhiFull;
		OxPhiFull[0].resize(Lhetero);
		OxPhiFull[1].resize(Lhetero);
		std::array<vector<Mps<MODEL::Symmetry,double>>,2> OxPhiFull_tmp;
		OxPhiFull_tmp[0] = uDMRG.create_Mps(Ncells, g, H, Ofull[0][x0], Ofull[0], false);
		OxPhiFull_tmp[1] = uDMRG.create_Mps(Ncells, g, H, Ofull[1][x0], Ofull[1], false);
		for (int l=0; l<Lhetero; ++l)
		{
			OxPhiFull[0][l] = OxPhiFull_tmp[0][l].template cast<complex<double>>();
			OxPhiFull[1][l] = OxPhiFull_tmp[1][l].template cast<complex<double>>();
		}
		
		// GreenPropagator
		#ifdef HEISENBERG
		Green[0] = GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> >("SSF",tmax,Nt,wmin,wmax,501,ZERO_2PI,qpoints,DIRECT);
		Green[1] = GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> >("SSF",tmax,Nt,wmin,wmax,501,ZERO_2PI,qpoints,DIRECT);
		#elif defined(FERMIONS) // TIME_FORWARDS=true/false
		Green[0] = GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> >("PES",tmax,Nt,wmin,wmax,501,ZERO_2PI,qpoints,DIRECT);
		Green[1] = GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> >("IPE",tmax,Nt,wmin,wmax,501,ZERO_2PI,qpoints,DIRECT);
		#endif
		
		Green[0].set_verbosity(DMRG::VERBOSITY::ON_EXIT);
		
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
		for (int z=0; z<2; ++z)
		{
			#ifdef HEISENBERG
			Green[z].set_measurement(Measure,4,"SdagS","measure");
			#elif defined(FERMIONS)
			Green[z].set_measurement(Measure,4,"n","measure");
			#endif
		}
		
		double Eg = avg_hetero(Phi, H_hetero, Phi, true); // USE_BOUNDARY=true
		lout << setprecision(14) << "Eg=" << Eg << endl;
		
		std::array<bool,2> TIME_DIR;
		TIME_DIR[0] = false;
		TIME_DIR[1] = true;
		
//		cout << endl;
//		auto Phic = Phi.cast<complex<double>>();
//		auto PhiTmp = Phi.cast<complex<double>>();
//		TDVPPropagator<MODEL,MODEL::Symmetry,double,complex<double>,Mps<MODEL::Symmetry,complex<double>>> TDVP(H_hetero, PhiTmp);
//		TDVP.t_step0(H_hetero, PhiTmp, 1.i*0.2, 1,1e-8);
//		cout << TDVP.info() << endl;
//		complex<double> resdot = dot(PhiTmp,Phic);
//		cout << resdot << ", " << exp(-1.i*Eg*0.2) << ", diff=" << resdot-exp(-1.i*Eg*0.2) << ", " << abs(resdot-exp(-1.i*Eg*0.2)) << endl;
//		cout << endl;
		
		#pragma omp parallel for
		for (int z=0; z<2; ++z)
		{
//			Green[z].OxPhiFull = OxPhiFull[z];
			
			if (CELL)
			{
				Green[z].compute_cell(H_hetero, OxPhi[z], Eg, TIME_DIR[z]);
				Green[z].FT_allSites();
			}
			else
			{
				Green[z].compute(H_hetero, OxPhiFull[z][Lhetero/2], Eg, TIME_DIR[z]);
			}
		}
		
		if (CELL)
		{
			vector<vector<MatrixXcd>> GinCell(L); for (int i=0; i<L; ++i) {GinCell[i].resize(L);}
			for (int i=0; i<L; ++i) 
			for (int j=0; j<L; ++j)
			{
				GinCell[i][j].resize(Nt,Ncells);
				GinCell[i][j].setZero();
			}
			
			for (int i=0; i<L; ++i)
			for (int j=0; j<L; ++j)
			{
				GinCell[i][j] += Green[0].get_GtxCell()[i][j] + Green[1].get_GtxCell()[i][j];
			}
			
			Gfull = GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> >("A1P",tmax,GinCell,true,ZERO_2PI); // GAUSSINT=true
			
			Gfull.recalc_FTwCell(wmin,wmax,1000);
			Gfull.FT_allSites();
			Gfull.diagonalize_and_save_cell();
		}
		else
		{
			MatrixXcd Gin = Green[0].get_Gtx() + Green[1].get_Gtx();
			
			Gfull = GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> >("A1P",L,tmax,Gin,true,ZERO_2PI); // GAUSSINT=true
			
			Gfull.recalc_FTw(wmin,wmax,1000);
		}
		
		Gfull.save();
		
		IntervalIterator mu(wmin,wmax,101);
		for (mu=mu.begin(); mu!=mu.end(); ++mu)
		{
			double res = spinfac * Gfull.integrate_Glocw(*mu);
			mu << res;
		}
		mu.save("n(μ).dat");
		RootFinder R(n_mu,wmin,wmax);
		lout << "μ=" << R.root() << endl;
	}
}
