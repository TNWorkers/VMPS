#ifdef BLAS
#include "util/LapackManager.h"
#pragma message("LapackManager")
#endif

#define LANCZOS_MAX_ITERATIONS 1e2

#define USE_HDF5_STORAGE
#define DMRG_DONT_USE_OPENMP
#define MPSQCOMPRESSOR_DONT_USE_OPENMP

#include <iostream>
#include <fstream>
#include <complex>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

#include <gsl/gsl_sf_ellint.h>

#include "Logger.h"
Logger lout;

#include "StringStuff.h"
#include "Stopwatch.h"

#include <Eigen/Core>
using namespace Eigen;
#include <unsupported/Eigen/FFT>

#include "VUMPS/VumpsSolver.h"
#include "VUMPS/VumpsLinearAlgebra.h"
#include "DmrgLinearAlgebra.h"
#include "solvers/TDVPPropagator.h"
#include "solvers/EntropyObserver.h"

#include "IntervalIterator.h"
#include "Quadrator.h"
#define CHEBTRANS_DONT_USE_FFTWOMP
#include "SuperQuadrator.h"
#define GREENPROPAGATOR_USE_HDF5
#include "solvers/GreenPropagator.h"
#include "RootFinder.h" // from ALGS

size_t L, Lhetero, Ncells, x0;
int M, Dtot, N;
double U, V, Vxy, Vz, J;
double tAB, tpx, tpxPrime;
double tmax, dt, tol_compr;
int Nt;
size_t Chi, max_iter, min_iter, Qinit, D;
double tol_eigval, tol_var, tol_state, tol_oxv;
GREEN_INTEGRATION INT;
double wmin, wmax; int wpoints, qpoints;
string wd;
string RELOAD;
DMRG::VERBOSITY::OPTION VERB;
vector<string> specs; int Nspec;

//#define USE_HEISENBERG
//string MODELNAME = "Heisenberg";
//Q_RANGE QR = ZERO_2PI;

//#include "models/HeisenbergSU2.h"
//typedef VMPS::HeisenbergSU2 MODEL; double spinfac = 1.;

//#include "models/HeisenbergU1.h"
//typedef VMPS::HeisenbergU1 MODEL; double spinfac = 1.;

//#include "models/SpinlessFermionsU1.h"
//typedef VMPS::SpinlessFermionsU1 MODEL; double spinfac = 1.;

#define USE_HUBBARD
string MODELNAME = "Hubbard";
Q_RANGE QR = MPI_PPI;

#include "models/HubbardSU2xU1.h"
typedef VMPS::HubbardSU2xU1 MODEL; double spinfac = 1.;

//#include "models/HubbardSU2xSU2.h"
//typedef VMPS::HubbardSU2xSU2 MODEL; double spinfac = 1.;

vector<GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double>>> Green;
GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double>> Gfull;

double n_mu (double mu, void*)
{
	return spinfac * Gfull.integrate_Glocw_cell(mu) - double(N)/L;
}

#include "models/SpectralFunctionHelpers.h"
#include "gsl_integration.h"

/////////////////////////////////
int main (int argc, char* argv[])
{
	#ifdef _OPENMP
	omp_set_nested(1);
	#endif
	
	gsl_set_error_handler_off(); lout << "gsl_set_error_handler_off()" << endl;
	
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",2);
	Ncells = args.get<size_t>("Ncells",20);
	
	wd = args.get<string>("wd","./");
	if (wd.back() != '/') {wd += "/";}
	
	Lhetero = L*Ncells;
	x0 = Lhetero/2;
	U = args.get<double>("U",0.);
	J = args.get<double>("J",1.);
	U = args.get<double>("U",6.);
	D = args.get<size_t>("D",3ul);
	N = args.get<int>("N",L);
	specs = args.get_list<string>("specs",{"SSF"});
	Nspec = specs.size();
	Green.resize(Nspec);
	INT = static_cast<GREEN_INTEGRATION>(args.get<int>("INT",1)); //0=DIRECT, 1=INTERP
	
	RELOAD = args.get<string>("RELOAD","");
	wmin = args.get<double>("wmin",-10.);
	#ifdef USE_HEISENBERG
	wmin = 0;
	#endif
	wmax = args.get<double>("wmax",+10.);
	wpoints = args.get<int>("wpoints",501);
	qpoints = args.get<int>("qpoints",501);
	VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",DMRG::VERBOSITY::ON_EXIT));
	
	dt = args.get<double>("dt",0.2);
	tmax = args.get<double>("tmax",6.);
	Nt = static_cast<int>(tmax/dt);
	
	tol_eigval = args.get<double>("tol_eigval",1e-5);
	tol_var = args.get<double>("tol_var",1e-5);
	tol_state = args.get<double>("tol_state",1e-4);
	
	min_iter = args.get<size_t>("min_iter",50ul);
	max_iter = args.get<size_t>("max_iter",150ul);
	
	Chi = args.get<size_t>("Chi",4ul);
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
	GlobParams.max_iter_without_expansion = 30ul;
	
	string base = make_string("Lcell=",L,"_model=",MODELNAME,"_sym=",MODEL::Symmetry::name());
	#ifdef USE_HEISENBERG
	base += make_string("_J=",J);
	#elif defined(USE_HUBBARD)
	base += make_string("_U=",U);
	#endif
	if (RELOAD=="") lout.set(make_string(base,"_Ncells=",Ncells,"_tmax=",tmax,".log"),wd+"log");
	lout << "Nspec=" << Nspec << endl;
	
	lout << args.info() << endl;
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	// reload:
	if (RELOAD != "")
	{
		GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double>> 
		Gnew(wd+RELOAD+"_"+base,L,tmax,
		     {wd+RELOAD+"_"+base+make_string("_L=",Lhetero,"_tmax=",tmax,"_INT=",INT)},
		     QR,qpoints,INT);
		Gnew.recalc_FTw(wmin,wmax,wpoints);
		Gnew.FT_allSites();
		Gnew.save(true); // IGNORE_CELL=true
	}
	else
	{
		#ifdef USE_HEISENBERG
		qarray<MODEL::Symmetry::Nq> Q = MODEL::singlet();
		#elif defined(USE_HUBBARD)
		qarray<MODEL::Symmetry::Nq> Q = MODEL::singlet(N);
		#endif
		
		vector<Param> params;
		params.push_back({"CALC_SQUARE",false});
		params.push_back({"OPEN_BC",false});
		params.push_back({"t",1.});
		#ifdef USE_HEISENBERG
		if (J!=0.)
		{
			params.push_back({"J",J});
			params.push_back({"D",D});
		}
		#endif
		#ifdef USE_HUBBARD
		if (U!=0.)
		{
			params.push_back({"Uph",U});
		}
		#endif
		
		MODEL H(L,params,BC::INFINITE);
		H.transform_base(Q,true); // PRINT=true
		lout << H.info() << endl;
		
		MODEL::uSolver uDMRG(VERB);
		Eigenstate<MODEL::StateUd> g;
		uDMRG.userSetGlobParam();
		uDMRG.GlobParam = GlobParams;
		uDMRG.edgeState(H,g,Q);
		
		MODEL H_hetero(Lhetero,params,BC::INFINITE);
		lout << H_hetero.info() << endl;
		H_hetero.transform_base(Q,false,L); // PRINT=false
		H_hetero.precalc_TwoSiteData(true); // FORCE=true
		
		// create vector of O
		vector<vector<Mpo<MODEL::Symmetry,double>>> O(Nspec);
		for (int z=0; z<Nspec; ++z)
		{
			O[z].resize(L);
			
			for (int l=0; l<L; ++l)
			{
				O[z][l] = VMPS::get_Op<MODEL,MODEL::Symmetry>(H_hetero,Lhetero/2+l,specs[z]);
				O[z][l].transform_base(Q,false,L); // PRINT=false
			}
		}
		
		// OxV in cell
		Stopwatch<> OxVTimer;
		Mps<MODEL::Symmetry,double> Phi = uDMRG.create_Mps(Ncells, g, H, x0); // ground state as heterogenic MPS
		
		vector<vector<Mps<MODEL::Symmetry,complex<double>>>> OxPhiCell(Nspec);
		for (int z=0; z<Nspec; ++z)
		{
			OxPhiCell[z].resize(L);
			auto OxPhiCellReal = uDMRG.create_Mps(Ncells, g, H, O[z][0], O[z]); // O[z][0] for boundaries, O[z] is multiplied
			
			for (int l=0; l<L; ++l)
			{
				OxPhiCell[z][l] = OxPhiCellReal[l].template cast<complex<double>>();
			}
		}
		
		lout << OxVTimer.info("OxV for all sites") << endl;
		
		for (int z=0; z<Nspec; ++z)
		{
			string spec = specs[z];
			Green[z] = GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> >
			           (wd+spec+"_"+base,tmax,Nt,wmin,wmax,wpoints,QR,qpoints,INT);
			Green[z].set_verbosity(DMRG::VERBOSITY::ON_EXIT);
		}
		Green[0].set_verbosity(DMRG::VERBOSITY::STEPWISE);
		
//		// set operator to measure
//		vector<Mpo<MODEL::Symmetry,double>> Measure;
//		Measure.resize(Lhetero);
//		for (int l=0; l<Measure.size(); ++l)
//		{
//			Measure[l] = H_hetero.Sz(l);
//			Measure[l].transform_base(Q,false,L); // PRINT=false
//		}
//		for (int z=0; z<Nspec; ++z) Green[z].set_measurement(Measure,1,"Sz","measure");
		
		double Eg = avg_hetero(Phi, H_hetero, Phi, true); // USE_BOUNDARY=true
		double Eg_ = Lhetero * g.energy;
		lout << setprecision(14) << "Eg=" << Eg << ", " << Eg_ << ", diff=" << abs(Eg-Eg_) << ", eg=" << g.energy << endl;
		
		#pragma omp parallel for
		for (int z=0; z<Nspec; ++z)
		{
			string spec = specs[z];
//			Green[z].set_tol_DeltaS(tol_DeltaS);
			Green[z].set_lim_Nsv(100ul);
			Green[z].compute_cell(H_hetero, OxPhiCell[z], Eg, VMPS::TIME_DIR(spec), true); // COUNTERPROPAGATE=true
			Green[z].FT_allSites();
			Green[z].save(true); // IGNORE_CELL=true
		}
		
		// add PES & IPE
		auto itPES = find(specs.begin(), specs.end(), "PES");
		auto itIPE = find(specs.begin(), specs.end(), "IPE");
		
		if (itPES != specs.end() and itIPE != specs.end())
		{
			lout << "adding PES & IPE..." << endl;
			int iPES = distance(specs.begin(), itPES);
			int iIPE = distance(specs.begin(), itIPE);
			
			// Add PES+IPE
			vector<vector<MatrixXcd>> GinA1P(L); for (int i=0; i<L; ++i) {GinA1P[i].resize(L);}
			for (int i=0; i<L; ++i) 
			for (int j=0; j<L; ++j)
			{
				GinA1P[i][j].resize(Green[iPES].get_GtxCell()[0][0].rows(),
				                    Green[iPES].get_GtxCell()[0][0].cols());
				GinA1P[i][j].setZero();
			}
			
			for (int i=0; i<L; ++i)
			for (int j=0; j<L; ++j)
			{
				GinA1P[i][j] += Green[iPES].get_GtxCell()[i][j] + Green[iIPE].get_GtxCell()[i][j];
			}
			
			Gfull = GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> >
			(wd+"A1P_"+base,tmax,GinA1P,QR,qpoints,INT);
			Gfull.recalc_FTwCell(wmin,wmax,wpoints);
			Gfull.FT_allSites();
			
			IntervalIterator mu(wmin,wmax,101);
			for (mu=mu.begin(); mu!=mu.end(); ++mu)
			{
				double res = spinfac * Gfull.integrate_Glocw(*mu);
				mu << res;
			}
			mu.save(make_string(wd,"n(μ)_tmax=",tmax,"_",base,".dat"));
			RootFinder R(n_mu,wmin,wmax);
			lout << "μ=" << R.root() << endl;
			
			Gfull.mu = R.root();
	//		Gfull.ncell = ncell;
			Gfull.save(true); // IGNORE_CELL=true
		}
	}
}
