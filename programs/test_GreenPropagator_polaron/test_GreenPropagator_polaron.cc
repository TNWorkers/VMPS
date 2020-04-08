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
#include "util/LapackManager.h"

#include "StringStuff.h"
#include "Stopwatch.h"

#include <Eigen/Core>
using namespace Eigen;
#include <unsupported/Eigen/FFT>

#include "solvers/DmrgSolver.h"
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

#define USE_KONDO
string MODELNAME = "Kondo";
Q_RANGE QR = MPI_PPI;

#include "models/KondoSU2xU1.h"
typedef VMPS::KondoSU2xU1 MODEL;

//#include "models/KondoU1xU1.h"
//typedef VMPS::KondoU1xU1 MODEL;

vector<GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double>>> Green;
GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double>> Gfull;

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
	L = args.get<size_t>("L",20);
	
	wd = args.get<string>("wd","./");
	if (wd.back() != '/') {wd += "/";}
	
	x0 = L/2;
	U = args.get<double>("U",0.);
	J = args.get<double>("J",8.);
	N = args.get<int>("N",0);
	specs = args.get_list<string>("specs",{"IPEUP"});
	Nspec = specs.size();
	Green.resize(Nspec);
	INT = static_cast<GREEN_INTEGRATION>(args.get<int>("INT",0)); //0=DIRECT, 1=INTERP
	
	RELOAD = args.get<string>("RELOAD","");
	wmin = args.get<double>("wmin",-15);
	wmax = args.get<double>("wmax",+15);
	wpoints = args.get<int>("wpoints",501);
	qpoints = args.get<int>("qpoints",501);
	VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",DMRG::VERBOSITY::HALFSWEEPWISE));
	
	dt = args.get<double>("dt",0.1);
	tmax = args.get<double>("tmax",6.);
	Nt = static_cast<int>(tmax/dt);
	
	tol_eigval = args.get<double>("tol_eigval",1e-5);
	tol_var = args.get<double>("tol_var",1e-5);
	tol_state = args.get<double>("tol_state",1e-4);
	
	min_iter = args.get<size_t>("min_iter",50ul);
	max_iter = args.get<size_t>("max_iter",150ul);
	
	Chi = args.get<size_t>("Chi",4ul);
	Qinit = args.get<size_t>("Qinit",6ul);
	
//	VUMPS::CONTROL::GLOB GlobParams;
//	GlobParams.min_iterations = min_iter;
//	GlobParams.max_iterations = max_iter;
//	GlobParams.Dinit = Chi;
//	GlobParams.Dlimit = Chi;
//	GlobParams.Qinit = Qinit;
//	GlobParams.tol_eigval = tol_eigval;
//	GlobParams.tol_var = tol_var;
//	GlobParams.tol_state = tol_state;
//	GlobParams.max_iter_without_expansion = 30ul;
	
	string base = make_string("L=",L,"_model=",MODELNAME,"_sym=",MODEL::Symmetry::name(),"_J=",J);
//	if (RELOAD=="") lout.set(make_string(base,"_Ncells=",Ncells,"_tmax=",tmax,".log"),wd+"log");
	lout << "Nspec=" << Nspec << endl;
	
	lout << args.info() << endl;
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	// reload:
//	if (RELOAD != "")
//	{
//		GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double>> 
//		Gnew(wd+RELOAD+"_"+base,L,tmax,
//		     {wd+RELOAD+"_"+base+make_string("_L=",Lhetero,"_tmax=",tmax,"_INT=",INT)},
//		     QR,qpoints,INT);
//		Gnew.recalc_FTw(wmin,wmax,wpoints);
//		Gnew.FT_allSites();
//		Gnew.save(true); // IGNORE_CELL=true
//	}
//	else
	{
		qarray<MODEL::Symmetry::Nq> Q = MODEL::polaron(L,N);
		lout << "Q=" << Q << endl;
		
		vector<Param> params;
		params.push_back({"CALC_SQUARE",false});
		params.push_back({"OPEN_BC",false});
		params.push_back({"t",1.});
		params.push_back({"J",J});
		
		MODEL H(L,params,BC::OPEN);
		lout << H.info() << endl;
		
		MODEL::Solver DMRG(VERB);
		Eigenstate<MODEL::StateXd> g;
//		DMRG.userSetGlobParam();
//		DMRG.GlobParam = GlobParams;
		DMRG.edgeState(H,g,Q);
		
		// create vector of O
		vector<vector<Mpo<MODEL::Symmetry,double>>> O(Nspec);
		for (int z=0; z<Nspec; ++z)
		{
			O[z].resize(L);
			
			for (int l=0; l<L; ++l)
			{
				O[z][l] = VMPS::get_Op<MODEL,MODEL::Symmetry>(H,l,specs[z]);
			}
		}
		
		vector<vector<Mps<MODEL::Symmetry,complex<double>>>> OxPhiCell(Nspec);
		for (int z=0; z<Nspec; ++z)
		{
			OxPhiCell[z].resize(L);
			
			for (int l=0; l<L; ++l)
			{
				Mps<MODEL::Symmetry,double> OxPhiCellReal;
				OxV_exact(O[z][l], g.state, OxPhiCellReal, 2.);
				OxPhiCell[z][l] = OxPhiCellReal.template cast<complex<double>>();
//				OxPhiCell[z][l].graph(make_string("OxPhi_l=",l));
			}
		}
		
		for (int z=0; z<Nspec; ++z)
		{
			string spec = specs[z];
			Green[z] = GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> >
			           (wd+spec+"_"+base,tmax,Nt,wmin,wmax,wpoints,QR,qpoints,INT);
			Green[z].set_verbosity(DMRG::VERBOSITY::ON_EXIT);
		}
		Green[0].set_verbosity(DMRG::VERBOSITY::STEPWISE);
		
		// set operator to measure
		vector<Mpo<MODEL::Symmetry,double>> Measure;
		Measure.resize(L);
		for (int l=0; l<Measure.size(); ++l)
		{
			Measure[l] = H.n(l);
		}
		for (int z=0; z<Nspec; ++z) Green[z].set_measurement(Measure,1,"n","measure");
		
		double Eg = g.energy;
		lout << "Eg=" << Eg << endl;
		
		#pragma omp parallel for
		for (int z=0; z<Nspec; ++z)
		{
			string spec = specs[z];
//			Green[z].set_tol_DeltaS(tol_DeltaS);
			Green[z].set_lim_Nsv(100ul);
			Green[z].set_OxPhiFull(OxPhiCell[z]);
			if constexpr (MODEL::Symmetry::IS_SPIN_SU2()) Green[z].set_Qmulti(2);
			if (N==0) Green[z].set_tol_DeltaS(0.);
			Green[z].compute(H, OxPhiCell[z], OxPhiCell[z][x0], Eg, VMPS::TIME_DIR(spec));
			Green[z].save(true); // IGNORE_CELL=true
		}
	}
}
