#ifdef BLAS
#include "util/LapackManager.h"
#pragma message("LapackManager")
#endif

//#define USE_OLD_COMPRESSION
#define USE_HDF5_STORAGE
#define DMRG_DONT_USE_OPENMP
//#define VUMPS_SOLVER_DONT_USE_OPENMP
#define GREENPROPAGATOR_USE_HDF5
//#define LINEARSOLVER_DIMK 100
//#define TIME_PROP_USE_TERMPLOT

#include <iostream>
#include <fstream>
#include <complex>
#include <iterator>

#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

#include "Logger.h"
Logger lout;
#include "ArgParser.h"

#include "StringStuff.h"
#include "Stopwatch.h"

#include "solvers/DmrgSolver.h"
#include "DmrgLinearAlgebra.h"

#include "models/HubbardSU2xU1.h"
typedef VMPS::HubbardSU2xU1 MODEL;

#ifdef TIME_PROP_USE_TERMPLOT
#include "plot.hpp"
#include "TerminalPlot.h"
#endif

#include "solvers/SpectralManager.h"
#include "models/ParamCollection.h"

int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	lout << args.info() << endl;
	
	#ifdef _OPENMP
	omp_set_nested(1);
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	string Ufile = args.get<string>("Ufile","U");
	ArrayXd U = loadMatrix(Ufile+".dat");
	string tfile = args.get<string>("tfile","t");
	ArrayXXd tFull = loadMatrix(tfile+".dat");
	string Efile = args.get<string>("Efile","E");
	ArrayXd t0 = loadMatrix(Efile+".dat");
	
	int L = U.rows();
	assert(U.cols() == 1);
	assert(t0.cols() == 1);
	assert(tFull.rows() == L and tFull.cols() == L);
	
	int j0 = args.get<int>("j0",L/2);
	
	int N = args.get<int>("N",L); // Teilchenzahl
	double n = double(N)/L;
	qarray<MODEL::Symmetry::Nq> Q = MODEL::singlet(N); // Quantenzahl des Grundzustandes
	lout << "Q=" << Q << endl;
	
	bool SAVE_GS = args.get<bool>("SAVE_GS",false);
	bool LOAD_GS = args.get<bool>("LOAD_GS",false);
	bool RELOAD = args.get<bool>("RELOAD",false);
	
	vector<string> specs = args.get_list<string>("specs",{"PES","IPE"}); // welche Spektren? PES:Photoemission, IPE:inv. Photoemission, HSF: Hybridisierung, IHSF: inverse Hybridisierung
	string specstring = "";
	int Nspec = specs.size();
	size_t Mlim = args.get<size_t>("Mlim",800ul); // Bonddimension fuer Dynamik
	double dt = args.get<double>("dt",0.1);
	double tol_DeltaS = args.get<double>("tol_DeltaS",1e-2);
	double tmax = args.get<double>("tmax",4.);
	double tol_compr = args.get<double>("tol_compr",1e-4);
	
	GREEN_INTEGRATION INT = static_cast<GREEN_INTEGRATION>(args.get<int>("INT",2)); // DIRECT=0, INTERP=1, OOURA=2
	Q_RANGE QR = static_cast<Q_RANGE>(args.get<int>("QR",1)); // MPI_PPI=0, ZERO_2PI=1
	
	double wmin = args.get<double>("wmin",-10.);
	double wmax = args.get<double>("wmax",+10.);
	int wpoints = args.get<int>("wpoints",501);
	int qpoints = args.get<int>("qpoints",501);
	
	// Steuert die Menge der Ausgaben
	DMRG::VERBOSITY::OPTION VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",DMRG::VERBOSITY::HALFSWEEPWISE));
	
	string wd = args.get<string>("wd","./"); correct_foldername(wd); // Arbeitsvereichnis
	string param_base = make_string("Ufile=",Ufile,"_tfile=",tfile,"_Efile=",Efile); // Dateiname
	string base = make_string("L=",L,"_N=",N,"_",param_base); // Dateiname
	lout << base << endl;
	lout.set(base+".log",wd+"log"); // Log-Datei im Unterordner log
	
	// Sweep-Parameter fuer den Grundzustand:
	DMRG::CONTROL::GLOB GlobSweepParams;
	GlobSweepParams.min_halfsweeps = args.get<size_t>("min_halfsweeps",12ul);
	GlobSweepParams.max_halfsweeps = args.get<size_t>("max_halfsweeps",32ul);
	GlobSweepParams.Minit = args.get<size_t>("Minit",1ul);
	GlobSweepParams.Qinit = args.get<size_t>("Qinit",1ul);
	GlobSweepParams.Mlimit = args.get<size_t>("Mlimit",800ul);
	GlobSweepParams.tol_eigval = args.get<double>("tol_eigval",1e-5);
	GlobSweepParams.tol_state = args.get<double>("tol_state",1e-4);
	GlobSweepParams.CALC_S_ON_EXIT = false;
	
	// Parameter
	vector<Param> params;
	params.push_back({"tFull",tFull});
	for (size_t l=0; l<L; ++l)
	{
		params.push_back({"U",U(l),l});
		params.push_back({"t0",t0(l),l});
	}
	
	// Aufbau des Modells
	MODEL H(L,params,BC::OPEN);
	lout << H.info() << endl;
	
	SpectralManager<MODEL> SpecMan;
	if (!RELOAD)
	{
		SpecMan = SpectralManager<MODEL>(specs, H, params, GlobSweepParams, Q, "gs_"+base, LOAD_GS, SAVE_GS, VERB);
		
		if (specs.size() != 0)
		{
			SpecMan.compute_finite(j0, wd, param_base, 1, tmax, dt, wmin, wmax, wpoints, INT, Mlim, tol_DeltaS, tol_compr);
		}
	}
//	else
//	{
//		SpecMan.reload(wd, specs, param_base, L, Ncells, Ns, tmax, wmin, wmax, wpoints, QR, qpoints, INT);
//	}
	
	// A1P berechnen
	auto itPES = find(specs.begin(), specs.end(), "PES");
	auto itIPE = find(specs.begin(), specs.end(), "IPE");
	if (itPES != specs.end() and itIPE != specs.end())
	{
		GreenPropagator<MODEL,MODEL::Symmetry,MODEL::Scalar_,complex<double>> Gfull;
		SpecMan.make_A1P_finite(Gfull, wd, param_base, tmax, wmin, wmax, wpoints, INT);
		Gfull.save(true); // IGNORE_TX=true
	}
}
