#ifdef BLAS
#include "util/LapackManager.h"
#pragma message("LapackManager")
#endif

//#define USE_OLD_COMPRESSION
#define USE_HDF5_STORAGE
#define DMRG_DONT_USE_OPENMP
//#define DEBUG_VERBOSITY 3

#include <iostream>
#include <fstream>
#include <complex>

#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

#include "Logger.h"
Logger lout;
#include "ArgParser.h"

#include "LanczosWrappers.h"
#include "HxV.h"
#include "LanczosSolver.h"

#include "plot.hpp"
#include "StringStuff.h"
#include "Stopwatch.h"
#include "TextTable.h"
#define HELPERS_IO_TABLE

#include "solvers/DmrgSolver.h"
#include "models/HubbardSU2xU1.h"
typedef VMPS::HubbardSU2xU1 MODEL;
#include "DmrgLinearAlgebra.h"
#include "VUMPS/VumpsSolver.h"

////////////////////////////////
int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	
	size_t L = args.get<size_t>("L",6);
	size_t N = args.get<size_t>("N",L);
	qarray<MODEL::Symmetry::Nq> Q = MODEL::singlet(N);
	double U = args.get<double>("U",8.);
	DMRG::VERBOSITY::OPTION VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",DMRG::VERBOSITY::ON_EXIT));
	
	DMRG::CONTROL::GLOB GlobParams;
	GlobParams.Minit = args.get<size_t>("Minit",1ul);
	GlobParams.Mlimit = args.get<size_t>("Mlimit",500ul);
	GlobParams.Qinit = args.get<size_t>("Qinit",1ul);
	GlobParams.min_halfsweeps = args.get<size_t>("min_halfsweeps",20ul);
	GlobParams.tol_eigval = args.get<double>("tol_eigval",1e-5);
	GlobParams.tol_state = args.get<double>("tol_state",1e-4);
	GlobParams.CALC_S_ON_EXIT = false;
	
	Eigenstate<MODEL::StateXd> g;
	
	MODEL H(L,{{"U",8.}},BC::OPEN);
	lout << H.info() << endl;
	
	MODEL::Solver DMRG1(VERB);
	DMRG1.userSetGlobParam();
	DMRG1.GlobParam = GlobParams;
	
	DMRG1.edgeState(H, g, Q);
	
	Mpo<MODEL::Symmetry> Hmpo = H;
	Mpo<MODEL::Symmetry> O = sum(H.n(L/2),H.n(L/2+1));
	Mpo<MODEL::Symmetry> OxH = prod(O,H);
	Mpo<MODEL::Symmetry> HxO = prod(H,O);
	OxH.scale(-1.);
	Mpo<MODEL::Symmetry> Commutator = sum(HxO,OxH);
	MODEL::StateXd Tmp;
	OxV_exact(Commutator, g.state, Tmp, 2., DMRG::VERBOSITY::SILENT);
	// verschiedene Ergebnisse:
	cout << "A variant1=" << -avg(g.state, Commutator, Commutator, g.state) << ", variant2=" << dot(Tmp,Tmp) << endl;
	
	O = H.n(L/2);
	OxH = prod(O,H);
	HxO = prod(H,O);
	OxH.scale(-1.);
	Commutator = sum(HxO,OxH);
	OxV_exact(Commutator, g.state, Tmp, 2., DMRG::VERBOSITY::SILENT);
	// gleiches Ergebnis:
	cout << "B variant1=" << -avg(g.state, Commutator, Commutator, g.state) << ", variant2=" << dot(Tmp,Tmp) << endl;
	
	O = prod(H.n(L/2),H.n(L/2+1));
	OxH = prod(O,H);
	HxO = prod(H,O);
	OxH.scale(-1.);
	Commutator = sum(HxO,OxH);
	OxV_exact(Commutator, g.state, Tmp, 2., DMRG::VERBOSITY::SILENT);
	// verschiedene Ergebnisse:
	cout << "C variant1=" << -avg(g.state, Commutator, Commutator, g.state) << ", variant2=" << dot(Tmp,Tmp) << endl;
}
