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
#include "solvers/GreenPropagator.h"

size_t L, Ncells, Lhetero;
int M, Dtot, N;
double J;
double tmax, dt, tol_compr;
int Nt;
size_t Chi, max_iter, min_iter, Qinit, D;
double tol_eigval, tol_var, tol_state;

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
	
	dt = args.get<double>("dt",0.1);
	tmax = args.get<double>("tmax",10.);
	Nt = static_cast<int>(tmax/dt);
	tol_compr = 1e-4;
	lout << "Nt=" << Nt << endl;
	
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
	
	typedef VMPS::HeisenbergU1 MODEL;
	qarray<MODEL::Symmetry::Nq> Q = MODEL::Symmetry::qvacuum();
	
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
	
	MODEL H_hetero(Lhetero,{{"J",1.},{"D",D},{"OPEN_BC",false}});
	
	GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> > Green(tmax,Nt,0.,3.,1000,1e-4,true);
	
	Mpo<MODEL::Symmetry,double> Oj = H_hetero.Scomp(SP,Ncells);
	vector<Mpo<MODEL::Symmetry,double>> Odagi(Lhetero);
	for (int l=0; l<Lhetero; ++l) Odagi[l] = H_hetero.Scomp(SM,l);
	
	Green.compute(H_hetero,Phi,Oj,Odagi);
}
