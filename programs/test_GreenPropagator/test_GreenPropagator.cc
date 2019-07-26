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
#include "models/HubbardU1.h"

#include "IntervalIterator.h"
#include "Quadrator.h"
#define CHEBTRANS_DONT_USE_FFTWOMP
#include "SuperQuadrator.h"
#include "solvers/GreenPropagator.h"

size_t L, Ncells, Lhetero;
int M, Dtot, N;
double J, V, U;
double tmax, dt, tol_compr;
int Nt;
size_t Chi, max_iter, min_iter, Qinit, D;
double tol_eigval, tol_var, tol_state;
bool GAUSSINT;

// set model here:
#define FERMIONS
//#define HEISENBERG

#ifdef HEISENBERG
typedef VMPS::HeisenbergU1 MODEL;
#elif defined(FERMIONS)
//typedef VMPS::SpinlessFermionsU1 MODEL; // spinless
typedef VMPS::HubbardU1xU1 MODEL; // spinful
#endif

int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",2);
	Ncells = args.get<size_t>("Ncells",30);
	Lhetero = L*Ncells+1;
	J = args.get<double>("J",1.);
	V = args.get<double>("V",4.);
	U = args.get<double>("U",6.);
	M = args.get<int>("M",0);
	Dtot = abs(M)+1;
	N = args.get<int>("N",L/2);
	D = args.get<size_t>("D",3ul);
	GAUSSINT = static_cast<bool>(args.get<int>("GAUSSINT",true));
	
	dt = args.get<double>("dt",0.1);
	tmax = args.get<double>("tmax",10.);
	Nt = static_cast<int>(tmax/dt);
	tol_compr = 1e-4;
	lout << "Nt=" << Nt << endl;
	
	Chi = args.get<size_t>("Chi",4);
	tol_eigval = args.get<double>("tol_eigval",1e-5);
	tol_var = args.get<double>("tol_var",1e-5);
	tol_state = args.get<double>("tol_state",1e-2);
	
	max_iter = args.get<size_t>("max_iter",100ul);
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
	
	#ifdef HEISENBERG
	qarray<MODEL::Symmetry::Nq> Q = MODEL::Symmetry::qvacuum();
	#elif defined(FERMIONS)
//	qarray<MODEL::Symmetry::Nq> Q = {N};   // spinless
	qarray<MODEL::Symmetry::Nq> Q = {0,N}; // spinful
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
//	g.state.graph("Umps");
	
	#if defined(FERMIONS)
	for (int l=0; l<L; ++l)
	{
		lout << "l=" << l << " n=" << avg(g.state, H.n(l), g.state) << endl;
//		<< ", d=" << avg(g.state, H.d(l), g.state) << endl;
	}
	#endif
	
	Mps<MODEL::Symmetry,double> Phi = uDMRG.create_Mps(Ncells, g, H, H.JordanWignerString(), true); // ADD_ODD_SITE=true
	lout << Phi.info() << endl;
//	Phi.graph("hetero");
	
	#ifdef HEISENBERG
	MODEL H_hetero(Lhetero,{{"J",1.},{"D",D},{"OPEN_BC",false},{"CALC_SQUARE",false}});
	#elif defined(FERMIONS)
//	MODEL H_hetero(Lhetero,{{"V",V},{"OPEN_BC",false},{"CALC_SQUARE",false}}); // spinless
	MODEL H_hetero(Lhetero,{{"U",U},{"OPEN_BC",false},{"CALC_SQUARE",false}}); // spinful
	#endif
	lout << H_hetero.info() << endl;
	H_hetero.transform_base(Q,false,L); // PRINT=false
	H_hetero.precalc_TwoSiteData(true); // FORCE=true
	
	GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> > Green;
//	
//	auto Op1 = H_hetero.cdagc<UP>(Ncells-3,Ncells); Op1.transform_base(Q,false,L);
//	cout << "cdagc=" << avg_hetero(Phi, Op1, Phi) << endl;
//	
//	Mps<MODEL::Symmetry,double> aPhi;
//	auto Op2 = H_hetero.a<UP>(Ncells); Op2.transform_base(Q,false,L);
//	OxV_exact(Op2, Phi, aPhi, 2., DMRG::VERBOSITY::SILENT);
//	
//	Mps<MODEL::Symmetry,double> adagPhi;
//	auto Op3 = H_hetero.a<UP>(Ncells-3); Op3.transform_base(Q,false,L);
//	OxV_exact(Op3, Phi, adagPhi, 2., DMRG::VERBOSITY::SILENT);
//	
//	auto Op4 = H_hetero.JWstring(Ncells-3,Ncells); Op4.transform_base(Q,false,L);
//	cout << "adagJWa=" << avg_hetero(adagPhi, Op4, aPhi) << endl;
//	cout << "dot=" << dot(adagPhi, aPhi) << endl;
	
	#ifdef HEISENBERG
	Green = GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> >(tmax,Nt,0.,3.,1000,1e-4,GAUSSINT,ZERO_2PI);
	#elif defined(FERMIONS)
	Green = GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> >(tmax,Nt,-4.,2.,1000,1e-4,GAUSSINT,MPI_PPI);
	#endif
	
	// Oj
	#ifdef HEISENBERG
	Mpo<MODEL::Symmetry,double> Oj = H_hetero.Scomp(SP,Ncells);
	#elif defined(FERMIONS)
	Mpo<MODEL::Symmetry,double> Oj = H_hetero.c<UP>(Lhetero/2);
	#endif
	Oj.transform_base(Q,false,L); // PRINT=false
	
	// O†i
	vector<Mpo<MODEL::Symmetry,double>> Odagi(Lhetero);
	for (int l=0; l<Lhetero; ++l)
	{
		#ifdef HEISENBERG
		Odagi[l] = H_hetero.Scomp(SM,l);
		#elif defined(FERMIONS)
		Odagi[l] = H_hetero.cdag<UP>(l);
		#endif
		
		Odagi[l].transform_base(Q,false,L); // PRINT=false
	}
	
	// set operator to measure
	vector<Mpo<MODEL::Symmetry,double>> Measure(Lhetero);
	#ifdef HEISENBERG
	for (int l=0; l<Lhetero; ++l)
	{
		Measure[l] = H_hetero.Sz(l);
		Measure[l].transform_base(Q,false,L); // PRINT=false
	}
	#elif defined(FERMIONS)
	for (int l=0; l<Lhetero; ++l)
	{
		Measure[l] = H_hetero.n(l);
		Measure[l].transform_base(Q,false,L); // PRINT=false
//		lout << "l=" << l << ", avg=" << avg_hetero(Phi, nvec[l], Phi) << endl;
	}
	#endif
	Green.set_measurement(Measure,10);
	
	Green.compute(H_hetero, Phi, Oj, Odagi);
	
//	// reload:
//	GreenPropagator<MODEL,MODEL::Symmetry,double,complex<double> > Green(tmax,Nt,loadMatrix("GtxRe_tmax=10_Nt=100.dat"),loadMatrix("GtxIm_tmax=10_Nt=100.dat"));
//	Green.recalc_FTw(-18.,0.,10000);
//	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
}
