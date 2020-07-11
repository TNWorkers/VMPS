#ifdef BLAS
#include "util/LapackManager.h"
#pragma message("LapackManager")
#endif

#define DEBUG_VERBOSITY 0

#define USE_HDF5_STORAGE

// with Eigen:
#define DMRG_DONT_USE_OPENMP
//#define MPSQCOMPRESSOR_DONT_USE_OPENMP

// with own parallelization:
//#define EIGEN_DONT_PARALLELIZE

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_DEFAULT_INDEX_TYPE int

#include <iostream>
#include <fstream>
#include <complex>
#include <variant>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

#include "Logger.h"
Logger lout;
#include "ArgParser.h"

#include "tensors/Qbasis.h"
#include "TwoSiteGate.h"
#include "bases/SpinBase.h"

#include "solvers/DmrgSolver.h"
#include "models/HeisenbergSU2.h"
#include "models/HeisenbergU1.h"
#include "models/Heisenberg.h"

size_t L1,L2,D1,D2,L,D;
int M,Dtot;
double J,Jprime;
DMRG::VERBOSITY::OPTION VERB;
Eigenstate<VMPS::HeisenbergSU2::StateXd> g_SU2;
Eigenstate<VMPS::HeisenbergU1::StateXd> g_U1;
Eigenstate<VMPS::Heisenberg::StateXd> g_U0;

int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);

	L1 = args.get<size_t>("L1",2);
	L2 = args.get<size_t>("L2",2);

	D1 = args.get<size_t>("D1",2);
	D2 = args.get<size_t>("D2",2);

	VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",1));

	L = args.get<size_t>("L",10);
	J = args.get<double>("J",1.);
	Jprime = args.get<double>("Jprime",0.);

	M = args.get<int>("M",0);
	Dtot = abs(M)+1;
	D = args.get<size_t>("D",2);

	// typedef Sym::SU2<Sym::SpinSU2> Symmetry;
	// typedef Sym::U1<Sym::SpinU1> Symmetry;
	typedef Sym::U0 Symmetry;
	
	SpinBase<Symmetry> B1(L1,D1);
	SpinBase<Symmetry> B2(L2,D2);

	TwoSiteGate<Symmetry,double> Swap(B1.get_basis(),B2.get_basis());
	Swap.setSwapGate();
	Swap.print();

	VMPS::HeisenbergSU2 H_SU2(L,{{"J",J},{"Jprime",Jprime},{"D",D}});
	VMPS::HeisenbergU1 H_U1(L,{{"J",J},{"Jprime",Jprime},{"D",D}});
	VMPS::Heisenberg H_U0(L,{{"J",J},{"Jprime",Jprime},{"D",D}});
	cout << H_U0.info() << endl;
	VMPS::HeisenbergSU2::Solver DMRG_SU2(VERB);
	VMPS::HeisenbergU1::Solver DMRG_U1(VERB);
	VMPS::Heisenberg::Solver DMRG_U0(VERB);
	
	// DMRG_SU2.edgeState(H_SU2, g_SU2, {Dtot}, LANCZOS::EDGE::GROUND);
	// DMRG_U1.edgeState(H_U1, g_U1, {M}, LANCZOS::EDGE::GROUND);
	DMRG_U0.edgeState(H_U0, g_U0, {}, LANCZOS::EDGE::GROUND);
		
	cout << g_U0.state.info() << endl;
	cout << g_U0.state << endl;
	cout << "E=" << avg(g_U0.state,H_U0,g_U0.state) << endl;
	g_U0.state.applyGate(Swap,0,DMRG::DIRECTION::RIGHT);
	cout << g_U0.state << endl;
	cout << "dot=" << g_U0.state.dot(g_U0.state) << endl;
	g_U0.state.applyGate(Swap,0,DMRG::DIRECTION::RIGHT);
	cout << "Applied Swap Gate twice:" << endl;
	cout << g_U0.state.info() << endl;
	cout << "E=" << avg(g_U0.state,H_U0,g_U0.state) << endl;
}

