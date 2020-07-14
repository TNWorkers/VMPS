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

size_t L1,L2,D1,D2,L,D,Ly,swapSite;
int M,Dtot;
double J,Jprime;
DMRG::VERBOSITY::OPTION VERB;
Eigenstate<VMPS::HeisenbergSU2::StateXd> g_SU2;
Eigenstate<VMPS::HeisenbergU1::StateXd> g_U1;
Eigenstate<VMPS::Heisenberg::StateXd> g_U0;

int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);

	swapSite = args.get<size_t>("swapSite",0);
		
	Ly = args.get<size_t>("Ly",1);
	L1 = args.get<size_t>("L1",Ly);
	L2 = args.get<size_t>("L2",Ly);

	D1 = args.get<size_t>("D1",2);
	D2 = args.get<size_t>("D2",2);

	VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",1));

	DMRG::DIRECTION::OPTION DIR = static_cast<DMRG::DIRECTION::OPTION>(args.get<int>("DIR",0));
		
	L = args.get<size_t>("L",10);
	J = args.get<double>("J",1.);
	Jprime = args.get<double>("Jprime",0.);

	M = args.get<int>("M",0);
	Dtot = abs(M)+1;
	D = args.get<size_t>("D",2);

	bool SU2 = args.get<bool>("SU2",1);
	bool U1 = args.get<bool>("U1",1);
	bool U0 = args.get<bool>("U0",1);
	
	if (SU2)
	{
		cout << "=======================SU2=========================" << endl;
		VMPS::HeisenbergSU2 H_SU2(L,{{"J",J},{"Jprime",Jprime},{"D",D},{"Ly",Ly}});
		VMPS::HeisenbergSU2::Solver DMRG_SU2(VERB);
		DMRG_SU2.edgeState(H_SU2, g_SU2, {Dtot}, LANCZOS::EDGE::GROUND);
		Qbasis<VMPS::HeisenbergSU2::Symmetry> qloc_l; qloc_l.pullData(H_SU2.locBasis(swapSite),true);
		Qbasis<VMPS::HeisenbergSU2::Symmetry> qloc_lp1; qloc_lp1.pullData(H_SU2.locBasis(swapSite+1),true);
		auto locBasis = qloc_l.combine(qloc_lp1);
		TwoSiteGate<VMPS::HeisenbergSU2::Symmetry,double> Swap_SU2(qloc_l,qloc_lp1);
		Swap_SU2.setSwapGate();
		// Swap_SU2.setIdentity();
		g_SU2.state.sweep(swapSite,DMRG::BROOM::QR);
		cout << g_SU2.state.info() << endl << g_SU2.state.test_ortho() << endl;
		// cout << g_SU2.state << endl;
		cout << "E=" << avg(g_SU2.state,H_SU2,g_SU2.state) << endl;
		auto copy_state = g_SU2.state;
		// vector<Biped<VMPS::HeisenbergSU2::Symmetry,Matrix<double,Dynamic,Dynamic> > > Apair;
		// contract_AA2(g_SU2.state.A_at(swapSite), g_SU2.state.locBasis(swapSite), g_SU2.state.A_at(swapSite+1), g_SU2.state.locBasis(swapSite+1), Apair);
		// split_AA2(DIR, locBasis, Apair,
		// 		  g_SU2.state.locBasis(swapSite), g_SU2.state.A_at(swapSite),
		// 		  g_SU2.state.locBasis(swapSite+1), g_SU2.state.A_at(swapSite+1),
		// 		  g_SU2.state.QoutTop[swapSite], g_SU2.state.QoutBot[swapSite], g_SU2.state.eps_svd, g_SU2.state.min_Nsv, g_SU2.state.max_Nsv);
		
		g_SU2.state.applyGate(Swap_SU2,swapSite,DIR);
		cout << "Applied Swap Gate once:" << endl;
		cout << "dot=" << copy_state.dot(g_SU2.state) << endl;
		g_SU2.state.applyGate(Swap_SU2,swapSite,DIR);
		// cout << g_SU2.state << endl;
		cout << "Applied Swap Gate twice:" << endl;
		cout << "dot=" << copy_state.dot(g_SU2.state) << endl;
		// g_SU2.state.applyGate(Swap_SU2,swapSite,DIR);
		cout << g_SU2.state.info() << endl << g_SU2.state.test_ortho() << endl << endl;
		cout << "E=" << avg(g_SU2.state,H_SU2,g_SU2.state) << endl;
		cout << endl << endl << endl;
	}

	if (U1)
	{
		cout << "========================U1=========================" << endl;
		VMPS::HeisenbergU1::Solver DMRG_U1(VERB);
		VMPS::HeisenbergU1 H_U1(L,{{"J",J},{"Jprime",Jprime},{"D",D},{"Ly",Ly}});
		DMRG_U1.edgeState(H_U1, g_U1, {M}, LANCZOS::EDGE::GROUND);
		Qbasis<VMPS::HeisenbergU1::Symmetry> qloc_l; qloc_l.pullData(H_U1.locBasis(swapSite),true);
		Qbasis<VMPS::HeisenbergU1::Symmetry> qloc_lp1; qloc_lp1.pullData(H_U1.locBasis(swapSite+1),true);
		TwoSiteGate<VMPS::HeisenbergU1::Symmetry,double> Swap_U1(qloc_l,qloc_lp1);
		// Swap_U1.setSwapGate();
		Swap_U1.setIdentity();
		g_U1.state.sweep(swapSite,DMRG::BROOM::QR);
		cout << g_U1.state.info() << endl << g_U1.state.test_ortho() << endl << endl;
		// cout << g_U1.state << endl;
		cout << "E=" << avg(g_U1.state,H_U1,g_U1.state) << endl;
		auto copy_state = g_U1.state;
		g_U1.state.applyGate(Swap_U1,swapSite,DIR);
		// cout << g_U1.state << endl;
		// g_U1.state.applyGate(Swap_U1,swapSite,DIR);
		cout << g_U1.state.info() << endl << g_U1.state.test_ortho() << endl << endl;
		cout << "Applied Swap Gate once:" << endl;
		cout << "dot=" << copy_state.dot(g_U1.state) << endl;
		cout << "E=" << avg(g_U1.state,H_U1,g_U1.state) << endl;
		cout << endl << endl << endl;
	}
	if (U0)
	{
		cout << "========================U0=========================" << endl;
		VMPS::Heisenberg::Solver DMRG_U0(VERB);	
		VMPS::Heisenberg H_U0(L,{{"J",J},{"Jprime",Jprime},{"D",D},{"Ly",Ly}});
		DMRG_U0.edgeState(H_U0, g_U0, {}, LANCZOS::EDGE::GROUND);
		g_U0.state.sweep(swapSite,DMRG::BROOM::QR);
		cout << g_U0.state.info() << endl << g_U0.state.test_ortho() << endl << endl;
		Qbasis<VMPS::Heisenberg::Symmetry> qloc_l; qloc_l.pullData(H_U0.locBasis(swapSite),true);
		Qbasis<VMPS::Heisenberg::Symmetry> qloc_lp1; qloc_lp1.pullData(H_U0.locBasis(swapSite+1),true);
		TwoSiteGate<VMPS::Heisenberg::Symmetry,double> Swap_U0(qloc_l,qloc_lp1);
		// Swap_U0.setSwapGate();
		Swap_U0.setIdentity();
		cout << "E=" << avg(g_U0.state,H_U0,g_U0.state) << endl;
		g_U0.state.applyGate(Swap_U0,swapSite,DIR);
		cout << "Applied Swap Gate once:" << endl;
		cout << "dot=" << g_U0.state.dot(g_U0.state) << endl;
		// g_U0.state.applyGate(Swap_U0,swapSite,DIR);
		cout << g_U0.state.info() << endl << g_U0.state.test_ortho() << endl << endl;
		cout << g_U0.state.info() << endl;
		cout << "E=" << avg(g_U0.state,H_U0,g_U0.state) << endl;
	}
}
