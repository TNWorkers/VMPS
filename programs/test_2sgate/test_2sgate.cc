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
#include "models/ParamCollection.h"

#include "solvers/DmrgSolver.h"
#include "models/HeisenbergSU2.h"
#include "models/HeisenbergU1.h"
#include "models/Heisenberg.h"

size_t L,D,Ly,swapSite;
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

	VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",1));

	DMRG::DIRECTION::OPTION DIR = static_cast<DMRG::DIRECTION::OPTION>(args.get<int>("DIR",0));
		
	L = args.get<size_t>("L",10);
	J = args.get<double>("J",1.);
	double Janiso = args.get<double>("Jansio",J);
	Jprime = args.get<double>("Jprime",0.);

	M = args.get<int>("M",0);
	Dtot = abs(M)+1;
	D = args.get<size_t>("D",2);

	bool SU2 = args.get<bool>("SU2",1);
	bool U1 = args.get<bool>("U1",1);
	bool U0 = args.get<bool>("U0",1);

	Eigen::ArrayXXd Jcoupl = create_1D_PBC(L,J);
	Jcoupl(0,1) = Janiso;
	Jcoupl(1,0) = Janiso;

	SpinBase<VMPS::HeisenbergSU2::Symmetry> B(L,D);
	cout << "J" << endl << Jcoupl << endl;
	auto H = B.HeisenbergHamiltonian((0.5*Jcoupl).eval());
	EDSolver<SiteOperatorQ<VMPS::HeisenbergSU2::Symmetry, MatrixXd> > Jim(H);
	cout << Jim.eigenvalues().data() << endl;
	auto rotate = [](auto &s, const auto &g) -> void
		 {
			 for (int l=L-2; l>-1; l--)
			 {
				 s.applyGate(g,static_cast<size_t>(l),DMRG::DIRECTION::LEFT);
			 }
		 };
	
	if (SU2)
	{
		cout << "=======================SU2=========================" << endl;
		VMPS::HeisenbergSU2 H_SU2(L,{{"Jfull",Jcoupl},{"D",D}});
		cout << H_SU2.info() << endl;
		VMPS::HeisenbergSU2::Solver DMRG_SU2(VERB);
		DMRG_SU2.edgeState(H_SU2, g_SU2, {Dtot}, LANCZOS::EDGE::GROUND);
		Qbasis<VMPS::HeisenbergSU2::Symmetry> qloc_l; qloc_l.pullData(H_SU2.locBasis(swapSite),true);
		Qbasis<VMPS::HeisenbergSU2::Symmetry> qloc_lp1; qloc_lp1.pullData(H_SU2.locBasis(swapSite+1),true);
		TwoSiteGate<VMPS::HeisenbergSU2::Symmetry,double> Swap_SU2(qloc_l,qloc_lp1);
		Swap_SU2.setSwapGate();
		// Swap_SU2.setIdentity();
		// g_SU2.state.sweep(swapSite,DMRG::BROOM::QR);
		cout << g_SU2.state.info() << endl << g_SU2.state.test_ortho() << endl;
		cout << "E=" << avg(g_SU2.state,H_SU2,g_SU2.state) << endl;
		auto copy_state = g_SU2.state;
		for (size_t l=1; l<=L; l++)
		{
			rotate(copy_state,Swap_SU2);
			cout << "Rotated by " << l*360./L << " degree:" << endl;
			cout << "dot=" << copy_state.dot(g_SU2.state) << endl;
		}
		// g_SU2.state.applyGate(Swap_SU2,swapSite,DIR);
		cout << g_SU2.state.info() << endl << g_SU2.state.test_ortho() << endl << endl;
		cout << "E=" << avg(copy_state,H_SU2,copy_state) << endl;
		cout << endl << endl << endl;
	}

	if (U1)
	{
		cout << "========================U1=========================" << endl;
		VMPS::HeisenbergU1::Solver DMRG_U1(VERB);
		VMPS::HeisenbergU1 H_U1(L,{{"Jfull",Jcoupl},{"D",D}});
		DMRG_U1.edgeState(H_U1, g_U1, {M}, LANCZOS::EDGE::GROUND);
		Qbasis<VMPS::HeisenbergU1::Symmetry> qloc_l; qloc_l.pullData(H_U1.locBasis(swapSite),true);
		Qbasis<VMPS::HeisenbergU1::Symmetry> qloc_lp1; qloc_lp1.pullData(H_U1.locBasis(swapSite+1),true);
		TwoSiteGate<VMPS::HeisenbergU1::Symmetry,double> Swap_U1(qloc_l,qloc_lp1);
		Swap_U1.setSwapGate();
		cout << g_U1.state.info() << endl << g_U1.state.test_ortho() << endl << endl;
		cout << "E=" << avg(g_U1.state,H_U1,g_U1.state) << endl;
		auto copy_state = g_U1.state;
		for (size_t l=1; l<=L; l++)
		{
			rotate(copy_state,Swap_U1);
			cout << "Rotated by " << l*360./L << " degree:" << endl;
			cout << "dot=" << copy_state.dot(g_U1.state) << endl;
		}
		cout << "E=" << avg(copy_state,H_U1,copy_state) << endl;
		cout << endl << endl << endl;
	}
	if (U0)
	{
		cout << "========================U0=========================" << endl;
		VMPS::Heisenberg::Solver DMRG_U0(VERB);	
		VMPS::Heisenberg H_U0(L,{{"Jfull",Jcoupl},{"D",D}});
		DMRG_U0.edgeState(H_U0, g_U0, {}, LANCZOS::EDGE::GROUND);
		g_U0.state.sweep(swapSite,DMRG::BROOM::QR);
		cout << g_U0.state.info() << endl << g_U0.state.test_ortho() << endl << endl;
		Qbasis<VMPS::Heisenberg::Symmetry> qloc_l; qloc_l.pullData(H_U0.locBasis(swapSite),true);
		Qbasis<VMPS::Heisenberg::Symmetry> qloc_lp1; qloc_lp1.pullData(H_U0.locBasis(swapSite+1),true);
		TwoSiteGate<VMPS::Heisenberg::Symmetry,double> Swap_U0(qloc_l,qloc_lp1);
		Swap_U0.setSwapGate();
		cout << "E=" << avg(g_U0.state,H_U0,g_U0.state) << endl;
		auto copy_state = g_U0.state;
		for (size_t l=1; l<=L; l++)
		{
			rotate(copy_state,Swap_U0);
			cout << "Rotated by " << l*360./L << " degree:" << endl;
			cout << "dot=" << copy_state.dot(g_U0.state) << endl;
		}
		cout << "E=" << avg(copy_state,H_U0,copy_state) << endl;
	}
}
