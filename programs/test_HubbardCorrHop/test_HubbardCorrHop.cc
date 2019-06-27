#ifdef BLAS
#include "util/LapackManager.h"
#pragma message("LapackManager")
#endif

//#define USE_HDF5_STORAGE

// with Eigen:
#define DMRG_DONT_USE_OPENMP
//#define MPSQCOMPRESSOR_DONT_USE_OPENMP

// with own parallelization:
//#define EIGEN_DONT_PARALLELIZE

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_DEFAULT_INDEX_TYPE int

//Also calculate SO4, implies no tPrime
#define SU2XSU2

#include <iostream>
#include <fstream>
#include <complex>
#include <variant>
#include <fenv.h> 
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

#include "Logger.h"
Logger lout;
#include "ArgParser.h"

// ED stuff
#include "HubbardModel.h"
#include "LanczosWrappers.h"
#include "LanczosSolver.h"
#include "Photo.h"
#include "Auger.h"

#include "solvers/DmrgSolver.h"

#include "models/Hubbard.h"
#include "models/HubbardU1xU1.h"
#include "models/HubbardU1.h"
#include "models/HubbardSU2xU1.h"
#ifdef SU2XSU2
#include "models/HubbardSU2xSU2.h"
#endif

size_t L;
double t, U, V, X;
int N;
int Dinit, Dlimit, Imin, Imax, Qinit;
double alpha, tol_eigval, tol_state;
DMRG::VERBOSITY::OPTION VERB;
bool U1, SU2, SO4;

Eigenstate<VMPS::HubbardU1xU1::StateXd> g_U1;
Eigenstate<VMPS::HubbardSU2xU1::StateXd> g_SU2;
Eigenstate<VMPS::HubbardSU2xSU2::StateXd> g_SO4;


int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",4);
	t = args.get<double>("t",1.);
	U = args.get<double>("U",8.);
	V = args.get<double>("V",0.);
	X = args.get<double>("X",X);
	N = args.get<int>("N",L);
	
	U1 = args.get<bool>("U1",true);
	SO4 = args.get<bool>("SO4",true);
	SU2 = args.get<bool>("U1",true);
	
	DMRG::CONTROL::GLOB GlobParam;
	DMRG::CONTROL::DYN  DynParam;
	size_t min_Nsv = args.get<size_t>("min_Nsv",0ul);
	DynParam.min_Nsv = [min_Nsv] (size_t i) {return min_Nsv;};
	
	alpha = args.get<double>("alpha",100.);
	
	VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",2));
	
	GlobParam.Dinit  = args.get<int>("Dmin",2);
	GlobParam.Dlimit = args.get<int>("Dmax",100);
	GlobParam.Qinit = args.get<int>("Qinit",10);
	GlobParam.min_halfsweeps = args.get<int>("Imin",6);
	GlobParam.max_halfsweeps = args.get<int>("Imax",20);
	GlobParam.tol_eigval = args.get<double>("tol_eigval",1e-6);
	GlobParam.tol_state = args.get<double>("tol_state",1e-5);
	
	lout << args.info() << endl;
	lout.set(make_string("L=",L,"_t=",t,"_U=",U,"_V=",V,"_X=",X,".log"),"log");
	
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	//--------U(1)---------
	if (U1)
	{
		lout << endl << termcolor::red << "--------U(1)---------" << termcolor::reset << endl << endl;
		
		Stopwatch<> Watch_U1;
		
		//,{"tPara",tParaA,0},{"tPara",tParaB,1}
		VMPS::HubbardU1xU1 H_U1(L,{{"t",t},{"Uph",U},{"X",X}});
		lout << H_U1.info() << endl;
		
		VMPS::HubbardU1xU1::Solver DMRG_U1(VERB);
		DMRG_U1.userSetGlobParam();
		DMRG_U1.userSetDynParam();
		DMRG_U1.GlobParam = GlobParam;
		DMRG_U1.DynParam = DynParam;
		DMRG_U1.edgeState(H_U1, g_U1, {0,N}, LANCZOS::EDGE::GROUND);
		
		lout << Watch_U1.info("U(1)") << endl;
		
		cout << avg(g_U1.state, H_U1.ns(L/2), g_U1.state) << endl;
	}
	
//	// --------SU(2)xSU(2)---------
	if (SO4)
	{
		lout << endl << termcolor::red << "--------SU(2)xSU(2)---------" << termcolor::reset << endl << endl;
		
		Stopwatch<> Watch_SO4;
		
		VMPS::HubbardSU2xSU2 H_SO4(L,{{"t",t},{"U",U},{"X",X}});
		lout << H_SO4.info() << endl;
		
		VMPS::HubbardSU2xSU2::Solver DMRG_SO4(VERB);
		DMRG_SO4.userSetGlobParam();
		DMRG_SO4.userSetDynParam();
		DMRG_SO4.GlobParam = GlobParam;
		DMRG_SO4.DynParam = DynParam;
		DMRG_SO4.edgeState(H_SO4, g_SO4, {1,1}, LANCZOS::EDGE::GROUND);
		
		lout << Watch_SO4.info("SO(4)") << endl;
		
		cout << avg(g_SO4.state, H_SO4.ns(L/2), g_SO4.state) << endl;
	}
	
	if (SU2)
	{
		lout << endl << termcolor::red << "--------SU(2)---------" << termcolor::reset << endl << endl;
		
		Stopwatch<> Watch_SU2;
		
		VMPS::HubbardSU2xU1 H_SU2(L,{{"t",t},{"Uph",U},{"X",X}});
		lout << H_SU2.info() << endl;
		
		VMPS::HubbardSU2xU1::Solver DMRG_SU2(VERB);
		DMRG_SU2.userSetGlobParam();
		DMRG_SU2.userSetDynParam();
		DMRG_SU2.GlobParam = GlobParam;
		DMRG_SU2.DynParam = DynParam;
		DMRG_SU2.edgeState(H_SU2, g_SU2, {1,N}, LANCZOS::EDGE::GROUND);
		
		lout << Watch_SU2.info("SU(2)") << endl;
		
		cout << avg(g_SU2.state, H_SU2.ns(L/2), g_SU2.state) << endl;
	}
}
