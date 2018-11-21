#define USE_HDF5_STORAGE
#include "util/LapackManager.h"


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

#include "solvers/DmrgSolver.h"
#include "VUMPS/VumpsSolver.h"
#include "models/HubbardSU2xSU2.h"

template<typename Scalar>
string to_string_prec (Scalar x, bool COLOR=false, int n=14)
{
	ostringstream ss;
	if (x < 1e-5 and COLOR)
	{
		ss << termcolor::colorize << termcolor::green << setprecision(n) << x << termcolor::reset;
	}
	else if (x >= 1e-5 and COLOR)
	{
		ss << termcolor::colorize << termcolor::red << setprecision(n) << x << termcolor::reset;
	}
	else
	{
		ss << setprecision(n) << x;
	}
	return ss.str();
}

complex<double> Ptot (const MatrixXd &densityMatrix, int L)
{
	complex<double> P=0.;
	int L_2 = static_cast<int>(L)/2;
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	for (int n=-L_2; n<L_2; ++n)
	{
		double k = 2.*M_PI*n/L;
		P += k * exp(-1.i*k*static_cast<double>(i-j)) * densityMatrix(i,j);
	}
	P /= (L*L);
	return P;
}

size_t L, Ly, Ly2;
int volume;
double t, tPrime, tRung, U, mu, Bz, J, V;
int M, N, S, Nup, Ndn;
double alpha;
double t_tot;
int Dinit, Dlimit, Imin, Imax, Qinit;
double tol_eigval, tol_state;
int i0;
DMRG::VERBOSITY::OPTION VERB;
double Emin = 0.;
double emin = 0.;

Eigenstate<VMPS::HubbardSU2xSU2::StateXd> g_fix;
Eigenstate<VMPS::HubbardSU2xSU2::StateUd> g_foxy;

int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",4);
	Ly = args.get<size_t>("Ly",1);
	Ly2 = args.get<size_t>("Ly2",Ly);
	t = args.get<double>("t",1.);
	tPrime = args.get<double>("tPrime",0.);
	tRung = args.get<double>("tRung",0.);
	U = args.get<double>("U",8.);
	J = args.get<double>("J",0.);
	V = args.get<double>("V",0.);
	mu = args.get<double>("mu",0.5*U);
	N = args.get<int>("N",L*Ly);
	M = args.get<int>("M",0);;
	S = abs(M)+1;

	bool VUMPS = args.get<bool>("VUMPS",true);
	
	DMRG::CONTROL::GLOB GlobParam_fix;
	DMRG::CONTROL::DYN  DynParam_fix;
	VUMPS::CONTROL::GLOB GlobParam_foxy;
	VUMPS::CONTROL::DYN  DynParam_foxy;

	size_t min_Nsv = args.get<size_t>("min_Nsv",0ul);
	DynParam_fix.min_Nsv = [min_Nsv] (size_t i) {return min_Nsv;};
	
	alpha = args.get<double>("alpha",100.);
	
	VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",2));
	
	i0 = args.get<int>("i0",L/2);

	
	GlobParam_fix.Dinit  = args.get<size_t>("Dinit",4ul);
	GlobParam_fix.Dlimit = args.get<size_t>("Dlimit",200ul);
	GlobParam_fix.Qinit = args.get<size_t>("Qinit",10ul);
	GlobParam_fix.min_halfsweeps = args.get<size_t>("Imin",6);
	GlobParam_fix.max_halfsweeps = args.get<size_t>("Imax",40);
	GlobParam_fix.tol_eigval = args.get<double>("tol_eigval",1e-6);
	GlobParam_fix.tol_state = args.get<double>("tol_state",1e-5);

	GlobParam_foxy.Dinit  = args.get<size_t>("Dinit",4ul);
	GlobParam_foxy.Dlimit = args.get<size_t>("Dlimit",200ul);
	GlobParam_foxy.Qinit = args.get<size_t>("Qinit",10ul);
	GlobParam_foxy.min_iterations = args.get<size_t>("Imin",6);
	GlobParam_foxy.max_iterations = args.get<size_t>("Imax",1000);
	GlobParam_foxy.tol_eigval = args.get<double>("tol_eigval",1e-6);
	GlobParam_foxy.tol_var = args.get<double>("tol_var",1e-6);
	GlobParam_foxy.tol_state = args.get<double>("tol_state",1e-5);

	lout << args.info() << endl;
	lout.set(make_string("L=",L,"_Ly=",Ly,"_t=",t,"_t'=",tPrime,"_U=",U,".log"),"log");
	
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
		
	Stopwatch<> Watch;
	
	vector<Param> params;
	params.push_back({"t",t,0});
	params.push_back({"t",t,1});
	params.push_back({"U",U,0});
	params.push_back({"U",U,1});
	params.push_back({"V",V,0});
	params.push_back({"V",V,1});
	params.push_back({"J",J,0});
	params.push_back({"J",J,1});
	params.push_back({"Ly",Ly,0});
	params.push_back({"Ly",Ly,1});
	if (VUMPS) {cout << "set open bc to false." << endl; params.push_back({"OPEN_BC",false});}
	
	VMPS::HubbardSU2xSU2 H(L,params);
	lout << H.info() << endl;

	volume = H.volume();
	int T = volume-N+1;

	if (VUMPS)
	{
		VMPS::HubbardSU2xSU2::uSolver Foxy(VERB);

		Foxy.userSetGlobParam();
		Foxy.userSetDynParam();
		Foxy.GlobParam = GlobParam_foxy;
		Foxy.DynParam = DynParam_foxy;
		Foxy.edgeState(H, g_foxy, {0,0});

		emin = g_foxy.energy;
	}
	else
	{
		VMPS::HubbardSU2xSU2::Solver Fix(VERB);
		Fix.userSetGlobParam();
		Fix.userSetDynParam();
		Fix.GlobParam = GlobParam_fix;
		Fix.DynParam = DynParam_fix;
		Fix.edgeState(H, g_fix, {S,T}, LANCZOS::EDGE::GROUND); 

		Emin = g_fix.energy-0.5*U*(V-N);
		emin = Emin/volume;
	}
	
	t_tot = Watch.time();
	lout << "emin=" << emin << endl;

	if (VUMPS) {}
	else
	{
		for (size_t l=0; l<L; ++l)
		{
			double Tcorr = avg(g_fix.state, H.Tdag(0), H.T(l), g_fix.state);
			double Scorr = avg(g_fix.state, H.Sdag(0), H.S(l), g_fix.state);
			cout << "l=" << l << "\t" << Tcorr << "\t" << Scorr << endl;
		}
	}
}
