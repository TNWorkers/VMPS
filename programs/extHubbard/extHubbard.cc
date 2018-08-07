//#define USE_HDF5_STORAGE
//#define EIGEN_USE_THREADS

// with Eigen:
#define DMRG_DONT_USE_OPENMP
//#define MPSQCOMPRESSOR_DONT_USE_OPENMP

// with own parallelization:
//#define EIGEN_DONT_PARALLELIZE

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_DEFAULT_INDEX_TYPE int

//Also calculate SU2xSU2, implies no tPrime
#define SU2XSU2

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
double t_U0, t_U1, t_SU2, t_SU2xSU2;
int Dinit, Dlimit, Imin, Imax, Qinit;
double tol_eigval, tol_state;
int i0;
DMRG::VERBOSITY::OPTION VERB;
double overlap_ED = 0.;
double overlap_U1_zipper = 0.;
double emin_U0 = 0.;
double Emin_U0 = 0.;
double Emin_SU2xSU2 = 0.;
double emin_SU2xSU2 = 0.;
bool ED, U0, U1, SU2, SU22, CORR, PRINT;

Eigenstate<VMPS::HubbardSU2xSU2::StateXd> g_SU2xSU2;

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
	
	DMRG::CONTROL::GLOB GlobParam;
	DMRG::CONTROL::DYN  DynParam;
	size_t min_Nsv = args.get<size_t>("min_Nsv",0ul);
	DynParam.min_Nsv = [min_Nsv] (size_t i) {return min_Nsv;};
	
	alpha = args.get<double>("alpha",100.);
	
	VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",2));
	
	i0 = args.get<int>("i0",L/2);
	
	GlobParam.Dinit  = args.get<int>("Dmin",2);
	GlobParam.Dlimit = args.get<int>("Dmax",100);
	GlobParam.Qinit = args.get<int>("Qinit",10);
	GlobParam.min_halfsweeps = args.get<int>("Imin",6);
	GlobParam.max_halfsweeps = args.get<int>("Imax",20);
	GlobParam.tol_eigval = args.get<double>("tol_eigval",1e-6);
	GlobParam.tol_state = args.get<double>("tol_state",1e-5);
	
	lout << args.info() << endl;
	lout.set(make_string("L=",L,"_Ly=",Ly,"_t=",t,"_t'=",tPrime,"_U=",U,".log"),"log");
	
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
//	// --------SU(2)xSU(2)---------
	lout << endl << termcolor::red << "--------SU(2)xSU(2)---------" << termcolor::reset << endl << endl;
	
	Stopwatch<> Watch_SU2xSU2;
	
	vector<Param> paramsSU2xSU2;
	paramsSU2xSU2.push_back({"t",t,0});
	paramsSU2xSU2.push_back({"t",t,1});
	paramsSU2xSU2.push_back({"U",U,0});
	paramsSU2xSU2.push_back({"U",U,1});
	paramsSU2xSU2.push_back({"V",V,0});
	paramsSU2xSU2.push_back({"V",V,1});
	paramsSU2xSU2.push_back({"J",J,0});
	paramsSU2xSU2.push_back({"J",J,1});
	paramsSU2xSU2.push_back({"Ly",Ly,0});
	paramsSU2xSU2.push_back({"Ly",Ly,1});
	
	VMPS::HubbardSU2xSU2 H_SU2xSU2(L,paramsSU2xSU2);
	lout << H_SU2xSU2.info() << endl;
	
	volume = H_SU2xSU2.volume();
	int T = volume-N+1;
	
	VMPS::HubbardSU2xSU2::Solver DMRG_SU2xSU2(VERB);
	DMRG_SU2xSU2.GlobParam = GlobParam;
	DMRG_SU2xSU2.DynParam = DynParam;
	DMRG_SU2xSU2.edgeState(H_SU2xSU2, g_SU2xSU2, {S,T}, LANCZOS::EDGE::GROUND); 
	
	Emin_SU2xSU2 = g_SU2xSU2.energy-0.5*U*(V-N);
	emin_SU2xSU2 = Emin_SU2xSU2/volume;
	t_SU2xSU2 = Watch_SU2xSU2.time();
	lout << "emin=" << emin_SU2xSU2 << endl;
	
//	for (size_t l=0; l<L; ++l)
//	{
//		double Tcorr = avg(g_SU2xSU2.state, H_SU2xSU2.Tdag(0), H_SU2xSU2.T(l) g_SU2xSU2.state);
//		double Scorr = avg(g_SU2xSU2.state, H_SU2xSU2.Sdag(0), H_SU2xSU2.S(l) g_SU2xSU2.state);
//		cout << "l=" << l << "\t" << Tcorr << "\t" << Scorr << endl;
//	}
}
