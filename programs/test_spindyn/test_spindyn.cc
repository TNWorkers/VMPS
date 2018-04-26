#define DONT_USE_LAPACK_SVD
#define DONT_USE_LAPACK_QR
//#define USE_HDF5_STORAGE
//#define EIGEN_USE_THREADS

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

#include "Stopwatch.h"

#include "solvers/DmrgSolver.h"
#include "solvers/TDVPPropagator.h"
#include "solvers/MpsCompressor.h"

#include "models/KondoU1xU1.h"
#include "models/KondoU1.h"
#include "models/KondoU0xSU2.h"
#include "models/KondoU0xSU2.h"

template<typename Scalar>
string to_string_prec (Scalar x, int n=14)
{
	ostringstream ss;
	ss << setprecision(n) << x;
	return ss.str();
}

bool CALC_DYNAMICS;
int N, T, D;
size_t L, Ly;
double Bxi, Bzi, Bxf, Bzf;
double J, U;
double alpha;
double t_U1, t_SU2;
int Dinit, Dlimit, Imin, Imax, Qinit;
double tol_eigval, tol_state;
double t, dt, tmax, tol_compr, tol_Lanczos;
DMRG::VERBOSITY::OPTION VERB;

int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",10ul);
	Ly = args.get<size_t>("Ly",1ul);
	
	J = args.get<double>("J",-1.);
	U = args.get<double>("U",0.);
	Bxi = args.get<double>("Bxi",1.);
	Bzi = args.get<double>("Bzi",0.);
	Bxf = args.get<double>("Bxf",0.);
	Bzf = args.get<double>("Bzf",1.);
	
	N = args.get<int>("N",L*Ly);
	D = args.get<size_t>("D",2);
	T = args.get<int>("T",1);
	
	alpha = args.get<double>("alpha",100.);
	VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",2));
	
	dt = args.get<double>("dt",0.2);
	tmax = args.get<double>("tmax",6.);
	tol_compr = args.get<double>("tol_compr",1e-4);
	tol_Lanczos = args.get<double>("tol_compr",1e-8);
	
	Dinit  = args.get<int>("Dmin",2);
	Dlimit = args.get<int>("Dmax",100);
	Imin   = args.get<int>("Imin",6);
	Imax   = args.get<int>("Imax",20);
	Qinit  = args.get<int>("Qinit",2);
	tol_eigval = args.get<double>("tol_eigval",1e-6);
	tol_state  = args.get<double>("tol_state",1e-5);
	
	string tinfo;
	
	lout << args.info() << endl;
	lout.set(make_string("L=",L,"_Ly=",Ly,"_D=",D,"_J=",J,".log"),"log");
	
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	vector<Param> params_init;
	params_init.push_back({"Ly",Ly});
	params_init.push_back({"J",J});
	params_init.push_back({"t",1.});
	params_init.push_back({"U",U});
	params_init.push_back({"D",2ul,0});
	params_init.push_back({"CALC_SQUARE",false});
	
	for (size_t l=1; l<L; ++l)
	{
		params_init.push_back({"D",1ul,l});
		params_init.push_back({"Bz",0.,l});
		params_init.push_back({"Bx",0.,l});
	}
	
	vector<Param> params_prop = params_init;
	
	params_init.push_back({"Bx",Bxi,0});
	params_init.push_back({"Bz",Bzi,0});
	
	params_prop.push_back({"Bx",Bxf,0});
	params_prop.push_back({"Bz",Bzf,0});
	
	//--------U(1)---------
	lout << endl << "--------U(1)---------" << endl << endl;
	
	Stopwatch<> Watch_U1;
	
	VMPS::KondoU1 H_U1i(L,params_init);
	VMPS::KondoU1 H_U1f(L,params_prop);
	lout << H_U1i.info() << endl;
	lout << H_U1f.info() << endl;
	assert(H_U1i.validate({N}) and "Bad total quantum number of the MPS.");
	Eigenstate<VMPS::KondoU1::StateXd> g_U1;
	
	VMPS::KondoU1::Solver DMRG_U1(VERB);
	DMRG_U1.edgeState(H_U1i, g_U1, {N}, LANCZOS::EDGE::GROUND, DMRG::CONVTEST::VAR_2SITE, 
	                  tol_eigval,tol_state, Dinit,Dlimit,Qinit, Imax,Imin, alpha);
	g_U1.state.graph("g");
	
	t_U1 = Watch_U1.time();
	
	VMPS::KondoU1::StateXcd Psi_U1 = g_U1.state.cast<complex<double> >();
	Psi_U1.eps_svd = tol_compr;
	TDVPPropagator<VMPS::KondoU1,Sym::U1<Sym::ChargeU1>,double,complex<double>,VMPS::KondoU1::StateXcd> TDVP_U1(H_U1f,Psi_U1);
	
	t = 0;
	ofstream FilerU1(make_string("U1.dat"));
	tinfo = "";
	for (int i=0; i<=static_cast<int>(tmax/dt); ++i)
	{
		double res = isReal(avg(Psi_U1, H_U1f.Simp(SZ,0), Psi_U1));
		FilerU1 << t << "\t" << res << endl;
		lout << "t=" << t << "\t" << res << "\t" << tinfo << endl;
		
		Stopwatch<> Steptimer;
		TDVP_U1.t_step(H_U1f, Psi_U1, -1.i*dt, 1,tol_Lanczos);
		tinfo = Steptimer.info();
		
		if (Psi_U1.get_truncWeight().sum() > 0.5*tol_compr)
		{
			Psi_U1.N_sv = min(static_cast<size_t>(max(Psi_U1.N_sv*1.1,Psi_U1.N_sv+1.)),200ul);
		}
		
		if (VERB != DMRG::VERBOSITY::SILENT) {lout << TDVP_U1.info() << endl << Psi_U1.info() << endl;}
		t += dt;
	}
	FilerU1.close();
	
	//--------SU(2)---------
	lout << endl << "--------SU(2)---------" << endl << endl;
	
	Stopwatch<> Watch_SU2;
	
	VMPS::KondoU0xSU2 H_SU2i(L,params_init);
	VMPS::KondoU0xSU2 H_SU2f(L,params_prop);
	lout << H_SU2i.info() << endl;
	lout << H_SU2f.info() << endl;
	Eigenstate<VMPS::KondoU0xSU2::StateXd> g_SU2;
	
	VMPS::KondoU0xSU2::Solver DMRG_SU2(VERB);
	DMRG_SU2.edgeState(H_SU2i, g_SU2, {T}, LANCZOS::EDGE::GROUND, DMRG::CONVTEST::VAR_2SITE, 
	                   tol_eigval,tol_state, Dinit,Dlimit,Qinit, Imax,Imin, alpha);
	
	t_SU2 = Watch_SU2.time();
	
	VMPS::KondoU0xSU2::StateXcd Psi_SU2 = g_SU2.state.cast<complex<double> >();
	Psi_SU2.eps_svd = tol_compr;
	TDVPPropagator<VMPS::KondoU0xSU2,Sym::SU2<Sym::ChargeSU2>,double,complex<double>,VMPS::KondoU0xSU2::StateXcd> TDVP_SU2(H_SU2f,Psi_SU2);
	
	t = 0;
	ofstream FilerSU2(make_string("SU2.dat"));
	tinfo = "";
	for (int i=0; i<=static_cast<int>(tmax/dt); ++i)
	{
		double res = isReal(avg(Psi_SU2, H_SU2f.Simp(SZ,0), Psi_SU2));
		FilerSU2 << t << "\t" << res << endl;
		lout << "t=" << t << "\t" << res << "\t" << tinfo << endl;
		
		Stopwatch<> Steptimer;
		TDVP_SU2.t_step(H_SU2f, Psi_SU2, -1.i*dt, 1,tol_Lanczos);
		tinfo = Steptimer.info();
		
		if (Psi_SU2.get_truncWeight().sum() > 0.5*tol_compr)
		{
			Psi_SU2.N_sv = min(static_cast<size_t>(max(Psi_SU2.N_sv*1.1,Psi_SU2.N_sv+1.)),200ul);
		}
		
		if (VERB != DMRG::VERBOSITY::SILENT) {lout << TDVP_SU2.info() << endl << Psi_SU2.info() << endl;}
		t += dt;
	}
	FilerSU2.close();
	
}
