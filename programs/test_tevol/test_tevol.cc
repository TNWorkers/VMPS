#ifdef BLAS
#include "util/LapackManager.h"
#pragma message("LapackManager")
#endif
//#define USE_HDF5_STORAGE

// with Eigen:
#define DMRG_DONT_USE_OPENMP
#define MPSQCOMPRESSOR_DONT_USE_OPENMP

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

#include "solvers/DmrgSolver.h"
#include "solvers/TDVPPropagator.h"
//#include "solvers/MpsCompressor.h"

#include "models/HeisenbergSU2.h"
//include "models/HeisenbergU1XXZ.h"
//include "models/HeisenbergXYZ.h"

bool CALC_DYNAMICS;
int M, S;
size_t D;
size_t L, Ly;
double J, Jprime;
double alpha;
double t_U0, t_U1, t_SU2;
int Dinit, Dlimit, Imin, Imax;
double tol_eigval, tol_state;
double dt, tmax, tol_compr, tol_Lanczos;
size_t i0;
DMRG::VERBOSITY::OPTION VERB;

int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",10);
	Ly = args.get<size_t>("Ly",1);
	J = args.get<double>("J",1.);
	Jprime = args.get<double>("Jprime",0.);
	M = args.get<int>("M",0);
	D = args.get<size_t>("D",2);
	S = abs(M)+1;
	alpha = args.get<double>("alpha",1.);
	VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",2));
	i0 = args.get<size_t>("i0",0);
	
	dt = args.get<double>("dt",0.1);
	tmax = args.get<double>("tmax",6.);
	tol_compr = args.get<double>("tol_compr",1e-5);
	tol_Lanczos = args.get<double>("tol_compr",1e-8);
	
	Dinit  = args.get<int>("Dmin",2);
	Dlimit = args.get<int>("Dmax",100);
	Imin   = args.get<int>("Imin",10);
	Imax   = args.get<int>("Imax",50);
	tol_eigval = args.get<double>("tol_eigval",1e-6);
	tol_state  = args.get<double>("tol_state",1e-5);
	
	CALC_DYNAMICS = args.get<bool>("CALC_DYN",0);
	
	lout << args.info() << endl;
	lout.set(make_string("L=",L,"_Ly=",Ly,"_M=",M,"_D=",D,"_J=",J,".log"),"log");
	
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	//--------U(1)---------
	/*lout << termcolor::red << endl << "--------U(1)---------" << termcolor::reset << endl << endl;
	
	Stopwatch<> Watch_U1;
	VMPS::HeisenbergU1 H_U1(L,{{"J",J},{"Jprime",Jprime},{"D",D},{"Ly",Ly}});
	H_U1.precalc_TwoSiteData();
	lout << H_U1.info() << endl;
	Eigenstate<VMPS::HeisenbergU1::StateXd> g_U1;
	
	VMPS::HeisenbergU1::Solver DMRG_U1(VERB);
	DMRG_U1.edgeState(H_U1, g_U1, {M}, LANCZOS::EDGE::GROUND);
	
	t_U1 = Watch_U1.time();
	
	VMPS::HeisenbergU1::StateXd Psi_U1tmp;
	Psi_U1tmp.eps_svd = 1e-15;
//	OxV(H_U1.Sz(0), g_U1.state, Psi_U1tmp, DMRG::BROOM::SVD);
//	VMPS::HeisenbergU1::CompressorXd CompadreU1(VERB);
//	CompadreU1.prodCompress(H_U1.Sz(i0), H_U1.Sz(i0), g_U1.state, Psi_U1tmp, {M}, g_U1.state.calc_Dmax());
	OxV_exact(H_U1.Sz(i0), g_U1.state, Psi_U1tmp, 1e-7);
	
	Psi_U1tmp.max_Nsv = Psi_U1tmp.calc_Dmax();
	Psi_U1tmp.eps_svd = tol_compr;
	
	VMPS::HeisenbergU1::StateXcd init_U1 = Psi_U1tmp.cast<complex<double> >();
	VMPS::HeisenbergU1::StateXcd Psi_U1 = init_U1;
	TDVPPropagator<VMPS::HeisenbergU1,Sym::U1<Sym::SpinU1>,double,complex<double>,VMPS::HeisenbergU1::StateXcd> TDVP_U1(H_U1,Psi_U1);
	
	double t = 0;
	ofstream FilerU1(make_string("U(1).dat"));
	ArrayXd resU1(static_cast<int>(tmax/dt)+1);
	string tinfo="";
	for (int i=0; i<=static_cast<int>(tmax/dt); ++i)
	{
		double res = isReal(dot(init_U1,Psi_U1));
		FilerU1 << t << "\t" << res << endl;
		lout << "t=" << t << "\t" << res << "\t" << tinfo << endl;
		resU1(i) = res;
		
		Stopwatch<> Steptimer;
		TDVP_U1.t_step0(H_U1, Psi_U1, -1.i*dt, 1,tol_Lanczos);
		tinfo = Steptimer.info();
		
		if (Psi_U1.get_truncWeight().sum() > 0.5*tol_compr)
		{
			Psi_U1.max_Nsv = min(static_cast<size_t>(max(Psi_U1.max_Nsv*1.1,Psi_U1.max_Nsv+1.)),200ul);
		}
		
		if (VERB != DMRG::VERBOSITY::SILENT) {lout << TDVP_U1.info() << endl << Psi_U1.info() << endl;}
		t += dt;
	}
	FilerU1.close();*/
	
	// --------SU(2)---------
    
    double t = 0;
    std::string tinfo = "";
	lout << termcolor::red << endl << "--------SU(2)---------" << termcolor::reset << endl << endl;
	
	Stopwatch<> Watch_SU2;
	VMPS::HeisenbergSU2 H_SU2(L,{{"J",J},{"Jprime",Jprime},{"D",D},{"Ly",Ly}});
	H_SU2.precalc_TwoSiteData();
	lout << H_SU2.info() << endl;
	Eigenstate<VMPS::HeisenbergSU2::StateXd> g_SU2;
	
	VMPS::HeisenbergSU2::Solver DMRG_SU2(VERB);
	DMRG_SU2.edgeState(H_SU2, g_SU2, {S}, LANCZOS::EDGE::GROUND);
	
	t_SU2 = Watch_SU2.time();
	
	VMPS::HeisenbergSU2::StateXd Psi_SU2tmp;
	VMPS::HeisenbergSU2::CompressorXd Compadre(VERB);
	
	cout << "avg=" << avg(g_SU2.state, H_SU2.Sdag(i0), H_SU2.S(i0), g_SU2.state, {1}) << endl;
	cout << "avg=" << avg(g_SU2.state, H_SU2.SdagS(i0,i0), g_SU2.state) << endl;
	
//	Compadre.prodCompress(H_SU2.S(i0), H_SU2.Sdag(i0), g_SU2.state, Psi_SU2tmp, {3}, g_SU2.state.calc_Dmax());
	OxV_exact(H_SU2.S(i0), g_SU2.state, Psi_SU2tmp, 1e-7);
//	Compadre.stateCompress(Psi_SU2tmp_, Psi_SU2tmp, g_SU2.state.calc_Dmax()/2, 1e-10);
	
	Psi_SU2tmp.max_Nsv = Psi_SU2tmp.calc_Dmax();
	Psi_SU2tmp.eps_svd = tol_compr;
	
	VMPS::HeisenbergSU2::StateXcd init_SU2 = Psi_SU2tmp.cast<complex<double> >();
	VMPS::HeisenbergSU2::StateXcd Psi_SU2 = init_SU2;
	TDVPPropagator<VMPS::HeisenbergSU2,Sym::SU2<Sym::SpinSU2>,double,complex<double>,VMPS::HeisenbergSU2::StateXcd> TDVP_SU2(H_SU2,Psi_SU2);
	
	// --------propagation---------
	t = 0;
	ofstream FilerSU2(make_string("SU(2).dat"));
	ArrayXd resSU2(static_cast<int>(tmax/dt)+1);
	tinfo="";
	for (int i=0; i<=static_cast<int>(tmax/dt); ++i)
	{
		double res = isReal(dot(init_SU2,Psi_SU2))/3.;
		FilerSU2 << t << "\t" << res << endl;
		lout << "t=" << t << "\t" << res << "\t" << tinfo << endl;
		resSU2(i) = res;
		
		Stopwatch<> Steptimer;
		TDVP_SU2.t_step0(H_SU2, Psi_SU2, -1.i*dt, 1,tol_Lanczos);
		tinfo = Steptimer.info();
		
		if (Psi_SU2.get_truncWeight().sum() > 0.5*tol_compr)
		{
			Psi_SU2.max_Nsv = min(static_cast<size_t>(max(Psi_SU2.max_Nsv*1.1,Psi_SU2.max_Nsv+1.)),200ul);
		}
		
		if (VERB != DMRG::VERBOSITY::SILENT) {lout << TDVP_SU2.info() << endl << Psi_SU2.info() << endl;}
		t += dt;
	}
	FilerSU2.close();
	
	//cout << abs(resU1-resSU2) << endl;
}
