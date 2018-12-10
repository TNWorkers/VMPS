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

//Also calculate SU2xSU2, implies no tPrime
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

//// ED stuff
//#include "HubbardModel.h"
//#include "LanczosWrappers.h"
//#include "LanczosSolver.h"
//#include "Photo.h"
//#include "Auger.h"

#include "solvers/DmrgSolver.h"

#include "models/Hubbard.h"
#include "models/HubbardU1xU1.h"
#include "models/HubbardU1.h"
#include "models/HubbardSU2xU1.h"
#ifdef SU2XSU2
#include "models/HubbardSU2xSU2.h"
#endif

#include "models/KitaevChain.h"
#include "models/Heisenberg.h"

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

size_t L, Ly, Ly2;
int V, Vsq;
double t, tPrime, tRung, Jxy, Jxyprime, U, mu, Delta, Bz;
int M, N, S, Nup, Ndn;
double alpha;
double t_U0, t_Z2, t_U1, t_SU2, t_SU2xSU2;
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
bool ED, U0, U1, SU2, SU22, Z_2, CORR, PRINT;

Eigenstate<VectorXd> g_ED;
Eigenstate<VMPS::Heisenberg::StateXd> g_U0;
Eigenstate<VMPS::HubbardU1xU1::StateXd> g_U1;
Eigenstate<VMPS::KitaevChain::StateXd> g_Z2;
Eigenstate<VMPS::HubbardSU2xU1::StateXd> g_SU2;
Eigenstate<VMPS::HubbardSU2xSU2::StateXd> g_SU2xSU2;

MatrixXd densityMatrix_ED;
VectorXd d_ED, h_ED;

MatrixXd densityMatrix_U1A, densityMatrix_U1B;
VectorXd d_U1;

MatrixXd densityMatrix_SU2A, densityMatrix_SU2B;
VectorXd d_SU2;

MatrixXd densityMatrix_SU2xSU2A, densityMatrix_SU2xSU2B;

VectorXd nh_SU2xSU2, ns_SU2xSU2;

int main (int argc, char* argv[])
{
//	feenableexcept(FE_INVALID | FE_OVERFLOW);
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",4);
	Ly = args.get<size_t>("Ly",1);
	Ly2 = args.get<size_t>("Ly2",Ly);
	t = args.get<double>("t",1.);
	tPrime = args.get<double>("tPrime",0.);
	tRung = args.get<double>("tRung",0.);
	U = args.get<double>("U",8.);
	mu = args.get<double>("mu",1.);
	Delta = args.get<double>("Delta",1.5);
	Nup = args.get<int>("Nup",L*Ly);
	Ndn = args.get<int>("Ndn",0);
	M = args.get<int>("M",0);;
	S = abs(M)+1;
	// for ED:
	N = Nup+Ndn;
	size_t D = args.get<int>("D",2);
	
//	ArrayXXd tParaA(1,2); tParaA = t;
//	ArrayXXd tParaB(2,1); tParaB = t;
	
	U0 = args.get<bool>("U0",false);
	Z_2 = args.get<bool>("Z_2",true);
	CORR = args.get<bool>("CORR",false);
//	PRINT = args.get<bool>("PRINT",false);
//	if (CORR == false) {PRINT = false;}
	
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
	GlobParam.min_halfsweeps = args.get<int>("Imin",1);
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
	
	if (CORR)
	{
		// resize all to prevent crashes at the end:
		d_ED.resize(L); d_ED.setZero();
		h_ED.resize(L); h_ED.setZero();
		d_U1.resize(L); d_U1.setZero();
		d_SU2.resize(L); d_SU2.setZero();
		nh_SU2xSU2.resize(L); nh_SU2xSU2.setZero();
		ns_SU2xSU2.resize(L); ns_SU2xSU2.setZero();
		
		densityMatrix_ED.resize(L,L); densityMatrix_ED.setZero();
		densityMatrix_U1A.resize(L,L); densityMatrix_U1A.setZero();
		densityMatrix_U1B.resize(L,L); densityMatrix_U1B.setZero();
		densityMatrix_SU2A.resize(L,L); densityMatrix_SU2A.setZero();
		densityMatrix_SU2B.resize(L,L); densityMatrix_SU2B.setZero();
		densityMatrix_SU2xSU2A.resize(L,L); densityMatrix_SU2xSU2A.setZero();
		densityMatrix_SU2xSU2B.resize(L,L); densityMatrix_SU2xSU2B.setZero();
	}
	
	//--------U(0)---------
	if (U0)
	{
		lout << endl << termcolor::red << "--------U(0)---------" << termcolor::reset << endl << endl;
		
		Stopwatch<> Watch_U0;
		VMPS::Heisenberg H_U0(L,{{"J",0.},{"t",t},{"Delta",Delta},{"mu",mu}});
		lout << H_U0.info() << endl;
		V = H_U0.volume();
		Vsq = V*V;
		
		VMPS::Heisenberg::Solver DMRG_U0(VERB);
		DMRG_U0.GlobParam = GlobParam;
		DMRG_U0.DynParam = DynParam;
		DMRG_U0.userSetGlobParam();
		DMRG_U0.userSetDynParam();
		DMRG_U0.edgeState(H_U0, g_U0, {}, LANCZOS::EDGE::GROUND);
		
		t_U0 = Watch_U0.time();
		
		lout << endl;
		double Ntot = 0.;
		for (size_t lx=0; lx<L; ++lx)
		for (size_t ly=0; ly<Ly; ++ly)
		{
			double n_l = avg(g_U0.state, H_U0.n(lx,ly), g_U0.state);
			cout << "lx=" << lx << ", ly=" << ly << "\tn=" << n_l << endl;
			Ntot += n_l;
		}
		cout << "Ntot=" << Ntot << endl;
	}
	
	//--------Z(2)---------
	if (Z_2)
	{
		lout << endl << termcolor::red << "--------Z(2)---------" << termcolor::reset << endl << endl;
		
		Stopwatch<> Watch_Z2;
		
		VMPS::KitaevChain H_Z2(L,{{"J",0.},{"t",t},{"Delta",Delta},{"mu",mu},{"CALC_SQUARE",false}});
		cout << H_Z2.info() << endl;
		V = H_Z2.volume();
		Vsq = V*V;
		
		VMPS::KitaevChain::Solver DMRG_Z2(VERB);
		DMRG_Z2.GlobParam = GlobParam;
		DMRG_Z2.DynParam = DynParam;
		DMRG_Z2.userSetGlobParam();
		DMRG_Z2.userSetDynParam();
		DMRG_Z2.edgeState(H_Z2, g_Z2, {0}, LANCZOS::EDGE::GROUND);
		g_Z2.state.graph("Z2");
		
		t_Z2 = Watch_Z2.time();
		
		lout << endl;
		double Ntot = 0.;
		for (size_t lx=0; lx<L; ++lx)
		for (size_t ly=0; ly<Ly; ++ly)
		{
			double n_l = avg(g_Z2.state, H_Z2.n(lx,ly), g_Z2.state);
			cout << "lx=" << lx << ", ly=" << ly << "\tn=" << n_l << endl;
			Ntot += n_l;
		}
		cout << "Ntot=" << Ntot << endl;
		
//		if (CORR)
//		{
////			Eigenstate<VMPS::HubbardU1xU1::StateXd> g_U1m;
////			DMRG_U1.set_verbosity(DMRG::VERBOSITY::SILENT);
////			DMRG_U1.edgeState(H_U1, g_U1m, {Nup-1,Ndn}, LANCZOS::EDGE::GROUND, DMRG::CONVTEST::VAR_2SITE, 
////			                tol_eigval,tol_state, Dinit,Dlimit,Qinit, Imax,Imin, alpha);
////			lout << "g_U1m.energy=" << g_U1m.energy << endl;
////		
////			ArrayXd c_U1(L);
////			for (int l=0; l<L; ++l)
////			{
////				c_U1(l) = avg(g_U1m.state, H_U1.c(UP,l), g_U1.state);
////				cout << "l=" << l << ", <c>=" << c_U1(l) << endl;
////			}
//			
//			for (size_t i=0; i<L; ++i) 
//			for (size_t j=0; j<L; ++j)
//			{
//				densityMatrix_U1A(i,j) = avg(g_U1.state, H_U1.cdagc<UP>(i,j), g_U1.state)+
//				                         avg(g_U1.state, H_U1.cdagc<DN>(i,j), g_U1.state);
//			}
//			
//			for (size_t i=0; i<L; ++i) 
//			for (size_t j=0; j<L; ++j)
//			{
//				densityMatrix_U1B(i,j) = avg(g_U1.state, H_U1.cdag<UP>(i), H_U1.c<UP>(j), g_U1.state)+
//				                         avg(g_U1.state, H_U1.cdag<DN>(i), H_U1.c<DN>(j), g_U1.state);
//			}
//			
////			lout << "P U(1): " << Ptot(densityMatrix_U1,L) << "\t" << Ptot(densityMatrix_U1B,L) << endl;
//			
//			for (size_t i=0; i<L; ++i) 
//			{
//				d_U1(i) = avg(g_U1.state, H_U1.d(i), g_U1.state);
//			}
//		}
	}
	
//	if (PRINT)
//	{
//		lout << endl << termcolor::blue << "--------Observables---------" << termcolor::reset << endl << endl;
//		
//		cout << "density matrix ED: " << endl;
//		cout << densityMatrix_ED << endl << endl;
//		
//		cout << "density matrix U(1)⊗U(1) A: " << endl;
//		cout << densityMatrix_U1A << endl << endl;
//		cout << "density matrix U(1)⊗U(1) B: " << endl;
//		cout << densityMatrix_U1B << endl << endl;
//		
//		cout << "density matrix SU(2)⊗U(1) A: " << endl;
//		cout << densityMatrix_SU2A << endl << endl;
//		cout << "density matrix SU(2)⊗U(1) B: " << endl;
//		cout << densityMatrix_SU2B << endl << endl;
//		
//		cout << "density matrix SU(2)⊗SU(2) A: " << endl;
//		cout << densityMatrix_SU2xSU2A << endl << endl;
//		cout << "density matrix SU(2)⊗SU(2) B: " << endl;
//		cout << densityMatrix_SU2xSU2B << endl << endl;
//	}
	
	//--------output---------
	TextTable T( '-', '|', '+' );
	
	T.add("");
	T.add("U(0)");
	T.add("Z(2)");
	T.endOfRow();
	
	T.add("E/V");
	T.add(to_string_prec(g_U0.energy/V));
	T.add(to_string_prec(g_Z2.energy/V));
	T.endOfRow();
	
	T.add("E/V diff");
	T.add("-");
	T.add(to_string_prec(abs(g_Z2.energy-g_U0.energy)/V,true));
	T.endOfRow();
	
	T.add("t/s");
	T.add(to_string_prec(t_U0,false,2));
	T.add(to_string_prec(t_Z2,false,2));
	T.endOfRow();
	
	T.add("t gain");
	T.add("-");
	T.add(to_string_prec(t_Z2/t_U0,false,2));
	T.endOfRow();
	
//	if (CORR)
//	{
//		T.add("d diff");
//		T.add("0");
//		T.add("-");
//		T.add(to_string_prec((d_U1-d_ED).norm(),true));
//		T.add(to_string_prec((d_SU2-d_ED).norm(),true));
//		T.add(to_string_prec((nh_SU2xSU2-d_ED-h_ED).norm(),true));
//		T.endOfRow();
//		
//		T.add("rhoA diff");
//		T.add("0");
//		T.add("-");
//		T.add(to_string_prec((densityMatrix_U1A-densityMatrix_ED).norm(),true));
//		T.add(to_string_prec((densityMatrix_SU2A-densityMatrix_ED).norm(),true));
//		T.add(to_string_prec((densityMatrix_SU2xSU2A-densityMatrix_ED).norm(),true));
//		T.endOfRow();
//		
//		T.add("rhoB diff");
//		T.add("0");
//		T.add("-");
//		T.add(to_string_prec((densityMatrix_U1B-densityMatrix_ED).norm(),true));
//		T.add(to_string_prec((densityMatrix_SU2B-densityMatrix_ED).norm(),true));
//		T.add(to_string_prec((densityMatrix_SU2xSU2B-densityMatrix_ED).norm(),true));
//		T.endOfRow();
//	}
	
	T.add("Dmax");
	T.add(to_string(g_U0.state.calc_Dmax()));
	T.add(to_string(g_Z2.state.calc_Dmax()));
	T.endOfRow();
	
	T.add("Mmax");
	T.add(to_string(g_U0.state.calc_Dmax()));
	T.add(to_string(g_Z2.state.calc_Mmax()));
	T.endOfRow();
	
	lout << endl << T;
	
	lout << "ref=" << VMPS::Hubbard::ref({{"n",static_cast<double>((N)/V)},{"U",U},{"t",t},{"Ly",Ly},{"tRung",tRung},{"tPrime",tPrime},{"Delta",Delta}}) << endl;
}
