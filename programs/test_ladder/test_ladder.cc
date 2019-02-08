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

// ED stuff
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
#include "models/HubbardSU2xSU2.h"
#include "models/KondoU0xSU2.h"

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
double t, tPrime, tRung, U, mu, Bz;
int M, N, S, Nup, Ndn;
double alpha;
double t_U0, t_U0SU2, t_SU2, t_SU2xSU2;
int Dinit, Dlimit, Imin, Imax, Qinit;
double tol_eigval, tol_state;
int i0;
DMRG::VERBOSITY::OPTION VERB;
double overlap_ED = 0.;
double overlap_U0SU2_zipper = 0.;
double emin_U0 = 0.;
double Emin_U0 = 0.;
double Emin_SU2xSU2 = 0.;
double emin_SU2xSU2 = 0.;
bool U0SU2, SU2, SO4, CORR, PRINT;

Eigenstate<VectorXd> g_ED;
Eigenstate<VMPS::Hubbard::StateXd> g_U0;
typedef VMPS::HubbardU1xU1 HUBBARD;
Eigenstate<VMPS::KondoU0xSU2::StateXd> g_U0SU2;
Eigenstate<VMPS::HubbardSU2xU1::StateXd> g_SU2;
Eigenstate<VMPS::HubbardSU2xSU2::StateXd> g_SU2xSU2;

MatrixXd densityMatrix_ED;
VectorXd d_ED, h_ED;

MatrixXd densityMatrix_U0SU2A, densityMatrix_U0SU2B;
VectorXd d_U0SU2;

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
	tRung = args.get<double>("tRung",t);
	U = args.get<double>("U",8.);
	mu = args.get<double>("mu",0.5*U);
	N = args.get<int>("N",L*Ly);
	M = args.get<int>("M",0);;
	S = abs(M)+1;
	// for ED:
	Nup = (N+M)/2;
	Ndn = (N-M)/2;
	
	U0SU2 = args.get<bool>("U0SU2",true);
	SU2 = args.get<bool>("SU2",true);
	SO4 = args.get<bool>("SO4",true);
	CORR = args.get<bool>("CORR",false);
	PRINT = args.get<bool>("PRINT",false);
	if (CORR == false) {PRINT = false;}
	
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
	
	lout << "ref=" << setprecision(16) << VMPS::Hubbard::ref({{"n",1.},{"t",t},{"tRung",tRung},{"Ly",2ul}}) << endl;
	
	if (CORR)
	{
		// resize all to prevent crashes at the end:
		d_ED.resize(L); d_ED.setZero();
		h_ED.resize(L); h_ED.setZero();
		d_U0SU2.resize(L); d_U0SU2.setZero();
		d_SU2.resize(L); d_SU2.setZero();
		nh_SU2xSU2.resize(L); nh_SU2xSU2.setZero();
		ns_SU2xSU2.resize(L); ns_SU2xSU2.setZero();
		
		densityMatrix_ED.resize(L,L); densityMatrix_ED.setZero();
		densityMatrix_U0SU2A.resize(L,L); densityMatrix_U0SU2A.setZero();
		densityMatrix_U0SU2B.resize(L,L); densityMatrix_U0SU2B.setZero();
		densityMatrix_SU2A.resize(L,L); densityMatrix_SU2A.setZero();
		densityMatrix_SU2B.resize(L,L); densityMatrix_SU2B.setZero();
		densityMatrix_SU2xSU2A.resize(L,L); densityMatrix_SU2xSU2A.setZero();
		densityMatrix_SU2xSU2B.resize(L,L); densityMatrix_SU2xSU2B.setZero();
	}
	
	//--------Kondo U(0)xSU(2)---------
	if (U0SU2)
	{
		lout << endl << termcolor::red << "--------Kondo U(0)xSU(2)---------" << termcolor::reset << endl << endl;
		
		Stopwatch<> Watch_U0SU2;
		
		vector<Param> paramsU0xSU2;
		paramsU0xSU2.push_back({"t",tRung,0});
		paramsU0xSU2.push_back({"t",t,1});
		paramsU0xSU2.push_back({"J",0.,0});
		paramsU0xSU2.push_back({"J",0.,1});
		paramsU0xSU2.push_back({"Ly",1ul,0});
		paramsU0xSU2.push_back({"Ly",1ul,1});
		paramsU0xSU2.push_back({"tPrimePrime",t,0});
		paramsU0xSU2.push_back({"tPrimePrime",0.,1});
		VMPS::KondoU0xSU2 H_U0SU2(L,paramsU0xSU2);
		V = H_U0SU2.volume();
		Vsq = V*V;
		lout << H_U0SU2.info() << endl;
		
		VMPS::KondoU0xSU2::Solver DMRG_U0SU2(VERB);
		DMRG_U0SU2.userSetGlobParam();
		DMRG_U0SU2.userSetDynParam();
		DMRG_U0SU2.GlobParam = GlobParam;
		DMRG_U0SU2.DynParam = DynParam;
		DMRG_U0SU2.edgeState(H_U0SU2, g_U0SU2, {V-N+1}, LANCZOS::EDGE::GROUND);
		g_U0SU2.state.graph("U1");
		
		t_U0SU2 = Watch_U0SU2.time();
	}
	
//	// --------SU(2)---------
	if (SU2)
	{
		lout << endl << termcolor::red << "--------SU(2)---------" << termcolor::reset << endl << endl;
		
		Stopwatch<> Watch_SU2;
		
		VMPS::HubbardSU2xU1 H_SU2(L/2,{{"t",t},{"tRung",tRung},{"U",U},{"Ly",2ul}});
		V = H_SU2.volume();
		Vsq = V*V;
		lout << H_SU2.info() << endl;
		
		VMPS::HubbardSU2xU1::Solver DMRG_SU2(VERB);
		DMRG_SU2.userSetGlobParam();
		DMRG_SU2.userSetDynParam();
		DMRG_SU2.GlobParam = GlobParam;
		DMRG_SU2.DynParam = DynParam;
		DMRG_SU2.edgeState(H_SU2, g_SU2, {S,N}, LANCZOS::EDGE::GROUND);
		g_SU2.state.graph("SU2");
		
		t_SU2 = Watch_SU2.time();
		
		if (CORR)
		{
//			Eigenstate<VMPS::HubbardSU2xU1::StateXd> g_SU2m;
//			DMRG_SU2.set_verbosity(DMRG::VERBOSITY::SILENT);
//			DMRG_SU2.edgeState(H_SU2, g_SU2m, {abs(Nup-1-Ndn)+1,N-1}, LANCZOS::EDGE::GROUND, DMRG::CONVTEST::VAR_2SITE,
//				               tol_eigval,tol_state, Dinit,Dlimit,Qinit, Imax,Imin, alpha);
//			lout << "g_SU2m.energy=" << g_SU2m.energy << endl;
//			
//			ArrayXd c_SU2(L);
//			for (int l=0; l<L; ++l)
//			{
//				c_SU2(l) = avg(g_SU2m.state, H_SU2.c(l), g_SU2.state);
//				cout << "l=" << l << ", <c>=" << c_SU2(l) << "\t" << c_SU2(l)/c_U1(l) << endl;
//			}
			
			for (size_t i=0; i<L; ++i) 
			for (size_t j=0; j<L; ++j)
			{
				densityMatrix_SU2A(i,j) = avg(g_SU2.state, H_SU2.cdagc(i,j), g_SU2.state);
			}
			
			for (size_t i=0; i<L; ++i) 
			for (size_t j=0; j<L; ++j)
			{
				densityMatrix_SU2B(i,j) = avg(g_SU2.state, H_SU2.cdag(i), H_SU2.c(j), g_SU2.state);
			}
			
//			lout << "P SU(2): " << Ptot(densityMatrix_SU2,L) << "\t" << Ptot(densityMatrix_SU2B,L) << endl;
			
			for (size_t i=0; i<L; ++i) 
			{
				d_SU2(i) = avg(g_SU2.state, H_SU2.d(i), g_SU2.state);
			}
		}
	}
	
//	// --------SU(2)xSU(2)---------
	if (SO4)
	{
		lout << endl << termcolor::red << "--------SU(2)xSU(2)---------" << termcolor::reset << endl << endl;
		
		Stopwatch<> Watch_SU2xSU2;
		
		vector<Param> paramsSU2xSU2;
		paramsSU2xSU2.push_back({"t",tRung,0});
		paramsSU2xSU2.push_back({"t",t,1});
		paramsSU2xSU2.push_back({"U",U,0});
		paramsSU2xSU2.push_back({"U",U,1});
		paramsSU2xSU2.push_back({"Ly",1ul,0});
		paramsSU2xSU2.push_back({"Ly",1ul,1});
		paramsSU2xSU2.push_back({"tPrimePrime",t,0});
		paramsSU2xSU2.push_back({"tPrimePrime",0.,1});
		VMPS::HubbardSU2xSU2 H_SU2xSU2(L,paramsSU2xSU2);
		V = H_SU2xSU2.volume();
		Vsq = V*V;
		lout << H_SU2xSU2.info() << endl;
		
		VMPS::HubbardSU2xSU2::Solver DMRG_SU2xSU2(VERB);
		DMRG_SU2xSU2.userSetGlobParam();
		DMRG_SU2xSU2.userSetDynParam();
		DMRG_SU2xSU2.GlobParam = GlobParam;
		DMRG_SU2xSU2.DynParam = DynParam;
		DMRG_SU2xSU2.edgeState(H_SU2xSU2, g_SU2xSU2, {S,V-N+1}, LANCZOS::EDGE::GROUND); 
		//Todo: check Pseudospin quantum number... (1 <==> half filling)
		g_SU2xSU2.state.graph("SU2xSU2");
		
		Emin_SU2xSU2 = g_SU2xSU2.energy-0.5*U*(V-N);
		emin_SU2xSU2 = Emin_SU2xSU2/V;
		t_SU2xSU2 = Watch_SU2xSU2.time();
		
		if (CORR)
		{
//			Eigenstate<VMPS::HubbardSU2xSU2::StateXd> g_SU2xSU2m;
//			DMRG_SU2xSU2.set_verbosity(DMRG::VERBOSITY::SILENT);
//			DMRG_SU2xSU2.edgeState(H_SU2xSU2, g_SU2xSU2m, {abs(Nup-1-Ndn)+1,V-(N)+2}, LANCZOS::EDGE::GROUND, DMRG::CONVTEST::VAR_2SITE,
//			                    tol_eigval,tol_state, Dinit,Dlimit, Imax,Imin, alpha);
//			lout << "g_SU2xSU2m.energy=" << g_SU2xSU2m.energy-0.5*U*(V-Nup+1-Ndn) << endl;
//			
//			ArrayXd c_SU2xSU2(L);
//			for (int l=0; l<L; ++l)
//			{
//				c_SU2xSU2(l) = avg(g_SU2xSU2m.state, H_SU2xSU2.c(l), g_SU2xSU2.state);
//				cout << "l=" << l << ", <c>=" << c_SU2xSU2(l) << "\t" << c_SU2xSU2(l)/c_U1(l) << endl;
//			}
			
			for (size_t i=0; i<L; ++i) 
			for (size_t j=0; j<L; ++j)
			{
				densityMatrix_SU2xSU2A(i,j) = 0.5*avg(g_SU2xSU2.state, H_SU2xSU2.cdagc(i,j), g_SU2xSU2.state);
			}
			
			for (size_t i=0; i<L; ++i) 
			for (size_t j=0; j<L; ++j)
			{
				//factor 1/2 because we have computed cdagc+cdagc
				densityMatrix_SU2xSU2B(i,j) = 0.5*avg(g_SU2xSU2.state, H_SU2xSU2.cdag(i), H_SU2xSU2.c(j), g_SU2xSU2.state);
			}
			
		//	lout << "P SU(2): " << Ptot(0.5*densityMatrix_SU2xSU2,L) << "\t" << Ptot(0.5*densityMatrix_SU2xSU2B,L) << endl;
			
			for (size_t i=0; i<L; ++i) 
			{
				nh_SU2xSU2(i) = avg(g_SU2xSU2.state, H_SU2xSU2.nh(i), g_SU2xSU2.state);
				ns_SU2xSU2(i) = avg(g_SU2xSU2.state, H_SU2xSU2.ns(i), g_SU2xSU2.state);
			}
//			lout << "<nh>=" << endl << nh_SU2xSU2 << endl;
//			lout << "error(<nh>=<h>+<d>)=" << (nh_SU2xSU2-d_ED-h_ED).matrix().norm() << endl;
		}
	}
	
	if (PRINT)
	{
		lout << endl << termcolor::blue << "--------Observables---------" << termcolor::reset << endl << endl;
		
		cout << "density matrix ED: " << endl;
		cout << densityMatrix_ED << endl << endl;
		
		cout << "density matrix U(1)⊗U(1) A: " << endl;
		cout << densityMatrix_U0SU2A << endl << endl;
		cout << "density matrix U(1)⊗U(1) B: " << endl;
		cout << densityMatrix_U0SU2B << endl << endl;
		
		cout << "density matrix SU(2)⊗U(1) A: " << endl;
		cout << densityMatrix_SU2A << endl << endl;
		cout << "density matrix SU(2)⊗U(1) B: " << endl;
		cout << densityMatrix_SU2B << endl << endl;
		
		cout << "density matrix SU(2)⊗SU(2) A: " << endl;
		cout << densityMatrix_SU2xSU2A << endl << endl;
		cout << "density matrix SU(2)⊗SU(2) B: " << endl;
		cout << densityMatrix_SU2xSU2B << endl << endl;
	}
	
	//--------output---------
	TextTable T( '-', '|', '+' );
	
	T.add("");
	T.add("ED");
	T.add("U(0)");
	T.add("Kondo U0xSU2");
	T.add("SU(2)⊗U(1) Ly=2");
	T.add("SU(2)⊗SU(2)");
	T.endOfRow();
	
	T.add("E/V");
	T.add(to_string_prec(g_ED.energy/V));
	T.add(to_string_prec(emin_U0));
	T.add(to_string_prec(g_U0SU2.energy/V));
	T.add(to_string_prec(g_SU2.energy/V));
	T.add(to_string_prec(emin_SU2xSU2));
	T.endOfRow();
	
	T.add("E/V diff");
	T.add("-");
	T.add(to_string_prec(abs(Emin_U0-g_SU2.energy)/V,true));
	T.add(to_string_prec(abs(g_U0SU2.energy-g_SU2.energy)/V,true));
	T.add(to_string_prec(abs(g_SU2.energy-g_SU2.energy)/V,true));
	T.add(to_string_prec(abs(Emin_SU2xSU2-g_SU2.energy)/V,true));
	T.endOfRow();
	
	T.add("t/s");
	T.add("-");
	T.add(to_string_prec(t_U0,false,2));
	T.add(to_string_prec(t_U0SU2,false,2));
	T.add(to_string_prec(t_SU2,false,2));
	T.add(to_string_prec(t_SU2xSU2,false,2));
	T.endOfRow();
	
	T.add("t gain");
	T.add("-");
	T.add(to_string_prec(t_U0/t_SU2,false,2));
	T.add(to_string_prec(t_U0SU2/t_SU2,false,2));
	T.add("1");
	T.add(to_string_prec(t_SU2xSU2/t_SU2,false,2));
	T.endOfRow();
	
	if (CORR)
	{
		T.add("d diff");
		T.add("0");
		T.add("-");
		T.add(to_string_prec((d_U0SU2-d_ED).norm(),true));
		T.add(to_string_prec((d_SU2-d_ED).norm(),true));
		T.add(to_string_prec((nh_SU2xSU2-d_ED-h_ED).norm(),true));
		T.endOfRow();
		
		T.add("rhoA diff");
		T.add("0");
		T.add("-");
		T.add(to_string_prec((densityMatrix_U0SU2A-densityMatrix_ED).norm(),true));
		T.add(to_string_prec((densityMatrix_SU2A-densityMatrix_ED).norm(),true));
		T.add(to_string_prec((densityMatrix_SU2xSU2A-densityMatrix_ED).norm(),true));
		T.endOfRow();
		
		T.add("rhoB diff");
		T.add("0");
		T.add("-");
		T.add(to_string_prec((densityMatrix_U0SU2B-densityMatrix_ED).norm(),true));
		T.add(to_string_prec((densityMatrix_SU2B-densityMatrix_ED).norm(),true));
		T.add(to_string_prec((densityMatrix_SU2xSU2B-densityMatrix_ED).norm(),true));
		T.endOfRow();
	}
	
	T.add("Dmax");
	T.add("-");
	T.add(to_string(g_U0.state.calc_Dmax()));
	T.add(to_string(g_U0SU2.state.calc_Dmax()));
	T.add(to_string(g_SU2.state.calc_Dmax()));
	T.add(to_string(g_SU2xSU2.state.calc_Dmax()));
	T.endOfRow();
	
	T.add("Mmax");
	T.add("-");
	T.add(to_string(g_U0.state.calc_Dmax()));
	T.add(to_string(g_U0SU2.state.calc_Mmax()));
	T.add(to_string(g_SU2.state.calc_Mmax()));
	T.add(to_string(g_SU2xSU2.state.calc_Mmax()));
	T.endOfRow();
	
	lout << endl << T;
	
	lout << "ref=" << setprecision(16) << VMPS::Hubbard::ref({{"n",1.},{"t",t},{"tRung",tRung},{"Ly",2ul}}) << endl;
}
