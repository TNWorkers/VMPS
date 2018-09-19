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
#include "solvers/MpsCompressor.h"

#include "models/HeisenbergSU2.h"
#include "models/HeisenbergU1XXZ.h"
#include "models/HeisenbergXYZ.h"

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

VMPS::HeisenbergU1XXZ::StateXcd Neel (const VMPS::HeisenbergU1XXZ &H)
{
	vector<qarray<1> > Neel_config(H.length());
	for (int l=0; l<H.length(); l+=2)
	{
		Neel_config[l]   = qarray<1>{+1};
		Neel_config[l+1] = qarray<1>{-1};
	}
	
	VMPS::HeisenbergU1XXZ::StateXcd Psi; 
	Psi.setProductState(H,Neel_config);
	
	return Psi;
}

bool CALC_DYNAMICS;
int M, S;
size_t D, D1;
size_t L, Ly, Ldyn;
double J, Jprime, Jrung;
double alpha;
double t_U0, t_U1, t_SU2;
size_t Dinit, Dlimit, Qinit, Imin, Imax;
int max_Nrich;
double tol_eigval, tol_state, eps_svd;
double dt, tmax;
DMRG::VERBOSITY::OPTION VERB;
bool U0, U1, SU2;

double E_U0_compressor=0., E_U0_zipper=0.;
MatrixXd SpinCorr_U0;
Eigenstate<VMPS::Heisenberg::StateXd>    g_U0;
Eigenstate<VMPS::HeisenbergU1::StateXd>  g_U1;
Eigenstate<VMPS::HeisenbergSU2::StateXd> g_SU2;

double E_U1_compressor=0;
double E_U1_zipper=0;
MatrixXd SpinCorr_U1;

MatrixXd SpinCorr_SU2;

int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",10);
	Ly = args.get<size_t>("Ly",1);
	Ldyn = args.get<size_t>("Ldyn",12);
	J = args.get<double>("J",1.);
	Jrung = args.get<double>("Jrung",J);
	Jprime = args.get<double>("Jprime",0.);
	M = args.get<int>("M",0);
	D = args.get<size_t>("D",2);
	D1 = args.get<size_t>("D1",D);
	S = abs(M)+1;
	size_t min_Nsv = args.get<size_t>("min_Nsv",0ul);
	VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",2));
	
	U0 = args.get<bool>("U0",false);
	U1 = args.get<bool>("U1",true);
	SU2 = args.get<bool>("SU2",true);
	
	eps_svd = args.get<double>("eps_svd",1e-7);
	alpha = args.get<double>("alpha",1e2);
	
	Dinit  = args.get<size_t>("Dinit",2ul);
	Dlimit = args.get<size_t>("Dmax",100ul);
	Qinit  = args.get<size_t>("Qinit",10ul);
	Imin   = args.get<size_t>("Imin",2ul);
	Imax   = args.get<size_t>("Imax",50ul);
	tol_eigval = args.get<double>("tol_eigval",1e-7);
	tol_state  = args.get<double>("tol_state",1e-7);
	max_Nrich = args.get<int>("max_Nrich",-1);
	
	vector<Param> SweepParams;
	// SweepParams.push_back({"max_alpha",alpha});
	// SweepParams.push_back({"eps_svd",eps_svd});
	SweepParams.push_back({"max_halfsweeps",Imax});
	SweepParams.push_back({"min_halfsweeps",Imin});
	// SweepParams.push_back({"Dinit",Dinit});
	SweepParams.push_back({"Qinit",Qinit});
	SweepParams.push_back({"min_Nsv",min_Nsv});
	// SweepParams.push_back({"Dlimit",Dlimit});
	// SweepParams.push_back({"tol_eigval",tol_eigval});
	// SweepParams.push_back({"tol_state",tol_state});
	// SweepParams.push_back({"max_Nrich",max_Nrich});
//	SweepParams.push_back({"CONVTEST",DMRG::CONVTEST::VAR_HSQ});
	
	CALC_DYNAMICS = args.get<bool>("CALC_DYN",0);
	dt = args.get<double>("dt",0.1);
	tmax = args.get<double>("tmax",6.);
	
	lout << args.info() << endl;
	lout.set(make_string("L=",L,"_Ly=",Ly,"_M=",M,"_D=",D,"_J=",J,".log"),"log");
	
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	//--------U(0)---------
	if (U0)
	{
		lout << endl << "--------U(0)---------" << endl << endl;
		
		Stopwatch<> Watch_U0;
		VMPS::Heisenberg H_U0(L,{{"J",J},{"Jprime",Jprime},{"Jrung",Jrung},{"D",D,0},{"D",D1,1},{"Ly",Ly}});
		lout << H_U0.info() << endl;
		
		VMPS::Heisenberg::Solver DMRG_U0(VERB);
		DMRG_U0.GlobParam = H_U0.get_GlobParam(SweepParams);
		DMRG_U0.DynParam = H_U0.get_DynParam(SweepParams);
		DMRG_U0.edgeState(H_U0, g_U0, {}, LANCZOS::EDGE::GROUND);
		
		t_U0 = Watch_U0.time();
	}
	
	//--------U(1)---------
	if (U1)
	{
		lout << endl << "--------U(1)---------" << endl << endl;
		
		Stopwatch<> Watch_U1;
		VMPS::HeisenbergU1 H_U1(L,{{"J",J},{"Jprime",Jprime},{"Jrung",Jrung},{"D",D,0},{"D",D1,1},{"Ly",Ly}});
		lout << H_U1.info() << endl;
		
		VMPS::HeisenbergU1::Solver DMRG_U1(VERB);
		DMRG_U1.userSetGlobParam();
		DMRG_U1.userSetDynParam();
		DMRG_U1.GlobParam = H_U1.get_GlobParam(SweepParams);
		DMRG_U1.DynParam = H_U1.get_DynParam(SweepParams);
		DMRG_U1.edgeState(H_U1, g_U1, {M}, LANCZOS::EDGE::GROUND);
		g_U1.state.graph("U1");
		
		t_U1 = Watch_U1.time();
		
		// dynamics (of Néel state)
		if (CALC_DYNAMICS)
		{
			lout << "-------DYNAMICS-------" << endl;
			vector<double> Jz_list = {0., -1., -2., -4.};
	//		vector<double> Jz_list = {0.};
			
			for (const auto& Jz:Jz_list)
			{
				VMPS::HeisenbergU1XXZ H_U1t(Ldyn,{{"Jxy",J},{"Jz",Jz},{"D",D}});
				VMPS::HeisenbergU1XXZ::StateXcd Psi = Neel(H_U1t);
				TDVPPropagator<VMPS::HeisenbergU1XXZ,Sym::U1<Sym::SpinU1>,double,complex<double>,VMPS::HeisenbergU1XXZ::StateXcd> TDVP(H_U1t,Psi);
			
				double t = 0;
				ofstream Filer(make_string("Mstag_Jxy=",J,"_Jz=",Jz,".dat"));
				for (int i=0; i<=static_cast<int>(tmax/dt); ++i)
				{
					double res = 0;
					for (int l=0; l<Ldyn; ++l)
					{
						res += pow(-1.,l) * isReal(avg(Psi, H_U1t.Sz(l), Psi));
					}
					res /= Ldyn;
					if (VERB != DMRG::VERBOSITY::SILENT) {lout << "t=" << t << ", <Sz>=" << res << endl;}
					Filer << t << "\t" << res << endl;
				
					TDVP.t_step(H_U1t,Psi, -1.i*dt, 1,1e-8);
					if (VERB != DMRG::VERBOSITY::SILENT) {lout << TDVP.info() << endl << Psi.info() << endl;}
					t += dt;
				}
				Filer.close();
			}
		}
	}
	
	// --------SU(2)---------
	if (SU2)
	{
		lout << endl << "--------SU(2)---------" << endl << endl;
		
		Stopwatch<> Watch_SU2;
		VMPS::HeisenbergSU2 H_SU2(L,{{"J",J},{"Jprime",Jprime},{"Jrung",Jrung},{"D",D,0},{"D",D1,1},{"Ly",Ly},{"CALC_SQUARE",false}});
		lout << H_SU2.info() << endl;
		
		VMPS::HeisenbergSU2::Solver DMRG_SU2(VERB);
		DMRG_SU2.GlobParam = H_SU2.get_GlobParam(SweepParams);
		DMRG_SU2.DynParam = H_SU2.get_DynParam(SweepParams);
		DMRG_SU2.edgeState(H_SU2, g_SU2, {S}, LANCZOS::EDGE::GROUND);
		g_SU2.state.graph("SU2");
		
		t_SU2 = Watch_SU2.time();
	}
	
	//--------output---------
	TextTable T( '-', '|', '+' );
	
	double V = L*Ly; double Vsq = V*V;
	
	// header
	T.add("");
	T.add("U(0)");
	T.add("U(1)");
	T.add("SU(2)");
	T.endOfRow();
	
	// energy
	T.add("E/L");
	T.add(to_string_prec(g_U0.energy/V));
	T.add(to_string_prec(g_U1.energy/V));
	T.add(to_string_prec(g_SU2.energy/V));
	T.endOfRow();
	
	// energy error
	T.add("E/L diff");
	T.add(to_string_prec(abs(g_U0.energy-g_SU2.energy)/V,true));
	T.add(to_string_prec(abs(g_U1.energy-g_SU2.energy)/V,true));
	T.add("0");
	T.endOfRow();
	
	// Compressor
	T.add("E/L Compressor");
	T.add(to_string_prec(E_U0_compressor/V));
	T.add(to_string_prec(E_U1_compressor/V));
	T.add("-"); T.endOfRow();
	
	// Zipper
	T.add("E/L Zipper");
	T.add(to_string_prec(E_U0_zipper/V));
	T.add(to_string_prec(E_U1_zipper/V));
	T.add("-"); T.endOfRow();
	
	// time
	T.add("t/s");
	T.add(to_string_prec(t_U0,false,2));
	T.add(to_string_prec(t_U1,false,2));
	T.add(to_string_prec(t_SU2,false,2));
	T.endOfRow();
	
	// time gain
	T.add("t gain");
	T.add(to_string_prec(t_U0/t_SU2,false,2));
	T.add(to_string_prec(t_U1/t_SU2,false,2));
	T.add("1");
	T.endOfRow();
	
	// bond dimensions
	T.add("Dmax");
	T.add(to_string(g_U0.state.calc_Dmax()));
	T.add(to_string(g_U1.state.calc_Dmax()));
	T.add(to_string(g_SU2.state.calc_Dmax()));
	T.endOfRow();
	T.add("Mmax");
	T.add(to_string(g_U0.state.calc_Dmax()));
	T.add(to_string(g_U1.state.calc_Mmax()));
	T.add(to_string(g_SU2.state.calc_Mmax()));
	T.endOfRow();
	
	lout << endl << T;
	
	lout << "ref=" << VMPS::Heisenberg::ref({{"J",J},{"Jprime",Jprime},{"D",D},{"Ly",Ly},{"m",static_cast<double>(M/(L*Ly))}}) << endl;
}
