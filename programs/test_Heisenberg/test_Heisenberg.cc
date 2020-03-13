//#ifdef BLAS
//#include "util/LapackManager.h"
//#pragma message("LapackManager")
//#endif
#include "util/LapackManager.h"

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

#include "Geometry2D.h"
#include "Lattice2D.h"

#include "solvers/DmrgSolver.h"
#include "solvers/TDVPPropagator.h"
#include "solvers/MpsCompressor.h"

#include "models/HeisenbergSU2.h"
#include "models/HeisenbergU1.h"
#include "models/HeisenbergU1XXZ.h"
#include "models/Heisenberg.h"
#include "models/HeisenbergXYZ.h"
#include "HDF5Interface.h"

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
int M, Dtot;
double Stot;
size_t D, D1;
size_t L, Ly, Ldyn;
double J, Jprime, Jrung, Jloc, Jtri;
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
MatrixXd SpinCorr_U1,SpinCorr_U1B;

MatrixXd SpinCorr_SU2, SpinCorr_SU2B;

int main (int argc, char* argv[])
{
	Sym::initialize(100,"cgc_hash/table_50.3j","cgc_hash/table_40.6j","cgc_hash/table_24.9j");

	ArgParser args(argc,argv);
	L = args.get<size_t>("L",10);
	Ly = args.get<size_t>("Ly",1);
	Ldyn = args.get<size_t>("Ldyn",12);
	J = args.get<double>("J",1.);
	Jrung = args.get<double>("Jrung",J);
	Jprime = args.get<double>("Jprime",0.);
	Jloc = args.get<double>("Jloc",0.);
	Jtri = args.get<double>("Jtri",0.);

	M = args.get<int>("M",0);
	D = args.get<size_t>("D",2);
	D1 = args.get<size_t>("D1",D);
	Dtot = abs(M)+1;
	Stot = (Dtot-1.)/2.;
	size_t min_Nsv = args.get<size_t>("min_Nsv",0ul);
	VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",2));
	
	double sigma = args.get<double>("sigma",1.);
	U0 = args.get<bool>("U0",false);
	U1 = args.get<bool>("U1",true);
	SU2 = args.get<bool>("SU2",true);

	bool PERIODIC = args.get<bool>("PER",false);
	bool RKKY = args.get<bool>("RKKY",false);
	bool ED_RKKY = args.get<bool>("ED",false);
	bool CALC_SQUARE = args.get<bool>("SQUARE",false);
	
	eps_svd = args.get<double>("eps_svd",1e-7);
	alpha = args.get<double>("alpha",1e2);
	
	Dinit  = args.get<size_t>("Dinit",2ul);
	Dlimit = args.get<size_t>("Dmax",100ul);
	Qinit  = args.get<size_t>("Qinit",7ul);
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
	SweepParams.push_back({"Dinit",Dinit});
	SweepParams.push_back({"Qinit",Qinit});
	SweepParams.push_back({"min_Nsv",min_Nsv});
	// SweepParams.push_back({"Dlimit",Dlimit});
	// SweepParams.push_back({"tol_eigval",tol_eigval});
	// SweepParams.push_back({"tol_state",tol_state});
	// SweepParams.push_back({"max_Nrich",max_Nrich});
	SweepParams.push_back({"CONVTEST",DMRG::CONVTEST::VAR_2SITE});
	
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
		VMPS::Heisenberg H_U0(L,{{"J",J},{"Jprime",Jprime},{"D",D,0},{"D",D1,1},{"Ly",Ly},{"CALC_SQUARE",CALC_SQUARE}});
		lout << H_U0.info() << endl;
		
		VMPS::Heisenberg::Solver DMRG_U0(VERB);
		DMRG_U0.GlobParam = H_U0.get_DmrgGlobParam(SweepParams);
		DMRG_U0.DynParam = H_U0.get_DmrgDynParam(SweepParams);
		DMRG_U0.edgeState(H_U0, g_U0, {}, LANCZOS::EDGE::GROUND);
		
		t_U0 = Watch_U0.time();
		for (size_t l=0; l<L; ++l)
		for (size_t lp=0; lp<L; ++lp)
		for (size_t c=0; c<Ly; ++c)
		for (size_t cp=0; cp<Ly; ++cp)
		{
			// cout << "(" << l << "," << c << "); " << "(" << lp << "," << cp << "): " << 3*avg(g_U0.state,H_U0.SzSz(l,lp,c,cp),g_U0.state) << endl;
			// SpinCorr_U1(l,lp) = 3*avg(g_U1.state,H_U1.SzSz(l,lp),g_U1.state);
		}
	}
	
	//--------U(1)---------
	if (U1)
	{
		lout << endl << "--------U(1)---------" << endl << endl;
		
		Stopwatch<> Watch_U1;
		VMPS::HeisenbergU1 H_U1(L,{{"J",J},{"Jprime",Jprime},{"D",D,0},{"D",D1,1},{"Ly",Ly},{"CALC_SQUARE",CALC_SQUARE}});
		lout << H_U1.info() << endl;
		
		VMPS::HeisenbergU1::Solver DMRG_U1(VERB);
		DMRG_U1.userSetGlobParam();
		DMRG_U1.userSetDynParam();
		DMRG_U1.GlobParam = H_U1.get_DmrgGlobParam(SweepParams);
		DMRG_U1.DynParam = H_U1.get_DmrgDynParam(SweepParams);
		DMRG_U1.edgeState(H_U1, g_U1, {M}, LANCZOS::EDGE::GROUND);
		g_U1.state.graph("U1");
		
		t_U1 = Watch_U1.time();

		// SpinCorr_U1.resize(L,L); SpinCorr_U1.setZero();
		// for (size_t l=0; l<L; ++l)
		// for (size_t lp=0; lp<L; ++lp)
		// for (size_t c=0; c<Ly; ++c)
		// for (size_t cp=0; cp<Ly; ++cp)
		// {
		// 	SpinCorr_U1(l,lp) = avg(g_U1.state,H_U1.SzSz(l,lp,c,cp),g_U1.state);
		// }
		// cout << "Spin correlations" << endl << SpinCorr_U1 << endl;
		// SpinCorr_U1B.resize(L,L); SpinCorr_U1B.setZero();
		// for (size_t l=0; l<L; ++l)
		// for (size_t lp=0; lp<L; ++lp)
		// {
		// 	VMPS::HeisenbergU1::StateXd Smg;
		// 	OxV_exact(H_U1.Scomp(SZ,l),g_U1.state,Smg,10.,DMRG::VERBOSITY::SILENT);
		// 	VMPS::HeisenbergU1::StateXd Spg;
		// 	OxV_exact(H_U1.Scomp(SZ,lp),g_U1.state,Spg,10.,DMRG::VERBOSITY::SILENT);
		// 	SpinCorr_U1B(l,lp) = Spg.dot(Smg); 
		// }
		// cout << "Spin correlations check" << endl << SpinCorr_U1B << endl;

		// for (size_t l=0; l<L; ++l)
		// {
		// 	lout << "l=" << l << "\t" 
		// 	     << "<S^z>=" << isReal(avg(g_U1.state, H_U1.Scomp(SZ,l), g_U1.state))
		// 	     << endl;
		// }
		
		// for (size_t l=0; l<L-1; ++l)
		// {
		// 	lout << "l=" << l << ", <S(i)S(i+1)>=" << isReal(avg(g_U1.state, H_U1.SpSm(l,l+1), g_U1.state)) + 
		// 	                                          isReal(avg(g_U1.state, H_U1.SzSz(l,l+1), g_U1.state)) << endl;
		// }
		
		// VMPS::HeisenbergU1XXZ H_U1XXZ(L,{{"Jxy",J},{"Jz",1.2*J},{"Jprime",Jprime},{"Jxyrung",Jrung},{"D",D,0},{"D",D1,1},{"Ly",Ly}});
		// VMPS::HeisenbergU1XXZ::Solver DMRG_U1XXZ(VERB);
		// Eigenstate<VMPS::HeisenbergU1XXZ::StateXd>  g_U1XXZ;
		// DMRG_U1XXZ.edgeState(H_U1XXZ, g_U1XXZ, {M}, LANCZOS::EDGE::GROUND);
		// cout << "dot=" << dot(g_U1.state,g_U1XXZ.state) << endl;
		
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
		typedef Sym::SU2<Sym::SpinSU2> Symmetry;
		
		Stopwatch<> Watch_SU2;
		
		VMPS::HeisenbergSU2 H_SU2;
		// H_SU2 = VMPS::HeisenbergSU2(L,{{"J",J},{"Jprime",Jprime},{"D",D,0},{"D",D1,1},{"Ly",Ly},{"CALC_SQUARE",CALC_SQUARE}});
		Lattice2D triag({1*L,Ly},{false,false});// periodic BC in y = true
		Geometry2D Geo1cell(triag,SNAKE); 
		ArrayXXd Jarray    = J * Geo1cell.hopping();
		H_SU2 = VMPS::HeisenbergSU2(L*Ly,{{"Jfull",Jarray},{"CALC_SQUARE",CALC_SQUARE}});
		
		lout << H_SU2.info() << endl;
		// H_SU2.precalc_TwoSiteData();
		VMPS::HeisenbergSU2::Solver DMRG_SU2(VERB);
		DMRG_SU2.userSetGlobParam();
		DMRG_SU2.userSetDynParam();
		DMRG_SU2.GlobParam = H_SU2.get_DmrgGlobParam(SweepParams);
		DMRG_SU2.DynParam = H_SU2.get_DmrgDynParam(SweepParams);
		DMRG_SU2.edgeState(H_SU2, g_SU2, {Dtot}, LANCZOS::EDGE::GROUND);
		g_SU2.state.graph("SU2");
		
		t_SU2 = Watch_SU2.time();
		
		cout << "E0=" << avg(g_SU2.state,H_SU2.SdagS(L/2,L/2+1),g_SU2.state) << endl;
		
		// SpinCorr_SU2.resize(L,L); SpinCorr_SU2.setZero();
		// for (size_t l=0; l<L; ++l)
		// for (size_t lp=0; lp<L; ++lp)
		// for (size_t c=0; c<Ly; ++c)
		// for (size_t cp=0; cp<Ly; ++cp)
		// {
		// 	SpinCorr_SU2(l,lp) = avg(g_SU2.state,H_SU2.SdagS(l,lp,c,cp),g_SU2.state);
		// }
		// cout << "Spin correlations" << endl << SpinCorr_SU2 << endl;
		// SpinCorr_SU2B.resize(L,L); SpinCorr_SU2B.setZero();
		// for (size_t l=0; l<L; ++l)
		// for (size_t lp=0; lp<L; ++lp)
		// {
		// 	VMPS::HeisenbergSU2::StateXd Sg;
		// 	OxV_exact(H_SU2.S(l),g_SU2.state,Sg,10.,DMRG::VERBOSITY::SILENT);
		// 	VMPS::HeisenbergSU2::StateXd Sdagg;
		// 	OxV_exact(H_SU2.S(lp),g_SU2.state,Sdagg,10.,DMRG::VERBOSITY::SILENT);
		// 	SpinCorr_SU2B(l,lp) = Sdagg.dot(Sg); 
		// }
		// cout << "Spin correlations check" << endl << SpinCorr_SU2B << endl;
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
	
	lout << "ref=" << VMPS::Heisenberg::ref({{"J",J},{"Jprime",Jprime},{"D",D},{"Ly",Ly},{"m",static_cast<double>(M)/(L*Ly)}}) << endl;

	Sym::finalize(true);
}
