#ifdef BLAS
#include "util/LapackManager.h"
#pragma message("LapackManager")
#endif

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

#include "DmrgTypedefs.h"
#include "solvers/DmrgSolver.h"

#include "models/KondoU1xU1.h"
#include "models/KondoU1.h"
#include "models/KondoSU2xU1.h"
#include "models/KondoU0xSU2.h"
#include "models/KondoSU2xSU2.h"

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

bool CALC_DYNAMICS;
int M, S;
size_t D;
size_t L, Ly;
int N, T;
double J, U, t, tPrime, Bx, Bz;
double alpha;
double t_U1, t_U1xU1, t_SU2xU1, t_U0xSU2, t_SO4;
int Dinit, Dlimit, Imin, Imax;
double tol_eigval, tol_state;
double dt;
DMRG::VERBOSITY::OPTION VERB;
bool U1, U1U1, SU2, U0SU2, SO4, CORR, PRINT;
string OP;

MatrixXd SpinCorr_U1xU1, nCorr_U1xU1, densityMatrixA_U1xU1, densityMatrixB_U1xU1;
MatrixXd SpinCorr_SU2xU1, nCorr_SU2xU1, densityMatrixA_SU2xU1, densityMatrixB_SU2xU1;

VectorXd d_U1, d_U1xU1, d_SU2xU1;
VectorXd n_U1, n_U1xU1, n_SU2xU1;
VectorXd expS_U1, expS1_U1xU1, expS2_U1xU1, expS_SU2xU1, expS1dag_U1xU1, expS2dag_U1xU1, expSdag_SU2xU1;

qarray<2> Qc_U1xU1, Qc_SU2xU1;

Eigenstate<VMPS::KondoU1::StateXd> g_U1;
Eigenstate<VMPS::KondoU1xU1::StateXd> g_U1xU1;
Eigenstate<VMPS::KondoSU2xU1::StateXd> g_SU2xU1;
Eigenstate<VMPS::KondoU0xSU2::StateXd> g_U0xSU2;
Eigenstate<VMPS::KondoSU2xSU2::StateXd> g_SO4;

int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",4ul);
	Ly = args.get<size_t>("Ly",1ul);
	
	J = args.get<double>("J",1.);
	t = args.get<double>("t",1.);
	double tRung = args.get<double>("tRung",1.);
	tPrime = args.get<double>("tPrime",0.);
	U = args.get<double>("U",0.);
	Bx = args.get<double>("Bx",0.);
	Bz = args.get<double>("Bz",0.);
	vector<size_t> imps = args.get_list<size_t>("imps",{1,2});
	
	N = args.get<int>("N",L*Ly);
	M = args.get<int>("M",0);
	D = args.get<size_t>("D",2);
	T = args.get<int>("T",1);
	S = abs(M)+1;
	
	U1 = args.get<bool>("U1",true);
	U1U1 = args.get<bool>("U1U1",true);
	SU2 = args.get<bool>("SU2",true);
	U0SU2 = args.get<bool>("U0SU2",true);
	SO4 = args.get<bool>("SO4",true);
	CORR = args.get<bool>("CORR",false);
	PRINT = args.get<bool>("PRINT",false);
	if (CORR == false) {PRINT = false;}
	OP = args.get<string>("OP","Simp");
	
	
	alpha = args.get<double>("alpha",1.);
	VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",2));
	dt = 0.2;
	
	Dinit  = args.get<int>("Dmin",2);
	Dlimit = args.get<int>("Dmax",100);
	Imin   = args.get<int>("Imin",2);
	Imax   = args.get<int>("Imax",50);
	tol_eigval = args.get<double>("tol_eigval",1e-6);
	tol_state  = args.get<double>("tol_state",1e-5);
	
	lout << args.info() << endl;
	lout.set(make_string("L=",L,"_Ly=",Ly,"_M=",M,"_D=",D,"_J=",J,".log"),"log");
	
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	ArrayXXd tPara(2,2); tPara.setZero(); tPara(0,0)=t; tPara(1,1)=t;
	vector<Param> params;
	params.push_back({"Ly",Ly});
	params.push_back({"J",J});
	params.push_back({"t",t});
	// params.push_back({"tRung",tRung});
	// params.push_back({"tPara",tPara});
	params.push_back({"tPrime",tPrime});
	params.push_back({"U",U});
	//params.push_back({"Bz",Bz,0});
	//params.push_back({"Bx",Bx,0});
//	params.push_back({"D",2ul});
	// params.push_back({"D",1ul,1});
	// params.push_back({"CALC_SQUARE",true});
	
	for (size_t l=0; l<L; ++l)
	{
		auto it = find(imps.begin(), imps.end(), l);
		if (it != imps.end())
		{
			params.push_back({"D",D,l});
		}
		else
		{
			params.push_back({"D",1ul,l});
		}
	}
	
//	for (size_t l=1; l<L; ++l)
//	{
//		params.push_back({"D",1ul,l});
//		// params.push_back({"Bz",0.,l});
//		// params.push_back({"Bx",0.,l});
//	}
	
	//--------U(1)---------
	if (U1)
	{
		lout << endl << termcolor::red << "--------U(1)---------" << termcolor::reset << endl << endl;
		
		Stopwatch<> Watch_U1;
		
		VMPS::KondoU1 H_U1(L,params);
		lout << H_U1.info() << endl;
//		assert(H_U1.validate({N}) and "Bad total quantum number of the MPS.");
		
		VMPS::KondoU1::Solver DMRG_U1(VERB);
		DMRG_U1.edgeState(H_U1, g_U1, {N}, LANCZOS::EDGE::GROUND);
		
		t_U1 = Watch_U1.time();
	}
	
	//--------U(1)xU(1)---------
	if (U1U1)
	{
		lout << endl << termcolor::red << "--------U(1)⊗U(1)---------" << termcolor::reset << endl << endl;
		
		Stopwatch<> Watch_U1xU1;
		
		VMPS::KondoU1xU1 H_U1xU1(L,params);
		lout << H_U1xU1.info() << endl;
//		assert(H_U1xU1.validate({M,N}) and "Bad total quantum number of the MPS.");
		
		VMPS::KondoU1xU1::Solver DMRG_U1xU1(VERB);
		DMRG_U1xU1.edgeState(H_U1xU1, g_U1xU1, {M,N}, LANCZOS::EDGE::GROUND);
		
		t_U1xU1 = Watch_U1xU1.time();
		
//		VMPS::KondoU1::StateXd Vred = g_U1.state;
//		Vred.graph("V");
//		Vred.reduce_symmetry(0,g_U1xU1.state);
//		cout << "Vred done!" << endl;
//		Vred.graph("Vred");
//		cout << Vred.validate() << endl;
//		g_U1xU1.state.graph("Vfull");
//		VMPS::KondoU1 H_U1(L,params);
//		cout << "dot=" << Vred.squaredNorm() << endl;
//		cout << Vred << endl;
//		cout << "red.avg=" << avg(Vred, H_U1, Vred) << endl;
		
		if (CORR)
		{
			d_U1xU1.resize(L); d_U1xU1.setZero();
			n_U1xU1.resize(L); n_U1xU1.setZero();
			for (size_t i=0; i<L; i++)
			{
				d_U1xU1(i) = avg(g_U1xU1.state, H_U1xU1.d(i), g_U1xU1.state);
				n_U1xU1(i) = avg(g_U1xU1.state, H_U1xU1.n(i), g_U1xU1.state);
			}
			
			SpinCorr_U1xU1.resize(L,L);
			SpinCorr_U1xU1.setZero();
			for (size_t i=0; i<L; i++)
			for (size_t j=0; j<L; j++)
			{
				SpinCorr_U1xU1(i,j) = avg(g_U1xU1.state, H_U1xU1.SimpSimp(i,j), g_U1xU1.state);
			}
			
			nCorr_U1xU1.resize(L,L);
			nCorr_U1xU1.setZero();
			for (size_t i=0; i<L; i++)
			for (size_t j=0; j<L; j++)
			{
				nCorr_U1xU1(i,j) = avg(g_U1xU1.state, H_U1xU1.nn(i,j), g_U1xU1.state);
			}
			
			densityMatrixA_U1xU1.resize(L,L);
			densityMatrixA_U1xU1.setZero();
			for (size_t i=0; i<L; i++)
			for (size_t j=0; j<L; j++)
			{
				densityMatrixA_U1xU1(i,j) = avg(g_U1xU1.state, H_U1xU1.cdagc<UP>(i,j), g_U1xU1.state)+
				                            avg(g_U1xU1.state, H_U1xU1.cdagc<DN>(i,j), g_U1xU1.state);
			}
			
			densityMatrixB_U1xU1.resize(L,L);
			densityMatrixB_U1xU1.setZero();
			for (size_t i=0; i<L; i++)
			for (size_t j=0; j<L; j++)
			{
				densityMatrixB_U1xU1(i,j) = avg(g_U1xU1.state, H_U1xU1.cdag<UP>(i), H_U1xU1.c<UP>(j), g_U1xU1.state)
					                      + avg(g_U1xU1.state, H_U1xU1.cdag<DN>(i), H_U1xU1.c<DN>(j), g_U1xU1.state);
			}			
		}
	}
	// --------SU(2)---------
	if (SU2)
	{
		lout << endl << termcolor::red << "--------SU(2)---------" << termcolor::reset << endl << endl;
		
		Stopwatch<> Watch_SU2xU1;
		VMPS::KondoSU2xU1 H_SU2xU1(L,params);
		lout << H_SU2xU1.info() << endl;
//		assert(H_SU2xU1.validate({S,N}) and "Bad total quantum number of the MPS.");
		
		VMPS::KondoSU2xU1::Solver DMRG_SU2xU1(VERB);
		DMRG_SU2xU1.edgeState(H_SU2xU1, g_SU2xU1, {S,N}, LANCZOS::EDGE::GROUND);
		
		t_SU2xU1 = Watch_SU2xU1.time();
		
		if (CORR)
		{
			cout << "corr" << endl;
			d_SU2xU1.resize(L); d_SU2xU1.setZero();
			n_SU2xU1.resize(L); n_SU2xU1.setZero();
			for (size_t i=0; i<L; i++)
			{
				d_SU2xU1(i) = avg(g_SU2xU1.state, H_SU2xU1.d(i), g_SU2xU1.state);
				n_SU2xU1(i) = avg(g_SU2xU1.state, H_SU2xU1.n(i), g_SU2xU1.state);
			}
			
			SpinCorr_SU2xU1.resize(L,L);
			SpinCorr_SU2xU1.setZero();
			for (size_t i=0; i<L; i++)
			for (size_t j=0; j<L; j++)
			{
				SpinCorr_SU2xU1(i,j) = avg(g_SU2xU1.state, H_SU2xU1.SimpSimp(i,j), g_SU2xU1.state);
			}
			cout << "corr" << endl;
			
			nCorr_SU2xU1.resize(L,L);
			nCorr_SU2xU1.setZero();
			for (size_t i=0; i<L; i++)
			for (size_t j=0; j<L; j++)
			{
				nCorr_SU2xU1(i,j) = avg(g_SU2xU1.state, H_SU2xU1.nn(i,j), g_SU2xU1.state);
			}
			
			densityMatrixA_SU2xU1.resize(L,L);
			densityMatrixA_SU2xU1.setZero();
			for (size_t i=0; i<L; i++)
			for (size_t j=0; j<L; j++)
			{
				densityMatrixA_SU2xU1(i,j) = avg(g_SU2xU1.state, H_SU2xU1.cdagc(i,j), g_SU2xU1.state);
			}
			
			densityMatrixB_SU2xU1.resize(L,L);
			densityMatrixB_SU2xU1.setZero();
			for (size_t i=0; i<L; i++)
			for (size_t j=0; j<L; j++)
			{
				densityMatrixB_SU2xU1(i,j) = avg(g_SU2xU1.state, H_SU2xU1.cdag(i,0,sqrt(2.)), H_SU2xU1.c(j,0,1.), g_SU2xU1.state);
			}
		}
	}
	// --------U(0)xSU(2)---------
	if (U0SU2)
	{
		lout << endl << termcolor::red << "--------U(0)⊗SU(2)---------" << termcolor::reset << endl << endl;
		
//		auto params_ = params;
//		params_.push_back({"subL",SUB_LATTICE::A,0});
//		params_.push_back({"subL",SUB_LATTICE::B,1});
		
		Stopwatch<> Watch_U0xSU2;
		VMPS::KondoU0xSU2 H_U0xSU2(L,params);
		lout << H_U0xSU2.info() << endl;
		
		VMPS::KondoU0xSU2::Solver DMRG_U0xSU2(VERB);
		DMRG_U0xSU2.edgeState(H_U0xSU2, g_U0xSU2, {T}, LANCZOS::EDGE::GROUND);
		
		t_U0xSU2 = Watch_U0xSU2.time();
	}
	
	if (SO4)
	{
		lout << endl << termcolor::red << "--------SU(2)⊗SU(2)---------" << termcolor::reset << endl << endl;
				
		Stopwatch<> Watch_SO4;
		VMPS::KondoSU2xSU2 H_SO4(L,params);
		lout << H_SO4.info() << endl;
		
		VMPS::KondoSU2xSU2::Solver DMRG_SO4(VERB);
		DMRG_SO4.edgeState(H_SO4, g_SO4, {S,T}, LANCZOS::EDGE::GROUND);
		
		t_SO4 = Watch_SO4.time();
	}
	
	//-------------correlations-----------------
	if (PRINT)
	{
		lout << endl << termcolor::blue << "--------Observables---------" << termcolor::reset << endl << endl;
		
		cout << "<SS> U(1)⊗U(1):" << endl;
		cout << SpinCorr_U1xU1 << endl;
		cout << endl;
		cout << "<SS> SU(2)⊗U(1):" << endl;
		cout << SpinCorr_SU2xU1 << endl;
		cout << endl;
		
		cout << "<nn> U(1)⊗U(1):" << endl;
		cout << nCorr_U1xU1 << endl;
		cout << endl;
		cout << "<nn> SU(2)⊗U(1):" << endl;
		cout << nCorr_SU2xU1 << endl;
		cout << endl;
		
		cout << "density matrixA U(1)⊗U(1): " << endl;
		cout << densityMatrixA_U1xU1 << endl << endl;
		cout << "density matrixA SU(2)⊗U(1): " << endl;
		cout << densityMatrixA_SU2xU1 << endl << endl;

		cout << "density matrixB U(1)⊗U(1): " << endl;
		cout << densityMatrixB_U1xU1 << endl << endl;
		cout << "density matrixB SU(2)⊗U(1): " << endl;
		cout << densityMatrixB_SU2xU1 << endl << endl;		
	}
	
	//--------output---------
	
	TextTable T( '-', '|', '+' );
	
	double V = L*Ly; double Vsq = V*V;
	
	T.add("");
	T.add("U(1)");
	T.add("U(1)xU(1)");
	T.add("SU(2)xU(1)");
	T.add("U(0)xSU(2)");
	T.add("SO(4)");
	T.endOfRow();
	
	T.add("E/L");
	T.add(to_string_prec(g_U1.energy/V));
	T.add(to_string_prec(g_U1xU1.energy/V));
	T.add(to_string_prec(g_SU2xU1.energy/V));
	T.add(to_string_prec(g_U0xSU2.energy/V));
	T.add(to_string_prec(g_SO4.energy/V));
	T.endOfRow();
	
	T.add("E/L diff");
	T.add(to_string_prec(abs(g_U1.energy-g_SU2xU1.energy)/V,true));
	T.add(to_string_prec(abs(g_U1xU1.energy-g_SU2xU1.energy)/V,true));
	T.add("0");
	T.add(to_string_prec(abs(g_U0xSU2.energy-g_SU2xU1.energy)/V,true));
	T.add(to_string_prec(abs(g_SO4.energy-g_SU2xU1.energy)/V,true));
	T.endOfRow();
	
	T.add("t/s");
	T.add(to_string_prec(t_U1,false,2));
	T.add(to_string_prec(t_U1xU1,false,2));
	T.add(to_string_prec(t_SU2xU1,false,2));
	T.add(to_string_prec(t_U0xSU2,false,2));
	T.add(to_string_prec(t_SO4,false,2));
	T.endOfRow();
	
	T.add("t gain");
	T.add(to_string_prec(t_U1/t_SU2xU1,false,2));
	T.add(to_string_prec(t_U1xU1/t_SU2xU1,false,2));
	T.add("1");
	T.add(to_string_prec(t_U0xSU2/t_SU2xU1,false,2));
	T.add(to_string_prec(t_SO4/t_SU2xU1,false,2));
	T.endOfRow();
	
	if (CORR)
	{
		T.add("<d>");
		T.add("-");
		T.add(to_string_prec(d_U1xU1.sum()));
		T.add(to_string_prec(d_SU2xU1.sum()));
		T.add("-");
		T.endOfRow();
		
		T.add("<d> diff");
		T.add("-");
		T.add(to_string_prec((d_U1xU1-d_SU2xU1).norm(),true));
		T.add("0");
		T.add("-");
		T.endOfRow();
		
		T.add("<SS>");
		T.add("-");
		T.add(to_string_prec(SpinCorr_U1xU1.sum()));
		T.add(to_string_prec(SpinCorr_SU2xU1.sum()));
		T.add("-");
		T.endOfRow();
		
		T.add("<SS> diff");
		T.add("-");
		T.add(to_string_prec((SpinCorr_U1xU1-SpinCorr_SU2xU1).norm(),true));
		T.add("0");
		T.add("-");
		T.endOfRow();
		
		T.add("rhoA");
		T.add("-");
		T.add(to_string_prec(densityMatrixA_U1xU1.sum()));
		T.add(to_string_prec(densityMatrixA_SU2xU1.sum()));
		T.add("-");
		T.endOfRow();
		
		T.add("rhoA diff");
		T.add("-");
		T.add(to_string_prec((densityMatrixA_U1xU1-densityMatrixA_SU2xU1).norm(),true));
		T.add("0");
		T.add("-");
		T.endOfRow();

		T.add("rhoB");
		T.add("-");
		T.add(to_string_prec(densityMatrixB_U1xU1.sum()));
		T.add(to_string_prec(densityMatrixB_SU2xU1.sum()));
		T.add("-");
		T.endOfRow();
		
		T.add("rhoB diff");
		T.add("-");
		T.add(to_string_prec((densityMatrixB_U1xU1-densityMatrixB_SU2xU1).norm(),true));
		T.add("0");
		T.add("-");
		T.endOfRow();
	}
	
	T.add("Dmax");
	T.add(to_string(g_U1.state.calc_Dmax()));
	T.add(to_string(g_U1xU1.state.calc_Dmax()));
	T.add(to_string(g_SU2xU1.state.calc_Dmax()));
	T.add(to_string(g_U0xSU2.state.calc_Dmax()));
	T.add(to_string(g_SO4.state.calc_Dmax()));
	T.endOfRow();
	
	T.add("Mmax");
	T.add(to_string(g_U1.state.calc_Dmax()));
	T.add(to_string(g_U1xU1.state.calc_Mmax()));
	T.add(to_string(g_SU2xU1.state.calc_Mmax()));
	T.add(to_string(g_U0xSU2.state.calc_Mmax()));
	T.add(to_string(g_SO4.state.calc_Mmax()));
	T.endOfRow();
	
	lout << T << endl;
}
