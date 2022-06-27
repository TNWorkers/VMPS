#if defined(BLAS) or defined(BLIS) or defined(MKL)
#include "util/LapackManager.h"
#pragma message("LapackManager")
#endif

//#define USE_OLD_COMPRESSION
#define USE_HDF5_STORAGE
#define DMRG_DONT_USE_OPENMP
#define LINEARSOLVER_DIMK 100
//#define DEBUG_VERBOSITY 3

#include <iostream>
#include <fstream>
#include <complex>

#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

#include "Logger.h"
Logger lout;
#include "ArgParser.h"

#include "StringStuff.h"
#include "Stopwatch.h"

#include "solvers/DmrgSolver.h"
#include "VUMPS/VumpsSolver.h"
#include "VUMPS/VumpsLinearAlgebra.h"

#include "models/HubbardSU2xU1.h"
typedef VMPS::HubbardSU2xU1 MODEL;

#include "IntervalIterator.h"
#include "models/ParamCollection.h"

////////////////////////////////
int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	
	size_t Ly = args.get<size_t>("Ly",1);
	assert(Ly==1 and "Only Ly=1 implemented");
	size_t L = args.get<size_t>("L",12);
	int N = args.get<int>("N",L);
	qarray<MODEL::Symmetry::Nq> Q = MODEL::singlet(N);
	lout << "Q=" << Q << endl;
	double U = args.get<double>("U",4.);
	double tPrime = args.get<double>("tPrime",1.); // bonds connecting the triangles
	double V = args.get<double>("V",0.);
	double tinter = args.get<double>("tinter",1.);
	bool PAIR_BINDING = args.get<bool>("PAIR_BINDING",false);
	bool PRINT_FREE = args.get<bool>("PRINT_FREE",false);
	int dmax = args.get<int>("dmax",30);
	
	DMRG::VERBOSITY::OPTION VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",DMRG::VERBOSITY::HALFSWEEPWISE));
	
	string wd = args.get<string>("wd","./"); correct_foldername(wd);
	string param_base = make_string("U=",U,"_V=",V,"_tinter=",tinter);
	string base = make_string("L=",L,"_N=",N,"_",param_base);
	lout << base << endl;
	lout.set(base+".log",wd+"log");
	
	VUMPS::CONTROL::GLOB GlobSweepParams;
	GlobSweepParams.min_iterations = args.get<size_t>("min_iterations",50ul);
	GlobSweepParams.max_iterations = args.get<size_t>("max_iterations",200ul);
	GlobSweepParams.Minit = args.get<size_t>("Minit",1ul);
	GlobSweepParams.Mlimit = args.get<size_t>("Mlimit",800ul);
	GlobSweepParams.Qinit = args.get<size_t>("Qinit",1ul);
	GlobSweepParams.tol_eigval = args.get<double>("tol_eigval",1e-5);
	GlobSweepParams.tol_var = args.get<double>("tol_var",1e-5);
	GlobSweepParams.tol_state = args.get<double>("tol_state",1e-4);
	GlobSweepParams.max_iter_without_expansion = args.get<size_t>("max_iter_without_expansion",10ul);
	GlobSweepParams.CALC_S_ON_EXIT = false;
	
	VUMPS::CONTROL::DYN DynSweepParams;
	double eps_svd = args.get<double>("eps_svd",1e-10);
	int SEQ_PERIOD = args.get<int>("SEQ_PERIOD",10);
	DynSweepParams.iteration = [SEQ_PERIOD] (size_t i) {return (i%SEQ_PERIOD==0)? UMPS_ALG::SEQUENTIAL : UMPS_ALG::PARALLEL;};
	
	// parameters
	vector<Param> params_IBC;
	
	//ArrayXXd t1cell = hopping_Archimedean("3.4.3.4",0,1.,tPrime);
	//ArrayXXd t1cell = hopping_Platonic(L, 0, 1.);
//	
//	// coupling between two molecules
//	ArrayXXd t2cell(24,24); t2cell = 0.;
//	t2cell.topLeftCorner(12,12) = t1cell;
//	t2cell.bottomRightCorner(12,12) = t1cell;
//	t2cell(10,12) = tinter;
//	t2cell(12,10) = tinter;
	
//	ArrayXXd t1cell(2,2);
//	t1cell.setZero();
//	t1cell(0,1) = 1.;
//	t1cell(1,0) = 1.;
	
	ArrayXXd t1cell = hopping_Platonic(L);
//	
//	lout << t1cell << endl;
//	lout << endl;
////	lout << t2cell << endl;
	
//	// IBC
//	params_IBC.push_back({"U",U});
//	params_IBC.push_back({"tFull",t2cell});
//	params_IBC.push_back({"Vfull",V*t2cell});
//	params_IBC.push_back({"maxPower",1ul});
	
	// free fermions
	if (PRINT_FREE)
	{
		SelfAdjointEigenSolver<MatrixXd> Eugen(-1.*t1cell.matrix());
		lout << Eugen.eigenvalues().transpose() << endl;
		VectorXd occ = Eugen.eigenvalues().head(N/2);
		VectorXd unocc = Eugen.eigenvalues().tail(L-N/2);
		lout << "orbital energies occupied:" << endl << occ.transpose()  << endl;
		lout << "orbital energies unoccupied:" << endl << unocc.transpose()  << endl << endl;
		double E0 = 2.*occ.sum();
		lout << setprecision(16) << "non-interacting fermions: E0=" << E0 << ", E0/L=" << E0/L << setprecision(6) << endl << endl;
	}
	
	//--------------pair binding energy to check results by White et al. (1991)--------------
	if (PAIR_BINDING)
	{
		IntervalIterator Uit(0.,100.,101);
		for (Uit=Uit.begin(); Uit!=Uit.end(); ++Uit)
		{
			MODEL H(L,{{"U",*Uit},{"tFull",t1cell}},BC::OPEN);
			lout << H.info() << endl;
			
			Eigenstate<MODEL::StateXd> g0s;
			Eigenstate<MODEL::StateXd> g0t;
			
			Eigenstate<MODEL::StateXd> g1d;
			Eigenstate<MODEL::StateXd> g1q;
			
			Eigenstate<MODEL::StateXd> g2s;
			Eigenstate<MODEL::StateXd> g2t;
			Eigenstate<MODEL::StateXd> g2q;
			
			DMRG::CONTROL::GLOB GlobParams;
			GlobParams.tol_eigval = args.get<double>("tol_eigval",1e-5);
			GlobParams.tol_state = args.get<double>("tol_state",1e-4);
			GlobParams.min_halfsweeps = args.get<size_t>("min_halfsweeps",1ul);
			GlobParams.max_halfsweeps = args.get<size_t>("max_halfsweeps",36ul);
			GlobParams.Minit = args.get<size_t>("Minit",2ul);
			GlobParams.Qinit = args.get<size_t>("Qinit",2ul);
			GlobParams.CALC_S_ON_EXIT = false;
			
			DMRG::CONTROL::DYN DynParams;
			size_t Mincr_abs = args.get<size_t>("Mincr_abs",100ul);
			DynParams.Mincr_abs = [Mincr_abs] (size_t i) {return Mincr_abs;};
			
			size_t start_2site = args.get<size_t>("start_2site",0ul);
			size_t end_2site = args.get<size_t>("end_2site",0ul); //GlobParam.max_halfsweeps-3
			size_t period_2site = args.get<size_t>("period_2site",2ul);
			DynParams.iteration = [start_2site,end_2site,period_2site] (size_t i) {return (i>=start_2site and i<=end_2site and i%period_2site==0)? DMRG::ITERATION::TWO_SITE : DMRG::ITERATION::ONE_SITE;};
			
			size_t Mincr_per = args.get<size_t>("Mincr_per",4ul);
			DynParams.Mincr_per = [Mincr_per] (size_t i) {return Mincr_per;};
			
//			DMRG::ITERATION::OPTION ITALG = static_cast<DMRG::ITERATION::OPTION>(args.get<int>("ITALG",2));
//			DynParams.iteration = [ITALG] (size_t i) {return ITALG;};
			
			#pragma omp parallel sections
			{
				#pragma omp section
				{
					MODEL::Solver DMRG(VERB);
					DMRG.GlobParam = GlobParams;
					DMRG.userSetGlobParam();
					DMRG.DynParam = DynParams;
					DMRG.userSetDynParam();
					DMRG.edgeState(H, g0s, {1,N});
				}
				#pragma omp section
				{
					MODEL::Solver DMRG(DMRG::VERBOSITY::SILENT);
					DMRG.GlobParam = GlobParams;
					DMRG.userSetGlobParam();
					DMRG.DynParam = DynParams;
					DMRG.userSetDynParam();
					DMRG.edgeState(H, g0t, {3,N});
				}
				#pragma omp section
				{
					MODEL::Solver DMRG(DMRG::VERBOSITY::SILENT);
					DMRG.GlobParam = GlobParams;
					DMRG.userSetGlobParam();
					DMRG.DynParam = DynParams;
					DMRG.userSetDynParam();
					DMRG.edgeState(H, g1d, {2,N+1});
				}
				#pragma omp section
				{
					MODEL::Solver DMRG(DMRG::VERBOSITY::SILENT);
					DMRG.GlobParam = GlobParams;
					DMRG.userSetGlobParam();
					DMRG.DynParam = DynParams;
					DMRG.userSetDynParam();
					DMRG.edgeState(H, g1q, {4,N+1});
				}
				#pragma omp section
				{
					MODEL::Solver DMRG(DMRG::VERBOSITY::SILENT);
					DMRG.GlobParam = GlobParams;
					DMRG.userSetGlobParam();
					DMRG.DynParam = DynParams;
					DMRG.userSetDynParam();
					DMRG.edgeState(H, g2s, {1,N+2});
				}
				#pragma omp section
				{
					MODEL::Solver DMRG(DMRG::VERBOSITY::SILENT);
					DMRG.GlobParam = GlobParams;
					DMRG.userSetGlobParam();
					DMRG.DynParam = DynParams;
					DMRG.userSetDynParam();
					DMRG.edgeState(H, g2t, {3,N+2});
				}
				#pragma omp section
				{
					MODEL::Solver DMRG(DMRG::VERBOSITY::SILENT);
					DMRG.GlobParam = GlobParams;
					DMRG.userSetGlobParam();
					DMRG.DynParam = DynParams;
					DMRG.userSetDynParam();
					DMRG.edgeState(H, g2q, {5,N+2});
				}
			}
			double E0 = min(g0s.energy,g0t.energy);
			double E1 = min(g1d.energy,g1q.energy);
			double E2 = min(min(g2s.energy,g2t.energy),g2q.energy);
			double Epair = E0+E2-2.*E1;
			lout << "U=" << *Uit << ", Epair=" << Epair << endl;
			Uit << Epair;
			Uit.save("Epair(U).dat");
		}
	}
	//----------------------------
	
//	Eigenstate<MODEL::StateUd> g;
//	
//	// model
//	MODEL H(L,params_IBC,BC::INFINITE);
//	H.transform_base(Q,false); // PRINT=false
//	H.precalc_TwoSiteData(true); // FORCE=true
//	lout << H.info() << endl;
//	
//	// VUMPS solver
//	MODEL::uSolver DMRG(VERB);
//	DMRG.userSetGlobParam();
//	DMRG.userSetDynParam();
//	DMRG.GlobParam = GlobSweepParams;
//	DMRG.DynParam = DynSweepParams;
//	DMRG.edgeState(H, g, Q, LANCZOS::EDGE::GROUND);
//	
//	lout << setprecision(16) << "g.energy=" << g.energy << endl;
//	
//	for (int d=1; d<=dmax; ++d)
//	{
//		MODEL Haux(12*d+24,{{"maxPower",1ul}},BC::INFINITE,DMRG::VERBOSITY::SILENT); Haux.transform_base(Q,false); // PRINT=false
//		double res1 = avg(g.state, Haux.TpTm(0,12*d), g.state);
//		double res2 = avg(g.state, Haux.TzTz(0,12*d), g.state);
//		double res21 = avg(g.state, Haux.Tz(0), g.state);
//		double res22 = avg(g.state, Haux.Tz(12*d), g.state);
//		lout << "d=" << d << ", <c†c†cc>=" << res1 << ", <nn>=" << res2-res21*res22 << endl;
//	}
}
