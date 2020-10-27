#ifdef BLAS
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
#include "models/PeierlsHubbardSU2xU1.h"
typedef VMPS::PeierlsHubbardSU2xU1 MODELC;

#include "IntervalIterator.h"
#include "models/ParamCollection.h"

MatrixXcd onsite (int L, double Eevn, double Eodd)
{
	MatrixXcd res(L,L); res.setZero();
	for (int i=0; i<L; i+=2)
	{
		res(i,i) = Eevn;
		res(i+1,i+1) = Eodd;
	}
	return res;
}

////////////////////////////////
int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	
	size_t Ly = args.get<size_t>("Ly",1); // Ly=1: entpackt, Ly=2: Supersites
	assert(Ly==1 and "Only Ly=1 implemented");
	size_t L = args.get<size_t>("L",2); // Groesse der Einheitszelle
	int N = args.get<int>("N",L); // Teilchenzahl
	qarray<MODEL::Symmetry::Nq> Q = MODEL::singlet(N); // Quantenzahl des Grundzustandes
	lout << "Q=" << Q << endl;
	double U = args.get<double>("U",4.); // U auf den f-Plaetzen
	double V = args.get<double>("V",0.); // V*nf*nc
	double tfc = args.get<double>("tfc",0.5); // Hybridisierung fc
	double tcc = args.get<double>("tcc",1.); // Hopping fc
	double tff = args.get<double>("tff",0.); // Hopping ff
	double Retx = args.get<double>("Retx",0.); // Re Hybridisierung f(i)c(i+1)
	double Imtx = args.get<double>("Imtx",0.5); // Im Hybridisierung f(i)c(i+1)
	double Rety = args.get<double>("Rety",0.); // Re Hybridisierung c(i)f(i+1)
	double Imty = args.get<double>("Imty",0.); // Im Hybridisierung c(i)f(i+1)
	double Ec = args.get<double>("Ec",0.); // onsite-Energie fuer c
	double Ef = args.get<double>("Ef",-2.); // onsite-Energie fuer f
	bool CALC_NEUTRAL_GAP = args.get<bool>("CALC_NEUTRAL_GAP",false);
	
	// Steuert die Menge der Ausgaben
	DMRG::VERBOSITY::OPTION VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",DMRG::VERBOSITY::HALFSWEEPWISE));
	
	string wd = args.get<string>("wd","./"); correct_foldername(wd); // Arbeitsvereichnis
	string param_base = make_string("tfc=",tfc,"_tcc=",tcc,"_tff=",tff,"_tx=",Retx,",",Imtx,"_ty=",Rety,",",Imty,"_Efc=",Ef,",",Ec,"_U=",U,"_V=",V); // Dateiname
	string base = make_string("L=",L,"_N=",N,"_",param_base); // Dateiname
	lout << base << endl;
	lout.set(base+".log",wd+"log"); // Log-Datei im Unterordner log
	
	// Parameter fuer den Grundzustand:
	VUMPS::CONTROL::GLOB GlobParams;
	GlobParams.min_iterations = args.get<size_t>("min_iterations",50ul);
	GlobParams.max_iterations = args.get<size_t>("max_iterations",200ul);
	GlobParams.Minit = args.get<size_t>("Minit",2ul);
	GlobParams.Mlimit = args.get<size_t>("Mlimit",500ul);
	GlobParams.Qinit = args.get<size_t>("Qinit",2ul);
	GlobParams.tol_eigval = args.get<double>("tol_eigval",1e-5);
	GlobParams.tol_var = args.get<double>("tol_var",1e-5);
	GlobParams.tol_state = args.get<double>("tol_state",1e-4);
	GlobParams.max_iter_without_expansion = 30ul;
	
	// Gemeinsame Parameter bei unendlichen und offenen Randbedingungen
	vector<Param> params_common;
	
	// Parameter des Modells
	vector<Param> params_IBC;
	vector<Param> params_OBC;
	if (Ly==1)
	{
		// Ungerade Plaetze sollen f-Plaetze mit U sein:
		params_common.push_back({"U",0.,0});
		params_common.push_back({"U",U,1});
		
		params_common.push_back({"V",V,0});
		params_common.push_back({"V",0.,1});
		
		params_common.push_back({"t0",Ec,0});
		params_common.push_back({"t0",Ef,1});
		
		params_IBC = params_common;
		
		// Hopping
		ArrayXXcd t2cell = hopping_PAM(L,tfc+0.i,tcc+0.i,tff+0.i,Retx+1.i*Imtx,Rety+1.i*Imty);
		if (L<=4)
		{
			lout << "hopping:" << endl << t2cell << endl;
		}
		params_IBC.push_back({"tFull",t2cell});
		params_IBC.push_back({"maxPower",1ul}); // hoechste Potenz von H
		
		params_OBC = params_common;
		ArrayXXcd tOBC = hopping_PAM(L/2,tfc+0.i,tcc+0.i,tff+0.i,Retx+1.i*Imtx,Rety+1.i*Imty);
		params_OBC.push_back({"tFull",tOBC});
		params_OBC.push_back({"maxPower",2ul}); // hoechste Potenz von H
	}
	
//	// Test von komplexem Hopping
//	// Referenz:
//	// "Persistent current of a Hubbard ring threaded with a magnetic flux", Yu & Fowler (1991)
//	IntervalIterator phi(-M_PI,M_PI,51);
//	vector<double> Uvals = {20., 40., 200.};
//	for (const auto& Uval:Uvals)
//	{
//		for (phi=phi.begin(); phi!=phi.end(); ++phi)
//		{
//			lout << "phi/pi=" << *phi*M_1_PI << endl;
//			double A = *phi/L;
//			ArrayXXcd tcomplex(8,8); tcomplex.setZero();
//			tcomplex.matrix().diagonal<1>().setConstant(exp(-1.i*A));
//			tcomplex.matrix().diagonal<-1>().setConstant(exp(+1.i*A));
//			tcomplex(0,7) = exp(+1.i*A);
//			tcomplex(7,0) = exp(-1.i*A);
//			
////			cout << tcomplex << endl << endl;
//			MODELC Hc(8,{{"tFull",tcomplex},{"U",Uval}});
//			lout << Hc.info() << endl;
//			Eigenstate<MODELC::StateXcd> gc;
//			
//			DmrgSolver<MODELC::Symmetry,MODELC,complex<double>> DMRGc(VERB);
//			DMRGc.edgeState(Hc, gc, {1,4}, LANCZOS::EDGE::GROUND);
//			phi << gc.energy;
//		}
//		phi.save(make_string("E(Φ)_U=",Uval,".dat"));
//	}
	
	int Lfinite = args.get<int>("Lfinite",1000);
	auto Hfree = hopping_PAM(Lfinite/2,tfc+0.i,tcc+0.i,tff+0.i,Retx+1.i*Imtx,Rety+1.i*Imty);
	SelfAdjointEigenSolver<MatrixXcd> Eugen(Hfree.matrix()+onsite(Lfinite,Ec,Ef));
	VectorXd occ = Eugen.eigenvalues().head(Lfinite/2);
	VectorXd unocc = Eugen.eigenvalues().tail(Lfinite/2);
	double e0free = 2.*occ.sum()/Lfinite;
	lout << setprecision(16) << "e0free/(L="<<Lfinite<<",half-filling)=" << e0free << endl;
	
	if (CALC_NEUTRAL_GAP)
	{
		// Grundzustand fuer unendliches System
		Eigenstate<MODELC::StateXcd> g;
		
		MODELC H(L,params_OBC,BC::OPEN);
		lout << H.info() << endl;
		
		MODELC::Solver Salvator(VERB);
		
		DMRG::CONTROL::GLOB GlobParamsOBC;
		GlobParamsOBC.min_halfsweeps = args.get<size_t>("min_halfsweeps",30ul);
		GlobParamsOBC.max_halfsweeps = args.get<size_t>("max_halfsweeps",60ul);
		GlobParams.Minit = args.get<size_t>("Minit",10ul);
		GlobParams.Qinit = args.get<size_t>("Qinit",10ul);
		GlobParamsOBC.CONVTEST = DMRG::CONVTEST::VAR_HSQ;
		
		DMRG::CONTROL::DYN DynParamsOBC;
		size_t lim2site = args.get<size_t>("lim2site",30ul);
		DynParamsOBC.iteration = [lim2site] (size_t i) {return DMRG::ITERATION::ONE_SITE;};
		
		Salvator.userSetGlobParam();
		Salvator.GlobParam = GlobParamsOBC;
		Salvator.userSetDynParam();
		Salvator.DynParam = DynParamsOBC;
		
		Salvator.edgeState(H, g, Q, LANCZOS::EDGE::GROUND);
		
		MODELC::Solver Salvator2(VERB);
		Salvator2.userSetGlobParam();
		Salvator2.GlobParam = GlobParamsOBC;
		Salvator2.userSetDynParam();
		Salvator2.DynParam = DynParamsOBC;
		Salvator2.Epenalty = args.get<double>("Epenalty",1e4);
		
		Salvator2.push_back(g.state);
		
		Eigenstate<MODEL::StateXcd> excited1;
		excited1.state = g.state;
		excited1.state.setRandom();
		excited1.state.sweep(0,DMRG::BROOM::QR);
		excited1.state /= sqrt(dot(excited1.state,excited1.state));
		excited1.state.eps_svd = 1e-8;
		
		double overlap = abs(dot(g.state,excited1.state));
		lout << "initial overlap=" << overlap << endl;
		Salvator2.edgeState(H, excited1, Q, LANCZOS::EDGE::GROUND, true);
		lout << "excited1.energy=" << setprecision(16) << excited1.energy << setprecision(6) << endl;
		overlap = abs(dot(g.state,excited1.state));
		lout << "overlap=" << overlap << endl;
		
		lout << termcolor::blue << "L=" << L << "\t" << setprecision(16) << g.energy << "\t" << excited1.energy << ", gap=" << excited1.energy-g.energy << setprecision(6) << termcolor::reset << endl;
	}
	else
	{
		// Grundzustand fuer unendliches System
		Eigenstate<MODELC::StateUcd> g;
		
		// Aufbau des Modells
		MODELC H(L,params_IBC,BC::INFINITE);
		H.transform_base(Q,false); // PRINT=false
		H.precalc_TwoSiteData(true); // FORCE=true
		lout << H.info() << endl;
		
		// VUMPS-Solver
		MODELC::uSolver Salvator(VERB);
		Salvator.userSetGlobParam();
		Salvator.GlobParam = GlobParams;
		Salvator.edgeState(H, g, Q, LANCZOS::EDGE::GROUND);
		
		lout << setprecision(16) << "g.energy=" << g.energy << ", e0free=" << e0free << endl;
		
		// Besetzungszahlen
		for (int l=0; l<L; ++l)
		{
			lout << "l=" << l << ", n=" << isReal(avg(g.state, H.n(l), g.state)) << endl;
		}
		// fc Spin-Spin-Korrelationen
		if (Ly==1)
		{
			for (int l=0; l<L; l=l+2)
			{
				lout << "l=" << l << "," << l+1 << ", SdagS=" << isReal(avg(g.state, H.SdagS(l,l+1), g.state)) << endl;
			}
		}
	}
}
