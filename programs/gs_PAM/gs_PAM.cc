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
	double tfc = args.get<double>("tfc",1.); // Hybridisierung fc
	double tcc = args.get<double>("tcc",1.); // Hopping fc
	double tff = args.get<double>("tff",0.); // Hopping ff
	double Retx = args.get<double>("Retx",0.); // Re Hybridisierung f(i)c(i+1)
	double Imtx = args.get<double>("Imtx",0.); // Im Hybridisierung f(i)c(i+1)
	double Rety = args.get<double>("Rety",0.); // Re Hybridisierung c(i)f(i+1)
	double Imty = args.get<double>("Imty",0.); // Im Hybridisierung c(i)f(i+1)
	double Ec = args.get<double>("Ec",0.); // onsite-Energie fuer c
	double Ef = args.get<double>("Ef",-0.5*U); // onsite-Energie fuer f
	bool CALC_NEUTRAL_GAP = args.get<bool>("CALC_NEUTRAL_GAP",false);
	bool CALC_TRIPLET_GAP = args.get<bool>("CALC_TRIPLET_GAP",false);
	
	// Steuert die Menge der Ausgaben
	DMRG::VERBOSITY::OPTION VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",DMRG::VERBOSITY::HALFSWEEPWISE));
	
	string wd = args.get<string>("wd","./"); correct_foldername(wd); // Arbeitsvereichnis
	string param_base = make_string("tfc=",tfc,"_tcc=",tcc,"_tff=",tff,"_tx=",Retx,",",Imtx,"_ty=",Rety,",",Imty,"_Efc=",Ef,",",Ec,"_U=",U,"_V=",V); // Dateiname
	string base = make_string("L=",L,"_N=",N,"_",param_base); // Dateiname
	lout << base << endl;
	lout.set(base+".log",wd+"log"); // Log-Datei im Unterordner log
	
	// Parameter fuer den Grundzustand:
	VUMPS::CONTROL::GLOB GlobSweepParams;
	GlobSweepParams.min_iterations = args.get<size_t>("min_iterations",50ul);
	GlobSweepParams.max_iterations = args.get<size_t>("max_iterations",200ul);
	GlobSweepParams.Minit = args.get<size_t>("Minit",1ul);
	GlobSweepParams.Mlimit = args.get<size_t>("Mlimit",500ul);
	GlobSweepParams.Qinit = args.get<size_t>("Qinit",1ul);
	GlobSweepParams.tol_eigval = args.get<double>("tol_eigval",1e-12);
	GlobSweepParams.tol_var = args.get<double>("tol_var",1e-8);
	GlobSweepParams.tol_state = args.get<double>("tol_state",1e-6);
	GlobSweepParams.max_iter_without_expansion = args.get<size_t>("max_iter_without_expansion",20ul);
	GlobSweepParams.CALC_S_ON_EXIT = false;
	
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
	MatrixXcd Hfree = -1.*hopping_PAM(Lfinite/2,tfc+0.i,tcc+0.i,tff+0.i,Retx+1.i*Imtx,Rety+1.i*Imty);
	Hfree += onsite(Lfinite,Ec,Ef);
	SelfAdjointEigenSolver<MatrixXcd> Eugen(Hfree);
	VectorXd occ = Eugen.eigenvalues().head(Lfinite/2);
	VectorXd unocc = Eugen.eigenvalues().tail(Lfinite/2);
	double e0free = 2.*occ.sum()/Lfinite;
	lout << termcolor::bold << setprecision(16) 
	     << "e0free/(L="<<Lfinite<<",half-filling)=" << e0free << setprecision(6) << ", HOMO-LUMO=" << unocc(0)-occ(occ.rows()-1) 
	     << termcolor::reset << endl << endl;
	
	if (CALC_NEUTRAL_GAP or CALC_TRIPLET_GAP)
	{
		// Grundzustand fuer endliches System
		Eigenstate<MODELC::StateXcd> g;
		
		MODELC H(L,params_OBC,BC::OPEN);
		lout << H.info() << endl;
		
		MODELC::Solver DMRG(VERB);
		
		DMRG::CONTROL::GLOB GlobSweepParamsOBC;
		GlobSweepParamsOBC.min_halfsweeps = args.get<size_t>("min_halfsweeps",10ul);
		GlobSweepParamsOBC.max_halfsweeps = args.get<size_t>("max_halfsweeps",20ul);
		GlobSweepParamsOBC.Minit = args.get<size_t>("Minit",2ul);
		GlobSweepParamsOBC.Qinit = args.get<size_t>("Qinit",2ul);
		GlobSweepParamsOBC.CALC_S_ON_EXIT = false;
		
		DMRG.userSetGlobParam();
		DMRG.GlobParam = GlobSweepParamsOBC;
		
		DMRG.edgeState(H, g, Q);
		
		Eigenstate<MODELC::StateXcd> gt;
		Eigenstate<MODELC::StateXcd> excited1;
		
		if (CALC_TRIPLET_GAP)
		{
			qarray<MODEL::Symmetry::Nq> Qt = {3,N};
			
			MODELC::Solver DMRG2(VERB);
			DMRG2.userSetGlobParam();
			DMRG2.GlobParam = GlobSweepParamsOBC;
			
			DMRG2.edgeState(H, gt, Qt);
			
			lout << endl;
			lout << "L=" << L << "\t" << setprecision(16) << g.energy << "\t" << gt.energy << ", triplet gap=" << gt.energy-g.energy << setprecision(6) << endl;
			
			ofstream Filer(make_string("tgap_L=",L,"_",param_base,".dat"));
			Filer << "#E0\tEtriplet\ttriplet gap" << endl;
			Filer << setprecision(16) << g.energy << "\t" << gt.energy << "\t" << gt.energy-g.energy << endl;
			Filer.close();
		}
		if (CALC_NEUTRAL_GAP)
		{
			MODELC::Solver DMRG2(VERB);
			DMRG2.userSetGlobParam();
			DMRG2.GlobParam = GlobSweepParamsOBC;
			GlobSweepParamsOBC.CONVTEST = DMRG::CONVTEST::VAR_HSQ;
			DMRG2.Epenalty = args.get<double>("Epenalty",1e4);
			
			DMRG::CONTROL::DYN DynParamsOBC;
			DMRG::ITERATION::OPTION ITALG = static_cast<DMRG::ITERATION::OPTION>(args.get<int>("ITALG",2));
			DynParamsOBC.iteration = [ITALG] (size_t i) {return ITALG;}; // [lim2site]
			
			DMRG2.userSetDynParam();
			DMRG2.DynParam = DynParamsOBC;
			
			DMRG2.push_back(g.state);
			
			excited1.state = g.state;
//			excited1.state.setRandom();
			excited1.state.sweep(0,DMRG::BROOM::QR);
			excited1.state.sweepStep(DMRG::DIRECTION::LEFT, 0, DMRG::BROOM::QR); // eliminates large numbers
			excited1.state /= sqrt(dot(excited1.state,excited1.state));
			excited1.state.eps_svd = 1e-8;
			
			double overlap = abs(dot(g.state,excited1.state));
			lout << endl << "initial overlap=" << overlap << endl;
			DMRG2.edgeState(H, excited1, Q, LANCZOS::EDGE::GROUND, true);
			lout << "excited1.energy=" << setprecision(16) << excited1.energy << setprecision(6) << endl;
			overlap = abs(dot(g.state,excited1.state));
			lout << "overlap=" << overlap << endl;
			
			ofstream Filer(make_string("ngap_L=",L,"_",param_base,".dat"));
			Filer << "#E0\tEneutral\tneutral gap\toverlap" << endl;
			Filer << setprecision(16) << g.energy << "\t" << excited1.energy << "\t" << excited1.energy-g.energy << "\t" << overlap << endl;
			Filer.close();
		}
		
		lout << endl;
		if (CALC_TRIPLET_GAP) lout << termcolor::bold << "L=" << L << "\t" << setprecision(16) << g.energy << "\t" << gt.energy << ", triplet gap=" << gt.energy-g.energy << setprecision(6) << termcolor::reset << endl;
		if (CALC_NEUTRAL_GAP) lout << termcolor::bold << "L=" << L << "\t" << setprecision(16) << g.energy << "\t" << excited1.energy << ", neutral gap=" << excited1.energy-g.energy << setprecision(6) << termcolor::reset << endl;
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
		MODELC::uSolver DMRG(VERB);
		DMRG.userSetGlobParam();
		DMRG.GlobParam = GlobSweepParams;
		DMRG.edgeState(H, g, Q, LANCZOS::EDGE::GROUND);
		
		lout << setprecision(16) << "g.energy=" << g.energy << ", e0free=" << e0free << endl;
		
		// observables
		HDF5Interface target("obs_"+param_base+make_string("_Lcell=",L)+".h5",WRITE);
		
		VectorXd n(L); n.setZero();
		VectorXd d(L); d.setZero();
		for (int l=0; l<L; ++l)
		{
			n(l) = isReal(avg(g.state, H.n(l), g.state));
			d(l) = isReal(avg(g.state, H.d(l), g.state));
			lout << "l=" << l << ", n=" << n(l) << endl;
			lout << "l=" << l << ", d=" << d(l) << endl;
		}
		target.save_vector(n,"n");
		target.save_vector(d,"d");
		
		VectorXd SdagS_cc(50); SdagS_cc.setZero();
		VectorXd SdagS_ff(50); SdagS_ff.setZero();
		VectorXd SdagS_cf(50); SdagS_cf.setZero();
		VectorXd SdagS_fc(50); SdagS_fc.setZero();
		VectorXd nn_cc(50); nn_cc.setZero();
		VectorXd nn_ff(50); nn_ff.setZero();
		VectorXd nn_cf(50); nn_cf.setZero();
		VectorXd nn_fc(50); nn_fc.setZero();
		VectorXd SdagSc(100); SdagSc.setZero();
		VectorXd SdagSf(100); SdagSf.setZero();
		
		double SdagS_loc = isReal(avg(g.state, H.SdagS(0,1), g.state));
		double nn_loc = isReal(avg(g.state, H.nn(0,1), g.state));
		lout << "SdagS=" << SdagS_loc << endl;
		lout << "nn=" << nn_loc << endl;
		
		#pragma omp parallel for
		for (int l=0; l<100; l+=2)
		{
			int Laux=0;
			while (Laux<l+3) Laux += L;
			MODEL Haux(Laux, {{"maxPower",1ul}}, BC::INFINITE, DMRG::VERBOSITY::SILENT);
			Haux.transform_base(Q,false); // PRINT=false
			
			SdagS_cc(l/2) = isReal(avg(g.state, Haux.SdagS(0,l), g.state));
			SdagS_ff(l/2) = isReal(avg(g.state, Haux.SdagS(1,l+1), g.state));
			SdagS_cf(l/2) = isReal(avg(g.state, Haux.SdagS(0,l+1), g.state));
			SdagS_fc(l/2) = isReal(avg(g.state, Haux.SdagS(1,l), g.state));
			
			nn_cc(l/2) = isReal(avg(g.state, Haux.nn(0,l), g.state));
			nn_ff(l/2) = isReal(avg(g.state, Haux.nn(1,l+1), g.state));
			nn_cf(l/2) = isReal(avg(g.state, Haux.nn(0,l+1), g.state));
			nn_fc(l/2) = isReal(avg(g.state, Haux.nn(1,l), g.state));
		}
		
		#pragma omp parallel for
		for (int l=0; l<100; l+=1)
		{
			int Laux=0;
			while (Laux<l+3) Laux += L;
			MODEL Haux(Laux, {{"maxPower",1ul}}, BC::INFINITE, DMRG::VERBOSITY::SILENT);
			Haux.transform_base(Q,false); // PRINT=false
			
			SdagSc(l) = isReal(avg(g.state, Haux.SdagS(0,l), g.state));
			SdagSf(l) = isReal(avg(g.state, Haux.SdagS(1,l+1), g.state));
		}
		
		lout << "first 10 non-local spin-spin correlations:" << endl;
		lout << "cc:" << SdagS_cc.head(10).transpose() << endl;
		lout << "ff:" << SdagS_ff.head(10).transpose() << endl;
		lout << "cf:" << SdagS_cf.head(10).transpose() << endl;
		lout << "fc:" << SdagS_fc.head(10).transpose() << endl;
		
		target.save_vector(SdagS_cc,"SdagScc");
		target.save_vector(SdagS_ff,"SdagSff");
		target.save_vector(SdagS_cf,"SdagScf");
		target.save_vector(SdagS_fc,"SdagSfc");
		
		target.save_vector(SdagSf,"SdagSf");
		target.save_vector(SdagSc,"SdagSc");
		
		target.save_vector(nn_cc,"nn_cc");
		target.save_vector(nn_ff,"nn_ff");
		target.save_vector(nn_cf,"nn_cf");
		target.save_vector(nn_fc,"nn_fc");
		
		target.save_scalar(SdagS_loc,"SdagS_loc");
		target.save_scalar(nn_loc,"nn_loc");
		
		target.save_scalar(g.state.calc_Mmax(),"Mmax");
		target.save_scalar(g.state.calc_fullMmax(),"fullMmax");
		
		target.close();
	}
}
