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

#include "IntervalIterator.h"
#include "models/ParamCollection.h"

MatrixXd onsite (int L, double Eevn, double Eodd)
{
	MatrixXd res(L,L); res.setZero();
	for (int i=0; i<L; i+=2)
	{
		res(i,i)     = Eevn;
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
	double tx = args.get<double>("tx",0.); // Hybridisierung f(i)c(i+1)
	double ty = args.get<double>("ty",0.); // Hybridisierung c(i)f(i+1)
	double Ec = args.get<double>("Ec",0.); // onsite-Energie fuer c
	double Ef = args.get<double>("Ef",-0.5*U); // onsite-Energie fuer f
	bool CALC_NEUTRAL_GAP = args.get<bool>("CALC_NEUTRAL_GAP",false);
	bool CALC_TRIPLET_GAP = args.get<bool>("CALC_TRIPLET_GAP",false);
	bool PBC = args.get<bool>("PBC",false);
	string BC = (PBC)? "PBC":"OBC";
	size_t Mlimit = args.get<size_t>("Mlimit",800ul);
	
	// Steuert die Menge der Ausgaben
	DMRG::VERBOSITY::OPTION VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",DMRG::VERBOSITY::HALFSWEEPWISE));
	
	string wd = args.get<string>("wd","./"); correct_foldername(wd); // Arbeitsvereichnis
	string param_base = make_string("tfc=",tfc,"_tcc=",tcc,"_tff=",tff,"_tx=",tx,"_ty=",ty,"_Efc=",Ef,",",Ec,"_U=",U,"_V=",V,"_Mlimit=",Mlimit); // Dateiname
	string base = make_string("L=",L,"_N=",N,"_",param_base); // Dateiname
	lout << base << endl;
	lout.set(base+".log",wd+"log"); // Log-Datei im Unterordner log
	
	// Parameter fuer den Grundzustand:
	VUMPS::CONTROL::GLOB GlobSweepParams;
	GlobSweepParams.min_iterations = args.get<size_t>("min_iterations",50ul);
	GlobSweepParams.max_iterations = args.get<size_t>("max_iterations",200ul);
	GlobSweepParams.Minit = args.get<size_t>("Minit",1ul);
	GlobSweepParams.Mlimit = Mlimit;
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
	vector<Param> params_finite;
	if (Ly==1)
	{
		// Ungerade Plaetze sollen f-Plaetze mit U sein:
		params_common.push_back({"U",0.,0});
		params_common.push_back({"U",U,1});
		
		params_common.push_back({"V",V,0});
		params_common.push_back({"V",0.,1});
		
		params_common.push_back({"t0",Ec,0});
		params_common.push_back({"t0",Ef,1});
		
		// IBC
		params_IBC = params_common;
		ArrayXXd t2cell = hopping_PAM(L,tfc,tcc,tff,tx,ty);
		if (L<=4) lout << "hopping:" << endl << t2cell << endl;
		params_IBC.push_back({"tFull",t2cell});
		params_IBC.push_back({"maxPower",1ul}); // hoechste Potenz von H
		
		// OBC/PBC
		params_finite = params_common;
		ArrayXXd t_finite = hopping_PAM(L/2,tfc,tcc,tff,tx,ty,PBC);
		params_finite.push_back({"tFull",t_finite});
		params_finite.push_back({"maxPower",2ul}); // hoechste Potenz von H
	}
	
	int Lfinite = args.get<int>("Lfinite",1000);
	MatrixXd Hfree = -1.*hopping_PAM(Lfinite/2,tfc,tcc,tff,tx,ty,PBC);
	Hfree += onsite(Lfinite,Ec,Ef);
	SelfAdjointEigenSolver<MatrixXd> Eugen(Hfree);
	VectorXd occ = Eugen.eigenvalues().head(Lfinite/2);
	VectorXd unocc = Eugen.eigenvalues().tail(Lfinite/2);
	double e0free = 2.*occ.sum()/Lfinite;
	lout << termcolor::bold << setprecision(16) 
	     << "e0free/(L="<<Lfinite<<",half-filling)=" << e0free << setprecision(6) << ", HOMO-LUMO=" << unocc(0)-occ(occ.rows()-1) 
	     << termcolor::reset << endl << endl;
	
	if (CALC_NEUTRAL_GAP or CALC_TRIPLET_GAP)
	{
		// Grundzustand fuer endliches System
		Eigenstate<MODEL::StateXd> g;
		
		MODEL H(L,params_finite,BC::OPEN);
		lout << H.info() << endl << endl;
		
		MODEL::Solver DMRG(VERB);
		
		DMRG::CONTROL::GLOB GlobSweepParamsFinite;
		GlobSweepParamsFinite.tol_eigval = args.get<double>("tol_eigval",1e-5);
		GlobSweepParamsFinite.tol_state = args.get<double>("tol_state",1e-4);
		GlobSweepParamsFinite.min_halfsweeps = args.get<size_t>("min_halfsweeps",12ul);
		GlobSweepParamsFinite.max_halfsweeps = args.get<size_t>("max_halfsweeps",36ul);
		GlobSweepParamsFinite.Minit = args.get<size_t>("Minit",2ul);
		GlobSweepParamsFinite.Qinit = args.get<size_t>("Qinit",2ul);
		GlobSweepParamsFinite.CALC_S_ON_EXIT = false;
		
		DMRG::CONTROL::DYN DynSweepParamsFinite;
		size_t Mincr_abs = args.get<size_t>("Mincr_abs",100ul);
		DynSweepParamsFinite.Mincr_abs = [Mincr_abs] (size_t i) {return Mincr_abs;};
		
		size_t Mincr_per = args.get<size_t>("Mincr_per",4ul);
		DynSweepParamsFinite.Mincr_per = [Mincr_per] (size_t i) {return Mincr_per;};
		
		DMRG::ITERATION::OPTION ITALG = static_cast<DMRG::ITERATION::OPTION>(args.get<int>("ITALG",2));
		DynSweepParamsFinite.iteration = [ITALG] (size_t i) {return ITALG;};
		
		DMRG.userSetGlobParam();
		DMRG.GlobParam = GlobSweepParamsFinite;
		DMRG.userSetDynParam();
		DMRG.DynParam = DynSweepParamsFinite;
		
		DMRG.edgeState(H, g, Q);
		
		Eigenstate<MODEL::StateXd> gt;
		Eigenstate<MODEL::StateXd> excited1;
		
		if (CALC_TRIPLET_GAP)
		{
			qarray<MODEL::Symmetry::Nq> Qt = {3,N};
			
			MODEL::Solver DMRG2(VERB);
			DMRG2.userSetGlobParam();
			DMRG2.GlobParam = GlobSweepParamsFinite;
			DMRG2.userSetDynParam();
			DMRG2.DynParam = DynSweepParamsFinite;
			
			DMRG2.edgeState(H, gt, Qt);
			
			lout << endl;
			lout << "L=" << L << "\t" << setprecision(16) << g.energy << "\t" << gt.energy << ", triplet gap=" << gt.energy-g.energy << setprecision(6) << endl;
			
			ofstream Filer(make_string("tgap_L=",L,"_",param_base,"_BC=",BC,".dat"));
			Filer << "#E0\tEtriplet\ttriplet gap" << endl;
			Filer << setprecision(16) << g.energy << "\t" << gt.energy << "\t" << gt.energy-g.energy << endl;
			Filer.close();
		}
		if (CALC_NEUTRAL_GAP)
		{
			MODEL::Solver DMRG2(VERB);
			DMRG2.userSetGlobParam();
			DMRG2.GlobParam = GlobSweepParamsFinite;
			GlobSweepParamsFinite.CONVTEST = DMRG::CONVTEST::VAR_HSQ;
			DMRG2.Epenalty = args.get<double>("Epenalty",1e4);
			
			DMRG2.userSetDynParam();
			DMRG2.DynParam = DynSweepParamsFinite;
			
			DMRG2.push_back(g.state);
			
			excited1.state = g.state;
//			excited1.state.setRandom();
			excited1.state.sweep(0,DMRG::BROOM::QR);
			excited1.state.sweepStep(DMRG::DIRECTION::LEFT, 0, DMRG::BROOM::QR); // eliminates large numbers
			excited1.state /= sqrt(dot(excited1.state,excited1.state));
			excited1.state.eps_svd = 1e-8;
			
			double overlap = abs(dot(g.state,excited1.state));
			DMRG2.edgeState(H, excited1, Q, LANCZOS::EDGE::GROUND, true);
			lout << "excited1.energy=" << setprecision(16) << excited1.energy << setprecision(6) << endl;
			overlap = abs(dot(g.state,excited1.state));
			lout << "overlap=" << overlap << endl;
			
			ofstream Filer(make_string("ngap_L=",L,"_",param_base,"_BC=",BC,".dat"));
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
		Eigenstate<MODEL::StateUd> g;
		
		// Aufbau des Modells
		MODEL H(L,params_IBC,BC::INFINITE);
		H.transform_base(Q,false); // PRINT=false
		H.precalc_TwoSiteData(true); // FORCE=true
		lout << H.info() << endl;
		
		// VUMPS-Solver
		MODEL::uSolver DMRG(VERB);
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
