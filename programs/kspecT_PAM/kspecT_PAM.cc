#ifdef BLAS
#include "util/LapackManager.h"
#pragma message("LapackManager")
#endif

#define USE_HDF5_STORAGE
#define DMRG_DONT_USE_OPENMP
#define GREENPROPAGATOR_USE_HDF5

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
#include "models/ParamCollection.h"

#include "models/HubbardSU2xU1.h"
typedef VMPS::HubbardSU2xU1 MODEL;
#include "models/PeierlsHubbardSU2xU1.h"
typedef VMPS::PeierlsHubbardSU2xU1 MODELC;

#include "solvers/GreenPropagator.h"
#include "DmrgLinearAlgebra.h"
#include "solvers/SpectralManager.h"

vector<GreenPropagator<MODELC,MODELC::Symmetry,complex<double>,complex<double>>> Green;

int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	lout << args.info() << endl;
	
	#ifdef _OPENMP
	omp_set_nested(1);
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	size_t Ly = args.get<size_t>("Ly",1); // Ly=1: entpackt, Ly=2: Supersites
	int dLphys = (Ly==2)? 1:2;
	assert(Ly==1 and "Only Ly=1 implemented");
	size_t L = args.get<size_t>("L",16ul); // Groesse der Kette
	size_t Lcell = 2;
	int N = args.get<int>("N",L); // Teilchenzahl
	int x0 = L/2;
	
	qarray<MODELC::Symmetry::Nq> Q = MODELC::singlet(2*N); // Quantenzahl des Grundzustandes
	lout << "Q=" << Q << endl;
	double U = args.get<double>("U",8.); // U auf den f-Plaetzen
	double Uc = args.get<double>("Uc",0.); // U auf den c-Plaetzen
	double mu = args.get<double>("mu",0.5*U); // U auf den f-Plaetzen
	double V = args.get<double>("V",0.); // V*nc*nf
	double tfc = args.get<double>("tfc",1.); // Hybridisierung fc
	double tcc = args.get<double>("tcc",1.); // Hopping fc
	double tff = args.get<double>("tff",0.); // Hopping ff
	double Retx = args.get<double>("Retx",0.); // Re Hybridisierung f(i)c(i+1)
	double Imtx = args.get<double>("Imtx",1.); // Im Hybridisierung f(i)c(i+1)
	double Rety = args.get<double>("Rety",0.); // Re Hybridisierung c(i)f(i+1)
	double Imty = args.get<double>("Imty",0.); // Im Hybridisierung c(i)f(i+1)
	double Ec = args.get<double>("Ec",0.); // onsite-Energie fuer c
	double Ef = args.get<double>("Ef",-2.); // onsite-Energie fuer f
	
	bool SAVE_GS = args.get<bool>("SAVE_GS",false);
	bool LOAD_GS = args.get<bool>("LOAD_GS",false);
	bool RELOAD = args.get<bool>("RELOAD",false);
	bool CALC_SPEC = args.get<bool>("CALC_SPEC",true);
	
	vector<string> specs = args.get_list<string>("specs",{"PES","IPE"}); // welche Spektren? PES:Photoemission, IPE:inv. Photoemission
	string specstring = "";
	int Nspec = specs.size();
	Green.resize(Nspec);
	size_t Mlim = args.get<size_t>("Mlim",800ul);
	double dt = args.get<double>("dt",0.025);
	double tol_DeltaS = args.get<double>("tol_DeltaS",5e-3);
	double tmax = args.get<double>("tmax",4.);
	double tol_compr = args.get<double>("tol_compr",1e-4);
	int Nt = static_cast<int>(tmax/dt);
	
	double dbeta = args.get<double>("dbeta",0.1);
	double beta = args.get<double>("beta",1.);
	double tol_compr_beta = args.get<double>("tol_compr_beta",1e-5);
	int Nbeta = static_cast<int>(beta/dbeta);
	
	GREEN_INTEGRATION INT = static_cast<GREEN_INTEGRATION>(args.get<int>("INT",2)); // DIRECT=0, INTERP=1, OOURA=2
	Q_RANGE QR = static_cast<Q_RANGE>(args.get<int>("QR",1)); // MPI_PPI=0, ZERO_2PI=1
	
	double wmin = args.get<double>("wmin",-10.);
	double wmax = args.get<double>("wmax",+10.);
	int wpoints = args.get<int>("wpoints",501);
	int qpoints = args.get<int>("qpoints",501);
	
	// Steuert die Menge der Ausgaben
	DMRG::VERBOSITY::OPTION VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",DMRG::VERBOSITY::HALFSWEEPWISE));
	
	string wd = args.get<string>("wd","./"); correct_foldername(wd); // Arbeitsvereichnis
	string param_base = make_string("tfc=",tfc,"_tcc=",tcc,"_tff=",tff,"_tx=",Retx,",",Imtx,"_ty=",Rety,",",Imty,"_Efc=",Ef,",",Ec,"_U=",U);
	if (Uc!=0.) param_base += make_string("_Uc=",Uc);
	param_base += make_string("_V=",V,"_beta=",beta); // Dateiname
	string base = make_string("L=",L,"_N=",N,"_",param_base); // Dateiname
	lout << base << endl;
	lout.set(base+".log",wd+"log"); // Log-Datei im Unterordner log
	
	lout << base << endl;
	lout.set(base+".log",wd+"log"); // Log-Datei im Unterordner log
	
	// Parameter fuer den Grundzustand:
	DMRG::CONTROL::GLOB GlobParams;
	GlobParams.min_halfsweeps = args.get<size_t>("min_halfsweeps",4);
	GlobParams.max_halfsweeps = args.get<size_t>("max_halfsweeps",8);
	GlobParams.Minit = args.get<size_t>("Minit",1ul);
	GlobParams.Qinit = args.get<size_t>("Qinit",1ul);
	
	// Parameter des Modells
	vector<Param> params_Tfin;
	// l%4=2 Plaetze sollen f-Plaetze mit U sein:
	params_Tfin.push_back({"Uph",Uc,0});
	params_Tfin.push_back({"Uph",0.,1});
	params_Tfin.push_back({"Uph",U,2});
	params_Tfin.push_back({"Uph",0.,3});
//	params_Tfin.push_back({"mu",mu,0});
//	params_Tfin.push_back({"mu",0.,1});
//	params_Tfin.push_back({"mu",mu,2});
//	params_Tfin.push_back({"mu",0.,3});
	
	// Hopping
	ArrayXXcd tFull = hopping_PAM_T(L,tfc+0.i,tcc+0.i,tff+0.i,Retx+1.i*Imtx,Rety+1.i*Imty,false); // ANCILLA_HOPPING=false
	params_Tfin.push_back({"tFull",tFull});
	params_Tfin.push_back({"maxPower",2ul}); // hoechste Potenz von H
	
	// Parameter fuer die t-Propagation mit beta: Rueckpropagation der Badplaetze
	vector<Param> pparams;
	pparams.push_back({"Uph",+Uc,0});
	pparams.push_back({"Uph",-Uc,1});
	pparams.push_back({"Uph",+U,2});
	pparams.push_back({"Uph",-U,3});
//	pparams.push_back({"mu",+mu,0});
//	pparams.push_back({"mu",-mu,1});
//	pparams.push_back({"mu",+mu,2});
//	pparams.push_back({"mu",-mu,3});
	ArrayXXcd tFull_ancilla = hopping_PAM_T(L,tfc+0.i,tcc+0.i,tff+0.i,Retx+1.i*Imtx,Rety+1.i*Imty,true,0.); // ANCILLA_HOPPING=true
	pparams.push_back({"tFull",tFull_ancilla});
	pparams.push_back({"maxPower",2ul});
	
	// Aufbau des Modells bei β=0
	MODEL H_Tinf(dLphys*L,Tinf_params_fermions(Ly));
	lout << endl << "β=0 Entangler " << H_Tinf.info() << endl;
	
	// Modell fuer die β-Propagation
	MODELC H_Tfin(dLphys*L,params_Tfin); H_Tfin.precalc_TwoSiteData();
	lout << endl << "physical Hamiltonian " << H_Tfin.info() << endl << endl;
	
	// Modell fuer die t-propagation
	MODELC Hp(dLphys*L,pparams); Hp.precalc_TwoSiteData();
	lout << endl << "propagation Hamiltonian " << Hp.info() << endl << endl;
	
	SpectralManager<MODELC> SpecMan(specs,Hp);
	SpecMan.beta_propagation<MODEL>(H_Tfin, H_Tinf, dLphys, beta, dbeta, tol_compr_beta, Mlim, Q, param_base, LOAD_GS, SAVE_GS, VERB);
	if (CALC_SPEC)
	{
		SpecMan.apply_operators_on_thermal_state(Lcell,dLphys);
		SpecMan.compute_thermal(wd, param_base, dLphys, tmax, dt, wmin, wmax, wpoints, QR, qpoints, INT, Mlim, tol_DeltaS, tol_compr);
	}
}
