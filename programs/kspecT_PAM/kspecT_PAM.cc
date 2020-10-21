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
#include "models/SpectralFunctionHelpers.h"
#include "DmrgLinearAlgebra.h"

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
	size_t L = args.get<size_t>("L",16); // Groesse der Kette
	size_t Lcell = 2;
	int N = args.get<int>("N",L); // Teilchenzahl
	int x0 = L/2;
	qarray<MODELC::Symmetry::Nq> Q = MODELC::singlet(N); // Quantenzahl des Grundzustandes
	lout << "Q=" << Q << endl;
	double U = args.get<double>("U",8.); // U auf den f-Plaetzen
	double tfc = args.get<double>("tfc",1.); // Hybridisierung fc
	double tcc = args.get<double>("tcc",1.); // Hopping fc
	double tff = args.get<double>("tff",0.); // Hopping ff
	double Retx = args.get<double>("Retx",0.); // Re Hybridisierung f(i)c(i+1)
	double Imtx = args.get<double>("Imtx",1.); // Im Hybridisierung f(i)c(i+1)
	double Rety = args.get<double>("Rety",0.); // Re Hybridisierung c(i)f(i+1)
	double Imty = args.get<double>("Imty",0.); // Im Hybridisierung c(i)f(i+1)
	
	bool SAVE_GS = args.get<double>("SAVE_GS",false);
	string LOAD_GS = args.get<string>("LOAD_GS","");
	
	vector<string> specs = args.get_list<string>("specs",{"PES","IPE"}); // welche Spektren? PES:Photoemission, IPE:inv. Photoemission
	string specstring = "";
	int Nspec = specs.size();
	Green.resize(Nspec);
	size_t Dlim = args.get<size_t>("Dlim",100ul);
	double dt = args.get<double>("dt",0.025);
	double tol_DeltaS = args.get<double>("tol_DeltaS",1e-2);
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
	string param_base = make_string("tfc=",tfc,"_tcc=",tcc,"_tff=",tff,"_tx=",Retx,",",Imtx,"_ty=",Rety,",",Imty,"_U=",U); // Dateiname
	string base = make_string("L=",L,"_N=",N,"_") + param_base;
	string tbase = make_string("dt=",dt,"_tolΔS=",tol_DeltaS);
	model_info += make_string();
	model_info += (BETAPROP)? make_string("_βmax=",betamax) : "_βmax=inf";
	
	lout << base << endl;
	lout.set(base+".log",wd+"log"); // Log-Datei im Unterordner log
	
	// Parameter fuer den Grundzustand:
	DMRG::CONTROL::GLOB GlobParams;
	GlobParams.min_halfsweeps = args.get<size_t>("min_halfsweeps",4);
	GlobParams.max_halfsweeps = args.get<size_t>("max_halfsweeps",8);
	GlobParams.Dinit = args.get<size_t>("Dinit",2ul);
	GlobParams.Qinit = args.get<size_t>("Qinit",6ul);
	
	// Parameter des Modells
	vector<Param> params;
	if (Ly==1)
	{
		// Ungerade Plaetze sollen f-Plaetze mit U sein:
		params.push_back({"U",0.,0});
		params.push_back({"U",U,1});
		
		// Hopping
		ArrayXXcd tFull = hopping_PAM_T(L,tfc+0.i,tcc+0.i,tff+0.i,Retx+1.i*Imtx,Rety+1.i*Imty,false); // ANCILLA=false
//		lout << "hopping:" << endl << tFull << endl;
		params.push_back({"tFull",tFull});
		
		params.push_back({"maxPower",2ul}); // hoechste Potenz von H
	}
	
	// Parameter fuer die t-Propagation mit beta: Rueckpropagation der Badplaetze
	vector<Param> pparams;
	pparams.push_back({"U",0.,0});
	pparams.push_back({"U",U,2});
	pparams.push_back({"U",0.,1});
	pparams.push_back({"U",-U,3});
	ArrayXXcd tFull_ancilla = hopping_PAM_T(L,tfc+0.i,tcc+0.i,tff+0.i,Retx+1.i*Imtx,Rety+1.i*Imty,true,0.); // ANCILLA=true
	pparams.push_back({"tFull",tFull_ancilla});
	pparams.push_back({"maxPower",2ul});
	
	// Aufbau des Modells bei β=0
	MODEL H_Tinf(dLphys*L,Tinf_params_fermions(Ly));
	lout << endl << "β=0 Entangler " << H_Tinf.info() << endl;
	
	// Modell fuer die β-Propagation
	MODELC H(dLphys*L,params); H.precalc_TwoSiteData();
	lout << endl << "physical Hamiltonian " << H.info() << endl << endl;
	
	// Modell fuer die t-propagation
	MODELC Hp(dLphys*L,pparams); Hp.precalc_TwoSiteData();
	lout << endl << "propagation Hamiltonian " << Hp.info() << endl << endl;
	
	// DMRG solver
	MODELC::StateXcd PsiT;
	Eigenstate<MODEL::StateXd> g;
	MODEL::Solver DMRG(VERB);
	DMRG.userSetGlobParam();
	DMRG.GlobParam = GlobParams;
	
	// groundstate beta=0 -> g
	DMRG.edgeState(H_Tinf, g, MODEL::singlet(2*N), LANCZOS::EDGE::GROUND, false);
	lout << endl;
	
	// Zero hopping may cause problems. Restart until the correct product state is reached.
	if (Ly==1)
	{
		vector<bool> ENTROPY_CHECK;
		for (int l=1; l<2*L-1; l+=2) ENTROPY_CHECK.push_back(abs(g.state.entropy()(l))<1e-10);
		bool ALL = all_of(ENTROPY_CHECK.begin(), ENTROPY_CHECK.end(), [](const bool v){return v;});
		
		while (ALL == false)
		{
			lout << termcolor::yellow << "restarting..." << termcolor::reset << endl;
			DMRG.edgeState(H_Tinf, g, MODELC::singlet(2*N), LANCZOS::EDGE::GROUND, false);
			ENTROPY_CHECK.clear();
			for (int l=1; l<2*L-1; l+=2) ENTROPY_CHECK.push_back(abs(g.state.entropy()(l))<1e-10);
			for (int l=1; l<2*L-1; l+=2)
			{
				bool TEST = abs(g.state.entropy()(l))<1e-10;
//				lout << "l=" << l << ", S=" << abs(g.state.entropy()(l)) << "\t" << boolalpha << TEST << endl;
			}
			ALL = all_of(ENTROPY_CHECK.begin(), ENTROPY_CHECK.end(), [](const bool v){return v;});
//			lout << boolalpha << "ALL=" << ALL << endl;
		}
	}
	
	PsiT = g.state.cast<complex<double>>();
	PsiT.eps_svd = tol_compr_beta;
	PsiT.min_Nsv = 0ul;
	PsiT.max_Nsv = 100ul;
	TDVPPropagator<MODELC,MODELC::Symmetry,complex<double>,complex<double>,MODELC::StateXcd> TDVPT(H,PsiT);
	
	// or propagation of T=inf solution g to finite beta
	double betaval = 0.;
	ofstream ThermoFiler(make_string("thermodyn_",base,".dat"));
	for (int i=0; i<Nbeta; ++i)
	{
		Stopwatch<> betaStepper;
		TDVPT.t_step(H, PsiT, -0.5*dbeta, 1);
		PsiT /= sqrt(dot(PsiT,PsiT));
		betaval = (i+1)*dbeta;
		lout << TDVPT.info() << endl;
		lout << setprecision(16) << PsiT.info() << setprecision(6) << endl;
		double e = isReal(avg(PsiT,H,PsiT))/L;
		double C = betaval*betaval*isReal(avg(PsiT,H,PsiT,2)-pow(avg(PsiT,H,PsiT),2))/N;
		
		auto PsiTtmp = PsiT; PsiTtmp.entropy_skim();
		lout << "S=" << PsiTtmp.entropy().transpose() << endl;
		
		double Nphys = 0.;
		for (int i=0; i<dLphys*L; i+=dLphys)
		{
			double ni = isReal(avg(PsiT, H.n(i,0), PsiT));
			double di = isReal(avg(PsiT, H.d(i,0), PsiT));
			double ns = isReal(avg(PsiT, H.ns(i,0), PsiT));
			double nh = isReal(avg(PsiT, H.nh(i,0), PsiT));
			cout << "i=" << i << ", n=" << ni << ", d=" << di << ", ns=" << ns << ", nh=" << nh << endl;
			Nphys += ni;
		}
		double Nancl = 0.;
		for (int i=dLphys-1; i<dLphys*L; i+=dLphys)
		{
			Nancl += isReal(avg(PsiT, H.n(i,dLphys%2), PsiT));
		}
		
		lout << "β=" << betaval << ", T=" << 1./betaval << ", e=" << e << ", C=" << C << ", Nphys=" << Nphys << ", Nancl=" << Nancl << endl;
		ThermoFiler << 1./betaval << "\t" << C << "\t" << e << endl;
		lout << betaStepper.info("βstep") << endl;
		lout << endl;
	}
	ThermoFiler.close();
	
	PsiT.entropy_skim();
	lout << "entropy in thermal state: " << PsiT.entropy().transpose() << endl;
	
	// =============== GREENSFUNKTION ===============
	MODELC::StateXcd Phi = PsiT.cast<complex<double>>();
	Phi.eps_svd = tol_compr_beta;
	Phi.min_Nsv = 0ul;
	Phi.max_Nsv = 100ul;
	
	// OxV for time propagation
	vector<vector<Mpo<MODELC::Symmetry,complex<double>>>> O(Nspec);
	vector<vector<Mpo<MODELC::Symmetry,complex<double>>>> Odag(Nspec);
	for (int z=0; z<Nspec; ++z) O[z].resize(L);
	for (int z=0; z<Nspec; ++z) Odag[z].resize(L);
	
	for (int z=0; z<Nspec; ++z)
	for (int l=0; l<L; ++l)
	{
		//O[z][l] = VMPS::get_Op<MODELC,MODELC::Symmetry,complex<double>>(H_hetero,Lhetero/2+l,specs[z]);
		O[z][l] = VMPS::get_Op<MODELC,MODELC::Symmetry,complex<double>>(H,dLphys*l,specs[z]);
		double dagfactor;
		if (specs[z] == "SSF")
		{
			dagfactor = sqrt(3);
		}
		else if (specs[z] == "PES")
		{
			dagfactor = -sqrt(2);
		}
		else if (specs[z] == "IPE")
		{
			dagfactor = +sqrt(2);
		}
		else
		{
			dagfactor = 1.;
		}
		Odag[z][l] = VMPS::get_Op<MODELC,MODELC::Symmetry,complex<double>>(H,dLphys*l,VMPS::DAG(specs[z]),dagfactor);
	}
	
	//---------check---------
//	lout << endl;
//	for (int z=0; z<Nspec; ++z)
//	{
//		lout << "check z=" << z 
//			 << ", spec=" << specs[z] << ", dag=" << VMPS::DAG(specs[z]) 
//			 << ", Phi: " << avg(Phi, Odag[z][L/2], O[z][L/2], Phi) 
//			 << ", g.state: " << avg(g.state, Odag[z][L/2], O[z][L/2], g.state) 
//			 << endl;
//	}
	
	vector<vector<MODELC::StateXcd>> OxPhi0(Nspec);
	for (int z=0; z<Nspec; ++z)
	{
		OxPhi0[z].resize(Lcell);
		for (int i=0; i<Lcell; ++i)
		{
			OxV_exact(O[z][L/2+i], Phi, OxPhi0[z][i], 2., DMRG::VERBOSITY::ON_EXIT);
		}
	}
	
	for (int z=0; z<Nspec; ++z)
	{
		string spec = specs[z];
		Green[z] = GreenPropagator<MODELC,MODELC::Symmetry,complex<double>,complex<double> >
			       (wd+spec+"_"+param_base+"_"+tbase,tmax,Nt,wmin,wmax,wpoints,QR,qpoints,INT);
		Green[z].set_verbosity(DMRG::VERBOSITY::ON_EXIT);
	}
	Green[0].set_verbosity(DMRG::VERBOSITY::HALFSWEEPWISE);
	
	#pragma omp parallel for
	for (int z=0; z<Nspec; ++z)
	{
		string spec = specs[z];
		Green[z].set_lim_Nsv(Dlim);
//		Green[z].set_h_ooura(1e-4);
		Green[z].set_tol_DeltaS(tol_DeltaS);
		Green[z].compute_thermal_cell(Hp, Odag[z], Phi, OxPhi0[z], VMPS::TIME_DIR(spec));
		Green[z].FT_allSites();
		Green[z].save(false); // IGNORE_CELL=false
	}
}
