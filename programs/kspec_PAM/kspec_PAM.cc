#ifdef BLAS
#include "util/LapackManager.h"
#pragma message("LapackManager")
#endif

//#define USE_OLD_COMPRESSION
#define USE_HDF5_STORAGE
#define DMRG_DONT_USE_OPENMP
//#define VUMPS_SOLVER_DONT_USE_OPENMP
#define GREENPROPAGATOR_USE_HDF5
//#define LINEARSOLVER_DIMK 100
//#define TIME_PROP_USE_TERMPLOT

#include <iostream>
#include <fstream>
#include <complex>
#include <iterator>

#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

#include "Logger.h"
Logger lout;
#include "ArgParser.h"

#include "StringStuff.h"
#include "Stopwatch.h"

// Test von komplexem Hopping
//#include "solvers/DmrgSolver.h"
//#include "IntervalIterator.h"

#include "VUMPS/VumpsSolver.h"
#include "VUMPS/VumpsLinearAlgebra.h"

//#include "models/HubbardSU2xU1.h"
//typedef VMPS::HubbardSU2xU1 MODEL; // reell

#include "models/PeierlsHubbardSU2xU1.h"
typedef VMPS::PeierlsHubbardSU2xU1 MODEL; // complex

#ifdef TIME_PROP_USE_TERMPLOT
#include "plot.hpp"
#include "TerminalPlot.h"
#endif

#include "solvers/SpectralManager.h"
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

double n = 1.;
GreenPropagator<MODEL,MODEL::Symmetry,complex<double>,complex<double>> Gfull;
static double integrand (double mu, void*)
{
	return MODEL::spinfac * Gfull.integrate_Glocw_cell(mu) - n;
}

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
	assert(Ly==1 and "Only Ly=1 is implemented");
	size_t L = args.get<size_t>("L",2); // Groesse der Einheitszelle
	int Ns = args.get<int>("Ns",L/2);
	int N = args.get<int>("N",L); // Teilchenzahl
	n = double(N)/L;
	int Ncells = args.get<int>("Ncells",16); // Anzahl der Einheitszellen fuer Spektralfunktion
	int Lhetero = L*Ncells;
	lout << "L=" << L << ", N=" << N << ", Ly=" << Ly << ", Ncells=" << Ncells << ", Lhetero=" << Lhetero << ", Ns=" << Ns << endl;
	
	qarray<MODEL::Symmetry::Nq> Q = MODEL::singlet(N); // Quantenzahl des Grundzustandes
	lout << "Q=" << Q << endl;
	double U = args.get<double>("U",4.); // U auf den f-Plaetzen
	double V = args.get<double>("V",0.); // V*nc*nf
	double tfc = args.get<double>("tfc",1.); // Hybridisierung fc
	double tcc = args.get<double>("tcc",1.); // Hopping fc
	double tff = args.get<double>("tff",0.); // Hopping ff
	double Retx = args.get<double>("Retx",0.); // Re Hybridisierung f(i)c(i+1)
	double Imtx = args.get<double>("Imtx",0.); // Im Hybridisierung f(i)c(i+1)
	double Rety = args.get<double>("Rety",0.); // Re Hybridisierung c(i)f(i+1)
	double Imty = args.get<double>("Imty",0.); // Im Hybridisierung c(i)f(i+1)
	double Ec = args.get<double>("Ec",0.); // onsite-Energie fuer c
	double Ef = args.get<double>("Ef",-2.); // onsite-Energie fuer f
	
	bool SAVE_GS = args.get<bool>("SAVE_GS",false);
	bool LOAD_GS = args.get<bool>("LOAD_GS",false);
	bool RELOAD = args.get<bool>("RELOAD",false);
	
	vector<string> specs = args.get_list<string>("specs",{"HSF","CSF","PES","IPE"}); // welche Spektren? PES:Photoemission, IPE:inv. Photoemission, HSF: Hybridisierung, IHSF: inverse Hybridisierung
	string specstring = "";
	int Nspec = specs.size();
	size_t Mlim = args.get<size_t>("Mlim",800ul); // Bonddimension fuer Dynamik
	double dt = args.get<double>("dt",0.2);
	double tol_DeltaS = args.get<double>("tol_DeltaS",1e-2);
	double tmax = args.get<double>("tmax",4.);
	double tol_compr = args.get<double>("tol_compr",1e-4);
	
	GREEN_INTEGRATION INT = static_cast<GREEN_INTEGRATION>(args.get<int>("INT",2)); // DIRECT=0, INTERP=1, OOURA=2
	Q_RANGE QR = static_cast<Q_RANGE>(args.get<int>("QR",1)); // MPI_PPI=0, ZERO_2PI=1
	
	double wmin = args.get<double>("wmin",-10.);
	double wmax = args.get<double>("wmax",+10.);
	int wpoints = args.get<int>("wpoints",501);
	int qpoints = args.get<int>("qpoints",501);
	
	// Steuert die Menge der Ausgaben
	DMRG::VERBOSITY::OPTION VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",DMRG::VERBOSITY::HALFSWEEPWISE));
	
	string wd = args.get<string>("wd","./"); correct_foldername(wd); // Arbeitsvereichnis
	string param_base = make_string("tfc=",tfc,"_tcc=",tcc,"_tff=",tff,"_tx=",Retx,",",Imtx,"_ty=",Rety,",",Imty,"_Efc=",Ef,",",Ec,"_U=",U,"_V=",V); // Dateiname
	string base = make_string("L=",L,"_N=",N,"_",param_base); // Dateiname
	lout << base << endl;
	lout.set(base+".log",wd+"log"); // Log-Datei im Unterordner log
	
	// Parameter fuer den Grundzustand:
	VUMPS::CONTROL::GLOB GlobSweepParams;
	GlobSweepParams.min_iterations = args.get<size_t>("min_iterations",100ul);
	GlobSweepParams.max_iterations = args.get<size_t>("max_iterations",150ul);
	GlobSweepParams.Minit = args.get<size_t>("Minit",1ul);
	GlobSweepParams.Mlimit = args.get<size_t>("Mlimit",800ul);
	GlobSweepParams.Qinit = args.get<size_t>("Qinit",1ul);
	GlobSweepParams.tol_eigval = args.get<double>("tol_eigval",1e-5);
	GlobSweepParams.tol_var = args.get<double>("tol_var",1e-5);
	GlobSweepParams.tol_state = args.get<double>("tol_state",1e-4);
	GlobSweepParams.max_iter_without_expansion = 20ul;
	GlobSweepParams.CALC_S_ON_EXIT = false;
	
	// Gemeinsame Parameter bei unendlichen und offenen Randbedingungen
	vector<Param> params_common, params, params_hetero;
	
	if (Ly==1)
	{
		// Ungerade Plaetze sollen f-Plaetze mit U sein:
		params_common.push_back({"U",0.,0});
		params_common.push_back({"U",U,1});
		
		params_common.push_back({"V",V,0});
		params_common.push_back({"V",0.,1});
		
		params_common.push_back({"t0",Ec,0});
		params_common.push_back({"t0",Ef,1});
		
		params_common.push_back({"maxPower",1ul}); // hoechste Potenz von H
		
		// params
		params = params_common;
		// Hopping
		ArrayXXcd t2cell = hopping_PAM(L,tfc+0.i,tcc+0.i,tff+0.i,Retx+1.i*Imtx,Rety+1.i*Imty);
//		ArrayXXd t2cell = hopping_PAM(L,tfc+0.i,tcc+0.i,tff+0.i,Retx+1.i*Imtx,Rety+1.i*Imty).real(); // reell
		lout << "hopping:" << endl << t2cell << endl;
		params.push_back({"tFull",t2cell});
		
		// params_hetero
		params_hetero = params_common;
		// Hopping
		ArrayXXcd tLhetero = hopping_PAM(2*Lhetero,tfc+0.i,tcc+0.i,tff+0.i,Retx+1.i*Imtx,Rety+1.i*Imty);
//		ArrayXXd tLhetero = hopping_PAM(2*Lhetero,tfc+0.i,tcc+0.i,tff+0.i,Retx+1.i*Imtx,Rety+1.i*Imty).real(); // reell
		params_hetero.push_back({"tFull",tLhetero});
	}
	
	// Aufbau des Modells
	MODEL H(L,params,BC::INFINITE);
	H.transform_base(Q,false); // PRINT=false
	lout << H.info() << endl;
	
//	//-----<Test von komplexem Hopping>-----
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
//			MODEL Hc(8,{{"tFull",tcomplex},{"U",Uval}});
//			lout << Hc.info() << endl;
//			Eigenstate<MODEL::StateXcd> gc;
//			
//			DmrgSolver<MODEL::Symmetry,MODEL,complex<double>> DMRGc(VERB);
//			DMRGc.edgeState(Hc, gc, {1,4}, LANCZOS::EDGE::GROUND);
//			phi << gc.energy;
//		}
//		phi.save(make_string("E(Φ)_U=",Uval,".dat"));
//	}
//	//-----</Test von komplexem Hopping>-----
	
	int Lfinite = args.get<int>("Lfinite",1000);
	auto Hfree = hopping_PAM(Lfinite/2,tfc+0.i,tcc+0.i,tff+0.i,Retx+1.i*Imtx,Rety+1.i*Imty);
	SelfAdjointEigenSolver<MatrixXcd> Eugen(Hfree.matrix()+onsite(Lfinite,Ec,Ef));
	cout << "N*Lfinite/(L*2)=" << N*Lfinite/(L*2) << endl;
	VectorXd occ = Eugen.eigenvalues().head(N*Lfinite/(L*2));
	VectorXd unocc = Eugen.eigenvalues().tail(N*Lfinite/(L*2));
	double e0free = 2.*occ.sum()/Lfinite;
	lout << setprecision(16) << "e0free/L=("<<Lfinite<<",half-filling)=" << e0free << endl;
	
	SpectralManager<MODEL> SpecMan;
	if (!RELOAD)
	{
		SpecMan = SpectralManager<MODEL>(specs, H, params, GlobSweepParams, Q, Ncells, params_hetero, "gs_"+base, LOAD_GS, SAVE_GS);
		
		if (U==0. and V==0.)
		{
			lout << "hopping matrix diagonalization: " << e0free << ", VUMPS (should be slightly lower): " << SpecMan.energy() << endl;
		}
		
//		lout << "cdagc3=" << isReal(avg(SpecMan.ground(), Haux.cdagc3(0,1), Haux.cdagc3(0,1), SpecMan.ground())) << endl;
		
		if (specs.size() != 0)
		{
			auto itSSF = find(specs.begin(), specs.end(), "SSF");
			if (itSSF != specs.end())
			{
				int iz = distance(specs.begin(), itSSF);
				SpecMan.resize_Green(wd, param_base, Ns, tmax, dt, wmin, wmax, wpoints, QR, qpoints, INT);
				SpecMan.set_measurement(iz, "SSF",1.,1, Q, L, 1,"S","wavepacket",true); // TRANSFORM=true
			}
			SpecMan.compute(wd, param_base, Ns, tmax, dt, wmin, wmax, wpoints, QR, qpoints, INT, Mlim, tol_DeltaS, tol_compr);
		}
	}
	else
	{
		SpecMan.reload(wd, specs, param_base, L, Ncells, Ns, tmax, wmin, wmax, wpoints, QR, qpoints, INT);
	}
	
	// μ berechnen
	auto itPES = find(specs.begin(), specs.end(), "PES");
	auto itIPE = find(specs.begin(), specs.end(), "IPE");
	if (itPES != specs.end() and itIPE != specs.end())
	{
		SpecMan.make_A1P(Gfull, wd, param_base, Ns, tmax, -20., +20., 1001, QR, qpoints, INT, true);
		RootFinder R(integrand,wmin,wmax);
		lout << "μ=" << R.root() << endl;
		Gfull.mu = R.root();
		Gfull.save(true); // IGNORE_TX=true
	}
}
