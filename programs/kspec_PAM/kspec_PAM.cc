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

#include "models/HubbardSU2xU1.h"
typedef VMPS::HubbardSU2xU1 MODEL;
#include "models/PeierlsHubbardSU2xU1.h"
typedef VMPS::PeierlsHubbardSU2xU1 MODELC;

#ifdef TIME_PROP_USE_TERMPLOT
#include "plot.hpp"
#include "TerminalPlot.h"
#endif

#include "solvers/GreenPropagator.h"
#include "models/SpectralFunctionHelpers.h"
#include "DmrgLinearAlgebra.h"
#include "models/ParamCollection.h"

vector<GreenPropagator<MODELC,MODELC::Symmetry,complex<double>,complex<double>>> Green;

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
	int Lphys = (Ly==1)? 2:1;
	assert(Ly==1 and "Only Ly=1 implemented so far");
	size_t L = args.get<size_t>("L",2); // Groesse der Einheitszelle
	int N = args.get<int>("N",L); // Teilchenzahl
	int Ncells = args.get<int>("Ncells",16); // Anzahl der Einheitszellen fuer Spektralfunktion
	int Lhetero = L*Ncells;
	int x0 = Lhetero/2;
	qarray<MODEL::Symmetry::Nq> Q = MODEL::singlet(N); // Quantenzahl des Grundzustandes
	lout << "Q=" << Q << endl;
	double U = args.get<double>("U",4.); // U auf den f-Plaetzen
	double V = args.get<double>("V",0.); // V*nc*nf
	double tfc = args.get<double>("tfc",0.5); // Hybridisierung fc
	double tcc = args.get<double>("tcc",1.); // Hopping fc
	double tff = args.get<double>("tff",0.); // Hopping ff
	double Retx = args.get<double>("Retx",0.); // Re Hybridisierung f(i)c(i+1)
	double Imtx = args.get<double>("Imtx",0.5); // Im Hybridisierung f(i)c(i+1)
	double Rety = args.get<double>("Rety",0.); // Re Hybridisierung c(i)f(i+1)
	double Imty = args.get<double>("Imty",0.); // Im Hybridisierung c(i)f(i+1)
	double Ec = args.get<double>("Ec",0.); // onsite-Energie fuer c
	double Ef = args.get<double>("Ef",-2.); // onsite-Energie fuer f
	
	bool SAVE_GS = args.get<bool>("SAVE_GS",false);
	bool LOAD_GS = args.get<bool>("LOAD_GS",false);
	
	vector<string> specs = args.get_list<string>("specs",{"HSF","CSF","PES","IPE"}); // welche Spektren? PES:Photoemission, IPE:inv. Photoemission, HSF: Hybridisierung, IHSF: inverse Hybridisierung
	string specstring = "";
	int Nspec = specs.size();
	size_t Mlim = args.get<size_t>("Mlim",500ul); // Bonddimension fuer Dynamik
	double dt = args.get<double>("dt",(L==2)?0.2:0.1);
	double tol_DeltaS = args.get<double>("tol_DeltaS",1e-2);
	double tmax = args.get<double>("tmax",4.);
	double tol_compr = args.get<double>("tol_compr",1e-4);
	int Nt = static_cast<int>(tmax/dt);
	
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
	VUMPS::CONTROL::GLOB GlobParams;
	GlobParams.min_iterations = args.get<size_t>("min_iterations",100ul);
	GlobParams.max_iterations = args.get<size_t>("max_iterations",150ul);
	GlobParams.Minit = args.get<size_t>("Minit",10ul);
	GlobParams.Mlimit = args.get<size_t>("Mlimit",500ul);
	GlobParams.Qinit = args.get<size_t>("Qinit",10ul);
	GlobParams.tol_eigval = args.get<double>("tol_eigval",1e-5);
	GlobParams.tol_var = args.get<double>("tol_var",1e-5);
	GlobParams.tol_state = args.get<double>("tol_state",1e-4);
	GlobParams.max_iter_without_expansion = 20ul;
	
	// Gemeinsame Parameter bei unendlichen und offenen Randbedingungen
	vector<Param> params_common;
	
	// Parameter des Modells
	vector<Param> params;
	if (Ly==1)
	{
		// Ungerade Plaetze sollen f-Plaetze mit U sein:
		params_common.push_back({"U",0.,0});
		params_common.push_back({"U",U,1});
		
//		ArrayXXd Vmatrix = hopping_PAM(L,1.+0.i,0,0,0,0).real(); // 
		params_common.push_back({"V",V,0});
		params_common.push_back({"V",0.,1});
		
		params_common.push_back({"t0",Ec,0});
		params_common.push_back({"t0",Ef,1});
		
		params_common.push_back({"maxPower",1ul}); // hoechste Potenz von H
		
		params = params_common;
		
		// Hopping
		ArrayXXcd t2cell = hopping_PAM(L,tfc+0.i,tcc+0.i,tff+0.i,Retx+1.i*Imtx,Rety+1.i*Imty);
		lout << "hopping:" << endl << t2cell << endl;
		params.push_back({"tFull",t2cell});
	}
	
	// Aufbau des Modells
	MODELC H(L,params,BC::INFINITE);
	H.transform_base(Q,false); // PRINT=false
	H.precalc_TwoSiteData(true); // FORCE=true
	lout << H.info() << endl;
	
//	// Test vom komplexen Hopping
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
	
	// Grundzustand fuer unendliches System
	Eigenstate<MODELC::StateUcd> g;
	
	int Lfinite = args.get<int>("Lfinite",1000);
	auto Hfree = hopping_PAM(Lfinite/2,tfc+0.i,tcc+0.i,tff+0.i,Retx+1.i*Imtx,Rety+1.i*Imty);
	SelfAdjointEigenSolver<MatrixXcd> Eugen(Hfree.matrix()+onsite(Lfinite,Ec,Ef));
	VectorXd occ = Eugen.eigenvalues().head(Lfinite/2);
	VectorXd unocc = Eugen.eigenvalues().tail(Lfinite/2);
	double e0free = 2.*occ.sum()/Lfinite;
	lout << setprecision(16) << "e0free/(L="<<Lfinite<<",half-filling)=" << e0free << endl;
	
	// VUMPS-Solver
	MODELC::uSolver uDMRG(VERB);
	if (LOAD_GS)
	{
		g.state.load("gs_"+base,g.energy);
		lout << "loaded: " << g.state.info() << endl;
	}
	else
	{
		uDMRG.userSetGlobParam();
		uDMRG.GlobParam = GlobParams;
		uDMRG.edgeState(H, g, Q, LANCZOS::EDGE::GROUND);
		if (SAVE_GS)
		{
			lout << "saving groundstate..." << endl;
			g.state.save("gs_"+base, "PAM groundstate", g.energy);
		}
	}
	
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
			lout << "l=" << l << "," << l+1 << ": SdagS=" << isReal(avg(g.state, H.SdagS(l,l+1), g.state)) << endl;
		}
	}
	
	// =============== GREENSFUNKTION ===============
	vector<Param> params_hetero = params_common;
	if (Ly==1)
	{
		// Hopping
		ArrayXXcd tLhetero = hopping_PAM(2*Lhetero,tfc+0.i,tcc+0.i,tff+0.i,Retx+1.i*Imtx,Rety+1.i*Imty);
		params_hetero.push_back({"tFull",tLhetero});
	}
	MODELC H_hetero(Lhetero,params_hetero,BC::INFINITE);
	lout << H_hetero.info() << endl;
	H_hetero.transform_base(Q,false,L); // PRINT=false
	H_hetero.precalc_TwoSiteData(true); // FORCE=true
	
	lout << endl;
	
	// shift values O→O-<O>
	vector<vector<double>> Oshift(Nspec);
	for (int z=0; z<Nspec; ++z)
	{
		Oshift[z].resize(Lphys);
		for (int l=0; l<Lphys; ++l)
		{
			complex<double> val;
			lout << "l=" << l << endl;
			if (specs[z] == "CSF")
			{
				val = avg(g.state, H.n(l), g.state);
				lout << "n(l)=" << val << endl;
				Oshift[z][l] = val.real();
			}
			else if (specs[z] == "HSF")
			{
				MODELC Haux(2*L, {{"maxPower",1ul}}, BC::INFINITE, DMRG::VERBOSITY::SILENT);
				val = avg(g.state, Haux.cdagc(l,l+1), g.state);
				lout << "cdagc(l,l+1)=" << val << endl;
				Oshift[z][l] = val.real();
			}
			else
			{
				Oshift[z][l] = 0.;
			}
			if (abs(val.imag()) > 1e-10)
			{
				lout << termcolor::red << "Warning: non-zero imaginary part!" << termcolor::reset << endl;
			}
			lout << "spec=" << specs[z] << ", l=" << l << ", Oshift=" << Oshift[z][l] << endl;
		}
	}
	
	// Phi
	Stopwatch<> OxVTimer;
	Mps<MODELC::Symmetry,complex<double>> Phi = uDMRG.create_Mps(Ncells, g, H, x0); // ground state as heterogenic MPS
	
	// O and OxPhiCell for counterpropagate
	vector<vector<Mpo<MODELC::Symmetry,complex<double>>>> O(Nspec);
	vector<vector<Mpo<MODELC::Symmetry,complex<double>>>> Odag(Nspec);
	for (int z=0; z<Nspec; ++z)
	{
		O[z].resize(Lphys);
		for (int l=0; l<Lphys; ++l)
		{
			O[z][l] = VMPS::get_Op<MODELC,MODELC::Symmetry,complex<double>>(H_hetero, Lhetero/2+l, specs[z]);
			if (specs[z] != "PES" and specs[z] != "IPE" and specs[z] != "PSZ")
			{
				O[z][l].scale(1.,-Oshift[z][l]);
			}
			O[z][l].transform_base(Q,false,L); // PRINT=false
			cout << O[z][l].info() << endl;
			cout << "avg(g.state, O[z][l], g.state)=" << avg(g.state, O[z][l], g.state) << endl;
			// l=0: c-electrons
			// l=1: f-electrons
		}
	}
	
	vector<vector<Mps<MODELC::Symmetry,complex<double>>>> OxPhiCell(Nspec);
	for (int z=0; z<Nspec; ++z)
	{
		OxPhiCell[z].resize(Lphys);
		OxPhiCell[z] = uDMRG.create_Mps(Ncells, g, H, O[z][0], O[z]); // O[z][0] for boundaries, O[z] is multiplied (vector in cell size)
	}
	
	// Ofull and OxPhiFull for simple propagate
	vector<vector<Mpo<MODEL::Symmetry,complex<double>>>> Ofull(Nspec);
	vector<vector<Mps<MODELC::Symmetry,complex<double>>>> OxPhiFull(Nspec);
	if (L>2)
	{
		// Ofull
		for (int z=0; z<Nspec; ++z)
		{
			Ofull[z].resize(Lhetero);
			for (int l=0; l<Lhetero; ++l)
			{
				Ofull[z][l] = VMPS::get_Op<MODELC,MODELC::Symmetry,complex<double>>(H_hetero,l,specs[z]);
				if (specs[z] != "PES" and specs[z] != "IPE" and specs[z] != "PSZ")
				{
					Ofull[z][l].scale(1.,-Oshift[z][l]);
				}
				Ofull[z][l].transform_base(Q,false,L); // PRINT=false
			}
		}
		
		// OxPhiFull
		for (int z=0; z<Nspec; ++z)
		{
			OxPhiFull[z].resize(Lhetero);
			OxPhiFull[z] = uDMRG.create_Mps(Ncells, g, H, Ofull[z][Lhetero/2], Ofull[z]); // Ofull[z][Lhetero/2] for boundaries, O[z] is multiplied
		}
	}
	
	// GreenPropagator
	Green.resize(Nspec);
	for (int z=0; z<Nspec; ++z)
	{
		string spec = specs[z];
		Green[z] = GreenPropagator<MODELC,MODELC::Symmetry,complex<double>,complex<double> >
		           (wd+spec+"_"+param_base,tmax,Nt,wmin,wmax,wpoints,QR,qpoints,INT);
		Green[z].set_verbosity(DMRG::VERBOSITY::ON_EXIT);
	}
	Green[0].set_verbosity(DMRG::VERBOSITY::STEPWISE);
	
	// Energie des heterogenen Teils
	double Eg = isReal(avg_hetero(Phi, H_hetero, Phi, true)); // USE_BOUNDARY=true
	lout << setprecision(14) << "Eg=" << Eg << ", eg=" << g.energy << ", egfree=" << 2.*occ.sum()/Lfinite << endl;
	
	// Propagation
	#pragma omp parallel for
	for (int z=0; z<Nspec; ++z)
	{
		string spec = specs[z];
		Green[z].set_tol_DeltaS(tol_DeltaS);
		Green[z].set_lim_Nsv(Mlim);
		if (L>2)
		{
			Green[z].set_OxPhiFull(OxPhiFull[z]);
			Green[z].compute_cell(H_hetero, OxPhiCell[z], Eg, VMPS::TIME_DIR(spec), false); // COUNTERPROPAGATE=false
		}
		else
		{
			Green[z].compute_cell(H_hetero, OxPhiCell[z], Eg, VMPS::TIME_DIR(spec), true); // COUNTERPROPAGATE=true
		}
//		Green[z].FT_allSites();
		Green[z].save(false); // IGNORE_CELL=false
	}
}
