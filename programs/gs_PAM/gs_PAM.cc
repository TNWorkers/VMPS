#ifdef BLAS
#include "util/LapackManager.h"
#pragma message("LapackManager")
#endif

#define USE_HDF5_STORAGE
#define DMRG_DONT_USE_OPENMP

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

template<typename Scalar>
Array<Scalar,Dynamic,Dynamic> create_hopping (int L, Scalar tfc, Scalar tcc, Scalar tff, Scalar tx, Scalar ty)
{
	Array<Scalar,Dynamic,Dynamic> t1site(2,2); t1site = 0;
	t1site(0,1) = tfc;
	
	// L: Anzahl der physikalischen fc-Sites
	Array<Scalar,Dynamic,Dynamic> res(2*L,2*L); res = 0;
	
	for (int l=0; l<L; ++l)
	{
		res.block(2*l,2*l, 2,2) = t1site;
	}
	
	for (int l=0; l<L-1; ++l)
	{
		res(2*l,   2*l+2) = tcc;
		res(2*l+1, 2*l+3) = tff;
		res(2*l+1, 2*l+2) = tx;
		res(2*l,   2*l+3) = ty;
	}
	
	res.matrix() += res.matrix().adjoint().eval();
	
	return res;
}

int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	
	size_t Ly = args.get<size_t>("Ly",1); // Ly=1: entpackt, Ly=2: Supersites
	assert(Ly==1 and "Only Ly=1 implemented");
	size_t L = args.get<size_t>("L",4); // Groesse der Einheitszelle
	int N = args.get<int>("N",L); // Teilchenzahl
	qarray<MODEL::Symmetry::Nq> Q = MODEL::singlet(N); // Quantenzahl des Grundzustandes
	lout << "Q=" << Q << endl;
	double U = args.get<double>("U",8.); // U auf den f-Plaetzen
	double tfc = args.get<double>("tfc",1.); // Hybridisierung fc
	double tcc = args.get<double>("tcc",1.); // Hopping fc
	double tff = args.get<double>("tff",0.); // Hopping ff
	double Retx = args.get<double>("Retx",0.); // Re Hybridisierung f(i)c(i+1)
	double Imtx = args.get<double>("Imtx",1.); // Im Hybridisierung f(i)c(i+1)
	double Rety = args.get<double>("Rety",0.); // Re Hybridisierung c(i)f(i+1)
	double Imty = args.get<double>("Imty",0.); // Im Hybridisierung c(i)f(i+1)
	
	// Steuert die Menge der Ausgaben
	DMRG::VERBOSITY::OPTION VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",DMRG::VERBOSITY::HALFSWEEPWISE));
	
	string wd = args.get<string>("wd","./"); correct_foldername(wd); // Arbeitsvereichnis
	string base = make_string("L=",L,"_N=",N,"_U=",U,"_tfc=",tfc,"_tcc=",tcc,"_tff=",tff,"_tx=",Retx,Imtx,"_ty=",Rety,Imty); // Dateiname
	lout << base << endl;
	lout.set(base+".log",wd+"log"); // Log-Datei im Unterordner log
	
	// Parameter fuer den Grundzustand:
	VUMPS::CONTROL::GLOB GlobParams;
	GlobParams.min_iterations = args.get<size_t>("min_iterations",50ul);
	GlobParams.max_iterations = args.get<size_t>("max_iterations",150ul);
	GlobParams.Dinit = args.get<size_t>("Dinit",30ul);
	GlobParams.Dlimit = GlobParams.Dinit;
	GlobParams.Qinit = args.get<size_t>("Qinit",6ul);
	GlobParams.tol_eigval = args.get<double>("tol_eigval",1e-5);
	GlobParams.tol_var = args.get<double>("tol_var",1e-5);
	GlobParams.tol_state = args.get<double>("tol_state",1e-4);
	GlobParams.max_iter_without_expansion = 30ul;
	
	// Parameter des Modells
	vector<Param> params;
	if (Ly==1)
	{
		// Ungerade Plaetze sollen f-Plaetze mit U sein:
		params.push_back({"U",0.,0});
		params.push_back({"U",U,1});
		
		// Hopping
		ArrayXXcd t2cell = create_hopping(L,tfc+0.i,tcc+0.i,tff+0.i,Retx+1.i*Imtx,Rety+1.i*Imty);
		lout << "hopping:" << endl << t2cell << endl;
		
		params.push_back({"tFull",t2cell});
		params.push_back({"maxPower",1ul}); // hoechste Potenz von H
	}
	
	// Aufbau des Modells
	MODELC H(L,params,BC::INFINITE);
	H.transform_base(Q,false); // PRINT=false
	H.precalc_TwoSiteData(true); // FORCE=true
	lout << H.info() << endl;
	
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
	
	// Grundzustand fuer unendliches System
	Eigenstate<MODELC::StateUcd> g;
	
	int Lfinite = args.get<int>("Lfinite",200);
	auto Hfree = create_hopping(Lfinite/2,tfc+0.i,tcc+0.i,tff+0.i,Retx+1.i*Imtx,Rety+1.i*Imty);
	SelfAdjointEigenSolver<MatrixXcd> Eugen(Hfree.matrix());
	VectorXd occ = Eugen.eigenvalues().head(Lfinite/2);
	VectorXd unocc = Eugen.eigenvalues().tail(Lfinite/2);
	double e0free = 2.*occ.sum()/Lfinite;
	lout << setprecision(16) << "e0free/(L="<<Lfinite<<",half-filling)=" << e0free << endl;
	
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
