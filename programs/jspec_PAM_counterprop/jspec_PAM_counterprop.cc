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
#include "IntervalIterator.h"

#include "LanczosWrappers.h"
#include "HubbardModel.h"

#include "solvers/DmrgSolver.h"
#include "solvers/MpsCompressor.h"
#include "solvers/EntropyObserver.h"
#include "solvers/TDVPPropagator.h"
#include "DmrgLinearAlgebra.h"

#include "models/HubbardSU2xU1.h"
//typedef VMPS::HubbardSU2xU1 MODEL; // reell

#include "models/PeierlsHubbardSU2xU1.h"
typedef VMPS::PeierlsHubbardSU2xU1 MODEL; // complex

#ifdef TIME_PROP_USE_TERMPLOT
#include "plot.hpp"
#include "TerminalPlot.h"
#endif

#include "models/ParamCollection.h"

#include <boost/math/quadrature/ooura_fourier_integrals.hpp>
#include "InterpolGSL.h"

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

complex<double> calc_Joverlap (const vector<MODEL::StateXcd> &Psi, const complex<double> &phase)
{
	int N = Psi.size()/2;
	MatrixXcd res(N,N);
	
	#pragma omp parallel for collapse(2)
	for (int i=0; i<N; ++i)
	for (int j=0; j<N; ++j)
	{
		res(i,j) = dot(Psi[i+N],Psi[j]);
	}
	
	return phase*res.sum();
}

enum DAMPING {GAUSS, LORENTZ, NODAMPING};

void FT_and_save (const VectorXd &tvals, double tmax, const VectorXcd &data, double wmin, double wmax, int wpoints, string filename, DAMPING DAMPING_input)
{
	boost::math::quadrature::ooura_fourier_sin<double> OouraSin = boost::math::quadrature::ooura_fourier_sin<double>();
	boost::math::quadrature::ooura_fourier_cos<double> OouraCos = boost::math::quadrature::ooura_fourier_cos<double>();
	Interpol<GSL> InterpRe(tvals);
	Interpol<GSL> InterpIm(tvals);
	for (int it=0; it<tvals.rows(); ++it)
	{
		InterpRe.insert(it,data(it).real());
		InterpIm.insert(it,data(it).imag());
	}
	InterpRe.set_splines();
	InterpIm.set_splines();
	
	auto fRe = [&InterpRe, &tmax, &DAMPING_input](double t)
	{
		if (t>tmax) return 0.;
		else
		{
			if (DAMPING_input == GAUSS)        return InterpRe(t)*exp(-pow(2.*t/tmax,2));
			else if (DAMPING_input == LORENTZ) return InterpRe(t)*exp(-4.*t/tmax);
			else                               return InterpRe(t);
		}
	};
	auto fIm = [&InterpIm, &tmax, &DAMPING_input](double t)
	{
		if (t>tmax) return 0.;
		else
		{
			if (DAMPING_input == GAUSS)        return InterpIm(t)*exp(-pow(2.*t/tmax,2));
			else if (DAMPING_input == LORENTZ) return InterpIm(t)*exp(-4.*t/tmax);
			else                               return InterpIm(t);
		}
	};
	
	double resReSin, resReCos, resImSin, resImCos;
	IntervalIterator w(wmin,wmax,wpoints);
	for (w=w.begin(2); w!=w.end(); ++w)
	{
		double wval = *w;
		complex<double> dataw;
		if (wval == 0.)
		{
			dataw = InterpIm.integrate() + 1.i * InterpIm.integrate();
		}
		else
		{
			resReSin = OouraSin.integrate(fRe,wval).first;
			resReCos = OouraCos.integrate(fRe,wval).first;
			resImSin = OouraSin.integrate(fIm,wval).first;
			resImCos = OouraCos.integrate(fIm,wval).first;
			dataw = resReCos-resImSin + 1.i*(resReSin+resImCos);
		}
		w << dataw;
	}
	w.save(filename);
	InterpRe.kill_splines();
	InterpIm.kill_splines();
}

void push_term (int i, int j, int ilast, complex<double> lambda, double tol_OxV, DMRG::VERBOSITY::OPTION CVERB, const MODEL &H, 
                const MODEL::StateXcd &target, vector<MODEL::StateXcd> &states, vector<complex<double>> &factors)
{
	if (i>=0 and i<=ilast and j>=0 and j<=ilast)
	{
		MODEL::StateXcd OxVres;
		OxV_exact(H.cdagc(i,j), target, OxVres, tol_OxV, CVERB);
		states.push_back(OxVres);
		factors.push_back(lambda);
	}
}

void push_corrhop (int i, int j, int ilast, complex<double> lambda, double tol_OxV, DMRG::VERBOSITY::OPTION CVERB, const MODEL &H, 
                   const MODEL::StateXcd &target, vector<MODEL::StateXcd> &states, vector<complex<double>> &factors, bool DAG=false)
{
	if (i>=0 and i<=ilast and j>=0 and j<=ilast)
	{
		MODEL::StateXcd OxVres;
		if (DAG)
		{
			OxV_exact(H.cdag_nc(i,j), target, OxVres, tol_OxV, CVERB);
		}
		else
		{
			OxV_exact(H.cdagn_c(i,j), target, OxVres, tol_OxV, CVERB);
		}
		states.push_back(OxVres);
		factors.push_back(lambda);
	}
}

/////////////////////////////////
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
	size_t L = args.get<size_t>("L",24); // Groesse der Kette
	int N = args.get<int>("N",L); // Teilchenzahl
	lout << "L=" << L << ", N=" << N << ", Ly=" << Ly << endl;
	
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
	
	string spec = args.get<string>("spec","JJC"); // JJC, JJE
	size_t Mstart = args.get<size_t>("Mstart",200ul); // anfaengliche Bonddimension fuer Dynamik
	size_t Mlimit = args.get<size_t>("Mlimit",800ul); // max. Bonddimension fuer Dynamik
	double dt = args.get<double>("dt",0.2);
	double tol_DeltaS = args.get<double>("tol_DeltaS",1e-2);
	double tmax = args.get<double>("tmax",4.);
	int Nt = tmax/dt+1;
	double tol_compr = args.get<double>("tol_compr",1e-4);
	
	double wmin = args.get<double>("wmin",-10.);
	double wmax = args.get<double>("wmax",+10.);
	int wpoints = args.get<int>("wpoints",501);
	int qpoints = args.get<int>("qpoints",501);
	
	// Steuert die Menge der Ausgaben
	DMRG::VERBOSITY::OPTION VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",DMRG::VERBOSITY::HALFSWEEPWISE));
	
	string wd = args.get<string>("wd","./"); correct_foldername(wd); // Arbeitsvereichnis
	string param_base = make_string("tfc=",tfc,"_tcc=",tcc,"_tff=",tff,"_tx=",Retx,",",Imtx,"_ty=",Rety,",",Imty,"_Efc=",Ef,",",Ec,"_U=",U,"_V=",V); // Dateiname
	string base = make_string("L=",L,"_N=",N,"_",param_base); // Dateiname
	string tbase = make_string("tmax=",tmax,"_dt=",dt);
	string wbase = make_string("wmin=",wmin,"_wmax=",wmax,"_wpoints=",wpoints);
	lout << base << endl;
	lout.set(base+".log",wd+"log"); // Log-Datei im Unterordner log
	
	// Parameter fuer den Grundzustand:
	DMRG::CONTROL::GLOB GlobSweepParamsOBC;
	GlobSweepParamsOBC.min_halfsweeps = args.get<size_t>("min_halfsweeps",10ul);
	GlobSweepParamsOBC.max_halfsweeps = args.get<size_t>("max_halfsweeps",30ul);
	GlobSweepParamsOBC.Minit = args.get<size_t>("Minit",2ul);
	GlobSweepParamsOBC.Qinit = args.get<size_t>("Qinit",2ul);
	GlobSweepParamsOBC.CALC_S_ON_EXIT = false;
	
	DMRG::CONTROL::DYN  DynParamsOBC;
	size_t start_alpha = args.get<size_t>("start_alpha",0);
	size_t end_alpha = args.get<size_t>("end_alpha",21ul);
	double alpha = args.get<double>("alpha",100.);
	DynParamsOBC.max_alpha_rsvd = [start_alpha, end_alpha, alpha] (size_t i) {return (i>=start_alpha and i<end_alpha)? alpha:0.;};
	
	// Gemeinsame Parameter bei unendlichen und offenen Randbedingungen
	vector<Param> params;
	
	if (Ly==1)
	{
		// Ungerade Plaetze sollen f-Plaetze mit U sein:
		params.push_back({"U",0.,0});
		params.push_back({"U",U,1});
		
		params.push_back({"V",V,0});
		params.push_back({"V",0.,1});
		
		params.push_back({"t0",Ec,0});
		params.push_back({"t0",Ef,1});
		
		params.push_back({"maxPower",1ul}); // hoechste Potenz von H
		
		// Hopping
		ArrayXXcd t1cell = hopping_PAM(L/2,tfc+0.i,tcc+0.i,tff+0.i,Retx+1.i*Imtx,Rety+1.i*Imty);
//		ArrayXXd t1cell = hopping_PAM(L/2,tfc+0.i,tcc+0.i,tff+0.i,Retx+1.i*Imtx,Rety+1.i*Imty).real(); // reell
		lout << "hopping:" << endl << t1cell << endl;
		params.push_back({"tFull",t1cell});
	}
	
	// Aufbau des Modells
	MODEL H(L,params,BC::OPEN);
	lout << H.info() << endl;
	
	auto Hfree = hopping_PAM(L/2,tfc+0.i,tcc+0.i,tff+0.i,Retx+1.i*Imtx,Rety+1.i*Imty);
	SelfAdjointEigenSolver<MatrixXcd> Eugen(Hfree.matrix()+onsite(L,Ec,Ef));
	VectorXd occ = Eugen.eigenvalues().head(N*L/(L*2));
	VectorXd unocc = Eugen.eigenvalues().tail(N*L/(L*2));
	double e0free = 2.*occ.sum()/L;
	lout << setprecision(16) << "e0free/L=("<<L<<",half-filling)=" << e0free << endl;
	
	Eigenstate<MODEL::StateXcd> g;
	MODEL::Solver DMRG(VERB);
	DMRG.GlobParam = GlobSweepParamsOBC;
	DMRG.userSetGlobParam();
	DMRG.DynParam = DynParamsOBC;
	DMRG.userSetDynParam();
	
	DMRG.edgeState(H, g, Q, LANCZOS::EDGE::GROUND);
	
	vector<MODEL::StateXcd> JCxg(L/2);
	
	lout << endl << "Applying J to ground state for all sites..." << endl;
	
	int ilast = L-2;
	double tol_OxV = 2.; // val>1 = do not compress
	
	// Test mit ED
	
	/*
	HubbardModel HED(L,L/2,L/2,U,BC_DANGLING);
	Eigenstate<VectorXd> gED;
	LanczosSolver<HubbardModel,VectorXd,double> Lutz;
	Lutz.edgeState(HED,gED);
	cout << "gED.energy=" << gED.energy << endl;
//	SparseMatrixXd Op = (HED.hopping_element(1,3,UP)+HED.hopping_element(1,3,DN)) * (HED.n(1,UP)+HED.n(1,DN));
	SparseMatrixXd Op =  (HED.n(2,UP)+HED.n(2,DN)) * (HED.hopping_element(4,2,UP)+HED.hopping_element(4,2,DN));
//	cout << "avgED=" << gED.state.dot(Op*gED.state) << endl;
	
	VMPS::HubbardSU2xU1 HDMRG(L,{{"U",U}},BC::OPEN);
	Eigenstate<VMPS::HubbardSU2xU1::StateXd> gDMRG;
	VMPS::HubbardSU2xU1::Solver DMRG_(DMRG::VERBOSITY::SILENT);
	DMRG_.GlobParam = GlobSweepParamsOBC;
	DMRG_.userSetGlobParam();
	DMRG_.edgeState(HDMRG, gDMRG, {1,static_cast<int>(L)}, LANCZOS::EDGE::GROUND);
	cout << "gDMRG.energy=" << gDMRG.energy << endl;
	
	cout << setprecision(3) << endl;
	MatrixXd MresED1(L,L);
	MatrixXd MresDMRG1(L,L);
	MatrixXd MresED2(L,L);
	MatrixXd MresDMRG2(L,L);
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		SparseMatrixXd Op1 = (HED.n(j,UP)+HED.n(j,DN)) * (HED.hopping_element(j,i,UP)+HED.hopping_element(j,i,DN)); // n(j)*cdagc(i,j) = cdag_nc
		SparseMatrixXd Op2 = (HED.hopping_element(j,i,UP)+HED.hopping_element(j,i,DN)) * (HED.n(i,UP)+HED.n(i,DN)); // cdagc(i,j)*n(i) = cdagn_c
		double resED1 = gED.state.dot(Op1*gED.state);
		double resED2 = gED.state.dot(Op2*gED.state);
		double resDMRG1 = avg(gDMRG.state, HDMRG.cdag_nc(i,j), gDMRG.state);
		double resDMRG2 = avg(gDMRG.state, HDMRG.cdagn_c(i,j), gDMRG.state);
		MresED1(i,j) = resED1;
		MresDMRG1(i,j) = resDMRG1;
		MresED2(i,j) = resED2;
		MresDMRG2(i,j) = resDMRG2;
		double nED = gED.state.dot((HED.n(i,UP)+HED.n(i,DN))*gED.state);
		double nDMRG = avg(gDMRG.state, HDMRG.n(i), gDMRG.state);
		cout << "i=" << i << ", j=" << j << ", ED=" << resED1 << ", DMRG=" <<  resDMRG1 << ", diff=" << abs(resED1-resDMRG1) << endl;
		cout << "i=" << i << ", j=" << j << ", ED=" << resED2 << ", DMRG=" <<  resDMRG2 << ", diff=" << abs(resED2-resDMRG2) << endl;
		cout << endl;
	}
	cout << endl << MresED1 << endl << endl;
	cout << endl << MresDMRG1 << endl << endl;
	cout << endl << MresED1-MresDMRG1 << endl << endl;
	cout << endl << MresED1.array()/MresDMRG1.array() << endl << endl;
	
	cout << endl << MresED2 << endl << endl;
	cout << endl << MresDMRG2 << endl << endl;
	cout << endl << MresED2-MresDMRG2 << endl << endl;
	cout << endl << MresED2.array()/MresDMRG2.array() << endl << endl;
	
	cout << setprecision(6) << endl;
	*/
	
	if (spec == "JJC")
	{
		#pragma omp parallel for
		for (int i=0; i<=ilast; i+=2)
		{
			int s = i/2;
			int slast = L/2-1;
			
			DMRG::VERBOSITY::OPTION CVERB = (i==L/2)? DMRG::VERBOSITY::HALFSWEEPWISE : DMRG::VERBOSITY::SILENT;
			JCxg[s] = g.state;
			
			vector<MODEL::StateXcd> states;
			vector<complex<double>> factors;
			
			if (tcc != 0.)
			{
				// cdag(i)*c(i+1)
				push_term(i, i+2, ilast, +1.i*tcc, tol_OxV, CVERB, H, g.state, states, factors);
				// cdag(i)*c(i-1)
				push_term(i, i-2, ilast, -1.i*tcc, tol_OxV, CVERB, H, g.state, states, factors);
			}
			
			if (tff != 0.)
			{
				// fdag(i)*f(i+1)
				push_term(i+1, i+3, ilast, +1.i*tff, tol_OxV, CVERB, H, g.state, states, factors);
				// fdag(i)f(i-1)
				push_term(i+1, i-1, ilast, -1.i*tff, tol_OxV, CVERB, H, g.state, states, factors);
			}
			
			if (states.size() > 0)
			{
				MpsCompressor<MODEL::Symmetry,complex<double>,complex<double>> Compadre(CVERB);
				Compadre.lincomboCompress(states, factors, JCxg[s], g.state, Mlimit, 1e-6, 32);
			}
			else
			{
				JCxg[s] = g.state;
			}
		}
		
//		//-----adjoint-----
//		#pragma omp parallel for
//		for (int i=0; i<=ilast; i+=2)
//		{
//			int s = L/2+i/2;
//			int slast = L/2+L/2-1;
//			
//			DMRG::VERBOSITY::OPTION CVERB = (i==L/2)? DMRG::VERBOSITY::HALFSWEEPWISE : DMRG::VERBOSITY::SILENT;
//			JCxg[s] = g.state;
//			
//			vector<MODEL::StateXcd> states;
//			vector<complex<double>> factors;
//			
//			if (tcc != 0.)
//			{
//				// cdag(i)*c(i+1)
//				push_term(i+2, i, ilast, -1.i*tcc, tol_OxV, CVERB, H, g.state, states, factors);
//				// cdag(i)*c(i-1)
//				push_term(i-2, i, ilast, +1.i*tcc, tol_OxV, CVERB, H, g.state, states, factors);
//			}
//			
//			if (tff != 0.)
//			{
//				// fdag(i)*f(i+1)
//				push_term(i+3, i+1, ilast, -1.i*tff, tol_OxV, CVERB, H, g.state, states, factors);
//				// fdag(i)f(i-1)
//				push_term(i-1, i+1, ilast, +1.i*tff, tol_OxV, CVERB, H, g.state, states, factors);
//			}
//			
//			if (states.size() > 0)
//			{
//				MpsCompressor<MODEL::Symmetry,complex<double>,complex<double>> Compadre(CVERB);
//				Compadre.lincomboCompress(states, factors, JCxg[s], g.state, Mlimit, 1e-6, 32);
//			}
//			else
//			{
//				JCxg[s] = g.state;
//			}
//		}
	}
	else if (spec == "JJE")
	{
		#pragma omp parallel for
		for (int i=0; i<=ilast; i+=2)
		{
			int s = i/2;
			int slast = L/2-1;
			
			DMRG::VERBOSITY::OPTION CVERB = (i==L/2)? DMRG::VERBOSITY::HALFSWEEPWISE : DMRG::VERBOSITY::SILENT;
			JCxg[s] = g.state;
			
			vector<MODEL::StateXcd> states;
			vector<complex<double>> factors;
			
			// term tcc*tcc
			if (tcc != 0.)
			{
				// cdag(i)*c(i+2)
				push_term(i, i+4, ilast, +1.i*tcc*tcc, tol_OxV, CVERB, H, g.state, states, factors);
				// cdag(i)*c(i-2)
				push_term(i, i-4, ilast, -1.i*tcc*tcc, tol_OxV, CVERB, H, g.state, states, factors);
			}
			
			if (tff != 0.)
			{
				// fdag(i)*f(i+2)
				push_term(i+1, i+5, ilast, +1.i*tff*tff, tol_OxV, CVERB, H, g.state, states, factors);
				// fdag(i)*f(i-2)
				push_term(i+1, i-3, ilast, -1.i*tff*tff, tol_OxV, CVERB, H, g.state, states, factors);
			}
			
			// term 0.5*tfc*(tcc+tff)
			if (tfc != 0.)
			{
				// cdag(i)*f(i+1)
				push_term(i,   i+3, ilast, +0.5i*(tcc+tff)*tfc, tol_OxV, CVERB, H, g.state, states, factors);
				// fdag(i)*c(i-1)
				push_term(i+1, i-2, ilast, -0.5i*(tcc+tff)*tfc, tol_OxV, CVERB, H, g.state, states, factors);
				// fdag(i)*c(i+1)
				push_term(i+1, i+2, ilast, +0.5i*(tcc+tff)*tfc, tol_OxV, CVERB, H, g.state, states, factors);
				// cdag(i)*f(i-1)
				push_term(i,   i-1, ilast, -0.5i*(tcc+tff)*tfc, tol_OxV, CVERB, H, g.state, states, factors);
			}
			
			// term Ec*tcc
			if (Ec != 0. and tcc != 0.)
			{
				// cdag(i)*c(i+1)
				push_term(i, i+2, ilast, +1.i*tcc*Ec, tol_OxV, CVERB, H, g.state, states, factors);
				// cdag(i)*c(i-1)
				push_term(i, i-2, ilast, -1.i*tcc*Ec, tol_OxV, CVERB, H, g.state, states, factors);
			}
			
			// term Ef*tff
			if (Ef != 0. and tff != 0.)
			{
				// fdag(i)*f(i+1)
				push_term(i+1, i+3, ilast, +1.i*tff*Ef, tol_OxV, CVERB, H, g.state, states, factors);
				// fdag(i)*f(i-1)
				push_term(i+1, i-1, ilast, -1.i*tff*Ef, tol_OxV, CVERB, H, g.state, states, factors);
			}
			
			// term U*tff
			if (tff != 0. and U != 0. and i+1 <= ilast)
			{
				// fdag(i)*nf(i)*f(i+1)
				push_corrhop(i+1, i+3, ilast, +0.5i*U*tff, tol_OxV, CVERB, H, g.state, states, factors, false); // false=cdagn_c
				// fdag(i)*nf(i)*f(i-1)
				push_corrhop(i+1, i-1, ilast, -0.5i*U*tff, tol_OxV, CVERB, H, g.state, states, factors, false);
				// fdag(i+1)*nf(i)*f(i)
				push_corrhop(i+3, i+1, ilast, -0.5i*U*tff, tol_OxV, CVERB, H, g.state, states, factors, true); // true=cdag_nc
				// fdag(i-1)*nf(i)*f(i)
				push_corrhop(i-1, i+1, ilast, +0.5i*U*tff, tol_OxV, CVERB, H, g.state, states, factors, true);
			}
			
			if (states.size() > 0)
			{
				MpsCompressor<MODEL::Symmetry,complex<double>,complex<double>> Compadre(CVERB);
				Compadre.lincomboCompress(states, factors, JCxg[s], g.state, Mlimit, 1e-6, 32);
			}
			else
			{
				JCxg[s] = g.state;
			}
		}
		
//		//-----adjoint-----
//		#pragma omp parallel for
//		for (int i=0; i<=ilast; i+=2)
//		{
//			int s = L/2+i/2;
//			int slast = L/2+L/2-1;
//			
//			DMRG::VERBOSITY::OPTION CVERB = (i==L/2)? DMRG::VERBOSITY::HALFSWEEPWISE : DMRG::VERBOSITY::SILENT;
//			JCxg[s] = g.state;
//			
//			vector<MODEL::StateXcd> states;
//			vector<complex<double>> factors;
//			
//			// term tcc*tcc
//			if (tcc != 0.)
//			{
//				// cdag(i)*c(i+2)
//				push_term(i+4, i, ilast, -1.i*tcc*tcc, tol_OxV, CVERB, H, g.state, states, factors);
//				// cdag(i)*c(i-2)
//				push_term(i-4, i, ilast, +1.i*tcc*tcc, tol_OxV, CVERB, H, g.state, states, factors);
//			}
//				
//			if (tff != 0.)
//			{
//				// fdag(i)*f(i+2)
//				push_term(i+5, i+1, ilast, -1.i*tff*tff, tol_OxV, CVERB, H, g.state, states, factors);
//				// fdag(i)*f(i-2)
//				push_term(i-3, i+1, ilast, +1.i*tff*tff, tol_OxV, CVERB, H, g.state, states, factors);
//			}
//			
//			// term 0.5*tfc*(tcc+tff)
//			if (tfc != 0.)
//			{
//				// cdag(i)*f(i+1)
//				push_term(i+3, i,   ilast, -0.5i*(tcc+tff)*tfc, tol_OxV, CVERB, H, g.state, states, factors);
//				// fdag(i)*c(i-1)
//				push_term(i-2, i+1, ilast, +0.5i*(tcc+tff)*tfc, tol_OxV, CVERB, H, g.state, states, factors);
//				// fdag(i)*c(i+1)
//				push_term(i+2, i+1, ilast, -0.5i*(tcc+tff)*tfc, tol_OxV, CVERB, H, g.state, states, factors);
//				// cdag(i)*f(i-1)
//				push_term(i-1, i,   ilast, +0.5i*(tcc+tff)*tfc, tol_OxV, CVERB, H, g.state, states, factors);
//			}
//				
//			// term Ec*tcc
//			if (Ec != 0. and tcc != 0.)
//			{
//				// cdag(i)*c(i+1)
//				push_term(i+2, i, ilast, -1.i*tcc*Ec, tol_OxV, CVERB, H, g.state, states, factors);
//				// cdag(i)*c(i-1)
//				push_term(i-2, i, ilast, +1.i*tcc*Ec, tol_OxV, CVERB, H, g.state, states, factors);
//			}
//			
//			// term Ef*tff
//			if (Ef != 0. and tff != 0.)
//			{
//				// fdag(i)*f(i+1)
//				push_term(i+3, i+1, ilast, -1.i*tff*Ef, tol_OxV, CVERB, H, g.state, states, factors);
//				// fdag(i)*f(i-1)
//				push_term(i-1, i+1, ilast, +1.i*tff*Ef, tol_OxV, CVERB, H, g.state, states, factors);
//			}
//			
//			// term U*tff
//			if (tff != 0. and U != 0. and i+1 <= ilast)
//			{
//				// (fdag(i)*nf(i)*f(i+1))^dag = fdag(i+1)*nf(i)*f(i)
//				push_corrhop(i+3, i+1, ilast, -1.i*U*tff, tol_OxV, CVERB, H, g.state, states, factors, true);
//				// (fdag(i)*nf(i)*f(i-1))^dag = fdag(i-1)*nf(i)*f(i)
//				push_corrhop(i-1, i+1, ilast, +1.i*U*tff, tol_OxV, CVERB, H, g.state, states, factors, true);
//			}
//			
//			if (states.size() > 0)
//			{
//				MpsCompressor<MODEL::Symmetry,complex<double>,complex<double>> Compadre(CVERB);
//				Compadre.lincomboCompress(states, factors, JCxg[s], g.state, Mlimit, 1e-6, 32);
//			}
//			else
//			{
//				JCxg[s] = g.state;
//			}
//		}
	}
	
	lout << endl << "Applying J to ground state for all sites done!" << endl << endl;
	
	vector<MODEL::StateXcd> Psi = JCxg;
	for (int i=0; i<L/2; ++i)
	{
		Psi.push_back(JCxg[i]);
	}
	JCxg.resize(0);
	for (int i=0; i<L; ++i)
	{
		Psi[i].eps_svd = tol_compr;
		Psi[i].max_Nsv = max(Psi[i].calc_Mmax(),Mstart);
		lout << i << "\t" << Psi[i].info() << endl;
		if (i==L/2-1) lout << "----" << endl;
	}
	lout << endl;
	
	vector<TDVPPropagator<MODEL,MODEL::Symmetry,complex<double>,complex<double>,MODEL::StateXcd>> TDVP(L);
	for (int i=0; i<L; ++i)
	{
		TDVP[i] = TDVPPropagator<MODEL,MODEL::Symmetry,complex<double>,complex<double>,MODEL::StateXcd>(H,Psi[i]);
	}
	
	vector<EntropyObserver<MODEL::StateXcd>> Sobs(L);
	vector<vector<bool>> TWO_SITE(L);
	for (int i=0; i<L; ++i)
	{
		DMRG::VERBOSITY::OPTION SOBSVERB = (i==3*L/4)? VERB : DMRG::VERBOSITY::SILENT;
		Sobs[i] = EntropyObserver<MODEL::StateXcd>(H.length(), Nt, SOBSVERB, tol_DeltaS);
		TWO_SITE[i] = Sobs[i].TWO_SITE(0, Psi[i], 1.);
	}
	
//	vector<MatrixXcd> Joverlap(Nt);
	VectorXcd JoverlapSum(Nt);
	
	Stopwatch<> TpropTimer;
	IntervalIterator t(0.,tmax/2,Nt);
	IntervalIterator tfull(0.,tmax,Nt); //tfull=tfull.begin(2);
	
	for (t=t.begin(2), tfull=tfull.begin(2); t!=t.end(); ++t, ++tfull)
	{
		Stopwatch<> StepTimer;
		
		JoverlapSum(t.index()) = calc_Joverlap(Psi, exp(+1.i*g.energy*(*tfull)))/(0.5*L);
		tfull << JoverlapSum(t.index());
		lout << "save results at tfull=" << *tfull << ", res=" << JoverlapSum(t.index()) << endl;
		tfull.save(make_string(spec+"t_",base,"_",tbase,".dat"));
		
		if (t.index() != t.end()-1)
		{
			#pragma omp parallel for
			for (int i=0; i<L; ++i)
			{
				//-----------------------------------------------------------
				if (i<L/2)
				{
					TDVP[i].t_step_adaptive(H, Psi[i], -1.i*0.5*dt, TWO_SITE[i], 1); // forwards
				}
				else
				{
					TDVP[i].t_step_adaptive(H, Psi[i], +1.i*0.5*dt, TWO_SITE[i], 1); // backwards
				}
				//-----------------------------------------------------------
				
				if (i==L/2)
				{
					lout << "propagated to t=±" << *t+0.5*dt << ", tfull=±" << *tfull+dt << endl;
					lout << TDVP[i].info() << endl;
					lout << Psi[i].info() << endl;
				}
				
				if (Psi[i].get_truncWeight().sum() > 0.5*tol_compr)
				{
					Psi[i].max_Nsv = min(static_cast<size_t>(max(Psi[i].max_Nsv*1.1, Psi[i].max_Nsv+50.)),Mlimit);
					if (VERB >= DMRG::VERBOSITY::HALFSWEEPWISE and i==L/2)
					{
						lout << termcolor::yellow << "Setting Psi.max_Nsv to " << Psi[i].max_Nsv << termcolor::reset << endl;
					}
				}
				else
				{
					if (VERB >= DMRG::VERBOSITY::HALFSWEEPWISE and i==L/2)
					{
						lout << termcolor::green << "trunc_weight=" << Psi[i].get_truncWeight().sum() << " < " << 0.5*tol_compr << " => no bond dimension increase" << termcolor::reset << endl;
					}
				}
			}
			lout << StepTimer.info("time step") << endl;
			
			#pragma omp parallel for
			for (int i=0; i<L; ++i)
			{
				auto PsiTmp = Psi[i]; PsiTmp.entropy_skim();
				TWO_SITE[i] = Sobs[i].TWO_SITE(tfull.index(), PsiTmp);
			}
			
			if (VERB >= DMRG::VERBOSITY::HALFSWEEPWISE) lout << StepTimer.info("entropy calculation") << endl;
		}
		
		lout << TpropTimer.info("total running time",false) << endl;
		lout << endl;
	}
	
	lout << "saved to: " << make_string(spec+"t_",base,"_",tbase,".dat") << endl << endl;
	
	VectorXd tvals = tfull.get_abscissa();
	for (int i=0; i<tvals.rows(); ++i)
	{
		lout << "t=" << tvals(i) << "\t" << JoverlapSum(i) << endl;
	}
	FT_and_save(tvals, tmax, JoverlapSum, wmin, wmax, wpoints, make_string(spec+"w_",base,"_",tbase,"_",wbase,"_DAMPING=GAUSS",".dat"), GAUSS);
	FT_and_save(tvals, tmax, JoverlapSum, wmin, wmax, wpoints, make_string(spec+"w_",base,"_",tbase,"_",wbase,"_DAMPING=LORENTZ",".dat"), LORENTZ);
	FT_and_save(tvals, tmax, JoverlapSum, wmin, wmax, wpoints, make_string(spec+"w_",base,"_",tbase,"_",wbase,"_DAMPING=NO",".dat"), NODAMPING);
	
	lout << endl << "saved to: " << make_string(spec+"w_",base,"_",tbase,"_",wbase,"_DAMPING=[...].dat") << endl;
}
