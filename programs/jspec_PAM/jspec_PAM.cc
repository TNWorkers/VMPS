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

#include "solvers/DmrgSolver.h"
#include "solvers/MpsCompressor.h"
#include "solvers/EntropyObserver.h"
#include "solvers/TDVPPropagator.h"
#include "DmrgLinearAlgebra.h"

//#include "models/HubbardSU2xU1.h"
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

MatrixXcd calc_Joverlap (const vector<MODEL::StateXcd> &Psi0, const vector<MODEL::StateXcd> &Psi, const complex<double> &phase)
{
	int L = Psi.size();
	MatrixXcd res(L,L);
	
	#pragma omp parallel for collapse(2)
	for (int i=0; i<L; ++i)
	for (int j=0; j<L; ++j)
	{
		res(i,j) = dot(Psi0[i],Psi[j]);
	}
	
	res *= phase;
	
	return res;
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
	size_t Mlim = args.get<size_t>("Mlim",800ul); // Bonddimension fuer Dynamik
	double dt = args.get<double>("dt",0.1);
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
	lout << base << endl;
	lout.set(base+".log",wd+"log"); // Log-Datei im Unterordner log
	
	// Parameter fuer den Grundzustand:
	DMRG::CONTROL::GLOB GlobSweepParamsOBC;
	GlobSweepParamsOBC.min_halfsweeps = args.get<size_t>("min_halfsweeps",10ul);
	GlobSweepParamsOBC.max_halfsweeps = args.get<size_t>("max_halfsweeps",20ul);
	GlobSweepParamsOBC.Minit = args.get<size_t>("Minit",2ul);
	GlobSweepParamsOBC.Qinit = args.get<size_t>("Qinit",2ul);
	GlobSweepParamsOBC.CALC_S_ON_EXIT = false;
	
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
//	DMRG.DynParam = DynParam;
//	DMRG.userSetDynParam();
	
	DMRG.edgeState(H, g, Q, LANCZOS::EDGE::GROUND);
	
	vector<MODEL::StateXcd> JCxg(L/2);
	
	lout << endl << "Applying J to ground state for all sites..." << endl;
	
	int ilast = L-2;
	double tol_OxV = 2.; // val>1 = do not compress
	
	if (spec == "JJC")
	{
		#pragma omp parallel for
		for (int i=0; i<=ilast; i+=2)
		{
			int s = i/2;
			int slast = L/2-1;
			
			DMRG::VERBOSITY::OPTION CVERB = (i==L/2)? DMRG::VERBOSITY::HALFSWEEPWISE : DMRG::VERBOSITY::SILENT;
			JCxg[s] = g.state;
			
			vector<MODEL::StateXcd> tmp;
			vector<complex<double>> factors;
			
			// cdag(i)*c(i+1)
			if (i+2 <= ilast)
			{
//				cout << "set forw: " << i << "," << i+2 << endl;
				MODEL::StateXcd OxVres;
				OxV_exact(H.cdagc(i,i+2), g.state, OxVres, tol_OxV, CVERB);
				tmp.push_back(OxVres);
				factors.push_back(+1.i*tcc);
			}
			// cdag(i)*c(i-1)
			if (i-2 >= 0)
			{
//				cout << "set back: " << i << "," << i-2 << endl;
				MODEL::StateXcd OxVres;
				OxV_exact(H.cdagc(i,i-2), g.state, OxVres, tol_OxV, CVERB);
				tmp.push_back(OxVres);
				factors.push_back(-1.i*tcc);
			}
			
			MpsCompressor<MODEL::Symmetry,complex<double>,complex<double>> Compadre(CVERB);
			Compadre.lincomboCompress(tmp, factors, JCxg[s], g.state, Mlim, 1e-7);
		}
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
			
			vector<MODEL::StateXcd> tmp;
			vector<complex<double>> factors;
			
			// term tcc*tcc
			// cdag(i)*c(i+2)
			if (i+4 <= ilast)
			{
				MODEL::StateXcd OxVres;
				OxV_exact(H.cdagc(i,i+4), g.state, OxVres, tol_OxV, CVERB);
				tmp.push_back(OxVres);
				factors.push_back(+1.i*tcc*tcc);
			}
			// cdag(i-2)*c(i)
			if (i-4 >= 0)
			{
				MODEL::StateXcd OxVres;
				OxV_exact(H.cdagc(i,i-4), g.state, OxVres, tol_OxV, CVERB);
				tmp.push_back(OxVres);
				factors.push_back(-1.i*tcc*tcc);
			}
			
			// term 0.5*tfc*tcc
			if (tfc != 0.)
			{
				// cdag(i)*f(i+1)
				if (i+3 <= ilast)
				{
					MODEL::StateXcd OxVres;
					OxV_exact(H.cdagc(i,i+3), g.state, OxVres, tol_OxV, CVERB);
					tmp.push_back(OxVres);
					factors.push_back(+0.5i*tcc*tfc);
				}
				// fdag(i)*c(i-1)
				if (i+1 <= ilast and i-2>=0)
				{
					MODEL::StateXcd OxVres;
					OxV_exact(H.cdagc(i+1,i-2), g.state, OxVres, tol_OxV, CVERB);
					tmp.push_back(OxVres);
					factors.push_back(-0.5i*tcc*tfc);
				}
				
				// fdag(i)*c(i+1)
				if (i+2 <= ilast)
				{
					MODEL::StateXcd OxVres;
					OxV_exact(H.cdagc(i+1,i+2), g.state, OxVres, tol_OxV, CVERB);
					tmp.push_back(OxVres);
					factors.push_back(+0.5i*tcc*tfc);
				}
				// cdag(i)*f(i-1)
				if (i-1 >= 0)
				{
					MODEL::StateXcd OxVres;
					OxV_exact(H.cdagc(i,i-1), g.state, OxVres, tol_OxV, CVERB);
					tmp.push_back(OxVres);
					factors.push_back(-0.5i*tcc*tfc);
				}
			}
			
			// term Ec*tcc
			if (Ec != 0.)
			{
				// cdag(i)*c(i+1)
				if (i+2 <= ilast)
				{
					MODEL::StateXcd OxVres;
					OxV_exact(H.cdagc(i,i+2), g.state, OxVres, tol_OxV, CVERB);
					tmp.push_back(OxVres);
					factors.push_back(+1.i*tcc*Ec);
				}
				// cdag(i)*c(i-1)
				if (i-2 >= 0)
				{
					MODEL::StateXcd OxVres;
					OxV_exact(H.cdagc(i,i-2), g.state, OxVres, tol_OxV, CVERB);
					tmp.push_back(OxVres);
					factors.push_back(-1.i*tcc*Ec);
				}
			}
			
			MpsCompressor<MODEL::Symmetry,complex<double>,complex<double>> Compadre(CVERB);
			Compadre.lincomboCompress(tmp, factors, JCxg[s], g.state, Mlim, 1e-7);
		}
	}
	
	lout << endl << "Applying J to ground state for all sites done!" << endl << endl;
	
	vector<MODEL::StateXcd> Psi = JCxg;
	for (int i=0; i<L/2; ++i)
	{
		Psi[i].eps_svd = tol_compr;
		Psi[i].max_Nsv = max(Psi[i].calc_Mmax(),Mlim);
	}
	
	vector<TDVPPropagator<MODEL,MODEL::Symmetry,complex<double>,complex<double>,MODEL::StateXcd>> TDVP(L/2);
	for (int i=0; i<L/2; ++i)
	{
		TDVP[i] = TDVPPropagator<MODEL,MODEL::Symmetry,complex<double>,complex<double>,MODEL::StateXcd>(H,Psi[i]);
	}
	
	vector<EntropyObserver<MODEL::StateXcd>> Sobs(L/2);
	vector<vector<bool>> TWO_SITE(L);
	for (int i=0; i<L/2; ++i)
	{
		DMRG::VERBOSITY::OPTION SOBSVERB = (i==L/4)? VERB : DMRG::VERBOSITY::SILENT;
		Sobs[i] = EntropyObserver<MODEL::StateXcd>(H.length(), Nt, SOBSVERB, tol_DeltaS);
		TWO_SITE[i] = Sobs[i].TWO_SITE(0, Psi[i], 1.);
	}
	
	// calc overlap at t=0
	vector<MatrixXcd> Joverlap(Nt);
	VectorXcd JoverlapSum(Nt);
	
	Stopwatch<> TpropTimer;
	IntervalIterator t(0.,tmax,Nt);
	for (t=t.begin(2); t!=t.end(); ++t)
	{
		lout << "t=" << *t << endl;
		Stopwatch<> StepTimer;
		
		Joverlap[t.index()] = calc_Joverlap(JCxg, Psi, exp(+1.i*g.energy*(*t)));
		JoverlapSum(t.index()) = Joverlap[t.index()].sum()/(0.5*L);
		t << JoverlapSum(t.index());
		t.save(make_string(spec+"t_",base,".dat"));
		
		#pragma omp parallel for
		for (int i=0; i<L/2; ++i)
		{
			//-----------------------------------------------------------
			TDVP[i].t_step_adaptive(H, Psi[i], -1.i*dt, TWO_SITE[i], 1);
			//-----------------------------------------------------------
			
			if (i==L/4)
			{
				lout << "propagated to t=" << *t << endl;
				lout << TDVP[i].info() << endl;
				lout << Psi[i].info() << endl;
				lout << StepTimer.info("time step") << endl;
			}
			
			if (Psi[i].get_truncWeight().sum() > 0.5*tol_compr)
			{
				Psi[i].max_Nsv = min(static_cast<size_t>(max(Psi[i].max_Nsv*1.1, Psi[i].max_Nsv+1.)),Mlim);
				if (VERB >= DMRG::VERBOSITY::HALFSWEEPWISE and i==0)
				{
					lout << termcolor::yellow << "Setting Psi.max_Nsv to " << Psi[i].max_Nsv << termcolor::reset << endl;
				}
			}
		}
		
		for (int i=0; i<L/2; ++i)
		{
			auto PsiTmp = Psi[i]; PsiTmp.entropy_skim();
			TWO_SITE[i] = Sobs[i].TWO_SITE(t.index(), PsiTmp);
			if (VERB >= DMRG::VERBOSITY::HALFSWEEPWISE and i==L/4) lout << StepTimer.info("entropy calculation") << endl;
		}
		
//		cout << endl << endl << Joverlap[t.index()] << endl << endl;
		
		lout << TpropTimer.info("total running time",false) << endl;
		lout << endl;
	}
	
	VectorXd tvals = t.get_abscissa();
	FT_and_save(tvals, tmax, JoverlapSum, wmin, wmax, wpoints, make_string(spec+"w_",base,"_DAMPING=GAUSS",".dat"), GAUSS);
	FT_and_save(tvals, tmax, JoverlapSum, wmin, wmax, wpoints, make_string(spec+"w_",base,"_DAMPING=LORENTZ",".dat"), LORENTZ);
	FT_and_save(tvals, tmax, JoverlapSum, wmin, wmax, wpoints, make_string(spec+"w_",base,"_DAMPING=NO",".dat"), NODAMPING);
}
