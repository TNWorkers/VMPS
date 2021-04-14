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

#include "models/HubbardSU2xU1.h"
typedef VMPS::HubbardSU2xU1 MODEL;
#include "models/PeierlsHubbardSU2xU1.h"
typedef VMPS::PeierlsHubbardSU2xU1 MODELC;

#ifdef TIME_PROP_USE_TERMPLOT
#include "plot.hpp"
#include "TerminalPlot.h"
#endif

#include "models/ParamCollection.h"
#include "solvers/SpectralManager.h"

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

void push_term (int i, int j, complex<double> lambda, double tol_OxV, DMRG::VERBOSITY::OPTION CVERB, const MODELC &H, 
                const MODELC::StateXcd &target, vector<MODELC::StateXcd> &states, vector<complex<double>> &factors)
{
	if (i>=0 and i<H.length() and j>=0 and j<H.length())
	{
//		cout << "push term at i=" << i << ", j=" << j << ", lambda=" << lambda << endl;
		//cout << H.cdagc(i,j).info() << endl;
		MODELC::StateXcd OxVres;
		OxV_exact(H.cdagc(i,j), target, OxVres, tol_OxV, CVERB);
		states.push_back(OxVres);
		factors.push_back(lambda);
	}
}

void push_operator (int i, int j, complex<double> lambda, const MODELC &H, vector<Mpo<MODELC::Symmetry,complex<double>>> &operators, vector<complex<double>> &factors)
{
	if (i>=0 and i<H.length() and j>=0 and j<H.length())
	{
		//cout << "push operator at i=" << i << ", j=" << j << ", lambda=" << lambda << endl;
		//cout << H.cdagc(i,j).info() << endl;
		operators.push_back(H.cdagc(i,j));
		factors.push_back(lambda);
	}
}

void push_corrhop (int i, int j, complex<double> lambda, double tol_OxV, DMRG::VERBOSITY::OPTION CVERB, const MODELC &H, 
                   const MODELC::StateXcd &target, vector<MODELC::StateXcd> &states, vector<complex<double>> &factors, bool DAG=false)
{
	if (i>=0 and i<H.length() and j>=0 and j<H.length())
	{
		MODELC::StateXcd OxVres;
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

void push_corrhopOperator (int i, int j, complex<double> lambda, const MODELC &H, vector<Mpo<MODELC::Symmetry,complex<double>>> &operators, vector<complex<double>> &factors, bool DAG=false)
{
	if (i>=0 and i<H.length() and j>=0 and j<H.length())
	{
		if (DAG)
		{
			operators.push_back(H.cdag_nc(i,j));
		}
		else
		{
			operators.push_back(H.cdagn_c(i,j));
		}
		factors.push_back(lambda);
	}
}

tuple<vector<MODELC::StateXcd>,
      vector<Mpo<MODELC::Symmetry,complex<double>>>,
      vector<complex<double>>> 
apply_J (int j0, string spec, int L, int dLphys, double tol_OxV, const MODELC &H, const MODELC::StateXcd &Psi, 
         double tcc, double tff, double tfc, double Ec, double Ef, double U, int Mlimit)
{
	vector<MODELC::StateXcd> res(1);
	vector<Mpo<MODELC::Symmetry,complex<double>>> Res;
	vector<complex<double>> Fac;
	
	if (spec == "JJC")
	{
		#pragma omp parallel for
		for (int i=0; i<L; i+=2)
		{
//			int s = i/2;
			int s = 0;
			
			DMRG::VERBOSITY::OPTION CVERB = (i==j0)? DMRG::VERBOSITY::HALFSWEEPWISE : DMRG::VERBOSITY::SILENT;
			
			vector<MODEL::StateXcd> states;
			vector<complex<double>> factors;
			
			if (tcc != 0.)
			{
				int i1 = i*dLphys;
				int i2 = (i+2)*dLphys;
				complex<double> lambda = +1.i*(-tcc);
				// cdag(i)*c(i+1)
				if (i==j0) push_term(i1, i2, lambda, tol_OxV, CVERB, H, Psi, states, factors);
				#pragma omp critical
				{
					push_operator(i2, i1, conj(lambda), H, Res, Fac);
				}
				
				i1 = i*dLphys;
				i2 = (i-2)*dLphys;
				// cdag(i)*c(i-1)
				if (i==j0) push_term(i1, i2, conj(lambda), tol_OxV, CVERB, H, Psi, states, factors);
				push_operator(i2, i1, lambda, H, Res, Fac);
			}
			
			if (tff != 0.)
			{
				int i1 = (i+1)*dLphys;
				int i2 = (i+3)*dLphys;
				complex<double> lambda = +1.i*(-tff);
				// fdag(i)*f(i+1)
				if (i==j0) push_term(i1, i2, lambda, tol_OxV, CVERB, H, Psi, states, factors);
				#pragma omp critical
				{
					push_operator(i2, i1, conj(lambda), H, Res, Fac);
				}
				
				i1 = (i+1)*dLphys;
				i2 = (i-1)*dLphys;
				// fdag(i)f(i-1)
				if (i==j0) push_term(i1, i2, conj(lambda), tol_OxV, CVERB, H, Psi, states, factors);
				#pragma omp critical
				{
					push_operator(i2, i1, lambda, H, Res, Fac);
				}
			}
			
			if (i==j0)
			{
				if (states.size() > 0)
				{
					MpsCompressor<MODEL::Symmetry,complex<double>,complex<double>> Compadre(CVERB);
					Compadre.lincomboCompress(states, factors, res[s], Psi, Mlimit, 1e-6, 32);
				}
				else
				{
					lout << "no operator at site " << s << endl;
					res[s] = Psi;
				}
			}
		}
	}
	else if (spec == "JJE")
	{
		for (int i=0; i<L; i+=2)
		{
//			int s = i/2;
			int s = 0;
			
			DMRG::VERBOSITY::OPTION CVERB = (i==j0)? DMRG::VERBOSITY::HALFSWEEPWISE : DMRG::VERBOSITY::SILENT;
			
			vector<MODEL::StateXcd> states;
			vector<complex<double>> factors;
			
			// term tcc*tcc
			if (tcc != 0.)
			{
				int i1 = i*dLphys;
				int i2 = (i+4)*dLphys;
				complex<double> lambda = +1.i*tcc*tcc;
				// cdag(i)*c(i+2)
				if (i==j0) push_term(i1, i2, lambda, tol_OxV, CVERB, H, Psi, states, factors);
				#pragma omp critical
				{
					push_operator(i2, i1, conj(lambda), H, Res, Fac);
				}
				
				i1 = i*dLphys;
				i2 = (i-4)*dLphys;
				// cdag(i)*c(i-2)
				if (i==j0) push_term(i1, i2, conj(lambda), tol_OxV, CVERB, H, Psi, states, factors);
				#pragma omp critical
				{
					push_operator(i2, i1, lambda, H, Res, Fac);
				}
			}
			
			if (tff != 0.)
			{
				int i1 = (i+1)*dLphys;
				int i2 = (i+5)*dLphys;
				complex<double> lambda = +1.i*tff*tff;
				// fdag(i)*f(i+2)
				if (i==j0) push_term(i1, i2, lambda, tol_OxV, CVERB, H, Psi, states, factors);
				#pragma omp critical
				{
					push_operator(i2, i1, conj(lambda), H, Res, Fac);
				}
				i1 = (i+1)*dLphys;
				i2 = (i-3)*dLphys;
				// fdag(i)*f(i-2)
				if (i==j0) push_term(i1, i2, conj(lambda), tol_OxV, CVERB, H, Psi, states, factors);
				#pragma omp critical
				{
					push_operator(i2, i1, lambda, H, Res, Fac);
				}
			}
			
			// term 0.5*tfc*(tcc+tff)
			if (tfc != 0.)
			{
				int i1 = i*dLphys;
				int i2 = (i+3)*dLphys;
				complex<double> lambda = +0.5i*(tcc+tff)*tfc;
				// cdag(i)*f(i+1)
				if (i==j0) push_term(i1, i2, lambda, tol_OxV, CVERB, H, Psi, states, factors);
				#pragma omp critical
				{
					push_operator(i2, i1, conj(lambda), H, Res, Fac);
				}
				
				i1 = i*dLphys;
				i2 = (i-1)*dLphys;
				// cdag(i)*f(i-1)
				if (i==j0) push_term(i1, i2, conj(lambda), tol_OxV, CVERB, H, Psi, states, factors);
				#pragma omp critical
				{
					push_operator(i2, i1, lambda, H, Res, Fac);
				}
				
				i1 = (i+1)*dLphys;
				i2 = (i-2)*dLphys;
				// fdag(i)*c(i-1)
				if (i==j0) push_term(i1, i2, conj(lambda), tol_OxV, CVERB, H, Psi, states, factors);
				#pragma omp critical
				{
					push_operator(i2, i1, lambda, H, Res, Fac);
				}
				
				i1 = (i+1)*dLphys;
				i2 = (i+2)*dLphys;
				// fdag(i)*c(i+1)
				if (i==j0) push_term(i1, i2, lambda, tol_OxV, CVERB, H, Psi, states, factors);
				#pragma omp critical
				{
					push_operator(i2, i1, conj(lambda), H, Res, Fac);
				}
			}
			
			// term Ec*tcc
			if (Ec != 0. and tcc != 0.)
			{
				int i1 = i*dLphys;
				int i2 = (i+2)*dLphys;
				complex<double> lambda = +1.i*(-tcc)*Ec;
				// cdag(i)*c(i+1)
				if (i==j0) push_term(i1, i2, lambda, tol_OxV, CVERB, H, Psi, states, factors);
				#pragma omp critical
				{
					push_operator(i2, i1, conj(lambda), H, Res, Fac);
				}
				
				i1 = i*dLphys;
				i2 = (i-2)*dLphys;
				// cdag(i)*c(i-1)
				if (i==j0) push_term(i1, i2, conj(lambda), tol_OxV, CVERB, H, Psi, states, factors);
				#pragma omp critical
				{
					push_operator(i2, i1, lambda, H, Res, Fac);
				}
			}
			
			// term Ef*tff
			if (Ef != 0. and tff != 0.)
			{
				int i1 = (i+1)*dLphys;
				int i2 = (i+3)*dLphys;
				complex<double> lambda = +1.i*(-tff)*Ef;
				// fdag(i)*f(i+1)
				if (i==j0) push_term(i1, i2, lambda, tol_OxV, CVERB, H, Psi, states, factors);
				#pragma omp critical
				{
					push_operator(i2, i1, conj(lambda), H, Res, Fac);
				}
				
				i1 = (i+1)*dLphys;
				i2 = (i-1)*dLphys;
				// fdag(i)*f(i-1)
				if (i==j0) push_term(i1, i2, conj(lambda), tol_OxV, CVERB, H, Psi, states, factors);
				#pragma omp critical
				{
					push_operator(i2, i1, lambda, H, Res, Fac);
				}
			}
			
			// term U*tff
			if (tff != 0. and U != 0.)
			{
				int i1 = (i+1)*dLphys;
				int i2 = (i+3)*dLphys;
				complex<double> lambda = +0.5i*U*(-tff);
				// fdag(i)*nf(i)*f(i+1)
				if (i==j0) push_corrhop(i1, i2, lambda, tol_OxV, CVERB, H, Psi, states, factors, false); // false=cdagn_c
				#pragma omp critical
				{
					push_corrhopOperator(i2, i1, conj(lambda), H, Res, Fac, true);
				}
				
				i1 = (i+1)*dLphys;
				i2 = (i-1)*dLphys;
				// fdag(i)*nf(i)*f(i-1)
				if (i==j0) push_corrhop(i1, i2, conj(lambda), tol_OxV, CVERB, H, Psi, states, factors, false);
				#pragma omp critical
				{
					push_corrhopOperator(i2, i1, lambda, H, Res, Fac, true);
				}
				
				i1 = (i+3)*dLphys;
				i2 = (i+1)*dLphys;
				// fdag(i+1)*nf(i)*f(i)
				if (i==j0) push_corrhop(i1, i2, conj(lambda), tol_OxV, CVERB, H, Psi, states, factors, true); // true=cdag_nc
				#pragma omp critical
				{
					push_corrhopOperator(i2, i1, lambda, H, Res, Fac, false);
				}
				
				// fdag(i-1)*nf(i)*f(i)
				i1 = (i-1)*dLphys;
				i2 = (i+1)*dLphys;
				if (i==j0) push_corrhop(i1, i2, lambda, tol_OxV, CVERB, H, Psi, states, factors, true);
				#pragma omp critical
				{
					push_corrhopOperator(i2, i1, conj(lambda), H, Res, Fac, false);
				}
			}
			
			if (i==j0)
			{
				lout << "compress at j0=i=" << j0 << endl;
				if (states.size() > 0)
				{
					MpsCompressor<MODEL::Symmetry,complex<double>,complex<double>> Compadre(CVERB);
					Compadre.lincomboCompress(states, factors, res[s], Psi, Mlimit, 1e-6, 32);
				}
				else
				{
					lout << "no operator at site " << s << endl;
					res[s] = Psi;
				}
			}
		}
	}
	return make_tuple(res,Res,Fac);
}

complex<double> calc_Joverlap (const MODELC::StateXcd &PhiTt, const vector<MODELC::StateXcd> &Psi, 
                               vector<Mpo<MODELC::Symmetry,complex<double>>> Op, 
                               vector<complex<double>> Fac, 
                               const MODELC &H,
                               string spec, int L, int dLphys, double tol_OxV,
                               double tcc, double tff, double tfc, double Ec, double Ef, double U, int Mlimit)
{
	double resRe = 0.;
	double resIm = 0.;
	
	#pragma omp parallel for collapse(2) reduction(+:resRe,resIm)
	for (int i=0; i<Psi.size(); ++i)
	for (int k=0; k<Op.size(); ++k)
	{
		complex<double> c = Fac[k]*avg(PhiTt,Op[k],Psi[i]);
		resRe += c.real();
		resIm += c.imag();
	}
	complex<double> res = complex<double>(resRe,resIm);
	
	return res;
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
	
	qarray<MODELC::Symmetry::Nq> Q = MODELC::singlet(2*N); // Quantenzahl des Grundzustandes
	lout << "Q=" << Q << endl;
	double U = args.get<double>("U",8.); // U auf den f-Plaetzen
	double V = args.get<double>("V",0.); // V*nc*nf
	double tfc = args.get<double>("tfc",1.); // Hybridisierung fc
	double tcc = args.get<double>("tcc",1.); // Hopping fc
	double tff = args.get<double>("tff",0.); // Hopping ff
	double Retx = args.get<double>("Retx",0.); // Re Hybridisierung f(i)c(i+1)
	double Imtx = args.get<double>("Imtx",0.); // Im Hybridisierung f(i)c(i+1)
	double Rety = args.get<double>("Rety",0.); // Re Hybridisierung c(i)f(i+1)
	double Imty = args.get<double>("Imty",0.); // Im Hybridisierung c(i)f(i+1)
	//double Ec = args.get<double>("Ec",0.); // onsite-Energie fuer c
	//double Ef = args.get<double>("Ef",-4.); // onsite-Energie fuer f
	double Ec = 0.;
	double Ef = -0.5*U;
	
	int j0 = args.get<int>("j0",0);
	assert(j0>=0 and j0<=L/2-1);
	
	size_t Lcell = 2;
	int dLphys = 2;
	
	double dbeta = args.get<double>("dbeta",0.1);
	double beta = args.get<double>("beta",1.);
	double tol_compr_beta = args.get<double>("tol_compr_beta",1e-5);
	
	string spec = args.get<string>("spec","JJC"); // JJC, JJE
	size_t Mstart = args.get<size_t>("Mstart",200ul); // anfaengliche Bonddimension fuer Dynamik
	size_t Mlimit = args.get<size_t>("Mlimit",800ul); // max. Bonddimension fuer Dynamik
	double dt = args.get<double>("dt",0.025);
	double tol_DeltaS = args.get<double>("tol_DeltaS",5e-3);
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
	param_base += make_string("_beta=",beta);
	string base = make_string("j0=",j0,"_L=",L,"_N=",N,"_",param_base); // Dateiname
	string tbase = make_string("tmax=",tmax,"_dt=",dt);
	string wbase = make_string("wmin=",wmin,"_wmax=",wmax,"_wpoints=",wpoints);
	lout << base << endl;
	lout.set(base+".log",wd+"log"); // Log-Datei im Unterordner log
	
	// Parameter des Modells
	vector<Param> params_Tfin;
	// l%4=2 Plaetze sollen f-Plaetze mit U sein:
	params_Tfin.push_back({"U",0.,0}); // c
	params_Tfin.push_back({"U",0.,1}); // bath(c)
	params_Tfin.push_back({"U",U,2}); // f
	params_Tfin.push_back({"U",0.,3}); // bath(f)
	params_Tfin.push_back({"t0",Ec,0}); // c
	params_Tfin.push_back({"t0",0.,1}); // bath(c)
	params_Tfin.push_back({"t0",Ef,2}); // f
	params_Tfin.push_back({"t0",0.,3}); // bath(f)
	// Hopping
	ArrayXXcd tFull = hopping_PAM_T(L,tfc+0.i,tcc+0.i,tff+0.i,Retx+1.i*Imtx,Rety+1.i*Imty,false); // ANCILLA_HOPPING=false
	params_Tfin.push_back({"tFull",tFull});
	params_Tfin.push_back({"maxPower",2ul}); // hoechste Potenz von H
	
	// Parameter fuer die t-Propagation mit beta: Rueckpropagation der Badplaetze
	vector<Param> pparams;
	pparams.push_back({"U",0.,0}); // c
	pparams.push_back({"U",0.,1}); // bath(c)
	pparams.push_back({"U",+U,2}); // f
	pparams.push_back({"U",-U,3}); // bath(f)
	pparams.push_back({"t0",+Ec,0}); // c
	pparams.push_back({"t0",-Ec,1}); // bath(c)
	pparams.push_back({"t0",+Ef,2}); // f
	pparams.push_back({"t0",-Ef,3}); // bath(f)
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
	
	SpectralManager<MODELC> SpecMan({spec},Hp);
	SpecMan.beta_propagation<MODEL>(H_Tfin, H_Tinf, Lcell, dLphys, beta, dbeta, tol_compr_beta, Mlimit, Q, base, false, false, VERB);
	
	lout << endl << "Applying J to ground state for j0 sites..." << endl;
	double tol_OxV = 2.; // val>1 = do not compress
	auto PhiT = SpecMan.get_PhiT();
	auto [Psi, Op, Fac] = apply_J(2*j0, spec, L, dLphys, tol_OxV, Hp, PhiT, tcc, tff, tfc, Ec, Ef-0.5*U, U, Mlimit);
	lout << "Op.size()=" << Op.size() << endl;
	lout << "Fac.size()=" << Fac.size() << endl;
	lout << endl << "Applying J to ground state for j0 sites done!" << endl << endl;
	
	for (int i=0; i<Psi.size(); ++i)
	{
		Psi[i].eps_svd = tol_compr;
		Psi[i].max_Nsv = max(Psi[i].calc_Mmax(),Mstart);
		lout << i << "\t" << Psi[i].info() << endl;
	}
	lout << endl;
	
	vector<TDVPPropagator<MODELC,MODELC::Symmetry,complex<double>,complex<double>,MODELC::StateXcd>> TDVP(2);
	TDVP[0] = TDVPPropagator<MODELC,MODELC::Symmetry,complex<double>,complex<double>,MODELC::StateXcd>(Hp,Psi[0]);
	TDVP[1] = TDVPPropagator<MODELC,MODELC::Symmetry,complex<double>,complex<double>,MODELC::StateXcd>(Hp,PhiT);
	
	int iVERB = L/4;
	
	vector<EntropyObserver<MODELC::StateXcd>> Sobs(2);
	vector<vector<bool>> TWO_SITE(2);
	
	Sobs[0] = EntropyObserver<MODELC::StateXcd>(Hp.length(), Nt, VERB, tol_DeltaS);
	TWO_SITE[0] = Sobs[0].TWO_SITE(0, Psi[0], 1.);
	
	Sobs[1] = EntropyObserver<MODELC::StateXcd>(Hp.length(), Nt, DMRG::VERBOSITY::SILENT, tol_DeltaS);
	TWO_SITE[1] = Sobs[1].TWO_SITE(0, PhiT, 1.);
	
	vector<MatrixXcd> Joverlap(Nt);
	VectorXcd JoverlapSum(Nt);
	
	Stopwatch<> TpropTimer;
	IntervalIterator t(0.,tmax,Nt);
	for (t=t.begin(2); t!=t.end(); ++t)
	{
		lout << "t=" << *t << endl;
		Stopwatch<> StepTimer;
		
		JoverlapSum(t.index()) =  calc_Joverlap(PhiT, Psi, Op, Fac, Hp, spec, L, dLphys, tol_OxV, tcc, tff, tfc, Ec, Ef-0.5*U, U, Mlimit)/(0.5*L);
		
		t << JoverlapSum(t.index());
		lout << "save results at t=" << *t << ", res=" << JoverlapSum(t.index()) << endl;
		t.save(make_string(spec+"t_",base,"_",tbase,".dat"));
	
		if (t.index() != t.end()-1)
		{
			#pragma omp parallel for
			for (int i=0; i<1; ++i)
			{
				if (i==0)
				{
					//-----------------------------------------------------------
					TDVP[i].t_step_adaptive(Hp, Psi[i], -1.i*dt, TWO_SITE[i], 1);
					//-----------------------------------------------------------
					
					lout << "propagated to t=" << *t << endl;
					lout << TDVP[i].info() << endl;
					lout << Psi[i].info() << endl;
					
					if (Psi[i].get_truncWeight().sum() > 0.5*tol_compr)
					{
						Psi[i].max_Nsv = min(static_cast<size_t>(max(Psi[i].max_Nsv*1.1, Psi[i].max_Nsv+50.)),Mlimit);
						if (VERB >= DMRG::VERBOSITY::HALFSWEEPWISE and i==iVERB)
						{
							lout << termcolor::yellow << "i=" << i << ", Setting Psi.max_Nsv to " << Psi[i].max_Nsv << termcolor::reset << endl;
						}
					}
					else
					{
						if (VERB >= DMRG::VERBOSITY::HALFSWEEPWISE and i==iVERB)
						{
							lout << termcolor::green << "trunc_weight=" << Psi[i].get_truncWeight().sum() << " < " << 0.5*tol_compr << " => no bond dimension increase" << termcolor::reset << endl;
						}
					}
				}
				else
				{
					//-----------------------------------------------------------
					TDVP[i].t_step_adaptive(Hp, PhiT, -1.i*dt, TWO_SITE[i], 1);
					//-----------------------------------------------------------
					
					if (PhiT.get_truncWeight().sum() > 0.5*tol_compr)
					{
						PhiT.max_Nsv = min(static_cast<size_t>(max(PhiT.max_Nsv*1.1, PhiT.max_Nsv+50.)),Mlimit);
						if (VERB >= DMRG::VERBOSITY::HALFSWEEPWISE and i==iVERB)
						{
							lout << termcolor::yellow << "i=" << i << ", Setting PhiT.max_Nsv to " << PhiT.max_Nsv << termcolor::reset << endl;
						}
					}
				}
			}
			lout << StepTimer.info("time step") << endl;
			
			#pragma omp parallel for
			for (int i=0; i<1; ++i)
			{
				if (i==0)
				{
					auto PsiTmp = Psi[i]; PsiTmp.entropy_skim();
					TWO_SITE[i] = Sobs[i].TWO_SITE(t.index(), PsiTmp);
				}
				else
				{
					auto PhiTtmp = PhiT; PhiTtmp.entropy_skim();
					TWO_SITE[i] = Sobs[i].TWO_SITE(t.index(), PhiTtmp);
				}
			}
			
			if (VERB >= DMRG::VERBOSITY::HALFSWEEPWISE) lout << StepTimer.info("entropy calculation") << endl;
		}
		
		lout << TpropTimer.info("total running time",false) << endl;
		lout << endl;
	}
	
	lout << "saved to: " << make_string(spec+"t_",base,"_",tbase,".dat") << endl << endl;
	
	VectorXd tvals = t.get_abscissa();
	for (int i=0; i<tvals.rows(); ++i)
	{
		lout << "t=" << tvals(i) << "\t" << JoverlapSum(i) << endl;
	}
	FT_and_save(tvals, tmax, JoverlapSum, wmin, wmax, wpoints, make_string(spec+"w_",base,"_",tbase,"_",wbase,"_DAMPING=GAUSS",".dat"), GAUSS);
	FT_and_save(tvals, tmax, JoverlapSum, wmin, wmax, wpoints, make_string(spec+"w_",base,"_",tbase,"_",wbase,"_DAMPING=LORENTZ",".dat"), LORENTZ);
	FT_and_save(tvals, tmax, JoverlapSum, wmin, wmax, wpoints, make_string(spec+"w_",base,"_",tbase,"_",wbase,"_DAMPING=NO",".dat"), NODAMPING);

	lout << endl << "saved to: " << make_string(spec+"w_",base,"_",tbase,"_",wbase,"_DAMPING=[...].dat") << endl;
}
