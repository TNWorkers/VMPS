#if defined(BLAS) or defined(BLIS) or defined(MKL)
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
#define USING_SU2

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
		OxV_exact(H.cdagc(i,j), target, OxVres, tol_OxV, CVERB, 200, 1);
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
			OxV_exact(H.cdag_nc(i,j), target, OxVres, tol_OxV, CVERB, 200, 1);
		}
		else
		{
			OxV_exact(H.cdagn_c(i,j), target, OxVres, tol_OxV, CVERB, 200, 1);
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
	assert(spec=="JC" or spec=="JE");
	
	vector<MODELC::StateXcd> res(1);
	vector<Mpo<MODELC::Symmetry,complex<double>>> Res;
	vector<complex<double>> Fac;
	
	if (spec == "JC")
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
//					cout << "push: " << i2 << ", " << i1 << ", " << conj(lambda) << endl;
					push_operator(i2, i1, conj(lambda), H, Res, Fac);
				}
				
				i1 = i*dLphys;
				i2 = (i-2)*dLphys;
				// cdag(i)*c(i-1)
				if (i==j0) push_term(i1, i2, conj(lambda), tol_OxV, CVERB, H, Psi, states, factors);
				#pragma omp critical
				{
//					cout << "push: " << i2 << ", " << i1 << ", " << lambda << endl;
					push_operator(i2, i1, lambda, H, Res, Fac);
				}
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
//					cout << "push: " << i2 << ", " << i1 << ", " << conj(lambda) << endl;
					push_operator(i2, i1, conj(lambda), H, Res, Fac);
				}
				
				i1 = (i+1)*dLphys;
				i2 = (i-1)*dLphys;
				// fdag(i)f(i-1)
				if (i==j0) push_term(i1, i2, conj(lambda), tol_OxV, CVERB, H, Psi, states, factors);
				#pragma omp critical
				{
//					cout << "push: " << i2 << ", " << i1 << ", " << lambda << endl;
					push_operator(i2, i1, lambda, H, Res, Fac);
				}
			}
			
			if (i==j0)
			{
				if (states.size() > 0)
				{
					MpsCompressor<MODEL::Symmetry,complex<double>,complex<double>> Compadre(CVERB);
					Compadre.lincomboCompress(states, factors, res[s], Psi, Psi.calc_Mmax(), 50ul, Mlimit, 1e-6, 32);
				}
				else
				{
					lout << "no operator at site " << s << endl;
					res[s] = Psi;
				}
			}
		}
	}
	else if (spec == "JE")
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
					Compadre.lincomboCompress(states, factors, res[s], Psi, Psi.calc_Mmax(), 50ul, Mlimit, 1e-6, 32);
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

complex<double> calc_Joverlap (const MODELC::StateXcd &PhiTt, const vector<MODELC::StateXcd> &Psi, vector<Mpo<MODELC::Symmetry,complex<double>>> Op, vector<complex<double>> Fac)
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

Mpo<MODELC::Symmetry,complex<double>> sum_all_Ops (const vector<Mpo<MODELC::Symmetry,complex<double>>> &Op, const vector<complex<double>> &Fac)
{
	Mpo<MODELC::Symmetry,complex<double>> res = Op[0];
	res.scale(Fac[0]);
	
	for (int i=1; i<Op.size(); ++i)
	{
		auto Optmp = Op[i];
		Optmp.scale(Fac[i]);
		
		res = sum(res,Optmp);
	}
	
	return res;
}

//complex<double> calc_Joverlap (const MODELC::StateXcd &PhiTt, const vector<MODELC::StateXcd> &Psi, Mpo<MODELC::Symmetry,complex<double>> Optot)
//{
//	complex<double> res = 0;
//	
//	for (int i=0; i<Psi.size(); ++i)
//	{
//		res += avg(PhiTt,Optot,Psi[i]);
//	}
//	
//	return res;
//}

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
	double U = args.get<double>("U",4.); // U auf den f-Plaetzen
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
	
	int j0 = args.get<int>("j0",L/4);
	assert(j0>=0 and j0<=L/2-1);
	
	size_t Lcell = 2;
	int dLphys = 2;
	
	double betaswitch = 100.;
	double dbeta = args.get<double>("dbeta",0.1);
	double beta = args.get<double>("beta",1.);
	double tol_compr_beta = args.get<double>("tol_compr_beta",1e-9);
	bool SAVE_BETA = args.get<bool>("SAVE_BETA",true);
	bool LOAD_BETA = args.get<bool>("LOAD_BETA",false);
	size_t maxPower = args.get<int>("maxPower",2ul);
	
	bool CALC_SPEC = args.get<bool>("CALC_SPEC",true);
	
	string spec1 = args.get<string>("spec","JC"); // JC, JE
	string spec2 = spec1;
	string spec = spec1+spec2;
	//size_t Mstart = args.get<size_t>("Mstart",400ul); // anfaengliche Bonddimension fuer Dynamik
	//size_t Mlimit = args.get<size_t>("Mlimit",800ul); // max. Bonddimension fuer Dynamik
	
	size_t Mstart = args.get<size_t>("Mstart",800ul);
	size_t MlimitKet = args.get<size_t>("MlimitKet",4000ul);
	size_t MlimitBra = args.get<size_t>("MlimitBra",1200ul);
	
	double dt = args.get<double>("dt",0.05);
	bool FORW = args.get<bool>("FORW",true);
	bool BACK = args.get<bool>("BACK",true);
	double tmax = args.get<double>("tmax",20.);
	int Nt = tmax/dt+1;
	double tol_DeltaS = args.get<double>("tol_DeltaS",0.); // 5e-3
	double tol_compr_forw = args.get<double>("tol_compr_forw",1e-5);
	double tol_compr_back = args.get<double>("tol_compr_back",1e-4);
	
	double wmin = args.get<double>("wmin",-10.);
	double wmax = args.get<double>("wmax",+10.);
	int wpoints = args.get<int>("wpoints",501);
	int qpoints = args.get<int>("qpoints",501);
	
	// Steuert die Menge der Ausgaben
	DMRG::VERBOSITY::OPTION VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",DMRG::VERBOSITY::HALFSWEEPWISE));
	
	string wd = args.get<string>("wd","./"); correct_foldername(wd); // Arbeitsvereichnis
	string param_base = make_string("tfc=",tfc,"_tcc=",tcc,"_tff=",tff,"_tx=",Retx,",",Imtx,"_ty=",Rety,",",Imty,"_Efc=",Ef,",",Ec,"_U=",U,"_V=",V); // Dateiname
	param_base += make_string("_beta=",beta);
	string base_beta = make_string("L=",L,"_N=",N,"_",param_base);
	string base = make_string("L=",L,"_Ncells=",L/Lcell,"_Lprop=",dLphys*L,"_N=",N,"_",param_base); // Dateiname
	string tbase = make_string("tmax=",tmax,"_dt=",dt,"_DIR=",FORW,",",BACK,"_tol_DeltaS=",tol_DeltaS,"_tol_compr=",tol_compr_forw,",",tol_compr_back);
	string wbase = make_string("wmin=",wmin,"_wmax=",wmax,"_wpoints=",wpoints);
	lout << base << endl;
	lout.set(base+".log",wd+"log"); // Log-Datei im Unterordner log
	
	vector<double> betasave = args.get_list<double>("betasave",{});
	vector<string> betasaveLabels;
	for (int is=0; is<betasave.size(); ++is)
	{
		string param_base_ = make_string("tfc=",tfc,"_tcc=",tcc,"_tff=",tff,"_tx=",Retx,",",Imtx,"_ty=",Rety,",",Imty,"_Efc=",Ef,",",Ec,"_U=",U,"_V=",V);
		param_base_ += make_string("_beta=",betasave[is]);
		string base_ = make_string("L=",L,"_N=",N,"_",param_base_);
		
		betasaveLabels.push_back(base_);
	}
	for (int is=0; is<betasave.size(); ++is)
	{
		lout << "intermediate beta=" << betasave[is] << ", label=" << betasaveLabels[is] << endl;
	}
	bool CALC_C = args.get<bool>("CALC_C",false);
	bool CALC_CHI = args.get<bool>("CALC_CHI",false);
	int Ntaylor = args.get<int>("Ntaylor",0);
	
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
	params_Tfin.push_back({"maxPower",maxPower}); // hoechste Potenz von H
	
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
	pparams.push_back({"maxPower",maxPower});
	
	// Aufbau des Modells bei β=0
	MODEL H_Tinf(dLphys*L,Tinf_params_fermions(Ly));
	lout << endl << "β=0 Entangler " << H_Tinf.info() << endl;
	
	// Modell fuer die β-Propagation
	MODELC H_Tfin(dLphys*L,params_Tfin); H_Tfin.precalc_TwoSiteData();
	lout << endl << "physical Hamiltonian " << H_Tfin.info() << endl << endl;
	
	// Modell fuer die t-propagation
	MODELC Hp(dLphys*L,pparams); Hp.precalc_TwoSiteData();
	lout << endl << "propagation Hamiltonian " << Hp.info() << endl << endl;
	
	SpectralManager<MODELC> SpecMan({spec},Hp); // spec ist Dummy, brauchen nur die beta-Propagation hieraus
	SpecMan.beta_propagation<MODEL>(H_Tfin, H_Tinf, Lcell, dLphys, beta, dbeta, tol_compr_beta, Mstart, Q, log(4), betaswitch, make_string("thermodyn_M=",Mstart), base_beta, LOAD_BETA, SAVE_BETA, VERB, betasave, betasaveLabels, Ntaylor, CALC_C, CALC_CHI);
	
	if (CALC_SPEC)
	{
		Stopwatch<> JappWatch;
		lout << endl << "Applying J to ground state for j0..." << endl;
		double tol_OxV = 2.; // val>1 = do not compress
		auto PhiT = SpecMan.get_PhiT();
		
		/////////////////////
		bool TEST_EIGEN = args.get<bool>("TEST_EIGEN",false);
		if (TEST_EIGEN)
		{
			// test that PhiT is an eigenstate:
			auto avgHp = avg(PhiT, Hp, PhiT);
			auto avgHpHp = (maxPower==2)? avg(PhiT, Hp, PhiT, 2) : avg(PhiT, Hp, Hp, PhiT);
			double test = abs(avgHpHp-avgHp);
			if (test < 1e-4)
			{
				lout << termcolor::green;
			}
			else
			{
				lout << termcolor::red;
			}
			lout << "avgHpHp=" << avgHpHp << endl;
			lout << "avgHp=" << avgHp << ", avgHp^2=" << avgHp*avgHp << endl;
			lout << "eigenstate test for PhiT: " << abs(avgHpHp-avgHp*avgHp) << termcolor::reset << endl;
		}
		/////////////////////
		
		auto [Psi, Op, Fac] = apply_J(2*j0, spec1, L, dLphys, tol_OxV, Hp, PhiT, tcc, tff, tfc, Ec, Ef-0.5*U, U, MlimitBra);
		auto Optot = sum_all_Ops(Op,Fac);
		
		lout << endl;
		lout << "Op.size()=" << Op.size() << endl;
		lout << "Fac.size()=" << Fac.size() << endl;
		lout << "avg<spec>=" << calc_Joverlap(PhiT, {PhiT}, Op, Fac) << endl;
		lout << endl;
		lout << JappWatch.info("Applying J to ground state for j0 done!") << endl << endl;
		
	//	for (int i=0; i<Psi2.size(); ++i)
	//	{
			//Psi2[i].eps_svd = tol_compr;
			//Psi2[i].max_Nsv = max(Psi2[i].calc_Mmax(),Mstart);
			//lout << i << "\t" << Psi2[i].info() << endl;
	//	}
	//	lout << endl;
		
		Psi[0].eps_truncWeight = tol_compr_forw;
		Psi[0].min_Nsv = Psi[0].calc_Mmax();
		Psi[0].max_Nsv = MlimitKet;
		lout << "Psi: " << "min_Nsv=" << Psi[0].min_Nsv << ", max_Nsv=" << Psi[0].max_Nsv << endl;
		lout << Psi[0].info() << endl;
		
		MODEL::StateXcd PhiTtmp;
		HxV(Optot, PhiT, PhiTtmp);
		PhiT = PhiTtmp;
		lout << "Jtot*PhiT=" << PhiT.info() << endl;
		
		PhiT.eps_truncWeight = tol_compr_back;
		PhiT.min_Nsv = min(MlimitBra,PhiT.calc_Mmax());
		PhiT.max_Nsv = MlimitBra;
		lout << "Phi: " << "min_Nsv=" << PhiT.min_Nsv << ", max_Nsv=" << PhiT.max_Nsv << endl;
		lout << PhiT.info() << endl;
		
		TDVPPropagator<MODELC,MODELC::Symmetry,complex<double>,complex<double>,MODELC::StateXcd> TDVP_forw(Hp,Psi[0]);
		TDVPPropagator<MODELC,MODELC::Symmetry,complex<double>,complex<double>,MODELC::StateXcd> TDVP_back(Hp,PhiT);
		
		EntropyObserver<MODELC::StateXcd> Sobs;
		vector<bool> TWO_SITE;
		
		Sobs = EntropyObserver<MODELC::StateXcd>(Hp.length(), Nt, VERB, tol_DeltaS);
		TWO_SITE = Sobs.TWO_SITE(0, Psi[0], 1.);
		
		MatrixXd Joverlap(0,3);
		MatrixXd Stateinfo(0,7);
		double time = 0.;
		int it_forw = 0;
		int it = 0;
		
		//////////// PROPAGATION ////////////
		
		Stopwatch<> TpropTimer;
		
		//for (int it=0; it<Nt; ++it)
		while (time < tmax)
		{
			if (FORW)
			{
				lout << "t=" << time << ", it=" << it << endl;
				
				Joverlap.conservativeResize(Joverlap.rows()+1,Joverlap.cols());
				//complex<double> res = avg(PhiT, Optot, Psi[0]);
				complex<double> res = dot(PhiT,Psi[0]);
				Joverlap(Joverlap.rows()-1,0) = time;
				Joverlap(Joverlap.rows()-1,1) = res.real();
				Joverlap(Joverlap.rows()-1,2) = res.imag();
				
				Stateinfo.conservativeResize(Stateinfo.rows()+1,Stateinfo.cols());
				Stateinfo(Stateinfo.rows()-1,0) = time;
				Stateinfo(Stateinfo.rows()-1,1) = Psi[0].calc_Mmax();
				Stateinfo(Stateinfo.rows()-1,2) = Psi[0].calc_fullMmax();
				Stateinfo(Stateinfo.rows()-1,3) = PhiT.calc_Mmax();
				Stateinfo(Stateinfo.rows()-1,4) = PhiT.calc_fullMmax();
				Stateinfo(Stateinfo.rows()-1,5) = Psi[0].get_truncWeight().maxCoeff();
				Stateinfo(Stateinfo.rows()-1,6) = PhiT.get_truncWeight().maxCoeff();
				
				lout << "save results at t=" << time << ", res=" << res << endl;
				
				saveMatrix(Joverlap,make_string(wd,spec+"t_",base,"_",tbase,".dat"));
				saveMatrix(Stateinfo,make_string(wd,"StateInfo",spec+"t_",base,"_",tbase,".dat"));
				
				Stopwatch<> StepTimer;
				
				//-----------------------------------------------------------
				if (tol_DeltaS == 0.)
				{
					TDVP_forw.t_step(Hp, Psi[0], -1.i*dt, 1);
				}
				else
				{
					TDVP_forw.t_step_adaptive(Hp, Psi[0], -1.i*dt, TWO_SITE, 1);
				}
				it += 1;
				it_forw += 1;
				//-----------------------------------------------------------
				
				lout << StepTimer.info("time ket") << endl;
				lout << "ket: " << TDVP_forw.info() << endl;
				lout << "ket: " << Psi[0].info() << endl;
				
				time += dt;
				
				if (tol_DeltaS > 0.)
				{
					auto PsiTmp = Psi[0]; PsiTmp.entropy_skim();
					lout << "ket: ";
					TWO_SITE = Sobs.TWO_SITE(it_forw, PsiTmp);
				}
				
				lout << "propagated to t=" << time  << endl;
				
				lout << TpropTimer.info("total time",false) << endl;
				lout << endl;
			}
			
			if (BACK)
			{
				lout << "t=" << time << endl;
				
				Joverlap.conservativeResize(Joverlap.rows()+1,Joverlap.cols());
				complex<double> res = dot(PhiT,Psi[0]); //calc_Joverlap(PhiT, Psi, Optot);
				Joverlap(Joverlap.rows()-1,0) = time;
				Joverlap(Joverlap.rows()-1,1) = res.real();
				Joverlap(Joverlap.rows()-1,2) = res.imag();
				
				Stateinfo.conservativeResize(Stateinfo.rows()+1,Stateinfo.cols());
				Stateinfo(Stateinfo.rows()-1,0) = time;
				Stateinfo(Stateinfo.rows()-1,1) = Psi[0].calc_Mmax();
				Stateinfo(Stateinfo.rows()-1,2) = Psi[0].calc_fullMmax();
				Stateinfo(Stateinfo.rows()-1,3) = PhiT.calc_Mmax();
				Stateinfo(Stateinfo.rows()-1,4) = PhiT.calc_fullMmax();
				Stateinfo(Stateinfo.rows()-1,5) = Psi[0].get_truncWeight().maxCoeff();
				Stateinfo(Stateinfo.rows()-1,6) = PhiT.get_truncWeight().maxCoeff();
				
				lout << "save results at t=" << time << ", res=" << res << endl;
				
				saveMatrix(Joverlap,make_string(wd,spec+"t_",base,"_",tbase,".dat"));
				saveMatrix(Stateinfo,make_string(wd,"StateInfo",spec+"t_",base,"_",tbase,".dat"));
				
				Stopwatch<> StepTimer;
				
				//-----------------------------------------------------------
				TDVP_back.t_step(Hp, PhiT, +1.i*dt, 1);
				it += 1;
				//-----------------------------------------------------------
				
				lout << StepTimer.info("time bra") << endl;
				lout << "bra: " << TDVP_back.info() << endl;
				lout << "bra: " << PhiT.info() << endl;
				
				time += dt;
				
				lout << "propagated to t=" << time  << endl;
				
				lout << TpropTimer.info("running time",false) << endl;
				lout << endl;
			}
		}
		
//		int it = 0;
//		Stopwatch<> TpropTimer;
//		if (dt_forw != 0.)
//		{
//			while (Psi[0].calc_Mmax() < MlimitKet or it<=10)
//			{
//				lout << "t=" << time << ", it=" << it << endl;
//				
//				Joverlap.conservativeResize(Joverlap.rows()+1,Joverlap.cols());
//				//complex<double> res = calc_Joverlap(PhiT, Psi, Op, Fac);
//				//complex<double> res = calc_Joverlap(PhiT, Psi, Optot);
//				complex<double> res = avg(PhiT, Optot, Psi[0]);
//				Joverlap(Joverlap.rows()-1,0) = time;
//				Joverlap(Joverlap.rows()-1,1) = res.real();
//				Joverlap(Joverlap.rows()-1,2) = res.imag();
//				
//				lout << "save results at t=" << time << ", res=" << res << endl;
//				
//				saveMatrix(Joverlap,make_string(wd,spec+"t_",base,"_",tbase,".dat"));
//				
//				Stopwatch<> StepTimer;
//				
//				//-----------------------------------------------------------
//				if (tol_DeltaS == 0.)
//				{
//					TDVP.t_step(Hp, Psi[0], -1.i*dt_forw, 1);
//				}
//				else
//				{
//					TDVP.t_step_adaptive(Hp, Psi[0], -1.i*dt_forw, TWO_SITE, 1);
//				}
//				it += 1;
//				//if (Psi[0].calc_Mmax() >= Mstart) Psi[0].eps_truncWeight = tol_compr_forw;
//				//-----------------------------------------------------------
//				
//				lout << StepTimer.info("time ket") << endl;
//				lout << "ket: " << TDVP.info() << endl;
//				lout << "ket: " << Psi[0].info() << endl;
//				
//				time += dt_forw;
//				
//				if (tol_DeltaS > 0.)
//				{
//					auto PsiTmp = Psi[0]; PsiTmp.entropy_skim();
//					lout << "ket: ";
//					TWO_SITE = Sobs.TWO_SITE(it, PsiTmp);
//				}
//				
//				lout << "propagated to t=" << time  << endl;
//				
//				lout << TpropTimer.info("total time",false) << endl;
//				lout << endl;
//				
//				if (time > tmax) break;
//			}
//		}
//		
//		lout << endl;
//		lout << termcolor::blue << "End of ket propagation!" << termcolor::reset << endl;
//		lout << Psi[0].info() << endl;
//		lout << endl;
//		
//		if (dt_back!=0.)
//		{
//			MODEL::StateXcd PhiTtmp;
//			HxV(Optot, PhiT, PhiTtmp);
//			PhiT = PhiTtmp;
//			lout << "Jtot*PhiT=" << PhiT.info() << endl;
//			
//			PhiT.eps_truncWeight = tol_compr_back; //min(1e-9,tol_compr_back);
//			PhiT.min_Nsv = PhiT.calc_Mmax();
//			//Mstart_t = PhiT.calc_Mmax();
//			PhiT.max_Nsv = MlimitBra;
//			lout << "min_Nsv=" << PhiT.min_Nsv << ", max_Nsv=" << PhiT.max_Nsv << endl;
//			lout << PhiT.info() << endl;
//			
//			it = 0;
//			TDVP = TDVPPropagator<MODELC,MODELC::Symmetry,complex<double>,complex<double>,MODELC::StateXcd>(Hp,PhiT);
//			
//			while (PhiT.calc_Mmax() < MlimitBra or it<=10)
//			{
//				lout << "t=" << time << endl;
//				
//				Joverlap.conservativeResize(Joverlap.rows()+1,Joverlap.cols());
//				Joverlap(Joverlap.rows()-1,0) = time;
//				complex<double> res = dot(PhiT,Psi[0]); //calc_Joverlap(PhiT, Psi, Optot);
//				Joverlap(Joverlap.rows()-1,0) = time;
//				Joverlap(Joverlap.rows()-1,1) = res.real();
//				Joverlap(Joverlap.rows()-1,2) = res.imag();
//				
//				lout << "save results at t=" << time << ", res=" << res << endl;
//				
//				saveMatrix(Joverlap,make_string(wd,spec+"t_",base,"_",tbase,".dat"));
//				
//				Stopwatch<> StepTimer;
//				
//				//-----------------------------------------------------------
//				TDVP.t_step(Hp, PhiT, +1.i*dt_back, 1);
//				it += 1;
//				//if (PhiT.calc_Mmax() >= Mstart) PhiT.eps_truncWeight = tol_compr_back;
//				//-----------------------------------------------------------
//				
//				lout << StepTimer.info("time bra") << endl;
//				lout << "bra: " << TDVP.info() << endl;
//				lout << "bra: " << PhiT.info() << endl;
//				
//				time += dt_back;
//				
//				lout << "propagated to t=" << time  << endl;
//				
//				lout << TpropTimer.info("running time",false) << endl;
//				lout << endl;
//				
//				if (time > tmax) break;
//			}
//		}
		
		lout << "saved to: " << make_string(wd,spec+"t_",base,"_",tbase,".dat") << endl << endl;
	}
}
