#ifdef BLAS
#include "util/LapackManager.h"
#pragma message("LapackManager")
#endif

#include <iostream>
#include <fstream>
#include <complex>

#include "Logger.h"
Logger lout;
#include "ArgParser.h"

#include "GrandSpinfulFermions.h"
#include "LanczosWrappers.h"
#include "LanczosSolver.h"
#include "Photo.h"
#include "IntervalIterator.h"

#include "ParamHandler.h"

#include <boost/math/quadrature/ooura_fourier_integrals.hpp>
#include "InterpolGSL.h"

#include "models/ParamCollection.h"
//#include "OrthPolyGreen.h" // for Chebyshev
#include "EigenFiles.h"
#include "LanczosPropagator.h"

typedef ED::GrandSpinfulFermions MODEL;

using namespace std;

// For Chebyshev:
//double dot_green (const VectorXd &V1, const VectorXd &V2)
//{
//	return V1.dot(V2);
//}

enum DAMPING {GAUSS, LORENTZ, NODAMPING};

// continuous Fourier transform using Ooura integration, with t=tvals, and a complex column of data
// wmin: minimal frequency
// wmax: maximal frequency
// wpoints: number of frequency points
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

VectorXcd get_G_timeslice (int i0, int j0, const vector<MatrixXcd> &G)
{
	VectorXcd res(G.size());
	for (int it=0; it<G.size(); ++it)
	{
		res[it] = G[it](i0,j0);
	}
	return res;
}

int main (int argc, char* argv[]) 
{
	ArgParser args(argc,argv);
	size_t L = args.get<size_t>("L",12);
	double U = args.get<double>("U",0.);
	double mu = args.get<double>("mu",0.5*U);
	int j0 = args.get<int>("j0",L/2);
	string MOL = args.get<string>("MOL","RING");
	
	double wmin = args.get<double>("wmin",-20.);
	double wmax = args.get<double>("wmin",+20.);
	int wpoints = args.get<int>("wpoints",501);
	
	double tmax = args.get<double>("tmax",8.);
	double dt = args.get<double>("dt",0.1);
	int tpoints = tmax/dt+1;
	
	string spec = args.get<string>("spec","PES");
	assert(spec=="PES" or spec=="IPE");
	double tsign = (spec=="PES")? -1.:+1.;
	string wd = args.get<string>("wd","./"); correct_foldername(wd);
	string base = make_string("MOL=",MOL,"_L=",L,"_U=",U,"_mu=",mu);
	
	ArrayXXd tFull;
	if (MOL == "RING") // for testing
	{
		tFull = create_1D_PBC(L);
	}
	else if (MOL == "C12")
	{
		tFull = hopping_fullerene(L);
	}
	lout << "tFull=" << endl << tFull << endl;
	vector<Param> params;
	params.push_back({"tFull",tFull});
	params.push_back({"U",U});
	params.push_back({"mu",mu});
	
	MODEL Hket(L,params,ED::N_EVN_M0);
	lout << "Hket=" << endl << Hket.info() << endl;
	
	MODEL Hbra(L,params, (spec=="PES")? ED::N_ODD_MM1 : ED::N_ODD_MP1);
	lout << "Hbra=" << endl << Hbra.info() << endl;
	
	//-------------Get groundstate in the ket space (for Chebyshev: also the ground & roof state in the bra space)-------------
	Stopwatch<> Timer;
	Eigenstate<VectorXd> g;
	double Emin, Emax;
	{
		Eigenstate<VectorXd> min, max;
//		LanczosSolver<MODEL,VectorXd,double> Lucy(LANCZOS::REORTHO::FULL);
//		LanczosSolver<MODEL,VectorXd,double> Lena(LANCZOS::REORTHO::FULL);
		LanczosSolver<MODEL,VectorXd,double> Lutz(LANCZOS::REORTHO::FULL);
		//#pragma omp parallel sections
		{
//			#pragma omp section
//			{
//				Lucy.ground(Hbra,min,1e-7,1e-4);
//				#pragma omp critical
//				{
//					lout << "Emin done!" << endl;
//				}
//			}
//			#pragma omp section
//			{
//				Lena.roof(Hbra,max,1e-7,1e-4);
//				#pragma omp critical
//				{
//					lout << "Emax done!" << endl;
//				}
//			}
//			#pragma omp section
			{
				Lutz.ground(Hket,g,1e-7,1e-4);
				//#pragma omp critical
				{
					lout << "E0 done!" << endl;
				}
			}
		}
		Emin = min.energy;
		Emax = max.energy;
//		lout << Lucy.info() << endl;
//		lout << Lena.info() << endl;
		lout << Lutz.info() << endl;
	}
	lout << Timer.info("ground state") << endl;
	lout << "Emin=" << Emin << ", Emax=" << Emax << ", E0=" << g.energy << endl;
	
	// list of site indices
	vector<int> jlist = {j0};
	
	// Phi will become the bra state
	vector<VectorXcd> Phi(L);
	for (int i=0; i<L; ++i)
	{
		ED::Photo Ph;
		if (spec=="PES")
		{
			Ph = ED::Photo(i,UP,ANNIHILATE,Hbra,Hket); // annihilates UP electron
		}
		else if (spec == "IPE")
		{
			Ph = ED::Photo(i,UP,CREATE,Hbra,Hket); // creates UP electron
		}
		VectorXd Vtmp;
		OxV(Ph,g.state,Vtmp);
		lout << "i=" << i << ", dot=" << Vtmp.dot(Vtmp) << endl;
		Phi[i] = Vtmp.cast<complex<double> >();
	}
	
	// Psi is the ket state that will be propagated
	vector<VectorXcd> Psi(jlist.size());
	for (int j=0; j<jlist.size(); ++j)
	{
		Psi[j] = Phi[jlist[j]];
	}
	
	// Green's function
	vector<MatrixXcd> G(tpoints);
	for (int it=0; it<tpoints; ++it)
	{
		G[it].resize(L,L);
		G[it].setZero();
	}
	
	// Create the propagator
	vector<LanczosPropagator<MODEL,VectorXcd,complex<double> > > Lutz(jlist.size());
	for (int j=0; j<jlist.size(); ++j)
	{
		Lutz[j] = LanczosPropagator<MODEL,VectorXcd,complex<double> >(1e-5);
	}
	
	IntervalIterator t(0., tmax, tpoints);
	for (t=t.begin(); t!=t.end(); ++t)
	{
		for (int i=0; i<L; ++i)
		for (int j=0; j<jlist.size(); ++j)
		{
			G[t.index()](i,jlist[j]) = -1.i * exp(+1.i*tsign*g.energy*(*t)) * Phi[i].dot(Psi[j]);
		}
		
		Stopwatch<> Timer;
		#pragma omp parallel for // parallelize all propagations?
		for (int j=0; j<jlist.size(); ++j)
		{
			Lutz[j].t_step(Hbra, Psi[j], -1.i*tsign*dt);
		}
		
		lout << Timer.info("timestep") << endl;
		for (int j=0; j<jlist.size(); ++j)
		{
			lout << *t << "\t" << Lutz[j].info() << endl;
		}
	}
	
	VectorXd tvals = t.get_abscissa();
	// Fourier transform and save to disk
	string filename = make_string(spec+"_i=",j0,"_j=",j0,"_",base,"_DAMPING=LORENTZ",".dat");
	lout << "save result to: " << filename << endl;
	FT_and_save(tvals, tmax, get_G_timeslice(jlist[0],jlist[0],G), wmin, wmax, wpoints, filename, LORENTZ);
	
//	
//	bool USE_IDENTITIES = args.get<bool>("USE_IDENTITIES",true);
//	vector<double> dE = args.get_list<double>("dE",{0.2});
//	sort(dE.rbegin(), dE.rend());
//	vector<int> Msave;
//	int Mmax;
//	double spillage = args.get<double>("spillage",10.);
//	bool VERBOSE = true;
//	
//	OrthPolyGreen<MODEL,VectorXd,double,CHEBYSHEV> KPS(Emin-spillage,Emax+spillage,VERBOSE);
//	
//	for (size_t i=0; i<dE.size(); ++i)
//	{
//		int Mval = (KPS.get_Emax()-KPS.get_Emin()+2.*spillage)/dE[i];
//		if (USE_IDENTITIES and Mval%2==1) Mval += 1;
//		Msave.push_back(Mval);
//		lout << "dE=" << dE[i] << " => M=" << Msave[Msave.size()-1] << endl;
//	}
//	
//	Mmax = args.get<int>("Mmax",*max_element(Msave.begin(),Msave.end()));
//	lout << KPS.info() << endl;
//	
//	string momfile, datfile;
//	
//	for (int i=0; i<Msave.size(); ++i)
//	{
//		momfile = make_string("moments/","moments.dat");
//		datfile = make_string("spec.dat");
//		
//		if (spec == "PES")
//		{
//			ArrayXd Eoffset(1); Eoffset << KPS.get_Emax();
//			KPS.add_savepoint(Msave[i], wd+momfile, wd+datfile, Eoffset, true, LORENTZ); // REVERSE = true
//		}
//		else if (spec == "IPE")
//		{
//			ArrayXd Eoffset(1); Eoffset << KPS.get_Emin();
//			KPS.add_savepoint(Msave[i], wd+momfile, wd+datfile, Eoffset, false, LORENTZ); // REVERSE = false
//		}
//	}
//	
//	KPS.calc_ImAB(Hbra, initA, initA[i0], Mmax);
//	
//	MatrixXcd G(wpoints,L);
//	G.setZero();
//	IntervalIterator iw(wmin,wmax,wpoints);
//	for (iw=iw.begin(); iw<iw.end(); ++iw)
//	for (int l=0; l<L; ++l)
//	{
//		double w = *iw;
//		double Re =         KPS.evaluate_ReGAB(l,w,-1,KPS.get_Emax(),true,LORENTZ);
//		double Im = -M_PI * KPS.evaluate_ImAB (l,w,-1,KPS.get_Emax(),true,LORENTZ);
//		G(iw.index(),l) = complex<double>(Re,Im);
//	}
//	
//	MatrixXd DOS(wpoints,2);
//	for (iw=iw.begin(); iw<iw.end(); ++iw)
//	{
//		DOS(iw.index(),0) = *iw;
//		DOS(iw.index(),1) = -M_1_PI * G(iw.index(),i0).imag();
//	}
//	
//	saveMatrix(DOS,"DOS.dat");
	
	
}
