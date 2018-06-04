#include <iostream>
#include <fstream>
#include <complex>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

#include "Stopwatch.h"
#include "PolychromaticConsole.h"

#include "solvers/DmrgSolver.h"
#include "solvers/MpsCompressor.h"
#include "DmrgLinearAlgebra.h"

#include "ArgParser.h"
#include "OrthPolyGreen.h"

size_t L;
int N;
double U, J;
vector<int> Msave;
int Mmax;
string spec, wd, outfile, Efilename;
vector<double> dE;
double Emin, Emax, E0;
double d, n_sig;
bool CHEB;

OrthPolyGreen<MODEL,MODEL::StateXd> * KPS;

int main (int argc, char* argv[]) 
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L");
	N = args.get<size_t>("N",0);
	qarray<2> Qi = MODEL::polaron(L,N);
	qarray<2> Qc;
	spec = args.get<string>("spec","IPES");
	U = args.get<double>("U",0.);
	J = args.get<double>("J",-1.);
	CHEB = args.get<bool>("CHEB",true);
	wd = args.get<string>("wd","./");
	if (wd.back() != '/') {wd += "/";}
	
	dE = args.get_list<double>("dE",{1.,0.5,0.2});
	outfile = make_string(spec,"_L=",L,"_N=",N,"_J=",J,"_U=",U);
	Efilename = outfile;
	outfile += "_dE=";
	lout.set(outfile+str(dE)+".log",wd+"log");
	lout << args.info() << endl;
	
	lout << outfile << endl;
	
	#ifdef _OPENMP
	lout << "threads=" << omp_get_max_threads() << endl;
	#else
	lout << "not parallelized" << endl;
	#endif
	
	//--------------<Hamiltonian & transition operator>---------------
	MODEL H(L,{{"U",U},{"J",J},{"CALC_SQUARE",true}});
	lout << H.info() << endl << endl;
	
	MODEL::Operator A, Adag;
	if (spec == "AES")
	{
		A = H.cc(L/2);
		Adag = H.cdagcdag(L/2);
		Qc = MODEL::polaron(L,N-2);
	}
	else if (spec == "APS")
	{
		A = H.cdagcdag(L/2);
		Adag = H.cc(L/2);
		Qc = MODEL::polaron(L,N+2);
	}
	if (spec == "PES")
	{
		#ifdef USING_SU2
		{
			A = H.c(L/2, 0, 1.);
			Adag = H.cdag(L/2, 0, sqrt(2.));
			Qc = qarray<2>({L+2,N-1});
		}
		#else
		{
			A = H.c(DN,L/2);
			Adag = H.cdag(DN,L/2);
			Qc = Qi+A.Qtarget();
		}
		#endif
	}
	else if (spec == "IPES")
	{
		#ifdef USING_SU2
		{
			A = H.cdag(L/2, 0, 1.);
			Adag = H.c(L/2, 0, sqrt(2.));
			Qc = qarray<2>({L,N+1});
		}
		#else
		{
			A = H.cdag(DN, L/2);
			Adag = H.c(DN, L/2);
			Qc = Qi+A.Qtarget();
		}
		#endif
	}
	else if (spec == "CSF")
	{
		A = H.n(L/2);
		Adag = A;
		Qc = Qi;
	}
//	else if (spec == "SSF")
//	{
//		if constexpr (MODEL::Symmetry::NON_ABELIAN)
//		{
//			A = H.S(L/2);
//			Adag = H.Sdag(L/2);
//			Qc = {3,N};
//		}
////		else
////		{
////			A = H.Sz(L/2);
////			Adag = H.Sz(L/2);
////			Qc = Qi;
////		}
//	}
	lout << A.info() << endl;
	lout << Adag.info() << endl;
	//--------------</Hamiltonian & transition operator>---------------
	
	//--------------<ground state>---------------
	Eigenstate<MODEL::StateXd> * init = new Eigenstate<MODEL::StateXd>;
	
	Stopwatch<> Chronos;
	stringstream ginfo, rinfo;
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			Stopwatch<> nChronos;
			MODEL::Solver * DMRG = new MODEL::Solver(DMRG::VERBOSITY::HALFSWEEPWISE);
			
			DMRG->edgeState(H, *init, Qi, LANCZOS::EDGE::GROUND);
			lout << endl << nChronos.info(make_string("ground state ",Qi)) << endl;
			lout << DMRG->info() << endl;
			E0 = init->energy;
			delete DMRG;
		}
		#pragma omp section
		{
			Stopwatch<> gChronos;
			Eigenstate<MODEL::StateXd> * g = new Eigenstate<MODEL::StateXd>;
			
			MODEL::Solver * gDMRG = new MODEL::Solver(DMRG::VERBOSITY::SILENT);
			gDMRG->edgeState(H, *g, Qc, LANCZOS::EDGE::GROUND);
			ginfo << gChronos.info(make_string("ground state ",Qc)) << endl;
			Emin = g->energy;
			ginfo << gDMRG->info() << endl;
			delete gDMRG;
			delete g;
		}
		
		#pragma omp section
		{
			Stopwatch<> rChronos;
			Eigenstate<MODEL::StateXd> * r = new Eigenstate<MODEL::StateXd>;
			MODEL::Solver * rDMRG = new MODEL::Solver(DMRG::VERBOSITY::SILENT);
			
			rDMRG->edgeState(H, *r, Qc, LANCZOS::EDGE::ROOF);
			rinfo << rChronos.info(make_string("roof state ",Qc)) << endl;
			Emax = r->energy;
			rinfo << rDMRG->info() << endl;
			delete rDMRG;
			delete r;
		}
	}
	
	lout << ginfo.str() << rinfo.str() << endl;
	lout << Chronos.info("all edge states") << endl;
	lout << "E0=" << E0 << ", Emin=" << Emin << ", Emax=" << Emax << endl << endl;
	
	// save energies
	ofstream Efile(make_string(wd+"energies/"+Efilename,".dat"));
	Efile << E0 << endl;
	Efile << Emin << endl;
	Efile << Emax << endl;
	Efile.close();
	
	//--------------</ground state>---------------
	
	//--------------<A*init>---------------
	MODEL::StateXd initA;
	initA.eps_svd = 1e-15;
//	OxV(O, init->state, initA);
	MODEL::CompressorXd Compadre(DMRG::VERBOSITY::HALFSWEEPWISE);
	Compadre.prodCompress(A, Adag, init->state, initA, Qc, init->state.calc_Dmax());
//	cout << "c*cdag 1=" << avg(init->state, H.c(L/2), H.cdag(L/2), init->state) << endl;
//	cout << "c*cdag 2=" << avg(init->state, H.ccdag(L/2,L/2), init->state) << endl;
//	cout << "avg1=" << avg(init->state, H.cdag(L/2), H.c(L/2), init->state) << endl;
//	cout << "avg2=" << avg(init->state, H.cdagc(L/2,L/2), init->state) << endl;
//	cout << "avg3=" << avg(init->state, H.n(L/2), init->state) << endl;
	delete init;
	initA.eps_svd = 1e-7;
	cout << "AxV:" << endl << initA.info() << endl;
	initA.graph("initA");
	//--------------</A*init>---------------
	
	//--------------<KernelPolynomialSolver>---------------
	if (CHEB)
	{
		double spillage = 0.;
		if (spec == "PES" or spec == "IPES")
		{
			spillage = 6.*dE[0];
		}
		else if (spec == "SSF")
		{
			spillage = 0.5*(Emax-Emin);
		}
		KPS = new OrthPolyGreen<MODEL,MODEL::StateXd,CHEBYSHEV>(Emin-spillage, Emax+spillage);
		
		for (size_t i=0; i<dE.size(); ++i)
		{
			if (i>0) {assert(dE[i] < dE[i-1]);} // monotoncally decreasing resolution
			Msave.push_back((Emax-Emin+2.*spillage)/dE[i]);
			lout << "dE=" << dE[i] << " => M=" << Msave[Msave.size()-1] << endl;
		}
		lout << endl;
		
		Mmax = args.get<int>("Mmax",*max_element(Msave.begin(),Msave.end()));
		lout << KPS->info() << endl;
		
		string momfile = make_string(wd+"moments/"+outfile,str(dE),".dat");
		for (int i=0; i<Msave.size(); ++i)
		{
			string datfileJ = make_string(wd+outfile,make_string(dE[i]),".dat");
			string datfileL = make_string(wd+"Lorentz/"+outfile,make_string(dE[i]),".dat");
		
			if (spec == "AES" or spec == "PES")
			{
				KPS->add_savepoint(Msave[i], momfile, datfileJ, Emax, true, JACKSON);
				KPS->add_savepoint(Msave[i], momfile, datfileL, Emax, true, LORENTZ);
			}
			else
			{
				KPS->add_savepoint(Msave[i], momfile, datfileJ, Emin, false, JACKSON);
				KPS->add_savepoint(Msave[i], momfile, datfileL, Emin, false, LORENTZ);
			}
		}
		
		KPS->calc_ImAA(H,initA,Mmax,false);
		lout << "Chebyshev iteration done!" << endl;
	}
	//--------------</KernelPolynomialSolver>---------------
	
	delete KPS;
}
