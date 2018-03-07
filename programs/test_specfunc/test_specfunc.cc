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
//#include "IntervalIterator.h"
#include "OrthPolyGreen.h"

//#include "InterpolGSL.h"

size_t L;
int N;
double U, V;
vector<int> Msave;
int Mmax;
string spec, wd, outfile, Efilename;
vector<double> dE;
double Emin, Emax, E0;
double d, n_sig;

OrthPolyGreen<MODEL,MODEL::StateXd> * KPS;

int main (int argc, char* argv[]) 
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L");
	N = args.get<size_t>("N",L);
	qarray<2> Qi = MODEL::singlet(N);
	qarray<2> Qc;
	spec = args.get<string>("spec","AES");
	U = args.get<double>("U",6.);
	V = args.get<double>("V",0.);
	wd = args.get<string>("wd","./");
	if (wd.back() != '/') {wd += "/";}
	
	dE = args.get_list<double>("dE",{0.2});
	outfile = make_string(spec,"_L=",L,"_N=",N,"_U=",U);
	if (V != 0.) {outfile += make_string("_V=",V);}
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
	MODEL H(L,{{"U",U},{"V",V}});
	lout << H.info() << endl << endl;
	
	MODEL::Operator A, Adag;
	if (spec == "AES")
	{
		A = H.cc(L/2);
		Adag = H.cdagcdag(L/2);
		Qc = MODEL::singlet(N-2);
	}
	else if (spec == "APS")
	{
		A = H.cdagcdag(L/2);
		Adag = H.cc(L/2);
		Qc = MODEL::singlet(N+2);
	}
//	else if (spec == "PES")
//	{
//		Nc = qarray<2>({Nupdn-1,Nupdn});
//	}
//	else if (spec == "IPES")
//	{
//		Nc = qarray<2>({Nupdn+1,Nupdn});
//	}
	else if (spec == "CSF")
	{
		A = H.n(L/2);
		Adag = A;
		Qc = Qi;
	}
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
			
			DMRG->edgeState(H, *init, Qi, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::NORM_TEST, 1e-7,1e-6, 4,500, 50,10);
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
			gDMRG->edgeState(H, *g, Qc, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::NORM_TEST, 1e-8,1e-6, 8,500, 50,10);
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
			
			rDMRG->edgeState(H, *r, Qc, LANCZOS::EDGE::ROOF, LANCZOS::CONVTEST::NORM_TEST, 1e-8,1e-6, 8,500, 50,10);
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
	Compadre.varCompress(A, Adag, init->state, initA, Qc, init->state.calc_Dmax());
	delete init;
	initA.eps_svd = 1e-7;
	cout << "AxV:" << endl << initA.info() << endl;
	//--------------</A*init>---------------
	
//	auto Psi = init->state;
////	MODEL::StateXd rand = Psi; rand.setRandom();
////	Psi += 0.1 * rand;
//	MODEL::CompressorXd Compadrino(DMRG::VERBOSITY::HALFSWEEPWISE);
//	MODEL::StateXd Phi;
//	Compadrino.varCompress(Psi,Phi,2);
//	cout << endl;
//	cout << "ORIGINAL:" << endl;
//	cout << Psi.info() << endl;
//	cout << "COMPRESSED:" << endl;
//	cout << Phi.info() << endl;
	
	//--------------<KernelPolynomialSolver>---------------
	double spillage = 0.;
	if (spec == "PES" or spec == "IPES")
	{
		spillage = 4.*dE[0];
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
	//--------------</KernelPolynomialSolver>---------------
	
	delete KPS;	
}
