#define DONT_USE_LAPACK_SVD
#define DONT_USE_LAPACK_QR
#define DMRG_DONT_USE_OPENMP
#define EIGEN_DONT_PARALLELIZE

#include <iostream>
#include <fstream>
#include <complex>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

#include "Logger.h"
Logger lout;

#include "Stopwatch.h"
#include "PolychromaticConsole.h"

#include "solvers/DmrgSolver.h"
#include "solvers/MpsCompressor.h"
#include "models/HubbardU1xU1.h"
#include "DmrgLinearAlgebra.h"

#include "ArgParser.h"
//#include "IntervalIterator.h"
#include "OrthPolyGreen.h"

//#include "InterpolGSL.h"

size_t L;
int Nupdn;
double U, V;
vector<int> Msave;
int Mmax;
string spec, wd, outfile, Efilename;
vector<double> dE;
double Emin, Emax, E0;
double d, n_sig;

OrthPolyGreen<VMPS::HubbardU1xU1,VMPS::HubbardU1xU1::StateXd> * KPS;

int main (int argc, char* argv[]) 
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L");
	Nupdn = args.get<int>("Nupdn",L/2);
	qarray<2> Narray = qarray<2>({Nupdn,Nupdn});
	spec = args.get<string>("spec","AES");
	qarray<2> Nc;
	if (spec == "AES")
	{
		Nc = qarray<2>({Nupdn-1,Nupdn-1});
	}
	else if (spec == "APS")
	{
		Nc = qarray<2>({Nupdn+1,Nupdn+1});
	}
	else if (spec == "PES")
	{
		Nc = qarray<2>({Nupdn-1,Nupdn});
	}
	else if (spec == "IPES")
	{
		Nc = qarray<2>({Nupdn+1,Nupdn});
	}
	else if (spec == "SSF")
	{
		Nc = qarray<2>({Nupdn,Nupdn});
	}
	U = args.get<double>("U",6.);
	V = args.get<double>("V",0.);
	wd = args.get<string>("wd","./");
	if (wd.back() != '/') {wd += "/";}
	
	dE = args.get_list<double>("dE",{0.2});
	outfile = make_string(spec,"_L=",L,"_Nupdn=",Nupdn,"_U=",U);
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
	
	//--------------<Hamiltonians & Auger>---------------
	VMPS::HubbardU1xU1 H(L,{{"U",U},{"V",V}});
	lout << H.info() << endl << endl;
	//--------------</Hamiltonians & Auger>---------------
	
	//--------------<ground state>---------------
	Eigenstate<VMPS::HubbardU1xU1::StateXd> * init = new Eigenstate<VMPS::HubbardU1xU1::StateXd>;
	
	Stopwatch<> Chronos;
	stringstream ginfo, rinfo;
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			Stopwatch<> nChronos;
			VMPS::HubbardU1xU1::Solver * DMRG = new VMPS::HubbardU1xU1::Solver(DMRG::VERBOSITY::HALFSWEEPWISE);
			
			DMRG->edgeState(H, *init, Narray, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::SQ_TEST, 1e-7,1e-6, 4,500, 50,6);
			lout << endl << nChronos.info(make_string("ground state ",Narray)) << endl;
			lout << DMRG->info() << endl;
			E0 = init->energy;
			d = avg(init->state, H.d(L/2), init->state);
			n_sig = avg(init->state, H.n(UP,L/2), init->state);
			delete DMRG;
		}
		#pragma omp section
		{
			Stopwatch<> gChronos;
			Eigenstate<VMPS::HubbardU1xU1::StateXd> * g = new Eigenstate<VMPS::HubbardU1xU1::StateXd>;
			
			VMPS::HubbardU1xU1::Solver * gDMRG = new VMPS::HubbardU1xU1::Solver(DMRG::VERBOSITY::SILENT);
			gDMRG->edgeState(H, *g, Nc, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::SQ_TEST, 1e-8,1e-6, 8,500, 50,6);
			ginfo << gChronos.info(make_string("ground state ",Nc)) << endl;
			Emin = g->energy;
			ginfo << gDMRG->info() << endl;
			delete gDMRG;
			delete g;
		}
		
		#pragma omp section
		{
//			lout << "thread(r): " << omp_get_thread_num() << endl;
			Stopwatch<> rChronos;
			Eigenstate<VMPS::HubbardU1xU1::StateXd> * r = new Eigenstate<VMPS::HubbardU1xU1::StateXd>;
			VMPS::HubbardU1xU1::Solver * rDMRG = new VMPS::HubbardU1xU1::Solver(DMRG::VERBOSITY::SILENT);
			
			rDMRG->edgeState(H, *r, Nc, LANCZOS::EDGE::ROOF, LANCZOS::CONVTEST::SQ_TEST, 1e-8,1e-6, 8,500, 50,6);
			rinfo << rChronos.info(make_string("roof state ",Nc)) << endl;
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
	VMPS::HubbardU1xU1::StateXd initA;
	initA.eps_svd = 1e-15;
	if (spec == "AES")
	{
		OxV(H.cc(L/2), init->state, initA);
	}
	else if (spec == "APS")
	{
		OxV(H.cdagcdag(L/2), init->state, initA);
	}
	else if (spec == "PES")
	{
		OxV(H.c(UP,L/2), init->state, initA);
	}
	else if (spec == "IPES")
	{
		OxV(H.cdag(UP,L/2), init->state, initA);
	}
	else if (spec == "SSF")
	{
		OxV(H.Sz(0), init->state, initA);
	}
	delete init;
	initA.eps_svd = 1e-7;
	cout << initA.info() << endl;
	//--------------</A*init>---------------
	
// 	cout << "1,0: " << avg(init->state, H, init->state, true) << endl;
 //  	H.scale(2.);
//   	cout << "2,0: " << avg(init->state, H, init->state) << ", "  << 2.*E0 << endl;
//   	cout << "2,0: " << avg(init->state, H, init->state, true) << ", "  << pow(2.*E0,2) << endl;
//   	H.scale(0.5);
//   	cout << "1,0: " << avg(init->state, H, init->state) << ", "  << E0 << endl;
//   	cout << "2,0: " << avg(init->state, H, init->state, true) << ", "  << pow(E0,2) << endl;
	
 // 	H.scale(2.,2.);
//  	cout << "H: " << avg(init->state, H, init->state) << ", " <<  2.*E0+2. << endl;  
//  	cout << "Hsq: " << avg(init->state, H, init->state, true) << ", " << pow(2.*E0+2.,2) << endl;
	
// 	cout << "scale test: " << avg(init->state, H, init->state, true)-pow(2.*E0,2) << endl;
// 	H.scale(3.,5.);
// 	cout << "scale test: " << avg(init->state, H, init->state)-6.*E0-5. << endl;
	
	
// 	auto Psi = init->state;
// 	VMPS::HubbardU1xU1::StateXd rand = Psi; rand.setRandom();
// 	Psi += 0.1 * rand;
// 	VMPS::HubbardU1xU1::CompressorXd Compadre(DMRG::VERBOSITY::HALFSWEEPWISE);
// 	VMPS::HubbardU1xU1::StateXd Phi;
// 	Compadre.varCompress(Psi,Phi,2);
// 	cout << Psi.info() << endl;
// 	cout << Phi.info() << endl;
	
	
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
	KPS = new OrthPolyGreen<VMPS::HubbardU1xU1,VMPS::HubbardU1xU1::StateXd,CHEBYSHEV>(Emin-spillage, Emax+spillage);
	
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
