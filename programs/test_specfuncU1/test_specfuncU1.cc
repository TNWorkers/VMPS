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
#include "models/HubbardSU2xU1.h"
#include "DmrgLinearAlgebra.h"

#include "ArgParser.h"
//#include "IntervalIterator.h"
#include "OrthPolyGreen.h"

//#include "InterpolGSL.h"

size_t L;
int N;
int M;
double U, V;
vector<int> Msave;
int Mmax;
string spec, wd, outfile, Efilename;
vector<double> dE;
double Emin, Emax, E0;
qarray<2> Qi, Qc;

// typedef VMPS::HubbardU1xU1 MODEL;
typedef VMPS::HubbardU1xU1 MODEL;

OrthPolyGreen<MODEL,MODEL::StateXd> * KPS;

int main (int argc, char* argv[]) 
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L");
	N = args.get<int>("N",L);
	spec = args.get<string>("spec","AES");
	U = args.get<double>("U",6.);
	V = args.get<double>("V",0.);
	M = args.get<int>("M",M);
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
	
	//--------------<Hamiltonians & Auger>---------------
	MODEL H(L,{{"U",U},{"V",V}});
	lout << H.info() << endl << endl;
	cout << "H.check_SQUARE()=" << H.check_SQUARE() << endl;
	
	MODEL::Operator O, Odag;
	if (spec == "AES")
	{
// 		Nc = qarray<2>({Nupdn-1,Nupdn-1});
		O = H.cc(L/2);
		Odag = H.cdagcdag(L/2);
		Qi = H.singlet(N);
		Qc = H.singlet(N-2);
	}
	else if (spec == "APS")
	{
// 		Nc = qarray<2>({Nupdn+1,Nupdn+1});
// 		O = H.cdagcdag(L/2);
	}
	else if (spec == "PES")
	{
// 		Nc = qarray<2>({Nupdn-1,Nupdn});
		O = H.c(UP,L/2);
	}
	else if (spec == "IPES")
	{
// 		Nc = qarray<2>({Nupdn+1,Nupdn});
		O = H.cdag(UP,L/2);
	}
	else if (spec == "SSF")
	{
// 		Nc = qarray<2>({Nupdn,Nupdn});
	}
	else if (spec == "CSF")
	{
		O = H.n(L/2);
		Odag = O;
		Qi = H.singlet(N);
		Qc = H.singlet(N);
	}
	cout << "Qi=" << Qi << ", Qc=" << Qc << ", O.IS_HERMITIAN()=" << O.IS_HERMITIAN() << endl;

	//--------------</Hamiltonians & Auger>---------------
	
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
			
			DMRG->edgeState(H, *init, Qi, LANCZOS::EDGE::GROUND, LANCZOS::CONVTEST::NORM_TEST, 1e-7,1e-6, 4,500, 50,10, 1.);
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
//			lout << "thread(r): " << omp_get_thread_num() << endl;
			Stopwatch<> rChronos;
			Eigenstate<MODEL::StateXd> * r = new Eigenstate<MODEL::StateXd>;
			MODEL::Solver * rDMRG = new MODEL::Solver(DMRG::VERBOSITY::SILENT);
			
			rDMRG->edgeState(H, *r, Qc, LANCZOS::EDGE::ROOF, LANCZOS::CONVTEST::NORM_TEST, 1e-8,1e-6, 8,500, 50,10, 1.);
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
	
	auto Htmp = H;
	cout << "avg=" << isReal(avg(init->state,Htmp,init->state)) << ", " << pow(1.*E0,1) << endl;
	cout << "avg=" << isReal(avg(init->state,Htmp,Htmp,init->state)) << ", " << pow(1.*E0,2) << endl;
	
	Htmp.scale(2.);
	cout << "avg=" << isReal(avg(init->state,Htmp,init->state)) << ", " << pow(2.*E0,1) << endl;
	cout << "avg=" << isReal(avg(init->state,Htmp,Htmp,init->state)) << ", " << pow(2.*E0,2) << endl;
	
	Htmp.scale(0.5);
	cout << "avg=" << isReal(avg(init->state,Htmp,init->state)) << ", " << pow(1.*E0,1) << endl;
	cout << "avg=" << isReal(avg(init->state,Htmp,Htmp,init->state)) << ", " << pow(1.*E0,2) << endl;
	
	Htmp.scale(4.);
	cout << "avg=" << isReal(avg(init->state,Htmp,init->state)) << ", " << pow(4.*E0,1) << endl;
	cout << "avg=" << isReal(avg(init->state,Htmp,Htmp,init->state)) << ", " << pow(4.*E0,2) << endl;
	
//  	OxV(O, init->state, initA);
	MODEL::CompressorXd Compadre(DMRG::VERBOSITY::HALFSWEEPWISE);
	Compadre.varCompress(O, Odag, init->state, initA, Qc, init->state.calc_Dmax());
	cout << Compadre.info() << endl;
// 	if (MODEL::Symmetry ==  Sym::S1xS2<Sym::U1<Sym::ChargeUp>,Sym::U1<Sym::ChargeDn> >)
// 	{
// 		MODEL::StateXd initA_;
// 		OxV(O, init->state, initA_);
// 		cout << "overlap=" << dot(initA,initA_) << ", norms=" << dot(initA,initA) << ", " << dot(initA_,initA_) << endl;
// 	}
// 	delete init;
	initA.eps_svd = 1e-7;
	cout << "initA.info()=" << initA.info() << endl;
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
	
	
	auto Psi = init->state;
	// MODEL::StateXd rand = Psi; rand.setRandom();
// 	Psi += 0.1 * rand;
	MODEL::CompressorXd Cmp(DMRG::VERBOSITY::HALFSWEEPWISE);
	MODEL::StateXd Phi;
	Cmp.varCompress(Psi,Phi,1);
	cout << Psi.info() << endl;
	cout << Phi.info() << endl;
	
	
	//--------------<KernelPolynomialSolver>---------------
	double spillage = 0.;
	if (spec == "PES" or spec == "IPES" or spec=="CSF")
	{
		spillage = 4.*dE[0];
	}
	else if (spec == "SSF")
	{
		spillage = 0.5*(Emax-Emin);
	}
	KPS = new OrthPolyGreen<MODEL,MODEL::StateXd,CHEBYSHEV>(Emin-spillage, Emax+spillage);
	
// 	for (size_t i=0; i<dE.size(); ++i)
// 	{
// 		if (i>0) {assert(dE[i] < dE[i-1]);} // monotoncally decreasing resolution
// 		Msave.push_back((Emax-Emin+2.*spillage)/dE[i]);
// 		lout << "dE=" << dE[i] << " => M=" << Msave[Msave.size()-1] << endl;
// 	}
// 	lout << endl;
	Msave.push_back(M);
	
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
	delete KPS;	
	//--------------</KernelPolynomialSolver>---------------
}
