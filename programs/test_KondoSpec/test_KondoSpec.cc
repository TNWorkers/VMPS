#define DMRG_DONT_USE_OPENMP
#ifdef BLAS
#include "util/LapackManager.h"
// extern "C" void openblas_set_num_threads(int num_threads);
// openblas_set_num_threads(1);
#pragma message ( "Eigen uses openblas." )
#else
#pragma message ( "Eigen uses own code (not BLAS)." )
#define EIGEN_DONT_PARALLELIZE
#endif


#include <iostream>
#include <fstream>
#include <complex>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

#include "Stopwatch.h"
#include "PolychromaticConsole.h"
#include "Logger.h"
Logger lout;

#include "solvers/DmrgSolver.h"
#include "solvers/MpsCompressor.h"
#include "DmrgLinearAlgebra.h"

#include "ArgParser.h"
//#include "IntervalIterator.h"
#include "OrthPolyGreen.h"

//#include "InterpolGSL.h"

#include "models/KondoSU2xU1.h"

typedef VMPS::KondoSU2xU1 MODEL;

size_t L;
int N;
double J, tPrime;
vector<int> Msave;
int Mmax;
string spec, wd, outfile, Efilename;
vector<double> dE;
double Emin, Emax, E0;
double d, n_sig;
size_t Dinit, Dlimit, Qinit, Imin, Imax;
int max_Nrich;
double tol_eigval, tol_state;
double alpha, eps_svd;

OrthPolyGreen<MODEL,MODEL::StateXd> * KPS;

int main (int argc, char* argv[]) 
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L");
	N = args.get<size_t>("N",L);
	qarray<2> Qi = {1,N};
	qarray<2> Qc;
	spec = args.get<string>("spec","SSF");
	J = args.get<double>("J",-3.);
	tPrime = args.get<double>("tPrime",1.);
	wd = args.get<string>("wd","./");
	if (wd.back() != '/') {wd += "/";}

	alpha = args.get<double>("alpha",100.);
	eps_svd = args.get<double>("eps_svd",1.e-7);
	Dinit  = args.get<size_t>("Dinit",10ul);
	Dlimit = args.get<size_t>("Dmax",500ul);
	Qinit  = args.get<size_t>("Qinit",10ul);
	Imin   = args.get<size_t>("Imin",2ul);
	Imax   = args.get<size_t>("Imax",50ul);
	tol_eigval = args.get<double>("tol_eigval",1e-7);
	tol_state  = args.get<double>("tol_state",1e-7);
	max_Nrich = args.get<int>("max_Nrich",-1);
	size_t j = args.get<size_t>("j",L/2);

	dE = args.get_list<double>("dE",{5.});
	outfile = make_string(spec,"_L=",L,"_N=",N,"_J=",J,"_tPrime=",tPrime);
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
	MODEL H(L,{{"J",J},{"tPrime",tPrime},{"CALC_SQUARE",true}});
	lout << H.info() << endl << endl;
	MODEL::Operator A, Adag;
	if (spec == "SSF")
	{
		A = H.Simp(j);
		Adag = H.Simpdag(j);
		Qc = {3,N};
	}
	//else if (spec == "AES")
	// {
	// 	A = H.cc(L/2);
	// 	Adag = H.cdagcdag(L/2);
	// 	Qc = MODEL::singlet(N-2);
	// }
	// else if (spec == "APS")
	// {
	// 	A = H.cdagcdag(L/2);
	// 	Adag = H.cc(L/2);
	// 	Qc = MODEL::singlet(N+2);
	// }
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
	 	A = H.n(j);
	 	Adag = A;
	 	Qc = Qi;
	 }
	lout << A.info() << endl;
	lout << Adag.info() << endl;
	//--------------</Hamiltonian & transition operator>---------------
	
	//--------------<ground state>---------------
	Eigenstate<MODEL::StateXd> * init = new Eigenstate<MODEL::StateXd>;

	vector<Param> SweepParams;
	SweepParams.push_back({"max_alpha",alpha});
	SweepParams.push_back({"eps_svd",eps_svd});
	SweepParams.push_back({"max_halfsweeps",Imax});
	SweepParams.push_back({"min_halfsweeps",Imin});
	SweepParams.push_back({"Dinit",Dinit});
	SweepParams.push_back({"Qinit",Qinit});
	SweepParams.push_back({"Dlimit",Dlimit});
	SweepParams.push_back({"tol_eigval",tol_eigval});
	SweepParams.push_back({"tol_state",tol_state});
	SweepParams.push_back({"max_Nrich",max_Nrich});
	SweepParams.push_back({"min_Nsv",1ul});

	Stopwatch<> Chronos;
	stringstream ginfo, rinfo;
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			Stopwatch<> nChronos;
			MODEL::Solver * DMRG = new MODEL::Solver(DMRG::VERBOSITY::SILENT);
			
			DMRG->edgeState(H, *init, Qi, LANCZOS::EDGE::GROUND);
			DMRG->DynParam = H.get_DynParam(SweepParams);
			DMRG->GlobParam = H.get_GlobParam(SweepParams);
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
			gDMRG->DynParam = H.get_DynParam(SweepParams);
			gDMRG->GlobParam = H.get_GlobParam(SweepParams);
			gDMRG->GlobParam.tol_state = 1.e-4;
			gDMRG->GlobParam.tol_eigval = 1.e-4;
			gDMRG->edgeState(H, *g, Qc, LANCZOS::EDGE::GROUND);
			ginfo << gChronos.info(make_string("ground state ",Qc)) << endl;
			Emin = g->energy;
			ginfo << gDMRG->info() << endl;
			delete gDMRG;
			
//			MODEL::CompressorXd C(DMRG::VERBOSITY::HALFSWEEPWISE);
//			MODEL::StateXd Vout;
//			cout << "Emin=" << Emin << endl;
//			C.polyCompress(H, g->state, 10., g->state, Vout, g->state.calc_Dmax());
//			cout << "dot=" << Vout.dot(Vout) << ", " << pow(g->energy-10.,2) << endl;
//			assert(1!=1);
			
			delete g;
		}
		
		#pragma omp section
		{
			Stopwatch<> rChronos;
			Eigenstate<MODEL::StateXd> * r = new Eigenstate<MODEL::StateXd>;
			MODEL::Solver * rDMRG = new MODEL::Solver(DMRG::VERBOSITY::STEPWISE);
			rDMRG->DynParam = H.get_DynParam(SweepParams);
			rDMRG->GlobParam = H.get_GlobParam(SweepParams);
			rDMRG->GlobParam.tol_state = 1.e-4;
			rDMRG->GlobParam.tol_eigval = 1.e-4;
//			rDMRG->GlobParam.Dinit = 40;
//			rDMRG->GlobParam.Qinit = 30;
			rDMRG->DynParam.max_alpha_rsvd = [](size_t i) { return (i<=10)? 100.:0; };
			rDMRG->DynParam.min_alpha_rsvd = [](size_t i) { return (i<=10)? 1.e-11:0; };
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
	
	auto Psi = init->state;
//	MODEL::StateXd rand = Psi; rand.setRandom();
//	Psi += 0.1 * rand;
	MODEL::CompressorXd Compadrino(DMRG::VERBOSITY::HALFSWEEPWISE);
	MODEL::StateXd Phi;
	Compadrino.stateCompress(Psi,Phi,10,1e-10);
	cout << endl;
	cout << "ORIGINAL:" << endl;
	cout << Psi.info() << endl;
	cout << "COMPRESSED:" << endl;
	cout << Phi.info() << endl;
	
	//--------------<A*init>---------------
//	cout << "avg1=" << avg(init->state, Adag, A, init->state) << endl;
//	cout << "avg2=" << avg(init->state, H.SimpSimp(i,j), init->state) << endl;
	
	MODEL::StateXd initB;
	vector<MODEL::StateXd> initA(L);
	
	for(size_t i=0; i<L; i++)
	{
		cout << "i=" << i << endl;
		initA[i].eps_svd = 1e-15;
		MODEL::CompressorXd Compadre(DMRG::VERBOSITY::HALFSWEEPWISE);
		if (spec == "SSF")
		{
			Compadre.prodCompress(H.Simp(i), H.Simpdag(i), init->state, initA[i], Qc, init->state.calc_Dmax());
		}
		else if (spec == "CSF")
		{
			Compadre.prodCompress(H.n(i), H.n(i), init->state, initA[i], Qc, init->state.calc_Dmax());
		}
		initA[i].eps_svd = 1e-7;
		cout << "AxV:" << endl << initA[i].info() << endl;
	}
	initB.eps_svd = 1e-15;
	MODEL::CompressorXd Compadre(DMRG::VERBOSITY::HALFSWEEPWISE);
	if (spec == "SSF")
	{
		Compadre.prodCompress(H.Simpdag(j,0,1.), H.Simp(j,0,sqrt(3.)), init->state, initB, Qc, init->state.calc_Dmax());
	}
	else if (spec == "CSF")
	{
		Compadre.prodCompress(H.n(j), H.n(j), init->state, initB, Qc, init->state.calc_Dmax());
	}
	delete init;
	initB.eps_svd = 1e-7;
	cout << "BxV:" << endl << initB.info() << endl;
	cout << "initB.squaredNorm()=" << initB.squaredNorm() << endl;
	initB.graph("initB");
	//--------------</A*init>---------------
	
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
	
	KPS->calc_ImAB(H,initA,initB,Mmax);
	lout << "Chebyshev iteration done!" << endl;
	//--------------</KernelPolynomialSolver>---------------
	
	delete KPS;	
}
