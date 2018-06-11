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
int N, M;
double U, J;
vector<int> Msave;
int Mmax;
string spec, wd, outfile, Efilename;
vector<double> dE;
double Emin, Emax, E0;
double d, n_sig;
bool CHEB;

int main (int argc, char* argv[]) 
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",4);
	N = args.get<size_t>("N",0);
	M = args.get<size_t>("M",L);
//	qarray<2> Qi = MODEL::polaron(L,N); // {L+1,N}
	#ifdef USING_SU2
	qarray<2> Qi = {M+1,N};
	#else
	qarray<2> Qi = {M,N};
	#endif
	qarray<2> Qc;
	spec = args.get<string>("spec","IPES");
	U = args.get<double>("U",0.);
	J = args.get<double>("J",-1.);
	CHEB = args.get<bool>("CHEB",true);
	wd = args.get<string>("wd","./");
	if (wd.back() != '/') {wd += "/";}
	
	dE = args.get_list<double>("dE",{0.2});
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
			Adag = H.cdag(L/2, 0, 1./sqrt(2.));
//			Qc = qarray<2>({L,N-1});
			Qc = qarray<2>({M+2,N-1});
		}
		#else
		{
			A = H.c<DN>(L/2);
			Adag = H.cdag<DN>(L/2);
			Qc = Qi+A.Qtarget();
		}
		#endif
	}
	else if (spec == "IPES")
	{
		#ifdef USING_SU2
		{
			A = H.cdag(L/2, 0, 1.);
			Adag = H.c(L/2, 0, -1./sqrt(2.));
//			Qc = qarray<2>({L,N+1});
			Qc = qarray<2>({M+2,N+1});
		}
		#else
		{
			A = H.cdag<UP>(L/2);
			Adag = H.c<UP>(L/2);
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
	else if (spec == "SSF")
	{
        #ifdef USING_SU2
		{
			A = H.Simp(L/2);
			Adag = H.Simpdag(L/2,0,1./sqrt(3.));
			Qc = {M+3,N};
		}
		#else
		{
			A = H.Simp(SP, L/2, 0, 1./sqrt(2));
			Adag = H.Simp(SM, L/2, 0, 1./sqrt(2));
			Qc = Qi+A.Qtarget();

			// A = H.Sz(L/2);
			// Adag = H.Sz(L/2);
			// Qc = Qi;
		}
		#endif
	}
	lout << A.info() << endl;
	lout << Adag.info() << endl;
	//--------------</Hamiltonian & transition operator>---------------
	
	#include "programs/snippets/groundstate.txt"
//	#include "programs/snippets/state_compression.txt"
	#include "programs/snippets/AxInit.txt"
	#include "programs/snippets/KPS.txt"
}
