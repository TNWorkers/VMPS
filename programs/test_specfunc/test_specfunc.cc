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
int N, M, Nup, Ndn;
double U, V, tPrime;
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
	N = args.get<size_t>("N",L);
	M = args.get<size_t>("M",0);
	#ifdef USING_SU2
	qarray<2> Qi = {M+1,N};
	#else
	Nup = (N+M)/2;
	Ndn = (N-M)/2;
	qarray<2> Qi = {Nup,Ndn};
	#endif
	qarray<2> Qc;
	spec = args.get<string>("spec","AES");
	U = args.get<double>("U",6.);
	tPrime = args.get<double>("tPrime",0);
	V = args.get<double>("V",0.);
	CHEB = args.get<bool>("CHEB",true);
	wd = args.get<string>("wd","./");
	if (wd.back() != '/') {wd += "/";}
	
	dE = args.get_list<double>("dE",{0.2});
	outfile = make_string(spec,"_L=",L,"_N=",N,"_M=",M,"_U=",U);
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
	MODEL H(L,{{"U",U},{"tPrime",tPrime},{"V",V},{"CALC_SQUARE",true}});
	lout << H.info() << endl << endl;
	
	MODEL::Operator A, Adag;
	if (spec == "AES")
	{
		A = H.cc(L/2);
		Adag = H.cdagcdag(L/2);
		Qc = qarray<2>({Nup-1,Ndn-1});
	}
	else if (spec == "APS")
	{
		A = H.cdagcdag(L/2);
		Adag = H.cc(L/2);
		Qc = qarray<2>({Nup+1,Ndn+1});
	}
	else if (spec == "PES")
	{
		#ifdef USING_SU2
		{
			A = H.c(L/2, 0, 1.);
			Adag = H.cdag(L/2, 0, 1./sqrt(2.));
			Qc = qarray<2>({2,N-1});
		}
		#else
		{
			A = H.c<UP>(L/2);
			Adag = H.cdag<UP>(L/2);
			Qc = qarray<2>({Nup-1,Ndn});
		}
		#endif
	}
	else if (spec == "IPES")
	{
		#ifdef USING_SU2
		{
			A = H.cdag(L/2, 0, 1.);
			Adag = H.c(L/2, 0, -1./sqrt(2.));
			Qc = qarray<2>({2,N+1});
		}
		#else
		{
			A = H.cdag<UP>(L/2);
			Adag = H.c<UP>(L/2);
			Qc = qarray<2>({Nup+1,Ndn});
		}
		#endif
	}
	else if (spec == "CSF")
	{
		A = H.n(L/2);
		Adag = A;
		Qc = Qi;
	}
	else if (spec == "SSF" or spec == "SSFZ" or spec == "SSFP" or spec == "SSFM")
	{
		#ifdef USING_SU2
		{
			A = H.S(L/2);
			Adag = H.Sdag(L/2, 0, 1./sqrt(3.));
			Qc = {3,N};
		}
		#else
		{
			if (spec == "SSFM")
			{
				A = H.Scomp(SM, L/2, 0, 1./sqrt(2.));
				Adag = H.Scomp(SP, L/2, 0, 1./sqrt(2.));
				Qc = Qi+A.Qtarget();
			}
			else if (spec == "SSFP")
			{
				A = H.Scomp(SP, L/2, 0, 1./sqrt(2.));
				Adag = H.Scomp(SM, L/2, 0, 1./sqrt(2.));
				Qc = Qi+A.Qtarget();
			}
			else
			{
				A = H.Sz(L/2);
				Adag = H.Sz(L/2);
				Qc = Qi;
			}
		}
		#endif
	}
	lout << A.info() << endl;
	lout << Adag.info() << endl;
	//--------------</Hamiltonian & transition operator>---------------
	
	#include "programs/snippets/groundstate.txt"
	#include "programs/snippets/state_compression.txt"
	#include "programs/snippets/AxInit.txt"
	#include "programs/snippets/KPS.txt"
}
