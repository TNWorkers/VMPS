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
int M;
double J;
vector<int> Msave;
int Mmax;
string spec, wd, outfile, Efilename;
vector<double> dE;
double Emin, Emax, E0;
bool CHEB;

int main (int argc, char* argv[]) 
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",4);
	M = args.get<size_t>("M",0);
	#ifdef USING_SU2
	qarray<1> Qi = {M+1};
	#else
	qarray<1> Qi = {M};
	#endif
	qarray<1> Qc;
	spec = args.get<string>("spec","SSF");
	J = args.get<double>("J",-1.);
	CHEB = args.get<bool>("CHEB",true);
	wd = args.get<string>("wd","./");
	if (wd.back() != '/') {wd += "/";}
	
	dE = args.get_list<double>("dE",{0.2});
	outfile = make_string(spec,"_L=",L,"_J=",J,"_M=",M);
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
	MODEL H(L,{{"J",J},{"CALC_SQUARE",true}});
	lout << H.info() << endl << endl;
	
	MODEL::Operator A, Adag;
	#ifdef USING_SU2
	{
		A = H.S(L/2);
		Adag = H.Sdag(L/2, 0, 1./sqrt(3.));
		Qc = {M+3};
	}
	#else
	{
		if (spec == "SSFP")
		{
			A = H.Scomp(SP, L/2, 0, 1./sqrt(2));
			Adag = H.Scomp(SM, L/2, 0, 1./sqrt(2));
			Qc = Qi+A.Qtarget();
		}
		else if (spec == "SSFM")
		{
			A = H.Scomp(SM, L/2, 0, 1./sqrt(2));
			Adag = H.Scomp(SP, L/2, 0, 1./sqrt(2));
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
	lout << A.info() << endl;
	lout << Adag.info() << endl;
	//--------------</Hamiltonian & transition operator>---------------
	
	#include "programs/snippets/groundstate.txt"
	#include "programs/snippets/state_compression.txt"
	#include "programs/snippets/AxInit.txt"
	#include "programs/snippets/KPS.txt"
}
