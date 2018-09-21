#ifdef BLAS
#include "util/LapackManager.h"
#pragma message("LapackManager")
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

#include "solvers/DmrgSolver.h"
#include "solvers/MpsCompressor.h"
#include "DmrgLinearAlgebra.h"

#include "ArgParser.h"
#include "OrthPolyGreen.h"

size_t L;
int M, Sc;
double J;
size_t D;
vector<int> Msave;
int Mmax;
string spec, wd, outfile, Efilename;
vector<double> dE;
double Emin, Emax, E0;
bool CHEB, SHIFT;
DMRG::VERBOSITY::OPTION VERB1, VERB2, VERB3;
int Qinit, Dinit;
int N=0;

int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",4);
	M = args.get<size_t>("M",0);
	Sc = args.get<size_t>("Sc",3);
	Qinit = args.get<int>("Qinit",1000);
	Dinit = args.get<int>("Dinit",1000);
	D = args.get<size_t>("D",2ul);
	
	#ifdef USING_SU2
	qarray<1> Qi = {M+1};
	#else
		#ifdef USING_U0
		qarray<0> Qi = {};
		#else
		qarray<1> Qi = {M};
		#endif
	#endif
	#ifndef USING_U0
	vector<qarray<1> > Qc;
	#else
	vector<qarray<0> > Qc;
	#endif
	spec = args.get<string>("spec","SSF");
	J = args.get<double>("J",1.);
	CHEB = args.get<bool>("CHEB",true);
	SHIFT = args.get<bool>("SHIFT",true);
	wd = args.get<string>("wd","./");
	if (wd.back() != '/') {wd += "/";}
	
	VERB1 = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB1",2));
	VERB2 = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB2",0));
	VERB3 = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB3",0));
	
	dE = args.get_list<double>("dE",{0.2});
	outfile = make_string(spec,"_L=",L,"_J=",J,"_M=",M,"_D=",D);
	#ifdef USING_SU2
	{
		outfile += make_string("_Sc=",Sc);
	}
	#endif
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
	MODEL H(L,{{"J",J},{"D",D},{"CALC_SQUARE",true}});
	lout << H.info() << endl << endl;
	
	MODEL::Operator A, Adag;
	#ifdef USING_SU2
	{
		A = H.S(L/2);
		Adag = H.Sdag(L/2, 0, sqrt(3.));
		Qc.push_back(qarray<1>{Sc});
	}
	#else
	{
		if (spec == "SSFP")
		{
			A = H.Scomp(SP, L/2, 0, 1./sqrt(2));
			Adag = H.Scomp(SM, L/2, 0, 1./sqrt(2));
			Qc.push_back(Qi+A.Qtarget());
		}
		else if (spec == "SSFM")
		{
			A = H.Scomp(SM, L/2, 0, 1./sqrt(2));
			Adag = H.Scomp(SP, L/2, 0, 1./sqrt(2));
			Qc.push_back(Qi+A.Qtarget());
		}
		#ifdef USING_U0
		else if (spec == "SSFX")
		{
			A = H.Sx(L/2);
			Adag = H.Sx(L/2);
			Qc.push_back(Qi);
		}
		else if (spec == "SSFY")
		{
			A = H.Scomp(iSY,L/2);
			Adag = H.Scomp(iSY,L/2,0,-1.);
			Qc.push_back(Qi);
		}
		#endif
		else
		{
			A = H.Sz(L/2);
			Adag = H.Sz(L/2);
			Qc.push_back(Qi);
		}
	}
	#endif
	
	lout << A.info() << endl;
	lout << Adag.info() << endl;
	lout << "Qi=" << Qi << endl;
	for (size_t i=0; i<Qc.size(); ++i)
	{
		lout << "i=" << i << ", Qc=" << Qc[i] << endl;
	}
	//--------------</Hamiltonian & transition operator>---------------
	
	#include "programs/snippets/groundstate.txt"
	#include "programs/snippets/state_compression.txt"
	#include "programs/snippets/AxInit.txt"
	#include "programs/snippets/KPS.txt"
	#include "programs/snippets/ImAA.txt"
}
