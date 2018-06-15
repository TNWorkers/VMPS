#define DMRG_DONT_USE_OPENMP
//#define PRINT_SU2_FACTORS

#include <iostream>

#include "termcolor.hpp"

#include "ArgParser.h"

#include "Logger.h"
Logger lout;

using namespace std;
#include "bases/SpinBaseSU2.h"

#include "models/HeisenbergSU2.h"
#include "solvers/DmrgSolver.h"
#include "solvers/MpsCompressor.h"

size_t L;
double J, factor;
int S,Sc,D;
int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",4);
	J = args.get<double>("J",-1.);
	S = args.get<int>("S",1);
	Sc = args.get<int>("Sc",S+2);
	D = args.get<int>("D",1);
	factor = args.get<double>("factor",1.);
	
	bool OBSERVABLES = args.get<bool>("OBSERVABLES",true);

	typedef Sym::SU2<Sym::SpinSU2> Symmetry;
	typedef SpinBase<Symmetry> Base;
	typedef Base::Operator Op;
	Base B(L);

	if (OBSERVABLES)
	{
		qarray<1> Q1 = {S};

		auto H = B.HeisenbergHamiltonian(J);
		EDSolver<Op> John(H,{Q1},Eigen::DecompositionOptions::ComputeEigenvectors);
		cout << "gse=" << John.eigenvalues().data().block[0](0,0) << endl;

		Eigen::VectorXd SdagS(L); SdagS.setZero();
		for(size_t i=0; i<L; i++)
		{
			auto SQ1 = Op::prod(B.S(i), John.groundstate(Q1), {3});
			auto res = Op::prod(SQ1.adjoint(), SQ1, {1});
			SdagS(i) = res.data().block[0](0,0);
		}
		cout << "sqrt(1./3.)*SdagS" << endl << sqrt(1./3.)*SdagS << endl << endl;
		cout << "sqrt(3.)*SdagS" << endl << sqrt(3.)*SdagS << endl << endl;

		Eigen::MatrixXd SS(L,L); SS.setZero();
		for(size_t i=0; i<L; i++)
		for(size_t j=0; j<L; j++)
		{
			auto SQ1i = Op::prod(B.S(i), John.groundstate(Q1), {3});
			auto SQ1j = Op::prod(B.S(j), John.groundstate(Q1), {3});

			auto res = Op::prod(SQ1i.adjoint(), SQ1j, {1});
			SS(i,j) = res.data().block[0](0,0);
		}
		cout << "sqrt(1./3.)*SS" << endl << sqrt(1./3.)*SS << endl << endl;
		cout << "sqrt(3.)*SS" << endl << sqrt(3.)*SS << endl << endl;


		VMPS::HeisenbergSU2 H_SU2(L,{ {"J",J} });
		lout << H_SU2.info() << endl;

		Eigenstate<VMPS::HeisenbergSU2::StateXd> g;
		Eigenstate<VMPS::HeisenbergSU2::StateXd> h;

		vector<Param> SweepParams;
		SweepParams.push_back({"max_alpha",10.});

		VMPS::HeisenbergSU2::Solver Lana(DMRG::VERBOSITY::ON_EXIT);
		Lana.GlobParam = H_SU2.get_GlobParam(SweepParams);
		Lana.DynParam = H_SU2.get_DynParam(SweepParams);
		Lana.edgeState(H_SU2, g, {S}, LANCZOS::EDGE::GROUND);

		// VMPS::HeisenbergSU2::Solver Jim(DMRG::VERBOSITY::SILENT);
		// Jim.GlobParam = H_SU2.get_GlobParam(SweepParams);
		// Jim.DynParam = H_SU2.get_DynParam(SweepParams);
		// Jim.edgeState(H_SU2, h, {S+2}, LANCZOS::EDGE::GROUND);

		Eigen::VectorXd SdagS_DMRG(L); SdagS_DMRG.setZero();
		for(size_t i=0; i<L; i++)
		{
			VMPS::HeisenbergSU2::StateXd Sg;
			// VMPS::HeisenbergSU2::CompressorXd Compadre(DMRG::VERBOSITY::SILENT);
			// Compadre.prodCompress(H_SU2.S(i), H_SU2.Sdag(i,0,1./sqrt(3.)), g.state, Sg, {Sc}, 100);
			OxV_exact(H_SU2.S(i),g.state,Sg,{Sc});
			SdagS_DMRG(i) = Sg.dot(Sg);
		}
		cout << "sqrt(1./3.)*SdagS DMRG: " << endl << SdagS_DMRG << endl << endl;
		cout << "sqrt(3.)*SdagS DMRG" << endl << 3.*SdagS_DMRG << endl << endl;

		Eigen::MatrixXd SS_DMRG(L,L); SS_DMRG.setZero();
		for(size_t i=0; i<L; i++)
		for(size_t j=0; j<L; j++)
		{
			VMPS::HeisenbergSU2::StateXd Sig;
			// VMPS::HeisenbergSU2::CompressorXd Compadrei(DMRG::VERBOSITY::SILENT);
			// Compadrei.prodCompress(H_SU2.S(i), H_SU2.Sdag(i,0,1./sqrt(3.)), g.state, Sig, {Sc}, 100);
			VMPS::HeisenbergSU2::StateXd Sjg;
			// VMPS::HeisenbergSU2::CompressorXd Compadrej(DMRG::VERBOSITY::SILENT);
			// Compadrej.prodCompress(H_SU2.S(j), H_SU2.Sdag(j,0,1./sqrt(3.)), g.state, Sjg, {Sc}, 100);
			
			OxV_exact(H_SU2.S(i),g.state,Sig,{Sc});
			OxV_exact(H_SU2.S(j),g.state,Sjg,{Sc});
			SS_DMRG(i,j) = Sig.dot(Sjg);
		}
		cout << "sqrt(1./3.)*SS DMRG: " << endl << SS_DMRG << endl << endl;
		cout << "sqrt(3.)*SS DMRG" << endl << 3.*SS_DMRG << endl << endl;

		cout << "ratio: " << endl << 3.*(SS_DMRG.array())/(sqrt(3.)*SS.array()) << endl << endl;

	}
}
