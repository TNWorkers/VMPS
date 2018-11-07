#define DMRG_DONT_USE_OPENMP
//#define PRINT_SU2_FACTORS

#include <iostream>

#include "termcolor.hpp"

#include "ArgParser.h"

#include "Logger.h"
Logger lout;

using namespace std;
#include "bases/FermionBaseSU2xU1.h"

// #include "models/HubbardU1xU1.h"
// #include "models/HubbardSU2xU1.h"

#include "models/KondoU1xU1.h"
#include "models/KondoSU2xU1.h"

#include "solvers/DmrgSolver.h"
#include "solvers/MpsCompressor.h"

size_t L;
double U, J, tPrime;
int S,Sc,M,N,Nup,Ndn,D;
int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",4);
	N = args.get<int>("N",4);
	J = args.get<double>("J",-1.);
	tPrime = args.get<double>("tPrime",0.);
	U = args.get<double>("U",0.);
	S = args.get<int>("S",1);
	M = args.get<int>("M",0);
	Sc = args.get<int>("Sc",1);
	Nup = args.get<int>("Nup",L/2);
	Ndn = args.get<int>("Ndn",L/2);
	D = args.get<int>("D",1);
	
	bool OBSERVABLES = args.get<bool>("OBSERVABLES",true);

	// typedef Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > Symmetry;
	// typedef FermionBase<Symmetry> Base;
	// typedef Base::Operator Op;
	// Base F(L);

	if (OBSERVABLES)
	{
		// qarray<2> Q1 = {S,N};

		// auto H = F.HubbardHamiltonian(U);
		// EDSolver<Op> John(H,{Q1},Eigen::DecompositionOptions::ComputeEigenvectors);
		// cout << "gse=" << John.eigenvalues().data().block[0](0,0) << endl;

		// Eigen::VectorXd cdagc(L); cdagc.setZero();
		// for(size_t i=0; i<L; i++)
		// {
		// 	auto cQ1 = Op::prod(F.c(i), John.groundstate(Q1), {2,-1});
		// 	auto res = Op::prod(cQ1.adjoint(), cQ1, {1,0});
		// 	cdagc(i) = res.data().block[0](0,0);
		// }
		// cout << "sqrt(2.)*cdagc" << endl << sqrt(2.)*cdagc << endl << endl;

		// Eigen::MatrixXd cc(L,L); cc.setZero();
		// for(size_t i=0; i<L; i++)
		// for(size_t j=0; j<L; j++)
		// {
		// 	auto cQ1i = Op::prod(F.c(i), John.groundstate(Q1), {2,-1});
		// 	auto cQ1j = Op::prod(F.c(j), John.groundstate(Q1), {2,-1});

		// 	auto res = Op::prod(cQ1i.adjoint(), cQ1j, {1,0});
		// 	cc(i,j) = res.data().block[0](0,0);
		// }
		// cout << "sqrt(2.)*cc" << endl << sqrt(2.)*cc << endl << endl;

		VMPS::KondoSU2xU1 H_SU2(L,{ {"J",J}, {"tPrime",tPrime}, {"U",U} });
		lout << H_SU2.info() << endl;

		Eigenstate<VMPS::KondoSU2xU1::StateXd> g;

		vector<Param> SweepParams;
		SweepParams.push_back({"max_alpha",10.});

		VMPS::KondoSU2xU1::Solver Lana(DMRG::VERBOSITY::ON_EXIT);
		// Lana.GlobParam = H_SU2.get_DmrgGlobParam(SweepParams);
		// Lana.DynParam = H_SU2.get_DmrgDynParam(SweepParams);
		Lana.edgeState(H_SU2, g, {S,N}, LANCZOS::EDGE::GROUND);


		Eigen::VectorXd cdagc_DMRG(L); cdagc_DMRG.setZero();
		Eigen::VectorXd SdagS_DMRG(L); SdagS_DMRG.setZero();
		Eigen::VectorXd SdagS_DMRGcontrol(L); SdagS_DMRGcontrol.setZero();

		for(size_t i=0; i<L; i++)
		{
			VMPS::KondoSU2xU1::StateXd cg;
			OxV_exact(H_SU2.c(i),g.state,cg);
			cdagc_DMRG(i) = cg.dot(cg);
			VMPS::KondoSU2xU1::StateXd Sg;
			OxV_exact(H_SU2.Ssub(i),g.state,Sg);
			SdagS_DMRG(i) = Sg.dot(Sg);
			SdagS_DMRGcontrol(i) = 0.75*(avg(g.state,H_SU2.n(i),g.state)-2*avg(g.state,H_SU2.d(i),g.state));
		}
		cout << "SdagS: " << (SdagS_DMRG-SdagS_DMRGcontrol).norm()/L << endl << SdagS_DMRG << endl;

		Eigen::MatrixXd cc_DMRG(L,L); cc_DMRG.setZero();
		Eigen::MatrixXd cc_DMRG2(L,L); cc_DMRG.setZero();

		for(size_t i=0; i<L; i++)
		for(size_t j=0; j<L; j++)
		{
			VMPS::KondoSU2xU1::StateXd cig;
			VMPS::KondoSU2xU1::StateXd cjg;
			
			OxV_exact(H_SU2.c(i),g.state,cig);
			OxV_exact(H_SU2.c(j),g.state,cjg);
			cc_DMRG(i,j) = cig.dot(cjg);
			cc_DMRG2(i,j) = avg(g.state,H_SU2.cdagc(i,j),g.state);
		}

		VMPS::KondoU1xU1 H_U1(L,{ {"J",J}, {"tPrime",tPrime}, {"U",U} });
		lout << H_U1.info() << endl;

		Eigenstate<VMPS::KondoU1xU1::StateXd> h;


		VMPS::KondoU1xU1::Solver Jim(DMRG::VERBOSITY::ON_EXIT);
		// Lana.GlobParam = H_SU2.get_DmrgGlobParam(SweepParams);
		// Lana.DynParam = H_SU2.get_DmrgDynParam(SweepParams);
		Jim.edgeState(H_U1, h, {M,N}, LANCZOS::EDGE::GROUND);


		Eigen::VectorXd cdagc_DMRGU1(L); cdagc_DMRGU1.setZero();
		for(size_t i=0; i<L; i++)
		{
			VMPS::KondoU1xU1::StateXd cUPh;
			OxV_exact(H_U1.c<UP>(i),h.state,cUPh);
			VMPS::KondoU1xU1::StateXd cDNh;
			OxV_exact(H_U1.c<DN>(i),h.state,cDNh);
			cdagc_DMRGU1(i) = cUPh.dot(cUPh) + cDNh.dot(cDNh);
		}

		Eigen::MatrixXd cc_DMRGU1(L,L); cc_DMRGU1.setZero();
		
		for(size_t i=0; i<L; i++)
		for(size_t j=0; j<L; j++)
		{
			VMPS::KondoU1xU1::StateXd cihUP;
			VMPS::KondoU1xU1::StateXd cjhUP;
			
			OxV_exact(H_U1.c<UP>(i),h.state,cihUP);
			OxV_exact(H_U1.c<UP>(j),h.state,cjhUP);

			VMPS::KondoU1xU1::StateXd cihDN;
			VMPS::KondoU1xU1::StateXd cjhDN;
			
			OxV_exact(H_U1.c<DN>(i),h.state,cihDN);
			OxV_exact(H_U1.c<DN>(j),h.state,cjhDN);

			cc_DMRGU1(i,j) = cihUP.dot(cjhUP) + cihDN.dot(cjhDN);
		}
		cout << "cc SU2: " << endl << cc_DMRG << endl << endl;
		cout << "cc 2 SU2: " << endl << cc_DMRG2 << endl << endl;

		cout << "cc U1: " << endl << cc_DMRGU1 << endl << endl;

		cout << "cc: " << endl << cc_DMRG.array()/cc_DMRGU1.array() << endl << endl;
	}
}
