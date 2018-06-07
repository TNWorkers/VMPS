#define DMRG_DONT_USE_OPENMP
#define PRINT_SU2_FACTORS

#include <iostream>

#include "termcolor.hpp"

#include "ArgParser.h"

#include "Logger.h"
Logger lout;

using namespace std;
#include "bases/FermionBaseSU2xU1.h"

#include "models/HubbardU1xU1.h"
#include "models/HubbardSU2xU1.h"
#include "solvers/DmrgSolver.h"

size_t L;
double U, factor;
int S,N;
int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	L = args.get<size_t>("L",4);
	U = args.get<double>("U",8.);
	S = args.get<int>("S",1);
	N = args.get<int>("N",L);
	factor = args.get<double>("factor",1.);
	
	bool COMMUTATORS = args.get<bool>("COMMUTATORS",false);
	bool OBSERVABLES = args.get<bool>("OBSERVABLES",true);

	typedef Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > Symmetry;
	typedef FermionBase<Symmetry> Base;
	typedef Base::Operator Op;
	Base F(L);

	if(COMMUTATORS)
	{
		cout << termcolor::red << "-------Commutators--------" << termcolor::reset << endl;
		for(size_t i=0; i<L; i++)
		for(size_t j=0; j<L; j++)
		{
			auto commutator = Op::prod(F.cdag(i),F.c(j),Symmetry::qvacuum()) - Op::prod(F.c(j),F.cdag(i),Symmetry::qvacuum());
			cout << "i=" << i << ", j=" << j << endl << commutator.data().print(true) << endl << endl;
			
			auto anticommutator = Op::prod(F.cdag(i),F.c(j),Symmetry::qvacuum()) + Op::prod(F.c(j),F.cdag(i),Symmetry::qvacuum());
			cout << "i=" << i << ", j=" << j << endl << anticommutator.data().print(true) << endl << endl;
		}
	}

	if (OBSERVABLES)
	{
		qarray<2> Q1 = {S,N};
		qarray<2> Q2 = {S+1,N-1};

		auto H = F.HubbardHamiltonian(U);
		EDSolver<Op> John(H,{Q1,Q2},Eigen::DecompositionOptions::ComputeEigenvectors);
		cout << "gse1=" << John.eigenvalues().data().block[1](0,0) << endl;
		cout << "gse2=" << John.eigenvalues().data().block[0](0,0) << endl;

		Eigen::MatrixXd densityMatrix(L,L); densityMatrix.setZero();
		for(size_t i=0; i<L; i++)
		for(size_t j=0; j<L; j++)
		{
  			auto res = Op::prod(John.groundstate(Q1).adjoint(),Op::prod(F.cdag(i),Op::prod(F.c(j),John.groundstate(Q1),{2,-1}),{1,0}),{1,0});

			densityMatrix(i,j) = sqrt(2.) * res.data().block[0](0,0);
		}
		cout << "densityMatrix" << endl << densityMatrix << endl << endl;

		Eigen::VectorXd n(L); n.setZero();
		for(size_t i=0; i<L; i++)
		{
  			auto res = Op::prod(John.groundstate(Q1).adjoint(),Op::prod(F.n(i),John.groundstate(Q1),{1,0}),{1,0});
			n(i) = res.data().block[0](0,0);
		}
		cout << "n" << endl << n << endl << n.sum() << endl << endl;

		Eigen::VectorXd c(L); c.setZero();
		for(size_t i=0; i<L; i++)
		{
  			auto res = Op::prod(John.groundstate(Q2).adjoint(),Op::prod(F.c(i),John.groundstate(Q1),{2,-1}),{2,-1});
			c(i) = res.data().block[0](0,0);
		}
		cout << "c" << endl << c << endl << endl;

		Eigen::VectorXd adag(L); adag.setZero();
		for(size_t i=0; i<L; i++)
		{
			//cdag2 gibt den richtigen hermitisch konjugierten Operator!
			auto adagQ2 = Op::prod(F.cdag(i), John.groundstate(Q1), {2,+1});
			auto res = Op::prod(adagQ2.adjoint(), adagQ2, {1,0});
//  		auto res = Op::prod(John.groundstate(Q1).adjoint(), Op::prod(F.adag(i),John.groundstate(Q2),{2,1}),{2,1});
			adag(i) = res.data().block[0](0,0);
		}

		cout << "adag" << endl << adag << endl << endl;
//		cout << "ratio=" << endl << c.array()/(cdag.array()) << endl << endl;
		
		VMPS::HubbardU1xU1 H_DMRGU1(L,{{"U",U}});
		cout << H_DMRGU1.info() << endl;
		VMPS::HubbardU1xU1::Solver Jack(DMRG::VERBOSITY::ON_EXIT);
		Eigenstate<VMPS::HubbardU1xU1::StateXd> g1, g2;
		Jack.edgeState(H_DMRGU1,g1,{static_cast<int>(L/2),static_cast<int>(L/2)},LANCZOS::EDGE::GROUND);
		cout << endl << endl;
		VMPS::HubbardU1xU1::Solver Lisa(DMRG::VERBOSITY::ON_EXIT);
		Lisa.edgeState(H_DMRGU1,g2,{static_cast<int>(L/2)-1,static_cast<int>(L/2)},LANCZOS::EDGE::GROUND);
		Eigen::VectorXd c_checkU1(L); c_checkU1.setZero();
		Eigen::VectorXd c_checkU1_comp(L); c_checkU1_comp.setZero();
		Eigen::VectorXd n_checkU1_comp(L); n_checkU1_comp.setZero();
		
		for(size_t i=0; i<L; i++)
		{
			c_checkU1(i) = avg(g1.state, H_DMRGU1.cdag<UP>(i), g2.state);
		
			VMPS::HubbardU1xU1::CompressorXd Compadre(DMRG::VERBOSITY::SILENT);
			VMPS::HubbardU1xU1::StateXd cg2;
			Compadre.prodCompress(H_DMRGU1.cdag<UP>(i), H_DMRGU1.c<UP>(i), g2.state, cg2, {static_cast<int>(L/2),static_cast<int>(L/2)}, g2.state.calc_Dmax());
			c_checkU1_comp(i) = dot(g1.state, cg2);
			n_checkU1_comp(i) = dot(cg2,cg2);
		}
		cout << "c_checkU1" << endl << c_checkU1 << endl << endl;
		cout << "c_checkU1_comp" << endl << c_checkU1_comp << endl << endl;
		cout << "n_checkU1_comp" << endl << n_checkU1_comp << endl << endl;
		
		VMPS::HubbardSU2xU1 H_DMRGSU2(L,{{"U",U}});
		cout << H_DMRGSU2.info() << endl;
		VMPS::HubbardSU2xU1::Solver Jim(DMRG::VERBOSITY::ON_EXIT);
		Eigenstate<VMPS::HubbardSU2xU1::StateXd> h1, h2;
		Jim.edgeState(H_DMRGSU2,h1,{S,N},LANCZOS::EDGE::GROUND);
		cout << endl << endl;
		VMPS::HubbardSU2xU1::Solver Lana(DMRG::VERBOSITY::ON_EXIT);
		Lana.edgeState(H_DMRGSU2,h2,{S+1,N-1},LANCZOS::EDGE::GROUND);
		Eigen::VectorXd c_checkSU2(L); c_checkSU2.setZero();
		Eigen::VectorXd c_checkSU2_comp(L); c_checkSU2_comp.setZero();
		Eigen::VectorXd n_checkSU2_comp(L); n_checkSU2_comp.setZero();
		for(size_t i=0; i<L; i++)
		{
			c_checkSU2(i) = avg(h1.state, H_DMRGSU2.adag(i,0,factor), h2.state);
			
			VMPS::HubbardSU2xU1::CompressorXd Compadre(DMRG::VERBOSITY::SILENT);
			VMPS::HubbardSU2xU1::StateXd ch2;
			Compadre.prodCompress(H_DMRGSU2.cdag(i,0,1.), H_DMRGSU2.c(i,0,1./sqrt(2.)), h1.state, ch2, {S+1,N+1}, h1.state.calc_Dmax());
//			c_checkSU2_comp(i) = dot(h1.state, ch2);
			n_checkSU2_comp(i) = dot(ch2,ch2);
		}
		cout << "c_checkSU2" << endl << c_checkSU2 << endl << endl;
		cout << "c_checkSU2_comp" << endl << c_checkSU2_comp << endl << endl;
		cout << "n_checkSU2_comp" << endl << n_checkSU2_comp << endl << endl;
		
		Eigen::VectorXd cdag_checkSU2(L); cdag_checkSU2.setZero();
		for(size_t i=0; i<L; i++)
		{
			cdag_checkSU2(i) = avg(h1.state, H_DMRGSU2.adag(i,0,factor), h2.state);
		}
		cout << "cdag_checkSU2" << endl << -cdag_checkSU2 << endl << endl;
	}
}
