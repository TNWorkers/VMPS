#ifdef BLAS
#include "util/LapackManager.h"
#pragma message("LapackManager")
#endif

//#define USE_OLD_COMPRESSION
#define USE_HDF5_STORAGE
#define DMRG_DONT_USE_OPENMP
//#define DEBUG_VERBOSITY 3

#include <iostream>
#include <fstream>
#include <complex>

#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

#include "Logger.h"
Logger lout;
#include "ArgParser.h"

//include "LanczosWrappers.h"
//include "HxV.h"
#include "LanczosSolver.h"

//include "plot.hpp"
#include "StringStuff.h"
#include "Stopwatch.h"
//include "TextTable.h"
#define HELPERS_IO_TABLE

#include "solvers/DmrgSolver.h"
#include "models/HubbardSU2xU1.h"
typedef VMPS::HubbardSU2xU1 MODEL;
#include "DmrgLinearAlgebra.h"
#include "VUMPS/VumpsSolver.h"

////////////////////////////////
int main (int argc, char* argv[])
{
	ArgParser args(argc,argv);
	
	size_t L = args.get<size_t>("L",6);
	size_t N = args.get<size_t>("N",L);
	qarray<MODEL::Symmetry::Nq> Q = MODEL::singlet(N);
	double U = args.get<double>("U",8.);
	DMRG::VERBOSITY::OPTION VERB = static_cast<DMRG::VERBOSITY::OPTION>(args.get<int>("VERB",DMRG::VERBOSITY::ON_EXIT));
	
	DMRG::CONTROL::GLOB GlobParams;
	GlobParams.Minit = args.get<size_t>("Minit",1ul);
	GlobParams.Mlimit = args.get<size_t>("Mlimit",500ul);
	GlobParams.Qinit = args.get<size_t>("Qinit",1ul);
	GlobParams.min_halfsweeps = args.get<size_t>("min_halfsweeps",20ul);
	GlobParams.tol_eigval = args.get<double>("tol_eigval",1e-5);
	GlobParams.tol_state = args.get<double>("tol_state",1e-4);
	GlobParams.CALC_S_ON_EXIT = false;
	
	Eigenstate<MODEL::StateXd> g;
	
	MODEL H(L,{{"U",U}},BC::OPEN);
	lout << H.info() << endl;
	
	MODEL::Solver DMRG1(VERB);
	DMRG1.userSetGlobParam();
	DMRG1.GlobParam = GlobParams;
	
	DMRG1.edgeState(H, g, Q);
    
    auto set_operators = [H,L](Mpo<MODEL::Symmetry>& mpo)
    {
        assert(L == 6ul);

        for(std::size_t loc=0; loc<L; ++loc)
        {
            mpo.set_qPhys(loc, H.get_qPhys()[loc]);
        }
        std::vector<FermionBase<MODEL::Symmetry>> F(L);
        for(std::size_t loc=0; loc<L; ++loc)
        {
            F[loc] = FermionBase<MODEL::Symmetry>(1ul);
        }
        auto Hloc3 = F[3].d();
        auto Hloc4 = F[4].d();
        auto n3 = F[3].n();
        auto n4 = F[4].n();
        auto c2 = F[2].c();
        auto c3 = F[3].c();
        auto c4 = F[4].c();
        auto c5 = F[5].c().plain<double>();
        auto cdag2 = F[2].cdag();
        auto cdag3 = F[3].cdag();
        auto cdag4 = F[4].cdag();
        auto cdag5 = F[5].cdag().plain<double>();
        auto sign2 = F[2].sign();
        auto sign4 = F[4].sign();
        auto csign4 = SiteOperatorQ<MODEL::Symmetry,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>>::prod(c4,sign4,c4.Q());
        auto cdagsign4 = SiteOperatorQ<MODEL::Symmetry,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>>::prod(cdag4,sign4,cdag4.Q());
        auto Hlocn3 = SiteOperatorQ<MODEL::Symmetry,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>>::prod(Hloc3,n3, MODEL::Symmetry::qvacuum()).plain<double>();
        auto nHloc3 = SiteOperatorQ<MODEL::Symmetry,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>>::prod(n3,Hloc3, MODEL::Symmetry::qvacuum()).plain<double>();
        auto Hlocn4 = SiteOperatorQ<MODEL::Symmetry,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>>::prod(Hloc4,n4, MODEL::Symmetry::qvacuum()).plain<double>();
        auto nHloc4 = SiteOperatorQ<MODEL::Symmetry,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>>::prod(n4,Hloc4, MODEL::Symmetry::qvacuum()).plain<double>();
        auto ncdag3 = SiteOperatorQ<MODEL::Symmetry,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>>::prod(n3,cdag3, cdag3.Q()).plain<double>();
        auto cdagn3 = SiteOperatorQ<MODEL::Symmetry,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>>::prod(cdag3,n3, cdag3.Q()).plain<double>();
        auto nc3 = SiteOperatorQ<MODEL::Symmetry,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>>::prod(n3,c3, c3.Q()).plain<double>();
        auto cn3 = SiteOperatorQ<MODEL::Symmetry,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>>::prod(c3,n3, c3.Q()).plain<double>();

        auto ncdagsign4 = SiteOperatorQ<MODEL::Symmetry,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>>::prod(n4,cdagsign4, cdagsign4.Q()).plain<double>();
        auto cdagsignn4 = SiteOperatorQ<MODEL::Symmetry,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>>::prod(cdagsign4,n4, cdagsign4.Q()).plain<double>();
        auto ncsign4 = SiteOperatorQ<MODEL::Symmetry,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>>::prod(n4,csign4, csign4.Q()).plain<double>();
        auto csignn4 = SiteOperatorQ<MODEL::Symmetry,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>>::prod(csign4,n4, csign4.Q()).plain<double>();
        auto csign2 = SiteOperatorQ<MODEL::Symmetry,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>>::prod(c2,sign2,c2.Q()).plain<double>();
        auto cdagsign2 = SiteOperatorQ<MODEL::Symmetry,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>>::prod(cdag2,sign2,cdag2.Q()).plain<double>();
        
        mpo.push_local(3, 1., Hlocn3);
        mpo.push_local(3, -1., nHloc3);
        mpo.push_local(4, 1., Hlocn4);
        mpo.push_local(4, -1., nHloc4);
        mpo.push_tight(2, sqrt(2.), csign2, ncdag3);
        mpo.push_tight(2, -sqrt(2.), csign2, cdagn3);
        mpo.push_tight(2, sqrt(2.), cdagsign2, nc3);
        mpo.push_tight(2, -sqrt(2.), cdagsign2, cn3);
        mpo.push_tight(4, sqrt(2.), ncsign4, cdag5);
        mpo.push_tight(4, -sqrt(2.), csignn4, cdag5);
        mpo.push_tight(4, sqrt(2.), ncdagsign4, c5);
        mpo.push_tight(4, -sqrt(2.), cdagsignn4, c5);
        mpo.finalize(true,1ul,1e-8);
    };
	
    Mpo<MODEL::Symmetry> idmpo(L, MODEL::Symmetry::qvacuum(), "Identity");
    for(std::size_t loc=0; loc<L; ++loc)
    {
        idmpo.set_qPhys(loc, H.get_qPhys()[loc]);
    }
    idmpo.set_Identity();
    
    MODEL::StateXd Tmp_A1;
	Mpo<MODEL::Symmetry> O_A1 = sum(H.n(L/2),H.n(L/2+1));
	Mpo<MODEL::Symmetry> OxH_A1 = prod(O_A1,H);
	Mpo<MODEL::Symmetry> HxO_A1 = prod(H,O_A1);
	OxH_A1.scale(-1.);
	Mpo<MODEL::Symmetry> Commutator_A1 = sum(HxO_A1,OxH_A1);
    OxV_exact(Commutator_A1, g.state, Tmp_A1, 2., DMRG::VERBOSITY::SILENT);
    double dot_value_A1 = dot(Tmp_A1,Tmp_A1);
    double avg_value_A1 = -1.*avg(g.state,Commutator_A1, Commutator_A1, g.state);
    double idcheck_value_A1 = avg(Tmp_A1,idmpo,idmpo,Tmp_A1);
    std::cout << "Case A1: avg(state,HA-AH,HA-AH,state)=" << avg_value_A1 << ", dot((HA-AH)*state,(HA-AH)*state)=" << dot_value_A1 << ", avg((HA-AH)*state,id,id,(HA-AH)*state)=" << idcheck_value_A1 << std::endl;

    
    MODEL::StateXd Tmp_A2;
    Mpo<MODEL::Symmetry> Commutator_A2(L, MODEL::Symmetry::qvacuum(), "OwnMPO");
    set_operators(Commutator_A2);
    OxV_exact(Commutator_A2, g.state, Tmp_A2, 2., DMRG::VERBOSITY::SILENT);
    double dot_value_A2 = dot(Tmp_A2,Tmp_A2);
    double avg_value_A2 = -1.*avg(g.state, Commutator_A2, Commutator_A2, g.state);
    double idcheck_value_A2 = avg(Tmp_A2,idmpo,idmpo,Tmp_A2);
    std::cout << "Case A2: avg(state,HA'-A'H,HA'-A'H,state)=" << avg_value_A2 << ", dot((HA'-A'H)*state,(HA'-A'H)*state)=" << dot_value_A2 << ", avg((HA'-A'H)*state,id,id,(HA'-A'H)*state)=" << idcheck_value_A2 << std::endl;
    
    MODEL::StateXd Tmp_B;
    Mpo<MODEL::Symmetry> O_B = H.n(L/2);
    Mpo<MODEL::Symmetry> OxH_B = prod(O_B,H);
    Mpo<MODEL::Symmetry> HxO_B = prod(H,O_B);
	OxH_B.scale(-1.);
    Mpo<MODEL::Symmetry> Commutator_B = sum(HxO_B,OxH_B);
	OxV_exact(Commutator_B, g.state, Tmp_B, 2., DMRG::VERBOSITY::SILENT);
    double dot_value_B = dot(Tmp_B,Tmp_B);
    double avg_value_B =  -1.*avg(g.state, Commutator_B, Commutator_B, g.state);
    double idcheck_value_B = avg(Tmp_B,idmpo,idmpo,Tmp_B);
    std::cout << "Case B: avg(state,HB-BH,HB-BH,state)=" << avg_value_B << ", dot((HB-BH)*state,(HB-BH)*state)=" << dot_value_B << ", avg((HB-BH)*state,id,id,(HB-BH)*state)=" << idcheck_value_B << std::endl;

    MODEL::StateXd Tmp_C;
    Mpo<MODEL::Symmetry> O_C = prod(H.n(L/2),H.n(L/2+1));
    Mpo<MODEL::Symmetry> OxH_C = prod(O_C,H);
    Mpo<MODEL::Symmetry> HxO_C = prod(H,O_C);
	OxH_C.scale(-1.);
    Mpo<MODEL::Symmetry> Commutator_C = sum(HxO_C,OxH_C);
	OxV_exact(Commutator_C, g.state, Tmp_C, 2., DMRG::VERBOSITY::SILENT);
    double dot_value_C = dot(Tmp_C,Tmp_C);
    double avg_value_C =  -1.*avg(g.state, Commutator_C, Commutator_C, g.state);
    double idcheck_value_C = avg(Tmp_C,idmpo,idmpo,Tmp_C);
    std::cout << "Case C: avg(state,HC-CH,HC-CH,state)=" << avg_value_C << ", dot((HC-CH)*state,(HC-CH)*state)=" << dot_value_C << ", avg((HC-CH)*state,id,id,(HC-CH)*state)=" << idcheck_value_C << std::endl;
}
