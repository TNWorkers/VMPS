#ifndef HUBBARDMODELU1XZN_H_COMPLEX
#define HUBBARDMODELU1XZN_H_COMPLEX

#include "symmetry/S1xS2.h"
//#include "symmetry/U1.h"
//#include "bases/FermionBase.h"
//#include "models/HubbardObservables.h"
//#include "Mpo.h"
//#include "ParamReturner.h"
//#include "Geometry2D.h" // from TOOLS
#include "models/PeierlsHubbardU1xU1.h"

#ifndef YMOMENTUM
#define YMOMENTUM 6
#endif

namespace VMPS
{
class PeierlsHubbardU1xZN : public Mpo< Sym::S1xS2< Sym::U1<Sym::ChargeU1>, Sym::ZN<Sym::Momentum,YMOMENTUM> > ,complex<double> >,
                            public HubbardObservables< Sym::S1xS2<Sym::U1<Sym::ChargeU1>, Sym::ZN<Sym::Momentum,YMOMENTUM> >, complex<double> >,
                            public ParamReturner
{
public:
	
	typedef Sym::S1xS2< Sym::U1<Sym::ChargeU1>, Sym::ZN<Sym::Momentum,YMOMENTUM> > Symmetry;
	MAKE_TYPEDEFS(PeierlsHubbardU1xZN)
	typedef Eigen::Matrix<complex<double>,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	
public:
	
	PeierlsHubbardU1xZN() : Mpo(){};
	
	PeierlsHubbardU1xZN(Mpo<Symmetry,complex<double>> &Mpo_input, const vector<Param> &params)
	:Mpo<Symmetry,complex<double>>(Mpo_input),
	 HubbardObservables(this->N_sites,params,PeierlsHubbardU1xZN::defaults),
	 ParamReturner(PeierlsHubbardU1xZN::sweep_defaults)
	{
		ParamHandler P(params,PeierlsHubbardU1xZN::defaults);
		size_t Lcell = P.size();
		N_phys = 0;
		for (size_t l=0; l<N_sites; ++l) N_phys += P.get<size_t>("Ly",l%Lcell);
		this->calc(P.get<size_t>("maxPower"));
		this->precalc_TwoSiteData();
	};
	
	PeierlsHubbardU1xZN (const size_t &L, const vector<Param> &params, const BC &boundary=BC::OPEN, const DMRG::VERBOSITY::OPTION &VERB=DMRG::VERBOSITY::OPTION::ON_EXIT);
	
	static qarray<1> singlet (int N=0) {return qarray<1>{N};};
	static constexpr MODEL_FAMILY FAMILY = HUBBARD;
	static constexpr int spinfac = 2;
	
	static const map<string,any> defaults;
	static const map<string,any> sweep_defaults;
};

// V is standard next-nearest neighbour density interaction
// Vz and Vxy are anisotropic isospin-isospin next-nearest neighbour interaction
const map<string,any> PeierlsHubbardU1xZN::defaults = 
{
	{"t",1.+0.i}, {"tPrime",0.+0.i}, {"tRung",1.+0.i}, 
	{"mu",0.}, {"t0",0.}, 
	{"U",0.}, {"Uph",0.},
	{"V",0.}, {"Vrung",0.}, 
	{"Vxy",0.}, {"Vz",0.},
	{"Bz",0.}, {"Bx",0.}, 
	{"J",0.}, {"Jperp",0.}, {"J3site",0.},
	{"X",0.}, {"Xperp",0.},
	{"REMOVE_DOUBLE",false}, {"REMOVE_EMPTY",false}, {"REMOVE_UP",false}, {"REMOVE_DN",false}, {"mfactor",1}, {"k",0},
	{"maxPower",2ul}, {"CYLINDER",false}, {"Ly",1ul}
};

const map<string,any> PeierlsHubbardU1xZN::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.}, {"lim_alpha",11ul}, {"eps_svd",1e-7},
	{"Mincr_abs", 50ul}, {"Mincr_per", 2ul}, {"Mincr_rel", 1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",24ul}, {"min_halfsweeps",1ul},
	{"Minit",2ul}, {"Qinit",2ul}, {"Mlimit",1000ul},
	{"tol_eigval",1e-7}, {"tol_state",1e-6},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST",DMRG::CONVTEST::VAR_2SITE}
};

PeierlsHubbardU1xZN::
PeierlsHubbardU1xZN (const size_t &L, const vector<Param> &params, const BC &boundary, const DMRG::VERBOSITY::OPTION &VERB)
:Mpo<Symmetry,complex<double>> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, boundary, VERB),
 HubbardObservables(L,params,PeierlsHubbardU1xZN::defaults),
 ParamReturner(PeierlsHubbardU1xZN::sweep_defaults)
{
	ParamHandler P(params,defaults);
	size_t Lcell = P.size();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(F[l].get_basis().qloc(),l);
	}
	
	this->set_name("Peierls-Hubbard");
	
	PushType<SiteOperator<Symmetry,complex<double>>,complex<double>> pushlist;
	std::vector<std::vector<std::string>> labellist;
	PeierlsHubbardU1xU1::set_operators(F, P, pushlist, labellist, boundary);
	
	this->construct_from_pushlist(pushlist, labellist, Lcell);
	this->finalize(PROP::COMPRESS, P.get<size_t>("maxPower"));
	
	this->precalc_TwoSiteData();
}

} // end namespace VMPS::models

#endif
