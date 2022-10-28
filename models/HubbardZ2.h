#ifndef STRAWBERRY_HUBBARDMODEL_Z2
#define STRAWBERRY_HUBBARDMODEL_Z2

#include "symmetry/ZN.h"
#include "models/HubbardU1xU1.h"
#include "models/HubbardObservables.h"
#include "ParamReturner.h"
#include "Geometry2D.h" // from TOOLS

namespace VMPS
{
class HubbardZ2 : public Mpo<Sym::ZN<Sym::ChargeZ2,2>,double>,
                  public HubbardObservables<Sym::ZN<Sym::ChargeZ2,2> >, 
                  public ParamReturner
{
public:
	
	typedef Sym::ZN<Sym::ChargeZ2,2> Symmetry;
	MAKE_TYPEDEFS(HubbardZ2)
	
	///@{
	HubbardZ2() : Mpo(){};
	
	HubbardZ2(Mpo<Symmetry> &Mpo_input, const vector<Param> &params)
	:Mpo<Symmetry>(Mpo_input),
	 HubbardObservables(this->N_sites,params,HubbardZ2::defaults),
	 ParamReturner()
	{
		ParamHandler P(params,HubbardZ2::defaults);
		size_t Lcell = P.size();
		N_phys = 0;
		for (size_t l=0; l<N_sites; ++l) N_phys += P.get<size_t>("Ly",l%Lcell);
		this->precalc_TwoSiteData();
	};
	
	HubbardZ2 (const size_t &L, const vector<Param> &params, const BC &boundary=BC::OPEN, const DMRG::VERBOSITY::OPTION &VERB=DMRG::VERBOSITY::OPTION::ON_EXIT);
	///@}
	
	static qarray<1> singlet (int N=0) {return qarray<1>{0};};
	static constexpr MODEL_FAMILY FAMILY = HUBBARD;
	static constexpr int spinfac = 2;
	
	static const std::map<string,std::any> defaults;
};

const std::map<string,std::any> HubbardZ2::defaults = 
{
	{"t",1.}, {"tPrime",0.}, {"tRung",1.},
	{"mu",0.}, {"t0",0.},
	{"V",0.}, {"Vrung",0.},
	{"Delta",0.},
	{"REMOVE_DOUBLE",false}, {"REMOVE_EMPTY",false}, {"REMOVE_SINGLE",false}, {"mfactor",1},
	{"maxPower",2ul}, {"CYLINDER",false}, {"Ly",1ul}
};

HubbardZ2::
HubbardZ2 (const size_t &L, const vector<Param> &params, const BC &boundary, const DMRG::VERBOSITY::OPTION &VERB)
:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, boundary, VERB),
 HubbardObservables(L,params,HubbardZ2::defaults),
 ParamReturner()
{
	ParamHandler P(params, HubbardZ2::defaults);
	size_t Lcell = P.size();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(F[l].get_basis().qloc(),l);
	}
	
	this->set_name("Kitaev");
	
	PushType<SiteOperator<Symmetry,double>,double> pushlist;
	std::vector<std::vector<std::string>> labellist;
	HubbardU1xU1::set_operators(F, P, pushlist, labellist, boundary);
	
	this->construct_from_pushlist(pushlist, labellist, Lcell);
	this->finalize(PROP::COMPRESS, P.get<size_t>("maxPower"));
	this->precalc_TwoSiteData();
}

} // end namespace VMPS::models

#endif
