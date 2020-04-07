#ifndef VANILLA_HEISENBERGXXZ
#define VANILLA_HEISENBERGXXZ

#include "models/HeisenbergU1XXZ.h"
#include "models/Heisenberg.h" // for defaults etc.

namespace VMPS
{

class HeisenbergXXZ : public Mpo<Sym::U0,double>, public HeisenbergObservables<Sym::U0>, public ParamReturner
{
public:
	typedef Sym::U0 Symmetry;
	MAKE_TYPEDEFS(HeisenbergXXZ)
	
	static qarray<0> singlet() {return qarray<0>{};};
	
private:
	
	typedef typename Symmetry::qType qType;
	
public:
	
	///\{
	HeisenbergXXZ() : Mpo<Symmetry>(), ParamReturner(Heisenberg::defaults) {};
	HeisenbergXXZ (const size_t &L, const vector<Param> &params, const BC &boundary=BC::OPEN, const DMRG::VERBOSITY::OPTION &VERB=DMRG::VERBOSITY::OPTION::ON_EXIT);
	///\}
	
	static const std::map<string,std::any> defaults;
};

const std::map<string,std::any> HeisenbergXXZ::defaults = 
{
	{"Jxy",1.}, {"Jxyprime",0.}, {"Jxyrung",1.},
	{"Jz",0.}, {"Jzprime",0.}, {"Jzrung",0.},
	
	{"Bz",0.}, {"Bx",0.}, 
	{"Kz",0.}, {"Kx",0.},
	{"Dy",0.}, {"Dyprime",0.}, {"Dyrung",0.},
	{"D",2ul}, {"maxPower",2ul}, {"CYLINDER",false}, {"Ly",1ul}, 
	
	// for consistency during inheritance (should not be set!):
	{"J",0.}, {"Jprime",0.}
};

HeisenbergXXZ::
HeisenbergXXZ (const size_t &L, const vector<Param> &params, const BC &boundary, const DMRG::VERBOSITY::OPTION &VERB)
:Mpo<Symmetry> (L, qarray<0>({}), "", PROP::HERMITIAN, PROP::NON_UNITARY, boundary, VERB),
 HeisenbergObservables(L,params,HeisenbergXXZ::defaults),
 ParamReturner(Heisenberg::sweep_defaults)
{
	ParamHandler P(params,HeisenbergXXZ::defaults);
	size_t Lcell = P.size();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(B[l].get_basis().qloc(),l);
	}

	if (P.HAS_ANY_OF({"Jxy", "Jxypara", "Jxyperp", "Jxyfull"}))
	{
		this->set_name("XXZ");
	}
	else
	{
		this->set_name("Ising");
	}

	PushType<SiteOperator<Symmetry,double>,double> pushlist;
    std::vector<std::vector<std::string>> labellist;
	
	HeisenbergU1::set_operators(B,P,pushlist,labellist,boundary);
	Heisenberg::add_operators(B,P,pushlist,labellist,boundary);
	HeisenbergU1XXZ::add_operators(B,P,pushlist,labellist,boundary);
	
	this->construct_from_pushlist(pushlist, labellist, Lcell);
    this->finalize(PROP::COMPRESS, P.get<size_t>("maxPower"));
	
	this->precalc_TwoSiteData();
}

} // end namespace VMPS

#endif
