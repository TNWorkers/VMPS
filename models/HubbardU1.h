#ifndef STRAWBERRY_HUBBARDMODEL_U1CHARGEONLY
#define STRAWBERRY_HUBBARDMODEL_U1CHARGEONLY

//include "bases/FermionBase.h"
//include "symmetry/S1xS2.h"
//include "Mpo.h"
//include "ParamHandler.h" // from HELPERS
//include "models/HubbardObservables.h"
#include "models/HubbardU1xU1.h"

namespace VMPS
{

class HubbardU1 : public Mpo<Sym::U1<Sym::ChargeU1>,double>, public HubbardObservables<Sym::U1<Sym::ChargeU1> >, public ParamReturner
{
public:
	
	typedef Sym::U1<Sym::ChargeU1> Symmetry;
	MAKE_TYPEDEFS(HubbardU1)
	
	///@{
	HubbardU1() : Mpo(){};
	HubbardU1 (const size_t &L, const vector<Param> &params);
	///@}
	
	/**Default parameters.*/
	static const std::map<string,std::any> defaults;
};

const std::map<string,std::any> HubbardU1::defaults = 
{
	{"t",1.}, {"tPrime",0.}, {"tRung",1.}, 
	{"mu",0.}, {"t0",0.}, 
	{"U",0.},
	{"V",0.}, {"Vrung",0.}, 
	{"Bz",0.}, {"Bx",0.}, 
	{"J",0.}, {"Jperp",0.}, {"J3site",0.},
	{"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}
};

HubbardU1::
HubbardU1 (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, PROP::HAMILTONIAN),
 HubbardObservables(L,params,HubbardU1::defaults),
 ParamReturner()
{
	ParamHandler P(params,HubbardU1::defaults);
	
	size_t Lcell = P.size();
	//vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
    HamiltonianTermsXd<Symmetry> Terms(N_sites);
    
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(F[l].get_basis(),l);
	}
	
	/*for (size_t l=0; l<N_sites; ++l)
	{
		Terms[l] = HubbardU1xU1::set_operators(F,P,l%Lcell);
		
		stringstream ss;
		ss << "Ly=" << P.get<size_t>("Ly",l%Lcell);
		Terms[l].info.push_back(ss.str());
	}*/
    
    HubbardU1xU1::set_operators(F,P,Terms);
	
	this->construct_from_Terms(Terms, Lcell, false, P.get<bool>("OPEN_BC"));
	this->precalc_TwoSiteData();
}

} // end namespace VMPS::models

#endif
