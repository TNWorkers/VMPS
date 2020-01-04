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

class HubbardU1spin : public Mpo<Sym::U1<Sym::SpinU1>,double>, public HubbardObservables<Sym::U1<Sym::SpinU1> >, public ParamReturner
{
public:
	
	typedef Sym::U1<Sym::SpinU1> Symmetry;
	MAKE_TYPEDEFS(HubbardU1spin)
	
	///@{
	HubbardU1spin() : Mpo(){};
	HubbardU1spin (const size_t &L, const vector<Param> &params);
	///@}
	
	/**Default parameters.*/
	static const std::map<string,std::any> defaults;
};

const std::map<string,std::any> HubbardU1spin::defaults = 
{
	{"t",1.}, {"tPrime",0.}, {"tRung",1.},
	{"mu",0.}, {"t0",0.}, 
	{"U",0.}, {"Uph",0.},
	{"V",0.}, {"Vrung",0.},
	{"Vxy",0.}, {"Vz",0.},
	{"Bz",0.}, 
	{"J",0.}, {"Jperp",0.}, {"J3site",0.},
	{"X",0.}, {"Xperp",0.},
	{"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}
};

HubbardU1spin::
HubbardU1spin (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, PROP::HAMILTONIAN),
 HubbardObservables(L,params,HubbardU1spin::defaults),
 ParamReturner()
{
	ParamHandler P(params,HubbardU1spin::defaults);
	
	size_t Lcell = P.size();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(F[l].get_basis(),l);
	}
	
	HamiltonianTermsXd<Symmetry> Terms(N_sites, P.get<bool>("OPEN_BC"));
	HubbardU1xU1::set_operators<Symmetry>(F,P,Terms);
	
	this->construct_from_Terms(Terms, Lcell, false, P.get<bool>("OPEN_BC"));
	this->precalc_TwoSiteData();
}

} // end namespace VMPS::models

#endif
