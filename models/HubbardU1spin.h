#ifndef STRAWBERRY_HUBBARDMODEL_U1SPINONLY
#define STRAWBERRY_HUBBARDMODEL_U1SPINONLY

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
	HubbardU1spin (const size_t &L, const vector<Param> &params, const BC &boundary=BC::OPEN, const DMRG::VERBOSITY::OPTION &VERB=DMRG::VERBOSITY::OPTION::ON_EXIT);
	///@}
	
	static qarray<1> singlet (int N=0) {return qarray<1>{0};};
	static constexpr MODEL_FAMILY FAMILY = HUBBARD;
	static constexpr int spinfac = 2;
	
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
	{"maxPower",2ul}, {"CYLINDER",false}, {"Ly",1ul}
};

HubbardU1spin::
HubbardU1spin (const size_t &L, const vector<Param> &params, const BC &boundary, const DMRG::VERBOSITY::OPTION &VERB)
:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, boundary, VERB),
 HubbardObservables(L,params,HubbardU1spin::defaults),
 ParamReturner()
{
	ParamHandler P(params,HubbardU1spin::defaults);
	
	size_t Lcell = P.size();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(F[l].get_basis().qloc(),l);
	}
	
	param1d U = P.fill_array1d<double>("U", "Uorb", F[0].orbitals(), 0);	
	if (isfinite(U.a.sum()))
	{
		this->set_name("Hubbard");
	}
	else if (P.HAS_ANY_OF({"J", "J3site"}))
	{
		this->set_name("t-J");
	}
	else
	{
		this->set_name("U=∞-Hubbard");
	}
	
	PushType<SiteOperator<Symmetry,double>,double> pushlist;
    std::vector<std::vector<std::string>> labellist;
	HubbardU1xU1::set_operators(F, P, pushlist, labellist, boundary);

	this->construct_from_pushlist(pushlist, labellist, Lcell);
    this->finalize(PROP::COMPRESS, P.get<size_t>("maxPower"));

	this->precalc_TwoSiteData();
}

} // end namespace VMPS::models

#endif
