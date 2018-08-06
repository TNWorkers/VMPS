#ifndef STRAWBERRY_HUBBARDMODEL_NM
#define STRAWBERRY_HUBBARDMODEL_NM

#include "models/HubbardU1xU1.h"

namespace VMPS
{
class HubbardU1xU1MN : public Mpo<Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::ChargeU1> >,double>,
                       public HubbardObservables<Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::ChargeU1> > >, 
                       public ParamReturner
{
public:
	
	typedef Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::ChargeU1> > Symmetry;
	MAKE_TYPEDEFS(HubbardU1xU1MN)
	
	///@{
	HubbardU1xU1MN() : Mpo(){};
	HubbardU1xU1MN (const size_t &L, const vector<Param> &params);
	///@}
	
	static qarray<2> singlet (int N) {return qarray<2>{N,0};};
	
	/**Default parameters.*/
	static const std::map<string,std::any> defaults;
};

const std::map<string,std::any> HubbardU1xU1MN::defaults = HubbardU1xU1::defaults;

HubbardU1xU1MN::
HubbardU1xU1MN (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", true),
 HubbardObservables(L,params,HubbardU1xU1MN::defaults),
 ParamReturner()
{
	ParamHandler P(params,HubbardU1xU1MN::defaults);
	
	size_t Lcell = P.size();
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(F[l].get_basis(),l);
	}
	
	for (size_t l=0; l<N_sites; ++l)
	{
		Terms[l] = HubbardU1xU1::set_operators(F,P,l%Lcell);
		
		stringstream ss;
		ss << "Ly=" << P.get<size_t>("Ly",l%Lcell);
		Terms[l].info.push_back(ss.str());
	}
	
	this->construct_from_Terms(Terms, Lcell, false, P.get<bool>("OPEN_BC"));
	this->precalc_TwoSiteData();
}

} // end namespace VMPS::models

#endif
