#ifndef VANILLA_HEISENBERGXXZ
#define VANILLA_HEISENBERGXXZ

#include "models/Heisenberg.h"
#include "models/HeisenbergU1XXZ.h"

namespace VMPS
{

class HeisenbergXXZ : public Mpo<Sym::U0,double>, public HeisenbergObservables<Sym::U0>
{
public:
	typedef Sym::U0 Symmetry;
	
private:
	typedef typename Symmetry::qType qType;
	
public:
	
	///\{
	HeisenbergXXZ() : Mpo<Symmetry>() {};
	HeisenbergXXZ (const size_t &L, const vector<Param> &params);
	///\}
	
	static const std::map<string,std::any> defaults;
};

const std::map<string,std::any> HeisenbergXXZ::defaults = 
{
	{"Jxy",0.}, {"Jz",0.},
	{"Jxyprime",0.}, {"Jzprime",0.},
	{"Jxyperp",0.}, {"Jzperp",0.},
	
	{"Bz",0.}, {"Bx",0.}, {"Kz",0.}, {"Kx",0.},
	{"Dy",0.}, {"Dyperp",0.}, {"Dyprime",0.},
	{"D",2ul}, {"Bz",0.}, {"Kz",0.},
	{"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}, 
	
	// for consistency during inheritance (should not be set for XXZ!):
	{"J",0.}, {"Jprime",0.}, {"Jperp",0.}, {"Jpara",0.}
};


HeisenbergXXZ::
HeisenbergXXZ (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, qarray<0>({}), "", true),
 HeisenbergObservables(L,params,HeisenbergXXZ::defaults)
{
	ParamHandler P(params,HeisenbergXXZ::defaults);
	
	size_t Lcell = P.size();
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		
		setLocBasis(B[l].get_basis(),l);
		
		Terms[l] = HeisenbergU1::set_operators(B[l],P,l%Lcell);
		Heisenberg::add_operators(Terms[l],B[l],P,l%Lcell);
		HeisenbergU1XXZ::add_operators(Terms[l],B[l],P,l%Lcell);
	}
	
	this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
}

} // end namespace VMPS

#endif
