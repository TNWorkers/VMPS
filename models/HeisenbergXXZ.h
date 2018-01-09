#ifndef VANILLA_HEISENBERGXXZ
#define VANILLA_HEISENBERGXXZ

#include "models/Heisenberg.h"
#include "models/HeisenbergU1XXZ.h"

namespace VMPS
{

class HeisenbergXXZ : public MpoQ<Sym::U0,double>
{
public:
	typedef Sym::U0 Symmetry;
	
private:
	typedef typename Symmetry::qType qType;
	
public:
	
	///\{
	HeisenbergXXZ() : MpoQ<Symmetry>() {};
	HeisenbergXXZ (const size_t &L, const vector<Param> &params);
	///\}
	
	///@{
	/**Observables.*/
	MpoQ<Symmetry> SzSz (size_t loc1, size_t loc2);
	MpoQ<Symmetry> Sz   (size_t loc);
	///@}
	
	static const std::map<string,std::any> defaults;
	
protected:
	
	vector<SpinBase<Symmetry> > B;
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
:MpoQ<Symmetry> (L, qarray<0>({}), labeldummy, "")
{
	ParamHandler P(params,HeisenbergXXZ::defaults);
	
	size_t Lcell = P.size();
	vector<SuperMatrix<Symmetry,double> > G;
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	B.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		
		B[l] = SpinBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), P.get<size_t>("D",l%Lcell));
		setLocBasis(B[l].get_basis(),l);
		
		Terms[l] = HeisenbergU1::set_operators(B[l],P,l%Lcell);
		Heisenberg::add_operators(Terms[l],B[l],P,l%Lcell);
		HeisenbergU1XXZ::add_operators(Terms[l],B[l],P,l%Lcell);
		this->Daux = Terms[l].auxdim();
		
		G.push_back(Generator(Terms[l]));
		setOpBasis(G[l].calc_qOp(),l);
	}
	
	this->generate_label(Terms[0].name,Terms,Lcell);
	this->construct(G, this->W, this->Gvec, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
}

MpoQ<Sym::U0> HeisenbergXXZ::
Sz (size_t loc)
{
	assert(loc<N_sites);
	stringstream ss;
	ss << "Sz(" << loc << ")";
	MpoQ<Symmetry > Mout(N_sites, qarray<0>{}, labeldummy, "");
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(B[l].get_basis(),l); }
	Mout.setLocal(loc, B[loc].Scomp(SZ));
	return Mout;
}

MpoQ<Sym::U0> HeisenbergXXZ::
SzSz (size_t loc1, size_t loc2)
{
	assert(loc1<N_sites and loc2<N_sites);
	stringstream ss;
	ss << "Sz(" << loc1 << ")" <<  "Sz(" << loc2 << ")";
	MpoQ<Symmetry > Mout(N_sites, qarray<0>{}, labeldummy, "");
	for (size_t l=0; l<N_sites; ++l) { Mout.setLocBasis(B[l].get_basis(),l); }
	Mout.setLocal({loc1, loc2}, {B[loc1].Scomp(SZ), B[loc2].Scomp(SZ)});
	return Mout;
}

} // end namespace VMPS

#endif
