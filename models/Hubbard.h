#ifndef VANILLA_GRANDHUBBARDMODEL
#define VANILLA_GRANDHUBBARDMODEL

#include "HubbardU1xU1.h"
#include "LiebWu.h" // from TOOLS, depends on gsl

namespace VMPS
{
/**
 * \class Hubbard
 * \ingroup Hubbard
 * \brief Hubbard model without any symmetries.
 * MPO representation of the Hubbard model corresponding to HubbardU1xU1, but without symmetries and an additional possibility of adding
 * \f[
 * 	-B_x \sum_{i} \sigma^x_i
 * \f]
 * with
 * \f[
 * 	\sigma^x_i = \frac{1}{2} \left(\sigma^+_i+\sigma^-_i\right)
 * \f]
 * but is mainly needed for VUMPS.
 * \note The default variable settings can be seen in \p Hubbard::defaults.
 */
class Hubbard : public Mpo<Sym::U0,double>, public HubbardObservables<Sym::U0>
{
public:
	typedef Sym::U0 Symmetry;
	
	Hubbard() : Mpo() {};
	Hubbard (const size_t &L, const vector<Param> &params);
	
	template<typename Symmetry_>
	static void add_operators (HamiltonianTermsXd<Symmetry_> &Terms, const vector<FermionBase<Symmetry_> > &F, const ParamHandler &P, size_t loc=0);
	
	static const std::map<string,std::any> defaults;
	
	static refEnergy ref (const vector<Param> &params, double L=numeric_limits<double>::infinity());
};

const std::map<string,std::any> Hubbard::defaults = 
{
	{"t",1.}, {"tPrime",0.}, {"tRung",1.},
	{"mu",0.}, {"t0",0.}, 
	{"U",0.},
	{"V",0.}, {"Vrung",0.}, 
	{"Bz",0.}, {"Bx",0.}, 
	{"J",0.}, {"Jrung",0.},
	{"J3site",0.},
	{"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}
};

Hubbard::
Hubbard (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, qarray<0>({}), "", true),
 HubbardObservables(L,params,Hubbard::defaults)
{
	ParamHandler P(params,Hubbard::defaults);
	
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
		add_operators(Terms[l],F,P,l%Lcell);
	}
	
	this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
}

template<typename Symmetry_>
void Hubbard::
add_operators (HamiltonianTermsXd<Symmetry_> &Terms, const vector<FermionBase<Symmetry_> > &F, const ParamHandler &P, size_t loc)
{
	auto save_label = [&Terms] (string label)
	{
		if (label!="") {Terms.info.push_back(label);}
	};
	
	// Bx
	auto [Bx,Bxorb,Bxlabel] = P.fill_array1d<double>("Bx","Bxorb",F[loc].orbitals(),loc);
	save_label(Bxlabel);
	
	// Can also implement superconductivity terms c*c & cdag*cdag here
	
	Terms.name = "Hubbard";
	
	ArrayXd  Uorb  = F[loc].ZeroField();
	ArrayXd  Eorb  = F[loc].ZeroField();
	ArrayXd  Bzorb = F[loc].ZeroField();
	ArrayXXd tPerp = F[loc].ZeroHopping();
	ArrayXXd Vperp = F[loc].ZeroHopping();
	ArrayXXd Jperp = F[loc].ZeroHopping();
	
	Terms.local.push_back(make_tuple(1., F[loc].template HubbardHamiltonian<double>(Uorb,Eorb,Bzorb,Bxorb,tPerp,Vperp,Jperp)));
}

refEnergy Hubbard::
ref (const vector<Param> &params, double L)
{
	ParamHandler P(params,{{"n",1.},{"Ly",1ul},{"U",0.}});
	refEnergy out;
	
	if (isinf(L) and P.get<size_t>("Ly") == 1ul and P.get<double>("n") == 1. and P.HAS_NONE_OF({"tPrime","t0","V","Bz","Bx","J","J3site"}))
	{
		out.value = LiebWu_e0(P.get<double>("U"));
		out.source = "Elliott H. Lieb, F. Y. Wu, Absence of Mott Transition in an Exact Solution of the Short-Range, One-Band Model in One Dimension, Phys. Rev. Lett. 20, 1445 (1968)";
		out.method = "numerical integration";
	}
	
	return out;
}

}

#endif
