#ifndef VANILLA_GRANDHUBBARDMODEL
#define VANILLA_GRANDHUBBARDMODEL

#include "HubbardU1xU1.h"

namespace VMPS
{
typedef Sym::U0 Symmetry;

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
	
	Hubbard() : Mpo(){};
	Hubbard (const size_t &L, const vector<Param> &params);
	
	template<typename Symmetry_>
	static void add_operators (HamiltonianTermsXd<Symmetry_> &Terms, const FermionBase<Symmetry_> &F, const ParamHandler &P, size_t loc=0);
	
	static const std::map<string,std::any> defaults;
};

const std::map<string,std::any> Hubbard::defaults = 
{
	{"t",1.}, {"tPerp",0.}, {"tPrime",0.}, 
	{"mu",0.}, {"t0",0.}, 
	{"U",0.}, {"V",0.}, {"Vperp",0.}, 
	{"Bz",0.}, {"Bx",0.}, 
	{"J",0.}, {"Jperp",0.}, {"J3site",0.},
	{"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}
};

Hubbard::
Hubbard (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, qarray<0>({}), "", true),
 HubbardObservables(L,params,Hubbard::defaults)
{
	ParamHandler P(params,Hubbard::defaults);
	
	size_t Lcell = P.size();
	vector<SuperMatrix<Symmetry,double> > G;
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		
		setLocBasis(F[l].get_basis(),l);
		
		Terms[l] = HubbardU1xU1::set_operators(F[l],P,l%Lcell);
		add_operators(Terms[l],F[l],P,l%Lcell);
		this->Daux = Terms[l].auxdim();
		
		G.push_back(Generator(Terms[l])); // boost::multi_array has stupid assignment
		setOpBasis(G[l].calc_qOp(),l);
	}
	
	this->generate_label(Terms[0].name,Terms,Lcell);
	this->construct(G, this->W, this->Gvec, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
}

template<typename Symmetry_>
void Hubbard::
add_operators (HamiltonianTermsXd<Symmetry_> &Terms, const FermionBase<Symmetry_> &F, const ParamHandler &P, size_t loc)
{
	auto save_label = [&Terms] (string label)
	{
		if (label!="") {Terms.info.push_back(label);}
	};
	
	// Bx
	auto [Bx,Bxorb,Bxlabel] = P.fill_array1d<double>("Bx","Bxorb",F.orbitals(),loc);
	save_label(Bxlabel);
	
	// Can also implement superconductivity terms c*c & cdag*cdag here
	
	Terms.name = "Hubbard";
	
	Terms.local.push_back(make_tuple(1., F.HubbardHamiltonian(F.ZeroField(),F.ZeroField(),F.ZeroField(),Bxorb,0.,0.,0., P.get<bool>("CYLINDER"))));
}

}

#endif
