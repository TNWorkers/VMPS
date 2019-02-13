#ifndef HEISENBERG_EXTENDED_TO_KITAEVCHAIN
#define HEISENBERG_EXTENDED_TO_KITAEVCHAIN

#include "models/HeisenbergU1.h"
#include "symmetry/ZN.h"

namespace VMPS
{

/** \class KitaevChain
  * \ingroup Spinless
  *
  * \brief KitaevChain Model
  *
  * MPO representation of
  * \f[
  * H =  -t \sum_{<ij>} \left(c^\dagger_i c_j + h.c.\right) 
  *      +\Delta \sum_{<ij>} \left(c^\dagger_i c_j + h.c.\right) 
  *      -\mu \sum_i n_i
  * \f]
  *
  * \note Internally, a Heisenberg model is used with the following relations: \f$S^+=c\f$, \f$S^-=c^{\dagger}\f$, \f$S^z=1/2-n\f$, \f$F=2S^z\f$
  * \note Uses the Z(2) symmetry, corresponding to the parity of the fermions (odd or even).
  * \note The default variable settings can be seen in \p KitaevChain::defaults.
  */
class KitaevChain : public Mpo<Sym::ZN<Sym::ChargeZ2,2> >, public HeisenbergObservables<Sym::ZN<Sym::ChargeZ2,2> >, public ParamReturner
{
public:
	typedef Sym::ZN<Sym::ChargeZ2,2> Symmetry;
	MAKE_TYPEDEFS(KitaevChain)
	
private:
	typedef typename Symmetry::qType qType;
	
public:
	
	///@{
	KitaevChain() : Mpo<Symmetry>(), HeisenbergObservables(), ParamReturner(KitaevChain::sweep_defaults) {};
	KitaevChain (const size_t &L, const vector<Param> &params);
	///@}
	
	template<typename Symmetry_>
	static void add_operators (const std::vector<SpinBase<Symmetry_>> &B, const ParamHandler &P, HamiltonianTermsXd<Symmetry_> &Terms);
	
	static const std::map<string,std::any> defaults;
	static const std::map<string,std::any> sweep_defaults;
};

const std::map<string,std::any> KitaevChain::defaults = 
{
	{"J",0.}, {"Jprime",0.}, {"Jrung",0.},
	{"Bz",0.}, {"Bx",0.},
	{"Kz",0.}, {"Kx",0.},
	{"Dy",0.}, {"Dyprime",0.}, {"Dyrung",0.}, // Dzialoshinsky-Moriya terms
	{"t",1.}, {"mu",0.}, {"Delta",0.}, // Kitaev chain terms
	{"D",2ul}, {"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}
};

const std::map<string,std::any> KitaevChain::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.e-11}, {"lim_alpha",10ul}, {"eps_svd",1.e-7},
	{"Dincr_abs", 4ul}, {"Dincr_per", 2ul}, {"Dincr_rel", 1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",40ul}, {"min_halfsweeps",1ul},
	{"Dinit",10ul}, {"Qinit",2ul}, {"Dlimit",1000ul},
	{"tol_eigval",1.e-5}, {"tol_state",1.e-5},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
};

KitaevChain::
KitaevChain (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, PROP::HAMILTONIAN),
 HeisenbergObservables(L,params,KitaevChain::defaults),
 ParamReturner(KitaevChain::sweep_defaults)
{
	ParamHandler P(params,KitaevChain::defaults);
	size_t Lcell = P.size();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(B[l].get_basis(),l);
	}
	
	HamiltonianTermsXd<Symmetry> Terms(N_sites, P.get<bool>("OPEN_BC"));
	HeisenbergU1::set_operators(B,P,Terms);
	add_operators(B,P,Terms);
	
	this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
	this->precalc_TwoSiteData();
}

template<typename Symmetry_>
void KitaevChain::
add_operators (const std::vector<SpinBase<Symmetry_>> &B, const ParamHandler &P, HamiltonianTermsXd<Symmetry_> &Terms)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = Terms.size();
	
	if (P.HAS_ANY_OF({"Dy", "Dyperp", "Dyprime"}))
	{
		Terms.set_name("Dzyaloshinsky-Moriya");
	}
	else if (P.HAS_ANY_OF({"mu", "Delta"}))
	{
		Terms.set_name("KitaevChain");
	}
	else
	{
		Terms.set_name("Heisenberg");
	}
	
	for (std::size_t loc=0; loc<N_sites; ++loc)
	{
		size_t lp1 = (loc+1)%N_sites;
		size_t lp2 = (loc+2)%N_sites;
		
		std::size_t orbitals       = B[loc].orbitals();
		std::size_t next_orbitals  = B[lp1].orbitals();
		std::size_t nextn_orbitals = B[lp2].orbitals();
		
		param1d mu = P.fill_array1d<double>("mu", "muorb", orbitals, loc%Lcell);
		Terms.save_label(loc, mu.label);
		
		ArrayXd Bz_array = B[loc].ZeroField();
		ArrayXd Bx_array = B[loc].ZeroField();
		ArrayXd Kx_array = B[loc].ZeroField();
		ArrayXd Kz_array = B[loc].ZeroField();
		ArrayXd Dy_array = B[loc].ZeroField();
		ArrayXXd Jperp_array = B[loc].ZeroHopping();
		
		Terms.push_local(loc, 1., B[loc].HeisenbergHamiltonian(Jperp_array, Jperp_array, Bz_array, Bx_array, mu.a, Kz_array, Kx_array, Dy_array));
		
		// Nearest-neighbour terms: p-wave SC and hopping
		
		param2d DeltaPara = P.fill_array2d<double>("Delta", "DeltaPara", {orbitals, next_orbitals}, loc%Lcell);
		Terms.save_label(loc, DeltaPara.label);
		
		param2d tPara = P.fill_array2d<double>("t", "tPara", {orbitals, next_orbitals}, loc%Lcell);
		Terms.save_label(loc, tPara.label);
		
		if (loc < N_sites-1 or !P.get<bool>("OPEN_BC"))
		{
			for (std::size_t alfa=0; alfa < orbitals;      ++alfa)
			for (std::size_t beta=0; beta < next_orbitals; ++beta)
			{
				// variant 1:
//				Terms.push_tight(loc, +DeltaPara(alfa,beta), B[loc].Scomp(SP,alfa)*B[loc].sign(), B[lp1].Scomp(SP,beta));
//				Terms.push_tight(loc, -DeltaPara(alfa,beta), B[loc].Scomp(SM,alfa)*B[loc].sign(), B[lp1].Scomp(SM,beta));
//				
//				Terms.push_tight(loc, -tPara(alfa,beta), B[loc].Scomp(SP,alfa)*B[loc].sign(), B[lp1].Scomp(SM,beta));
//				Terms.push_tight(loc, +tPara(alfa,beta), B[loc].Scomp(SM,alfa)*B[loc].sign(), B[lp1].Scomp(SP,beta));
				
				// variant 2:
				// t=-Delta is mappable to Ising chain: S^x_i*S^x_j = 1/4*(S^+_i+S^-_j)*(S^+_j+S^-_i) = 1/4*(c†^i*c^j + c^i*c^j + h.c.)
				if ((tPara.a+DeltaPara.a).matrix().norm() < 1e-14)
				{
					Terms.push_tight(loc, +4.*DeltaPara(alfa,beta), B[loc].Scomp(SX,alfa), B[lp1].Scomp(SX,beta));
				}
				else
				{
					Terms.push_tight(loc, +DeltaPara(alfa,beta), B[loc].Scomp(SP,alfa), B[lp1].Scomp(SP,beta));
					Terms.push_tight(loc, +DeltaPara(alfa,beta), B[loc].Scomp(SM,alfa), B[lp1].Scomp(SM,beta));
					
					Terms.push_tight(loc, -tPara(alfa,beta), B[loc].Scomp(SP,alfa), B[lp1].Scomp(SM,beta));
					Terms.push_tight(loc, -tPara(alfa,beta), B[loc].Scomp(SM,alfa), B[lp1].Scomp(SP,beta));
				}
			}
		}
	}
}

} // end namespace VMPS

#endif
