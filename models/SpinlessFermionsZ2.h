#ifndef STRAWBERRY_SPINLESSFERMIONSZ2
#define STRAWBERRY_SPINLESSFERMIONSZ2

#include "symmetry/ZN.h"
#include "models/SpinlessFermionsU1.h"

namespace VMPS
{

/** \class SpinlessFermionsZ2
  * \ingroup Spinless
  *
  * \brief SpinlessFermionsZ2 Model
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
  * \note The default variable settings can be seen in \p SpinlessFermionsZ2::defaults.
  */
class SpinlessFermionsZ2 : public Mpo<Sym::ZN<Sym::ChargeZ2,2> >, public SpinlessFermionsObservables<Sym::ZN<Sym::ChargeZ2,2> >, public ParamReturner
{
public:
	typedef Sym::ZN<Sym::ChargeZ2,2> Symmetry;
	MAKE_TYPEDEFS(SpinlessFermionsZ2)
	
private:
	typedef typename Symmetry::qType qType;
	
public:
	
	///@{
	SpinlessFermionsZ2() : Mpo<Symmetry>(), SpinlessFermionsObservables(), ParamReturner(SpinlessFermionsZ2::sweep_defaults) {};
	SpinlessFermionsZ2 (const size_t &L, const vector<Param> &params);
	///@}
	
	template<typename Symmetry_>
	static void add_operators (const std::vector<SpinlessFermionBase<Symmetry_> > &B, const ParamHandler &P, HamiltonianTermsXd<Symmetry_> &Terms);
	
	static const std::map<string,std::any> defaults;
	static const std::map<string,std::any> sweep_defaults;
};

const std::map<string,std::any> SpinlessFermionsZ2::defaults = 
{
	{"t",1.}, {"Delta",0.}, {"tPrime",0.},
	{"V",0.}, {"Vph",0.},
	{"Vprime",0.}, {"VphPrime",0.},
	{"mu",0.},{"t0",0.},
	{"D",2ul}, {"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}, 
};

const std::map<string,std::any> SpinlessFermionsZ2::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.e-11}, {"lim_alpha",10ul}, {"eps_svd",1.e-7},
	{"Dincr_abs", 4ul}, {"Dincr_per", 2ul}, {"Dincr_rel", 1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",40ul}, {"min_halfsweeps",1ul},
	{"Dinit",10ul}, {"Qinit",2ul}, {"Dlimit",1000ul},
	{"tol_eigval",1.e-5}, {"tol_state",1.e-5},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
};

SpinlessFermionsZ2::
SpinlessFermionsZ2 (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, PROP::HAMILTONIAN),
 SpinlessFermionsObservables(L,params,SpinlessFermionsZ2::defaults),
 ParamReturner(SpinlessFermionsZ2::sweep_defaults)
{
	ParamHandler P(params,SpinlessFermionsZ2::defaults);
	size_t Lcell = P.size();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(F[l].get_basis(),l);
	}
	
	HamiltonianTermsXd<Symmetry> Terms(N_sites, P.get<bool>("OPEN_BC"));
	SpinlessFermionsU1::set_operators(F,P,Terms);
	add_operators(F,P,Terms);
	
	this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
	this->precalc_TwoSiteData();
}

template<typename Symmetry_>
void SpinlessFermionsZ2::
add_operators (const std::vector<SpinlessFermionBase<Symmetry_> > &F, const ParamHandler &P, HamiltonianTermsXd<Symmetry_> &Terms)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = Terms.size();
	
	Terms.set_name("SpinlessFermionsZ2");
	
	for (std::size_t loc=0; loc<N_sites; ++loc)
	{
		size_t lp1 = (loc+1)%N_sites;
		size_t lp2 = (loc+2)%N_sites;
		
		std::size_t orbitals       = F[loc].orbitals();
		std::size_t next_orbitals  = F[lp1].orbitals();
		std::size_t nextn_orbitals = F[lp2].orbitals();
		
		// Nearest-neighbour terms: p-wave SC
		
		param2d DeltaPara = P.fill_array2d<double>("Delta", "DeltaPara", {orbitals, next_orbitals}, loc%Lcell);
		Terms.save_label(loc, DeltaPara.label);
		
		if (loc < N_sites-1 or !P.get<bool>("OPEN_BC"))
		{
			for (std::size_t alfa=0; alfa < orbitals;      ++alfa)
			for (std::size_t beta=0; beta < next_orbitals; ++beta)
			{
				Terms.push_tight(loc, +DeltaPara(alfa,beta), F[loc].cdag(alfa) * F[loc].sign(), F[lp1].cdag(beta));
				Terms.push_tight(loc, -DeltaPara(alfa,beta), F[loc].c(alfa)    * F[loc].sign(), F[lp1].c(beta));
				
				// Old: mapped to Heisenberg
				// variant 1:
//				Terms.push_tight(loc, +DeltaPara(alfa,beta), B[loc].Scomp(SP,alfa)*B[loc].sign(), B[lp1].Scomp(SP,beta));
//				Terms.push_tight(loc, -DeltaPara(alfa,beta), B[loc].Scomp(SM,alfa)*B[loc].sign(), B[lp1].Scomp(SM,beta));
//				
//				Terms.push_tight(loc, -tPara(alfa,beta), B[loc].Scomp(SP,alfa)*B[loc].sign(), B[lp1].Scomp(SM,beta));
//				Terms.push_tight(loc, +tPara(alfa,beta), B[loc].Scomp(SM,alfa)*B[loc].sign(), B[lp1].Scomp(SP,beta));
				
				// variant 2:
				// t=-Delta is mappable to Ising chain: S^x_i*S^x_j = 1/4*(S^+_i+S^-_j)*(S^+_j+S^-_i) = 1/4*(c†^i*c^j + c^i*c^j + h.c.)
//				if ((tPara.a+DeltaPara.a).matrix().norm() < 1e-14)
//				{
//					Terms.push_tight(loc, +4.*DeltaPara(alfa,beta), B[loc].Scomp(SX,alfa), B[lp1].Scomp(SX,beta));
//				}
//				else
//				{
//					Terms.push_tight(loc, +DeltaPara(alfa,beta), B[loc].Scomp(SP,alfa), B[lp1].Scomp(SP,beta));
//					Terms.push_tight(loc, +DeltaPara(alfa,beta), B[loc].Scomp(SM,alfa), B[lp1].Scomp(SM,beta));
//					
//					Terms.push_tight(loc, -tPara(alfa,beta), B[loc].Scomp(SP,alfa), B[lp1].Scomp(SM,beta));
//					Terms.push_tight(loc, -tPara(alfa,beta), B[loc].Scomp(SM,alfa), B[lp1].Scomp(SP,beta));
//				}
			}
		}
	}
}

} // end namespace VMPS

#endif
