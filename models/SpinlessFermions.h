#ifndef STRAWBERRY_SpinlessFermions
#define STRAWBERRY_SpinlessFermions

#include "symmetry/U0.h"
#include "models/SpinlessFermionsU1.h"
#include "models/SpinlessFermionsZ2.h"

namespace VMPS
{

/** \class SpinlessFermions
  * \ingroup Spinless
  *
  * \brief SpinlessFermions Model
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
  * \note The default variable settings can be seen in \p SpinlessFermions::defaults.
  */
class SpinlessFermions : public Mpo<Sym::U0>, public SpinlessFermionsObservables<Sym::U0>, public ParamReturner
{
public:
	typedef Sym::U0 Symmetry;
	MAKE_TYPEDEFS(SpinlessFermions)
	
private:
	typedef typename Symmetry::qType qType;
	
public:
	
	///@{
	SpinlessFermions() : Mpo<Symmetry>(), SpinlessFermionsObservables(), ParamReturner(SpinlessFermions::sweep_defaults) {};
	SpinlessFermions (const size_t &L, const vector<Param> &params);
	///@}
	
	static qarray<0> singlet (int N) {return qarray<0>{};};
	static MODEL_FAMILY FAMILY = SPINLESS;
	
	template<typename Symmetry_>
	static void add_operators (const std::vector<SpinlessFermionBase<Symmetry_> > &B, const ParamHandler &P, HamiltonianTermsXd<Symmetry_> &Terms);
	
	static const std::map<string,std::any> defaults;
	static const std::map<string,std::any> sweep_defaults;
};

const std::map<string,std::any> SpinlessFermions::defaults = 
{
	{"t",1.}, {"Delta",0.}, {"tPrime",0.},
	{"V",0.}, {"Vph",0.},
	{"Vprime",0.}, {"VphPrime",0.},
	{"mu",0.},{"t0",0.},
	{"D",2ul}, {"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}, 
};

const std::map<string,std::any> SpinlessFermions::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.e-11}, {"lim_alpha",10ul}, {"eps_svd",1.e-7},
	{"Dincr_abs", 4ul}, {"Dincr_per", 2ul}, {"Dincr_rel", 1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",40ul}, {"min_halfsweeps",1ul},
	{"Dinit",10ul}, {"Qinit",2ul}, {"Dlimit",1000ul},
	{"tol_eigval",1.e-5}, {"tol_state",1.e-5},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
};

SpinlessFermions::
SpinlessFermions (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, PROP::HAMILTONIAN),
 SpinlessFermionsObservables(L,params,SpinlessFermions::defaults),
 ParamReturner(SpinlessFermions::sweep_defaults)
{
	ParamHandler P(params,SpinlessFermions::defaults);
	size_t Lcell = P.size();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(F[l].get_basis(),l);
	}
	
	HamiltonianTermsXd<Symmetry> Terms(N_sites, P.get<bool>("OPEN_BC"));
	SpinlessFermionsU1::set_operators(F,P,Terms);
	SpinlessFermionsZ2::add_operators(F,P,Terms);
	Terms.set_name("SpinlessFermions");
	
	this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
	this->precalc_TwoSiteData();
}

} // end namespace VMPS

#endif
