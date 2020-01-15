#ifndef STRAWBERRY_HUBBARDMODEL
#define STRAWBERRY_HUBBARDMODEL

//include "bases/FermionBase.h"
#include "symmetry/S1xS2.h"
#include "symmetry/U1.h"
//include "Mpo.h"
//include "ParamHandler.h" // from HELPERS
#include "models/HubbardObservables.h"
#include "ParamReturner.h"

namespace VMPS
{

/**
 * \class HubbardU1xU1
 * \ingroup Hubbard
 * \brief Hubbard model with U(1) symmetries.
 * MPO representation of the Hubbard model
 * \f[
 * 	H = -t \sum_{<ij>\sigma} \left( c^\dagger_{i\sigma}c_{j\sigma} + h.c. \right)
 * 	    -t^{\prime} \sum_{<<ij>>\sigma} \left( c^\dagger_{i\sigma}c_{j\sigma} +h.c. \right)
 * 	    +\sum_i \left(t_{0,i}-\mu\right) n_i
 * 	    +U \sum_i n_{i\uparrow} n_{i\downarrow}
 * 	    +V \sum_{<ij>} n_{i} n_{j}
 * 	    -B_z \sum_{i} \left(n_{i\uparrow}-n_{i\downarrow}\right)
 *      -X \sum_{<ij>\sigma} \left( c^\dagger_{i\sigma}c_{j\sigma} + h.c.\right) \left(n_{i,-\sigma}-n_{j,-\sigma}\right)^2
 * 	    +H_{tJ}
 * 	    +H_{3-site}
 * \f]
 * with
 * \f[
 * H_{tJ} = +J \sum_{<ij>} (\mathbf{S}_{i} \mathbf{S}_{j} - \frac{1}{4} n_in_j)
 * \f]
 * \note: The term before \f$n_i n_j\f$ is not set and has to be adjusted with \p V
 * \f[
 * H_{3-site} = -\frac{J}{4} \sum_{<ijk>\sigma} (c^\dagger_{i\sigma} n_{j,-\sigma} c_{k\sigma} - c^\dagger_{i\sigma} S^{-\sigma}_j c_{k,-\sigma} + h.c.) \
 * \f]
 * \note Makes use of the U(1) particle conservation symmetry for both spin components separatly.
 *       You can change this by choosing another symmetry class. For example, to use the magnetization and the particle number use:
 * \code{.cpp}
 *     Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::ChargeU1> >
 * \endcode
 * \note The default variable settings can be seen in \p HubbardU1xU1::defaults.
 * \note If the NNN-hopping is positive, the ground state energy is lowered.
 * \warning \f$J>0\f$ is antiferromagnetic
 */
class HubbardU1xU1 : public Mpo<Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::ChargeU1> >,double>,
                     public HubbardObservables<Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::ChargeU1> > >, 
                     public ParamReturner
{
public:
	
	typedef Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::ChargeU1> > Symmetry;
	MAKE_TYPEDEFS(HubbardU1xU1)
	
	///@{
	HubbardU1xU1() : Mpo(){};
	HubbardU1xU1 (const size_t &L, const vector<Param> &params);
	///@}
	
	static qarray<2> singlet (int N) {return qarray<2>{0,N};};
	
	template<typename Symmetry_> 
	//static HamiltonianTermsXd<Symmetry_> set_operators (const vector<FermionBase<Symmetry_> > &F, const ParamHandler &P, size_t loc=0);
	static void set_operators(const std::vector<FermionBase<Symmetry_>> &F, const ParamHandler &P, HamiltonianTermsXd<Symmetry_> &Terms);
	
	/**Default parameters.*/
	static const std::map<string,std::any> defaults;
};

const std::map<string,std::any> HubbardU1xU1::defaults = 
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

HubbardU1xU1::
HubbardU1xU1 (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, PROP::HAMILTONIAN),
 HubbardObservables(L,params,HubbardU1xU1::defaults),
 ParamReturner()
{
	ParamHandler P(params, HubbardU1xU1::defaults);
	
	size_t Lcell = P.size();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(F[l].get_basis(),l);
	}
	
	HamiltonianTermsXd<Symmetry> Terms(N_sites, P.get<bool>("OPEN_BC"));
	set_operators(F,P,Terms);
	
	this->construct_from_Terms(Terms, Lcell, false, P.get<bool>("OPEN_BC"));
	this->precalc_TwoSiteData();
}

template<typename Symmetry_>
void HubbardU1xU1::
set_operators (const std::vector<FermionBase<Symmetry_>> &F, const ParamHandler &P, HamiltonianTermsXd<Symmetry_> &Terms)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = Terms.size();
	bool U_infinite = true;
	
	for (std::size_t loc=0; loc<N_sites; ++loc)
	{
		size_t lp1 = (loc+1)%N_sites;
		size_t lp2 = (loc+2)%N_sites;
		
		std::size_t orbitals = F[loc].orbitals();
		std::size_t next_orbitals = F[lp1].orbitals();
		std::size_t nextn_orbitals = F[lp2].orbitals();
		
		stringstream ss;
		ss << "Ly=" << P.get<size_t>("Ly",loc%Lcell);
		Terms.save_label(loc, ss.str());
		
		// local terms: U, t0, μ, Bz, t⟂, V⟂, J⟂, X⟂
		
		param1d U = P.fill_array1d<double>("U", "Uorb", orbitals, loc%Lcell);
		param1d Uph = P.fill_array1d<double>("Uph", "Uorb", orbitals, loc%Lcell);
		param1d t0 = P.fill_array1d<double>("t0", "t0orb", orbitals, loc%Lcell);
		param1d mu = P.fill_array1d<double>("mu", "muorb", orbitals, loc%Lcell);
		param1d Bz = P.fill_array1d<double>("Bz", "Bzorb", orbitals, loc%Lcell);
		param2d tperp = P.fill_array2d<double>("tRung", "t", "tPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Vperp = P.fill_array2d<double>("Vrung", "V", "Vperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Jperp = P.fill_array2d<double>("Jrung", "J", "Jperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		
		Terms.save_label(loc, U.label);
		Terms.save_label(loc, Uph.label);
		Terms.save_label(loc, t0.label);
		Terms.save_label(loc, mu.label);
		Terms.save_label(loc, Bz.label);
		Terms.save_label(loc, tperp.label);
		Terms.save_label(loc, Vperp.label);
		Terms.save_label(loc, Jperp.label);
		
		ArrayXd Bx_array = F[loc].ZeroField();
		
		Terms.push_local(loc, 1., F[loc].template HubbardHamiltonian<double>(U.a, Uph.a, t0.a-mu.a, Bz.a, Bx_array, tperp.a, Vperp.a, Jperp.a));
		
		if (isfinite(U.a.sum()))
		{
			U_infinite = false;
		}
		
		// Nearest-neighbour terms: t, V, J, X
		
		param2d tpara = P.fill_array2d<double>("t", "tPara", {orbitals, next_orbitals}, loc%Lcell);
		param2d Vpara = P.fill_array2d<double>("V", "Vpara", {orbitals, next_orbitals}, loc%Lcell);
		param2d Jpara = P.fill_array2d<double>("J", "Jpara", {orbitals, next_orbitals}, loc%Lcell);
		param2d Xpara = P.fill_array2d<double>("X", "Xpara", {orbitals, next_orbitals}, loc%Lcell);
		param2d Vxypara = P.fill_array2d<double>("Vxy", "Vxypara", {orbitals, next_orbitals}, loc%Lcell);
		param2d Vzpara = P.fill_array2d<double>("Vz", "Vzpara", {orbitals, next_orbitals}, loc%Lcell);
		
		Terms.save_label(loc, tpara.label);
		Terms.save_label(loc, Vpara.label);
		Terms.save_label(loc, Jpara.label);
		Terms.save_label(loc, Xpara.label);
		Terms.save_label(loc, Vxypara.label);
		Terms.save_label(loc, Vzpara.label);
		
		if (loc < N_sites-1 or !P.get<bool>("OPEN_BC"))
		{
			for (std::size_t alfa=0; alfa<orbitals;      ++alfa)
			for (std::size_t beta=0; beta<next_orbitals; ++beta)
			{
				// t
				Terms.push_tight(loc, -tpara(alfa,beta), F[loc].cdag(UP,alfa)*F[loc].sign(), F[lp1].c(UP,beta));
				Terms.push_tight(loc, -tpara(alfa,beta), F[loc].cdag(DN,alfa)*F[loc].sign(), F[lp1].c(DN,beta));
				Terms.push_tight(loc, +tpara(alfa,beta), F[loc].c(UP,alfa)   *F[loc].sign(), F[lp1].cdag(UP,beta));
				Terms.push_tight(loc, +tpara(alfa,beta), F[loc].c(DN,alfa)   *F[loc].sign(), F[lp1].cdag(DN,beta));
				
				// V
				Terms.push_tight(loc, Vpara.a(alfa,beta), F[loc].n(alfa), F[lp1].n(beta));
				
				// J
				Terms.push_tight(loc, 0.5*Jpara(alfa,beta), F[loc].Sp(alfa), F[lp1].Sm(beta));
				Terms.push_tight(loc, 0.5*Jpara(alfa,beta), F[loc].Sm(alfa), F[lp1].Sp(beta));
				Terms.push_tight(loc,     Jpara(alfa,beta), F[loc].Sz(alfa), F[lp1].Sz(beta));
				
				// X, uncompressed variant with 12 operators
//				Terms.push_tight(loc, -Xpara(alfa,beta), F[loc].cdag(UP,alfa)*F[loc].sign() * F[loc].n(DN,alfa), F[lp1].c(UP,beta));
//				Terms.push_tight(loc, -Xpara(alfa,beta), F[loc].cdag(UP,alfa)*F[loc].sign(), F[lp1].c(UP,beta) * F[lp1].n(DN,beta));
//				Terms.push_tight(loc, +2.*Xpara(alfa,beta), F[loc].cdag(UP,alfa)*F[loc].sign() * F[loc].n(DN,alfa), F[lp1].c(UP,beta) * F[lp1].n(DN,beta));
//				
//				Terms.push_tight(loc, -Xpara(alfa,beta), F[loc].cdag(DN,alfa)*F[loc].sign() * F[loc].n(UP,alfa), F[lp1].c(DN,beta));
//				Terms.push_tight(loc, -Xpara(alfa,beta), F[loc].cdag(DN,alfa)*F[loc].sign(), F[lp1].c(DN,beta) * F[lp1].n(UP,beta));
//				Terms.push_tight(loc, +2.*Xpara(alfa,beta), F[loc].cdag(DN,alfa)*F[loc].sign() * F[loc].n(UP,alfa), F[lp1].c(DN,beta) * F[lp1].n(UP,beta));
//				
//				Terms.push_tight(loc, +Xpara(alfa,beta), F[loc].c(UP,alfa)*F[loc].sign() * F[loc].n(DN,alfa), F[lp1].cdag(UP,beta));
//				Terms.push_tight(loc, +Xpara(alfa,beta), F[loc].c(UP,alfa)*F[loc].sign(), F[lp1].cdag(UP,beta) * F[lp1].n(DN,beta));
//				Terms.push_tight(loc, -2.*Xpara(alfa,beta), F[loc].c(UP,alfa)*F[loc].sign() * F[loc].n(DN,alfa), F[lp1].cdag(UP,beta) * F[lp1].n(DN,beta));
//				
//				Terms.push_tight(loc, +Xpara(alfa,beta), F[loc].c(DN,alfa)*F[loc].sign() * F[loc].n(UP,alfa), F[lp1].cdag(DN,beta));
//				Terms.push_tight(loc, +Xpara(alfa,beta), F[loc].c(DN,alfa)*F[loc].sign(), F[lp1].cdag(DN,beta) * F[lp1].n(UP,beta));
//				Terms.push_tight(loc, -2.*Xpara(alfa,beta), F[loc].c(DN,alfa)*F[loc].sign() * F[loc].n(UP,alfa), F[lp1].cdag(DN,beta) * F[lp1].n(UP,beta));
				
				// X, compressed variant with 8 operators
				Terms.push_tight(loc, -Xpara(alfa,beta), F[loc].cdag(UP,alfa)*F[loc].sign() * F[loc].n(DN,alfa), 
				                                         F[lp1].c(UP,beta) * (F[lp1].Id()-2.*F[lp1].n(DN,alfa)));
				Terms.push_tight(loc, -Xpara(alfa,beta), F[loc].cdag(UP,alfa)*F[loc].sign(), 
				                                         F[lp1].c(UP,beta) * F[lp1].n(DN,beta));
				
				Terms.push_tight(loc, -Xpara(alfa,beta), F[loc].cdag(DN,alfa)*F[loc].sign() * F[loc].n(UP,alfa), 
				                                         F[lp1].c(DN,beta) * (F[lp1].Id()-2.*F[lp1].n(UP,alfa)));
				Terms.push_tight(loc, -Xpara(alfa,beta), F[loc].cdag(DN,alfa)*F[loc].sign(), 
				                                         F[lp1].c(DN,beta) * F[lp1].n(UP,beta));
				
				Terms.push_tight(loc, +Xpara(alfa,beta), F[loc].c(UP,alfa)*F[loc].sign() * F[loc].n(DN,alfa), 
				                                         F[lp1].cdag(UP,beta) * (F[lp1].Id()-2.*F[lp1].n(DN,alfa)));
				Terms.push_tight(loc, +Xpara(alfa,beta), F[loc].c(UP,alfa)*F[loc].sign(), 
				                                         F[lp1].cdag(UP,beta) * F[lp1].n(DN,beta));
				
				Terms.push_tight(loc, +Xpara(alfa,beta), F[loc].c(DN,alfa)*F[loc].sign() * F[loc].n(UP,alfa), 
				                                         F[lp1].cdag(DN,beta) * (F[lp1].Id()-2.*F[lp1].n(UP,alfa)));
				Terms.push_tight(loc, +Xpara(alfa,beta), F[loc].c(DN,alfa)*F[loc].sign(), 
				                                         F[lp1].cdag(DN,beta) * F[lp1].n(UP,beta));
				
				// Vxy, Vz
				Terms.push_tight(loc, 0.5*Vxypara(alfa,beta)*pow(-1,loc+lp1), F[loc].cc(alfa),       F[lp1].cdagcdag(beta));
				Terms.push_tight(loc, 0.5*Vxypara(alfa,beta)*pow(-1,loc+lp1), F[loc].cdagcdag(alfa), F[lp1].cc(beta));
				Terms.push_tight(loc,     Vzpara (alfa,beta),                 F[loc].Tz(alfa),       F[lp1].Tz(beta));
			}
		}
		
		// Next-nearest-neighbour terms: t'
		
		param2d tPrime = P.fill_array2d<double>("tPrime", "tPrime_array", {orbitals, nextn_orbitals}, loc%Lcell);
		Terms.save_label(loc, tPrime.label);
		
		if (loc < N_sites-2 or !P.get<bool>("OPEN_BC"))
		{
			for (std::size_t alfa=0; alfa<orbitals;       ++alfa)
			for (std::size_t beta=0; beta<nextn_orbitals; ++beta)
			{
				Terms.push_nextn(loc, -tPrime(alfa,beta),     F[loc].cdag(UP,alfa)*F[loc].sign(), F[lp1].sign(), F[lp2].c(UP,beta));
				Terms.push_nextn(loc, -tPrime(alfa,beta),     F[loc].cdag(DN,alfa)*F[loc].sign(), F[lp1].sign(), F[lp2].c(DN,beta));
				Terms.push_nextn(loc, -tPrime(alfa,beta), -1.*F[loc].c(UP,alfa)*F[loc].sign(),    F[lp1].sign(), F[lp2].cdag(UP,beta));
				Terms.push_nextn(loc, -tPrime(alfa,beta), -1.*F[loc].c(DN,alfa)*F[loc].sign(),    F[lp1].sign(), F[lp2].cdag(DN,beta));
			}
		}
		
		param0d J3site = P.fill_array0d<double>("J3site", "J3site", loc%Lcell);
		Terms.save_label(loc, J3site.label);
		
		if (J3site.x != 0.)
		{
			lout << "Warning! J3site has to be tested against ED!" << endl;
			
			assert(orbitals == 1 and "Cannot do a ladder with 3-site J terms!");
			if (loc < N_sites-2 or !P.get<bool>("OPEN_BC"))
			{
				SiteOperator<Symmetry_, double> cup_sign_local    = F[loc].c(UP)    * F[loc].sign();
				SiteOperator<Symmetry_, double> cdn_sign_local    = F[loc].c(DN)    * F[loc].sign();
				SiteOperator<Symmetry_, double> cupdag_sign_local = F[loc].cdag(UP) * F[loc].sign();
				SiteOperator<Symmetry_, double> cdndag_sign_local = F[loc].cdag(DN) * F[loc].sign();
				
				SiteOperator<Symmetry_, double> nup_sign_tight = F[lp1].n(UP) * F[lp1].sign();
				SiteOperator<Symmetry_, double> ndn_sign_tight = F[lp1].n(UP) * F[lp1].sign();
				SiteOperator<Symmetry_, double> Sp_sign_tight  = F[lp1].Sp()  * F[lp1].sign();
				SiteOperator<Symmetry_, double> Sm_sign_tight  = F[lp1].Sm()  * F[lp1].sign();
				
				SiteOperator<Symmetry_, double> cup_nextn    = F[lp2].c(UP);
				SiteOperator<Symmetry_, double> cdn_nextn    = F[lp2].c(DN);
				SiteOperator<Symmetry_, double> cupdag_nextn = F[lp2].cdag(UP);
				SiteOperator<Symmetry_, double> cdndag_nextn = F[lp2].cdag(DN);
				
				// three-site terms without spinflip
				Terms.push_nextn(loc, -0.25*J3site.x, cupdag_sign_local, ndn_sign_tight, cup_nextn);
				Terms.push_nextn(loc, -0.25*J3site.x, cdndag_sign_local, nup_sign_tight, cdn_nextn);
				Terms.push_nextn(loc, +0.25*J3site.x, cup_sign_local,    ndn_sign_tight, cupdag_nextn);
				Terms.push_nextn(loc, +0.25*J3site.x, cdn_sign_local,    nup_sign_tight, cdndag_nextn);
				
				// three-site terms with spinflip
				Terms.push_nextn(loc, -0.25*J3site.x, cupdag_sign_local, Sm_sign_tight, cdn_nextn);
				Terms.push_nextn(loc, -0.25*J3site.x, cdndag_sign_local, Sp_sign_tight, cup_nextn);
				Terms.push_nextn(loc, +0.25*J3site.x, cup_sign_local,    Sp_sign_tight, cdndag_nextn);
				Terms.push_nextn(loc, +0.25*J3site.x, cdn_sign_local,    Sm_sign_tight, cupdag_nextn);
			}
		}
	}
	
	if (!U_infinite)
	{
		Terms.set_name("Hubbard");
	}
	else if (P.HAS_ANY_OF({"J", "J3site"}))
	{
		Terms.set_name("t-J");
	}
	else
	{
		Terms.set_name("U=∞-Hubbard");
	}
}

} // end namespace VMPS::models

#endif
