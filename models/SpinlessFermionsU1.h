#ifndef STRAWBERRY_SPINLESSFERMIONSU1
#define STRAWBERRY_SPINLESSFERMIONSU1

#include "symmetry/U1.h"
#include "bases/SpinlessFermionBase.h"
#include "models/SpinlessFermionsObservables.h"
#include "ParamReturner.h"
#include "Geometry2D.h" // from TOOLS

namespace VMPS
{
class SpinlessFermionsU1 : public Mpo<Sym::U1<Sym::ChargeU1>,double>, 
                           public SpinlessFermionsObservables<Sym::U1<Sym::ChargeU1> >,
                           public ParamReturner
{
public:
	
	typedef Sym::U1<Sym::ChargeU1> Symmetry;
	MAKE_TYPEDEFS(SpinlessFermionsU1)
	
public:
	
	SpinlessFermionsU1() : Mpo<Symmetry>(), ParamReturner(SpinlessFermionsU1::sweep_defaults) {};
	SpinlessFermionsU1 (const size_t &L, const vector<Param> &params);
	
	template<typename Symmetry_>
	static void set_operators (const std::vector<SpinlessFermionBase<Symmetry_> > &F, const ParamHandler &P, HamiltonianTermsXd<Symmetry_> &Terms);
	
	static const std::map<string,std::any> defaults;
	static const std::map<string,std::any> sweep_defaults;
};

const std::map<string,std::any> SpinlessFermionsU1::defaults = 
{
	{"t",1.}, {"tPrime",0.},
	{"V",0.}, {"Vph",0.},
	{"Vprime",0.}, {"VphPrime",0.},
	{"mu",0.},{"t0",0.},
	{"D",2ul}, {"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}, 
};

const std::map<string,std::any> SpinlessFermionsU1::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.e-11}, {"lim_alpha",10ul}, {"eps_svd",1.e-7},
	{"Dincr_abs", 4ul}, {"Dincr_per", 2ul}, {"Dincr_rel", 1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",20ul}, {"min_halfsweeps",1ul},
	{"Dinit",5ul}, {"Qinit",10ul}, {"Dlimit",100ul},
	{"tol_eigval",1e-7}, {"tol_state",1e-6},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
};

SpinlessFermionsU1::
SpinlessFermionsU1 (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, PROP::HAMILTONIAN),
 SpinlessFermionsObservables<Sym::U1<Sym::ChargeU1> >(L,params,SpinlessFermionsU1::defaults),
 ParamReturner(SpinlessFermionsU1::sweep_defaults)
{
	ParamHandler P(params,SpinlessFermionsU1::defaults);
	
	size_t Lcell = P.size();
	HamiltonianTermsXd<Symmetry> Terms(N_sites, P.get<bool>("OPEN_BC"));
	F.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		
		setLocBasis(F[l].get_basis(),l);
	}
	
	set_operators(F,P,Terms);
	
	this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
	this->precalc_TwoSiteData();
}

template<typename Symmetry_>
void SpinlessFermionsU1::
set_operators (const std::vector<SpinlessFermionBase<Symmetry_> > &F, const ParamHandler &P, HamiltonianTermsXd<Symmetry_> &Terms)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = Terms.size();
	Terms.set_name("SpinlessFermions");
	
	for (std::size_t loc=0; loc<N_sites; ++loc)
	{
		size_t lp1 = (loc+1)%N_sites;
		size_t lp2 = (loc+2)%N_sites;
		
		std::size_t orbitals       = F[loc].orbitals();
		std::size_t next_orbitals  = F[lp1].orbitals();
		std::size_t nextn_orbitals = F[lp2].orbitals();
		
		if (P.HAS("tFull"))
		{
			ArrayXXd Full = P.get<Eigen::ArrayXXd>("tFull");
			vector<vector<std::pair<size_t,double> > > R = Geometry2D::rangeFormat(Full);
			
			if (P.get<bool>("OPEN_BC")) {assert(R.size() ==   N_sites and "Use an (N_sites)x(N_sites) hopping matrix for open BC!");}
			else                        {assert(R.size() >= 2*N_sites and "Use at least a (2*N_sites)x(N_sites) hopping matrix for infinite BC!");}
			
			for (size_t h=0; h<R[loc].size(); ++h)
			{
				size_t range = R[loc][h].first;
				double value = R[loc][h].second;
				
				size_t Ntrans = (range == 0)? 0:range-1;
				vector<SiteOperator<Symmetry_,double> > TransOps(Ntrans);
				for (size_t i=0; i<Ntrans; ++i)
				{
					TransOps[i] = F[(loc+i+1)%N_sites].sign();
				}
				
				if (range != 0)
				{
					auto c_sign_local    = F[loc].c(0) * F[loc].sign();
					auto cdag_sign_local = F[loc].cdag(0) * F[loc].sign();
					auto c_range         = F[(loc+range)%N_sites].c(0);
					auto cdag_range      = F[(loc+range)%N_sites].cdag(0);
					
					//hopping
					Terms.push(range, loc, -value, cdag_sign_local, TransOps, c_range);
					Terms.push(range, loc, +value, c_sign_local, TransOps, cdag_range);
				}
			}
			
			stringstream ss;
			ss << "tᵢⱼ(" << Geometry2D::hoppingInfo(Full) << ")";
			Terms.save_label(loc,ss.str());
		}
		
		if (P.HAS("Vfull"))
		{
			ArrayXXd Full = P.get<Eigen::ArrayXXd>("Vfull");
			vector<vector<std::pair<size_t,double> > > R = Geometry2D::rangeFormat(Full);
			
			if (P.get<bool>("OPEN_BC")) {assert(R.size() ==   N_sites and "Use an (N_sites)x(N_sites) hopping matrix for open BC!");}
			else                        {assert(R.size() >= 2*N_sites and "Use at least a (2*N_sites)x(N_sites) hopping matrix for infinite BC!");}
			
			for (size_t h=0; h<R[loc].size(); ++h)
			{
				size_t range = R[loc][h].first;
				double value = R[loc][h].second;
				
				size_t Ntrans = (range == 0)? 0:range-1;
				vector<SiteOperator<Symmetry_,double> > TransOps(Ntrans);
				for (size_t i=0; i<Ntrans; ++i)
				{
					TransOps[i] = F[(loc+i+1)%N_sites].Id();
				}
				
				if (range != 0)
				{
					auto n_loc = F[loc].n(0);
					auto n_hop = F[(loc+range)%N_sites].n(0);
					
					Terms.push(range, loc, value, n_loc, TransOps, n_hop);
				}
			}
			
			stringstream ss;
			ss << "Vᵢⱼ(" << Geometry2D::hoppingInfo(Full) << ")";
			Terms.save_label(loc,ss.str());
		}
		
		// Local terms: t0, mu
		
		param1d t0 = P.fill_array1d<double>("t0", "t0orb", orbitals, loc%Lcell);
		param1d mu = P.fill_array1d<double>("mu", "muorb", orbitals, loc%Lcell);
		
		for (std::size_t alfa=0; alfa<orbitals; ++alfa)
		{
			Terms.push_local(loc, +t0(alfa), F[loc].n(alfa));
			Terms.push_local(loc, -mu(alfa), F[loc].n(alfa));
		}
		
		// Nearest-neighbour terms: t, V, Vph
		
		if (!P.HAS("tFull") and !P.HAS("Vfull") and !P.HAS("VphFull"))
		{
			param2d tPara    = P.fill_array2d<double>("t", "tPara", {orbitals, next_orbitals}, loc%Lcell);
			param2d Vpara    = P.fill_array2d<double>("V", "Vpara",  {orbitals, next_orbitals}, loc%Lcell);
			param2d VphPara  = P.fill_array2d<double>("Vph", "VphPara",  {orbitals, next_orbitals}, loc%Lcell);
			
			Terms.save_label(loc, tPara.label);
			Terms.save_label(loc, Vpara.label);
			Terms.save_label(loc, VphPara.label);
			
			if (loc < N_sites-1 or !P.get<bool>("OPEN_BC"))
			{
				for (std::size_t alfa=0; alfa<orbitals; ++alfa)
				for (std::size_t beta=0; beta<next_orbitals; ++beta)
				{
					Terms.push_tight(loc, -tPara(alfa,beta), F[loc].cdag(alfa)*F[loc].sign(), F[lp1].c(beta));
					Terms.push_tight(loc, +tPara(alfa,beta), F[loc].c(alfa)   *F[loc].sign(), F[lp1].cdag(beta));
					
					Terms.push_tight(loc, Vpara(alfa,beta), F[loc].n(alfa), F[lp1].n(beta));
					Terms.push_tight(loc, VphPara(alfa,beta), F[loc].n(alfa)-0.5*F[loc].Id(), F[lp1].n(beta)-0.5*F[lp1].Id());
				}
			}
			
			// Next-nearest-neighbour terms: tPrime, Vprime, VphPrime
			
			param2d tPrime = P.fill_array2d<double>("tPrime", "tPrime_array", {orbitals, nextn_orbitals}, loc%Lcell);
			param2d Vprime = P.fill_array2d<double>("Vprime", "Vprime_array",  {orbitals, nextn_orbitals}, loc%Lcell);
			param2d VphPrime = P.fill_array2d<double>("VphPrime", "VphPrime_array",  {orbitals, nextn_orbitals}, loc%Lcell);
			
			Terms.save_label(loc, tPrime.label);
			Terms.save_label(loc, Vprime.label);
			Terms.save_label(loc, VphPrime.label);
			
			if (loc < N_sites-2 or !P.get<bool>("OPEN_BC"))
			{
				for (std::size_t alfa=0; alfa<orbitals;       ++alfa)
				for (std::size_t beta=0; beta<nextn_orbitals; ++beta)
				{
					Terms.push_nextn(loc, -tPrime(alfa,beta),     F[loc].cdag(alfa)*F[loc].sign(), F[lp1].sign(), F[lp2].c(beta));
					Terms.push_nextn(loc, -tPrime(alfa,beta), -1.*F[loc].c(alfa)*F[loc].sign(), F[lp1].sign(), F[lp2].cdag(beta));
					
					Terms.push_nextn(loc, Vprime(alfa,beta), F[loc].n(alfa), F[lp1].Id(), F[lp2].n(beta));
					Terms.push_nextn(loc, VphPrime(alfa,beta), F[loc].n(alfa)-0.5*F[loc].Id(), F[lp1].Id(), F[lp2].n(beta)-0.5*F[lp2].Id());
				}
			}
		}
	}
}

} //end namespace VMPS

#endif
