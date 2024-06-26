#ifndef VANILLA_GRANDHUBBARDMODEL
#define VANILLA_GRANDHUBBARDMODEL

#include "symmetry/U0.h"
#include "HubbardU1xU1.h"
#include "BetheAnsatzIntegrals.h" // from TOOLS, depends on gsl

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
class Hubbard : public Mpo<Sym::U0,double>, public HubbardObservables<Sym::U0>, public ParamReturner
{
public:
	typedef Sym::U0 Symmetry;
	MAKE_TYPEDEFS(Hubbard)
	
	Hubbard() : Mpo() {};
	
	Hubbard(Mpo<Symmetry> &Mpo_input, const vector<Param> &params)
	:Mpo<Symmetry>(Mpo_input),
	 HubbardObservables(this->N_sites,params,Hubbard::defaults),
	 ParamReturner()
	{
		ParamHandler P(params,Hubbard::defaults);
		size_t Lcell = P.size();
		N_phys = 0;
		for (size_t l=0; l<N_sites; ++l) N_phys += P.get<size_t>("Ly",l%Lcell);
		this->precalc_TwoSiteData();
		this->HERMITIAN = true;
		this->HAMILTONIAN = true;
	};
	
	Hubbard (const size_t &L, const vector<Param> &params, const BC &boundary=BC::OPEN, const DMRG::VERBOSITY::OPTION &VERB=DMRG::VERBOSITY::OPTION::ON_EXIT);
	
	static qarray<0> singlet (int N) {return qarray<0>{};};
	static constexpr MODEL_FAMILY FAMILY = HUBBARD;
	static constexpr int spinfac = 2;
	
	template<typename Symmetry_>
	static void add_operators (const std::vector<FermionBase<Symmetry_> > &F, const ParamHandler &P, 
	                           PushType<SiteOperator<Symmetry_,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, 
	                           const BC boundary=BC::OPEN);
	
	static const std::map<string,std::any> defaults;
	
	static refEnergy ref (const vector<Param> &params, double L=numeric_limits<double>::infinity());
};

const std::map<string,std::any> Hubbard::defaults = 
{
	{"t",1.}, {"tPrime",0.}, {"tRung",1.},
	{"mu",0.}, {"t0",0.}, {"Fp", 0.},
	{"U",0.}, {"Uph",0.},
	{"V",0.}, {"Vrung",0.}, 
	{"Vxy",0.}, {"Vz",0.},
	{"Bz",0.}, {"Bx",0.}, 
	{"J",0.}, {"Jrung",0.},
	{"J3site",0.},
	{"Delta",0.}, {"DeltaUP",0.}, {"DeltaDN",0.},
	{"X",0.}, {"Xperp",0.},
	{"REMOVE_DOUBLE",false}, {"REMOVE_EMPTY",false}, {"REMOVE_UP",false}, {"REMOVE_DN",false}, {"mfactor",1}, {"k",0}, 
	{"maxPower",2ul}, {"CYLINDER",false}, {"Ly",1ul}
};

Hubbard::
Hubbard (const size_t &L, const vector<Param> &params, const BC &boundary, const DMRG::VERBOSITY::OPTION &VERB)
:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, boundary, VERB),
 HubbardObservables(L,params,Hubbard::defaults),
 ParamReturner()
{
	ParamHandler P(params,Hubbard::defaults);
	
	size_t Lcell = P.size();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(F[l].get_basis().qloc(),l);
	}
	
	param1d U = P.fill_array1d<double>("U", "Uorb", F[0].orbitals(), 0);
	if (isfinite(U.a.sum()))
	{
		this->set_name("Hubbard");
	}
	else if (P.HAS_ANY_OF({"J", "J3site"}))
	{
		this->set_name("t-J");
	}
	else
	{
		this->set_name("U=∞-Hubbard");
	}
	PushType<SiteOperator<Symmetry,double>,double> pushlist;
	std::vector<std::vector<std::string>> labellist;
	HubbardU1xU1::set_operators(F, P, pushlist, labellist, boundary);
	add_operators(F, P, pushlist, labellist, boundary);
	
	this->construct_from_pushlist(pushlist, labellist, Lcell);
	this->finalize(PROP::COMPRESS, P.get<size_t>("maxPower"));
	
	this->precalc_TwoSiteData();
}

template<typename Symmetry_>
void Hubbard::
add_operators (const std::vector<FermionBase<Symmetry_> > &F, const ParamHandler &P, PushType<SiteOperator<Symmetry_,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = F.size();
	
	for(std::size_t loc=0; loc<N_sites; ++loc)
	{
		std::size_t lp1 = (loc+1)%N_sites;
		std::size_t lp2 = (loc+2)%N_sites;
		
		std::size_t orbitals = F[loc].orbitals();
		std::size_t next_orbitals = F[lp1].orbitals();
		std::size_t nextn_orbitals = F[lp2].orbitals();
		
		param2d DeltaUPpara = P.fill_array2d<double>("DeltaUP", "DeltaUPpara", {orbitals, next_orbitals}, loc%Lcell);
		param2d DeltaDNpara = P.fill_array2d<double>("DeltaDN", "DeltaDNpara", {orbitals, next_orbitals}, loc%Lcell);
		
		labellist[loc].push_back(DeltaUPpara.label);
		labellist[loc].push_back(DeltaDNpara.label);
		
		if (loc < N_sites-1 or !static_cast<bool>(boundary))
		{
			for (std::size_t alfa=0; alfa<orbitals;      ++alfa)
			for (std::size_t beta=0; beta<next_orbitals; ++beta)
			{
				if (!P.HAS("DeltaUPfull"))
				{
					if (DeltaUPpara(alfa,beta) != 0.)
					{
						pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction((F[loc].cdag(UP,alfa)*F[loc].sign()), F[lp1].cdag(UP,beta)), +DeltaUPpara(alfa,beta)));
						pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction((F[loc].c(UP,alfa)   *F[loc].sign()), F[lp1].c(UP,beta)),    -DeltaUPpara(alfa,beta)));
					}
				}
				if (!P.HAS("DeltaDNfull"))
				{
					if (DeltaDNpara(alfa,beta) != 0.)
					{
						pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction((F[loc].cdag(DN,alfa)*F[loc].sign()), F[lp1].c(DN,beta)),    +DeltaDNpara(alfa,beta)));
						pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction((F[loc].c(DN,alfa)   *F[loc].sign()), F[lp1].cdag(DN,beta)), -DeltaDNpara(alfa,beta)));
					}
				}
			}
		}
		
		// ArrayXd  U_array  = F[loc].ZeroField();
		// ArrayXd  Uph_array  = F[loc].ZeroField();
		// ArrayXd  E_array  = F[loc].ZeroField();
		// ArrayXd  Bz_array = F[loc].ZeroField();
		// ArrayXXd tperp_array = F[loc].ZeroHopping();
		// ArrayXXd Vperp_array = F[loc].ZeroHopping();
		// ArrayXXd Jperp_array = F[loc].ZeroHopping();
		
		param1d Bx = P.fill_array1d<double>("Bx", "Bxorb", orbitals, loc%Lcell);
		labellist[loc].push_back(Bx.label);
		
		param1d Fp = P.fill_array1d<double>("Fp", "Fporb", orbitals, loc%Lcell);
		labellist[loc].push_back(Fp.label);
		
		auto H_Bx = F[loc].template coupling_Bx<double>(Bx.a);
		auto H_Fp = F[loc].template coupling_singleFermion<double>(Fp.a);
		auto Hloc = Mpo<Symmetry,double>::get_N_site_interaction((H_Bx+H_Fp));
		pushlist.push_back(std::make_tuple(loc, Hloc, 1.));
	}
}

refEnergy Hubbard::
ref (const vector<Param> &params, double L)
{
	ParamHandler P(params,{{"t",1.},{"U",0.},{"n",1.},{"Ly",1ul},{"tRung",1.},{"tPrime",0.},
	                       {"t0",0.},{"V",0.},{"Bz",0.},{"Bx",0.},{"J",0.},{"J3site",0.}});
	refEnergy out;
	
	size_t Ly = P.get<size_t>("Ly");
	double n = P.get<double>("n");
	double U = P.get<double>("U");
	double t = P.get<double>("t");
	double tRung = P.get<double>("tRung");
	
	// half-filled chain
	if (isinf(L) and Ly == 1ul and n == 1. and P.ARE_ALL_ZERO<double>({"tPrime","t0","V","Bz","Bx","J","J3site"}))
	{
		out.value = BetheAnsatz::e0(U,t);
		out.source = "Elliott H. Lieb, F. Y. Wu, Absence of Mott Transition in an Exact Solution of the Short-Range, One-Band Model in One Dimension, Phys. Rev. Lett. 20, 1445 (1968)";
		out.method = "num. integration with gsl";
	}
	// U=0 ladder
	else if (Ly == 2ul and n == 1. and P.ARE_ALL_ZERO<double>({"U","tPrime","t0","V","Bz","Bx","J","J3site"}))
	{
		if (t/tRung <= 0.5) {out.value = -tRung;}
		else
		{
			if (isinf(L)) {out.value = -tRung-2.*M_1_PI*tRung*(sqrt(pow(2.*t/tRung,2)-1.)-acos(0.5*tRung/t));}
		}
		out.source = "Zheng Weihong, J. Oitmaa, C. J. Hamer, R. J. Bursill, Numerical studies of the two-leg Hubbard ladder, J. Phys.: Condens. Matter 13 (2001) 433–448";
		out.method = "analytical";
	}
	
	return out;
}

}

#endif
