#ifndef STRAWBERRY_HUBBARDMODEL_Z2
#define STRAWBERRY_HUBBARDMODEL_Z2

#include "symmetry/ZN.h"
#include "models/HubbardU1xU1.h"
#include "models/HubbardObservables.h"
#include "ParamReturner.h"
#include "Geometry2D.h" // from TOOLS

namespace VMPS
{
class HubbardZ2 : public Mpo<Sym::ZN<Sym::ChargeZ2,2>,double>,
                  public HubbardObservables<Sym::ZN<Sym::ChargeZ2,2> >, 
                  public ParamReturner
{
public:
	
	typedef Sym::ZN<Sym::ChargeZ2,2> Symmetry;
	MAKE_TYPEDEFS(HubbardZ2)
	
	///@{
	HubbardZ2() : Mpo(){};
	
	HubbardZ2(Mpo<Symmetry> &Mpo_input, const vector<Param> &params)
	:Mpo<Symmetry>(Mpo_input),
	 HubbardObservables(this->N_sites,params,HubbardZ2::defaults),
	 ParamReturner()
	{
		ParamHandler P(params,HubbardZ2::defaults);
		size_t Lcell = P.size();
		N_phys = 0;
		for (size_t l=0; l<N_sites; ++l) N_phys += P.get<size_t>("Ly",l%Lcell);
		this->precalc_TwoSiteData();
	};
	
	HubbardZ2 (const size_t &L, const vector<Param> &params, const BC &boundary=BC::OPEN, const DMRG::VERBOSITY::OPTION &VERB=DMRG::VERBOSITY::OPTION::ON_EXIT);
	
	template<typename Symmetry_>
	static void add_operators (const std::vector<FermionBase<Symmetry_> > &F, const ParamHandler &P, 
	                           PushType<SiteOperator<Symmetry_,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, 
	                           const BC boundary=BC::OPEN);
	///@}
	
	static qarray<1> singlet (int N=0) {return qarray<1>{0};};
	static constexpr MODEL_FAMILY FAMILY = HUBBARD;
	static constexpr int spinfac = 2;
	
	static const std::map<string,std::any> defaults;
};

const std::map<string,std::any> HubbardZ2::defaults = 
{
	{"t",1.}, {"tPrime",0.}, {"tRung",1.},
	{"mu",0.}, {"t0",0.}, {"DeltaUP",0.}, {"DeltaDN",0.},
	{"U",0.}, {"Uph",0.},
	{"V",0.}, {"Vrung",0.},
	{"Vxy",0.}, {"Vz",0.},
	{"Bz",0.}, 
	{"J",0.}, {"Jperp",0.}, {"J3site",0.},
	{"X",0.}, {"Xperp",0.},
	{"V",0.}, {"Vrung",0.},
	{"REMOVE_DOUBLE",false}, {"REMOVE_EMPTY",false}, {"REMOVE_UP",false}, {"REMOVE_DN",false}, {"mfactor",1}, {"k",0},
	{"maxPower",2ul}, {"CYLINDER",false}, {"Ly",1ul}
};

HubbardZ2::
HubbardZ2 (const size_t &L, const vector<Param> &params, const BC &boundary, const DMRG::VERBOSITY::OPTION &VERB)
:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, boundary, VERB),
 HubbardObservables(L,params,HubbardZ2::defaults),
 ParamReturner()
{
	ParamHandler P(params, HubbardZ2::defaults);
	size_t Lcell = P.size();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(F[l].get_basis().qloc(),l);
	}
	
	this->set_name("HubbardZ2");
	
	PushType<SiteOperator<Symmetry,double>,double> pushlist;
	std::vector<std::vector<std::string>> labellist;
	HubbardU1xU1::set_operators(F, P, pushlist, labellist, boundary);
	add_operators(F, P, pushlist, labellist, boundary);
	
	this->construct_from_pushlist(pushlist, labellist, Lcell);
	this->finalize(PROP::COMPRESS, P.get<size_t>("maxPower"));
	this->precalc_TwoSiteData();
}

template<typename Symmetry_>
void HubbardZ2::
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
	}
}

} // end namespace VMPS::models

#endif
