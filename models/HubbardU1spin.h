#ifndef STRAWBERRY_HUBBARDMODEL_U1SPINONLY
#define STRAWBERRY_HUBBARDMODEL_U1SPINONLY

//include "bases/FermionBase.h"
//include "symmetry/S1xS2.h"
//include "Mpo.h"
//include "ParamHandler.h" // from HELPERS
//include "models/HubbardObservables.h"
#include "models/HubbardU1xU1.h"

namespace VMPS
{

class HubbardU1spin : public Mpo<Sym::U1<Sym::SpinU1>,double>, public HubbardObservables<Sym::U1<Sym::SpinU1> >, public ParamReturner
{
public:
	
	typedef Sym::U1<Sym::SpinU1> Symmetry;
	MAKE_TYPEDEFS(HubbardU1spin)
	
	///@{
	HubbardU1spin() : Mpo(){};
	
	HubbardU1spin(Mpo<Symmetry> &Mpo_input, const vector<Param> &params)
	:Mpo<Symmetry>(Mpo_input),
	 HubbardObservables(this->N_sites,params,HubbardU1spin::defaults),
	 ParamReturner()
	{
		ParamHandler P(params,HubbardU1spin::defaults);
		size_t Lcell = P.size();
		N_phys = 0;
		for (size_t l=0; l<N_sites; ++l) N_phys += P.get<size_t>("Ly",l%Lcell);
		this->precalc_TwoSiteData();
	};
	
	HubbardU1spin (const size_t &L, const vector<Param> &params, const BC &boundary=BC::OPEN, const DMRG::VERBOSITY::OPTION &VERB=DMRG::VERBOSITY::OPTION::ON_EXIT);
	///@}
	
	template<typename Symmetry_>
	static void add_operators (const std::vector<FermionBase<Symmetry_> > &F, const ParamHandler &P, 
	                           PushType<SiteOperator<Symmetry_,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, 
	                           const BC boundary=BC::OPEN);
	
	static qarray<1> singlet (int N=0) {return qarray<1>{0};};
	static constexpr MODEL_FAMILY FAMILY = HUBBARD;
	static constexpr int spinfac = 2;
	
	/**Default parameters.*/
	static const std::map<string,std::any> defaults;
};

const std::map<string,std::any> HubbardU1spin::defaults = 
{
	{"t",1.}, {"tPrime",0.}, {"tRung",1.},
	{"mu",0.}, {"t0",0.}, 
	{"U",0.}, {"Uph",0.},
	{"V",0.}, {"Vrung",0.},
	{"Vxy",0.}, {"Vz",0.},
	{"Bz",0.}, 
	{"J",0.}, {"Jperp",0.}, {"J3site",0.},
	{"X",0.}, {"Xperp",0.},
	{"C",0.}, {"Cperp",0.},
	{"REMOVE_DOUBLE",false}, {"REMOVE_EMPTY",false}, {"REMOVE_SINGLE",false}, {"mfactor",1}, 
	{"maxPower",2ul}, {"CYLINDER",false}, {"Ly",1ul}
};

HubbardU1spin::
HubbardU1spin (const size_t &L, const vector<Param> &params, const BC &boundary, const DMRG::VERBOSITY::OPTION &VERB)
:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, boundary, VERB),
 HubbardObservables(L,params,HubbardU1spin::defaults),
 ParamReturner()
{
	ParamHandler P(params,HubbardU1spin::defaults);
	
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
void HubbardU1spin::
add_operators (const std::vector<FermionBase<Symmetry_> > &F, const ParamHandler &P, PushType<SiteOperator<Symmetry_,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = F.size();
	
	for(std::size_t loc=0; loc<N_sites; ++loc)
	{
		std::size_t orbitals = F[loc].orbitals();
		
		ArrayXd  U_array  = F[loc].ZeroField();
		ArrayXd  Uph_array  = F[loc].ZeroField();
		ArrayXd  E_array  = F[loc].ZeroField();
		ArrayXd  Bz_array = F[loc].ZeroField();
		ArrayXXd tperp_array = F[loc].ZeroHopping();
		ArrayXXd Vperp_array = F[loc].ZeroHopping();
		ArrayXXd Jperp_array = F[loc].ZeroHopping();
		
		param1d C = P.fill_array1d<double>("C", "Corb", orbitals, loc%Lcell);
		labellist[loc].push_back(C.label);
		
		// local superconducting term C_{ii}
		auto Hloc = Mpo<Symmetry,double>::get_N_site_interaction
		(
			F[loc].template HubbardHamiltonian<double>(U_array, Uph_array, E_array, Bz_array, tperp_array, Vperp_array, Vperp_array, Vperp_array, Jperp_array, Jperp_array, C.a)
		);
		pushlist.push_back(std::make_tuple(loc, Hloc, 1.));
		
		auto push_full = [&N_sites, &loc, &F, &P, &pushlist, &labellist, &boundary] (string xxxFull, string label,
		                                                                             const vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > &first,
		                                                                             const vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > &last,
		                                                                             vector<double> factor, bool FERMIONIC) -> void
		{
			ArrayXXd Full = P.get<Eigen::ArrayXXd>(xxxFull);
			vector<vector<std::pair<size_t,double> > > R = Geometry2D::rangeFormat(Full);
			
			if (static_cast<bool>(boundary)) {assert(R.size() ==   N_sites and "Use an (N_sites)x(N_sites) hopping matrix for open BC!");}
			else                             {assert(R.size() >= 2*N_sites and "Use at least a (2*N_sites)x(N_sites) hopping matrix for infinite BC!");}
			
			for (size_t j=0; j<first.size(); j++)
			for (size_t h=0; h<R[loc].size(); ++h)
			{
				size_t range = R[loc][h].first;
				double value = R[loc][h].second;
				
				if (range != 0)
				{
					vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > ops(range+1);
					ops[0] = first[j];
					for (size_t i=1; i<range; ++i)
					{
						if (FERMIONIC) {ops[i] = F[(loc+i)%N_sites].sign();}
						else {ops[i] = F[(loc+i)%N_sites].Id();}
					}
					ops[range] = last[j][(loc+range)%N_sites];
					pushlist.push_back(std::make_tuple(loc, ops, factor[j] * value));
				}
			}
			
			stringstream ss;
			ss << label << "(" << Geometry2D::hoppingInfo(Full) << ")";
			labellist[loc].push_back(ss.str());
		};
		
		// non-local superconducting term C_{ij}
		if (P.HAS("Cfull"))
		{
			SiteOperatorQ<Symmetry_,Eigen::MatrixXd> cUP_sign_local    = F[loc].c(UP,0)    * F[loc].sign();
			SiteOperatorQ<Symmetry_,Eigen::MatrixXd> cDN_sign_local    = F[loc].c(DN,0)    * F[loc].sign();
			SiteOperatorQ<Symmetry_,Eigen::MatrixXd> cdagUP_sign_local = F[loc].cdag(UP,0) * F[loc].sign();
			SiteOperatorQ<Symmetry_,Eigen::MatrixXd> cdagDN_sign_local = F[loc].cdag(DN,0) * F[loc].sign();
			
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > cUP_ranges(N_sites);    for (size_t i=0; i<N_sites; ++i) {cUP_ranges[i]    = F[i].c(UP,0);}
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > cDN_ranges(N_sites);    for (size_t i=0; i<N_sites; ++i) {cDN_ranges[i]    = F[i].c(DN,0);}
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > cdagUP_ranges(N_sites); for (size_t i=0; i<N_sites; ++i) {cdagUP_ranges[i] = F[i].cdag(UP,0);}
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > cdagDN_ranges(N_sites); for (size_t i=0; i<N_sites; ++i) {cdagDN_ranges[i] = F[i].cdag(DN,0);}
			
			// sort by lattice site:
			// c↑i*c↓j + c†↓j*c†↑i
			// = c↑i*c↓j - c†↑i*c†↓j
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> >          frst {cUP_sign_local, cdagUP_sign_local};
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {cDN_ranges, cdagDN_ranges};
			push_full("Cfull", "Cᵢⱼ", frst, last, {+1., -1.}, PROP::FERMIONIC);
		}
	}
}

} // end namespace VMPS::models

#endif
