#ifndef STRAWBERRY_SPINLESSFERMIONSU1
#define STRAWBERRY_SPINLESSFERMIONSU1

#include "symmetry/U1.h"
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
	
	///@{
	SpinlessFermionsU1() : Mpo<Symmetry>(), ParamReturner() {};
	
	SpinlessFermionsU1(Mpo<Symmetry> &Mpo_input, const vector<Param> &params)
	:Mpo<Symmetry>(Mpo_input),
	 SpinlessFermionsObservables(this->N_sites,params,SpinlessFermionsU1::defaults),
	 ParamReturner()
	{
		ParamHandler P(params,SpinlessFermionsU1::defaults);
		size_t Lcell = P.size();
		N_phys = 0;
		for (size_t l=0; l<N_sites; ++l) N_phys += P.get<size_t>("Ly",l%Lcell);
		this->precalc_TwoSiteData();
	};
	
	SpinlessFermionsU1 (const size_t &L, const vector<Param> &params, const BC &boundary=BC::OPEN, const DMRG::VERBOSITY::OPTION &VERB=DMRG::VERBOSITY::OPTION::ON_EXIT);
	///@}
	
	template<typename Symmetry_>
	static void set_operators (const std::vector<SpinlessFermionBase<Symmetry_> > &F, const ParamHandler &P,
	                           PushType<SiteOperator<Symmetry_,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary=BC::OPEN);
	
	static qarray<1> singlet (int N) {return qarray<1>{N};};
	static constexpr MODEL_FAMILY FAMILY = SPINLESS;
	static constexpr int spinfac = 1;
	
	static const std::map<string,std::any> defaults;
//	static const std::map<string,std::any> sweep_defaults;
};

const std::map<string,std::any> SpinlessFermionsU1::defaults = 
{
	{"t",1.}, {"tPrime",0.},
	{"V",0.}, {"Vph",0.},
	{"Vprime",0.}, {"VphPrime",0.},
	{"mu",0.},{"t0",0.},
	{"D",2ul}, {"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}, 
};

//const std::map<string,std::any> SpinlessFermionsU1::sweep_defaults = 
//{
//	{"max_alpha",100.}, {"min_alpha",1.e-11}, {"lim_alpha",10ul}, {"eps_svd",1.e-7},
//	{"Dincr_abs", 4ul}, {"Dincr_per", 2ul}, {"Dincr_rel", 1.1},
//	{"min_Nsv",0ul}, {"max_Nrich",-1},
//	{"max_halfsweeps",20ul}, {"min_halfsweeps",1ul},
//	{"Dinit",4ul}, {"Qinit",5ul}, {"Dlimit",100ul},
//	{"tol_eigval",1e-7}, {"tol_state",1e-6},
//	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
//};

SpinlessFermionsU1::
SpinlessFermionsU1 (const size_t &L, const vector<Param> &params, const BC &boundary, const DMRG::VERBOSITY::OPTION &VERB)
:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, boundary, VERB),
 SpinlessFermionsObservables<Sym::U1<Sym::ChargeU1> >(L,params,SpinlessFermionsU1::defaults),
 ParamReturner() //SpinlessFermionsU1::sweep_defaults
{
	ParamHandler P(params,SpinlessFermionsU1::defaults);
	size_t Lcell = P.size();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(F[l].get_basis().qloc(),l);
	}
	
	//HamiltonianTermsXd<Symmetry> Terms(N_sites, P.get<bool>("OPEN_BC"));
	//F.resize(N_sites);
	
	this->set_name("SpinlessFermions");
	
	PushType<SiteOperator<Symmetry,double>,double> pushlist;
	std::vector<std::vector<std::string>> labellist;
	set_operators(F, P, pushlist, labellist, boundary);
	
	//set_operators(F,P,Terms);
	//this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
	//this->precalc_TwoSiteData();
	
	this->construct_from_pushlist(pushlist, labellist, Lcell);
	this->finalize(PROP::COMPRESS, P.get<size_t>("maxPower"));
	this->precalc_TwoSiteData();
}

template<typename Symmetry_>
void SpinlessFermionsU1::
set_operators (const std::vector<SpinlessFermionBase<Symmetry_> > &F, const ParamHandler &P, 
               PushType<SiteOperator<Symmetry_,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = F.size();
	if(labellist.size() != N_sites) {labellist.resize(N_sites);}
	
	for (std::size_t loc=0; loc<N_sites; ++loc)
	{
		size_t lp1 = (loc+1)%N_sites;
		size_t lp2 = (loc+2)%N_sites;
		
		std::size_t orbitals       = F[loc].orbitals();
		std::size_t next_orbitals  = F[lp1].orbitals();
		std::size_t nextn_orbitals = F[lp2].orbitals();
		
		stringstream ss;
		ss << "Ly=" << P.get<size_t>("Ly",loc%Lcell);
		labellist[loc].push_back(ss.str());
		
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
		
		if (P.HAS("tFull"))
		{
//			ArrayXXd Full = P.get<Eigen::ArrayXXd>("tFull");
//			vector<vector<std::pair<size_t,double> > > R = Geometry2D::rangeFormat(Full);
//			
//			if (P.get<bool>("OPEN_BC")) {assert(R.size() ==   N_sites and "Use an (N_sites)x(N_sites) hopping matrix for open BC!");}
//			else                        {assert(R.size() >= 2*N_sites and "Use at least a (2*N_sites)x(N_sites) hopping matrix for infinite BC!");}
//			
//			for (size_t h=0; h<R[loc].size(); ++h)
//			{
//				size_t range = R[loc][h].first;
//				double value = R[loc][h].second;
//				
//				size_t Ntrans = (range == 0)? 0:range-1;
//				vector<SiteOperator<Symmetry_,double> > TransOps(Ntrans);
//				for (size_t i=0; i<Ntrans; ++i)
//				{
//					TransOps[i] = F[(loc+i+1)%N_sites].sign();
//				}
//				
//				if (range != 0)
//				{
//					auto c_sign_local    = F[loc].c(0) * F[loc].sign();
//					auto cdag_sign_local = F[loc].cdag(0) * F[loc].sign();
//					auto c_range         = F[(loc+range)%N_sites].c(0);
//					auto cdag_range      = F[(loc+range)%N_sites].cdag(0);
//					
//					//hopping
//					Terms.push(range, loc, -value, cdag_sign_local, TransOps, c_range);
//					Terms.push(range, loc, +value, c_sign_local, TransOps, cdag_range);
//				}
//			}
//			
//			stringstream ss;
//			ss << "tᵢⱼ(" << Geometry2D::hoppingInfo(Full) << ")";
//			Terms.save_label(loc,ss.str());
			
			SiteOperatorQ<Symmetry_,Eigen::MatrixXd> c_sign_local    = F[loc].c(0)    * F[loc].sign();
			SiteOperatorQ<Symmetry_,Eigen::MatrixXd> cdag_sign_local = F[loc].cdag(0) * F[loc].sign();
			
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > c_ranges(N_sites);    for (size_t i=0; i<N_sites; ++i) {c_ranges[i]    = F[i].c(0);}
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > cdag_ranges(N_sites); for (size_t i=0; i<N_sites; ++i) {cdag_ranges[i] = F[i].cdag(0);}
			
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> >          frst {cdag_sign_local, c_sign_local};
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {c_ranges, cdag_ranges};
			
			push_full("tFull", "tᵢⱼ", frst, last, {-1., +1.}, PROP::FERMIONIC);
		}
		
		if (P.HAS("VextFull"))
		{
//			ArrayXXd Full = P.get<Eigen::ArrayXXd>("Vfull");
//			vector<vector<std::pair<size_t,double> > > R = Geometry2D::rangeFormat(Full);
//			
//			if (P.get<bool>("OPEN_BC")) {assert(R.size() ==   N_sites and "Use an (N_sites)x(N_sites) hopping matrix for open BC!");}
//			else                        {assert(R.size() >= 2*N_sites and "Use at least a (2*N_sites)x(N_sites) hopping matrix for infinite BC!");}
//			
//			for (size_t h=0; h<R[loc].size(); ++h)
//			{
//				size_t range = R[loc][h].first;
//				double value = R[loc][h].second;
//				
//				size_t Ntrans = (range == 0)? 0:range-1;
//				vector<SiteOperator<Symmetry_,double> > TransOps(Ntrans);
//				for (size_t i=0; i<Ntrans; ++i)
//				{
//					TransOps[i] = F[(loc+i+1)%N_sites].Id();
//				}
//				
//				if (range != 0)
//				{
//					auto n_loc = F[loc].n(0);
//					auto n_hop = F[(loc+range)%N_sites].n(0);
//					
//					Terms.push(range, loc, value, n_loc, TransOps, n_hop);
//				}
//			}
//			
//			stringstream ss;
//			ss << "Vᵢⱼ(" << Geometry2D::hoppingInfo(Full) << ")";
//			Terms.save_label(loc,ss.str());
			
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first {F[loc].n(0)};
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > n_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++)
			{
				n_ranges[i] = F[i].n(0);
			}
			
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {n_ranges};
			push_full("VextFull", "Vᵢⱼ", first, last, {1.}, PROP::BOSONIC);
		}
		
		if (P.HAS("VphFull"))
		{
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first {F[loc].nph(0)};
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > nph_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++)
			{
				nph_ranges[i] = F[i].nph(0);
			}
			
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {nph_ranges};
			push_full("VphFull", "Vphᵢⱼ", first, last, {1.}, PROP::BOSONIC);
		}
		
		// Local terms: t0, mu
		
		param1d t0 = P.fill_array1d<double>("t0", "t0orb", orbitals, loc%Lcell);
		param1d mu = P.fill_array1d<double>("mu", "muorb", orbitals, loc%Lcell);
		
		labellist[loc].push_back(t0.label);
		labellist[loc].push_back(mu.label);
		
		// Nearest-neighbour terms: t, V, Vph
		
//		if (!P.HAS("tFull") and !P.HAS("Vfull") and !P.HAS("VphFull"))
//		{
//			param2d tPara    = P.fill_array2d<double>("t", "tPara", {orbitals, next_orbitals}, loc%Lcell);
//			param2d Vpara    = P.fill_array2d<double>("V", "Vpara",  {orbitals, next_orbitals}, loc%Lcell);
//			param2d VphPara  = P.fill_array2d<double>("Vph", "VphPara",  {orbitals, next_orbitals}, loc%Lcell);
//			
//			Terms.save_label(loc, tPara.label);
//			Terms.save_label(loc, Vpara.label);
//			Terms.save_label(loc, VphPara.label);
//			
//			if (loc < N_sites-1 or !P.get<bool>("OPEN_BC"))
//			{
//				for (std::size_t alfa=0; alfa<orbitals; ++alfa)
//				for (std::size_t beta=0; beta<next_orbitals; ++beta)
//				{
//					Terms.push_tight(loc, -tPara(alfa,beta), F[loc].cdag(alfa)*F[loc].sign(), F[lp1].c(beta));
//					Terms.push_tight(loc, +tPara(alfa,beta), F[loc].c(alfa)   *F[loc].sign(), F[lp1].cdag(beta));
//					
//					Terms.push_tight(loc, Vpara(alfa,beta), F[loc].n(alfa), F[lp1].n(beta));
//					Terms.push_tight(loc, VphPara(alfa,beta), F[loc].n(alfa)-0.5*F[loc].Id(), F[lp1].n(beta)-0.5*F[lp1].Id());
//				}
//			}
//			
//			// Next-nearest-neighbour terms: tPrime, Vprime, VphPrime
//			
//			param2d tPrime = P.fill_array2d<double>("tPrime", "tPrime_array", {orbitals, nextn_orbitals}, loc%Lcell);
//			param2d Vprime = P.fill_array2d<double>("Vprime", "Vprime_array",  {orbitals, nextn_orbitals}, loc%Lcell);
//			param2d VphPrime = P.fill_array2d<double>("VphPrime", "VphPrime_array",  {orbitals, nextn_orbitals}, loc%Lcell);
//			
//			Terms.save_label(loc, tPrime.label);
//			Terms.save_label(loc, Vprime.label);
//			Terms.save_label(loc, VphPrime.label);
//			
//			if (loc < N_sites-2 or !P.get<bool>("OPEN_BC"))
//			{
//				for (std::size_t alfa=0; alfa<orbitals;       ++alfa)
//				for (std::size_t beta=0; beta<nextn_orbitals; ++beta)
//				{
//					Terms.push_nextn(loc, -tPrime(alfa,beta),     F[loc].cdag(alfa)*F[loc].sign(), F[lp1].sign(), F[lp2].c(beta));
//					Terms.push_nextn(loc, -tPrime(alfa,beta), -1.*F[loc].c(alfa)*F[loc].sign(), F[lp1].sign(), F[lp2].cdag(beta));
//					
//					Terms.push_nextn(loc, Vprime(alfa,beta), F[loc].n(alfa), F[lp1].Id(), F[lp2].n(beta));
//					Terms.push_nextn(loc, VphPrime(alfa,beta), F[loc].n(alfa)-0.5*F[loc].Id(), F[lp1].Id(), F[lp2].n(beta)-0.5*F[lp2].Id());
//				}
//			}
//		}
	}
}

} //end namespace VMPS

#endif
