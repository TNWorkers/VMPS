#ifndef VANILLA_GRANDHUBBARDMODEL_COMPLEX
#define VANILLA_GRANDHUBBARDMODEL_COMPLEX

#include "symmetry/U0.h"
#include "bases/FermionBase.h"
#include "models/HubbardObservables.h"
#include "ParamReturner.h"
#include "Geometry2D.h" // from TOOLS

namespace VMPS
{
class HubbardComplex : public Mpo<Sym::U0,complex<double> >, public HubbardObservables<Sym::U0,complex<double> >, public ParamReturner
{
public:
	typedef Sym::U0 Symmetry;
	MAKE_TYPEDEFS(HubbardComplex)
	
	HubbardComplex() : Mpo() {};
	
	HubbardComplex(Mpo<Symmetry,complex<double> > &Mpo_input, const vector<Param> &params)
	:Mpo<Symmetry,complex<double> >(Mpo_input),
	 HubbardObservables(this->N_sites,params,HubbardComplex::defaults),
	 ParamReturner()
	{
		ParamHandler P(params,HubbardComplex::defaults);
		size_t Lcell = P.size();
		N_phys = 0;
		for (size_t l=0; l<N_sites; ++l) N_phys += P.get<size_t>("Ly",l%Lcell);
		this->precalc_TwoSiteData();
		this->HERMITIAN = true;
		this->HAMILTONIAN = true;
	};
	
	HubbardComplex (const size_t &L, const vector<Param> &params, const BC &boundary=BC::OPEN, const DMRG::VERBOSITY::OPTION &VERB=DMRG::VERBOSITY::OPTION::ON_EXIT);
	
	static qarray<0> singlet (int N) {return qarray<0>{};};
	static constexpr MODEL_FAMILY FAMILY = HUBBARD;
	static constexpr int spinfac = 2;
	
	template<typename Symmetry_>
	static void set_operators (const std::vector<FermionBase<Symmetry_> > &F, const ParamHandler &P, 
	                           PushType<SiteOperator<Symmetry_,complex<double> >,complex<double> >& pushlist, 
	                           std::vector<std::vector<std::string>>& labellist, 
	                           const BC boundary=BC::OPEN);
	
	static const std::map<string,std::any> defaults;
};

const std::map<string,std::any> HubbardComplex::defaults = 
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

HubbardComplex::
HubbardComplex (const size_t &L, const vector<Param> &params, const BC &boundary, const DMRG::VERBOSITY::OPTION &VERB)
:Mpo<Symmetry,complex<double> > (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, boundary, VERB),
 HubbardObservables(L,params,HubbardComplex::defaults),
 ParamReturner()
{
	ParamHandler P(params,HubbardComplex::defaults);
	
	size_t Lcell = P.size();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(F[l].get_basis().qloc(),l);
	}
	
	this->set_name("Hubbard");
	
	PushType<SiteOperator<Symmetry,complex<double> >,complex<double> > pushlist;
	std::vector<std::vector<std::string>> labellist;
	HubbardComplex::set_operators(F, P, pushlist, labellist, boundary);
	
	this->construct_from_pushlist(pushlist, labellist, Lcell);
	this->finalize(PROP::COMPRESS, P.get<size_t>("maxPower"));
	
	this->precalc_TwoSiteData();
}

template<typename Symmetry_>
void HubbardComplex::
set_operators (const std::vector<FermionBase<Symmetry_> > &F, const ParamHandler &P, PushType<SiteOperator<Symmetry_,complex<double> >,complex<double> >& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = F.size();
	if(labellist.size() != N_sites) {labellist.resize(N_sites);}
	
	for (std::size_t loc=0; loc<N_sites; ++loc)
	{
		size_t lp1 = (loc+1)%N_sites;
		size_t lp2 = (loc+2)%N_sites;
		
		std::size_t orbitals = F[loc].orbitals();
		std::size_t next_orbitals = F[lp1].orbitals();
		std::size_t nextn_orbitals = F[lp2].orbitals();
		
		stringstream ss;
		ss << "Ly=" << P.get<size_t>("Ly",loc%Lcell);
		labellist[loc].push_back(ss.str());
		
		auto push_full = [&N_sites, &loc, &F, &P, &pushlist, &labellist, &boundary] (string xxxFull, string label,
		                                                                             const vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> > &first,
		                                                                             const vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> > > &last,
		                                                                             vector<double> factor, bool FERMIONIC) -> void
		{
			ArrayXXcd Full = P.get<Eigen::ArrayXXcd>(xxxFull);
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
					vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> > ops(range+1);
					ops[0] = first[j];
					for (size_t i=1; i<range; ++i)
					{
						if (FERMIONIC) {ops[i] = F[(loc+i)%N_sites].sign().template cast<complex<double> >();}
						else {ops[i] = F[(loc+i)%N_sites].Id().template cast<complex<double> >();}
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
			SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> cUP_sign_local    = (F[loc].c(UP,0)    * F[loc].sign()).template cast<complex<double> >();
			SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> cDN_sign_local    = (F[loc].c(DN,0)    * F[loc].sign()).template cast<complex<double> >();
			SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> cdagUP_sign_local = (F[loc].cdag(UP,0) * F[loc].sign()).template cast<complex<double> >();
			SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> cdagDN_sign_local = (F[loc].cdag(DN,0) * F[loc].sign()).template cast<complex<double> >();
			
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> > cUP_ranges(N_sites);    for (size_t i=0; i<N_sites; ++i) {cUP_ranges[i]    = F[i].c(UP,0).template cast<complex<double> >();}
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> > cdagUP_ranges(N_sites); for (size_t i=0; i<N_sites; ++i) {cdagUP_ranges[i] = F[i].cdag(UP,0).template cast<complex<double> >();}
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> > cDN_ranges(N_sites);    for (size_t i=0; i<N_sites; ++i) {cDN_ranges[i]    = F[i].c(DN,0).template cast<complex<double> >();}
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> > cdagDN_ranges(N_sites); for (size_t i=0; i<N_sites; ++i) {cdagDN_ranges[i] = F[i].cdag(DN,0).template cast<complex<double> >();}
			
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> >          frst {cdagUP_sign_local, cUP_sign_local, cdagDN_sign_local, cDN_sign_local};
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> > > last {cUP_ranges, cdagUP_ranges, cDN_ranges, cdagDN_ranges};
			push_full("tFull", "tᵢⱼ", frst, last, {-1., +1., -1., +1.}, PROP::FERMIONIC);
		}
		
		// local terms: U, t0, μ, Bz, t⟂, V⟂, J⟂, X⟂
		
		param1d U = P.fill_array1d<double>("U", "Uorb", orbitals, loc%Lcell);
		param1d Uph = P.fill_array1d<double>("Uph", "Uorb", orbitals, loc%Lcell);
		param1d t0 = P.fill_array1d<double>("t0", "t0orb", orbitals, loc%Lcell);
		param1d mu = P.fill_array1d<double>("mu", "muorb", orbitals, loc%Lcell);
		param1d Bz = P.fill_array1d<double>("Bz", "Bzorb", orbitals, loc%Lcell);
		param2d tperp = P.fill_array2d<double>("tRung", "t", "tPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Vperp = P.fill_array2d<double>("Vrung", "V", "Vperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Vxyperp = P.fill_array2d<double>("Vxyrung", "Vxy", "Vxyperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Vzperp = P.fill_array2d<double>("Vzrung", "Vz", "Vzperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Jperp = P.fill_array2d<double>("Jrung", "J", "Jperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		
		labellist[loc].push_back(U.label);
		labellist[loc].push_back(Uph.label);
		labellist[loc].push_back(t0.label);
		labellist[loc].push_back(mu.label);
		labellist[loc].push_back(Bz.label);
		labellist[loc].push_back(tperp.label);
		labellist[loc].push_back(Vperp.label);
		labellist[loc].push_back(Jperp.label);
		ArrayXd  C_array = F[loc].ZeroField();
		
		//ArrayXd Vxy_array = F[loc].ZeroHopping();
		//ArrayXd Vz_array = F[loc].ZeroHopping();
		
		auto Hloc = Mpo<Symmetry,complex<double> >::get_N_site_interaction
		(
			F[loc].template HubbardHamiltonian<double>(U.a, Uph.a, t0.a-mu.a, Bz.a, tperp.a, Vperp.a, Vzperp.a, Vxyperp.a, Jperp.a, Jperp.a, C_array)
		).cast<complex<double> >());
		pushlist.push_back(std::make_tuple(loc, Hloc, 1.+0.i));
	}	
}

}

#endif
