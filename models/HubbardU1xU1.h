#ifndef STRAWBERRY_HUBBARDMODEL
#define STRAWBERRY_HUBBARDMODEL

//include "bases/FermionBase.h"
#include "symmetry/S1xS2.h"
#include "symmetry/U1.h"
//include "Mpo.h"
//include "ParamHandler.h" // from HELPERS
#include "models/HubbardObservables.h"
#include "ParamReturner.h"
#include "Geometry2D.h" // from TOOLS

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
	
	HubbardU1xU1(Mpo<Symmetry> &Mpo_input, const vector<Param> &params)
	:Mpo<Symmetry>(Mpo_input),
	 HubbardObservables(this->N_sites,params,HubbardU1xU1::defaults),
	 ParamReturner()
	{
		ParamHandler P(params,HubbardU1xU1::defaults);
		size_t Lcell = P.size();
		N_phys = 0;
		for (size_t l=0; l<N_sites; ++l) N_phys += P.get<size_t>("Ly",l%Lcell);
		this->precalc_TwoSiteData();
		this->HERMITIAN = true;
		this->HAMILTONIAN = true;
	};
	
	HubbardU1xU1 (const size_t &L, const vector<Param> &params, const BC &boundary=BC::OPEN, const DMRG::VERBOSITY::OPTION &VERB=DMRG::VERBOSITY::OPTION::ON_EXIT);
	///@}
	
	static qarray<2> singlet (int N=0, int L=0) {return qarray<2>{0,N};};
	static constexpr MODEL_FAMILY FAMILY = HUBBARD;
	static constexpr int spinfac = 2;
	
	/**
	 * \describe_set_operators
	 *
	 * \param B : Base class from which the local operators are received
	 * \param P : The parameters
	 * \param pushlist : All the local operators for the Mpo will be pushed into \p pushlist.
	 * \param labellist : All the labels for the Mpo will be put into \p labellist. Mpo::generate_label will produce a nice label from the data in labellist.
	 * \describe_boundary 
	*/	
	template<typename Symmetry_> 
	static void set_operators (const std::vector<FermionBase<Symmetry_> > &F, const ParamHandler &P,
	                           PushType<SiteOperator<Symmetry_,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary=BC::OPEN);
	
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
	{"REMOVE_DOUBLE",false}, {"REMOVE_EMPTY",false}, {"REMOVE_UP",false}, {"REMOVE_DN",false}, {"mfactor",1}, {"k",0},
	{"maxPower",2ul}, {"CYLINDER",false}, {"Ly",1ul}
};

HubbardU1xU1::
HubbardU1xU1 (const size_t &L, const vector<Param> &params, const BC &boundary, const DMRG::VERBOSITY::OPTION &VERB)
:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, boundary, VERB),
 HubbardObservables(L,params,HubbardU1xU1::defaults),
 ParamReturner()
{
	ParamHandler P(params, HubbardU1xU1::defaults);
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
	set_operators(F, P, pushlist, labellist, boundary);
	
	this->construct_from_pushlist(pushlist, labellist, Lcell);
	this->finalize(PROP::COMPRESS, P.get<size_t>("maxPower"));
	this->precalc_TwoSiteData();
}

template<typename Symmetry_>
void HubbardU1xU1::
set_operators (const std::vector<FermionBase<Symmetry_> > &F, const ParamHandler &P, PushType<SiteOperator<Symmetry_,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary)
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
			SiteOperatorQ<Symmetry_,Eigen::MatrixXd> cUP_sign_local    = F[loc].c(UP,0)    * F[loc].sign();
			SiteOperatorQ<Symmetry_,Eigen::MatrixXd> cDN_sign_local    = F[loc].c(DN,0)    * F[loc].sign();
			SiteOperatorQ<Symmetry_,Eigen::MatrixXd> cdagUP_sign_local = F[loc].cdag(UP,0) * F[loc].sign();
			SiteOperatorQ<Symmetry_,Eigen::MatrixXd> cdagDN_sign_local = F[loc].cdag(DN,0) * F[loc].sign();
			
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > cUP_ranges(N_sites);    for (size_t i=0; i<N_sites; ++i) {cUP_ranges[i]    = F[i].c(UP,0);}
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > cdagUP_ranges(N_sites); for (size_t i=0; i<N_sites; ++i) {cdagUP_ranges[i] = F[i].cdag(UP,0);}
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > cDN_ranges(N_sites);    for (size_t i=0; i<N_sites; ++i) {cDN_ranges[i]    = F[i].c(DN,0);}
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > cdagDN_ranges(N_sites); for (size_t i=0; i<N_sites; ++i) {cdagDN_ranges[i] = F[i].cdag(DN,0);}
			
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> >          frst {cdagUP_sign_local, cUP_sign_local, cdagDN_sign_local, cDN_sign_local};
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {cUP_ranges, cdagUP_ranges, cDN_ranges, cdagDN_ranges};
			push_full("tFull", "tᵢⱼ", frst, last, {-1., +1., -1., +1.}, PROP::FERMIONIC);
		}
		
		if (P.HAS("Vxyfull"))
		{
			auto Gloc = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,loc)));
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first {F[loc].Tp(0,Gloc), F[loc].Tm(0,Gloc)};
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > Tp_ranges(N_sites);
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > Tm_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++)
			{
				auto Gi = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,i)));
				Tp_ranges[i] = F[i].Tp(0,Gi);
				Tm_ranges[i] = F[i].Tm(0,Gi);
			}
			
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {Tm_ranges, Tp_ranges};
			push_full("Vxyfull", "Vxyᵢⱼ", first, last, {0.5,0.5}, PROP::BOSONIC);
		}
		
		if (P.HAS("Vzfull"))
		{
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first {F[loc].Tz(0)};
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > Tz_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++)
			{
				Tz_ranges[i] = F[i].Tz(0);
			}
			
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {Tz_ranges};
			push_full("Vzfull", "Vzᵢⱼ", first, last, {1.}, PROP::BOSONIC);
		}
		
		if (P.HAS("vzfull"))
		{
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first {F[loc].tz(0)};
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > tz_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++)
			{
				tz_ranges[i] = F[i].tz(0);
			}
			
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {tz_ranges};
			push_full("vzfull", "vzᵢⱼ", first, last, {1.}, PROP::BOSONIC);
		}
		
		if (P.HAS("VextFull"))
		{
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first {F[loc].n(0)};
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > n_ranges(N_sites); for (size_t i=0; i<N_sites; i++) {n_ranges[i] = F[i].n(0);}
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {n_ranges};
			push_full("VextFull", "Vextᵢⱼ", first, last, {1.}, PROP::BOSONIC);
		}
		
		if (P.HAS("Jfull"))
		{
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first {F[loc].Sp(0), F[loc].Sm(0), F[loc].Sz(0)};
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > Sp_ranges(N_sites);
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > Sm_ranges(N_sites);
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > Sz_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++)
			{
				Sp_ranges[i] = F[i].Sp(0);
				Sm_ranges[i] = F[i].Sm(0);
				Sz_ranges[i] = F[i].Sz(0);
			}
			
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {Sm_ranges, Sp_ranges, Sz_ranges};
			push_full("Jfull", "Jᵢⱼ", first, last, {0.5,0.5,1.}, PROP::BOSONIC);
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
		
		auto Hloc = Mpo<Symmetry,double>::get_N_site_interaction
		(
			F[loc].template HubbardHamiltonian<double>(U.a, Uph.a, t0.a-mu.a, Bz.a, tperp.a, Vperp.a, Vzperp.a, Vxyperp.a, Jperp.a, Jperp.a, C_array)
		);
		pushlist.push_back(std::make_tuple(loc, Hloc, 1.));
		
		// Nearest-neighbour terms: t, V, J, X
		
		param2d tpara = P.fill_array2d<double>("t", "tPara", {orbitals, next_orbitals}, loc%Lcell);
		param2d Vpara = P.fill_array2d<double>("V", "Vpara", {orbitals, next_orbitals}, loc%Lcell);
		param2d Jpara = P.fill_array2d<double>("J", "Jpara", {orbitals, next_orbitals}, loc%Lcell);
		param2d Xpara = P.fill_array2d<double>("X", "Xpara", {orbitals, next_orbitals}, loc%Lcell);
		param2d Vxypara = P.fill_array2d<double>("Vxy", "Vxypara", {orbitals, next_orbitals}, loc%Lcell);
		param2d Vzpara = P.fill_array2d<double>("Vz", "Vzpara", {orbitals, next_orbitals}, loc%Lcell);
		
		labellist[loc].push_back(tpara.label);
		labellist[loc].push_back(Vpara.label);
		labellist[loc].push_back(Jpara.label);
		labellist[loc].push_back(Xpara.label);
		labellist[loc].push_back(Vxypara.label);
		labellist[loc].push_back(Vzpara.label);
		
		if (loc < N_sites-1 or !static_cast<bool>(boundary))
		{
			for (std::size_t alfa=0; alfa<orbitals;      ++alfa)
			for (std::size_t beta=0; beta<next_orbitals; ++beta)
			{
				// t
				if (!P.HAS("tFull"))
				{
					if (tpara(alfa,beta) != 0.)
					{
						pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction((F[loc].cdag(UP,alfa)*F[loc].sign()), F[lp1].c(UP,beta)),    -tpara(alfa,beta)));
						pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction((F[loc].cdag(DN,alfa)*F[loc].sign()), F[lp1].c(DN,beta)),    -tpara(alfa,beta)));
						pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction((F[loc].c(UP,alfa)   *F[loc].sign()), F[lp1].cdag(UP,beta)), +tpara(alfa,beta)));
						pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction((F[loc].c(DN,alfa)   *F[loc].sign()), F[lp1].cdag(DN,beta)), +tpara(alfa,beta)));
					}
				}
				
				// V
				if (!P.HAS("Vfull"))
				{
					if (Vpara(alfa,beta) != 0.)
					{
						pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(F[loc].n(alfa), F[lp1].n(beta)), Vpara(alfa,beta)));
					}
				}
				
				// J
				if (!P.HAS("Jfull"))
				{
					if (Jpara(alfa,beta) != 0.)
					{
						pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(F[loc].Sp(alfa), F[lp1].Sm(beta)), 0.5*Jpara(alfa,beta)));
						pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(F[loc].Sm(alfa), F[lp1].Sp(beta)), 0.5*Jpara(alfa,beta)));
						pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(F[loc].Sz(alfa), F[lp1].Sz(beta)),     Jpara(alfa,beta)));
					}
				}
				
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
				if (!P.HAS("Xfull"))
				{
					if (Xpara(alfa,beta) != 0.)
					{
						pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction((F[loc].cdag(UP,alfa)*F[loc].sign() * F[loc].n(DN,alfa)),
							                                                                                (F[lp1].c(UP,beta) * (F[lp1].Id()-2.*F[lp1].n(DN,beta)))),
							                                                                                 -Xpara(alfa,beta)));
						pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction((F[loc].cdag(UP,alfa)*F[loc].sign()),
							                                                                                (F[lp1].c(UP,beta) * F[lp1].n(DN,beta))),
							                                                                                 -Xpara(alfa,beta)));
						pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction((F[loc].cdag(DN,alfa)*F[loc].sign() * F[loc].n(UP,alfa)),
																										 (F[lp1].c(DN,beta) * (F[lp1].Id()-2.*F[lp1].n(UP,beta)))),
														                                                 -Xpara(alfa,beta)));
						pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction((F[loc].cdag(DN,alfa)*F[loc].sign()),
																										 (F[lp1].c(DN,beta) * F[lp1].n(UP,beta))),
														                                                 -Xpara(alfa,beta)));
					
						pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction((F[loc].c(UP,alfa)*F[loc].sign() * F[loc].n(DN,alfa)),
																										 (F[lp1].cdag(UP,beta) * (F[lp1].Id()-2.*F[lp1].n(DN,beta)))),
														                                                 +Xpara(alfa,beta)));
						pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction((F[loc].c(UP,alfa)*F[loc].sign()),
																										 (F[lp1].cdag(UP,beta) * F[lp1].n(DN,beta))),
														                                                 +Xpara(alfa,beta)));
						pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction((F[loc].c(DN,alfa)*F[loc].sign() * F[loc].n(UP,alfa)),
																										 (F[lp1].cdag(DN,beta) * (F[lp1].Id()-2.*F[lp1].n(UP,beta)))),
														                                                 +Xpara(alfa,beta)));
						pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction((F[loc].c(DN,alfa)*F[loc].sign()),
																										 (F[lp1].cdag(DN,beta) * F[lp1].n(UP,beta))),
														                                                 +Xpara(alfa,beta)));
					}
				}
				
				// Vxy, Vz
				if (!P.HAS("Vxyfull"))
				{
					if (Vxypara(alfa,beta) != 0.)
					{
						pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(F[loc].cc(alfa), F[lp1].cdagcdag(beta)), 0.5*Vxypara(alfa,beta)*pow(-1,loc+lp1)));
						pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(F[loc].cdagcdag(alfa), F[lp1].cc(beta)), 0.5*Vxypara(alfa,beta)*pow(-1,loc+lp1)));
					}
				}
				if (!P.HAS("Vzfull"))
				{
					if (Vzpara(alfa,beta) != 0.)
					{
						pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(F[loc].Tz(alfa), F[lp1].Tz(beta)),           Vzpara (alfa,beta)));
					}
				}
			}
		}
		
		// Next-nearest-neighbour terms: t'
		param2d tPrime = P.fill_array2d<double>("tPrime", "tPrime_array", {orbitals, nextn_orbitals}, loc%Lcell);
		labellist[loc].push_back(tPrime.label);
		if (loc < N_sites-2 or !static_cast<bool>(boundary))
		{
			for (std::size_t alfa=0; alfa<orbitals;       ++alfa)
			for (std::size_t beta=0; beta<nextn_orbitals; ++beta)
			{
				if (tPrime(alfa,beta) != 0.)
				{
					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction((F[loc].cdag(UP,alfa)*F[loc].sign()),
					                                                                                      F[lp1].sign(),
					                                                                                      F[lp2].c(UP,beta)),    -tPrime(alfa,beta)));
					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction((F[loc].cdag(DN,alfa)*F[loc].sign()),
					                                                                                      F[lp1].sign(),
					                                                                                      F[lp2].c(DN,beta)),    -tPrime(alfa,beta)));
					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction((F[loc].c(UP,alfa)   *F[loc].sign()),
					                                                                                      F[lp1].sign(),
					                                                                                      F[lp2].cdag(UP,beta)), +tPrime(alfa,beta)));
					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction((F[loc].c(DN,alfa)   *F[loc].sign()),
					                                                                                      F[lp1].sign(),
					                                                                                      F[lp2].cdag(DN,beta)), +tPrime(alfa,beta)));
				}
			}
		}
		
		/*param0d J3site = P.fill_array0d<double>("J3site", "J3site", loc%Lcell);
		labellist[loc].push_back(J3site.label);
		
		if (J3site.x != 0.)
		{
			lout << "Warning! J3site has to be tested against ED!" << endl;
			
			assert(orbitals == 1 and "Cannot do a ladder with 3-site J terms!");
			if (loc < N_sites-2 or !static_cast<bool>(boundary) )
			{
				SiteOperatorQ<Symmetry_, Eigen::MatrixXd> cup_sign_local    = F[loc].c(UP)    * F[loc].sign();
				SiteOperatorQ<Symmetry_, Eigen::MatrixXd> cdn_sign_local    = F[loc].c(DN)    * F[loc].sign();
				SiteOperatorQ<Symmetry_, Eigen::MatrixXd> cupdag_sign_local = F[loc].cdag(UP) * F[loc].sign();
				SiteOperatorQ<Symmetry_, Eigen::MatrixXd> cdndag_sign_local = F[loc].cdag(DN) * F[loc].sign();
				
				SiteOperatorQ<Symmetry_, Eigen::MatrixXd> nup_sign_tight = F[lp1].n(UP) * F[lp1].sign();
				SiteOperatorQ<Symmetry_, Eigen::MatrixXd> ndn_sign_tight = F[lp1].n(UP) * F[lp1].sign();
				SiteOperatorQ<Symmetry_, Eigen::MatrixXd> Sp_sign_tight  = F[lp1].Sp()  * F[lp1].sign();
				SiteOperatorQ<Symmetry_, Eigen::MatrixXd> Sm_sign_tight  = F[lp1].Sm()  * F[lp1].sign();
				
				SiteOperatorQ<Symmetry_, Eigen::MatrixXd> cup_nextn    = F[lp2].c(UP);
				SiteOperatorQ<Symmetry_, Eigen::MatrixXd> cdn_nextn    = F[lp2].c(DN);
				SiteOperatorQ<Symmetry_, Eigen::MatrixXd> cupdag_nextn = F[lp2].cdag(UP);
				SiteOperatorQ<Symmetry_, Eigen::MatrixXd> cdndag_nextn = F[lp2].cdag(DN);
				
				// three-site terms without spinflip
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(cupdag_sign_local, ndn_sign_tight, cup_nextn),    -0.25*J3site.x));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(cdndag_sign_local, nup_sign_tight, cdn_nextn),    -0.25*J3site.x));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(cup_sign_local, ndn_sign_tight, cupdag_nextn),    +0.25*J3site.x));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(cdn_sign_local, nup_sign_tight, cdndag_nextn),    +0.25*J3site.x));
								
				// three-site terms with spinflip
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(cupdag_sign_local, Sm_sign_tight, cdn_nextn),    -0.25*J3site.x));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(cdndag_sign_local, Sp_sign_tight, cup_nextn),    -0.25*J3site.x));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(cup_sign_local, Sp_sign_tight, cdndag_nextn),    +0.25*J3site.x));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(cdn_sign_local, Sm_sign_tight, cupdag_nextn),    +0.25*J3site.x));				
			}
		}*/
	}	
}

} // end namespace VMPS::models

#endif
