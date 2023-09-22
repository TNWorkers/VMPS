#ifndef HUBBARDMODELSU2XSU2_H_
#define HUBBARDMODELSU2XSU2_H_

#include "symmetry/S1xS2.h"
#include "symmetry/SU2.h"
#include "bases/FermionBase.h"
#include "models/HubbardObservables.h"
#include "Mpo.h"
#include "ParamReturner.h"
#include "Geometry2D.h" // from TOOLS

namespace VMPS
{

/** 
 * \class HubbardSU2xSU2
 * \ingroup Hubbard
 *
 * \brief Hubbard Model
 *
 * MPO representation of 
 * 
 * \f[
 * H = -t \sum_{<ij>\sigma} (c^\dagger_{i\sigma}c_{j\sigma} + h.c.)
 *     +U \sum_i \left[\left(n_{i\uparrow}-\frac{1}{2}\right)\left(n_{i\downarrow}-\frac{1}{2}\right) +\frac{1}{4}\right]
 *     +V \sum_{<ij>} \mathbf{T}_i \mathbf{T}_j
 *     +J \sum_{<ij>} \mathbf{S}_i \mathbf{S}_j
 *     -X \sum_{<ij>\sigma} \left(c^\dagger_{i\sigma}c_{j\sigma} + h.c.\right) \left(n_{i,-\sigma}-n_{j,-\sigma}\right)^2
 * \f]
 * with \f$T^+_i = (-1)^i c^{\dagger}_{i\uparrow} c^{\dagger}_{i\downarrow}\f$, \f$Q^-_i = (T^+_i)^{\dagger}\f$, \f$T^z_i = 0.5(n_{i}-1)\f$
 *
 * \note Makes use of the spin-SU(2) symmetry and the charge-SU(2) symmetry.
 * \warning Bipartite hopping structure is mandatory (particle-hole symmetry)!
 * \warning \f$J>0\f$ is antiferromagnetic
 */
class HubbardSU2xSU2 : public Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > ,double>,
					   public HubbardObservables<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > >,
					   public ParamReturner
{
public:
	typedef Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > Symmetry;
	MAKE_TYPEDEFS(HubbardSU2xSU2)
    
private:
	
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	
public:
	
	///@{
	HubbardSU2xSU2() : Mpo(){};
	
	HubbardSU2xSU2(Mpo<Symmetry> &Mpo_input, const vector<Param> &params)
	:Mpo<Symmetry>(Mpo_input),
	 HubbardObservables(this->N_sites,params,HubbardSU2xSU2::defaults),
	 ParamReturner(HubbardSU2xSU2::sweep_defaults)
	{
		ParamHandler P(params,HubbardSU2xSU2::defaults);
		size_t Lcell = P.size();
		N_phys = 0;
		for (size_t l=0; l<N_sites; ++l) N_phys += P.get<size_t>("Ly",l%Lcell);
		this->precalc_TwoSiteData();
		this->HERMITIAN = true;
		this->HAMILTONIAN = true;
	};
	
	HubbardSU2xSU2 (const size_t &L, const vector<Param> &params, const BC &boundary=BC::OPEN, const DMRG::VERBOSITY::OPTION &VERB=DMRG::VERBOSITY::OPTION::ON_EXIT);
	///@}
	
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
	static void set_operators (const std::vector<FermionBase<Symmetry_> > &F, 
	                           const vector<SUB_LATTICE> &G, const ParamHandler &P,
	                           PushType<SiteOperator<Symmetry_,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, 
	                           const BC boundary=BC::OPEN);
	
	static qarray<2> singlet (int N=0, int L=0)
	{
		assert(N%2==0);
		int T = abs(0.5*(N-L));
		return qarray<2>{1,2*T+1};
	};
	static constexpr MODEL_FAMILY FAMILY = HUBBARD;
	static constexpr int spinfac = 1;
	
	Mpo<Symmetry> B (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const {return cdagc(locx1,locx2,locy1,locy2);};
	Mpo<Symmetry> C (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
		
	static const map<string,any> defaults;
	static const map<string,any> sweep_defaults;
};

const map<string,any> HubbardSU2xSU2::defaults = 
{
	{"t",1.}, {"tRung",1.}, {"tPrimePrime",0.}, 
	{"Uph",0.},
	{"V",0.}, {"Vrung",0.},
	{"J",0.}, {"Jrung",0.},
	{"X",0.}, {"Xrung",0.},
	{"REMOVE_DOUBLE",false}, {"REMOVE_EMPTY",false}, {"REMOVE_UP",false}, {"REMOVE_DN",false}, {"mfactor",1}, {"k",1}, 
	{"maxPower",2ul}, {"CYLINDER",false}, {"Ly",1ul}
};

const map<string,any> HubbardSU2xSU2::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1e-11}, {"lim_alpha",11ul}, {"eps_svd",1e-7},
	{"Dincr_abs", 2ul}, {"Dincr_per", 2ul}, {"Dincr_rel", 1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",30ul}, {"min_halfsweeps",6ul},
	{"Dinit",4ul}, {"Qinit",10ul}, {"Dlimit",500ul},
	{"tol_eigval",1e-6}, {"tol_state",1e-5},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
};

HubbardSU2xSU2::
HubbardSU2xSU2 (const size_t &L, const vector<Param> &params, const BC &boundary, const DMRG::VERBOSITY::OPTION &VERB)
:Mpo<Symmetry> (L, qarray<Symmetry::Nq>({1,1}), "", PROP::HERMITIAN, PROP::NON_UNITARY, boundary, VERB),
 HubbardObservables(L,params,HubbardSU2xSU2::defaults),
 ParamReturner(HubbardSU2xSU2::sweep_defaults)
{
	ParamHandler P(params,defaults);
	size_t Lcell = P.size();
	
//	assert(Lcell > 1 and "You need to set a unit cell with at least Lcell=2 for the charge-SU(2) symmetry!");	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);		
		setLocBasis(F[l].get_basis().qloc(),l);
	}

	this->set_name("Hubbard");

	PushType<SiteOperator<Symmetry,double>,double> pushlist;
    std::vector<std::vector<std::string>> labellist;
    set_operators(F, G, P, pushlist, labellist, boundary); // F, G are set in HubbardObservables
    
    this->construct_from_pushlist(pushlist, labellist, Lcell);
    this->finalize(PROP::COMPRESS, P.get<size_t>("maxPower"));

	this->precalc_TwoSiteData();
}

template<typename Symmetry_>
void HubbardSU2xSU2::
set_operators (const std::vector<FermionBase<Symmetry_> > &F, const vector<SUB_LATTICE> &G, const ParamHandler &P, PushType<SiteOperator<Symmetry_,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = F.size();
	if(labellist.size() != N_sites) {labellist.resize(N_sites);}
	
	for(std::size_t loc=0; loc<N_sites; ++loc)
	{
		size_t lp1 = (loc+1)%N_sites;
		size_t lp2 = (loc+2)%N_sites;
		size_t lp3 = (loc+3)%N_sites;
		
		//auto Gloc = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,loc)));
		//auto Glp1 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,lp1)));
		//auto Glp2 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,lp2)));
		//auto Glp3 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,lp3)));
		//lout << G[loc] << "\t" << G[lp1] << "\t" << G[lp2] << "\t" << G[lp3] << endl;
		
		std::size_t orbitals       = F[loc].orbitals();
		std::size_t next_orbitals  = F[lp1].orbitals();
		std::size_t next3_orbitals = F[lp3].orbitals();
		
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
			SiteOperatorQ<Symmetry_,Eigen::MatrixXd> cdag_sign_local = (F[loc].cdag(G[loc],0) * F[loc].sign());
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > c_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++)
			{
				c_ranges[i] = F[i].c(G[i],0);
			}
			
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first {cdag_sign_local};
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {c_ranges};
			push_full("tFull", "tᵢⱼ", first, last, {-std::sqrt(2.) * std::sqrt(2.)}, PROP::FERMIONIC);
		}
		
		if (P.HAS("Vfull"))
		{
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first {F[loc].Tdag(0)};
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > T_ranges(N_sites); for (size_t i=0; i<N_sites; i++) {T_ranges[i] = F[i].T(0);}
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {T_ranges};
			push_full("Vfull", "Jᵢⱼ", first, last, {std::sqrt(3.)}, PROP::BOSONIC);
		}
		
		if (P.HAS("Jfull"))
		{
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first {F[loc].Sdag(0)};
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > S_ranges(N_sites); for (size_t i=0; i<N_sites; i++) {S_ranges[i] = F[i].S(0);}
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {S_ranges};
			push_full("Jfull", "Jᵢⱼ", first, last, {std::sqrt(3.)}, PROP::BOSONIC);
		}
		
		if (P.HAS("Xfull"))
		{
			SiteOperatorQ<Symmetry_,Eigen::MatrixXd> PsidagRloc = ((F[loc].ns() * F[loc].cdag(G[loc])) * F[loc].sign());
			SiteOperatorQ<Symmetry_,Eigen::MatrixXd> PsidagLloc = ((F[loc].cdag(G[loc]) * F[loc].sign()) * F[loc].ns());

			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > PsiLran(N_sites);
			for(size_t i=0; i<N_sites; i++)
			{
				PsiLran[i] = (F[i].ns() * F[i].c(G[i]));
			}
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > PsiRran(N_sites);
			for(size_t i=0; i<N_sites; i++)
			{
				PsiRran[i] = (F[i].c(G[i]) * F[i].ns());
			}
			
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first {PsidagLloc,PsidagRloc};
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {PsiRran,PsiLran};
			push_full("Xfull", "Xᵢⱼ", first, last, {-std::sqrt(2.) * std::sqrt(2.), -std::sqrt(2.) * std::sqrt(2.)}, PROP::FERMIONIC);
		}
		
		// Local terms: Hubbard-U, t⟂, V⟂, J⟂
		
		param1d Uph = P.fill_array1d<double>("Uph", "Uphorb", orbitals, loc%Lcell);
		param2d tperp = P.fill_array2d<double>("tRung", "t", "tPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Vperp = P.fill_array2d<double>("Vrung", "V", "Vperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Jperp = P.fill_array2d<double>("Jrung", "J", "Jperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		
		labellist[loc].push_back(Uph.label);
		labellist[loc].push_back(tperp.label);
		labellist[loc].push_back(Vperp.label);
		labellist[loc].push_back(Jperp.label);
		
		auto Hloc = Mpo<Symmetry_,double>::get_N_site_interaction(F[loc].HubbardHamiltonian(Uph.a, tperp.a, Vperp.a, Jperp.a));
		pushlist.push_back(std::make_tuple(loc, Hloc, 1.));
		
		// Nearest-neighbour terms: t, V, J, X
		
		if (!P.HAS("tFull"))
		{
			param2d tpara = P.fill_array2d<double>("t", "tPara", {orbitals, next_orbitals}, loc%Lcell);
			labellist[loc].push_back(tpara.label);
			
			if (loc < N_sites-1 or !static_cast<bool>(boundary))
			{
				for (std::size_t alfa=0; alfa<orbitals;      ++alfa)
				for (std::size_t beta=0; beta<next_orbitals; ++beta)
				{
					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> cdag_sign_local = (F[loc].cdag(G[loc], alfa) * F[loc].sign());
					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> c_tight         = F[lp1].c(G[lp1], beta);

					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(cdag_sign_local, c_tight), -std::sqrt(2.)*std::sqrt(2.)*tpara(alfa,beta)));
				}
			}
		}
		
		if (!P.HAS("Vfull"))
		{
			param2d Vpara = P.fill_array2d<double>("V", "Vpara", {orbitals, next_orbitals}, loc%Lcell);
			labellist[loc].push_back(Vpara.label);
			
			if (loc < N_sites-1 or !static_cast<bool>(boundary))
			{
				for (std::size_t alfa=0; alfa<orbitals;      ++alfa)
				for (std::size_t beta=0; beta<next_orbitals; ++beta)
				{
					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> Tdag_local = F[loc].Tdag(alfa);
					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> T_tight  = F[lp1].T(beta);
					
					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(Tdag_local, T_tight), std::sqrt(3.)*Vpara(alfa,beta)));
				}
			}
		}
		
		if (!P.HAS("Jfull"))
		{
			param2d Jpara = P.fill_array2d<double>("J", "Jpara", {orbitals, next_orbitals}, loc%Lcell);
			labellist[loc].push_back(Jpara.label);
			
			if (loc < N_sites-1 or !static_cast<bool>(boundary))
			{
				for (std::size_t alfa=0; alfa<orbitals;      ++alfa)
				for (std::size_t beta=0; beta<next_orbitals; ++beta)
				{
					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> Sdag_local = F[loc].Sdag(alfa);
					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> S_tight  = F[lp1].S(beta);

					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(Sdag_local, S_tight), std::sqrt(3.)*Jpara(alfa,beta)));
				}
			}
		}
		
		if (!P.HAS("Xfull"))
		{
			param2d Xpara = P.fill_array2d<double>("X", "Xpara", {orbitals, next_orbitals}, loc%Lcell);
			labellist[loc].push_back(Xpara.label);
			
			if (loc < N_sites-1 or !static_cast<bool>(boundary))
			{
				for (std::size_t alfa=0; alfa<orbitals;      ++alfa)
				for (std::size_t beta=0; beta<next_orbitals; ++beta)
				{
					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> PsiRdag_loc = ((F[loc].ns(alfa) * F[loc].cdag(G[loc],alfa)) * F[loc].sign());
					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> PsiR_tight = (F[lp1].c(G[lp1],beta) * F[lp1].ns(beta));
					
					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> PsiLdag_loc = ((F[loc].cdag(G[loc],alfa) * F[loc].sign()) * F[loc].ns(alfa));
					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> PsiL_tight  = (F[lp1].ns(beta) * F[lp1].c(G[lp1],beta));

					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(PsiLdag_loc, PsiR_tight), -std::sqrt(2.)*std::sqrt(2.)*Xpara(alfa,beta)));
					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(PsiRdag_loc, PsiL_tight), -std::sqrt(2.)*std::sqrt(2.)*Xpara(alfa,beta)));
				}
			}
		}
		
		if (!P.HAS("tFull"))
		{
			// tPrimePrime
			param2d tPrimePrime = P.fill_array2d<double>("tPrimePrime", "tPrimePrime_array", {orbitals, next3_orbitals}, loc%Lcell);
			labellist[loc].push_back(tPrimePrime.label);
			
			if (loc < N_sites-2 or !static_cast<bool>(boundary))
			{
				SiteOperatorQ<Symmetry_,Eigen::MatrixXd> sign_tight = F[lp1].sign();
				SiteOperatorQ<Symmetry_,Eigen::MatrixXd> sign_nextn = F[lp2].sign();
				
				for (std::size_t alfa=0; alfa<orbitals;       ++alfa)
				for (std::size_t beta=0; beta<next3_orbitals; ++beta)
				{
					SiteOperatorQ<Symmetry_, Eigen::MatrixXd> cdag_sign_local = (F[loc].cdag(G[loc],alfa) * F[loc].sign());
					SiteOperatorQ<Symmetry_, Eigen::MatrixXd> c_nnextn         = F[lp3].c(G[lp3],beta);
					
					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(cdag_sign_local, sign_tight, sign_nextn, c_nnextn), -std::sqrt(2.)*std::sqrt(2.)*tPrimePrime(alfa,beta)));
				}
			}
		}
	}
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > HubbardSU2xSU2::
C (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	return Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > >::Identity(this->locBasis());
	// return make_corr("c†", "c", locx1, locx2, locy1, locy2, F[locx1].cdag(SUB_LATTICE::A,locy1), F[locx2].c(SUB_LATTICE::B,locy2), {3,1}, 2., PROP::FERMIONIC, PROP::HERMITIAN);
}

} //end namespace VMPS
#endif
