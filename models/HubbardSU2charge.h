#ifndef HUBBARDMODELSU2CHARGE_H_
#define HUBBARDMODELSU2CHARGE_H_

#include "models/HubbardSU2xU1.h"
#include "symmetry/SU2.h"
#include "bases/FermionBase.h"
#include "models/HubbardObservables.h"
#include "Mpo.h"
#include "ParamReturner.h"
#include "Geometry2D.h" // from TOOLS

namespace VMPS
{

/** \class HubbardSU2
  * \ingroup Hubbard
  *
  * \brief Hubbard Model
  *
  * MPO representation of 
  * 
  * \f$
  * H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
  * - t^{\prime} \sum_{<<ij>>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
  * + U \sum_i n_{i\uparrow} n_{i\downarrow}
  * + V \sum_{<ij>} n_{i} n_{j}
  * - X \sum_{<ij>\sigma} \left( c^\dagger_{i\sigma}c_{j\sigma} + h.c.\right) \left(n_{i,-\sigma}-n_{j,-\sigma}\right)^2
  * +H_{tJ}
  * \f$.
  * with
  * \f[
  * H_{tJ} = +J \sum_{<ij>} (\mathbf{S}_{i} \mathbf{S}_{j} - \frac{1}{4} n_in_j)
  * \f]
  * \note: The term before \f$n_i n_j\f$ is not set and has to be adjusted with \p V
  * \note Makes use only of the spin-SU(2) smmetry.
  * \note If the nnn-hopping is positive, the ground state energy is lowered.
  * \warning \f$J>0\f$ is antiferromagnetic
  */
class HubbardSU2charge : public Mpo<Sym::SU2<Sym::ChargeSU2> ,double>,
				   public HubbardObservables<Sym::SU2<Sym::ChargeSU2> >,
				   public ParamReturner
{
public:
	
	typedef Sym::SU2<Sym::ChargeSU2> Symmetry;
	MAKE_TYPEDEFS(HubbardSU2charge)
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	
private:
	
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	
public:
	
	///@{
	HubbardSU2charge() : Mpo(){};
	
	HubbardSU2charge(Mpo<Symmetry> &Mpo_input, const vector<Param> &params)
	:Mpo<Symmetry>(Mpo_input),
	 HubbardObservables(this->N_sites,params,HubbardSU2charge::defaults),
	 ParamReturner(HubbardSU2charge::sweep_defaults)
	{
		ParamHandler P(params,HubbardSU2charge::defaults);
		size_t Lcell = P.size();
		N_phys = 0;
		for (size_t l=0; l<N_sites; ++l) N_phys += P.get<size_t>("Ly",l%Lcell);
		this->precalc_TwoSiteData();
	};
	
	HubbardSU2charge (const size_t &L, const vector<Param> &params, const BC &boundary=BC::OPEN, const DMRG::VERBOSITY::OPTION &VERB=DMRG::VERBOSITY::OPTION::ON_EXIT);
	///@}
	
	template<typename Symmetry_> 
	static void set_operators (const std::vector<FermionBase<Symmetry_> > &F, const ParamHandler &P,
	                           PushType<SiteOperator<Symmetry_,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, 
	                           const BC boundary=BC::OPEN);
	
	static qarray<1> singlet (int N=0) {return qarray<1>{1};};
	static constexpr MODEL_FAMILY FAMILY = HUBBARD;
	static constexpr int spinfac = 1;
	
	static const map<string,any> defaults;
	static const map<string,any> sweep_defaults;
};

// V is standard next-nearest neighbour density interaction
// Vz and Vxy are anisotropic isospin-isospin next-nearest neighbour interaction
const map<string,any> HubbardSU2charge::defaults = 
{
	{"t",1.}, {"tPrime",0.}, {"tRung",1.}, {"tPrimePrime",0.},
	{"mu",0.}, {"t0",0.}, 
	{"U",0.}, {"Uph",0.},
	{"V",0.}, {"Vrung",0.},
	{"Jz",0.}, {"Jzrung",0.}, {"Jxy",0.}, {"Jxyrung",0.}, 
	{"J",0.}, {"Jperp",0.},
	{"Bz",0.}, {"Bx",0.},
	{"X",0.}, {"Xrung",0.},
	{"REMOVE_DOUBLE",false}, {"REMOVE_EMPTY",false}, {"REMOVE_SINGLE",false}, {"mfactor",1}, 
	{"maxPower",2ul}, {"CYLINDER",false}, {"Ly",1ul}
};

const map<string,any> HubbardSU2charge::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.}, {"lim_alpha",11ul}, {"eps_svd",1e-7},
	{"Dincr_abs", 4ul}, {"Dincr_per", 2ul}, {"Dincr_rel", 1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",24ul}, {"min_halfsweeps",6ul},
	{"Minit",1ul}, {"Qinit",1ul}, {"Mlimit",500ul},
	{"tol_eigval",1e-7}, {"tol_state",1e-6},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
};

HubbardSU2charge::
HubbardSU2charge (const size_t &L, const vector<Param> &params, const BC &boundary, const DMRG::VERBOSITY::OPTION &VERB)
:Mpo<Symmetry> (L, qarray<Symmetry::Nq>({1}), "", PROP::HERMITIAN, PROP::NON_UNITARY, boundary, VERB),
 HubbardObservables(L,params,HubbardSU2charge::defaults),
 ParamReturner(HubbardSU2charge::sweep_defaults)
{
	ParamHandler P(params,defaults);	
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
	else
	{
		this->set_name("U=∞-Hubbard");
	}
	
	PushType<SiteOperator<Symmetry,double>,double> pushlist;
	std::vector<std::vector<std::string>> labellist;
	HubbardSU2charge::set_operators(F, P, pushlist, labellist, boundary);
	//add_operators(F, P, pushlist, labellist, boundary);
	
	this->construct_from_pushlist(pushlist, labellist, Lcell);
	this->finalize(PROP::COMPRESS, P.get<size_t>("maxPower"));
	
	this->precalc_TwoSiteData();
}

template<typename Symmetry_>
void HubbardSU2charge::
set_operators (const std::vector<FermionBase<Symmetry_> > &F, const ParamHandler &P, PushType<SiteOperator<Symmetry_,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = F.size();
	if(labellist.size() != N_sites) {labellist.resize(N_sites);}
	
	for (std::size_t loc=0; loc<N_sites; ++loc)
	{
		size_t lp1 = (loc+1)%N_sites;
		size_t lp2 = (loc+2)%N_sites;
		size_t lp3 = (loc+3)%N_sites;
		
		std::size_t orbitals       = F[loc].orbitals();
		std::size_t next_orbitals  = F[lp1].orbitals();
		std::size_t nextn_orbitals = F[lp2].orbitals();
		std::size_t nnextn_orbitals = F[lp3].orbitals();
		
		vector<SUB_LATTICE> G(N_sites);
		if (P.HAS("G")) {G = P.get<vector<SUB_LATTICE> >("G");}
		else // set default (-1)^l
		{
			G[0] = static_cast<SUB_LATTICE>(1);
			for (int l=1; l<N_sites; l+=1) G[l] = static_cast<SUB_LATTICE>(-1*G[l-1]);
		}
		
//		auto Gloc = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,loc)));
//		auto Glp1 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,lp1)));
//		auto Glp2 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,lp2)));
//		auto Glp3 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,lp3)));
		
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
			SiteOperatorQ<Symmetry_,Eigen::MatrixXd> cdagup_sign_local = F[loc].cdag(UP,G[loc],0) * F[loc].sign();
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > cup_ranges(N_sites);
			SiteOperatorQ<Symmetry_,Eigen::MatrixXd> cdagdn_sign_local = F[loc].cdag(DN,G[loc],0) * F[loc].sign();
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > cdn_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++)
			{
				//auto Gi = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,i)));
				cup_ranges[i] = F[i].c(UP,G[i],0);
			}
			for (size_t i=0; i<N_sites; i++)
			{
				//auto Gi = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,i)));
				cdn_ranges[i] = F[i].c(DN,G[i],0);
			}
			
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first {cdagup_sign_local, cdagdn_sign_local};
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {cup_ranges,cdn_ranges};
			push_full("tFull", "tᵢⱼ", first, last, {-std::sqrt(2.),- std::sqrt(2.)}, PROP::FERMIONIC);
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
		// Local terms: U, t0, μ, t⟂, V⟂, J⟂
		
		param1d Uph = P.fill_array1d<double>("Uph", "Uphorb", orbitals, loc%Lcell);
		param1d V = P.fill_array1d<double>("V", "Vorb", orbitals, loc%Lcell);
		//param1d t0 = P.fill_array1d<double>("t0", "t0orb", orbitals, loc%Lcell);
		//param1d mu = P.fill_array1d<double>("mu", "muorb", orbitals, loc%Lcell);
		param2d tPerp = P.fill_array2d<double>("tRung", "t", "tPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Jxyperp = P.fill_array2d<double>("JxyRung", "Jxy", "JxyPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Jzperp = P.fill_array2d<double>("JzRung", "Jz", "JzPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Jperp = P.fill_array2d<double>("JRung", "J", "JPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param1d Bz = P.fill_array1d<double>("Bz", "Bzorb", orbitals, loc%Lcell);
		param1d Bx = P.fill_array1d<double>("Bx", "Bxorb", orbitals, loc%Lcell);
		
		labellist[loc].push_back(Uph.label);
		//labellist[loc].push_back(t0.label);
		//labellist[loc].push_back(mu.label);
		labellist[loc].push_back(tPerp.label);
		labellist[loc].push_back(Jxyperp.label);
		labellist[loc].push_back(Jzperp.label);
		labellist[loc].push_back(Jperp.label);
		labellist[loc].push_back(Bz.label);
		labellist[loc].push_back(Bx.label);
		
		ArrayXXd Vperp = F[loc].ZeroHopping();
		
		auto sum_array = [] (const ArrayXXd& a1, const ArrayXXd& a2)
		{
			ArrayXXd res(a1.rows(), a1.cols());
			for (int i=0; i<a1.rows(); ++i)
			for (int j=0; j<a1.rows(); ++j)
			{
				res(i,j) = a1(i,j) + a2(i,j);
			}
			return res;
		};
		
		auto Hloc = Mpo<Symmetry_,double>::get_N_site_interaction
		(
			//HubbardHamiltonian(U.a,tPerp.a,Vperp,Jzsubperp,Jxysubperp,Bzsub.a,Bxsub.a));
			F[loc].template HubbardHamiltonian<double>(Uph.a, tPerp.a, Vperp, sum_array(Jperp.a,Jzperp.a), sum_array(Jperp.a,Jxyperp.a), Bz.a, Bx.a)
		);
		pushlist.push_back(std::make_tuple(loc, Hloc, 1.));
		
		// Nearest-neighbour terms: t, V, J
		
//		if (!P.HAS("tFull") and !P.HAS("Jfull"))
//		{
//			param2d tpara = P.fill_array2d<double>("t", "tPara", {orbitals, next_orbitals}, loc%Lcell);
//			param2d Vpara = P.fill_array2d<double>("V", "Vpara", {orbitals, next_orbitals}, loc%Lcell);
//			param2d Vzpara = P.fill_array2d<double>("Vz", "Vzpara", {orbitals, next_orbitals}, loc%Lcell);
//			param2d Vxypara = P.fill_array2d<double>("Vxy", "Vxypara", {orbitals, next_orbitals}, loc%Lcell);
//			param2d Jpara = P.fill_array2d<double>("J", "Jpara", {orbitals, next_orbitals}, loc%Lcell);
//			param2d Xpara = P.fill_array2d<double>("X", "Xpara", {orbitals, next_orbitals}, loc%Lcell);
//			
//			labellist[loc].push_back(tpara.label);
//			labellist[loc].push_back(Vpara.label);
//			labellist[loc].push_back(Vzpara.label);
//			labellist[loc].push_back(Vxypara.label);
//			labellist[loc].push_back(Jpara.label);
//			labellist[loc].push_back(Xpara.label);
//			
//			if (loc < N_sites-1 or !static_cast<bool>(boundary))
//			{
//				for (std::size_t alfa=0; alfa<orbitals;      ++alfa)
//				for (std::size_t beta=0; beta<next_orbitals; ++beta)
//				{
//					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> c_sign_local    = (F[loc].c(alfa) *    F[loc].sign());
//					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> cdag_sign_local = (F[loc].cdag(alfa) * F[loc].sign());
//					
//					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> c_tight    = F[lp1].c   (beta);
//					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> cdag_tight = F[lp1].cdag(beta);
//					
//					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> n_local = F[loc].n(alfa);
//					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> n_tight = F[lp1].n(beta);
//					
//					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> tz_local = F[loc].Tz(alfa);
//					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> tz_tight = F[lp1].Tz(beta);
//					
//					auto Gloc = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,loc)));
//					auto Glp1 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,lp1)));
//					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> tp_local = F[loc].Tp(alfa,Gloc);
//					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> tm_tight = F[lp1].Tm(beta,Glp1);
//					
//					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> tm_local = F[loc].Tm(alfa,Gloc);
//					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> tp_tight = F[lp1].Tp(beta,Glp1);
//					
//					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> Sdag_local = F[loc].Sdag(alfa);
//					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> S_tight    = F[lp1].S   (beta);
//					
//					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> PsiLloc = ((F[loc].ns(alfa) * F[loc].c(alfa)) * F[loc].sign());
//					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> PsiRloc = ((F[loc].c(alfa) * F[loc].sign()) * F[loc].ns(alfa));
//					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> PsiLlp1 = (F[lp1].ns(beta) * F[lp1].c(beta));
//					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> PsiRlp1 = (F[lp1].c(beta) * F[lp1].ns(beta));
//					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> PsidagLloc = ((F[loc].cdag(alfa) * F[loc].sign()) * F[loc].ns(alfa));
//					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> PsidagRloc = ((F[loc].ns(alfa) * F[loc].cdag(alfa)) * F[loc].sign());
//					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> PsidagLlp1 = (F[lp1].cdag(beta) * F[lp1].ns(beta));
//					SiteOperatorQ<Symmetry_,Eigen::MatrixXd> PsidagRlp1 = (F[lp1].ns(beta) * F[lp1].cdag(beta));
//					
//					//hopping
//					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(cdag_sign_local, c_tight), -std::sqrt(2.)*tpara(alfa,beta)));
//					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(c_sign_local, cdag_tight), -std::sqrt(2.)*tpara(alfa,beta)));
//					
//					//density-density interaction
//					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(n_local, n_tight), Vpara(alfa,beta)));
//					
//					//isospin-isopsin interaction
//					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(tp_local, tm_tight), 0.5*Vxypara(alfa,beta)));
//					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(tm_local, tp_tight), 0.5*Vxypara(alfa,beta)));
//					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(tz_local, tz_tight),     Vzpara (alfa,beta)));
//					
//					//spin-spin interaction
//					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(Sdag_local, S_tight), std::sqrt(3.)*Jpara(alfa,beta)));
//					
//					//correlated hopping
//					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(PsidagLloc, PsiRlp1), -std::sqrt(2.)*Xpara(alfa,beta)));
//					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(PsidagRloc, PsiLlp1), -std::sqrt(2.)*Xpara(alfa,beta)));
//					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(PsiLloc, PsidagRlp1), -std::sqrt(2.)*Xpara(alfa,beta)));
//					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(PsiRloc, PsidagLlp1), -std::sqrt(2.)*Xpara(alfa,beta)));
//				}
//			}
//		}
	}
}

} // end namespace VMPS::models

#endif
