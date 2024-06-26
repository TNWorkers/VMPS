#ifndef KONDOMODEL_U0XU1_H_
#define KONDOMODEL_U0XU1_H_

#include "symmetry/SU2.h"
#include "bases/SpinBase.h"
#include "bases/FermionBase.h"
#include "models/KondoObservables.h"
#include "Mpo.h"
//include "ParamHandler.h" // from HELPERS
#include "ParamReturner.h"

namespace VMPS
{
/** \class KondoU0xSU2
  * \ingroup Kondo
  *
  * \brief Kondo Model
  *
  * MPO representation of
  * \f[
  * H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} -t^{\prime} \sum_{<<ij>>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
  * - J \sum_{i \in I} \mathbf{S}_i \cdot \mathbf{s}_i
  * \f].
  *
  * where further parameters from HubbardSU2xU1 and HeisenbergSU2 are possible.
  * \note The default variable settings can be seen in \p KondoU0xSU2::defaults.
  * \note Take use of the Spin SU(2) symmetry and U(1) charge symmetry.
  * \note If the nnn-hopping is positive, the ground state energy is lowered.
  * \warning \f$J<0\f$ is antiferromagnetic
  */
class KondoU0xSU2 : public Mpo<Sym::SU2<Sym::ChargeSU2>,double>,
					public KondoObservables<Sym::SU2<Sym::ChargeSU2> >,
					public ParamReturner
{
public:
	typedef Sym::SU2<Sym::ChargeSU2> Symmetry;
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	MAKE_TYPEDEFS(KondoU0xSU2)
	
private:
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	
public:
	
	///@{
	KondoU0xSU2 (): Mpo(), KondoObservables(), ParamReturner(KondoU0xSU2::sweep_defaults) {};
	KondoU0xSU2 (const size_t &L, const vector<Param> &params, const BC &boundary=BC::OPEN, const DMRG::VERBOSITY::OPTION &VERB=DMRG::VERBOSITY::OPTION::ON_EXIT);
	///@}
	
	static qarray<1> singlet (int N) {return qarray<1>{1};};
	static constexpr MODEL_FAMILY FAMILY = KONDO;
	
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
	static void set_operators (const std::vector<SpinBase<Symmetry_> > &B, const std::vector<FermionBase<Symmetry_> > &F, const vector<SUB_LATTICE> &G, const ParamHandler &P,
	                           PushType<SiteOperator<Symmetry_,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, 
	                           const BC boundary=BC::OPEN);
	
	// Mpo<Symmetry> Simp (SPINOP_LABEL Sa, size_t locx, size_t locy=0) const;
	// Mpo<Symmetry> Ssub (SPINOP_LABEL Sa, size_t locx, size_t locy=0) const;
	// Mpo<Symmetry> n (size_t locx, size_t locy=0) const;
	
	// Mpo<Symmetry> SimpSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	// Mpo<Symmetry> SsubSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	// Mpo<Symmetry> SimpSimp (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	
	static const std::map<string,std::any> defaults;
	static const map<string,any> sweep_defaults;
	
// protected:
	
// 	Mpo<Symmetry> make_corr (KONDO_SUBSYSTEM SUBSYS, string name1, string name2, size_t locx1, size_t locx2, size_t locy1, size_t locy2, 
// 	                         const OperatorType &Op1, const OperatorType &Op2, bool BOTH_HERMITIAN=false) const;
	
// 	vector<FermionBase<Symmetry> > F;
// 	vector<SpinBase<Symmetry> > B;
};

const std::map<string,std::any> KondoU0xSU2::defaults =
{
	{"t",1.}, {"tRung",0.}, {"tPrime",0.}, {"tPrimePrime",0.},
	{"J",1.}, {"Jz",0.}, {"U",0.}, 
	{"V",0.}, {"Vrung",0.},
	{"Bz",0.}, {"Bzsub",0.}, {"Kz",0.}, {"Bx",0.}, {"Bxsub",0.}, {"Kx",0.},
	{"Inext",0.}, {"Iprev",0.}, {"I3next",0.}, {"I3prev",0.}, {"I3loc",0.}, 
	{"D",2ul}, {"maxPower",1ul}, {"CYLINDER",false}, {"Ly",1ul}, {"LyF",1ul},
	{"SEMIOPEN_LEFT",false}, {"SEMIOPEN_RIGHT",false}, {"mfactor",1},
	{"REMOVE_DOUBLE",false}, {"REMOVE_EMPTY",false}, {"REMOVE_UP",false}, {"REMOVE_DN",false}, {"mfactor",1}, {"k",1}
};

const map<string,any> KondoU0xSU2::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.}, {"lim_alpha",11ul}, {"eps_svd",1e-7},
	{"Mincr_abs",50ul}, {"Mincr_per",2ul}, {"Mincr_rel",1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",30ul}, {"min_halfsweeps",10ul},
	{"Minit",1ul}, {"Qinit",1ul}, {"Dlimit",10000ul},
	{"tol_eigval",1e-6}, {"tol_state",1e-5},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST",DMRG::CONVTEST::VAR_2SITE}
};

KondoU0xSU2::
KondoU0xSU2 (const size_t &L, const vector<Param> &params, const BC &boundary, const DMRG::VERBOSITY::OPTION &VERB)
:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, boundary, VERB),
 KondoObservables(L,params,KondoU0xSU2::defaults),
 ParamReturner(KondoU0xSU2::sweep_defaults)
{
	ParamHandler P(params,defaults);
	size_t Lcell = P.size();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("LyF",l%Lcell);
		setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l);
	}
	//this->GOT_SEMIOPEN_LEFT  = P.get<bool>("SEMIOPEN_LEFT");
	//this->GOT_SEMIOPEN_RIGHT = P.get<bool>("SEMIOPEN_RIGHT");
	
	this->set_name("Kondo");
	
	PushType<SiteOperator<Symmetry,double>,double> pushlist;
	std::vector<std::vector<std::string>> labellist;
	set_operators(B, F, G, P, pushlist, labellist, boundary);
	
	this->construct_from_pushlist(pushlist, labellist, Lcell);
	this->finalize(PROP::COMPRESS, P.get<size_t>("maxPower"));
	
	this->precalc_TwoSiteData();
}

template<typename Symmetry_>
void KondoU0xSU2::
set_operators (const std::vector<SpinBase<Symmetry_> > &B, const std::vector<FermionBase<Symmetry_> > &F, const vector<SUB_LATTICE> &G, const ParamHandler &P,
               PushType<SiteOperator<Symmetry_,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = B.size();
	if(labellist.size() != N_sites) {labellist.resize(N_sites);}
	
	for (std::size_t loc=0; loc<N_sites; ++loc)
	{
		size_t lm1 = (loc==0)? N_sites-1 : loc-1;
		size_t lp1 = (loc+1)%N_sites;
		size_t lp2 = (loc+2)%N_sites;
		size_t lp3 = (loc+3)%N_sites;
		
		//auto Glm1 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,lm1)));
		//auto Glp1 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,lp1)));
		//auto Glp2 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,lp2)));
		//auto Glp3 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,lp3)));
		
		std::size_t Fprev_orbitals  = F[lm1].orbitals();
		std::size_t Forbitals       = F[loc].orbitals();
		std::size_t Fnext_orbitals  = F[lp1].orbitals();
		std::size_t Fnextn_orbitals = F[lp2].orbitals();
		std::size_t F3next_orbitals = F[lp3].orbitals();
		
		std::size_t Bprev_orbitals  = B[lm1].orbitals();
		std::size_t Borbitals       = B[loc].orbitals();
		std::size_t Bnext_orbitals  = B[lp1].orbitals();
		std::size_t Bnextn_orbitals = B[lp2].orbitals();
		std::size_t B3next_orbitals = B[lp3].orbitals();
		
		frac S = frac(B[loc].get_D()-1,2);
		stringstream Slabel;
		Slabel << "S=" << print_frac_nice(S);
		labellist[loc].push_back(Slabel.str());
		
		auto push_full = [&N_sites, &loc, &B, &F, &P, &pushlist, &labellist, &boundary] (string xxxFull, string label,
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
						if (FERMIONIC) {ops[i] = kroneckerProduct(B[(loc+i)%N_sites].Id(), F[(loc+i)%N_sites].sign());}
						else {ops[i] = kroneckerProduct(B[(loc+i)%N_sites].Id(), F[(loc+i)%N_sites].Id());}
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
			SiteOperatorQ<Symmetry_,Eigen::MatrixXd> cdagup_sign_local = kroneckerProduct(B[loc].Id(),(F[loc].cdag(UP,G[loc],0) * F[loc].sign()));
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > cup_ranges(N_sites);
			SiteOperatorQ<Symmetry_,Eigen::MatrixXd> cdagdn_sign_local = kroneckerProduct(B[loc].Id(),(F[loc].cdag(DN,G[loc],0) * F[loc].sign()));
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > cdn_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++)
			{
				auto Gi = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,i)));
				cup_ranges[i] = kroneckerProduct(B[loc].Id(),F[i].c(UP,Gi,0));
			}
			for (size_t i=0; i<N_sites; i++)
			{
				auto Gi = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,i)));
				cdn_ranges[i] = kroneckerProduct(B[loc].Id(),F[i].c(DN,Gi,0));
			}
			
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first {cdagup_sign_local, cdagdn_sign_local};
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {cup_ranges,cdn_ranges};
			push_full("tFull", "tᵢⱼ", first, last, {-std::sqrt(2.),- std::sqrt(2.)}, PROP::FERMIONIC);
		}
		
		// local terms
		
		// Kondo-J
		param1d J = P.fill_array1d<double>("J", "Jorb", Forbitals, loc%Lcell);
		labellist[loc].push_back(J.label);
		
		// Kondo-Jz
		param1d Jz = P.fill_array1d<double>("Jz", "Jzorb", Forbitals, loc%Lcell);
		labellist[loc].push_back(Jz.label);
		
		// t⟂
		param2d tPerp = P.fill_array2d<double>("tRung", "t", "tPerp", Forbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		labellist[loc].push_back(tPerp.label);
		
		// Hubbard-U
		param1d U = P.fill_array1d<double>("U", "Uorb", Forbitals, loc%Lcell);
		labellist[loc].push_back(U.label);
		
		// Bx substrate
		param1d Bxsub = P.fill_array1d<double>("Bxsub", "Bxsuborb", Forbitals, loc%Lcell);
		labellist[loc].push_back(Bxsub.label);
		
		// Bx impurities
		param1d Bx = P.fill_array1d<double>("Bx","Bxorb", Borbitals, loc%Lcell);
		labellist[loc].push_back(Bx.label);
		
		// Kx anisotropy
		param1d Kx = P.fill_array1d<double>("Kx","Kxorb", Borbitals, loc%Lcell);
		labellist[loc].push_back(Kx.label);
		
		// Bz substrate
		param1d Bzsub = P.fill_array1d<double>("Bzsub", "Bzsuborb", Forbitals, loc%Lcell);
		labellist[loc].push_back(Bzsub.label);
		
		// Bz impurities
		param1d Bz = P.fill_array1d<double>("Bz", "Bzorb", Borbitals, loc%Lcell);
		labellist[loc].push_back(Bz.label);
		
		// Kz anisotropy
		param1d Kz = P.fill_array1d<double>("Kz","Kzorb", Borbitals, loc%Lcell);
		labellist[loc].push_back(Kz.label);
		
		ArrayXXd muPerp  = B[loc].ZeroHopping();
		ArrayXXd nuPerp  = B[loc].ZeroHopping();
		ArrayXXd Jxyperp = B[loc].ZeroHopping();
		ArrayXXd Jzperp  = B[loc].ZeroHopping();
		ArrayXXd DyPerp  = B[loc].ZeroHopping();
		
		//set Heisenberg part of Kondo Hamiltonian
		auto KondoHamiltonian = kroneckerProduct(B[loc].HeisenbergHamiltonian(Jxyperp,Jzperp,Bz.a,Bx.a,muPerp,nuPerp,Kz.a,Kx.a,DyPerp), F[loc].Id());
		
		ArrayXXd Vperp      = F[loc].ZeroHopping();
		ArrayXXd Jxysubperp = F[loc].ZeroHopping();
		ArrayXXd Jzsubperp  = F[loc].ZeroHopping();
		
		//set Hubbard part of Kondo Hamiltonian
		KondoHamiltonian += kroneckerProduct(B[loc].Id(), F[loc].HubbardHamiltonian(U.a,tPerp.a,Vperp,Jzsubperp,Jxysubperp,Bzsub.a,Bxsub.a));
		
		//set Kondo part of Hamiltonian
		for (int alfa=0; alfa<Forbitals; ++alfa)
		{
			if (J(alfa) != 0.)
			{
				assert(Borbitals == Forbitals and "Can only do a Kondo ladder with the same amount of spins and fermionic orbitals in y-direction!");
				KondoHamiltonian += 0.5*J(alfa) * kroneckerProduct(B[loc].Scomp(SP,alfa), F[loc].Sm(alfa));
				KondoHamiltonian += 0.5*J(alfa) * kroneckerProduct(B[loc].Scomp(SM,alfa), F[loc].Sp(alfa));
				KondoHamiltonian +=     J(alfa) * kroneckerProduct(B[loc].Scomp(SZ,alfa), F[loc].Sz(alfa));
			}
			if (Jz(alfa) != 0.)
			{
				KondoHamiltonian += Jz(alfa) * kroneckerProduct(B[loc].Scomp(SZ,alfa), F[loc].Sz(alfa));
			}
		}
		pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(KondoHamiltonian), 1.));
		
		// NN terms
		
		// V∥
		param2d Vpara = P.fill_array2d<double>("V", "Vpara", {Forbitals, Fnext_orbitals}, loc%Lcell);
		labellist[loc].push_back(Vpara.label);
		
		// t∥
		if (!P.HAS("tFull"))
		{
			param2d tPara = P.fill_array2d<double>("t", "tPara", {Forbitals, Fnext_orbitals}, loc%Lcell);
			labellist[loc].push_back(tPara.label);
			
			if (loc < N_sites-1 or !static_cast<bool>(boundary))
			{
				for (int alfa=0; alfa<Forbitals;      ++alfa)
				for (int beta=0; beta<Fnext_orbitals; ++beta)
				{
					auto PsiDagUp_loc = kroneckerProduct(B[loc].Id(), F[loc].cdag(UP,G[loc],alfa));
					auto PsiDagDn_loc = kroneckerProduct(B[loc].Id(), F[loc].cdag(DN,G[loc],alfa));
					auto Sign_loc     = kroneckerProduct(B[loc].Id(), F[loc].sign());
					auto PsiUp_lp1    = kroneckerProduct(B[lp1].Id(), F[lp1].c(UP,G[lp1],beta));
					auto PsiDn_lp1    = kroneckerProduct(B[lp1].Id(), F[lp1].c(DN,G[lp1],beta));
					
					auto Otmp_loc = PsiDagUp_loc * Sign_loc;
					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(Otmp_loc, PsiUp_lp1), -tPara(alfa,beta) * sqrt(2.)) );
										
					//c†DNcDN
					Otmp_loc = PsiDagDn_loc * Sign_loc;
					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(Otmp_loc, PsiDn_lp1), -tPara(alfa,beta) * sqrt(2.)) );
				}
			}
		}
		
		// NN spin exchange terms
		
		param2d InextPara = P.fill_array2d<double>("Inext", "InextPara", {Borbitals, Fnext_orbitals}, loc%Lcell);
		labellist[loc].push_back(InextPara.label);
		
		if (loc < N_sites-1 or !static_cast<bool>(boundary))
		{
			for (std::size_t alfa=0; alfa<Borbitals;      ++alfa)
			for (std::size_t beta=0; beta<Fnext_orbitals; ++beta)
			{
				pushlist.push_back(std::make_tuple(loc,
												   Mpo<Symmetry_,double>::get_N_site_interaction(kroneckerProduct(B[loc].Scomp(SP,alfa), F[loc].Id()),
																								 kroneckerProduct(B[lp1].Id(), F[lp1].Sm(beta))),
												   0.5*InextPara(alfa,beta)));
				pushlist.push_back(std::make_tuple(loc,
												   Mpo<Symmetry_,double>::get_N_site_interaction(kroneckerProduct(B[loc].Scomp(SM,alfa), F[loc].Id()),
																								 kroneckerProduct(B[lp1].Id(), F[lp1].Sp(beta))),
												   0.5*InextPara(alfa,beta)));
				pushlist.push_back(std::make_tuple(loc,
												   Mpo<Symmetry_,double>::get_N_site_interaction(kroneckerProduct(B[loc].Scomp(SZ,alfa), F[loc].Id()),
																								 kroneckerProduct(B[lp1].Id(), F[lp1].Sz(beta))),
												   InextPara(alfa,beta)));
			}
		}
		
		param2d IprevPara = P.fill_array2d<double>("Iprev", "IprevPara", {Fprev_orbitals, Borbitals}, loc%Lcell);
		labellist[loc].push_back(IprevPara.label);
		
		if (lm1 < N_sites-1 or !static_cast<bool>(boundary))
		{
			for (std::size_t alfa=0; alfa<Fprev_orbitals;  ++alfa)
			for (std::size_t beta=0; beta<Borbitals;       ++beta)
			{
				pushlist.push_back(std::make_tuple(lm1,
												   Mpo<Symmetry_,double>::get_N_site_interaction(kroneckerProduct(B[lm1].Id(), F[lm1].Sm(alfa)),
																								 kroneckerProduct(B[loc].Scomp(SP,beta), F[lp1].Id())),
												   0.5*IprevPara(alfa,beta)));
				pushlist.push_back(std::make_tuple(lm1,
												   Mpo<Symmetry_,double>::get_N_site_interaction(kroneckerProduct(B[lm1].Id(), F[lm1].Sp(alfa)),
																								 kroneckerProduct(B[loc].Scomp(SM,beta), F[lp1].Id())),
												   0.5*IprevPara(alfa,beta)));
				pushlist.push_back(std::make_tuple(lm1,
												   Mpo<Symmetry_,double>::get_N_site_interaction(kroneckerProduct(B[lm1].Id(), F[lm1].Sz(alfa)),
																								 kroneckerProduct(B[loc].Scomp(SZ,beta), F[lp1].Id())),
												   IprevPara(alfa,beta)));
			}
		}
		
		// NN 3-orbital spin exchange terms
		
		param2d I3nextPara = P.fill_array2d<double>("I3next", "I3nextPara", {Forbitals, Fnext_orbitals}, loc%Lcell);
		labellist[loc].push_back(I3nextPara.label);
		
		if (loc < N_sites-1 or !static_cast<bool>(boundary))
		{
			for (std::size_t alfa=0; alfa<Forbitals;      ++alfa)
			for (std::size_t beta=0; beta<Fnext_orbitals; ++beta)
			{
				assert(Borbitals == 1);
				
				auto Sm_PsiDagUp_loc = kroneckerProduct(B[loc].Scomp(SM,0), F[loc].cdag(UP,G[loc],alfa));
				auto Sp_PsiDagDn_loc = kroneckerProduct(B[loc].Scomp(SP,0), F[loc].cdag(DN,G[loc],alfa));
				auto Sign_loc        = kroneckerProduct(B[loc].Id(), F[loc].sign());
				auto PsiUp_lp1       = kroneckerProduct(B[lp1].Id(), F[lp1].c(UP,G[lp1],beta));
				auto PsiDn_lp1       = kroneckerProduct(B[lp1].Id(), F[lp1].c(DN,G[lp1],beta));
				
				auto Otmp_loc = Sm_PsiDagUp_loc * Sign_loc;
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(Otmp_loc, PsiDn_lp1),0.5*sqrt(2.)*I3nextPara(alfa,beta)));
							   				
				Otmp_loc = Sp_PsiDagDn_loc * Sign_loc;
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(Otmp_loc, PsiUp_lp1),0.5*sqrt(2.)*I3nextPara(alfa,beta)));

				pushlist.push_back(std::make_tuple(loc,
												   Mpo<Symmetry_,double>::get_N_site_interaction(kroneckerProduct(B[loc].Scomp(SZ,0), F[loc].Id()),
																								 kroneckerProduct(B[lp1].Id(), F[lp1].Sz(beta))),
												   I3nextPara(alfa,beta)));
			}
		}
		
		param2d I3prevPara = P.fill_array2d<double>("I3prev", "I3prevPara", {Fprev_orbitals, Forbitals}, loc%Lcell);
		labellist[loc].push_back(I3prevPara.label);
		
		if (lm1 < N_sites-1 or !static_cast<bool>(boundary))
		{
			for (std::size_t alfa=0; alfa<Fprev_orbitals;  ++alfa)
			for (std::size_t beta=0; beta<Forbitals;       ++beta)
			{
				assert(Borbitals == 1);
				
				auto Sm_PsiDagUp_lm1 = kroneckerProduct(B[lm1].Scomp(SM,0), F[lm1].cdag(UP,G[lm1],alfa));
				auto Sp_PsiDagDn_lm1 = kroneckerProduct(B[lm1].Scomp(SP,0), F[lm1].cdag(DN,G[lm1],alfa));
				auto Sign_lm1        = kroneckerProduct(B[lm1].Id(), F[lm1].sign());
				auto PsiUp_loc       = kroneckerProduct(B[loc].Id(), F[loc].c(UP,G[loc],beta));
				auto PsiDn_loc       = kroneckerProduct(B[loc].Id(), F[loc].c(DN,G[loc],beta));
				
				auto Otmp_lm1 = Sm_PsiDagUp_lm1 * Sign_lm1;
				pushlist.push_back(std::make_tuple(lm1, Mpo<Symmetry_,double>::get_N_site_interaction(Otmp_lm1, PsiDn_loc),0.5*sqrt(2.)*I3prevPara(alfa,beta)));
				
				Otmp_lm1 = Sp_PsiDagDn_lm1 * Sign_lm1;
				pushlist.push_back(std::make_tuple(lm1, Mpo<Symmetry_,double>::get_N_site_interaction(Otmp_lm1, PsiUp_loc),0.5*sqrt(2.)*I3prevPara(alfa,beta)));

				pushlist.push_back(std::make_tuple(lm1,
												   Mpo<Symmetry_,double>::get_N_site_interaction(kroneckerProduct(B[lm1].Scomp(SZ,0), F[lm1].Id()),
																								 kroneckerProduct(B[loc].Id(), F[loc].Sz(beta))),
												   I3prevPara(alfa,beta)));
			}
		}
		
		// tPrime
		if (!P.HAS("tFull") and P.HAS("tPrime",loc%Lcell))
		{			
			param2d tPrime = P.fill_array2d<double>("tPrime", "tPrime_array", {Forbitals, F3next_orbitals}, loc%Lcell);
			labellist[loc].push_back(tPrime.label);
			
			if (loc < N_sites-2 or !static_cast<bool>(boundary))
			{
				auto Sign_loc     = kroneckerProduct(B[loc].Id(), F[loc].sign());
				auto Sign_lp1     = kroneckerProduct(B[lp1].Id(), F[lp1].sign());
								
				for (std::size_t alfa=0; alfa<Forbitals;       ++alfa)
				for (std::size_t beta=0; beta<F3next_orbitals; ++beta)
				{
					auto PsiDagUp_loc = kroneckerProduct(B[loc].Id(), F[loc].cdag(UP,G[loc],alfa));
					auto PsiDagDn_loc = kroneckerProduct(B[loc].Id(), F[loc].cdag(DN,G[loc],alfa));
					auto Sign_loc     = kroneckerProduct(B[loc].Id(), F[loc].sign());
					auto PsiUp_lp2    = kroneckerProduct(B[lp1].Id(), F[lp1].c(UP,G[lp2],beta));
					auto PsiDn_lp2    = kroneckerProduct(B[lp1].Id(), F[lp1].c(DN,G[lp2],beta));
					
					auto Otmp_loc = PsiDagUp_loc * Sign_loc;
					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(Otmp_loc, Sign_lp1, PsiUp_lp2), -tPrime(alfa,beta) * sqrt(2.)) );
					
					//c†DNcDN
					Otmp_loc = PsiDagDn_loc * Sign_loc;
					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(Otmp_loc, Sign_lp1, PsiDn_lp2), -tPrime(alfa,beta) * sqrt(2.)) );
				}
			}
		}
		
		// tPrimePrime
		if (!P.HAS("tFull") and P.HAS("tPrimePrime",loc%Lcell))
		{
			param2d tPrimePrime = P.fill_array2d<double>("tPrimePrime", "tPrimePrime_array", {Forbitals, F3next_orbitals}, loc%Lcell);
			labellist[loc].push_back(tPrimePrime.label);
			
			if (loc < N_sites-3 or !static_cast<bool>(boundary))
			{
				auto Sign_loc     = kroneckerProduct(B[loc].Id(), F[loc].sign());
				auto Sign_lp1     = kroneckerProduct(B[lp1].Id(), F[lp1].sign());
				auto Sign_lp2     = kroneckerProduct(B[lp2].Id(), F[lp2].sign());
								
				for (std::size_t alfa=0; alfa<Forbitals;       ++alfa)
				for (std::size_t beta=0; beta<F3next_orbitals; ++beta)
				{
					auto PsiDagUp_loc = kroneckerProduct(B[loc].Id(), F[loc].cdag(UP,G[loc],alfa));
					auto PsiDagDn_loc = kroneckerProduct(B[loc].Id(), F[loc].cdag(DN,G[loc],alfa));
					auto Sign_loc     = kroneckerProduct(B[loc].Id(), F[loc].sign());
					auto PsiUp_lp3    = kroneckerProduct(B[lp1].Id(), F[lp1].c(UP,G[lp3],beta));
					auto PsiDn_lp3    = kroneckerProduct(B[lp1].Id(), F[lp1].c(DN,G[lp3],beta));
					
					auto Otmp_loc = PsiDagUp_loc * Sign_loc;
					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(Otmp_loc, Sign_lp1, Sign_lp2, PsiDn_lp3), -tPrimePrime(alfa,beta) * sqrt(2.)));
										
					//c†DNcDN
					Otmp_loc = PsiDagDn_loc * Sign_loc;
					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(Otmp_loc, Sign_lp1, Sign_lp2, PsiUp_lp3), -tPrimePrime(alfa,beta) * sqrt(2.)));
				}
			}
		}
	}
}

//HamiltonianTermsXd<Sym::SU2<Sym::ChargeSU2> > KondoU0xSU2::
//set_operators (const vector<SpinBase<Symmetry> > &B, const vector<FermionBase<Symmetry> > &F, const ParamHandler &P, size_t loc)
//{
//	HamiltonianTermsXd<Symmetry> Terms;
//	
//	frac S = frac(B[loc].get_D()-1,2);
//	stringstream Slabel;
//	Slabel << "S=" << print_frac_nice(S);
//	Terms.info.push_back(Slabel.str());
//	
//	auto save_label = [&Terms] (string label)
//	{
//		if (label!="") {Terms.info.push_back(label);}
//	};
//	
//	size_t lp1 = (loc+1)%F.size();
//	
//	// NN terms
//	
//	auto [t,tPara,tlabel] = P.fill_array2d<double>("t","tPara",{{F[loc].orbitals(),F[lp1].orbitals()}},loc);
//	save_label(tlabel);
//	
//	auto [V,Vpara,Vlabel] = P.fill_array2d<double>("V","Vpara",{{F[loc].orbitals(),F[lp1].orbitals()}},loc);
//	save_label(Vlabel);
//	
//	for (int i=0; i<F[loc].orbitals(); ++i)
//	for (int j=0; j<F[lp1].orbitals(); ++j)
//	{
//		if (tPara(i,j) != 0.)
//		{
//			//-------------------------------------------------------------------------------------------------------------------------------------//
//			// Terms.tight.push_back(make_tuple(-tPara(i,j),
//			//                                  kroneckerProduct(B.Id(), F.cdag(UP,i) * F.sign()),
//			//                                  kroneckerProduct(B.Id(), F.c(UP,j))));
//			// Terms.tight.push_back(make_tuple(-tPara(i,j),
//			//                                  kroneckerProduct(B.Id(), F.cdag(DN,i) * F.sign()),
//			//                                  kroneckerProduct(B.Id(), F.c(DN,j))));
//			// Terms.tight.push_back(make_tuple(-tPara(i,j),
//			//                                  kroneckerProduct(B.Id(), -1.*F.c(UP,i) * F.sign()),
//			//                                  kroneckerProduct(B.Id(), F.cdag(UP,j))));
//			// Terms.tight.push_back(make_tuple(-tPara(i,j),
//			//                                  kroneckerProduct(B.Id(), -1.*F.c(DN,i) * F.sign()),
//			//                                  kroneckerProduct(B.Id(), F.cdag(DN,j))));
//			
//			// Mout += -t*std::sqrt(2.)*(Operator::prod(psidag(UP,i),psi(UP,i+1),{1})+Operator::prod(psidag(DN,i),psi(DN,i+1),{1}));
//			//-------------------------------------------------------------------------------------------------------------------------------------//
//			
//			//c†UPcUP
//			
//			auto Otmp = OperatorType::prod(OperatorType::outerprod(B[loc].Id().structured(), F[loc].psidag(UP,i), {2}),
//										   OperatorType::outerprod(B[loc].Id().structured(), F[loc].sign()      , {1}),
//										   {2});
//			Terms.tight.push_back(make_tuple(-tPara(i,j)*sqrt(2.),
//											 Otmp.plain<double>(),
//											 OperatorType::outerprod(B[loc].Id().structured(), F[loc].psi(UP,i), {2}).plain<double>()));
//			
//			//c†DNcDN
//			Otmp = OperatorType::prod(OperatorType::outerprod(B[loc].Id().structured(), F[loc].psidag(DN,i), {2}),
//									  OperatorType::outerprod(B[loc].Id().structured(), F[loc].sign()      , {1}),
//									  {2});
//			Terms.tight.push_back(make_tuple(-tPara(i,j)*sqrt(2.),
//											 Otmp.plain<double>(),
//											 OperatorType::outerprod(B[loc].Id().structured(), F[loc].psi(DN,i), {2}).plain<double>()));
//			
//			//-cUPc†UP
//			// Otmp = OperatorType::prod(OperatorType::outerprod(B.Id().structured(),F.psi(UP,i),{2}),
//			// 						  OperatorType::outerprod(B.Id().structured(),F.sign()   ,{1}),
//			// 						  {2});
//			// Terms.tight.push_back(make_tuple(tPara(i,j)*sqrt(2.),
//			// 								 Otmp.plain<double>(),
//			// 								 OperatorType::outerprod(B.Id().structured(),F.psidag(UP,j),{2}).plain<double>()));
//			
//			//-cDNc†DN
//			// Otmp = OperatorType::prod(OperatorType::outerprod(B.Id().structured(),F.psi(DN,i),{2}),
//			// 						  OperatorType::outerprod(B.Id().structured(),F.sign()   ,{1}),
//			// 						  {2});
//			// Terms.tight.push_back(make_tuple(tPara(i,j)*sqrt(2.),
//			// 								 Otmp.plain<double>(),
//			// 								 OperatorType::outerprod(B.Id().structured(),F.psidag(DN,j),{2}).plain<double>()));
//			
//			//-------------------------------------------------------------------------------------------------------------------------------------//
//		}
//	}
//	
//	// local terms
//	
//	// t⟂
//	auto [tRung,tPerp,tPerplabel] = P.fill_array2d<double>("tRung","t","tPerp",F[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
//	save_label(tPerplabel);
//	
//	// Hubbard U
//	auto [U,Uorb,Ulabel] = P.fill_array1d<double>("U","Uorb",F[loc].orbitals(),loc);
//	save_label(Ulabel);

//	// Bx substrate
//	auto [Bxsub,Bxsuborb,Bxsublabel] = P.fill_array1d<double>("Bxsub","Bxsuborb",F[loc].orbitals(),loc);
//	save_label(Bxsublabel);
//	
//	// Bx impurities
//	auto [Bx,Bxorb,Bxlabel] = P.fill_array1d<double>("Bx","Bxorb",F[loc].orbitals(),loc);
//	save_label(Bxlabel);
//	
//	// Kx anisotropy
//	auto [Kx,Kxorb,Kxlabel] = P.fill_array1d<double>("Kx","Kxorb",B[loc].orbitals(),loc);
//	save_label(Kxlabel);
//	
//	// Bz substrate
//	auto [Bzsub,Bzsuborb,Bzsublabel] = P.fill_array1d<double>("Bzsub","Bzsuborb",F[loc].orbitals(),loc);
//	save_label(Bzsublabel);
//	
//	// Bz impurities
//	auto [Bz,Bzorb,Bzlabel] = P.fill_array1d<double>("Bz","Bzorb",F[loc].orbitals(),loc);
//	save_label(Bzlabel);
//	
//	// Kz anisotropy
//	auto [Kz,Kzorb,Kzlabel] = P.fill_array1d<double>("Kz","Kzorb",B[loc].orbitals(),loc);
//	save_label(Kzlabel);
//	
//	// OperatorType KondoHamiltonian({1},B[loc].get_structured_basis().combine(F[loc].get_basis()));
//	
//	ArrayXXd Jxyperp = B[loc].ZeroHopping();
//	ArrayXXd Jzperp  = B[loc].ZeroHopping();
//	ArrayXXd Dyperp  = B[loc].ZeroHopping();
//	
//	//set Heisenberg part of Kondo Hamiltonian
//	auto KondoHamiltonian = OperatorType::outerprod(B[loc].HeisenbergHamiltonian(Jxyperp,Jzperp,Bzorb,Bxorb,Kzorb,Kxorb,Dyperp).structured(),
//	                                                F[loc].Id(),
//	                                                {1});
//	
//	ArrayXXd Vperp      = F[loc].ZeroHopping();
//	ArrayXXd Jxysubperp = F[loc].ZeroHopping();
//	ArrayXXd Jzsubperp  = F[loc].ZeroHopping();
//	
//	//set Hubbard part of Kondo Hamiltonian
//	KondoHamiltonian += OperatorType::outerprod(B[loc].Id().structured(),
//	                                            F[loc].HubbardHamiltonian(Uorb,tPerp,Vperp,Jxysubperp,Jzsubperp,Bzsuborb,Bxsuborb),
//	                                            {1});
//	
//	//set Heisenberg part of Hamiltonian
////	KondoHamiltonian += OperatorType::outerprod(B.HeisenbergHamiltonian(0.,P.get<bool>("CYLINDER")),F[loc].Id(),{1,0});
//	
//	// Kondo-J
//	auto [J,Jorb,Jlabel] = P.fill_array1d<double>("J","Jorb",F[loc].orbitals(),loc);
//	save_label(Jlabel);
//	
//	//set interaction part of Hamiltonian.
//	for (int i=0; i<F[loc].orbitals(); ++i)
//	{
//		if (Jorb(i) != 0.)
//		{
//			KondoHamiltonian +=     Jorb(i) * OperatorType::outerprod(B[loc].Scomp(SZ,i).structured(), F[loc].Sz(i), {1});
//			KondoHamiltonian += 0.5*Jorb(i) * OperatorType::outerprod(B[loc].Scomp(SP,i).structured(), F[loc].Sm(i), {1});
//			KondoHamiltonian += 0.5*Jorb(i) * OperatorType::outerprod(B[loc].Scomp(SM,i).structured(), F[loc].Sp(i), {1});
//		}
//	}
//	
//	Terms.name = "Kondo U(0)⊗SU(2)";
//	Terms.local.push_back(make_tuple(1.,KondoHamiltonian.plain<double>()));
//	
//	return Terms;
//}

// Mpo<Sym::SU2<Sym::ChargeSU2> > KondoU0xSU2::
// Simp (SPINOP_LABEL Sa, size_t locx, size_t locy) const
// {
// 	assert(locx < this->N_sites);
// 	std::stringstream ss;
	
// 	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
// 	for (std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis((B[l].get_structured_basis().combine(F[l].get_basis())).qloc(),l); }
	
// 	auto Sop = OperatorType::outerprod(B[locx].Scomp(Sa,locy).structured(), F[locx].Id(), {1});
	
// 	Mout.setLocal(locx, Sop.plain<double>());
// 	return Mout;
// }

// Mpo<Sym::SU2<Sym::ChargeSU2> > KondoU0xSU2::
// Ssub (SPINOP_LABEL Sa, size_t locx, size_t locy) const
// {
// 	assert(locx < this->N_sites);
// 	std::stringstream ss;
	
// 	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
// 	for (std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis((B[l].get_structured_basis().combine(F[l].get_basis())).qloc(),l); }
	
// 	auto Sop = OperatorType::outerprod(B[locx].Id().structured(), F[locx].Scomp(Sa,locy), {1});
	
// 	Mout.setLocal(locx, Sop.plain<double>());
// 	return Mout;
// }

// Mpo<Sym::SU2<Sym::ChargeSU2> > KondoU0xSU2::
// n (size_t locx, size_t locy) const
// {
// 	assert(locx < this->N_sites);
// 	std::stringstream ss;
	
// 	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
// 	for (std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis((B[l].get_structured_basis().combine(F[l].get_basis())).qloc(),l); }
	
// 	auto nop = OperatorType::outerprod(B[locx].Id().structured(), F[locx].n(locy), {1});
	
// 	Mout.setLocal(locx, nop.plain<double>());
// 	return Mout;
// }

/*Mpo<Sym::SU2<Sym::ChargeSU2> > KondoU0xSU2::*/
/*SimpSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const*/
/*{*/
/*	assert(locx1 < this->N_sites and locx2 < this->N_sites);*/
/*	std::stringstream ss;*/
/*	*/
/*	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());*/
/*	for (std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis((B[l].get_structured_basis().combine(F[l].get_basis())).qloc(),l); }*/
/*	*/
/*	auto Sop1 = OperatorType::outerprod(B[locx1].Scomp(SOP1,locy1).structured(), F[locx2].Id(), {1});*/
/*	auto Sop2 = OperatorType::outerprod(B[locx1].Id().structured(), F[locx2].Scomp(SOP2,locy2), {1});*/
/*	*/
/*	Mout.setLocal({locx1,locx2}, {Sop1.plain<double>(),Sop2.plain<double>()});*/
/*	return Mout;*/
/*}*/

// Mpo<Sym::SU2<Sym::ChargeSU2> > KondoU0xSU2::
// make_corr (KONDO_SUBSYSTEM SUBSYS, string name1, string name2, 
//            size_t locx1, size_t locx2, size_t locy1, size_t locy2, 
//            const OperatorType &Op1, const OperatorType &Op2, 
//            bool BOTH_HERMITIAN) const
// {
// 	assert(locx1<F.size() and locx2<F.size() and locy1<F[locx1].dim() and locy2<F[locx2].dim());
// 	stringstream ss;
// 	ss << name1 << "(" << locx1 << "," << locy1 << ")"
// 	   << name2 << "(" << locx2 << "," << locy2 << ")";
	
// 	bool HERMITIAN = (BOTH_HERMITIAN and locx1==locx2 and locy1==locy2)? true:false;
	
// 	OperatorType Op1Ext;
// 	OperatorType Op2Ext;
	
// 	Mpo<Symmetry> Mout(F.size(), Symmetry::qvacuum(), ss.str(), HERMITIAN);
// 	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis((B[l].get_structured_basis().combine(F[l].get_basis())).qloc(),l);}
	
// 	if (SUBSYS == SUB)
// 	{
// 		Op1Ext = OperatorType::outerprod(B[locx1].Id().structured(), Op1, {1});
// 		Op2Ext = OperatorType::outerprod(B[locx2].Id().structured(), Op2, {1});
// 	}
// 	else if (SUBSYS == IMP)
// 	{
// 		Op1Ext = OperatorType::outerprod(Op1, F[locx1].Id(), {1});
// 		Op2Ext = OperatorType::outerprod(Op2, F[locx2].Id(), {1});
// 	}
// 	else if (SUBSYS == IMPSUB and locx1 != locx2)
// 	{
// 		Op1Ext = OperatorType::outerprod(Op1, F[locx1].Id(), {1});
// 		Op2Ext = OperatorType::outerprod(B[locx2].Id().structured(), Op2, {1});
// 	}
// 	else if (SUBSYS == IMPSUB and locx1 == locx2)
// 	{
// 		OperatorType OpExt = OperatorType::outerprod(Op1, Op2, {1});
		
// 		Mout.setLocal(locx1, OpExt.plain<double>());
// 		return Mout;
// 	}
	
// 	Mout.setLocal({locx1,locx2}, {Op1Ext.plain<double>(),Op2Ext.plain<double>()});
// 	return Mout;
// }

// Mpo<Sym::SU2<Sym::ChargeSU2> > KondoU0xSU2::
// SimpSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
// {
// 	stringstream ss1; ss1 << SOP1 << "imp";
// 	stringstream ss2; ss2 << SOP2 << "sub";
	
// 	return make_corr(IMPSUB, ss1.str(),ss2.str(), locx1,locx2,locy1,locy2, B[locx1].Scomp(SOP1,locy1).structured(), F[locx2].Scomp(SOP2,locy2));
// }

// Mpo<Sym::SU2<Sym::ChargeSU2> > KondoU0xSU2::
// SimpSimp (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
// {
// 	stringstream ss1; ss1 << SOP1 << "imp";
// 	stringstream ss2; ss2 << SOP2 << "imp";
	
// 	return make_corr(IMP, ss1.str(),ss2.str(), locx1,locx2,locy1,locy2, B[locx1].Scomp(SOP1,locy1).structured(), B[locx2].Scomp(SOP2,locy2).structured());
// }

// Mpo<Sym::SU2<Sym::ChargeSU2> > KondoU0xSU2::
// SsubSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
// {
// 	stringstream ss1; ss1 << SOP1 << "sub";
// 	stringstream ss2; ss2 << SOP2 << "sub";
	
// 	return make_corr(SUB, ss1.str(),ss2.str(), locx1,locx2,locy1,locy2, F[locx1].Scomp(SOP1,locy1), F[locx2].Scomp(SOP2,locy2));
// }

} //end namespace VMPS

#endif
