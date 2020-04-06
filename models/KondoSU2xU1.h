#ifndef KONDOMODELSU2XU1_H_
#define KONDOMODELSU2XU1_H_

//include "ParamHandler.h" // from HELPERS
#include "symmetry/S1xS2.h"
#include "symmetry/SU2.h"
#include "symmetry/U1.h"
#include "bases/SpinBase.h"
#include "bases/FermionBase.h"
//include "Mpo.h"
//include "models/HubbardSU2xU1.h"
#include "models/KondoObservables.h"
#include "ParamReturner.h"

#include "Geometry2D.h" // from TOOLS

namespace VMPS
{
/** 
 * \class KondoSU2xU1
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
 * \note The default variable settings can be seen in \p KondoSU2xU1::defaults.
 * \note Take use of the Spin SU(2) symmetry and U(1) charge symmetry.
 * \note If the nnn-hopping is positive, the ground state energy is lowered.
 * \warning \f$J<0\f$ is antiferromagnetic
 */
class KondoSU2xU1 : public Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,double>,
					public KondoObservables<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > >,
					public ParamReturner
{
public:
	typedef Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > Symmetry;
	MAKE_TYPEDEFS(KondoSU2xU1)
	
private:
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	
public:
	
	///@{
	KondoSU2xU1 (): Mpo(), KondoObservables(), ParamReturner(KondoSU2xU1::sweep_defaults) {};
	KondoSU2xU1 (const size_t &L, const vector<Param> &params, const BC &boundary=BC::OPEN);
	///@}
	
	static qarray<2> singlet (int N) {return qarray<2>{1,N};};
	static qarray<2> polaron (int L, int N=0) {return qarray<2>{L-N+1,N};};
	
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
    static void set_operators (const std::vector<SpinBase<Symmetry_> > &B, const std::vector<FermionBase<Symmetry_> > &F, const ParamHandler &P,
							   PushType<SiteOperator<Symmetry_,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary=BC::OPEN);
	
	/**Validates whether a given \p qnum is a valid combination of \p N and \p M for the given model.
	\returns \p true if valid, \p false if not*/
	bool validate (qType qnum) const;
	
	///@{
	// Mpo<Symmetry> c (size_t locx, size_t locy=0, double factor=1.);
	// Mpo<Symmetry> cdag (size_t locx, size_t locy=0, double factor=sqrt(2.));
	// Mpo<Symmetry> cc (size_t locx, size_t locy=0);
	// Mpo<Symmetry> cdagcdag (size_t locx, size_t locy=0);
	// ///@}

	
	// ///@{
	// Mpo<Symmetry> n (size_t locx, size_t locy=0);
	// Mpo<Symmetry> nh (size_t locx, size_t locy=0);
	// Mpo<Symmetry> ns (size_t locx, size_t locy=0);
	// Mpo<Symmetry> d (size_t locx, size_t locy=0);
	// Mpo<Symmetry> cdagc (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	// Mpo<Symmetry> ccdag (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	// Mpo<Symmetry> dh_excitation (size_t locx);
	// ///@}
	
	// ///@{
	// Mpo<Symmetry> nn (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	//*\warning not implemented
//	Mpo<Symmetry> cdagcdagcc (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	///@}
	
	///@{
	// Mpo<Symmetry> Simp (size_t locx, size_t locy=0, double factor=1.);
	// Mpo<Symmetry> Simpdag (size_t locx, size_t locy=0, double factor=sqrt(3.));
	// Mpo<Symmetry> Ssub (size_t locx, size_t locy=0, double factor=1.);
	// Mpo<Symmetry> Ssubdag (size_t locx, size_t locy=0, double factor=sqrt(3.));
	// Mpo<Symmetry,complex<double> > Simp_ky    (vector<complex<double> > phases) const;
	// Mpo<Symmetry,complex<double> > Simpdag_ky (vector<complex<double> > phases, double factor=sqrt(3.)) const;
	// // for compatibility:
	// Mpo<Symmetry> S (size_t locx, size_t locy=0, double factor=1.) {return Simp(locx,locy,factor);};
	// Mpo<Symmetry> Sdag (size_t locx, size_t locy=0, double factor=sqrt(3.)) {return Simpdag(locx,locy,factor);};
	// ///@}
	
	// ///@{
	// Mpo<Symmetry> SimpSimp (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	// Mpo<Symmetry> SsubSsub (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	// Mpo<Symmetry> SimpSsub (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	///@}
	
//	///@{ \warning not implemented
//	Mpo<Symmetry> SimpSsubSimpSimp (size_t locx1, size_t locx2,
//									 size_t loc3x, size_t loc4x,
//									 size_t locy1=0, size_t locy2=0, size_t loc3y=0, size_t loc4y=0);
//	Mpo<Symmetry> SimpSsubSimpSsub (size_t locx1, size_t locx2,
//									 size_t loc3x, size_t loc4x,
//									 size_t locy1=0, size_t locy2=0, size_t loc3y=0, size_t loc4y=0);
//	///@}
	
	static const map<string,any> defaults;
	static const map<string,any> sweep_defaults;	
};

const map<string,any> KondoSU2xU1::defaults =
{
	{"t",1.}, {"tPrime",0.}, {"tRung",0.},
	{"J",1.}, {"Jdir",0.}, 
	{"U",0.}, {"Uph",0.},
	{"V",0.}, {"Vrung",0.}, 
	{"mu",0.}, {"t0",0.},
	{"Inext",0.}, {"Iprev",0.}, {"I3next",0.}, {"I3prev",0.}, {"I3loc",0.}, 
	{"D",2ul}, {"maxPower",2ul}, {"CYLINDER",false}, {"Ly",1ul}, {"LyF",1ul}
};

const map<string,any> VMPS::KondoSU2xU1::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.}, {"lim_alpha",11ul}, {"eps_svd",1.e-7},
	{"Dincr_abs", 4ul}, {"Dincr_per", 2ul}, {"Dincr_rel", 1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",30ul}, {"min_halfsweeps",6ul},
	{"Dinit",5ul}, {"Qinit",15ul}, {"Dlimit",100ul},
	{"tol_eigval",1.e-7}, {"tol_state",1.e-6},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
};

KondoSU2xU1::
KondoSU2xU1 (const size_t &L, const vector<Param> &params, const BC &boundary)
:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, PROP::HAMILTONIAN, boundary),
 KondoObservables(L,params,KondoSU2xU1::defaults),
 ParamReturner(KondoSU2xU1::sweep_defaults)
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("LyF",l%Lcell);		
		setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l);
	}
	
	this->set_name("Kondo");

	PushType<SiteOperator<Symmetry,double>,double> pushlist;
    std::vector<std::vector<std::string>> labellist;
    set_operators(B, F, P, pushlist, labellist, boundary);
    
    this->construct_from_pushlist(pushlist, labellist, Lcell);
    this->finalize(PROP::COMPRESS, P.get<size_t>("maxPower"));

	this->precalc_TwoSiteData();
}

bool KondoSU2xU1::
validate (qType qnum) const
{
	frac S_elec(qnum[1],2); //electrons have spin 1/2
	frac Smax = S_elec;
	for (size_t l=0; l<N_sites; ++l) { Smax+=static_cast<int>(B[l].orbitals())*frac(B[l].get_D()-1,2); } //add local spins to Smax
	
	frac S_tot(qnum[0]-1,2);
	if (Smax.denominator()==S_tot.denominator() and S_tot<=Smax and qnum[0]<=2*2*static_cast<int>(this->N_phys) and qnum[0]>0) {return true;}
	else {return false;}
}

template<typename Symmetry_> 
void KondoSU2xU1::
set_operators (const std::vector<SpinBase<Symmetry_> > &B, const std::vector<FermionBase<Symmetry_> > &F, const ParamHandler &P,
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
		
		std::size_t Forbitals       = F[loc].orbitals();
		std::size_t Fnext_orbitals  = F[(loc+1)%N_sites].orbitals();
		std::size_t Fnextn_orbitals = F[(loc+2)%N_sites].orbitals();
		
		std::size_t Borbitals       = B[loc].orbitals();
		std::size_t Bnext_orbitals  = B[(loc+1)%N_sites].orbitals();
		std::size_t Bnextn_orbitals = B[(loc+2)%N_sites].orbitals();
		
		stringstream Slabel;
		Slabel << "S=" << print_frac_nice(frac(B[loc].get_D()-1,2));
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
			SiteOperatorQ<Symmetry_,Eigen::MatrixXd> c_sign_local = kroneckerProduct(B[loc].Id(),F[loc].c(0) * F[loc].sign());
			SiteOperatorQ<Symmetry_,Eigen::MatrixXd> cdag_sign_local = kroneckerProduct(B[loc].Id(),(F[loc].cdag(0) * F[loc].sign()));
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > c_ranges(N_sites); for (size_t i=0; i<N_sites; i++) {c_ranges[i] = kroneckerProduct(B[loc].Id(),F[i].c(0));}
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > cdag_ranges(N_sites); for (size_t i=0; i<N_sites; i++) {cdag_ranges[i] = kroneckerProduct(B[loc].Id(),F[i].cdag(0));}
			
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first {cdag_sign_local,c_sign_local};
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {c_ranges,cdag_ranges};
			push_full("tFull", "tᵢⱼ", first, last, {-std::sqrt(2.),-std::sqrt(2.)}, PROP::FERMIONIC);
		}

		if (P.HAS("JdirFull"))
		{
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first {kroneckerProduct(B[loc].Sdag(0),F[loc].Id())};
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > S_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++) {S_ranges[i] = kroneckerProduct(B[i].S(0),F[loc].Id());}
			
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {S_ranges};
			push_full("Jdirfull", "Jdirᵢⱼ", first, last, {std::sqrt(3.)}, PROP::BOSONIC);
		}
		
		// local terms
		
		// Kondo-J
		param1d J = P.fill_array1d<double>("J", "Jorb", Forbitals, loc%Lcell);
		labellist[loc].push_back(J.label);
		
		// t⟂
		param2d tPerp = P.fill_array2d<double>("tRung", "t", "tPerp", Forbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		labellist[loc].push_back(tPerp.label);
		
		// V⟂
		param2d Vperp = P.fill_array2d<double>("Vrung", "V", "Vperp", Forbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		labellist[loc].push_back(Vperp.label);
		
		// Hubbard-U
		param1d U = P.fill_array1d<double>("U", "Uorb", Forbitals, loc%Lcell);
		labellist[loc].push_back(U.label);
		
		// Hubbard-U
		param1d Uph = P.fill_array1d<double>("Uph", "Uphorb", Forbitals, loc%Lcell);
		labellist[loc].push_back(Uph.label);
		
		// mu
		param1d mu = P.fill_array1d<double>("mu", "muorb", Forbitals, loc%Lcell);
		labellist[loc].push_back(mu.label);
		
		// t0
		param1d t0 = P.fill_array1d<double>("t0", "t0orb", Forbitals, loc%Lcell);
		labellist[loc].push_back(t0.label);
		
		if (F[loc].dim() > 1)
		{
			OperatorType KondoHamiltonian({1,0}, B[loc].get_basis().combine(F[loc].get_basis()));
			
			ArrayXXd Jperp    = B[loc].ZeroHopping();
			ArrayXXd Jperpsub = F[loc].ZeroHopping();
			ArrayXXd Vz       = F[loc].ZeroHopping();
			ArrayXXd Vxy      = F[loc].ZeroHopping();
			
			//set Hubbard part of Kondo Hamiltonian
			KondoHamiltonian = kroneckerProduct(B[loc].Id(), F[loc].template HubbardHamiltonian<double>(U.a,Uph.a,t0.a-mu.a,tPerp.a,Vperp.a,Vz,Vxy,Jperpsub));
			
			//set Heisenberg part of Hamiltonian
//			KondoHamiltonian += OperatorType::outerprod(B[loc].HeisenbergHamiltonian(Jperp), F[loc].Id(), {1,0});
			
			//set interaction part of Hamiltonian.
			for (int alfa=0; alfa<Forbitals; ++alfa)
			{
				assert(Borbitals == Forbitals and "Can only do a Kondo ladder with the same amount of spins and fermionic orbitals in y-direction!");
				KondoHamiltonian += J(alfa) * sqrt(3.) * SiteOperatorQ<Symmetry_,Eigen::MatrixXd>::outerprod(B[loc].Sdag(alfa), F[loc].S(alfa), {1,0});
			}

			pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(KondoHamiltonian), 1.));
		}

		// NN terms
		if (!P.HAS("tFull"))
		{
			// t∥
			param2d tPara = P.fill_array2d<double>("t", "tPara", {Forbitals, Fnext_orbitals}, loc%Lcell);
			labellist[loc].push_back(tPara.label);
		
			// V∥
			param2d Vpara = P.fill_array2d<double>("V", "Vpara", {Forbitals, Fnext_orbitals}, loc%Lcell);
			labellist[loc].push_back(Vpara.label);
		
			// JdirPara∥
			param2d JdirPara = P.fill_array2d<double>("Jdir", "JdirPara", {Borbitals, Bnext_orbitals}, loc%Lcell);
			labellist[loc].push_back(JdirPara.label);
		
			if (loc < N_sites-1 or !static_cast<bool>(boundary))
			{
				for (int alfa=0; alfa<Forbitals;      ++alfa)
				for (int beta=0; beta<Fnext_orbitals; ++beta)
				{
					auto cF_loc    = kroneckerProduct(B[loc].Id(), F[loc].c(alfa)) * kroneckerProduct(B[loc].Id(), F[loc].sign());
					auto cdagF_loc    = kroneckerProduct(B[loc].Id(), F[loc].cdag(alfa)) * kroneckerProduct(B[loc].Id(), F[loc].sign());

					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(cdagF_loc,kroneckerProduct(B[lp1].Id(),F[lp1].c(beta))),-std::sqrt(2.)*tPara(alfa,beta)));
					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(cF_loc,kroneckerProduct(B[lp1].Id(),F[lp1].cdag(beta))),-std::sqrt(2.)*tPara(alfa,beta)));

					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(kroneckerProduct(B[loc].Id(),F[loc].n(alfa)),
																										  kroneckerProduct(B[lp1].Id(),F[lp1].n(beta))),Vpara(alfa,beta)));								
				}
			
				for (int alfa=0; alfa<Borbitals;      ++alfa)
				for (int beta=0; beta<Bnext_orbitals; ++beta)
				{
					auto Sdag_loc = kroneckerProduct(B[loc].Sdag(alfa), F[loc].Id());
					auto S_tight = kroneckerProduct(B[lp1].S(beta), F[lp1].Id());
				
					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(Sdag_loc, S_tight), JdirPara(alfa,beta) * std::sqrt(3.)));
				}
			}
			
			// NN Kondo terms
		
			param2d InextPara = P.fill_array2d<double>("Inext", "InextPara", {Borbitals, Fnext_orbitals}, loc%Lcell);
			labellist[loc].push_back(InextPara.label);
		
			if (loc < N_sites-1 or !static_cast<bool>(boundary))
			{
				for (int alfa=0; alfa<Borbitals;      ++alfa)
				for (int beta=0; beta<Fnext_orbitals; ++beta)
				{				
					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction( kroneckerProduct(B[loc].Sdag(alfa), F[loc].Id()),
																											kroneckerProduct(B[lp1].Id(), F[lp1].S(beta))), sqrt(3.) * InextPara(alfa,beta)));
				}
			}
		
			param2d IprevPara = P.fill_array2d<double>("Iprev", "IprevPara", {Forbitals, Bnext_orbitals}, loc%Lcell);
			labellist[loc].push_back(IprevPara.label);
		
			if (lm1 < N_sites-1 or !static_cast<bool>(boundary))
			{
				for (int alfa=0; alfa<Forbitals;      ++alfa)
				for (int beta=0; beta<Bnext_orbitals; ++beta)
				{						
					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction( kroneckerProduct(B[lm1].Id(), F[lm1].Sdag(alfa)),
																											kroneckerProduct(B[loc].S(beta), F[loc].Id())), sqrt(3.) * IprevPara(alfa,beta)));
				}
			}
		
			// NNN terms
		
			param2d tPrime = P.fill_array2d<double>("tPrime", "tPrime_array", {Forbitals, Fnextn_orbitals}, loc%Lcell);
			labellist[loc].push_back(tPrime.label);
		
			if (loc < N_sites-2 or !static_cast<bool>(boundary))
			{
				for (std::size_t alfa=0; alfa<Forbitals;       ++alfa)
				for (std::size_t beta=0; beta<Fnextn_orbitals; ++beta)
				{
					auto cF_loc    = kroneckerProduct(B[loc].Id(), F[loc].c(alfa)) * kroneckerProduct(B[loc].Id(), F[loc].sign());
					auto cdagF_loc    = kroneckerProduct(B[loc].Id(), F[loc].cdag(alfa)) * kroneckerProduct(B[loc].Id(), F[loc].sign());
						
					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(cdagF_loc,
																										  kroneckerProduct(B[lp1].Id(),F[lp1].sign()),
																										  kroneckerProduct(B[lp2].Id(),F[lp2].c(beta))),-std::sqrt(2.)*tPrime(alfa,beta)));
					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(cF_loc,
																										  kroneckerProduct(B[lp1].Id(),F[lp1].sign()),
																										  kroneckerProduct(B[lp2].Id(),F[lp2].cdag(beta))),-std::sqrt(2.)*tPrime(alfa,beta)));
				}
			}
		}		
	}
}


// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// make_local (KONDO_SUBSYSTEM SUBSYS, 
//             string name, 
//             size_t locx, size_t locy, 
//             const OperatorType &Op, 
//             double factor, bool FERMIONIC, bool HERMITIAN) const
// {
// 	assert(locx<F.size() and locy<F[locx].dim());
// 	assert(SUBSYS != IMPSUB);
// 	stringstream ss;
// 	ss << name << "(" << locx << "," << locy << ")";
	
// 	Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > Mout(N_sites, Op.Q(), ss.str(), HERMITIAN);
// 	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l);}
	
// 	OperatorType OpExt;
// 	vector<SiteOperator<Symmetry,MatrixType::Scalar> > SignExt(locx);
	
// 	if (SUBSYS == SUB)
// 	{
// 		OpExt = OperatorType::outerprod(B[locx].Id(), Op, Op.Q());
// 		for (size_t l=0; l<locx; ++l)
// 		{
// 			SignExt[l] = OperatorType::outerprod(B[l].Id(), F[l].sign(), Symmetry::qvacuum()).plain<double>();
// 		}
// 	}
// 	else if (SUBSYS == IMP)
// 	{
// 		assert(!FERMIONIC and "Impurity cannot be fermionic!");
// 		OpExt = OperatorType::outerprod(Op, F[locx].Id(), Op.Q());
// 	}
	
// 	Mout.set_locality(locx);
// 	Mout.set_localOperator(OpExt.plain<double>());
	
// 	(FERMIONIC)? Mout.setLocal(locx, (factor * OpExt).plain<double>(), SignExt)
// 	           : Mout.setLocal(locx, (factor * OpExt).plain<double>());
	
// 	return Mout;
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// make_corr (KONDO_SUBSYSTEM SUBSYS,
//            string name1, string name2, 
//            size_t locx1, size_t locx2, size_t locy1, size_t locy2, 
//            const OperatorType &Op1, const OperatorType &Op2, 
//            qarray<Symmetry::Nq> Qtot, 
//            double factor,
//            bool BOTH_HERMITIAN)
// {
// 	assert(locx1<this->N_sites and locx2<this->N_sites);
// 	stringstream ss;
// 	ss << name1 << "(" << locx1 << "," << locy1 << ")" << name2 << "(" << locx2 << "," << locy2 << ")";
	
// 	bool HERMITIAN = (BOTH_HERMITIAN and locx1==locx2 and locy1==locy2)? true:false;
	
// 	Mpo<Symmetry> Mout(N_sites, Qtot, ss.str(), HERMITIAN);
// 	for (size_t l=0; l<this->N_sites; l++) {Mout.setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l);}
	
// 	OperatorType Op1Ext;
// 	OperatorType Op2Ext;
	
// 	if (SUBSYS == SUB)
// 	{
// 		Op1Ext = OperatorType::outerprod(B[locx1].Id(), Op1, Op1.Q());
// 		Op2Ext = OperatorType::outerprod(B[locx2].Id(), Op2, Op2.Q());
// 	}
// 	else if (SUBSYS == IMP)
// 	{
// 		Op1Ext = OperatorType::outerprod(Op1, F[locx1].Id(), Op1.Q());
// 		Op2Ext = OperatorType::outerprod(Op2, F[locx2].Id(), Op2.Q());
// 	}
// 	else if (SUBSYS == IMPSUB)
// 	{
// 		Op2Ext = OperatorType::outerprod(Op1, F[locx1].Id(), Op1.Q());
// 		Op1Ext = OperatorType::outerprod(B[locx2].Id(), Op2, Op2.Q());
// 	}
	
// 	if (locx1 == locx2)
// 	{
// 		auto LocProd = OperatorType::prod(Op1Ext, Op2Ext, Qtot);
// 		Mout.setLocal(locx1, factor * LocProd.plain<double>());
// 	}
// 	else
// 	{
// 		Mout.setLocal({locx1, locx2}, {factor * Op1Ext.plain<double>(), Op2Ext.plain<double>()});
// 	}
	
// 	return Mout;
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,complex<double> > KondoSU2xU1::
// make_FourierYSum (string name, const vector<OperatorType> &Ops, 
//                   double factor, bool HERMITIAN, const vector<complex<double> > &phases) const
// {
// 	stringstream ss;
// 	ss << name << "_ky(";
// 	for (int l=0; l<phases.size(); ++l)
// 	{
// 		ss << phases[l];
// 		if (l!=phases.size()-1) {ss << ",";}
// 		else                    {ss << ")";}
// 	}
	
// 	// all Ops[l].Q() must match
// 	Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,complex<double> > Mout(N_sites, Ops[0].Q(), ss.str(), HERMITIAN);
// 	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l);}
	
// 	vector<complex<double> > phases_x_factor = phases;
// 	for (int l=0; l<phases.size(); ++l)
// 	{
// 		phases_x_factor[l] = phases[l] * factor;
// 	}
	
// 	vector<SiteOperator<Symmetry,complex<double> > > OpsPlain(Ops.size());
// 	for (int l=0; l<OpsPlain.size(); ++l)
// 	{
// 		OpsPlain[l] = Ops[l].plain<double>().cast<complex<double> >();
// 	}
	
// 	Mout.setLocalSum(OpsPlain, phases_x_factor);
	
// 	return Mout;
// }

// //-----------------------------------------

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// n (size_t locx, size_t locy)
// {
// 	return make_local(SUB, "n", locx,locy, F[locx].n(locy), 1., false, true);
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// nh (size_t locx, size_t locy)
// {
// 	return make_local(SUB, "nh", locx,locy, F[locx].nh(locy), 1., false, true);
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// ns (size_t locx, size_t locy)
// {
// 	return make_local(SUB, "ns", locx,locy, F[locx].ns(locy), 1., false, true);
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// d (size_t locx, size_t locy)
// {
// 	return make_local(SUB, "d", locx,locy, F[locx].d(locy), 1., false, true);
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// c (size_t locx, size_t locy, double factor)
// {
// 	return make_local(SUB, "c", locx,locy, F[locx].c(locy), factor, true, false);
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// cdag (size_t locx, size_t locy, double factor)
// {
// 	return make_local(SUB, "c†", locx,locy, F[locx].cdag(locy), factor, true, false);
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// cc (size_t locx, size_t locy)
// {
// 	stringstream ss;
// 	ss << "c" << UP << "c" << DN;
// 	return make_local(SUB, ss.str(), locx,locy, F[locx].cc(locy), 1., false, false);
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// cdagcdag (size_t locx, size_t locy)
// {
// 	stringstream ss;
// 	ss << "c†" << UP << "c†" << DN;
// 	return make_local(SUB, ss.str(), locx,locy, F[locx].cdagcdag(locy), 1., false, false);
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// Ssub (size_t locx, size_t locy, double factor)
// {
// 	return make_local(SUB, "Ssub", locx,locy, F[locx].S(locy), factor, false, false);
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// Ssubdag (size_t locx, size_t locy, double factor)
// {
// 	return make_local(SUB, "Ssub†", locx,locy, F[locx].Sdag(locy), factor, false, false);
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// Simp (size_t locx, size_t locy, double factor)
// {
// 	return make_local(IMP, "Simp", locx,locy, B[locx].S(locy), factor, false, false);
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// Simpdag (size_t locx, size_t locy, double factor)
// {
// 	return make_local(IMP, "Simp†", locx,locy, B[locx].Sdag(locy), factor, false, false);
// }


// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >, complex<double> > KondoSU2xU1::
// Simp_ky (vector<complex<double> > phases) const
// {
// 	vector<OperatorType> Ops(N_sites);
// 	for (size_t l=0; l<N_sites; ++l)
// 	{
// 		Ops[l] = F[l].S(0);
// 	}
// 	return make_FourierYSum("S", Ops, 1., false, phases);
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >, complex<double> > KondoSU2xU1::
// Simpdag_ky (vector<complex<double> > phases, double factor) const
// {
// 	vector<OperatorType> Ops(N_sites);
// 	for (size_t l=0; l<N_sites; ++l)
// 	{
// 		Ops[l] = F[l].Sdag(0);
// 	}
// 	return make_FourierYSum("S†", Ops, 1., false, phases);
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// nn (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
// {
// 	return make_corr (SUB, "n","n", locx1,locx2,locy1,locy2, F[locx1].n(locy1), F[locx2].n(locy2), Symmetry::qvacuum(), 1., true);
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// SsubSsub (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
// {
// 	return make_corr (SUB, "Ssub","Ssub", locx1,locx2,locy1,locy2, F[locx1].Sdag(locy1),F[locx2].S(locy2), Symmetry::qvacuum(), sqrt(3.), false);
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// SimpSimp (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
// {
// 	return make_corr (IMP, "Simp","Simp", locx1,locx2,locy1,locy2, B[locx1].Sdag(locy1),B[locx2].S(locy2), Symmetry::qvacuum(), sqrt(3.), false);
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// SimpSsub (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
// {
// 	return make_corr (IMPSUB, "Simp","Ssub", locx1,locx2,locy1,locy2, B[locx1].Sdag(locy1),F[locx2].S(locy2), Symmetry::qvacuum(), sqrt(3.), false);
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// cdagc (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
// {
// 	assert(locx1<this->N_sites and locx2<this->N_sites);
// 	stringstream ss;
// 	ss << "c†(" << locx1 << "," << locy1 << ")" << "c(" << locx2 << "," << locy2 << ")";
	
// 	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
// 	for (size_t l=0; l<this->N_sites; l++) {Mout.setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l);}
	
// 	auto cdag = OperatorType::outerprod(B[locx1].Id(), F[locx1].cdag(locy1),{2,+1});
// 	auto c    = OperatorType::outerprod(B[locx2].Id(), F[locx2].c(locy2),   {2,-1});
// 	auto sign1 = OperatorType::outerprod(B[locx1].Id(), F[locx1].sign(),    {1, 0});
// 	auto sign2 = OperatorType::outerprod(B[locx2].Id(), F[locx2].sign(),    {1, 0});
	
// 	vector<SiteOperator<Symmetry,MatrixType::Scalar> > signs;
// 	for (size_t l=min(locx1,locx2)+1; l<max(locx1,locx2); l++)
// 	{
// 		signs.push_back(OperatorType::outerprod(B[l].Id(), F[l].sign(), {1, 0}).plain<double>());
// 	}
	
// 	if (locx1 == locx2)
// 	{
// 		Mout.setLocal(locx1, sqrt(2.) * OperatorType::prod(cdag,c,Symmetry::qvacuum()).plain<double>());
// 	}
// 	else if (locx1<locx2)
// 	{
// 		Mout.setLocal({locx1, locx2}, {sqrt(2.) * OperatorType::prod(cdag, sign1, {2,+1}).plain<double>(), c.plain<double>()}, signs);
// 	}
// 	else if (locx1>locx2)
// 	{
// 		Mout.setLocal({locx2, locx1}, {sqrt(2.) * OperatorType::prod(c, sign2, {2,-1}).plain<double>(), cdag.plain<double>()}, signs);
// 	}
// 	return Mout;
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// ccdag (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
// {
// 	assert(locx1<this->N_sites and locx2<this->N_sites);
// 	stringstream ss;
// 	ss << "c(" << locx1 << "," << locy1 << ")" << "c†(" << locx2 << "," << locy2 << ")";
	
// 	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
// 	for(size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l); }
	
// 	auto cdag = OperatorType::outerprod(B[locx1].Id(),F[locx1].cdag(locy1),{2,+1});
// 	auto c    = OperatorType::outerprod(B[locx2].Id(),F[locx2].c(locy2),{2,-1});
// 	auto sign = OperatorType::outerprod(B[locx2].Id(),F[locx2].sign(),{1,0});
	
// 	if (locx1 == locx2)
// 	{
// 		auto product = sqrt(2.)*OperatorType::prod(c,cdag,Symmetry::qvacuum());
// 		Mout.setLocal(locx1,product.plain<double>());
// 	}
// 	else if (locx1<locx2)
// 	{
// 		Mout.setLocal({locx1, locx2}, {sqrt(2.) * OperatorType::prod(c, sign, {2,-1}).plain<double>(), 
// 		                               cdag.plain<double>()}, 
// 		                               sign.plain<double>());
// 	}
// 	else if (locx1>locx2)
// 	{
// 		Mout.setLocal({locx2, locx1}, {sqrt(2.)*OperatorType::prod(cdag, sign, {2,+1}).plain<double>(), 
// 		                               c.plain<double>()}, 
// 			                           sign.plain<double>());
// 	}
// 	return Mout;
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// dh_excitation (size_t loc)
// {
// 	size_t lp1 = loc+1;
	
// 	OperatorType PsiRloc = OperatorType::prod(OperatorType::prod(F[loc].c(0), F[loc].sign(), {2,-1}), F[loc].ns(0), {2,-1});
// 	OperatorType PsidagLlp1 = OperatorType::prod(F[lp1].cdag(0), F[lp1].ns(0),{2,1});
	
// 	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), "dh");
// 	for(size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l); }
	
// 	OperatorType Op1Ext = OperatorType::outerprod(B[loc].Id(), PsiRloc, PsiRloc.Q());
// 	OperatorType Op2Ext = OperatorType::outerprod(B[lp1].Id(), PsidagLlp1, PsidagLlp1.Q());
	
// 	Mout.setLocal({loc, lp1}, {Op1Ext.plain<double>(), Op2Ext.plain<double>()});
	
// 	return Mout;
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// cdagcdagcc (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
// {
// 	assert(locx1<this->N_sites and locx2<this->N_sites);
// 	stringstream ss;
// 	ss << "η†(" << locx1 << "," << locy1 << ")" << "η(" << locx2 << "," << locy2 << ")";

// 	Mpo<Symmetry> Mout(N_sites, N_legs);
// 	for(size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(Spins.basis().combine(F.basis()),l); }

// 	auto Etadag = Operator::outerprod(Spins.Id(),F.cdagcdag(locy1),{1,2});
// 	auto Eta = Operator::outerprod(Spins.Id(),F.cc(locy2),{1,-2});
// 	Mout.label = ss.str();
// 	Mout.setQtarget(Symmetry::qvacuum());
// 	Mout.qlabel = KondoSU2xU1::SNlabel;
// 	if(locx1 == locx2)
// 	{
// 		auto product = Operator::prod(Etadag,Eta,Symmetry::qvacuum());
// 		Mout.setLocal(locx1,product,Symmetry::qvacuum());
// 		return Mout;
// 	}
// 	else
// 	{
// 		Mout.setLocal({locx1, locx2}, {Etadag, Eta}, {{1,2},{1,-2}});
// 		return Mout;
// 	}
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// SimpSsubSimpSimp (size_t locx1, size_t locx2, size_t loc3x, size_t loc4x,
//                   size_t locy1, size_t locy2, size_t loc3y, size_t loc4y)
// {
// 	assert(locx1<this->N_sites and locx2<this->N_sites and loc3x<this->N_sites and loc4x<this->N_sites);
// 	stringstream ss;
// 	ss << SOP1 << "(" << locx1 << "," << locy1 << ")" << SOP2 << "(" << locx2 << "," << locy2 << ")" <<
// 	      SOP3 << "(" << loc3x << "," << loc3y << ")" << SOP4 << "(" << loc4x << "," << loc4y << ")";
// 	Mpo<2> Mout(this->N_sites, this->N_legs, locBasis(), {0,0}, KondoSU2xU1::NMlabel, ss.str());
// 	MatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
// 	MatrixXd IdImp(Mpo<2>::qloc[locx2].size()/F.dim(), Mpo<2>::qloc[locx2].size()/F.dim()); IdImp.setIdentity();
// 	Mout.setLocal({locx1, locx2, loc3x, loc4x}, {kroneckerProduct(S.Scomp(SOP1,locy1),IdSub), 
// 	                                             kroneckerProduct(IdImp,F.Scomp(SOP2,locy2)),
// 	                                             kroneckerProduct(S.Scomp(SOP3,loc3y),IdSub),
// 	                                             kroneckerProduct(S.Scomp(SOP4,loc4y),IdSub)});
// 	return Mout;
// }

// Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > KondoSU2xU1::
// SimpSsubSimpSsub (size_t locx1, size_t locx2, size_t loc3x, size_t loc4x,
// 				  size_t locy1, size_t locy2, size_t loc3y, size_t loc4y)
// {
// 	assert(locx1<this->N_sites and locx2<this->N_sites and loc3x<this->N_sites and loc4x<this->N_sites);
// 	stringstream ss;
// 	ss << SOP1 << "(" << locx1 << "," << locy1 << ")" << SOP2 << "(" << locx2 << "," << locy2 << ")" <<
// 	      SOP3 << "(" << loc3x << "," << loc3y << ")" << SOP4 << "(" << loc4x << "," << loc4y << ")";
// 	Mpo<2> Mout(this->N_sites, this->N_legs, locBasis(), {0,0}, KondoSU2xU1::NMlabel, ss.str());
// 	MatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
// 	MatrixXd IdImp(Mpo<2>::qloc[locx2].size()/F.dim(), Mpo<2>::qloc[locx2].size()/F.dim()); IdImp.setIdentity();
// 	Mout.setLocal({locx1, locx2, loc3x, loc4x}, {kroneckerProduct(S.Scomp(SOP1,locy1),IdSub), 
// 	                                             kroneckerProduct(IdImp,F.Scomp(SOP2,locy2)),
// 	                                             kroneckerProduct(S.Scomp(SOP3,loc3y),IdSub),
// 	                                             kroneckerProduct(IdImp,F.Scomp(SOP4,loc4y))}
// 		);
// 	return Mout;
// }


// bool KondoSU2xU1::
// validate (qType qnum) const
// {
// 	int Sx2 = static_cast<int>(D-1); // necessary because of size_t
// 	return (qnum[0]-1+N_legs*Sx2*imploc.size())%2 == qnum[1]%2;
// }

} //end namespace VMPS

#endif
