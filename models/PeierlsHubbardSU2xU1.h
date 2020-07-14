#ifndef HUBBARDMODELSU2XU1_H_COMPLEX
#define HUBBARDMODELSU2XU1_H_COMPLEX

#include "symmetry/S1xS2.h"
#include "symmetry/U1.h"
#include "symmetry/SU2.h"
#include "bases/FermionBase.h"
#include "models/HubbardObservables.h"
//include "tensors/SiteOperatorQ.h"
//include "tensors/SiteOperator.h"
#include "Mpo.h"
//include "DmrgExternal.h"
//include "ParamHandler.h"
#include "ParamReturner.h"
#include "Geometry2D.h" // from TOOLS

namespace VMPS
{
class PeierlsHubbardSU2xU1 : public Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,complex<double>>,
                             public HubbardObservables<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,complex<double>>,
                             public ParamReturner
{
public:
	
	typedef Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > Symmetry;
	MAKE_TYPEDEFS(PeierlsHubbardSU2xU1)
	typedef Eigen::Matrix<complex<double>,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	
private:
	
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	
public:
	
	PeierlsHubbardSU2xU1() : Mpo(){};
	PeierlsHubbardSU2xU1 (const size_t &L, const vector<Param> &params, const BC &boundary=BC::OPEN, const DMRG::VERBOSITY::OPTION &VERB=DMRG::VERBOSITY::OPTION::ON_EXIT);
	
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
	                           PushType<SiteOperator<Symmetry_,complex<double>>,complex<double>>& pushlist, std::vector<std::vector<std::string>>& labellist, 
	                           const BC boundary=BC::OPEN);
	
	static qarray<2> singlet (int N=0) {return qarray<2>{1,N};};
	static constexpr MODEL_FAMILY FAMILY = HUBBARD;
	
	static const map<string,any> defaults;
	static const map<string,any> sweep_defaults;
};

// V is standard next-nearest neighbour density interaction
// Vz and Vxy are anisotropic isospin-isospin next-nearest neighbour interaction
const map<string,any> PeierlsHubbardSU2xU1::defaults = 
{
	{"t",1.+0.i}, {"tPrime",0.i}, {"tRung",1.+0.i}, {"tPrimePrime",0.i},
	{"mu",0.}, {"t0",0.}, 
	{"U",0.}, {"Uph",0.},
	{"V",0.}, {"Vext",0.}, {"Vrung",0.},
	{"Vz",0.}, {"Vzrung",0.}, {"Vxy",0.}, {"Vxyrung",0.}, 
	{"J",0.}, {"Jperp",0.},
	{"X",0.}, {"Xrung",0.},
	{"maxPower",2ul}, {"CYLINDER",false}, {"Ly",1ul}
};

const map<string,any> PeierlsHubbardSU2xU1::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.}, {"lim_alpha",11ul}, {"eps_svd",1e-7},
	{"Dincr_abs", 4ul}, {"Dincr_per", 2ul}, {"Dincr_rel", 1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",24ul}, {"min_halfsweeps",1ul},
	{"Dinit",8ul}, {"Qinit",10ul}, {"Dlimit",500ul},
	{"tol_eigval",1e-7}, {"tol_state",1e-6},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
};

PeierlsHubbardSU2xU1::
PeierlsHubbardSU2xU1 (const size_t &L, const vector<Param> &params, const BC &boundary, const DMRG::VERBOSITY::OPTION &VERB)
:Mpo<Symmetry,complex<double>> (L, qarray<Symmetry::Nq>({1,0}), "", PROP::HERMITIAN, PROP::NON_UNITARY, boundary, VERB),
 HubbardObservables(L,params,PeierlsHubbardSU2xU1::defaults),
 ParamReturner(PeierlsHubbardSU2xU1::sweep_defaults)
{
	ParamHandler P(params,defaults);
	size_t Lcell = P.size();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(F[l].get_basis().qloc(),l);
	}
	
	this->set_name("Peierls-Hubbard");
	
	PushType<SiteOperator<Symmetry,complex<double>>,complex<double>> pushlist;
	std::vector<std::vector<std::string>> labellist;
	set_operators(F, P, pushlist, labellist, boundary);
	
	this->construct_from_pushlist(pushlist, labellist, Lcell);
	this->finalize(PROP::COMPRESS, P.get<size_t>("maxPower"));
	
	this->precalc_TwoSiteData();
}

template<typename Symmetry_>
void PeierlsHubbardSU2xU1::
set_operators (const std::vector<FermionBase<Symmetry_> > &F, const ParamHandler &P, PushType<SiteOperator<Symmetry_,complex<double>>,complex<double>>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary)
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
		
		stringstream ss;
		ss << "Ly=" << P.get<size_t>("Ly",loc%Lcell);
		labellist[loc].push_back(ss.str());
		
		auto push_full = [&N_sites, &loc, &F, &P, &pushlist, &labellist, &boundary] (string xxxFull, string label,
																					 const vector<SiteOperatorQ<Symmetry_,MatrixType>> &first,
																					 const vector<vector<SiteOperatorQ<Symmetry_,MatrixType>>> &last,
																					 vector<complex<double>> factor, 
																					 vector<bool> CONJ,
																					 bool FERMIONIC) -> void
		{
			ArrayXXcd Full = P.get<Eigen::ArrayXXcd>(xxxFull);
			vector<vector<std::pair<size_t,complex<double>> > > R = Geometry2D::rangeFormat(Full);
			
			if (static_cast<bool>(boundary)) {assert(R.size() ==   N_sites and "Use an (N_sites)x(N_sites) hopping matrix for open BC!");}
			else                             {assert(R.size() >= 2*N_sites and "Use at least a (2*N_sites)x(N_sites) hopping matrix for infinite BC!");}
			
			for (size_t j=0; j<first.size(); j++)
			for (size_t h=0; h<R[loc].size(); ++h)
			{
				size_t range = R[loc][h].first;
				complex<double> value = R[loc][h].second;
				
				if (range != 0)
				{
					vector<SiteOperatorQ<Symmetry_,MatrixType> > ops(range+1);
					ops[0] = first[j];
					for (size_t i=1; i<range; ++i)
					{
						if (FERMIONIC) {ops[i] = F[(loc+i)%N_sites].sign().template cast<complex<double>>();}
						else {ops[i] = F[(loc+i)%N_sites].Id().template cast<complex<double>>();}
					}
					ops[range] = last[j][(loc+range)%N_sites];
					complex<double> total_value =  factor[j] * value;
					if (CONJ[j]) total_value = conj(total_value);
					pushlist.push_back(std::make_tuple(loc, ops, total_value));
				}
			}
			
			stringstream ss;
			ss << label << "(" << Geometry2D::hoppingInfo(Full) << ")";
			labellist[loc].push_back(ss.str());
		};
		
		if (P.HAS("tFull"))
		{
			SiteOperatorQ<Symmetry_,MatrixType> c_sign_local = (F[loc].c(0) * F[loc].sign()).template cast<complex<double>>();
			SiteOperatorQ<Symmetry_,MatrixType> cdag_sign_local = (F[loc].cdag(0) * F[loc].sign()).template cast<complex<double>>();
			vector<SiteOperatorQ<Symmetry_,MatrixType> > c_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++) {c_ranges[i] = F[i].c(0).template cast<complex<double>>();}
			vector<SiteOperatorQ<Symmetry_,MatrixType> > cdag_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++) {cdag_ranges[i] = F[i].cdag(0).template cast<complex<double>>();}
			
			vector<SiteOperatorQ<Symmetry_,MatrixType> > first {cdag_sign_local,c_sign_local};
			vector<vector<SiteOperatorQ<Symmetry_,MatrixType> > > last {c_ranges,cdag_ranges};
			push_full("tFull", "tᵢⱼ", first, last, {-std::sqrt(2.), -std::sqrt(2.)}, {false, true}, PROP::FERMIONIC);
		}
//		if (P.HAS("Vzfull"))
//		{
//			vector<SiteOperatorQ<Symmetry_,MatrixType> > first {F[loc].Tz(0)};
//			vector<SiteOperatorQ<Symmetry_,MatrixType> > Tz_ranges(N_sites); for (size_t i=0; i<N_sites; i++) {Tz_ranges[i] = F[i].Tz(0);}
//			vector<vector<SiteOperatorQ<Symmetry_,MatrixType> > > last {Tz_ranges};
//			push_full("Vzfull", "Vzᵢⱼ", first, last, {1.}, PROP::BOSONIC);			
//		}
//		if (P.HAS("Vxyfull"))
//		{
//			auto Gloc = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,loc)));
//			vector<SiteOperatorQ<Symmetry_,MatrixType> > first {F[loc].Tp(0,Gloc), F[loc].Tm(0,Gloc)};
//			vector<SiteOperatorQ<Symmetry_,MatrixType> > Tp_ranges(N_sites);
//			vector<SiteOperatorQ<Symmetry_,MatrixType> > Tm_ranges(N_sites);
//			for (size_t i=0; i<N_sites; i++)
//			{auto Gi = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,i))); Tp_ranges[i] = F[i].Tp(0,Gi); Tm_ranges[i] = F[i].Tm(0,Gi);}
//			
//			vector<vector<SiteOperatorQ<Symmetry_,MatrixType> > > last {Tm_ranges, Tp_ranges};
//			push_full("Vxyfull", "Vxyᵢⱼ", first, last, {0.5,0.5}, PROP::BOSONIC);
//		}
//		if (P.HAS("VextFull"))
//		{
//			vector<SiteOperatorQ<Symmetry_,MatrixType> > first {F[loc].n(0)};
//			vector<SiteOperatorQ<Symmetry_,MatrixType> > n_ranges(N_sites); for (size_t i=0; i<N_sites; i++) {n_ranges[i] = F[i].n(0);}
//			vector<vector<SiteOperatorQ<Symmetry_,MatrixType> > > last {n_ranges};
//			push_full("VextFull", "Vextᵢⱼ", first, last, {1.}, PROP::BOSONIC);
//		}
//		if (P.HAS("Jfull"))
//		{
//			vector<SiteOperatorQ<Symmetry_,MatrixType> > first {F[loc].Sdag(0)};
//			vector<SiteOperatorQ<Symmetry_,MatrixType> > S_ranges(N_sites); for (size_t i=0; i<N_sites; i++) {S_ranges[i] = F[i].S(0);}
//			vector<vector<SiteOperatorQ<Symmetry_,MatrixType> > > last {S_ranges};
//			push_full("Jfull", "Jᵢⱼ", first, last, {std::sqrt(3.)}, PROP::BOSONIC);
//		}
//		if (P.HAS("Xfull"))
//		{
//			SiteOperatorQ<Symmetry_,MatrixType> PsiLloc = ((F[loc].ns() * F[loc].c()) * F[loc].sign());
//			SiteOperatorQ<Symmetry_,MatrixType> PsiRloc = ((F[loc].c() * F[loc].sign()) * F[loc].ns());
//			
//			vector<SiteOperatorQ<Symmetry_,MatrixType> > PsiLran(N_sites); for(size_t i=0; i<N_sites; i++) {PsiLran[i] = (F[i].ns() * F[i].c());}
//			vector<SiteOperatorQ<Symmetry_,MatrixType> > PsiRran(N_sites); for(size_t i=0; i<N_sites; i++) {PsiRran[i] = (F[i].c() * F[i].ns());}
//			
//			SiteOperatorQ<Symmetry_,MatrixType> PsidagLloc = ((F[loc].cdag() * F[loc].sign()) * F[loc].ns());
//			SiteOperatorQ<Symmetry_,MatrixType> PsidagRloc = ((F[loc].ns() * F[loc].cdag()) * F[loc].sign());
//			
//			vector<SiteOperatorQ<Symmetry_,MatrixType> > PsidagLran(N_sites); for(size_t i=0; i<N_sites; i++) {PsidagLran[i] = (F[i].cdag() * F[i].ns());}
//			vector<SiteOperatorQ<Symmetry_,MatrixType> > PsidagRran(N_sites); for(size_t i=0; i<N_sites; i++) {PsidagRran[i] = (F[i].ns() * F[i].cdag());}
//			
//			vector<SiteOperatorQ<Symmetry_,MatrixType> > first {PsidagLloc,PsidagRloc,PsiLloc,PsiRloc};
//			vector<vector<SiteOperatorQ<Symmetry_,MatrixType> > > last {PsiRran,PsiLran,PsidagRran,PsidagLran};
//			push_full("Xfull", "Xᵢⱼ", first, last, {-std::sqrt(2.), -std::sqrt(2.), -std::sqrt(2.), -std::sqrt(2.)}, PROP::FERMIONIC);
//		}
		
		// Local terms: U, t0, μ, t⟂, V⟂, J⟂
		param1d U = P.fill_array1d<double>("U", "Uorb", orbitals, loc%Lcell);
		param1d Uph = P.fill_array1d<double>("Uph", "Uphorb", orbitals, loc%Lcell);
		param1d t0 = P.fill_array1d<double>("t0", "t0orb", orbitals, loc%Lcell);
		param1d mu = P.fill_array1d<double>("mu", "muorb", orbitals, loc%Lcell);
		param2d tperp = P.fill_array2d<complex<double>>("tRung", "t", "tPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Vperp = P.fill_array2d<double>("VRung", "V", "VPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Vzperp = P.fill_array2d<double>("VzRung", "Vz", "VzPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Vxyperp = P.fill_array2d<double>("VxyRung", "Vxy", "VxyPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Jperp = P.fill_array2d<double>("JRung", "J", "JPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		
		labellist[loc].push_back(U.label);
		labellist[loc].push_back(Uph.label);
		labellist[loc].push_back(t0.label);
		labellist[loc].push_back(mu.label);
		labellist[loc].push_back(tperp.label);
		labellist[loc].push_back(Vperp.label);
		labellist[loc].push_back(Vzperp.label);
		labellist[loc].push_back(Vxyperp.label);
		labellist[loc].push_back(Jperp.label);
		
		auto Hloc = Mpo<Symmetry_,complex<double>>::get_N_site_interaction
		(
			F[loc].template HubbardHamiltonian<complex<double>,Symmetry_>(U.a.cast<complex<double>>(), 
			                                                              Uph.a.cast<complex<double>>(),
			                                                              (t0.a-mu.a).cast<complex<double>>(),
			                                                              tperp.a,
			                                                              Vperp.a.cast<complex<double>>(),
			                                                              Vzperp.a.cast<complex<double>>(),
			                                                              Vxyperp.a.cast<complex<double>>(),
			                                                              Jperp.a.cast<complex<double>>())
		);
		pushlist.push_back(std::make_tuple(loc, Hloc, 1.+0.i));
		
		// Nearest-neighbour terms: t, V, J
//		if (!P.HAS("tFull") and !P.HAS("Vzfull") and !P.HAS("Vxyfull") and !P.HAS("Jfull") and !P.HAS("Xfull"))
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
//					SiteOperatorQ<Symmetry_,MatrixType> c_sign_local    = (F[loc].c(alfa) *    F[loc].sign());
//					SiteOperatorQ<Symmetry_,MatrixType> cdag_sign_local = (F[loc].cdag(alfa) * F[loc].sign());
//					
//					SiteOperatorQ<Symmetry_,MatrixType> c_tight    = F[lp1].c   (beta);
//					SiteOperatorQ<Symmetry_,MatrixType> cdag_tight = F[lp1].cdag(beta);
//					
//					SiteOperatorQ<Symmetry_,MatrixType> n_local = F[loc].n(alfa);
//					SiteOperatorQ<Symmetry_,MatrixType> n_tight = F[lp1].n(beta);
//					
//					SiteOperatorQ<Symmetry_,MatrixType> tz_local = F[loc].Tz(alfa);
//					SiteOperatorQ<Symmetry_,MatrixType> tz_tight = F[lp1].Tz(beta);

//					auto Gloc = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,loc)));
//					auto Glp1 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,lp1)));
//					SiteOperatorQ<Symmetry_,MatrixType> tp_local = F[loc].Tp(alfa,Gloc);
//					SiteOperatorQ<Symmetry_,MatrixType> tm_tight = F[lp1].Tm(beta,Glp1);
//					
//					SiteOperatorQ<Symmetry_,MatrixType> tm_local = F[loc].Tm(alfa,Gloc);
//					SiteOperatorQ<Symmetry_,MatrixType> tp_tight = F[lp1].Tp(beta,Glp1);
//					
//					SiteOperatorQ<Symmetry_,MatrixType> Sdag_local = F[loc].Sdag(alfa);
//					SiteOperatorQ<Symmetry_,MatrixType> S_tight    = F[lp1].S   (beta);
//					
//					SiteOperatorQ<Symmetry_,MatrixType> PsiLloc = ((F[loc].ns(alfa) * F[loc].c(alfa)) * F[loc].sign());
//					SiteOperatorQ<Symmetry_,MatrixType> PsiRloc = ((F[loc].c(alfa) * F[loc].sign()) * F[loc].ns(alfa));
//					SiteOperatorQ<Symmetry_,MatrixType> PsiLlp1 = (F[lp1].ns(beta) * F[lp1].c(beta));
//					SiteOperatorQ<Symmetry_,MatrixType> PsiRlp1 = (F[lp1].c(beta) * F[lp1].ns(beta));
//					SiteOperatorQ<Symmetry_,MatrixType> PsidagLloc = ((F[loc].cdag(alfa) * F[loc].sign()) * F[loc].ns(alfa));
//					SiteOperatorQ<Symmetry_,MatrixType> PsidagRloc = ((F[loc].ns(alfa) * F[loc].cdag(alfa)) * F[loc].sign());
//					SiteOperatorQ<Symmetry_,MatrixType> PsidagLlp1 = (F[lp1].cdag(beta) * F[lp1].ns(beta));
//					SiteOperatorQ<Symmetry_,MatrixType> PsidagRlp1 = (F[lp1].ns(beta) * F[lp1].cdag(beta));
//					
//					//hopping
//					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(cdag_sign_local, c_tight), -std::sqrt(2.)*tpara(alfa,beta)));
//					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(c_sign_local, cdag_tight), -std::sqrt(2.)*tpara(alfa,beta)));
//					
//					//density-density interaction
//					pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry_,double>::get_N_site_interaction(n_local, n_tight), Vpara(alfa,beta)));
//					
//					//isospin-isospin interaction
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
		
//		// Next-nearest-neighbour terms: t'
//		if (!P.HAS("tFull"))
//		{
//			param2d tPrime = P.fill_array2d<double>("tPrime", "tPrime_array", {orbitals, nextn_orbitals}, loc%Lcell);
//			labellist[loc].push_back(tPrime.label);
//			
//			if (loc < N_sites-2 or !static_cast<bool>(boundary))
//			{
//				for (std::size_t alfa=0; alfa<orbitals;       ++alfa)
//				for (std::size_t beta=0; beta<nextn_orbitals; ++beta)
//				{
//					SiteOperatorQ<Symmetry_,MatrixType> c_sign_local    = (F[loc].c(alfa) *    F[loc].sign());
//					SiteOperatorQ<Symmetry_,MatrixType> cdag_sign_local = (F[loc].cdag(alfa) * F[loc].sign());
//					
//					SiteOperatorQ<Symmetry_,MatrixType> sign_tight = F[lp1].sign();
//					
//					SiteOperatorQ<Symmetry_,MatrixType> c_nextn    = F[lp2].c(beta);
//					SiteOperatorQ<Symmetry_,MatrixType> cdag_nextn = F[lp2].cdag(beta);
//					
//					pushlist.push_back(std::make_tuple(loc, 
//					                   Mpo<Symmetry_,double>::get_N_site_interaction(cdag_sign_local, sign_tight, c_nextn), 
//					                   -std::sqrt(2.)*tPrime(alfa,beta)));
//					pushlist.push_back(std::make_tuple(loc, 
//					                   Mpo<Symmetry_,double>::get_N_site_interaction(c_sign_local, sign_tight, cdag_nextn), 
//					                   -std::sqrt(2.)*tPrime(alfa,beta)));
//				}
//			}
//		}
//		
//		// Next-next-nearest-neighbour terms: t''
//		if (!P.HAS("tFull"))
//		{
//			param2d tPrimePrime = P.fill_array2d<double>("tPrimePrime", "tPrimePrime_array", {orbitals, nnextn_orbitals}, loc%Lcell);
//			labellist[loc].push_back(tPrimePrime.label);
//			
//			if (loc < N_sites-3 or !static_cast<bool>(boundary))
//			{
//				SiteOperatorQ<Symmetry_,MatrixType> sign_tight = F[lp1].sign();
//				SiteOperatorQ<Symmetry_,MatrixType> sign_nextn = F[lp2].sign();
//				
//				for (std::size_t alfa=0; alfa<orbitals;        ++alfa)
//				for (std::size_t beta=0; beta<nnextn_orbitals; ++beta)
//				{
//					SiteOperatorQ<Symmetry_,MatrixType> c_sign_local    = (F[loc].c(alfa) *    F[loc].sign());
//					SiteOperatorQ<Symmetry_,MatrixType> cdag_sign_local = (F[loc].cdag(alfa) * F[loc].sign());
//					
//					SiteOperatorQ<Symmetry_,MatrixType> c_nnextn    = F[lp3].c(beta);
//					SiteOperatorQ<Symmetry_,MatrixType> cdag_nnextn = F[lp3].cdag(beta);
//					
//					pushlist.push_back(std::make_tuple(loc, 
//					                   Mpo<Symmetry_,double>::get_N_site_interaction(cdag_sign_local, sign_tight, sign_nextn, c_nnextn), 
//					                   -std::sqrt(2.)*tPrimePrime(alfa,beta)));
//					pushlist.push_back(std::make_tuple(loc, 
//					                   Mpo<Symmetry_,double>::get_N_site_interaction(c_sign_local, sign_tight, sign_nextn, cdag_nnextn), 
//					                   -std::sqrt(2.)*tPrimePrime(alfa,beta)));
//				}
//			}
//		}
	}
}

} // end namespace VMPS::models

#endif
