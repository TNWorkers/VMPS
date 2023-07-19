#ifndef HUBBARDMODELU1XU1_H_COMPLEX
#define HUBBARDMODELU1XU1_H_COMPLEX

#include "symmetry/S1xS2.h"
#include "symmetry/U1.h"
#include "bases/FermionBase.h"
#include "models/HubbardObservables.h"
#include "Mpo.h"
#include "ParamReturner.h"
#include "Geometry2D.h" // from TOOLS

namespace VMPS
{
class PeierlsHubbardU1xU1 : public Mpo<Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::ChargeU1> >,complex<double>>,
                            public HubbardObservables<Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::ChargeU1> >,complex<double>>,
                            public ParamReturner
{
public:
	
	typedef Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::ChargeU1> > Symmetry;
	MAKE_TYPEDEFS(PeierlsHubbardU1xU1)
	typedef Eigen::Matrix<complex<double>,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	
//private:
	
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	
public:
	
	PeierlsHubbardU1xU1() : Mpo(){};
	
	PeierlsHubbardU1xU1(Mpo<Symmetry,complex<double>> &Mpo_input, const vector<Param> &params)
	:Mpo<Symmetry,complex<double>>(Mpo_input),
	 HubbardObservables(this->N_sites,params,PeierlsHubbardU1xU1::defaults),
	 ParamReturner(PeierlsHubbardU1xU1::sweep_defaults)
	{
		ParamHandler P(params,PeierlsHubbardU1xU1::defaults);
		size_t Lcell = P.size();
		N_phys = 0;
		for (size_t l=0; l<N_sites; ++l) N_phys += P.get<size_t>("Ly",l%Lcell);
		this->precalc_TwoSiteData();
		this->HERMITIAN = true;
		this->HAMILTONIAN = true;
	};
	
	PeierlsHubbardU1xU1 (const size_t &L, const vector<Param> &params, const BC &boundary=BC::OPEN, const DMRG::VERBOSITY::OPTION &VERB=DMRG::VERBOSITY::OPTION::ON_EXIT);
	
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
	
	static qarray<2> singlet (int N=0) {return qarray<2>{0,N};};
	static constexpr MODEL_FAMILY FAMILY = HUBBARD;
	static constexpr int spinfac = 2;
	
	static const map<string,any> defaults;
	static const map<string,any> sweep_defaults;
};

// V is standard next-nearest neighbour density interaction
// Vz and Vxy are anisotropic isospin-isospin next-nearest neighbour interaction
const map<string,any> PeierlsHubbardU1xU1::defaults = 
{
	{"t",1.+0.i}, {"tPrime",0.i}, {"tRung",1.+0.i}, {"tPrimePrime",0.i},
	{"mu",0.}, {"t0",0.}, 
	{"U",0.}, {"Uph",0.},
	{"V",0.}, {"Vext",0.}, {"Vrung",0.},
	{"Bz",0.},
	{"Vz",0.}, {"Vzrung",0.}, {"Vxy",0.}, {"Vxyrung",0.}, 
	{"J",0.}, {"Jperp",0.},
	{"X",0.}, {"Xrung",0.},
	{"REMOVE_DOUBLE",false}, {"REMOVE_EMPTY",false}, {"REMOVE_UP",false}, {"REMOVE_DN",false}, {"mfactor",1}, {"k",0},
	{"maxPower",2ul}, {"CYLINDER",false}, {"Ly",1ul}
};

const map<string,any> PeierlsHubbardU1xU1::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.}, {"lim_alpha",11ul}, {"eps_svd",1e-7},
	{"Mincr_abs", 50ul}, {"Mincr_per", 2ul}, {"Mincr_rel", 1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",24ul}, {"min_halfsweeps",1ul},
	{"Minit",2ul}, {"Qinit",2ul}, {"Mlimit",1000ul},
	{"tol_eigval",1e-7}, {"tol_state",1e-6},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST",DMRG::CONVTEST::VAR_2SITE}
};

PeierlsHubbardU1xU1::
PeierlsHubbardU1xU1 (const size_t &L, const vector<Param> &params, const BC &boundary, const DMRG::VERBOSITY::OPTION &VERB)
:Mpo<Symmetry,complex<double>> (L, Symmetry::qvacuum(), "", PROP::HERMITIAN, PROP::NON_UNITARY, boundary, VERB),
 HubbardObservables(L,params,PeierlsHubbardU1xU1::defaults),
 ParamReturner(PeierlsHubbardU1xU1::sweep_defaults)
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
void PeierlsHubbardU1xU1::
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
					complex<double> total_value = factor[j] * value;
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
			SiteOperatorQ<Symmetry_,MatrixType> cUP_sign_local    = (F[loc].c(UP,0)    * F[loc].sign()).template cast<complex<double>>();
			SiteOperatorQ<Symmetry_,MatrixType> cDN_sign_local    = (F[loc].c(DN,0)    * F[loc].sign()).template cast<complex<double>>();
			SiteOperatorQ<Symmetry_,MatrixType> cdagUP_sign_local = (F[loc].cdag(UP,0) * F[loc].sign()).template cast<complex<double>>();
			SiteOperatorQ<Symmetry_,MatrixType> cdagDN_sign_local = (F[loc].cdag(DN,0) * F[loc].sign()).template cast<complex<double>>();
			
			vector<SiteOperatorQ<Symmetry_,MatrixType> > cUP_ranges(N_sites);    for (size_t i=0; i<N_sites; ++i) {cUP_ranges[i]    = F[i].c(UP,0).template cast<complex<double>>();}
			vector<SiteOperatorQ<Symmetry_,MatrixType> > cdagUP_ranges(N_sites); for (size_t i=0; i<N_sites; ++i) {cdagUP_ranges[i] = F[i].cdag(UP,0).template cast<complex<double>>();}
			vector<SiteOperatorQ<Symmetry_,MatrixType> > cDN_ranges(N_sites);    for (size_t i=0; i<N_sites; ++i) {cDN_ranges[i]    = F[i].c(DN,0).template cast<complex<double>>();}
			vector<SiteOperatorQ<Symmetry_,MatrixType> > cdagDN_ranges(N_sites); for (size_t i=0; i<N_sites; ++i) {cdagDN_ranges[i] = F[i].cdag(DN,0).template cast<complex<double>>();}
			
			vector<SiteOperatorQ<Symmetry_,MatrixType> >          frst {cdagUP_sign_local, cUP_sign_local, cdagDN_sign_local, cDN_sign_local};
			vector<vector<SiteOperatorQ<Symmetry_,MatrixType> > > last {cUP_ranges, cdagUP_ranges, cDN_ranges, cdagDN_ranges};
			push_full("tFull", "tᵢⱼ", frst, last, {-1., +1., -1., +1.}, {false, true, false, true}, PROP::FERMIONIC);
		}
		
		if (P.HAS("Vzfull"))
		{
			vector<SiteOperatorQ<Symmetry_,MatrixType> > first {F[loc].Tz(0).template cast<complex<double>>()};
			vector<SiteOperatorQ<Symmetry_,MatrixType> > Tz_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++)
			{
				Tz_ranges[i] = F[i].Tz(0).template cast<complex<double>>();
			}
			
			vector<vector<SiteOperatorQ<Symmetry_,MatrixType> > > last {Tz_ranges};
			push_full("Vzfull", "Vzᵢⱼ", first, last, {1.}, {false}, PROP::BOSONIC);
		}
		
		if (P.HAS("vzfull"))
		{
			vector<SiteOperatorQ<Symmetry_,MatrixType> > first {F[loc].tz(0).template cast<complex<double>>()};
			vector<SiteOperatorQ<Symmetry_,MatrixType> > tz_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++)
			{
				tz_ranges[i] = F[i].tz(0).template cast<complex<double>>();
			}
			
			vector<vector<SiteOperatorQ<Symmetry_,MatrixType> > > last {tz_ranges};
			push_full("vzfull", "vzᵢⱼ", first, last, {1.}, {false}, PROP::BOSONIC);
		}
		
		if (P.HAS("VextFull"))
		{
			vector<SiteOperatorQ<Symmetry_,MatrixType> > first {F[loc].n(0).template cast<complex<double>>()};
			vector<SiteOperatorQ<Symmetry_,MatrixType> > n_ranges(N_sites); for (size_t i=0; i<N_sites; i++) {n_ranges[i] = F[i].n(0).template cast<complex<double>>();}
			vector<vector<SiteOperatorQ<Symmetry_,MatrixType> > > last {n_ranges};
			push_full("VextFull", "Vextᵢⱼ", first, last, {1.}, {false}, PROP::BOSONIC);
		}
		
		if (P.HAS("Jfull"))
		{
			vector<SiteOperatorQ<Symmetry_,MatrixType> > first {F[loc].Sp(0).template cast<complex<double>>(), 
			                                                         F[loc].Sm(0).template cast<complex<double>>(),
			                                                         F[loc].Sz(0).template cast<complex<double>>()};
			vector<SiteOperatorQ<Symmetry_,MatrixType> > Sp_ranges(N_sites);
			vector<SiteOperatorQ<Symmetry_,MatrixType> > Sm_ranges(N_sites);
			vector<SiteOperatorQ<Symmetry_,MatrixType> > Sz_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++)
			{
				Sp_ranges[i] = F[i].Sp(0).template cast<complex<double>>();
				Sm_ranges[i] = F[i].Sm(0).template cast<complex<double>>();
				Sz_ranges[i] = F[i].Sz(0).template cast<complex<double>>();
			}
			
			vector<vector<SiteOperatorQ<Symmetry_,MatrixType> > > last {Sm_ranges, Sp_ranges, Sz_ranges};
			push_full("Jfull", "Jᵢⱼ", first, last, {0.5,0.5,1.}, {false,false,false}, PROP::BOSONIC);
		}
		
		// Local terms: U, t0, μ, t⟂, V⟂, J⟂
		param1d U = P.fill_array1d<double>("U", "Uorb", orbitals, loc%Lcell);
		param1d Uph = P.fill_array1d<double>("Uph", "Uphorb", orbitals, loc%Lcell);
		param1d t0 = P.fill_array1d<double>("t0", "t0orb", orbitals, loc%Lcell);
		param1d mu = P.fill_array1d<double>("mu", "muorb", orbitals, loc%Lcell);
		param1d Bz = P.fill_array1d<double>("Bz", "Bzorb", orbitals, loc%Lcell);
		param2d tperp = P.fill_array2d<complex<double>>("tRung", "t", "tPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Vperp = P.fill_array2d<double>("VRung", "V", "VPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Vzperp = P.fill_array2d<double>("VzRung", "Vz", "VzPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Vxyperp = P.fill_array2d<double>("VxyRung", "Vxy", "VxyPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Jperp = P.fill_array2d<double>("JRung", "J", "JPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		
		labellist[loc].push_back(U.label);
		labellist[loc].push_back(Uph.label);
		labellist[loc].push_back(t0.label);
		labellist[loc].push_back(mu.label);
		labellist[loc].push_back(Bz.label);
		labellist[loc].push_back(tperp.label);
		labellist[loc].push_back(Vperp.label);
		labellist[loc].push_back(Vzperp.label);
		labellist[loc].push_back(Vxyperp.label);
		labellist[loc].push_back(Jperp.label);
		ArrayXd C_array = F[loc].ZeroField();
		
		auto Hloc = Mpo<Symmetry_,complex<double>>::get_N_site_interaction
		(
			F[loc].template HubbardHamiltonian<complex<double>,Symmetry_>(U.a.cast<complex<double>>(), 
			                                                              Uph.a.cast<complex<double>>(),
			                                                              (t0.a-mu.a).cast<complex<double>>(),
			                                                              Bz.a.cast<complex<double>>(),
			                                                              tperp.a,
			                                                              Vperp.a.cast<complex<double>>(),
			                                                              Vzperp.a.cast<complex<double>>(),
			                                                              Vxyperp.a.cast<complex<double>>(),
			                                                              Jperp.a.cast<complex<double>>(),
			                                                              Jperp.a.cast<complex<double>>(),
			                                                              C_array.cast<complex<double>>())
		);
		pushlist.push_back(std::make_tuple(loc, Hloc, 1.+0.i));
	}
}

} // end namespace VMPS::models

#endif
