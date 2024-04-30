#ifndef STRAWBERRY_HEISENBERGU1
#define STRAWBERRY_HEISENBERGU1

//include <array>

//include "Mpo.h"
#include "symmetry/U1.h"
#include "models/HeisenbergObservables.h"
//include "bases/SpinBase.h"
//include "DmrgExternal.h"
//include "ParamHandler.h" // from HELPERS
//include "symmetry/kind_dummies.h"
#include "ParamReturner.h"
#include "Geometry2D.h" // from TOOLS

namespace VMPS
{

/** \class HeisenbergU1
  * \ingroup Heisenberg
  *
  * \brief Heisenberg Model
  *
  * MPO representation of
  * \f[
  * H =  J \sum_{<ij>} \left(\mathbf{S_i} \cdot \mathbf{S_j}\right) 
  *	 +J' \sum_{<<ij>>} \left(\mathbf{S_i} \cdot \mathbf{S_j}\right)
  *	 -B_z \sum_i S^z_i
  *	 +K_z \sum_i \left(S^z_i\right)^2
  *	 -D_y \sum_{<ij>} \left(\mathbf{S_i} \times \mathbf{S_j}\right)_y
  *	 -D_y' \sum_{<<ij>>} \left(\mathbf{S_i} \times \mathbf{S_j}\right)_y
  * \f]
  *
  * \param D : \f$D=2S+1\f$ where \f$S\f$ is the spin
  * \note Makes use of the \f$S^z\f$ U(1) symmetry.
  * \note The default variable settings can be seen in \p HeisenbergU1::defaults.
  * \note \f$J>0\f$ is antiferromagnetic
  * \note Isotropic \f$J\f$ is required here. For XXZ coupling, use VMPS::HeisenbergU1XXZ.
  */
class HeisenbergU1 : public Mpo<Sym::U1<Sym::SpinU1>,double>, public HeisenbergObservables<Sym::U1<Sym::SpinU1> >, public ParamReturner
{
public:
	
	typedef Sym::U1<Sym::SpinU1> Symmetry;
	MAKE_TYPEDEFS(HeisenbergU1)
	
	static qarray<1> singlet (int N=0) {return qarray<1>{0};};
	static constexpr MODEL_FAMILY FAMILY = HEISENBERG;
	
public:
	typedef Symmetry::qType qType;
	typedef SiteOperator<Symmetry,SparseMatrix<double> > OperatorType;
	
public:
	
	///@{
	HeisenbergU1() : Mpo<Symmetry>(), ParamReturner(HeisenbergU1::sweep_defaults) {};
	
	HeisenbergU1(Mpo<Symmetry> &Mpo_input, const vector<Param> &params)
	:Mpo<Symmetry>(Mpo_input),
	 HeisenbergObservables(this->N_sites,params,HeisenbergU1::defaults),
	 ParamReturner(HeisenbergU1::sweep_defaults)
	{
		ParamHandler P(params,HeisenbergU1::defaults);
		size_t Lcell = P.size();
		N_phys = 0;
		for (size_t l=0; l<N_sites; ++l) N_phys += P.get<size_t>("Ly",l%Lcell);
		this->precalc_TwoSiteData();
		this->HERMITIAN = true;
		this->HAMILTONIAN = true;
	};
	
	HeisenbergU1 (const size_t &L, const BC &boundary=BC::OPEN, const DMRG::VERBOSITY::OPTION &VERB=DMRG::VERBOSITY::OPTION::ON_EXIT);
	
	HeisenbergU1 (const size_t &L, const vector<Param> &params, const BC & boundary=BC::OPEN, const DMRG::VERBOSITY::OPTION &VERB=DMRG::VERBOSITY::OPTION::ON_EXIT);
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
	static void set_operators (const std::vector<SpinBase<Symmetry_> > &B, const ParamHandler &P, PushType<SiteOperator<Symmetry_,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary=BC::OPEN);
	
	/**
	 * Validates whether a given total quantum number \p qnum is a possible target quantum number for an Mps.
	 * \returns \p true if valid, \p false if not
	 */
	bool validate (qarray<1> qnum) const;
	
	static const std::map<string,std::any> defaults;
	static const std::map<string,std::any> sweep_defaults;
};

const std::map<string,std::any> HeisenbergU1::defaults = 
{
	{"J",0.}, {"Jprime",0.}, {"Jrung",0.},
	{"Jxy",0.}, {"Jxyprime",0.}, {"Jxyrung",0.},
	{"Jz",0.}, {"Jzprime",0.}, {"Jzrung",0.},
	{"R",0.},
	{"Dy",0.}, {"Dyprime",0.}, {"Dyrung",0.},
	{"Bz",0.}, {"Kz",0.},
	{"mu",0.}, {"nu",0.}, // couple to Sz_i-1/2 and Sz_i+1/2
	{"D",2ul}, {"maxPower",2ul}, {"CYLINDER",false}, {"Ly",1ul}, {"mfactor",1}
};

const std::map<string,std::any> HeisenbergU1::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.e-11}, {"lim_alpha",10ul}, {"eps_svd",1.e-7},
	{"Mincr_abs", 50ul}, {"Mincr_per", 2ul}, {"Mincr_rel", 1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",20ul}, {"min_halfsweeps",4ul},
	{"Minit",1ul}, {"Qinit",1ul}, {"Mlimit",1000ul},
	{"tol_eigval",1e-7}, {"tol_state",1e-6},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
};

HeisenbergU1::
HeisenbergU1 (const size_t &L, const BC &boundary, const DMRG::VERBOSITY::OPTION &VERB)
:Mpo<Symmetry> (L, qarray<Symmetry::Nq>({0}), "", PROP::HERMITIAN, PROP::NON_UNITARY, boundary,VERB),
HeisenbergObservables(L),
ParamReturner(HeisenbergU1::sweep_defaults)
{}

HeisenbergU1::
HeisenbergU1 (const size_t &L, const vector<Param> &params, const BC &boundary, const DMRG::VERBOSITY::OPTION &VERB)
:Mpo<Symmetry> (L, qarray<Symmetry::Nq>({0}), "", PROP::HERMITIAN, PROP::NON_UNITARY, boundary, VERB),
 HeisenbergObservables(L,params,HeisenbergU1::defaults),
 ParamReturner(HeisenbergU1::sweep_defaults)
{
	ParamHandler P(params,defaults);
	size_t Lcell = P.size();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(B[l].get_basis().qloc(),l);
	}
	
	if (P.HAS_ANY_OF({"Jxy", "Jxypara", "Jxyperp", "Jxyfull"}))
	{
		this->set_name("XXZ");
	}
	else if (P.HAS_ANY_OF({"Jz", "Jzpara", "Jzperp", "Jzfull"}))
	{
		this->set_name("Ising");
	}
	else
	{
		this->set_name("Heisenberg");
	}
	
	PushType<SiteOperator<Symmetry,double>,double> pushlist;
	std::vector<std::vector<std::string>> labellist;
	set_operators(B, P, pushlist, labellist, boundary);
	
	this->construct_from_pushlist(pushlist, labellist, Lcell);
	this->finalize(PROP::COMPRESS, P.get<size_t>("maxPower"));
	
	this->precalc_TwoSiteData();
}

bool HeisenbergU1::
validate (qarray<1> qnum) const
{
	frac Smax(0,1);
	frac q_in(qnum[0],2);
	for (size_t l=0; l<N_sites; ++l) { Smax+=frac(B[l].get_D()-1,2); }
	if (Smax.denominator()==q_in.denominator() and q_in <= Smax) {return true;}
	else {return false;}
}

template<typename Symmetry_>
void HeisenbergU1::
set_operators (const std::vector<SpinBase<Symmetry_> > &B, const ParamHandler &P, PushType<SiteOperator<Symmetry_,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = B.size();
	
	if (labellist.size() != N_sites) {labellist.resize(N_sites);}
	
	for (std::size_t loc=0; loc<N_sites; ++loc)
	{
		size_t lp1 = (loc+1)%N_sites;
		size_t lp2 = (loc+2)%N_sites;
		
		std::size_t orbitals = B[loc].orbitals();
		std::size_t next_orbitals = B[lp1].orbitals();
		std::size_t nextn_orbitals = B[lp2].orbitals();
		
		stringstream ss1, ss2;
		ss1 << "S=" << print_frac_nice(frac(P.get<size_t>("D",loc%Lcell)-1,2));
		ss2 << "Ly=" << P.get<size_t>("Ly",loc%Lcell);
		labellist[loc].push_back(ss1.str());
		labellist[loc].push_back(ss2.str());
		
		auto push_full = [&N_sites, &loc, &B, &P, &pushlist, &labellist, &boundary] 
		(string xxxFull, string label,
		 const vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > &first,
		 const vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > &last,
		 vector<double> factor) -> void
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
						ops[i] = B[(loc+i)%N_sites].Id();
					}
					ops[range] = last[j][(loc+range)%N_sites];
					pushlist.push_back(std::make_tuple(loc, ops, factor[j] * value));
				}
			}
			
			stringstream ss;
			ss << label << "(" << Geometry2D::hoppingInfo(Full) << ")";
			labellist[loc].push_back(ss.str());
		};
		
		// Local terms: B, K and J⟂
		
		param1d Bz = P.fill_array1d<double>("Bz", "Bzorb", orbitals, loc%Lcell);
		param1d Kz = P.fill_array1d<double>("Kz", "Kzorb", orbitals, loc%Lcell);
		param2d Jperp = P.fill_array2d<double>("Jrung", "J", "Jperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Jxyperp = P.fill_array2d<double>("Jxyrung", "Jxy", "Jxyperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Jzperp  = P.fill_array2d<double>("Jzrung",  "Jz",  "Jzperp",  orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param1d mu = P.fill_array1d<double>("mu", "muorb", orbitals, loc%Lcell);
		param1d nu = P.fill_array1d<double>("nu", "nuorb", orbitals, loc%Lcell);
		
		labellist[loc].push_back(Bz.label);
		labellist[loc].push_back(Kz.label);
		labellist[loc].push_back(Jperp.label);
		labellist[loc].push_back(Jxyperp.label);
		labellist[loc].push_back(Jzperp.label);
		labellist[loc].push_back(mu.label);
		labellist[loc].push_back(nu.label);
		
		Eigen::ArrayXd Bx_array = B[loc].ZeroField();
		Eigen::ArrayXd Kx_array = B[loc].ZeroField();
		Eigen::ArrayXXd Dyperp_array = B[loc].ZeroHopping();
		
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
		
		auto Hloc = Mpo<Symmetry,double>::get_N_site_interaction(
		            B[loc].HeisenbergHamiltonian(sum_array(Jperp.a,Jxyperp.a), sum_array(Jperp.a,Jzperp.a), Bz.a, mu.a, nu.a, Kz.a));
		pushlist.push_back(std::make_tuple(loc, Hloc, 1.));
		
		// Full J-matrices
		if (P.HAS("Jfull"))
		{
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first {B[loc].Sp(0), B[loc].Sm(0), B[loc].Sz(0)};
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > Sp_ranges(N_sites);
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > Sm_ranges(N_sites);
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > Sz_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++) {Sp_ranges[i] = B[i].Sp(0); Sm_ranges[i] = B[i].Sm(0); Sz_ranges[i] = B[i].Sz(0);}
			
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {Sm_ranges, Sp_ranges, Sz_ranges};
			push_full("Jfull", "Jᵢⱼ", first, last, {0.5,0.5,1.0});
		}
		if (P.HAS("Jxyfull"))
		{
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first {B[loc].Sp(0), B[loc].Sm(0)};
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > Sp_ranges(N_sites);
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > Sm_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++) {Sp_ranges[i] = B[i].Sp(0); Sm_ranges[i] = B[i].Sm(0);}
			
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {Sm_ranges, Sp_ranges};
			push_full("Jxyfull", "Jxyᵢⱼ", first, last, {0.5,0.5});
		}
		if (P.HAS("Jzfull"))
		{
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first {B[loc].Sz(0)};
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > Sz_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++) {Sz_ranges[i] = B[i].Sz(0);}
			
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {Sz_ranges};
			push_full("Jzfull", "Jzᵢⱼ", first, last, {1.0});
		}
		if (P.HAS("JxyfullA"))
		{
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first {B[loc].Sp(1), B[loc].Sm(1)};
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > Sp_ranges(N_sites);
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > Sm_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++) {Sp_ranges[i] = B[i].Sp(1); 
			                                  Sm_ranges[i] = B[i].Sm(1);}
			
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {Sm_ranges, Sp_ranges};
			push_full("JxyfullA", "JxyAᵢⱼ", first, last, {0.5,0.5});
		}
		if (P.HAS("JzfullA"))
		{
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first {B[loc].Sz(1)};
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > Sz_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++) {Sz_ranges[i] = B[i].Sz(1);}
			
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {Sz_ranges};
			push_full("JzfullA", "JzAᵢⱼ", first, last, {1.0});
		}
		if (P.HAS("Rfull"))
		{
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > first {B[loc].Qp(0), B[loc].Qm(0), B[loc].Qz(0), B[loc].Qpz(0), B[loc].Qmz(0)};
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > Qp_ranges(N_sites);
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > Qm_ranges(N_sites);
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > Qz_ranges(N_sites);
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > Qpz_ranges(N_sites);
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > Qmz_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++)
			{
				Qp_ranges[i] = B[i].Qp(0);
				Qm_ranges[i] = B[i].Qm(0);
				Qz_ranges[i] = B[i].Qz(0);
				Qpz_ranges[i] = B[i].Qpz(0);
				Qmz_ranges[i] = B[i].Qmz(0);
			}
			
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXd> > > last {Qm_ranges, Qp_ranges, Qz_ranges, Qmz_ranges, Qpz_ranges};
			push_full("Rfull", "Rᵢⱼ", first, last, {0.5, 0.5, 1.0, 0.5, 0.5});
		}
		
		if (P.HAS("Jfull") or P.HAS("Jxyfull") or P.HAS("Jzfull") or P.HAS("Rfull")) continue;
		
		// Nearest-neighbour terms: Jxy, Jz, J
		param2d Jxypara = P.fill_array2d<double>("Jxy", "Jxypara", {orbitals, next_orbitals}, loc%Lcell);
		param2d Jzpara  = P.fill_array2d<double>("Jz",  "Jzpara",  {orbitals, next_orbitals}, loc%Lcell);
		param2d Jpara = P.fill_array2d<double>("J", "Jpara", {orbitals, next_orbitals}, loc%Lcell);
		labellist[loc].push_back(Jxypara.label);
		labellist[loc].push_back(Jzpara.label);
		labellist[loc].push_back(Jpara.label);
		
		if (loc < N_sites-1 or !static_cast<bool>(boundary))
		{
			for (std::size_t alfa=0; alfa < orbitals; ++alfa)
			for (std::size_t beta=0; beta < next_orbitals; ++beta)
			{
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(B[loc].Scomp(SP,alfa),
				                                                                                     B[lp1].Scomp(SM,beta)),
				                                                                                     0.5*Jxypara(alfa,beta)+0.5*Jpara(alfa,beta)));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(B[loc].Scomp(SM,alfa),
				                                                                                     B[lp1].Scomp(SP,beta)),
				                                                                                     0.5*Jxypara(alfa,beta)+0.5*Jpara(alfa,beta)));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(B[loc].Scomp(SZ,alfa),
				                                                                                     B[lp1].Scomp(SZ,beta)),
				                                                                                     Jzpara(alfa,beta)+Jpara(alfa,beta)));
			}
		}
		
		// Nearest-neighbour terms: R
		param2d Rpara = P.fill_array2d<double>("R", "Rpara", {orbitals, next_orbitals}, loc%Lcell);
		labellist[loc].push_back(Rpara.label);
		
		if (loc < N_sites-1 or !static_cast<bool>(boundary))
		{
			for (std::size_t alfa=0; alfa < orbitals; ++alfa)
			for (std::size_t beta=0; beta < next_orbitals; ++beta)
			{
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(B[loc].Qp(alfa),
				                                                                                     B[lp1].Qm(beta)),
				                                                                                     0.5*Rpara(alfa,beta)));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(B[loc].Qm(alfa),
				                                                                                     B[lp1].Qp(beta)),
				                                                                                     0.5*Rpara(alfa,beta)));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(B[loc].Qpz(alfa),
				                                                                                     B[lp1].Qmz(beta)),
				                                                                                     0.5*Rpara(alfa,beta)));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(B[loc].Qmz(alfa),
				                                                                                     B[lp1].Qpz(beta)),
				                                                                                     0.5*Rpara(alfa,beta)));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(B[loc].Qz(alfa),
				                                                                                     B[lp1].Qz(beta)),
				                                                                                     Rpara(alfa,beta)));
			}
		}
		
		// Next-nearest-neighbour terms: Jxy, Jz, J
		param2d Jxyprime = P.fill_array2d<double>("Jxyprime", "Jxyprime_array", {orbitals, nextn_orbitals}, loc%Lcell);
		param2d Jzprime  = P.fill_array2d<double>("Jzprime",  "Jzprime_array",  {orbitals, nextn_orbitals}, loc%Lcell);
		param2d Jprime = P.fill_array2d<double>("Jprime", "Jprime_array", {orbitals, nextn_orbitals}, loc%Lcell);
		labellist[loc].push_back(Jxyprime.label);
		labellist[loc].push_back(Jzprime.label);
		labellist[loc].push_back(Jprime.label);
		
		if (loc < N_sites-2 or !static_cast<bool>(boundary))
		{
			for (std::size_t alfa=0; alfa < orbitals; ++alfa)
			for (std::size_t beta=0; beta < nextn_orbitals; ++beta)
			{
				pushlist.push_back(std::make_tuple(loc,Mpo<Symmetry,double>::get_N_site_interaction(B[loc].Scomp(SP,alfa),
				                                                                                    B[lp1].Id(),
				                                                                                    B[lp2].Scomp(SM,beta)),
				                                                                                    0.5*Jxyprime(alfa,beta)+0.5*Jprime(alfa,beta)));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(B[loc].Scomp(SM,alfa),
				                                                                                     B[lp1].Id(),
				                                                                                     B[lp2].Scomp(SP,beta)),
				                                                                                     0.5*Jxyprime(alfa,beta)+0.5*Jprime(alfa,beta)));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(B[loc].Scomp(SZ,alfa),
				                                                                                     B[lp1].Id(),
				                                                                                     B[lp2].Scomp(SZ,beta)),
				                                                                                     Jzprime(alfa,beta)+Jprime(alfa,beta)));
			}
		}
	}
}

} //end namespace VMPS

#endif
