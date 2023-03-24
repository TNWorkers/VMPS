#ifndef STRAWBERRY_HeisenbergU1Complex
#define STRAWBERRY_HeisenbergU1Complex

#include "symmetry/U1.h"
#include "models/HeisenbergU1.h"
#include "models/HeisenbergObservables.h"
#include "ParamReturner.h"
#include "Geometry2D.h" // from TOOLS

namespace VMPS
{
class HeisenbergU1Complex : public Mpo<Sym::U1<Sym::SpinU1>,complex<double> >, public HeisenbergObservables<Sym::U1<Sym::SpinU1>,complex<double> >, public ParamReturner
{
public:
	
	typedef Sym::U1<Sym::SpinU1> Symmetry;
	MAKE_TYPEDEFS(HeisenbergU1Complex)
	typedef complex<double> Scalar;
	
	static qarray<1> singlet (int N=0) {return qarray<1>{0};};
	static constexpr MODEL_FAMILY FAMILY = HEISENBERG;
	
public:
	typedef Symmetry::qType qType;
	typedef SiteOperator<Symmetry,SparseMatrix<complex<double> > > OperatorType;
	
public:
	
	///@{
	HeisenbergU1Complex() : Mpo<Symmetry,complex<double> >(), ParamReturner(HeisenbergU1Complex::sweep_defaults) {};
	
	HeisenbergU1Complex(Mpo<Symmetry,complex<double> > &Mpo_input, const vector<Param> &params)
	:Mpo<Symmetry,complex<double> >(Mpo_input),
	 HeisenbergObservables(this->N_sites,params,HeisenbergU1Complex::defaults),
	 ParamReturner(HeisenbergU1Complex::sweep_defaults)
	{
		ParamHandler P(params,HeisenbergU1Complex::defaults);
		size_t Lcell = P.size();
		N_phys = 0;
		for (size_t l=0; l<N_sites; ++l) N_phys += P.get<size_t>("Ly",l%Lcell);
		this->precalc_TwoSiteData();
		this->HERMITIAN = true;
		this->HAMILTONIAN = true;
	};
	
	HeisenbergU1Complex (const size_t &L, const BC &boundary=BC::OPEN, const DMRG::VERBOSITY::OPTION &VERB=DMRG::VERBOSITY::OPTION::ON_EXIT);
	
	HeisenbergU1Complex (const size_t &L, const vector<Param> &params, const BC & boundary=BC::OPEN, const DMRG::VERBOSITY::OPTION &VERB=DMRG::VERBOSITY::OPTION::ON_EXIT);
	
	template<typename Symmetry_>
	static void set_operators (const std::vector<SpinBase<Symmetry_> > &B, const ParamHandler &P, 
	                           PushType<SiteOperator<Symmetry_,complex<double> >,complex<double> >& pushlist, 
	                           std::vector<std::vector<std::string>>& labellist, 
	                           const BC boundary=BC::OPEN);
	///@}
	
	/**
	 * Validates whether a given total quantum number \p qnum is a possible target quantum number for an Mps.
	 * \returns \p true if valid, \p false if not
	 */
	bool validate (qarray<1> qnum) const;
	
	static const std::map<string,std::any> defaults;
	static const std::map<string,std::any> sweep_defaults;
};

const std::map<string,std::any> HeisenbergU1Complex::defaults = 
{
	{"J",0.}, {"Jprime",0.}, {"Jrung",0.},
	{"Jxy",0.}, {"Jxyprime",0.}, {"Jxyrung",0.},
	{"Jxy3site",0.}, {"Jxy4site",0.},
	{"Jz",0.}, {"Jzprime",0.}, {"Jzrung",0.},
	{"R",0.},
	{"Dy",0.}, {"Dyprime",0.}, {"Dyrung",0.},
	{"Bz",0.}, {"Kz",0.},
	{"mu",0.}, {"nu",0.}, // couple to Sz_i-1/2 and Sz_i+1/2
	{"D",2ul}, {"maxPower",2ul}, {"CYLINDER",false}, {"Ly",1ul}
};

const std::map<string,std::any> HeisenbergU1Complex::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.e-11}, {"lim_alpha",10ul}, {"eps_svd",1.e-7},
	{"Mincr_abs", 50ul}, {"Mincr_per", 2ul}, {"Mincr_rel", 1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",20ul}, {"min_halfsweeps",4ul},
	{"Minit",1ul}, {"Qinit",1ul}, {"Mlimit",1000ul},
	{"tol_eigval",1e-7}, {"tol_state",1e-6},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
};

HeisenbergU1Complex::
HeisenbergU1Complex (const size_t &L, const BC &boundary, const DMRG::VERBOSITY::OPTION &VERB)
:Mpo<Symmetry,complex<double> > (L, qarray<Symmetry::Nq>({0}), "", PROP::HERMITIAN, PROP::NON_UNITARY, boundary,VERB),
HeisenbergObservables(L),
ParamReturner(HeisenbergU1Complex::sweep_defaults)
{}

HeisenbergU1Complex::
HeisenbergU1Complex (const size_t &L, const vector<Param> &params, const BC &boundary, const DMRG::VERBOSITY::OPTION &VERB)
:Mpo<Symmetry,complex<double> > (L, qarray<Symmetry::Nq>({0}), "", PROP::HERMITIAN, PROP::NON_UNITARY, boundary, VERB),
 HeisenbergObservables(L,params,HeisenbergU1Complex::defaults),
 ParamReturner(HeisenbergU1Complex::sweep_defaults)
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
		this->set_name("XXZcomplex");
	}
	else if (P.HAS_ANY_OF({"Jz", "Jzpara", "Jzperp", "Jzfull"}))
	{
		this->set_name("IsingComplex");
	}
	else
	{
		this->set_name("HeisenbergComplex");
	}
	
	PushType<SiteOperator<Symmetry,complex<double> >,complex<double> > pushlist;
	std::vector<std::vector<std::string>> labellist;
	HeisenbergU1Complex::set_operators(B, P, pushlist, labellist, boundary);
	
	this->construct_from_pushlist(pushlist, labellist, Lcell);
	this->finalize(PROP::COMPRESS, P.get<size_t>("maxPower"));
	
	this->precalc_TwoSiteData();
}

template<typename Symmetry_>
void HeisenbergU1Complex::
set_operators (const std::vector<SpinBase<Symmetry_> > &B, const ParamHandler &P, PushType<SiteOperator<Symmetry_,complex<double> >,complex<double> >& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = B.size();
	
	if (labellist.size() != N_sites) {labellist.resize(N_sites);}
	
	for (std::size_t loc=0; loc<N_sites; ++loc)
	{
		size_t lp1 = (loc+1)%N_sites;
		size_t lp2 = (loc+2)%N_sites;
		size_t lp3 = (loc+3)%N_sites;
		size_t lp4 = (loc+4)%N_sites;
		
		std::size_t orbitals = B[loc].orbitals();
		std::size_t next_orbitals = B[lp1].orbitals();
		std::size_t nextn_orbitals = B[lp2].orbitals();
		std::size_t next3_orbitals = B[lp3].orbitals();
		std::size_t next4_orbitals = B[lp4].orbitals();
		
		stringstream ss1, ss2;
		ss1 << "S=" << print_frac_nice(frac(P.get<size_t>("D",loc%Lcell)-1,2));
		ss2 << "Ly=" << P.get<size_t>("Ly",loc%Lcell);
		labellist[loc].push_back(ss1.str());
		labellist[loc].push_back(ss2.str());
		
		param2d Jxyperp = P.fill_array2d<double>("Jxyrung", "Jxy", "Jxyperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Jzperp  = P.fill_array2d<double>("Jzrung",  "Jz",  "Jzperp",  orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Jperp  = P.fill_array2d<double>("Jrung",  "J",  "Jperp",  orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		
		param1d Bz = P.fill_array1d<double>("Bz", "Bzorb", orbitals, loc%Lcell);
		param1d mu = P.fill_array1d<double>("mu", "muorb", orbitals, loc%Lcell);
		param1d nu = P.fill_array1d<double>("nu", "nuorb", orbitals, loc%Lcell);
		param1d Kz = P.fill_array1d<double>("Kz", "Kzorb", orbitals, loc%Lcell);
		
		labellist[loc].push_back(Bz.label);
		labellist[loc].push_back(mu.label);
		labellist[loc].push_back(nu.label);
		labellist[loc].push_back(Kz.label);
		
		labellist[loc].push_back(Jxyperp.label);
		labellist[loc].push_back(Jzperp.label);
		labellist[loc].push_back(Jperp.label);
		
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
		
		auto Hloc = Mpo<Symmetry,complex<double> >::get_N_site_interaction(
		            B[loc].HeisenbergHamiltonian(Jxyperp.a+Jperp.a, Jzperp.a+Jperp.a, Bz.a, mu.a, nu.a, Kz.a).template cast<complex<double> >());
		pushlist.push_back(std::make_tuple(loc, Hloc, 1.+0.i));
		
//		// Nearest-neighbour terms: Jxy, Jz, J
//		param2d Jxypara = P.fill_array2d<complex<double> >("Jxy", "Jxypara", {orbitals, next_orbitals}, loc%Lcell);
//		param2d Jzpara  = P.fill_array2d<complex<double> >("Jz",  "Jzpara",  {orbitals, next_orbitals}, loc%Lcell);
//		labellist[loc].push_back(Jxypara.label);
//		labellist[loc].push_back(Jzpara.label);
//		
//		if (loc < N_sites-1 or !static_cast<bool>(boundary))
//		{
//			for (std::size_t alfa=0; alfa < orbitals; ++alfa)
//			for (std::size_t beta=0; beta < next_orbitals; ++beta)
//			{
//				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,complex<double> >::get_N_site_interaction(B[loc].Scomp(SP,alfa),
//				                                                                                     B[lp1].Scomp(SM,beta)),
//				                                                                                     0.5*Jxypara(alfa,beta)));
//				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,complex<double> >::get_N_site_interaction(B[loc].Scomp(SM,alfa),
//				                                                                                     B[lp1].Scomp(SP,beta)),
//				                                                                                     0.5*Jxypara(alfa,beta)));
//				
//				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,complex<double> >::get_N_site_interaction(B[loc].Scomp(SZ,alfa),
//				                                                                                     B[lp1].Scomp(SZ,beta)),
//				                                                                                     Jzpara(alfa,beta)));
//			}
//		}
		
		auto push_full = [&N_sites, &loc, &B, &P, &pushlist, &labellist, &boundary] 
		(string xxxFull, string label,
		 const vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> > &first,
		 const vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> > > &last,
		 vector<complex<double> > factor,
		 vector<bool> CONJ) -> void
		{
			ArrayXXcd Full = P.get<Eigen::ArrayXXcd>(xxxFull);
			vector<vector<std::pair<size_t,complex<double> > > > R = Geometry2D::rangeFormat(Full);
			
			if (static_cast<bool>(boundary)) {assert(R.size() ==   N_sites and "Use an (N_sites)x(N_sites) hopping matrix for open BC!");}
			else                             {assert(R.size() >= 2*N_sites and "Use at least a (2*N_sites)x(N_sites) hopping matrix for infinite BC!");}
			
			for (size_t j=0; j<first.size(); j++)
			for (size_t h=0; h<R[loc].size(); ++h)
			{
				size_t range = R[loc][h].first;
				complex<double> value = R[loc][h].second;
				
				if (range != 0)
				{
					vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> > ops(range+1);
					ops[0] = first[j];
					for (size_t i=1; i<range; ++i)
					{
						ops[i] = B[(loc+i)%N_sites].Id().template cast<complex<double> >();
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
		
		// Full J-matrices
		if (P.HAS("Jzfull"))
		{
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> > first {B[loc].Sz(0).template cast<complex<double> >()};
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> > Sz_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++) {Sz_ranges[i] = B[i].Sz(0).template cast<complex<double> >();}
			
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> > > last {Sz_ranges};
			push_full("Jzfull", "Jzᵢⱼ", first, last, {1.0}, {false});
		}
		if (P.HAS("Jxyfull"))
		{
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> > first {B[loc].Sp(0).template cast<complex<double> >(), 
			                                                          B[loc].Sm(0).template cast<complex<double> >()};
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> > Sp_ranges(N_sites);
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> > Sm_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++) {Sp_ranges[i] = B[i].Sp(0).template cast<complex<double> >(); 
			                                  Sm_ranges[i] = B[i].Sm(0).template cast<complex<double> >();}
			
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> > > last {Sm_ranges, Sp_ranges};
			push_full("Jxyfull", "Jxyᵢⱼ", first, last, {0.5,0.5}, {false,true});
		}
		if (P.HAS("JzfullA"))
		{
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> > first {B[loc].Sz(1).template cast<complex<double> >()};
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> > Sz_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++) {Sz_ranges[i] = B[i].Sz(1).template cast<complex<double> >();}
			
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> > > last {Sz_ranges};
			push_full("JzfullA", "JzAᵢⱼ", first, last, {1.0}, {false});
		}
		if (P.HAS("JxyfullA"))
		{
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> > first {B[loc].Sp(1).template cast<complex<double> >(), 
			                                                          B[loc].Sm(1).template cast<complex<double> >()};
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> > Sp_ranges(N_sites);
			vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> > Sm_ranges(N_sites);
			for (size_t i=0; i<N_sites; i++) {Sp_ranges[i] = B[i].Sp(1).template cast<complex<double> >(); 
			                                  Sm_ranges[i] = B[i].Sm(1).template cast<complex<double> >();}
			
			vector<vector<SiteOperatorQ<Symmetry_,Eigen::MatrixXcd> > > last {Sm_ranges, Sp_ranges};
			push_full("JxyfullA", "JxyAᵢⱼ", first, last, {0.5,0.5}, {false,true});
		}
		
		param2d Jxy3site = P.fill_array2d<double>("Jxy3site", "Jxy3site_array", {orbitals, next3_orbitals}, loc%Lcell);
		param2d Jxy4site = P.fill_array2d<double>("Jxy4site", "Jxy4site_array", {orbitals, next4_orbitals}, loc%Lcell);
		
		labellist[loc].push_back(Jxy3site.label);
		labellist[loc].push_back(Jxy4site.label);
		
		if (loc < N_sites-3 or !static_cast<bool>(boundary))
		{
			assert(orbitals == next3_orbitals);
			for (std::size_t alfa=0; alfa<orbitals; ++alfa)
			{
//				if (std::abs(complex<double>(+0.25*Jxy3site(alfa,alfa),0.)) > 1e-10)
//				{
//					cout << "loc=" << loc << ", alfa=" << alfa << ", 0.25*val3=" << complex<double>(+0.25*Jxy3site(alfa,alfa),0.) << endl;
////					cout << "B[loc].Sp(alfa)=" << B[loc].Sp(alfa).template plain<double>() << endl;
//					cout << endl;
//				}
				// +-+-, sign:+ i,A-i,B--i+1,A-i+1,B
				pushlist.push_back(std::make_tuple(loc,Mpo<Symmetry,complex<double> >::get_N_site_interaction(B[loc].Sp(alfa).template cast<complex<double> >(),
				                                                                                              B[lp1].Sm(alfa).template cast<complex<double> >(),
				                                                                                              B[lp2].Sp(alfa).template cast<complex<double> >(),
				                                                                                              B[lp3].Sm(alfa).template cast<complex<double> >()),
				                                                                                              complex<double>(+0.25*Jxy3site(alfa,alfa),0.)
				                                                                                              ));
				// -+-+, sign:+ i,A-i,B--i+1,A-i+1,B
				pushlist.push_back(std::make_tuple(loc,Mpo<Symmetry,complex<double> >::get_N_site_interaction(B[loc].Sm(alfa).template cast<complex<double> >(),
				                                                                                              B[lp1].Sp(alfa).template cast<complex<double> >(),
				                                                                                              B[lp2].Sm(alfa).template cast<complex<double> >(),
				                                                                                              B[lp3].Sp(alfa).template cast<complex<double> >()),
				                                                                                              complex<double>(+0.25*Jxy3site(alfa,alfa),0.)
				                                                                                              ));
				// +--+, sign:- i,A-i,B--i+1,A-i+1,B
				pushlist.push_back(std::make_tuple(loc,Mpo<Symmetry,complex<double> >::get_N_site_interaction(B[loc].Sp(alfa).template cast<complex<double> >(),
				                                                                                              B[lp1].Sm(alfa).template cast<complex<double> >(),
				                                                                                              B[lp2].Sm(alfa).template cast<complex<double> >(),
				                                                                                              B[lp3].Sp(alfa).template cast<complex<double> >()),
				                                                                                              complex<double>(-0.25*Jxy3site(alfa,alfa),0.)
				                                                                                              ));
				// -++-, sign:- i,A-i,B--i+1,A-i+1,B
				pushlist.push_back(std::make_tuple(loc,Mpo<Symmetry,complex<double> >::get_N_site_interaction(B[loc].Sm(alfa).template cast<complex<double> >(),
				                                                                                              B[lp1].Sp(alfa).template cast<complex<double> >(),
				                                                                                              B[lp2].Sp(alfa).template cast<complex<double> >(),
				                                                                                              B[lp3].Sm(alfa).template cast<complex<double> >()),
				                                                                                              complex<double>(-0.25*Jxy3site(alfa,alfa),0.)
				                                                                                              ));
			}
		}
		
		if (loc < N_sites-4 or !static_cast<bool>(boundary))
		{
			assert(orbitals == next4_orbitals);
			for (std::size_t alfa=0; alfa<orbitals; ++alfa)
			{
				//cout << "loc=" << loc << ", alfa=" << alfa << ", 0.25*val4=" << complex<double>(+0.25*Jxy4site(alfa,alfa)) << endl;
				pushlist.push_back(std::make_tuple(lp1,Mpo<Symmetry,complex<double> >::get_N_site_interaction(B[lp1].Sp(alfa).template cast<complex<double> >(),
				                                                                                              B[lp2].Sm(alfa).template cast<complex<double> >(),
				                                                                                              B[lp3].Sp(alfa).template cast<complex<double> >(),
				                                                                                              B[lp4].Sm(alfa).template cast<complex<double> >()),
				                                                                                              complex<double>(+0.25*Jxy4site(alfa,alfa),0.)
				                                                                                              ));
				pushlist.push_back(std::make_tuple(lp1,Mpo<Symmetry,complex<double> >::get_N_site_interaction(B[lp1].Sm(alfa).template cast<complex<double> >(),
				                                                                                              B[lp2].Sp(alfa).template cast<complex<double> >(),
				                                                                                              B[lp3].Sm(alfa).template cast<complex<double> >(),
				                                                                                              B[lp4].Sp(alfa).template cast<complex<double> >()),
				                                                                                              complex<double>(+0.25*Jxy4site(alfa,alfa),0.)
				                                                                                              ));
				pushlist.push_back(std::make_tuple(lp1,Mpo<Symmetry,complex<double> >::get_N_site_interaction(B[lp1].Sp(alfa).template cast<complex<double> >(),
				                                                                                              B[lp2].Sm(alfa).template cast<complex<double> >(),
				                                                                                              B[lp3].Sm(alfa).template cast<complex<double> >(),
				                                                                                              B[lp4].Sp(alfa).template cast<complex<double> >()),
				                                                                                              complex<double>(-0.25*Jxy4site(alfa,alfa),0.)
				                                                                                              ));
				pushlist.push_back(std::make_tuple(lp1,Mpo<Symmetry,complex<double> >::get_N_site_interaction(B[lp1].Sm(alfa).template cast<complex<double> >(),
				                                                                                              B[lp2].Sp(alfa).template cast<complex<double> >(),
				                                                                                              B[lp3].Sp(alfa).template cast<complex<double> >(),
				                                                                                              B[lp4].Sm(alfa).template cast<complex<double> >()),
				                                                                                              complex<double>(-0.25*Jxy4site(alfa,alfa),0.)
				                                                                                              ));
			}
		}
	}
}

} //end namespace VMPS

#endif
