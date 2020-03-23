#ifndef STRAWBERRY_HEISENBERGU1XXZ
#define STRAWBERRY_HEISENBERGU1XXZ

#include "models/HeisenbergU1.h"

namespace VMPS
{
	
/** \class HeisenbergU1XXZ
  * \ingroup Heisenberg
  *
  * \brief Heisenberg Model with XXZ-coupling
  *
  * MPO representation of
  \f[
  H =  J_{xy} \sum_{<ij>} \left(S^x_iS^x_j+S^y_iS^y_j\right) + J_z \sum_{<ij>} S^z_iS^z_j 
      +J'_{xy} \sum_{<<ij>>} \left(S^x_iS^x_j+S^y_iS^y_j\right) + J'_z \sum_{<<ij>>} S^z_iS^z_j 
      -B_z \sum_i S^z_i
      +K_z \sum_i \left(S^z_i\right)^2
      -D_y \sum_{<ij>} \left(\mathbf{S_i} \times \mathbf{S_j}\right)_y
      -D_y' \sum_{<<ij>>} \left(\mathbf{S_i} \times \mathbf{S_j}\right)_y
  \f]
  *
  \param D : \f$D=2S+1\f$ where \f$S\f$ is the spin
  \note Makes use of the \f$S^z\f$ U(1) symmetry.
  \note The default variable settings can be seen in \p HeisenbergU1XXZ::defaults.
  \note \f$J>0\f$ is antiferromagnetic.
*/
class HeisenbergU1XXZ : public HeisenbergU1
{
public:
	typedef Sym::U1<Sym::SpinU1> Symmetry;
	MAKE_TYPEDEFS(HeisenbergU1XXZ)
	
	static qarray<1> singlet() {return qarray<1>{0};};
	
public:
	
	HeisenbergU1XXZ() : HeisenbergU1() {};
	HeisenbergU1XXZ (const size_t &L, const BC &boundary=BC::OPEN);
	HeisenbergU1XXZ (const size_t &L, const vector<Param> &params={}, const BC &boundary = BC::OPEN);
	
	template<typename Symmetry_>
	static void add_operators(const std::vector<SpinBase<Symmetry_>> &B, const ParamHandler &P, PushType<SiteOperator<Symmetry_,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary=BC::OPEN);
	
	static const std::map<string,std::any> defaults;
};

const std::map<string,std::any> HeisenbergU1XXZ::defaults = 
{
	{"Jxy",1.}, {"Jxyprime",0.}, {"Jxyrung",1.},
	{"Jz",0.}, {"Jzprime",0.}, {"Jzrung",0.},
	
	{"Dy",0.}, {"Dyprime",0.}, {"Dyrung",0.},
	{"Bz",0.}, {"Kz",0.},
	{"D",2ul}, {"maxPower",2ul}, {"CYLINDER",false}, {"Ly",1ul}, 
	
	// for consistency during inheritance (should not be set for XXZ!):
	{"J",0.}, {"Jprime",0.}
};

HeisenbergU1XXZ::
HeisenbergU1XXZ (const size_t &L, const BC &boundary)
:HeisenbergU1(L, boundary)
{}
	
HeisenbergU1XXZ::
HeisenbergU1XXZ (const size_t &L, const vector<Param> &params, const BC &boundary)
:HeisenbergU1(L, boundary)
{
	ParamHandler P(params,HeisenbergU1XXZ::defaults);	
	size_t Lcell = P.size();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		
		B[l] = SpinBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), P.get<size_t>("D",l%Lcell));
		setLocBasis(B[l].get_basis().qloc(),l);
	}

	if (P.HAS_ANY_OF({"Jxy", "Jxypara", "Jxyperp", "Jxyfull"}))
	{
		this->set_name("XXZ");
	}
	else
	{
		this->set_name("Ising");
	}

	PushType<SiteOperator<Symmetry,double>,double> pushlist;
    std::vector<std::vector<std::string>> labellist;
	set_operators(B, P, pushlist, labellist, boundary);
	add_operators(B, P, pushlist, labellist, boundary);

	this->construct_from_pushlist(pushlist, labellist, Lcell);
    this->finalize(PROP::COMPRESS, P.get<size_t>("maxPower"));

	this->precalc_TwoSiteData();
}

template<typename Symmetry_>
void HeisenbergU1XXZ::
add_operators (const std::vector<SpinBase<Symmetry_>> &B, const ParamHandler &P, PushType<SiteOperator<Symmetry_,double>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = B.size();
	if(labellist.size() != N_sites) {labellist.resize(N_sites);}
		
	for (std::size_t loc=0; loc<N_sites; ++loc)
	{
		size_t lp1 = (loc+1)%N_sites;
		size_t lp2 = (loc+2)%N_sites;
		
		std::size_t orbitals       = B[loc].orbitals();
		std::size_t next_orbitals  = B[lp1].orbitals();
		std::size_t nextn_orbitals = B[lp2].orbitals();
		
//		stringstream ss1, ss2;
//		ss2 << "Ly=" << P.get<size_t>("Ly",loc%Lcell);
//		Terms.save_label(loc, ss2.str());

		auto push_full = [&N_sites, &loc, &B, &P, &pushlist, &labellist, &boundary] (string xxxFull, string label,
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
			ss << label << "ⱼ(" << Geometry2D::hoppingInfo(Full) << ")";
			labellist[loc].push_back(ss.str());
		};

		// Local terms: J⟂
		
		param2d Jxyperp = P.fill_array2d<double>("Jxyrung", "Jxy", "Jxyperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Jzperp  = P.fill_array2d<double>("Jzrung",  "Jz",  "Jzperp",  orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		
		labellist[loc].push_back(Jxyperp.label);
		labellist[loc].push_back(Jzperp.label);
		
		ArrayXd Bz_array      = B[loc].ZeroField();
		ArrayXd Bx_array      = B[loc].ZeroField();
		ArrayXd mu_array      = B[loc].ZeroField();
		ArrayXd Kz_array      = B[loc].ZeroField();
		ArrayXd Kx_array      = B[loc].ZeroField();
		ArrayXXd Dyperp_array = B[loc].ZeroHopping();

		auto Hloc = Mpo<Symmetry,double>::get_N_site_interaction(B[loc].HeisenbergHamiltonian(Jxyperp.a, Jzperp.a, Bz_array, mu_array, Kz_array));
		pushlist.push_back(std::make_tuple(loc, Hloc, 1.));

		// Case, where a full coupling-matrix is provided: Jᵢⱼ
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
		
		if (P.HAS("Jxyfull") or P.HAS("Jzfull")) continue;
				
		// Nearest-neighbour terms: J
		
		param2d Jxypara = P.fill_array2d<double>("Jxy", "Jxypara", {orbitals, next_orbitals}, loc%Lcell);
		param2d Jzpara  = P.fill_array2d<double>("Jz",  "Jzpara",  {orbitals, next_orbitals}, loc%Lcell);
		
		labellist[loc].push_back(Jxypara.label);
		labellist[loc].push_back(Jzpara.label);
		
//		if (!P.HAS("Jxy"))
//		{
//			labellist[loc].push_back("def.Jxy=1.");
//		}
		
		if (loc < N_sites-1 or !static_cast<bool>(boundary))
		{
			for (std::size_t alfa=0; alfa < orbitals; ++alfa)
			for (std::size_t beta=0; beta < next_orbitals; ++beta)
			{
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(B[loc].Scomp(SP,alfa), B[lp1].Scomp(SM,beta)), 0.5*Jxypara(alfa,beta)));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(B[loc].Scomp(SM,alfa), B[lp1].Scomp(SP,beta)), 0.5*Jxypara(alfa,beta)));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(B[loc].Scomp(SZ,alfa), B[lp1].Scomp(SZ,beta)),     Jzpara(alfa,beta)));
			}
		}
		
		// Next-nearest-neighbour terms: J'
		
		param2d Jxyprime = P.fill_array2d<double>("Jxyprime", "Jxyprime_array", {orbitals, nextn_orbitals}, loc%Lcell);
		param2d Jzprime  = P.fill_array2d<double>("Jzprime",  "Jzprime_array",  {orbitals, nextn_orbitals}, loc%Lcell);
		
		labellist[loc].push_back(Jxyprime.label);
		labellist[loc].push_back(Jzprime.label);
		
		if (loc < N_sites-2 or !static_cast<bool>(boundary))
		{
			for (std::size_t alfa=0; alfa < orbitals; ++alfa)
			for (std::size_t beta=0; beta < nextn_orbitals; ++beta)
			{
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(B[loc].Scomp(SP,alfa), B[lp1].Id(), B[lp2].Scomp(SM,beta)), 0.5*Jxyprime(alfa,beta)));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(B[loc].Scomp(SM,alfa), B[lp1].Id(), B[lp2].Scomp(SP,beta)), 0.5*Jxyprime(alfa,beta)));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,double>::get_N_site_interaction(B[loc].Scomp(SZ,alfa), B[lp1].Id(), B[lp2].Scomp(SZ,beta)),     Jzprime(alfa,beta)));
			}
		}
	}
}

} //end namespace VMPS

#endif
