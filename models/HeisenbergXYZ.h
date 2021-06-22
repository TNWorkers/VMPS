#ifndef STRAWBERRY_HEISENBERGXYZ
#define STRAWBERRY_HEISENBERGXYZ

#include "models/Heisenberg.h"

namespace VMPS
{

/** \class HeisenbergXYZ
  * \ingroup Heisenberg
  *
  * \brief Heisenberg Model with general XYZ coupling.
  *
  * MPO representation of
  \f[
  H = \sum_{\alfa=x,y,z} \left(
   J_{\alfa} \sum_{<ij>} \left(S^{\alfa}_i \cdot S^{\alfa}_j\right) 
  +J'_{\alfa} \sum_{<<ij>>} \left(S^{\alfa}_i \cdot S^{\alfa}_j\right) 
  -B_{\alfa} \sum_i S^{\alfa}_i
  +K_{\alfa} \sum_i \left(S^{\alfa}_i\right)^2
  \right)
  -\mathbf{D} \sum_{<ij>} \left(\mathbf{S}_i \times \mathbf{S}_j\right) 
  -\mathbf{D'} \sum_{<<ij>>} \left(\mathbf{S}_i \times \mathbf{S}_j\right) 
  \f]
  *
  \param D : \f$D=2S+1\f$ where \f$S\f$ is the spin
  \note Uses no symmetries. Any parameter constellations are allowed. For variants with symmetries, see VMPS::HeisenbergU1 or VMPS::HeisenbergSU2.
  \note The default variable settings can be seen in \p HeisenbergXYZ::defaults.
  \note \f$J>0\f$ is antiferromagnetic
  \note Due to the \f$S_y\f$ operator, this MPO is complex.
*/
class HeisenbergXYZ : public Mpo<Sym::U0,complex<double> >, public HeisenbergObservables<Sym::U0>, public ParamReturner
{
public:
	typedef Sym::U0 Symmetry;
	MAKE_TYPEDEFS(HeisenbergXYZ)
	
private:
	typedef typename Symmetry::qType qType;
	
	static qarray<0> singlet(int N=0) {return qarray<0>{};};
	static constexpr MODEL_FAMILY FAMILY = HEISENBERG;
	
public:
	
	///\{
	HeisenbergXYZ() : Mpo<Symmetry,complex<double> >(), ParamReturner(Heisenberg::sweep_defaults) {};
	HeisenbergXYZ (const size_t &L, const vector<Param> &params, const BC &boundary=BC::OPEN, const DMRG::VERBOSITY::OPTION &VERB=DMRG::VERBOSITY::OPTION::ON_EXIT);
	///\}
	
	template<typename Symmetry_>
	void add_operators (const std::vector<SpinBase<Symmetry_>> &B, const ParamHandler &P, 
	                    PushType<SiteOperator<Symmetry_,complex<double>>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, 
	                    const BC boundary=BC::OPEN);
	
	static const std::map<string,std::any> defaults;
};

const std::map<string,std::any> HeisenbergXYZ::defaults = 
{
	{"Jx",0.}, {"Jy",0.}, {"Jz",0.},
	{"Jxrung",0.}, {"Jyrung",0.}, {"Jzrung",0.},
	{"Jxprime",0.}, {"Jyprime",0.}, {"Jzprime",0.},
	
	// Dzialoshinsky-Moriya terms
	{"Dx",0.}, {"Dy",0.}, {"Dz",0.},
	{"Dxrung",0.}, {"Dyrung",0.}, {"Dzrung",0.},
	{"Dxprime",0.}, {"Dyprime",0.}, {"Dzprime",0.},
	
	{"Bx",0.}, {"By",0.}, {"Bz",0.},
	{"Kx",0.}, {"Ky",0.}, {"Kz",0.},
	
	{"D",2ul}, {"maxPower",2ul}, {"CYLINDER",false}, {"Ly",1ul},
	
	{"J",0.}, {"Jprime",0.}
};

HeisenbergXYZ::
HeisenbergXYZ (const size_t &L, const vector<Param> &params, const BC &boundary, const DMRG::VERBOSITY::OPTION &VERB)
:Mpo<Symmetry,complex<double> > (L, qarray<0>({}), "", PROP::HERMITIAN, PROP::NON_UNITARY, boundary, VERB),
 HeisenbergObservables(L,params,HeisenbergXYZ::defaults),
 ParamReturner(Heisenberg::sweep_defaults)
{
	ParamHandler P(params,HeisenbergXYZ::defaults);   
	size_t Lcell = P.size();
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(B[l].get_basis().qloc(),l);
	}

	if(P.HAS_ANY_OF({"Dx", "Dy", "Dz", "Dxprime", "Dyprime", "Dzprime", "Dxpara", "Dypara", "Dzpara", "Dxperp", "Dzperp"}))
	{
		this->set_name("Dzyaloshinsky-Moriya");
	}
	else
	{
		this->set_name("HeisenbergXYZ");
	}
	
	PushType<SiteOperator<Symmetry,double>,double> pushlist_aux;
	std::vector<std::vector<std::string>> labellist;
	
	HeisenbergU1::set_operators(B,P,pushlist_aux,labellist,boundary);
	Heisenberg::add_operators(B,P,pushlist_aux,labellist,boundary);
	
	PushType<SiteOperator<Symmetry,complex<double>>,double> pushlist = pushlist_aux.cast<SiteOperator<Symmetry,complex<double>>, double>();
	add_operators(B,P,pushlist,labellist,boundary);
	
	this->construct_from_pushlist(pushlist, labellist, Lcell);
	this->finalize(PROP::COMPRESS, P.get<size_t>("maxPower"));
	
	this->precalc_TwoSiteData();
}

template<typename Symmetry_>
void HeisenbergXYZ::
add_operators(const std::vector<SpinBase<Symmetry_>> &B, const ParamHandler &P, PushType<SiteOperator<Symmetry_,complex<double>>,double>& pushlist, std::vector<std::vector<std::string>>& labellist, const BC boundary)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = B.size();
	if(labellist.size() != N_sites) {labellist.resize(N_sites);}
	
	for(std::size_t loc=0; loc<N_sites; ++loc)
	{
		std::size_t orbitals = B[loc].orbitals();
		std::size_t next_orbitals = B[(loc+1)%N_sites].orbitals();
		std::size_t nextn_orbitals = B[(loc+2)%N_sites].orbitals();
		
		stringstream ss1, ss2;
		ss1 << "S=" << print_frac_nice(frac(P.get<size_t>("D",loc%Lcell)-1,2));
		ss2 << "Ly=" << P.get<size_t>("Ly",loc%Lcell);
		labellist[loc].push_back(ss1.str());
		labellist[loc].push_back(ss2.str());
		
		// Local terms: J⟂, DM⟂, B and K
		
		param2d Jxperp = P.fill_array2d<double>("Jxrung", "Jx", "Jxperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Jyperp = P.fill_array2d<double>("Jyrung", "Jy", "Jyperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Jzperp = P.fill_array2d<double>("Jzrung", "Jz", "Jzperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Dxperp = P.fill_array2d<double>("Dxrung", "Dx", "Dxperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Dzperp = P.fill_array2d<double>("Dzrung", "Dz", "Dzperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param1d By = P.fill_array1d<double>("By", "Byorb", orbitals, loc%Lcell);
		param1d Ky = P.fill_array1d<double>("Ky", "Kyorb", orbitals, loc%Lcell);
		
		labellist[loc].push_back(Jxperp.label);
		labellist[loc].push_back(Jyperp.label);
		labellist[loc].push_back(Jzperp.label);
		labellist[loc].push_back(Dxperp.label);
		labellist[loc].push_back(Dzperp.label);
		labellist[loc].push_back(By.label);
		labellist[loc].push_back(Ky.label);
		
		std::array<Eigen::ArrayXXd,3> Jperp = {Jxperp.a, Jyperp.a, Jzperp.a};
		std::array<Eigen::ArrayXd,3>  B_array = {B[loc].ZeroField(), By.a, B[loc].ZeroField()};
		std::array<Eigen::ArrayXd,3>  K_array = {B[loc].ZeroField(), Ky.a, B[loc].ZeroField()};
		std::array<Eigen::ArrayXXd,3> Dperp = {Dxperp.a, B[loc].ZeroHopping(), Dzperp.a};
		
		auto Hloc = Mpo<Symmetry,complex<double> >::get_N_site_interaction(B[loc].HeisenbergHamiltonian(Jperp,B_array,K_array,Dperp));
		pushlist.push_back(std::make_tuple(loc, Hloc, 1.));
		
		// Nearest-neighbour terms: J and DM
		param2d Jxpara = P.fill_array2d<double>("Jx", "Jxpara", {orbitals, next_orbitals}, loc%Lcell);
		param2d Jypara = P.fill_array2d<double>("Jy", "Jypara", {orbitals, next_orbitals}, loc%Lcell);
		param2d Jzpara = P.fill_array2d<double>("Jz", "Jzpara", {orbitals, next_orbitals}, loc%Lcell);
		param2d Dxpara = P.fill_array2d<double>("Dx", "Dxpara", {orbitals, next_orbitals}, loc%Lcell);
		param2d Dzpara = P.fill_array2d<double>("Dz", "Dzpara", {orbitals, next_orbitals}, loc%Lcell);
		param2d JyKTpara = P.fill_array2d<double>("JyKT", "JyKTpara", {orbitals, next_orbitals}, loc%Lcell);
		
		labellist[loc].push_back(Jxpara.label);
		labellist[loc].push_back(Jypara.label);
		labellist[loc].push_back(Jzpara.label);
		labellist[loc].push_back(Dxpara.label);
		labellist[loc].push_back(Dzpara.label);
		labellist[loc].push_back(JyKTpara.label);
		
		if (loc < N_sites-1 or !static_cast<bool>(boundary))
		{
			for (std::size_t alfa=0; alfa<orbitals; ++alfa)
			for (std::size_t beta=0; beta<next_orbitals; ++beta)
			{
				auto local_Sx = B[loc].Scomp(SX,alfa).template cast<complex<double> >();
				auto local_Sy = -1.i*B[loc].Scomp(iSY,alfa).template cast<complex<double> >();
				auto local_Sz = B[loc].Scomp(SZ,alfa).template cast<complex<double> >();
				
				auto tight_Sx = B[(loc+1)%N_sites].Scomp(SX,beta).template cast<complex<double> >();
				auto tight_Sy = -1.i*B[(loc+1)%N_sites].Scomp(iSY,beta).template cast<complex<double> >();
				auto tight_Sz = B[(loc+1)%N_sites].Scomp(SZ,beta).template cast<complex<double> >();
				
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,complex<double> >::get_N_site_interaction(local_Sx, tight_Sx), Jxpara(alfa,beta)));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,complex<double> >::get_N_site_interaction(local_Sy, tight_Sy), Jypara(alfa,beta)));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,complex<double> >::get_N_site_interaction(local_Sz, tight_Sz), Jzpara(alfa,beta)));
				
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,complex<double> >::get_N_site_interaction(local_Sy, tight_Sx), +Dzpara(alfa,beta)));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,complex<double> >::get_N_site_interaction(local_Sy, tight_Sz), -Dxpara(alfa,beta)));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,complex<double> >::get_N_site_interaction(local_Sz, tight_Sy), +Dxpara(alfa,beta)));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,complex<double> >::get_N_site_interaction(local_Sx, tight_Sy), -Dzpara(alfa,beta)));
			}
		}
	
		// Next-nearest-neighbour terms: J and DM
		param2d Jxprime = P.fill_array2d<double>("Jxprime", "Jxprime", {orbitals, nextn_orbitals}, loc%Lcell);
		param2d Jyprime = P.fill_array2d<double>("Jyprime", "Jyprime", {orbitals, nextn_orbitals}, loc%Lcell);
		param2d Jzprime = P.fill_array2d<double>("Jzprime", "Jzprime", {orbitals, nextn_orbitals}, loc%Lcell);
		param2d Dxprime = P.fill_array2d<double>("Dxprime", "Dxprime_array", {orbitals, nextn_orbitals}, loc%Lcell);
		param2d Dzprime = P.fill_array2d<double>("Dzprime", "Dzprime_array", {orbitals, nextn_orbitals}, loc%Lcell);
		
		labellist[loc].push_back(Jxprime.label);
		labellist[loc].push_back(Jyprime.label);
		labellist[loc].push_back(Jzprime.label);
		labellist[loc].push_back(Dxprime.label);
		labellist[loc].push_back(Dzprime.label);
		
		if(loc < N_sites-2 or !static_cast<bool>(boundary))
		{
			for(std::size_t alfa=0; alfa<orbitals; ++alfa)
			for(std::size_t beta=0; beta<nextn_orbitals; ++beta)
			{
				auto local_Sx = B[loc].Scomp(SX,alfa).template cast<complex<double> >();
				auto local_Sy = -1.i*B[loc].Scomp(iSY,alfa).template cast<complex<double> >();
				auto local_Sz = B[loc].Scomp(SZ,alfa).template cast<complex<double> >();
				auto tight_Id = B[(loc+1)%N_sites].Id().template cast<complex<double> >();
				auto nextn_Sx = B[(loc+2)%N_sites].Scomp(SX,beta).template cast<complex<double> >();
				auto nextn_Sy = -1.i*B[(loc+2)%N_sites].Scomp(iSY,beta).template cast<complex<double> >();
				auto nextn_Sz = B[(loc+2)%N_sites].Scomp(SZ,beta).template cast<complex<double> >();
				
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,complex<double> >::get_N_site_interaction(local_Sx, tight_Id, nextn_Sx), Jxprime(alfa,beta)));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,complex<double> >::get_N_site_interaction(local_Sy, tight_Id, nextn_Sy), Jyprime(alfa,beta)));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,complex<double> >::get_N_site_interaction(local_Sz, tight_Id, nextn_Sz), Jzprime(alfa,beta)));
				
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,complex<double> >::get_N_site_interaction(local_Sy, tight_Id, nextn_Sx), +Dzprime(alfa,beta)));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,complex<double> >::get_N_site_interaction(local_Sy, tight_Id, nextn_Sz), -Dxprime(alfa,beta)));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,complex<double> >::get_N_site_interaction(local_Sz, tight_Id, nextn_Sy), +Dxprime(alfa,beta)));
				pushlist.push_back(std::make_tuple(loc, Mpo<Symmetry,complex<double> >::get_N_site_interaction(local_Sx, tight_Id, nextn_Sy), -Dzprime(alfa,beta)));
			}
		}
	}
}

} //end namespace VMPS

#endif
