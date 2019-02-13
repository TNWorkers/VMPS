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
  H = \sum_{\alpha=x,y,z} \left(
       J_{\alpha} \sum_{<ij>} \left(S^{\alpha}_i \cdot S^{\alpha}_j\right) 
      +J'_{\alpha} \sum_{<<ij>>} \left(S^{\alpha}_i \cdot S^{\alpha}_j\right) 
      -B_{\alpha} \sum_i S^{\alpha}_i
      +K_{\alpha} \sum_i \left(S^{\alpha}_i\right)^2
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
	
	static qarray<0> singlet() {return qarray<0>{};};
	
public:
	
	///\{
	HeisenbergXYZ() : Mpo<Symmetry,complex<double> >(), ParamReturner(Heisenberg::sweep_defaults) {};
	HeisenbergXYZ (const size_t &L, const vector<Param> &params);
	///\}
	
	template<typename Symmetry_>
	//void add_operators (HamiltonianTerms<Symmetry_,complex<double> > &Terms, const vector<SpinBase<Symmetry_> > &B, const ParamHandler &P, size_t loc=0);
    void add_operators(const std::vector<SpinBase<Symmetry_>> &B, const ParamHandler &P, HamiltonianTerms<Symmetry_,std::complex<double>> &Terms);
    
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
	
	{"D",2ul}, {"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul},
	
	{"J",0.}, {"Jprime",0.}
};

HeisenbergXYZ::
HeisenbergXYZ (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry,complex<double> > (L, qarray<0>({}), "", PROP::HERMITIAN, PROP::NON_UNITARY, PROP::HAMILTONIAN),
 HeisenbergObservables(L,params,HeisenbergXYZ::defaults),
 ParamReturner(Heisenberg::sweep_defaults)
{
	/*ParamHandler P(params,HeisenbergXYZ::defaults);
	
	size_t Lcell = P.size();
	vector<HamiltonianTerms<Symmetry,complex<double> > > Terms(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(B[l].get_basis(),l);
	}
	
	for (size_t l=0; l<N_sites; ++l)
	{
		auto Terms_tmp = HeisenbergU1::set_operators(B,P,l%Lcell);
		Heisenberg::add_operators(Terms_tmp,B,P,l%Lcell);
		Terms[l] = Terms_tmp.cast<complex<double> >();
		add_operators(Terms[l],B,P,l%Lcell);
		
		stringstream ss;
		ss << "Ly=" << P.get<size_t>("Ly",l%Lcell);
		Terms[l].info.push_back(ss.str());
	}
	
	this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
	this->precalc_TwoSiteData();*/
    
    ParamHandler P(params,HeisenbergXYZ::defaults);
    
    size_t Lcell = P.size();
    
    for (size_t l=0; l<N_sites; ++l)
    {
        N_phys += P.get<size_t>("Ly",l%Lcell);
        setLocBasis(B[l].get_basis(),l);
    }
    
    HamiltonianTermsXd<Symmetry> Terms_aux(N_sites, P.get<bool>("OPEN_BC"));
    HeisenbergU1::set_operators(B,P,Terms_aux);
    Heisenberg::add_operators(B,P,Terms_aux);
    HamiltonianTermsXcd<Symmetry> Terms = Terms_aux.cast<std::complex<double>>();
    add_operators(B,P,Terms);
    
    this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
    this->precalc_TwoSiteData();
}

/*template<typename Symmetry_>
void HeisenbergXYZ::
add_operators (HamiltonianTerms<Symmetry_,complex<double> > &Terms, const vector<SpinBase<Symmetry_> > &B, const ParamHandler &P, size_t loc)
{
	auto save_label = [&Terms] (string label)
	{
		if (label!="") {Terms.info.push_back(label);}
	};
	
	size_t lp1 = (loc+1)%B.size();
	
	// J terms
	
	auto [Jx,Jxpara,Jxlabel] = P.fill_array2d<double>("Jx","Jxpara",{{B[loc].orbitals(),B[lp1].orbitals()}},loc);
	save_label(Jxlabel);
	
	auto [Jy,Jypara,Jylabel] = P.fill_array2d<double>("Jy","Jypara",{{B[loc].orbitals(),B[lp1].orbitals()}},loc);
	save_label(Jylabel);
	
	auto [Jz,Jzpara,Jzlabel] = P.fill_array2d<double>("Jz","Jzpara",{{B[loc].orbitals(),B[lp1].orbitals()}},loc);
	save_label(Jzlabel);
	
	for (int i=0; i<B[loc].orbitals(); ++i)
	for (int j=0; j<B[lp1].orbitals(); ++j)
	{
		if (Jxpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(Jxpara(i,j), B[loc].Scomp(SX,i).template cast<complex<double> >(), 
			                                              B[loc].Scomp(SX,i).template cast<complex<double> >()));
		}
		if (Jypara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(Jypara(i,j), -1.i*B[loc].Scomp(iSY,i).template cast<complex<double> >(), 
			                                              -1.i*B[loc].Scomp(iSY,i).template cast<complex<double> >()));
		}
		if (Jzpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(Jzpara(i,j), B[loc].Scomp(SZ,i).template cast<complex<double> >(), 
			                                              B[loc].Scomp(SZ,i).template cast<complex<double> >()));
		}
	}
	
	// DM terms
	
	auto [Dx,Dxpara,Dxlabel] = P.fill_array2d<double>("Dx","Dxpara",B[loc].orbitals(),loc);
	save_label(Dxlabel);
	
	auto [Dz,Dzpara,Dzlabel] = P.fill_array2d<double>("Dz","Dzpara",B[loc].orbitals(),loc);
	save_label(Dzlabel);
	
	for (int i=0; i<B[loc].orbitals(); ++i)
	for (int j=0; j<B[lp1].orbitals(); ++j)
	{
		if (Dxpara(i,j)!=0. or Dzpara(i,j)!=0.)
		{
			SiteOperator<Symmetry_,complex<double> > Sx = B[loc].Scomp(SX,i).template cast<complex<double> >();
			SiteOperator<Symmetry_,complex<double> > Sy = -1.i*B[loc].Scomp(iSY,i).template cast<complex<double> >();
			SiteOperator<Symmetry_,complex<double> > Sz = B[loc].Scomp(SZ,i).template cast<complex<double> >();
			
			Terms.tight.push_back(make_tuple(1., Sy, +Dzpara(i,j)*Sx-Dxpara(i,j)*Sz));
			Terms.tight.push_back(make_tuple(1., +Dxpara(i,j)*Sz-Dzpara(i,j)*Sx, Sy));
		}
	}
	
	// NNN terms
	
	param0d Dxprime = P.fill_array0d<double>("Dxprime","Dxprime",loc);
	save_label(Dxprime.label);
	
	param0d Dzprime = P.fill_array0d<double>("Dzprime","Dzprime",loc);
	save_label(Dzprime.label);
	
	if (Dxprime.x!=0. or Dzprime.x!=0.)
	{
		assert(B[loc].orbitals() == 1 and "Cannot do a ladder with Dx'/Dz' terms!");
		
		SiteOperator<Symmetry_,complex<double> > Sx = B[loc].Scomp(SX).template cast<complex<double> >();
		SiteOperator<Symmetry_,complex<double> > Sy = -1.i*B[loc].Scomp(iSY).template cast<complex<double> >();
		SiteOperator<Symmetry_,complex<double> > Sz = B[loc].Scomp(SZ).template cast<complex<double> >();
		SiteOperator<Symmetry_,complex<double> > Id = B[loc].Id().template cast<complex<double> >();
		
		Terms.nextn.push_back(make_tuple(1., Sy, +Dzprime.x*Sx-Dxprime.x*Sz, Id));
		Terms.nextn.push_back(make_tuple(1., +Dxprime.x*Sz-Dzprime.x*Sx, Sy, Id));
	}
	
	param0d Jxprime = P.fill_array0d<double>("Jxprime","Jxprime",loc);
	save_label(Jxprime.label);
	
	param0d Jyprime = P.fill_array0d<double>("Jyprime","Jyprime",loc);
	save_label(Jyprime.label);
	
	param0d Jzprime = P.fill_array0d<double>("Jzprime","Jzprime",loc);
	save_label(Jzprime.label);
	
	if (Jxprime.x != 0.)
	{
		assert(B[loc].orbitals() == 1 and "Cannot do a ladder with Jx' terms!");
		Terms.nextn.push_back(make_tuple(Jxprime.x, B[loc].Scomp(SX).template cast<complex<double> >(), 
		                                            B[loc].Scomp(SX).template cast<complex<double> >(), 
		                                            B[loc].Id().template cast<complex<double> >()));
	}
	
	if (Jyprime.x != 0.)
	{
		assert(B[loc].orbitals() == 1 and "Cannot do a ladder with Jy' terms!");
		Terms.nextn.push_back(make_tuple(Jyprime.x, -1.i*B[loc].Scomp(iSY).template cast<complex<double> >(), 
		                                            -1.i*B[loc].Scomp(iSY).template cast<complex<double> >(), 
		                                                 B[loc].Id().template cast<complex<double> >()));
	}
	
	if (Jzprime.x != 0.)
	{
		assert(B[loc].orbitals() == 1 and "Cannot do a ladder with Jz' terms!");
		Terms.nextn.push_back(make_tuple(Jzprime.x, B[loc].Scomp(SZ).template cast<complex<double> >(), 
		                                            B[loc].Scomp(SZ).template cast<complex<double> >(), 
		                                            B[loc].Id().template cast<complex<double> >()));
	}
	
	// local terms
	
	auto [Jx_,Jxperp,Jxperplabel] = P.fill_array2d<double>("Jxrung","Jx","Jxperp",B[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
	save_label(Jxperplabel);
	
	auto [Jy_,Jyperp,Jyperplabel] = P.fill_array2d<double>("Jyrung","Jy","Jyperp",B[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
	save_label(Jyperplabel);
	
	auto [Jz_,Jzperp,Jzperplabel] = P.fill_array2d<double>("Jzrung","Jz","Jzperp",B[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
	save_label(Jzperplabel);
	
	auto [Dx_,Dxperp,Dxperplabel] = P.fill_array2d<double>("Dxrung","Dx","Dxperp",B[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
	save_label(Dxperplabel);
	
	auto [Dz_,Dzperp,Dzperplabel] = P.fill_array2d<double>("Dzrung","Dz","Dzperp",B[loc].orbitals(),loc,P.get<bool>("CYLINDER"));
	save_label(Dzperplabel);
	
	auto [By,Byorb,Bylabel] = P.fill_array1d<double>("By","Byorb",B[loc].orbitals(),loc);
	save_label(Bylabel);
	
	auto [Ky,Kyorb,Kylabel] = P.fill_array1d<double>("Ky","Kyorb",B[loc].orbitals(),loc);
	save_label(Kylabel);
	
	Terms.name = (P.HAS_ANY_OF({"Dx","Dy","Dz","Dxprime","Dyprime","Dzprime","Dxpara","Dypara","Dzpara"},loc))? 
	"Dzyaloshinsky-Moriya":"HeisenbergXYZ";
	
	std::array<ArrayXXd,3> Jperp = {Jxperp, Jyperp, Jzperp};
	std::array<ArrayXd,3>  Borb = {B[loc].ZeroField(), Byorb, B[loc].ZeroField()};
	std::array<ArrayXd,3>  Korb = {B[loc].ZeroField(), Kyorb, B[loc].ZeroField()};
	std::array<ArrayXXd,3> Dperp = {Dxperp, B[loc].ZeroHopping(), Dzperp};
	
	Terms.local.push_back(make_tuple(1., B[loc].HeisenbergHamiltonian(Jperp,Borb,Korb,Dperp)));
}*/
    
template<typename Symmetry_>
void HeisenbergXYZ::
add_operators (const std::vector<SpinBase<Symmetry_>> &B, const ParamHandler &P, HamiltonianTerms<Symmetry_,std::complex<double>> &Terms)
{
    std::size_t Lcell = P.size();
    std::size_t N_sites = Terms.size();
    if(P.HAS_ANY_OF({"Dx", "Dy", "Dz", "Dxprime", "Dyprime", "Dzprime", "Dxpara", "Dypara", "Dzpara", "Dxperp", "Dzperp"}))
    {
        Terms.set_name("Dzyaloshinsky-Moriya");
    }
    else
    {
        Terms.set_name("HeisenbergXYZ");
    }
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        std::size_t orbitals = B[loc].orbitals();
        std::size_t next_orbitals = B[(loc+1)%N_sites].orbitals();
        std::size_t nextn_orbitals = B[(loc+2)%N_sites].orbitals();
        
        stringstream ss1, ss2;
        ss1 << "S=" << print_frac_nice(frac(P.get<size_t>("D",loc%Lcell)-1,2));
        ss2 << "Ly=" << P.get<size_t>("Ly",loc%Lcell);
        Terms.save_label(loc, ss1.str());
        Terms.save_label(loc, ss2.str());
        
        // Local terms: J⟂, DM⟂, B and K
        
        param2d Jxperp = P.fill_array2d<double>("Jxrung", "Jx", "Jxperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
        param2d Jyperp = P.fill_array2d<double>("Jyrung", "Jy", "Jyperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
        param2d Jzperp = P.fill_array2d<double>("Jzrung", "Jz", "Jzperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
        param2d Dxperp = P.fill_array2d<double>("Dxrung", "Dx", "Dxperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
        param2d Dzperp = P.fill_array2d<double>("Dzrung", "Dz", "Dzperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
        param1d By = P.fill_array1d<double>("By", "Byorb", orbitals, loc%Lcell);
        param1d Ky = P.fill_array1d<double>("Ky", "Kyorb", orbitals, loc%Lcell);
        
        Terms.save_label(loc, Jxperp.label);
        Terms.save_label(loc, Jyperp.label);
        Terms.save_label(loc, Jzperp.label);
        Terms.save_label(loc, Dxperp.label);
        Terms.save_label(loc, Dzperp.label);
        Terms.save_label(loc, By.label);
        Terms.save_label(loc, Ky.label);
        
        std::array<Eigen::ArrayXXd,3> Jperp = {Jxperp.a, Jyperp.a, Jzperp.a};
        std::array<Eigen::ArrayXd,3>  B_array = {B[loc].ZeroField(), By.a, B[loc].ZeroField()};
        std::array<Eigen::ArrayXd,3>  K_array = {B[loc].ZeroField(), Ky.a, B[loc].ZeroField()};
        std::array<Eigen::ArrayXXd,3> Dperp = {Dxperp.a, B[loc].ZeroHopping(), Dzperp.a};
        
        Terms.push_local(loc, 1., B[loc].HeisenbergHamiltonian(Jperp,B_array,K_array,Dperp));
        
        // Nearest-neighbour terms: J and DM
    
        param2d Jxpara = P.fill_array2d<double>("Jx", "Jxpara", {orbitals, next_orbitals}, loc%Lcell);
        param2d Jypara = P.fill_array2d<double>("Jy", "Jypara", {orbitals, next_orbitals}, loc%Lcell);
        param2d Jzpara = P.fill_array2d<double>("Jz", "Jzpara", {orbitals, next_orbitals}, loc%Lcell);
        param2d Dxpara = P.fill_array2d<double>("Dx", "Dxpara", {orbitals, next_orbitals}, loc%Lcell);
        param2d Dzpara = P.fill_array2d<double>("Dz", "Dzpara", {orbitals, next_orbitals}, loc%Lcell);
        
        Terms.save_label(loc, Jxpara.label);
        Terms.save_label(loc, Jypara.label);
        Terms.save_label(loc, Jzpara.label);
        Terms.save_label(loc, Dxpara.label);
        Terms.save_label(loc, Dzpara.label);
    
        if(loc < N_sites-1 || !P.get<bool>("OPEN_BC"))
        {
            for (std::size_t alpha=0; alpha<orbitals; ++alpha)
            {
                for (std::size_t beta=0; beta<next_orbitals; ++beta)
                {
                    SiteOperator<Symmetry_,complex<double> > local_Sx = B[loc].Scomp(SX,alpha).template cast<complex<double> >();
                    SiteOperator<Symmetry_,complex<double> > local_Sy = -1.i*B[loc].Scomp(iSY,alpha).template cast<complex<double> >();
                    SiteOperator<Symmetry_,complex<double> > local_Sz = B[loc].Scomp(SZ,alpha).template cast<complex<double> >();
                    SiteOperator<Symmetry_,complex<double> > tight_Sx = B[(loc+1)%N_sites].Scomp(SX,beta).template cast<complex<double> >();
                    SiteOperator<Symmetry_,complex<double> > tight_Sy = -1.i*B[(loc+1)%N_sites].Scomp(iSY,beta).template cast<complex<double> >();
                    SiteOperator<Symmetry_,complex<double> > tight_Sz = B[(loc+1)%N_sites].Scomp(SZ,beta).template cast<complex<double> >();
                    
                    Terms.push_tight(loc, Jxpara.a(alpha,beta), local_Sx, tight_Sx);
                    Terms.push_tight(loc, Jypara.a(alpha,beta), local_Sy, tight_Sy);
                    Terms.push_tight(loc, Jzpara.a(alpha,beta), local_Sz, tight_Sz);
                    
                    Terms.push_tight(loc, +Dzpara.a(alpha,beta), local_Sy, tight_Sx);
                    Terms.push_tight(loc, -Dxpara.a(alpha,beta), local_Sy, tight_Sz);
                    Terms.push_tight(loc, +Dxpara.a(alpha,beta), local_Sz, tight_Sy);
                    Terms.push_tight(loc, -Dzpara.a(alpha,beta), local_Sx, tight_Sy);
                }
            }
        }
    
    
        // Next-nearest-neighbour terms: J and DM

        param2d Jxprime = P.fill_array2d<double>("Jxprime", "Jxprime", {orbitals, nextn_orbitals}, loc%Lcell);
        param2d Jyprime = P.fill_array2d<double>("Jyprime", "Jyprime", {orbitals, nextn_orbitals}, loc%Lcell);
        param2d Jzprime = P.fill_array2d<double>("Jzprime", "Jzprime", {orbitals, nextn_orbitals}, loc%Lcell);
        param2d Dxprime = P.fill_array2d<double>("Dxprime", "Dxprime_array", {orbitals, nextn_orbitals}, loc%Lcell);
        param2d Dzprime = P.fill_array2d<double>("Dzprime", "Dzprime_array", {orbitals, nextn_orbitals}, loc%Lcell);

        Terms.save_label(loc, Jxprime.label);
        Terms.save_label(loc, Jyprime.label);
        Terms.save_label(loc, Jzprime.label);
        Terms.save_label(loc, Dxprime.label);
        Terms.save_label(loc, Dzprime.label);
        
        if(loc < N_sites-2 || !P.get<bool>("OPEN_BC"))
        {
            for(std::size_t alpha=0; alpha<orbitals; ++alpha)
            {
                for(std::size_t beta=0; beta<nextn_orbitals; ++beta)
                {
                    SiteOperator<Symmetry_,complex<double> > local_Sx = B[loc].Scomp(SX,alpha).template cast<complex<double> >();
                    SiteOperator<Symmetry_,complex<double> > local_Sy = -1.i*B[loc].Scomp(iSY,alpha).template cast<complex<double> >();
                    SiteOperator<Symmetry_,complex<double> > local_Sz = B[loc].Scomp(SZ,alpha).template cast<complex<double> >();
                    SiteOperator<Symmetry_,complex<double> > tight_Id = B[(loc+1)%N_sites].Id().template cast<complex<double> >();
                    SiteOperator<Symmetry_,complex<double> > nextn_Sx = B[(loc+2)%N_sites].Scomp(SX,beta).template cast<complex<double> >();
                    SiteOperator<Symmetry_,complex<double> > nextn_Sy = -1.i*B[(loc+2)%N_sites].Scomp(iSY,beta).template cast<complex<double> >();
                    SiteOperator<Symmetry_,complex<double> > nextn_Sz = B[(loc+2)%N_sites].Scomp(SZ,beta).template cast<complex<double> >();

                    Terms.push_nextn(loc, Jxprime.a(alpha,beta), local_Sx, tight_Id, nextn_Sx);
                    Terms.push_nextn(loc, Jyprime.a(alpha,beta), local_Sy, tight_Id, nextn_Sy);
                    Terms.push_nextn(loc, Jzprime.a(alpha,beta), local_Sz, tight_Id, nextn_Sz);
                    
                    Terms.push_nextn(loc, +Dzprime.a(alpha,beta), local_Sy, tight_Id, nextn_Sx);
                    Terms.push_nextn(loc, -Dxprime.a(alpha,beta), local_Sy, tight_Id, nextn_Sz);
                    Terms.push_nextn(loc, +Dxprime.a(alpha,beta), local_Sz, tight_Id, nextn_Sy);
                    Terms.push_nextn(loc, -Dzprime.a(alpha,beta), local_Sx, tight_Id, nextn_Sy);
                }
            }
        }
    }
}


}

#endif
