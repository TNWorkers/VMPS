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
      -J_{\alpha} \sum_{<ij>} \left(S^{\alpha}_i \cdot S^{\alpha}_j\right) 
      -J'_{\alpha} \sum_{<<ij>>} \left(S^{\alpha}_i \cdot S^{\alpha}_j\right) 
      -B_{\alpha} \sum_i S^{\alpha}_i
      +K_{\alpha} \sum_i \left(S^{\alpha}_i\right)^2
      \right)
      -\mathbf{D} \sum_{<ij>} \left(\mathbf{S}_i \times \mathbf{S}_j\right) 
      -\mathbf{D'} \sum_{<<ij>>} \left(\mathbf{S}_i \times \mathbf{S}_j\right) 
  \f]
  *
  \param D : \f$D=2S+1\f$ where \f$S\f$ is the spin
  \note Uses no symmetry. Any parameter constellations are allowed. For variants with symmetries, see VMPS::HeisenbergU1 or VMPS::HeisenbergSU2.
  \note The default variable settings can be seen in \p HeisenbergXYZ::defaults.
  \note \f$J<0\f$ is antiferromagnetic
  \note Due to the \f$S_y\f$ operator, this MPO is complex.
*/
class HeisenbergXYZ : public MpoQ<Sym::U0,complex<double> >
{
public:
	typedef Sym::U0 Symmetry;
	
private:
	typedef typename Symmetry::qType qType;
	
public:
	
	///\{
	HeisenbergXYZ() : MpoQ<Symmetry,complex<double> >() {};
	HeisenbergXYZ (const size_t &L, const vector<Param> &params);
	///\}
	
//	///@{
//	/**Observables.*/
//	MpoQ<Symmetry> SzSz (size_t loc1, size_t loc2);
//	MpoQ<Symmetry> Sz   (size_t loc);
//	///@}
	
	template<typename Symmetry_>
	void add_operators (HamiltonianTerms<Symmetry_,complex<double> > &Terms, const SpinBase<Symmetry_> &B, const ParamHandler &P, size_t loc=0);
	
	static const std::map<string,std::any> defaults;
	
protected:
	
	SpinBase<Symmetry> B;
};

const std::map<string,std::any> HeisenbergXYZ::defaults = 
{
	{"Jx",0.}, {"Jy",0.}, {"Jz",0.},
	{"Jxperp",0.}, {"Jyperp",0.}, {"Jzperp",0.},
	{"Jxprime",0.}, {"Jyprime",0.}, {"Jzprime",0.},
	
	 // Dzialoshinsky-Moriya terms
	{"Dx",0.}, {"Dy",0.}, {"Dz",0.},
	{"Dxperp",0.}, {"Dyperp",0.}, {"Dzperp",0.},
	{"Dxprime",0.}, {"Dyprime",0.}, {"Dzprime",0.},
	
	{"Bx",0.}, {"By",0.}, {"Bz",0.},
	{"Kx",0.}, {"Ky",0.}, {"Kz",0.},
	
	{"D",2ul}, {"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1},
	
	{"J",0.}, {"Jprime",0.}, {"Jperp",0.}
};

HeisenbergXYZ::
HeisenbergXYZ (const size_t &L, const vector<Param> &params)
:MpoQ<Symmetry,complex<double> > (L, qarray<0>({}), labeldummy, "")
{
	ParamHandler P(params,HeisenbergXYZ::defaults);
	
	size_t Lcell = P.size();
	vector<SuperMatrix<Symmetry,complex<double> > > G;
	vector<HamiltonianTerms<Symmetry,complex<double> > > Terms(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		
		B = SpinBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), P.get<size_t>("D",l%Lcell));
		setLocBasis(B.get_basis(),l);
		
		auto Terms_tmp = HeisenbergU1::set_operators(B,P,l%Lcell);
		Heisenberg::add_operators(Terms_tmp,B,P,l%Lcell);
		Terms[l] = Terms_tmp.cast<complex<double> >();
		add_operators(Terms[l],B,P,l%Lcell);
		
		this->Daux = Terms[l].auxdim();
		
		G.push_back(Generator(Terms[l]));
		setOpBasis(G[l].calc_qOp(),l);
	}
	
	this->generate_label(Terms[0].name,Terms,Lcell);
	this->construct(G, this->W, this->Gvec, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
}

template<typename Symmetry_>
void HeisenbergXYZ::
add_operators (HamiltonianTerms<Symmetry_,complex<double> > &Terms, const SpinBase<Symmetry_> &B, const ParamHandler &P, size_t loc)
{
	auto save_label = [&Terms] (string label)
	{
		if (label!="") {Terms.info.push_back(label);}
	};
	
	// J terms
	
	auto [Jx,Jxpara,Jxlabel] = P.fill_array2d<double>("Jx","Jxpara",B.orbitals(),loc);
	save_label(Jxlabel);
	
	auto [Jy,Jypara,Jylabel] = P.fill_array2d<double>("Jy","Jypara",B.orbitals(),loc);
	save_label(Jylabel);
	
	auto [Jz,Jzpara,Jzlabel] = P.fill_array2d<double>("Jz","Jzpara",B.orbitals(),loc);
	save_label(Jzlabel);
	
	for (int i=0; i<B.orbitals(); ++i)
	for (int j=0; j<B.orbitals(); ++j)
	{
		if (Jxpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(Jxpara(i,j), B.Scomp(SX,i).template cast<complex<double> >(), 
			                                              B.Scomp(SX,j).template cast<complex<double> >()));
		}
		if (Jypara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(Jypara(i,j), -1.i*B.Scomp(iSY,i).template cast<complex<double> >(), 
			                                              -1.i*B.Scomp(iSY,j).template cast<complex<double> >()));
		}
		if (Jzpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(Jzpara(i,j), B.Scomp(SZ,i).template cast<complex<double> >(), 
			                                              B.Scomp(SZ,j).template cast<complex<double> >()));
		}
	}
	
	// DM terms
	
	auto [Dx,Dxpara,Dxlabel] = P.fill_array2d<double>("Dx","Dxpara",B.orbitals(),loc);
	save_label(Dxlabel);
	
	auto [Dz,Dzpara,Dzlabel] = P.fill_array2d<double>("Dz","Dzpara",B.orbitals(),loc);
	save_label(Dzlabel);
	
	for (int i=0; i<B.orbitals(); ++i)
	for (int j=0; j<B.orbitals(); ++j)
	{
		if (Dxpara(i,j)!=0. or Dzpara(i,j)!=0.)
		{
			SiteOperator<Symmetry_,complex<double> > Sxi = B.Scomp(SX,i).template cast<complex<double> >();
			SiteOperator<Symmetry_,complex<double> > Sxj = B.Scomp(SX,j).template cast<complex<double> >();
			SiteOperator<Symmetry_,complex<double> > Syi = -1.i*B.Scomp(iSY,i).template cast<complex<double> >();
			SiteOperator<Symmetry_,complex<double> > Syj = -1.i*B.Scomp(iSY,j).template cast<complex<double> >();
			SiteOperator<Symmetry_,complex<double> > Szi = B.Scomp(SZ,i).template cast<complex<double> >();
			SiteOperator<Symmetry_,complex<double> > Szj = B.Scomp(SZ,j).template cast<complex<double> >();
			
			Terms.tight.push_back(make_tuple(1., Syi, +Dzpara(i,j)*Sxj-Dxpara(i,j)*Szj));
			Terms.tight.push_back(make_tuple(1., +Dxpara(i,j)*Szi-Dzpara(i,j)*Sxi, Syj));
		}
	}
	
	// NNN terms
	
	param0d Dxprime = P.fill_array0d<double>("Dxprime","Dxprime",loc);
	save_label(Dxprime.label);
	
	param0d Dzprime = P.fill_array0d<double>("Dzprime","Dzprime",loc);
	save_label(Dzprime.label);
	
	if (Dxprime.x!=0. or Dzprime.x!=0.)
	{
		assert(B.orbitals() == 1 and "Cannot do a ladder with Dx'/Dz' terms!");
		
		SiteOperator<Symmetry_,complex<double> > Sx = B.Scomp(SX).template cast<complex<double> >();
		SiteOperator<Symmetry_,complex<double> > Sy = -1.i*B.Scomp(iSY).template cast<complex<double> >();
		SiteOperator<Symmetry_,complex<double> > Sz = B.Scomp(SZ).template cast<complex<double> >();
		SiteOperator<Symmetry_,complex<double> > Id = B.Id().template cast<complex<double> >();
		
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
		assert(B.orbitals() == 1 and "Cannot do a ladder with Jx' terms!");
		Terms.nextn.push_back(make_tuple(Jxprime.x, B.Scomp(SX).template cast<complex<double> >(), 
		                                            B.Scomp(SX).template cast<complex<double> >(), 
		                                            B.Id().template cast<complex<double> >()));
	}
	
	if (Jyprime.x != 0.)
	{
		assert(B.orbitals() == 1 and "Cannot do a ladder with Jy' terms!");
		Terms.nextn.push_back(make_tuple(Jxprime.x, -1.i*B.Scomp(iSY).template cast<complex<double> >(), 
		                                            -1.i*B.Scomp(iSY).template cast<complex<double> >(), 
		                                                 B.Id().template cast<complex<double> >()));
	}
	
	if (Jzprime.x != 0.)
	{
		assert(B.orbitals() == 1 and "Cannot do a ladder with Jz' terms!");
		Terms.nextn.push_back(make_tuple(Jzprime.x, -1.i*B.Scomp(SZ).template cast<complex<double> >(), 
		                                            -1.i*B.Scomp(SZ).template cast<complex<double> >(), 
		                                                 B.Id().template cast<complex<double> >()));
	}
	
	// local terms
	
	param0d Jxperp = P.fill_array0d<double>("Jx","Jxperp",loc);
	save_label(Jxperp.label);
	
	param0d Jyperp = P.fill_array0d<double>("Jy","Jyperp",loc);
	save_label(Jyperp.label);
	
	param0d Jzperp = P.fill_array0d<double>("Jz","Jzperp",loc);
	save_label(Jzperp.label);
	
	param0d Dxperp = P.fill_array0d<double>("Dx","Dxperp",loc);
	save_label(Dxperp.label);
	
	param0d Dzperp = P.fill_array0d<double>("Dz","Dzperp",loc);
	save_label(Dzperp.label);
	
	auto [By,Byorb,Bylabel] = P.fill_array1d<double>("By","Byorb",B.orbitals(),loc);
	save_label(Bylabel);
	
	auto [Ky,Kyorb,Kylabel] = P.fill_array1d<double>("Ky","Kyorb",B.orbitals(),loc);
	save_label(Kylabel);
	
	Terms.name = (P.HAS_ANY_OF({"Dx","Dy","Dz","Dxprime","Dyprime","Dzprime","Dxpara","Dypara","Dzpara"},loc))? 
	"Dzyaloshinsky-Moriya":"HeisenbergXYZ";
	
	Array3d Jorb; Jorb << Jxperp.x, Jyperp.x, Jzperp.x;
	Array<double,Dynamic,3> Borb(B.orbitals(),3); Borb=0.; Borb.col(1)=Byorb;
	Array<double,Dynamic,3> Korb(B.orbitals(),3); Korb=0.; Korb.col(1)=Kyorb;
	Array3d Dorb; Dorb << Dxperp.x, 0., Dzperp.x;
	
	Terms.local.push_back(make_tuple(1., B.HeisenbergHamiltonian(Jorb,Borb,Korb,Dorb, P.get<bool>("CYLINDER"))));
}

}

#endif
