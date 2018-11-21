#ifndef STRAWBERRY_TRANSVERSEKONDOMODEL
#define STRAWBERRY_TRANSVERSEKONDOMODEL

#include "models/KondoU1xU1.h"
//include "symmetry/U1.h"

namespace VMPS
{
/** \class KondoU1
 * \ingroup Kondo
 *
 * \brief Kondo Model
 *
 * MPO representation of 
 * \f[
 * H = -t\sum_{<ij>\sigma} \left(c^\dagger_{i\sigma}c_{j\sigma} + h.c.\right)
 * - J \sum_{i \in I} \mathbf{S}_i \cdot \mathbf{s}_i
 * - \sum_{i \in I} B_i^x S_i^x
 * - \sum_{i \in I} B_i^z S_i^z
 * \f]
 *
 * where further parameters from Hubbard and Heisenberg are possible.
 *  \param D : \f$D=2S+1\f$ where \f$S\f$ is the spin of the impurity.
 * 
 * \note Take use of the U(1) particle conservation symmetry.
 * \note The \f$S_z\f$ U(1) symmetry is borken due to the field in x-direction.
 * \note The default variable settings can be seen in \p KondoU1::defaults.
 * \note \f$J<0\f$ is antiferromagnetic
 * \note If nnn-hopping is positive, the GS-energy is lowered.
 * \note The multi-impurity model can be received, by setting D=1 (S=0) for all sites without an impurity.
 */
class KondoU1 : public Mpo<Sym::U1<Sym::ChargeU1>,double>, public KondoObservables<Sym::U1<Sym::ChargeU1> >, public ParamReturner
{
public:
	typedef Sym::U1<Sym::ChargeU1> Symmetry;
	MAKE_TYPEDEFS(KondoU1)
	
private:
	typedef typename Symmetry::qType qType;
	
public:
	
	///@{
	KondoU1 () : Mpo(){};
	KondoU1 (const size_t &L, const vector<Param> &params);
	///@}
	
	static qarray<1> singlet (int N) {return qarray<1>{N};}; // not a real singlet, but useful for consistency when switching symmetries
	
	template<typename Symmetry_>
	static void add_operators (const vector<SpinBase<Symmetry_> > &B, 
	                           const vector<FermionBase<Symmetry_> > &F, 
	                           const ParamHandler &P,
	                           HamiltonianTermsXd<Symmetry_> &Terms);
	
	/**Validates whether a given \p qnum is a valid combination of \p N and \p M for the given model.
	\returns \p true if valid, \p false if not*/
	bool validate (qType qnum) const;
	
	static const std::map<string,std::any> defaults;
};

const std::map<string,std::any> KondoU1::defaults = 
{
	{"t",1.}, {"tPrime",0.}, {"tRung",0.},
	{"J",1.}, 
	{"U",0.},
	{"V",0.}, {"Vrung",0.}, 
	{"mu",0.}, {"t0",0.},
	{"Bz",0.}, {"Bx",0.}, {"Bzsub",0.}, {"Bxsub",0.}, {"Kz",0.}, {"Kx",0.},
	{"Inext",0.}, {"Iprev",0.}, {"I3next",0.}, {"I3prev",0.}, {"I3loc",0.},
	{"D",2ul}, {"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}, {"LyF",1ul}
};

KondoU1::
KondoU1 (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, qarray<Symmetry::Nq>({0}), "", PROP::HERMITIAN, PROP::NON_UNITARY, PROP::HAMILTONIAN),
 KondoObservables(L,params,defaults),
 ParamReturner()
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
//	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l);
	}
	
//	for (size_t l=0; l<N_sites; ++l)
//	{
//		Terms[l] = KondoU1xU1::set_operators(B,F,P,l%Lcell);
//		add_operators(Terms[l],B,F,P,l%Lcell);
//		
//		stringstream ss;
//		ss << "Ly=" << P.get<size_t>("Ly",l%Lcell);
//		Terms[l].info.push_back(ss.str());
//	}
	
	HamiltonianTermsXd<Symmetry> Terms(N_sites);
	KondoU1xU1::set_operators(B,F,P,Terms);
	add_operators(B,F,P,Terms);
	
	this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
	this->precalc_TwoSiteData();
}

bool KondoU1::
validate (qType qnum) const
{
	if (qnum[0]<=2*static_cast<int>(this->N_phys) and qnum[0]>0) {return true;}
	else {return false;}
}

template<typename Symmetry_>
void KondoU1::
add_operators (const vector<SpinBase<Symmetry_> > &B, const vector<FermionBase<Symmetry_> > &F, 
               const ParamHandler &P, HamiltonianTermsXd<Symmetry_> &Terms)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = Terms.size();
	
	Terms.set_name("Transverse-field Kondo");
	
	for (std::size_t loc=0; loc<N_sites; ++loc)
	{
		// Bx substrate
		param1d Bxsub = P.fill_array1d<double>("Bxsub", "Bxsuborb", F[loc].orbitals(), loc%Lcell);
		Terms.save_label(loc, Bxsub.label);
		
		// Bx impurities
		param1d Bx = P.fill_array1d<double>("Bx", "Bxorb", B[loc].orbitals(), loc%Lcell);
		Terms.save_label(loc, Bx.label);
		
		// Kx anisotropy
		param1d Kx = P.fill_array1d<double>("Kx", "Kxorb", B[loc].orbitals(), loc%Lcell);
		Terms.save_label(loc, Kx.label);
		
		ArrayXXd Jxyperp = B[loc].ZeroHopping();
		ArrayXXd Jzperp  = B[loc].ZeroHopping();
		ArrayXd  Bzorb   = B[loc].ZeroField();
		ArrayXd  Kzorb   = B[loc].ZeroField();
		ArrayXXd Dyperp  = B[loc].ZeroHopping();
		
		ArrayXd Uorb     = F[loc].ZeroField();
		ArrayXd Eorb     = F[loc].ZeroField();
		ArrayXd Bzsuborb = F[loc].ZeroField();
		ArrayXXd tPerp   = F[loc].ZeroHopping();
		ArrayXXd Vperp   = F[loc].ZeroHopping();
		ArrayXXd Jperp   = F[loc].ZeroHopping();
		
		auto Himp = kroneckerProduct(B[loc].HeisenbergHamiltonian(Jxyperp,Jzperp,Bzorb,Bx.a,Kzorb,Kx.a,Dyperp), F[loc].Id());
		auto Hsub = kroneckerProduct(B[loc].Id(), F[loc].HubbardHamiltonian(Uorb,Eorb,Bzsuborb,Bxsub.a,tPerp,Vperp,Jperp));
		
		Terms.push_local(loc, 1., Himp+Hsub);
	}
}

//template<typename Symmetry_>
//void KondoU1::
//add_operators (HamiltonianTermsXd<Symmetry_> &Terms, 
//               const vector<SpinBase<Symmetry_> > &B, const vector<FermionBase<Symmetry_> > &F, 
//               const ParamHandler &P, size_t loc)
//{
//	auto save_label = [&Terms] (string label)
//	{
//		if (label!="") {Terms.info.push_back(label);}
//	};
//	
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
//	ArrayXXd Jxyperp = B[loc].ZeroHopping();
//	ArrayXXd Jzperp  = B[loc].ZeroHopping();
//	ArrayXd  Bzorb   = B[loc].ZeroField();
//	ArrayXd  Kzorb   = B[loc].ZeroField();
//	ArrayXXd Dyperp  = B[loc].ZeroHopping();
//	
//	ArrayXd Uorb     = F[loc].ZeroField();
//	ArrayXd Eorb     = F[loc].ZeroField();
//	ArrayXd Bzsuborb = F[loc].ZeroField();
//	ArrayXXd tPerp   = F[loc].ZeroHopping();
//	ArrayXXd Vperp   = F[loc].ZeroHopping();
//	ArrayXXd Jperp   = F[loc].ZeroHopping();
//	
//	auto Himp = kroneckerProduct(B[loc].HeisenbergHamiltonian(Jxyperp,Jzperp,Bzorb,Bxorb,Kzorb,Kxorb,Dyperp), F[loc].Id());
//	auto Hsub = kroneckerProduct(B[loc].Id(), F[loc].HubbardHamiltonian(Uorb,Eorb,Bzsuborb,Bxsuborb,tPerp,Vperp,Jperp));
//	
//	Terms.local.push_back(make_tuple(1.,Himp+Hsub));
//	
//	Terms.name = "Transverse-field Kondo";
//}

};

#endif
