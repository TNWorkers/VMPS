#ifndef HUBBARDMODELSU2XU1_H_
#define HUBBARDMODELSU2XU1_H_

#include "symmetry/S1xS2.h"
#include "symmetry/U1.h"
#include "symmetry/SU2.h"
#include "bases/FermionBaseSU2xU1.h"
//include "tensors/SiteOperatorQ.h"
//include "tensors/SiteOperator.h"
#include "Mpo.h"
//include "DmrgExternal.h"
//include "ParamHandler.h"
#include "ParamReturner.h"
#include "Geometry2D.h" // from TOOLS

namespace VMPS
{

/** \class HubbardSU2xU1
  * \ingroup Hubbard
  *
  * \brief Hubbard Model
  *
  * MPO representation of 
  * 
  * \f$
  * H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
  * - t^{\prime} \sum_{<<ij>>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
  * + U \sum_i n_{i\uparrow} n_{i\downarrow}
  * + V \sum_{<ij>} n_{i} n_{j}
  * - X \sum_{<ij>\sigma} \left( c^\dagger_{i\sigma}c_{j\sigma} + h.c.\right) \left(n_{i,-\sigma}-n_{j,-\sigma}\right)^2
  * +H_{tJ}
  * \f$.
  * with
  * \f[
  * H_{tJ} = +J \sum_{<ij>} (\mathbf{S}_{i} \mathbf{S}_{j} - \frac{1}{4} n_in_j)
  * \f]
  * \note: The term before \f$n_i n_j\f$ is not set and has to be adjusted with \p V
  * \note Makes use of the spin-SU(2) symmetry and the U(1) charge symmetry.
  * \note If the nnn-hopping is positive, the ground state energy is lowered.
  * \warning \f$J>0\f$ is antiferromagnetic
  */
class HubbardSU2xU1 : public Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > ,double>, public ParamReturner
{
public:
	
	typedef Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > Symmetry;
	MAKE_TYPEDEFS(HubbardSU2xU1)
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	
private:
	
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	
public:
	
	HubbardSU2xU1() : Mpo(){};
	HubbardSU2xU1 (const size_t &L, const vector<Param> &params);
	
	//static HamiltonianTermsXd<Symmetry> set_operators (const vector<FermionBase<Symmetry> > &F, const ParamHandler &P, size_t loc=0);
	static void set_operators(const std::vector<FermionBase<Symmetry>> &F, const ParamHandler &P, HamiltonianTermsXd<Symmetry> &Terms);
	
	static qarray<2> singlet (int N) {return qarray<2>{1,N};};
	
	///@{
	Mpo<Symmetry> c (size_t locx, size_t locy=0, double factor=1.) const;
	Mpo<Symmetry> cdag (size_t locx, size_t locy=0, double factor=sqrt(2.)) const;
	Mpo<Symmetry> n (size_t locx, size_t locy=0) const;
	Mpo<Symmetry> d (size_t locx, size_t locy=0) const;
	Mpo<Symmetry> nh (size_t locx, size_t locy=0) const;
	Mpo<Symmetry> ns (size_t locx, size_t locy=0) const;
	///@}
	
	///@{
	Mpo<Symmetry> cc (size_t locx, size_t locy=0) const;
	Mpo<Symmetry> cdagcdag (size_t locx, size_t locy=0) const;
	Mpo<Symmetry> dh_excitation (size_t locx) const;
	///@}
	
	///@{
	Mpo<Symmetry> S (size_t locx, size_t locy=0) const;
	Mpo<Symmetry> Sdag (size_t locx, size_t locy=0, double factor=sqrt(3.)) const;
	///@}
	
	///@{
	Mpo<Symmetry> cdagc (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	Mpo<Symmetry> nn    (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	Mpo<Symmetry> SdagS (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	Mpo<Symmetry> TzTz  (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	Mpo<Symmetry> TpTm  (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	Mpo<Symmetry> TmTp  (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	Mpo<Symmetry> Tz    (size_t locx, size_t locy=0) const;
	Mpo<Symmetry> Tp    (size_t locx, size_t locy=0, double factor=1.) const;
	Mpo<Symmetry> Tm    (size_t locx, size_t locy=0, double factor=1.) const;
	///@}
	
	Mpo<Symmetry,complex<double> > S_ky    (vector<complex<double> > phases) const;
	Mpo<Symmetry,complex<double> > Sdag_ky (vector<complex<double> > phases, double factor=sqrt(3.)) const;
	Mpo<Symmetry,complex<double> > T_ky    (vector<complex<double> > phases) const;
	Mpo<Symmetry,complex<double> > Tdag_ky (vector<complex<double> > phases, double factor=1.) const;
	Mpo<Symmetry,complex<double> > c_ky    (vector<complex<double> > phases, double factor=sqrt(2.)) const;
	Mpo<Symmetry,complex<double> > cdag_ky (vector<complex<double> > phases, double factor=sqrt(2.)) const;
	
	static const map<string,any> defaults;
	static const map<string,any> sweep_defaults;
	
protected:
	
	Mpo<Symmetry>
	make_local (string name, 
	            size_t locx, size_t locy, 
	            const OperatorType &Op, 
	            double factor, bool FERMIONIC, bool HERMITIAN) const;

	Mpo<Symmetry,complex<double> >
	make_FourierYSum (string name, const vector<OperatorType> &Ops, double factor, bool HERMITIAN, const vector<complex<double> > &phases) const;
	
	vector<FermionBase<Symmetry> > F;
};

// V is standard next-nearest neighbour density interaction
// Vz and Vxy are anisotropic isospin-isospin next-nearest neighbour interaction
const map<string,any> HubbardSU2xU1::defaults = 
{
	{"t",1.}, {"tPrime",0.}, {"tRung",1.}, {"tPrimePrime",0.},
	{"mu",0.}, {"t0",0.}, 
	{"U",0.}, {"Uph",0.},
	{"V",0.}, {"Vext",0.}, {"Vrung",0.},
	{"Vz",0.}, {"Vzrung",0.}, {"Vxy",0.}, {"Vxyrung",0.}, 
	{"J",0.}, {"Jperp",0.},
	{"X",0.}, {"Xrung",0.},
	{"CALC_SQUARE",false}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}
};

const map<string,any> HubbardSU2xU1::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.}, {"lim_alpha",11ul}, {"eps_svd",1e-7},
	{"Dincr_abs", 4ul}, {"Dincr_per", 2ul}, {"Dincr_rel", 1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",24ul}, {"min_halfsweeps",6ul},
	{"Dinit",8ul}, {"Qinit",10ul}, {"Dlimit",500ul},
	{"tol_eigval",1e-7}, {"tol_state",1e-6},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
};

HubbardSU2xU1::
HubbardSU2xU1 (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, qarray<Symmetry::Nq>({1,0}), "", PROP::HERMITIAN, PROP::NON_UNITARY, PROP::HAMILTONIAN),
 ParamReturner(HubbardSU2xU1::sweep_defaults)
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
	HamiltonianTermsXd<Symmetry> Terms(N_sites, P.get<bool>("OPEN_BC"));
	F.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		
		F[l] = FermionBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), !isfinite(P.get<double>("U",l%Lcell)), !isfinite(P.get<double>("Uph",l%Lcell)));
		setLocBasis(F[l].get_basis().qloc(),l);
	}
	
	set_operators(F,P,Terms);
	
	this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
	this->precalc_TwoSiteData();
}

void HubbardSU2xU1::
set_operators (const std::vector<FermionBase<Symmetry> > &F, const ParamHandler &P, HamiltonianTermsXd<Symmetry> &Terms)
{
	std::size_t Lcell = P.size();
	std::size_t N_sites = Terms.size();
	Terms.set_name("Hubbard");
	
	for(std::size_t loc=0; loc<N_sites; ++loc)
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
		Terms.save_label(loc, ss.str());
		
		if (P.HAS("tFull"))
		{
			ArrayXXd Full = P.get<Eigen::ArrayXXd>("tFull");
			vector<vector<std::pair<size_t,double> > > R = Geometry2D::rangeFormat(Full);
			
			if (P.get<bool>("OPEN_BC")) {assert(R.size() ==   N_sites and "Use an (N_sites)x(N_sites) hopping matrix for open BC!");}
			else                        {assert(R.size() >= 2*N_sites and "Use at least a (2*N_sites)x(N_sites) hopping matrix for infinite BC!");}
			
			for (size_t h=0; h<R[loc].size(); ++h)
			{
				size_t range = R[loc][h].first;
				double value = R[loc][h].second;
				
				size_t Ntrans = (range == 0)? 0:range-1;
				vector<SiteOperator<Symmetry,double> > TransOps(Ntrans);
				for (size_t i=0; i<Ntrans; ++i)
				{
					TransOps[i] = F[(loc+i+1)%N_sites].sign().plain<double>();
				}
				
				if (range != 0)
				{
					SiteOperator<Symmetry,double> c_sign_local = OperatorType::prod(F[loc].c(0), F[loc].sign(), {2,-1}).plain<double>();
					SiteOperator<Symmetry,double> cdag_sign_local = OperatorType::prod(F[loc].cdag(0), F[loc].sign(), {2,1}).plain<double>();
					SiteOperator<Symmetry,double> c_range = F[(loc+range)%N_sites].c(0).plain<double>();
					SiteOperator<Symmetry,double> cdag_range = F[(loc+range)%N_sites].cdag(0).plain<double>();
					
					//hopping
					//cout << "loc=" << loc << ", pushing at range=" << range << ", value=" << value << endl;
					Terms.push(range, loc, -value * std::sqrt(2.), cdag_sign_local, TransOps, c_range);
					Terms.push(range, loc, -value * std::sqrt(2.), c_sign_local, TransOps, cdag_range);
				}
			}
			
			stringstream ss;
			ss << "tᵢⱼ(" << Geometry2D::hoppingInfo(Full) << ")";
			Terms.save_label(loc,ss.str());
		}
		
		if (P.HAS("Vzfull"))
		{
			ArrayXXd Full = P.get<Eigen::ArrayXXd>("Vzfull");
			vector<vector<std::pair<size_t,double> > > R = Geometry2D::rangeFormat(Full);
			
			if (P.get<bool>("OPEN_BC")) {assert(R.size() ==   N_sites and "Use an (N_sites)x(N_sites) hopping matrix for open BC!");}
			else                        {assert(R.size() >= 2*N_sites and "Use at least a (2*N_sites)x(N_sites) hopping matrix for infinite BC!");}
			
			for (size_t h=0; h<R[loc].size(); ++h)
			{
				size_t range = R[loc][h].first;
				double value = R[loc][h].second;
				
				size_t Ntrans = (range == 0)? 0:range-1;
				vector<SiteOperator<Symmetry,double> > TransOps(Ntrans);
				for (size_t i=0; i<Ntrans; ++i)
				{
					TransOps[i] = F[(loc+i+1)%N_sites].Id().plain<double>();
				}
				
				if (range != 0)
				{
					
					auto Tz_loc = F[loc].Tz(0);
					auto Tz_hop = F[(loc+range)%N_sites].Tz(0);
					
					Terms.push(range, loc, value, Tz_loc.plain<double>(), TransOps, Tz_hop.plain<double>());
				}
			}
			
			stringstream ss;
			ss << "Vzᵢⱼ(" << Geometry2D::hoppingInfo(Full) << ")";
			Terms.save_label(loc,ss.str());
		}
		
		if (P.HAS("Vxyfull"))
		{
			ArrayXXd Full = P.get<Eigen::ArrayXXd>("Vxyfull");
			vector<vector<std::pair<size_t,double> > > R = Geometry2D::rangeFormat(Full);
			
			if (P.get<bool>("OPEN_BC")) {assert(R.size() ==   N_sites and "Use an (N_sites)x(N_sites) hopping matrix for open BC!");}
			else                        {assert(R.size() >= 2*N_sites and "Use at least a (2*N_sites)x(N_sites) hopping matrix for infinite BC!");}
			
			for (size_t h=0; h<R[loc].size(); ++h)
			{
				size_t range = R[loc][h].first;
				double value = R[loc][h].second;
				
				size_t Ntrans = (range == 0)? 0:range-1;
				vector<SiteOperator<Symmetry,double> > TransOps(Ntrans);
				for (size_t i=0; i<Ntrans; ++i)
				{
					TransOps[i] = F[(loc+i+1)%N_sites].Id().plain<double>();
				}
				
				if (range != 0)
				{
					//The sign is hardcoded here.. maybe include this in Geometry class.
					auto Tp_loc    = pow(-1,loc) * F[loc].cc(0);
					auto Tm_hop    = pow(-1,(loc+range)%N_sites) * F[(loc+range)%N_sites].cdagcdag(0);
					auto Tm_loc    = pow(-1,loc) * F[loc].cdagcdag(0);
					auto Tp_hop    = pow(-1,(loc+range)%N_sites) * F[(loc+range)%N_sites].cc(0);
					
					Terms.push(range, loc, 0.5 * value, Tp_loc.plain<double>(), TransOps, Tm_hop.plain<double>());
					Terms.push(range, loc, 0.5 * value, Tm_loc.plain<double>(), TransOps, Tp_hop.plain<double>());
				}
			}
			
			stringstream ss;
			ss << "Vxyᵢⱼ(" << Geometry2D::hoppingInfo(Full) << ")";
			Terms.save_label(loc,ss.str());
		}
		
		if (P.HAS("VextFull"))
		{
			ArrayXXd Full = P.get<Eigen::ArrayXXd>("VextFull");
			vector<vector<std::pair<size_t,double> > > R = Geometry2D::rangeFormat(Full);
			
			if (P.get<bool>("OPEN_BC")) {assert(R.size() ==   N_sites and "Use an (N_sites)x(N_sites) hopping matrix for open BC!");}
			else                        {assert(R.size() >= 2*N_sites and "Use at least a (2*N_sites)x(N_sites) hopping matrix for infinite BC!");}
			
			for (size_t h=0; h<R[loc].size(); ++h)
			{
				size_t range = R[loc][h].first;
				double value = R[loc][h].second;
				
				size_t Ntrans = (range == 0)? 0:range-1;
				vector<SiteOperator<Symmetry,double> > TransOps(Ntrans);
				for (size_t i=0; i<Ntrans; ++i)
				{
					TransOps[i] = F[(loc+i+1)%N_sites].Id().plain<double>();
				}
				
				if (range != 0)
				{
					
					auto n_loc = F[loc].n(0);
					auto n_hop = F[(loc+range)%N_sites].n(0);
					
					Terms.push(range, loc, value, n_loc.plain<double>(), TransOps, n_hop.plain<double>());
				}
			}
			
			stringstream ss;
			ss << "Vextᵢⱼ(" << Geometry2D::hoppingInfo(Full) << ")";
			Terms.save_label(loc,ss.str());
		}
		
		if (P.HAS("Jfull"))
		{
			ArrayXXd Full = P.get<Eigen::ArrayXXd>("Jfull");
			vector<vector<std::pair<size_t,double> > > R = Geometry2D::rangeFormat(Full);
			
			if (P.get<bool>("OPEN_BC")) {assert(R.size() ==   N_sites and "Use an (N_sites)x(N_sites) hopping matrix for open BC!");}
			else                        {assert(R.size() >= 2*N_sites and "Use at least a (2*N_sites)x(N_sites) hopping matrix for infinite BC!");}
			
			for (size_t h=0; h<R[loc].size(); ++h)
			{
				size_t range = R[loc][h].first;
				double value = R[loc][h].second;
				
				size_t Ntrans = (range == 0)? 0:range-1;
				vector<SiteOperator<Symmetry,double> > TransOps(Ntrans);
				for (size_t i=0; i<Ntrans; ++i)
				{
					TransOps[i] = F[(loc+i+1)%N_sites].Id().plain<double>();
				}
				
				if (range != 0)
				{
					auto Sdag_loc = F[loc].Sdag(0);
					auto S_hop    = F[(loc+range)%N_sites].S(0);
					
					Terms.push(range, loc, std::sqrt(3.) * value,
					           Sdag_loc.plain<double>(), TransOps, S_hop.plain<double>());
				}
			}
			
			stringstream ss;
			ss << "Jᵢⱼ(" << Geometry2D::hoppingInfo(Full) << ")";
			Terms.save_label(loc,ss.str());
		}
		
		if (P.HAS("Xfull"))
		{
			ArrayXXd Full = P.get<Eigen::ArrayXXd>("Xfull");
			vector<vector<std::pair<size_t,double> > > R = Geometry2D::rangeFormat(Full);
			
			if (P.get<bool>("OPEN_BC")) {assert(R.size() ==   N_sites and "Use an (N_sites)x(N_sites) hopping matrix for open BC!");}
			else                        {assert(R.size() >= 2*N_sites and "Use at least a (2*N_sites)x(N_sites) hopping matrix for infinite BC!");}
			
			for (size_t h=0; h<R[loc].size(); ++h)
			{
				size_t range = R[loc][h].first;
				double value = R[loc][h].second;
				
				size_t Ntrans = (range == 0)? 0:range-1;
				vector<SiteOperator<Symmetry,double> > TransOps(Ntrans);
				for (size_t i=0; i<Ntrans; ++i)
				{
					TransOps[i] = F[(loc+i+1)%N_sites].sign().plain<double>();
				}
				
				if (range != 0)
				{
					size_t ran = (loc+range)%N_sites;
					
					SiteOperator<Symmetry,double> PsiLloc = OperatorType::prod(F[loc].ns(),
					                                        OperatorType::prod(F[loc].c(), F[loc].sign(), {2,-1}),
					                                        {2,-1}).plain<double>();
					SiteOperator<Symmetry,double> PsiRloc = OperatorType::prod(
					                                        OperatorType::prod(F[loc].c(), F[loc].sign(), {2,-1}),
					                                        F[loc].ns(),
					                                        {2,-1}).plain<double>();
					SiteOperator<Symmetry,double> PsiLran = OperatorType::prod(F[ran].ns(), F[ran].c(),
					                                        {2,-1}).plain<double>();
					SiteOperator<Symmetry,double> PsiRran = OperatorType::prod(F[ran].c(), F[ran].ns(),
					                                        {2,-1}).plain<double>();
					SiteOperator<Symmetry,double> PsidagLloc = OperatorType::prod(
					                                           OperatorType::prod(F[loc].cdag(), F[loc].sign(), {2,1}),
					                                           F[loc].ns(),
					                                           {2,1}).plain<double>();
					SiteOperator<Symmetry,double> PsidagRloc = OperatorType::prod(F[loc].ns(),
					                                           OperatorType::prod(F[loc].cdag(), F[loc].sign(), {2,1}),
					                                           {2,1}).plain<double>();
					SiteOperator<Symmetry,double> PsidagLran = OperatorType::prod(F[ran].cdag(), F[ran].ns(),
					                                           {2,1}).plain<double>();
					SiteOperator<Symmetry,double> PsidagRran = OperatorType::prod(F[ran].ns(), F[ran].cdag(),
					                                           {2,1}).plain<double>();
					
					//hopping
					Terms.push(range, loc, -value * std::sqrt(2.), PsidagLloc, TransOps, PsiRran);
					Terms.push(range, loc, -value * std::sqrt(2.), PsidagRloc, TransOps, PsiLran);
					Terms.push(range, loc, -value * std::sqrt(2.), PsiLloc, TransOps, PsidagRran); // why no sign flip?
					Terms.push(range, loc, -value * std::sqrt(2.), PsiRloc, TransOps, PsidagLran); // why no sign flip?
				}
			}
			
			stringstream ss;
			ss << "Xᵢⱼ(" << Geometry2D::hoppingInfo(Full) << ")";
			Terms.save_label(loc,ss.str());
		}
		
		// Local terms: U, t0, μ, t⟂, V⟂, J⟂
		
		param1d U = P.fill_array1d<double>("U", "Uorb", orbitals, loc%Lcell);
		param1d Uph = P.fill_array1d<double>("Uph", "Uphorb", orbitals, loc%Lcell);
		param1d t0 = P.fill_array1d<double>("t0", "t0orb", orbitals, loc%Lcell);
		param1d mu = P.fill_array1d<double>("mu", "muorb", orbitals, loc%Lcell);
		param2d tperp = P.fill_array2d<double>("tRung", "t", "tPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Vperp = P.fill_array2d<double>("VRung", "V", "VPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Vzperp = P.fill_array2d<double>("VzRung", "Vz", "VzPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Vxyperp = P.fill_array2d<double>("VxyRung", "Vxy", "VxyPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Jperp = P.fill_array2d<double>("JRung", "J", "JPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		
		Terms.save_label(loc, U.label);
		Terms.save_label(loc, Uph.label);
		Terms.save_label(loc, t0.label);
		Terms.save_label(loc, mu.label);
		Terms.save_label(loc, tperp.label);
		Terms.save_label(loc, Vperp.label);
		Terms.save_label(loc, Vzperp.label);
		Terms.save_label(loc, Vxyperp.label);
		Terms.save_label(loc, Jperp.label);
		
		Terms.push_local(loc, 1., F[loc].HubbardHamiltonian(U.a, Uph.a, t0.a - mu.a, tperp.a, Vperp.a, Vzperp.a, Vxyperp.a, Jperp.a).plain<double>());
		
		// Nearest-neighbour terms: t, V, J
		
		if (!P.HAS("tFull") and !P.HAS("Vzfull") and !P.HAS("Vxyfull") and !P.HAS("Jfull") and !P.HAS("Xfull"))
		{
			param2d tpara = P.fill_array2d<double>("t", "tPara", {orbitals, next_orbitals}, loc%Lcell);
			param2d Vpara = P.fill_array2d<double>("V", "Vpara", {orbitals, next_orbitals}, loc%Lcell);
			param2d Vzpara = P.fill_array2d<double>("Vz", "Vzpara", {orbitals, next_orbitals}, loc%Lcell);
			param2d Vxypara = P.fill_array2d<double>("Vxy", "Vxypara", {orbitals, next_orbitals}, loc%Lcell);
			param2d Jpara = P.fill_array2d<double>("J", "Jpara", {orbitals, next_orbitals}, loc%Lcell);
			param2d Xpara = P.fill_array2d<double>("X", "Xpara", {orbitals, next_orbitals}, loc%Lcell);
			
			Terms.save_label(loc, tpara.label);
			Terms.save_label(loc, Vpara.label);
			Terms.save_label(loc, Vzpara.label);
			Terms.save_label(loc, Vxypara.label);
			Terms.save_label(loc, Jpara.label);
			Terms.save_label(loc, Xpara.label);
			
			if (loc < N_sites-1 or !P.get<bool>("OPEN_BC"))
			{
				for (std::size_t alfa=0; alfa<orbitals;      ++alfa)
				for (std::size_t beta=0; beta<next_orbitals; ++beta)
				{
					SiteOperator<Symmetry,double> c_sign_local    = OperatorType::prod(F[loc].c(alfa),    F[loc].sign(), {2,-1}).plain<double>();
					SiteOperator<Symmetry,double> cdag_sign_local = OperatorType::prod(F[loc].cdag(alfa), F[loc].sign(), {2,+1}).plain<double>();
					
					SiteOperator<Symmetry,double> c_tight    = F[lp1].c   (beta).plain<double>();
					SiteOperator<Symmetry,double> cdag_tight = F[lp1].cdag(beta).plain<double>();
					
					SiteOperator<Symmetry,double> n_local = F[loc].n(alfa).plain<double>();
					SiteOperator<Symmetry,double> n_tight = F[lp1].n(beta).plain<double>();
					
					SiteOperator<Symmetry,double> tz_local = F[loc].Tz(alfa).plain<double>();
					SiteOperator<Symmetry,double> tz_tight = F[lp1].Tz(beta).plain<double>();
					
					SiteOperator<Symmetry,double> tp_local = pow(-1,loc) * F[loc].cc      (alfa).plain<double>();
					SiteOperator<Symmetry,double> tm_tight = pow(-1,lp1) * F[lp1].cdagcdag(beta).plain<double>();
					
					SiteOperator<Symmetry,double> tm_local = pow(-1,loc) * F[loc].cdagcdag(alfa).plain<double>();
					SiteOperator<Symmetry,double> tp_tight = pow(-1,lp1) * F[lp1].cc      (beta).plain<double>();
					
					SiteOperator<Symmetry,double> Sdag_local = F[loc].Sdag(alfa).plain<double>();
					SiteOperator<Symmetry,double> S_tight    = F[lp1].S   (beta).plain<double>();
					
					SiteOperator<Symmetry,double> PsiLloc = OperatorType::prod(F[loc].ns(alfa),
					                                        OperatorType::prod(F[loc].c(alfa), F[loc].sign(), {2,-1}),
					                                        {2,-1}).plain<double>();
					SiteOperator<Symmetry,double> PsiRloc = OperatorType::prod(
					                                        OperatorType::prod(F[loc].c(alfa), F[loc].sign(), {2,-1}),
					                                        F[loc].ns(alfa),
					                                        {2,-1}).plain<double>();
					SiteOperator<Symmetry,double> PsiLlp1 = OperatorType::prod(F[lp1].ns(beta), F[lp1].c(beta),
					                                        {2,-1}).plain<double>();
					SiteOperator<Symmetry,double> PsiRlp1 = OperatorType::prod(F[lp1].c(beta), F[lp1].ns(beta),
					                                        {2,-1}).plain<double>();
					SiteOperator<Symmetry,double> PsidagLloc = OperatorType::prod(
					                                           OperatorType::prod(F[loc].cdag(alfa), F[loc].sign(), {2,1}),
					                                           F[loc].ns(alfa),
					                                           {2,1}).plain<double>();
					SiteOperator<Symmetry,double> PsidagRloc = OperatorType::prod(F[loc].ns(alfa),
					                                           OperatorType::prod(F[loc].cdag(alfa), F[loc].sign(), {2,1}),
					                                           {2,1}).plain<double>();
					SiteOperator<Symmetry,double> PsidagLlp1 = OperatorType::prod(F[lp1].cdag(beta), F[lp1].ns(beta),
					                                           {2,1}).plain<double>();
					SiteOperator<Symmetry,double> PsidagRlp1 = OperatorType::prod(F[lp1].ns(beta), F[lp1].cdag(beta),
					                                           {2,1}).plain<double>();
					
					//hopping
					Terms.push_tight(loc, -tpara(alfa,beta) * std::sqrt(2.), cdag_sign_local, c_tight);
					Terms.push_tight(loc, -tpara(alfa,beta) * std::sqrt(2.), c_sign_local, cdag_tight);
					
					//density-density interaction
					Terms.push_tight(loc, Vpara(alfa,beta), n_local, n_tight);
					
					//isospin-isopsin interaction
					Terms.push_tight(loc, 0.5*Vxypara(alfa,beta), tp_local, tm_tight);
					Terms.push_tight(loc, 0.5*Vxypara(alfa,beta), tm_local, tp_tight);
					Terms.push_tight(loc,     Vzpara (alfa,beta), tz_local, tz_tight);
					
					//spin-spin interaction
					Terms.push_tight(loc, Jpara(alfa, beta) * std::sqrt(3.), Sdag_local, S_tight);
					
					//correlated hopping
					Terms.push_tight(loc, -Xpara(alfa,beta) * std::sqrt(2.), PsidagLloc, PsiRlp1);
					Terms.push_tight(loc, -Xpara(alfa,beta) * std::sqrt(2.), PsidagRloc, PsiLlp1);
					Terms.push_tight(loc, -Xpara(alfa,beta) * std::sqrt(2.), PsiLloc, PsidagRlp1);
					Terms.push_tight(loc, -Xpara(alfa,beta) * std::sqrt(2.), PsiRloc, PsidagLlp1);
				}
			}
		}
		
		// Next-nearest-neighbour terms: t'
		if (!P.HAS("tFull"))
		{
			param2d tPrime = P.fill_array2d<double>("tPrime", "tPrime_array", {orbitals, nextn_orbitals}, loc%Lcell);
			Terms.save_label(loc, tPrime.label);
			
			if (loc < N_sites-2 or !P.get<bool>("OPEN_BC"))
			{
				for (std::size_t alfa=0; alfa<orbitals;       ++alfa)
				for (std::size_t beta=0; beta<nextn_orbitals; ++beta)
				{
					SiteOperator<Symmetry,double> c_sign_local    = OperatorType::prod(F[loc].c(alfa),    F[loc].sign(), {2,-1}).plain<double>();
					SiteOperator<Symmetry,double> cdag_sign_local = OperatorType::prod(F[loc].cdag(alfa), F[loc].sign(), {2,1} ).plain<double>();
					
					SiteOperator<Symmetry,double> sign_tight = F[lp1].sign().plain<double>();
					
					SiteOperator<Symmetry,double> c_nextn    = F[lp2].c(beta).plain<double>();
					SiteOperator<Symmetry,double> cdag_nextn = F[lp2].cdag(beta).plain<double>();
					
					Terms.push_nextn(loc, -tPrime(alfa,beta)*std::sqrt(2.), cdag_sign_local, sign_tight, c_nextn);
					Terms.push_nextn(loc, -tPrime(alfa,beta)*std::sqrt(2.), c_sign_local,    sign_tight, cdag_nextn);
				}
			}
		}
		
		// Next-next-nearest-neighbour terms: t''
		if (!P.HAS("tFull"))
		{
			param2d tPrimePrime = P.fill_array2d<double>("tPrimePrime", "tPrimePrime_array", {orbitals, nnextn_orbitals}, loc%Lcell);
			Terms.save_label(loc, tPrimePrime.label);
			
			if (loc < N_sites-3 or !P.get<bool>("OPEN_BC"))
			{
				vector<SiteOperator<Symmetry,double>> TransOps(2);
				TransOps[0] = F[lp1].sign().plain<double>();
				TransOps[1] = F[lp2].sign().plain<double>();
				
				for (std::size_t alfa=0; alfa<orbitals;        ++alfa)
				for (std::size_t beta=0; beta<nnextn_orbitals; ++beta)
				{
					SiteOperator<Symmetry,double> c_sign_local    = OperatorType::prod(F[loc].c(alfa),    F[loc].sign(), {2,-1}).plain<double>();
					SiteOperator<Symmetry,double> cdag_sign_local = OperatorType::prod(F[loc].cdag(alfa), F[loc].sign(), {2,1} ).plain<double>();
					
					SiteOperator<Symmetry,double> c_nnextn    = F[lp3].c(beta).plain<double>();
					SiteOperator<Symmetry,double> cdag_nnextn = F[lp3].cdag(beta).plain<double>();
					
					Terms.push(3, loc, -tPrimePrime(alfa,beta)*std::sqrt(2.), c_sign_local,    TransOps, c_nnextn);
					Terms.push(3, loc, -tPrimePrime(alfa,beta)*std::sqrt(2.), cdag_sign_local, TransOps, cdag_nnextn);
				}
			}
		}
	}
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
make_local (string name, size_t locx, size_t locy, const OperatorType &Op, double factor, bool FERMIONIC, bool HERMITIAN) const
{
	assert(locx<F.size() and locy<F[locx].dim());
	stringstream ss;
	ss << name << "(" << locx << "," << locy;
	if (factor != 1.) ss << ",factor=" << factor;
	ss << ")";
	
	Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > Mout(N_sites, Op.Q(), ss.str(), HERMITIAN);
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	
	if (FERMIONIC)
	{
		vector<SiteOperator<Symmetry,MatrixType::Scalar> > Signs(locx);
		for (size_t l=0; l<locx; ++l) Signs[l] = F[l].sign().plain<double>();
		
		Mout.setLocal(locx, (factor * Op).plain<double>(), Signs);
	}
	else
	{
		Mout.setLocal(locx, Op.plain<double>());
	}
	
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
n (size_t locx, size_t locy) const
{
	return make_local("n", locx,locy, F[locx].n(locy), 1., false, true);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
nh (size_t locx, size_t locy) const
{
	return make_local("nh", locx,locy, F[locx].nh(locy), 1., false, true);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
ns (size_t locx, size_t locy) const
{
	return make_local("ns", locx,locy, F[locx].ns(locy), 1., false, true);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
d (size_t locx, size_t locy) const
{
	return make_local("d", locx,locy, F[locx].d(locy), 1., false, true);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
c (size_t locx, size_t locy, double factor) const
{
	return make_local("c", locx,locy, F[locx].c(locy), factor, true, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
cdag (size_t locx, size_t locy, double factor) const
{
	return make_local("c†", locx,locy, F[locx].cdag(locy), factor, true, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
S (size_t locx, size_t locy) const
{
	return make_local("S", locx,locy, F[locx].S(locy), 1., false, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
Sdag (size_t locx, size_t locy, double factor) const
{
	return make_local("S†", locx,locy, F[locx].Sdag(locy), factor, false, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
cc (size_t locx, size_t locy) const
{
	stringstream ss;
	ss << "c" << UP << "c" << DN;
	return make_local(ss.str(), locx,locy, F[locx].cc(locy), 1., false, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
cdagcdag (size_t locx, size_t locy) const
{
	stringstream ss;
	ss << "c†" << DN << "c†" << UP;
	return make_local(ss.str(), locx,locy, F[locx].cdagcdag(locy), 1., false, false);
}

//Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
//make_corr (string name1, string name2, 
//           size_t locx1, size_t locx2, size_t locy1, size_t locy2, 
//           const OperatorType &Op1, const OperatorType &Op2,
//           qarray<Symmetry::Nq> Qtot, 
//           bool BOTH_HERMITIAN) const
//{
//	assert(locx1<F.size() and locx2<F.size() and locy1<F[locx1].dim() and locy2<F[locx2].dim());
//	stringstream ss;
//	ss << name1 << "(" << locx1 << "," << locy1 << ")"
//	   << name2 << "(" << locx2 << "," << locy2 << ")";
//	
//	bool HERMITIAN = (BOTH_HERMITIAN and locx1==locx2 and locy1==locy2)? true:false;
//	
//	Mpo<Symmetry> Mout(F.size(), Qtot, ss.str(), HERMITIAN);
//	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
//	
//	Mout.setLocal({locx1,locx2}, {Op1,Op2});
//	return Mout;
//}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
cdagc (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	assert(locx1<this->N_sites and locx2<this->N_sites);
	stringstream ss;
	ss << "c†(" << locx1 << "," << locy1 << ")" << "c(" << locx2 << "," << locy2 << ")";
	
	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
	for (size_t l=0; l<this->N_sites; l++) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	
	auto cdag = F[locx1].cdag(locy1);
	auto c    = F[locx2].c   (locy2);
	
	if (locx1 == locx2)
	{
		Mout.setLocal(locx1, sqrt(2.) * OperatorType::prod(cdag, c, Symmetry::qvacuum()).plain<double>());
	}
	else if (locx1<locx2)
	{
		Mout.setLocal({locx1, locx2}, {sqrt(2.) * OperatorType::prod(cdag, F[locx1].sign(), {2,+1}).plain<double>(), 
		                               c.plain<double>()},
		                               F[0].sign().plain<double>());
	}
	else if (locx1>locx2)
	{
		Mout.setLocal({locx2, locx1}, {sqrt(2.) * OperatorType::prod(c, F[locx2].sign(), {2,-1}).plain<double>(), 
		                               cdag.plain<double>()}, 
		                               F[0].sign().plain<double>());
	}
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
dh_excitation (size_t loc) const
{
	size_t lp1 = loc+1;
	
	OperatorType PsiRloc = OperatorType::prod(OperatorType::prod(F[loc].c(0), F[loc].sign(), {2,-1}), F[loc].ns(0), {2,-1});
	OperatorType PsidagLlp1 = OperatorType::prod(F[lp1].cdag(0), F[lp1].ns(0),{2,1});
	
	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), "dh");
	for(size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis((F[l].get_basis()).qloc(),l); }
	
	Mout.setLocal({loc, lp1}, {PsiRloc.plain<double>(), PsidagLlp1.plain<double>()});
	
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
nn (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	assert(locx1<this->N_sites and locx2<this->N_sites);
	stringstream ss;
	ss << "n(" << locx1 << "," << locy1 << ")" << "n(" << locx2 << "," << locy2 << ")";
	
	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
	for (size_t l=0; l<this->N_sites; l++) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	
	Mout.setLocal({locx1, locx2}, {F[locx1].n(locy1).plain<double>(), F[locx2].n(locy2).plain<double>()});
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
TzTz (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	assert(locx1<this->N_sites and locx2<this->N_sites);
	stringstream ss;
	ss << "Tz(" << locx1 << "," << locy1 << ")" << "Tz(" << locx2 << "," << locy2 << ")";
	
	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
	for (size_t l=0; l<this->N_sites; l++) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	
	auto Op1 = F[locx1].Tz(locy1);
	auto Op2 = F[locx2].Tz(locy2);
	
	if (locx1 == locx2)
	{
		auto product = OperatorType::prod(Op1, Op2, Symmetry::qvacuum());
		Mout.setLocal(locx1, product.plain<double>());
	}
	else
	{
		Mout.setLocal({locx1, locx2}, {F[locx1].Tz(locy1).plain<double>(), F[locx2].Tz(locy2).plain<double>()});
	}
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
TpTm (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	assert(locx1<this->N_sites and locx2<this->N_sites);
	stringstream ss;
	ss << "Tp(" << locx1 << "," << locy1 << ")" << "Tm(" << locx2 << "," << locy2 << ")";
	
	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
	for (size_t l=0; l<this->N_sites; l++) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	
	auto Op1 = pow(-1.,locx1+locy1) * F[locx1].cc(locy1);
	auto Op2 = pow(-1.,locx2+locy2) * F[locx2].cdagcdag(locy2);
	
	if (locx1 == locx2)
	{
		auto product = OperatorType::prod(Op1, Op2, Symmetry::qvacuum());
		Mout.setLocal(locx1, product.plain<double>());
	}
	else
	{
		Mout.setLocal({locx1, locx2}, {Op1.plain<double>(), Op2.plain<double>()});
	}
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
TmTp (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	assert(locx1<this->N_sites and locx2<this->N_sites);
	stringstream ss;
	ss << "Tm(" << locx1 << "," << locy1 << ")" << "Tp(" << locx2 << "," << locy2 << ")";
	
	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
	for (size_t l=0; l<this->N_sites; l++) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	
	auto Op1 = pow(-1.,locx1+locy1) * F[locx1].cdagcdag(locy1);
	auto Op2 = pow(-1.,locx2+locy2) * F[locx2].cc(locy2);
	
	if (locx1 == locx2)
	{
		auto product = OperatorType::prod(Op1, Op2, Symmetry::qvacuum());
		Mout.setLocal(locx1, product.plain<double>());
	}
	else
	{
		Mout.setLocal({locx1, locx2}, {Op1.plain<double>(), Op2.plain<double>()});
	}
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
SdagS (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	assert(locx1<this->N_sites and locx2<this->N_sites);
	stringstream ss;
	ss << "S†(" << locx1 << "," << locy1 << ")" << "S(" << locx2 << "," << locy2 << ")";
	
	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
	for (size_t l=0; l<this->N_sites; l++) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	
	auto Op1 = F[locx1].Sdag(locy1);
	auto Op2 = F[locx2].S(locy2);
	
	if (locx1 == locx2)
	{
		auto product = std::sqrt(3.) * OperatorType::prod(Op1, Op2, Symmetry::qvacuum());
		Mout.setLocal(locx1, product.plain<double>());
	}
	else
	{
		Mout.setLocal({locx1, locx2}, {(std::sqrt(3.) * Op1).plain<double>(), Op2.plain<double>()});
	}
	
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
Tz (size_t locx, size_t locy) const
{
	return make_local("Tz", locx,locy, F[locx].Tz(locy), 1., false, true);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
Tp (size_t locx, size_t locy, double factor) const
{
	return make_local("T+", locx,locy, factor*pow(-1.,locx+locy)*F[locx].cc(locy), 1., false, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
Tm (size_t locx, size_t locy, double factor) const
{
	return make_local("T-", locx,locy, factor*pow(-1.,locx+locy)*F[locx].cdagcdag(locy), 1., false, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,complex<double> > HubbardSU2xU1::
make_FourierYSum (string name, const vector<OperatorType> &Ops, 
                  double factor, bool HERMITIAN, const vector<complex<double> > &phases) const
{
	stringstream ss;
	ss << name << "_ky(";
	for (int l=0; l<phases.size(); ++l)
	{
		ss << phases[l];
		if (l!=phases.size()-1) {ss << ",";}
		else                    {ss << ")";}
	}
	
	// all Ops[l].Q() must match
	Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,complex<double> > Mout(N_sites, Ops[0].Q(), ss.str(), HERMITIAN);
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	
	vector<complex<double> > phases_x_factor = phases;
	for (int l=0; l<phases.size(); ++l)
	{
		phases_x_factor[l] = phases[l] * factor;
	}
	
	vector<SiteOperator<Symmetry,complex<double> > > OpsPlain(Ops.size());
	for (int l=0; l<OpsPlain.size(); ++l)
	{
		OpsPlain[l] = Ops[l].plain<double>().cast<complex<double> >();
	}
	
	Mout.setLocalSum(OpsPlain, phases_x_factor);
	
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,complex<double> > HubbardSU2xU1::
S_ky (vector<complex<double> > phases) const
{
	vector<OperatorType> Ops(N_sites);
	for (size_t l=0; l<N_sites; ++l)
	{
		Ops[l] = F[l].S(0);
	}
	return make_FourierYSum("S", Ops, 1., false, phases);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,complex<double> > HubbardSU2xU1::
Sdag_ky (vector<complex<double> > phases, double factor) const
{
	vector<OperatorType> Ops(N_sites);
	for (size_t l=0; l<N_sites; ++l)
	{
		Ops[l] = F[l].Sdag(0);
	}
	return make_FourierYSum("S†", Ops, 1., false, phases);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,complex<double> > HubbardSU2xU1::
T_ky (vector<complex<double> > phases) const
{
	vector<OperatorType> Ops(N_sites);
	for (size_t l=0; l<N_sites; ++l)
	{
		Ops[l] = F[l].S(0);
	}
	return make_FourierYSum("T", Ops, 1., false, phases);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,complex<double> > HubbardSU2xU1::
Tdag_ky (vector<complex<double> > phases, double factor) const
{
	vector<OperatorType> Ops(N_sites);
	for (size_t l=0; l<N_sites; ++l)
	{
		Ops[l] = F[l].S(0);
	}
	return make_FourierYSum("T†", Ops, 1., false, phases);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,complex<double> > HubbardSU2xU1::
c_ky (vector<complex<double> > phases, double factor) const
{
	vector<OperatorType> Ops(N_sites);
	for (size_t l=0; l<N_sites; ++l)
	{
		Ops[l] = F[l].S(0);
	}
	return make_FourierYSum("c", Ops, 1., false, phases);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,complex<double> > HubbardSU2xU1::
cdag_ky (vector<complex<double> > phases, double factor) const
{
	vector<OperatorType> Ops(N_sites);
	for (size_t l=0; l<N_sites; ++l)
	{
		Ops[l] = F[l].Sdag(0);
	}
	return make_FourierYSum("c†", Ops, 1., false, phases);
}

} // end namespace VMPS::models

#endif
