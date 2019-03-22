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
	Mpo<Symmetry> c (size_t locx, size_t locy=0, double factor=1.);
	Mpo<Symmetry> cdag (size_t locx, size_t locy=0, double factor=sqrt(2.));
	Mpo<Symmetry> n (size_t locx, size_t locy=0);
	Mpo<Symmetry> d (size_t locx, size_t locy=0);
	///@}
	
	///@{
	Mpo<Symmetry> cc (size_t locx, size_t locy=0);
	Mpo<Symmetry> cdagcdag (size_t locx, size_t locy=0);
	///@}
	
	///@{
	Mpo<Symmetry> S (size_t locx, size_t locy=0);
	Mpo<Symmetry> Sdag (size_t locx, size_t locy=0, double factor=sqrt(3.));
	///@}
	
	///@{
	Mpo<Symmetry> cdagc (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	Mpo<Symmetry> nn    (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	Mpo<Symmetry> SdagS (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	///@}
	
	static const map<string,any> defaults;
	static const map<string,any> sweep_defaults;
	
protected:
	
	Mpo<Symmetry>
	make_local (string name, 
	            size_t locx, size_t locy, 
	            const OperatorType &Op, 
	            double factor, bool FERMIONIC, bool HERMITIAN) const;
	
	vector<FermionBase<Symmetry> > F;
};

// V is standard next-nearest neighbour density interaction
// Vz and Vxy are anisotropic isospin-isospin next-nearest neighbour interaction
const map<string,any> HubbardSU2xU1::defaults = 
{
	{"t",1.}, {"tPrime",0.}, {"tRung",1.},
	{"mu",0.}, {"t0",0.}, 
	{"U",0.},
	{"V",0.}, {"Vrung",0.},
	{"Vz",0.}, {"Vzrung",0.}, {"Vxy",0.}, {"Vxyrung",0.}, 
	{"J",0.}, {"Jperp",0.},
	{"CALC_SQUARE",false}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}
};

const map<string,any> HubbardSU2xU1::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1.}, {"lim_alpha",12ul}, {"eps_svd",1e-7},
	{"Dincr_abs", 5ul}, {"Dincr_per", 2ul}, {"Dincr_rel", 1.1},
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
		
		F[l] = FermionBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), !isfinite(P.get<double>("U",l%Lcell)));
		setLocBasis(F[l].get_basis().qloc(),l);
	}
		
	set_operators(F, P, Terms);
	
	this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
	this->precalc_TwoSiteData();
}
    
void HubbardSU2xU1::
set_operators(const std::vector<FermionBase<Symmetry> > &F, const ParamHandler &P, HamiltonianTermsXd<Symmetry> &Terms)
{
    std::size_t Lcell = P.size();
    std::size_t N_sites = Terms.size();
    Terms.set_name("Hubbard");
    
    for(std::size_t loc=0; loc<N_sites; ++loc)
    {
        std::size_t orbitals = F[loc].orbitals();
        std::size_t next_orbitals = F[(loc+1)%N_sites].orbitals();
        std::size_t nextn_orbitals = F[(loc+2)%N_sites].orbitals();
        
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
					SiteOperator<Symmetry, double> c_sign_local = OperatorType::prod(F[loc].c(0), F[loc].sign(), {2,-1}).plain<double>();
                    SiteOperator<Symmetry, double> cdag_sign_local = OperatorType::prod(F[loc].cdag(0), F[loc].sign(), {2,1}).plain<double>();
					SiteOperator<Symmetry, double> c_range = F[(loc+range)%N_sites].c(0).plain<double>();
                    SiteOperator<Symmetry, double> cdag_range = F[(loc+range)%N_sites].cdag(0).plain<double>();
					
					//hopping
					//cout << "loc=" << loc << ", pushing at range=" << range << ", value=" << value << endl;
                    Terms.push(range, loc, -value * std::sqrt(2.), cdag_sign_local, TransOps, c_range);
					Terms.push(range, loc, -value * std::sqrt(2.), c_sign_local, TransOps, cdag_range);
				}
			}
			
			stringstream ss;
			ss << "tᵢⱼ(avg=" << Geometry2D::avg(Full) << ",σ=" << Geometry2D::sigma(Full) << ",max=" << Geometry2D::max(Full) << ")";
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
					auto Tz_hop    = F[(loc+range)%N_sites].Tz(0);
					
					Terms.push(range, loc, value,
					           Tz_loc.plain<double>(), TransOps, Tz_hop.plain<double>());
				}
			}
			
			stringstream ss;
			ss << "Vzᵢⱼ(avg=" << Geometry2D::avg(Full) << ",σ=" << Geometry2D::sigma(Full) << ",max=" << Geometry2D::max(Full) << ")";
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
					auto Tp_loc    = pow(-1,loc)*F[loc].Tp(0);
					auto Tm_hop    = pow(-1,(loc+range)%N_sites)*F[(loc+range)%N_sites].Tm(0);
					auto Tm_loc    = pow(-1,loc)*F[loc].Tm(0);
					auto Tp_hop    = pow(-1,(loc+range)%N_sites)*F[(loc+range)%N_sites].Tp(0);
					
					Terms.push(range, loc, 0.5 * value,
					           Tp_loc.plain<double>(), TransOps, Tm_hop.plain<double>());
					Terms.push(range, loc, 0.5 * value,
					           Tm_loc.plain<double>(), TransOps, Tp_hop.plain<double>());
				}
			}
			
			stringstream ss;
			ss << "Vxyᵢⱼ(avg=" << Geometry2D::avg(Full) << ",σ=" << Geometry2D::sigma(Full) << ",max=" << Geometry2D::max(Full) << ")";
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
			ss << "Jᵢⱼ(avg=" << Geometry2D::avg(Full) << ",σ=" << Geometry2D::sigma(Full) << ",max=" << Geometry2D::max(Full) << ")";
			Terms.save_label(loc,ss.str());
		}

        // Local terms: U, t0, μ, t⟂, V⟂, J⟂
        
        param1d U = P.fill_array1d<double>("U", "Uorb", orbitals, loc%Lcell);
        param1d t0 = P.fill_array1d<double>("t0", "t0orb", orbitals, loc%Lcell);
        param1d mu = P.fill_array1d<double>("mu", "muorb", orbitals, loc%Lcell);
        param2d tperp = P.fill_array2d<double>("tRung", "t", "tPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Vperp = P.fill_array2d<double>("VRung", "V", "VPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
        param2d Vzperp = P.fill_array2d<double>("VzRung", "Vz", "VzPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Vxyperp = P.fill_array2d<double>("VxyRung", "Vxy", "VxyPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
        param2d Jperp = P.fill_array2d<double>("JRung", "J", "JPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
        
        Terms.save_label(loc, U.label);
        Terms.save_label(loc, t0.label);
        Terms.save_label(loc, mu.label);
        Terms.save_label(loc, tperp.label);
		Terms.save_label(loc, Vperp.label);
        Terms.save_label(loc, Vzperp.label);
		Terms.save_label(loc, Vxyperp.label);
        Terms.save_label(loc, Jperp.label);
        
        Terms.push_local(loc, 1., F[loc].HubbardHamiltonian(U.a, t0.a - mu.a, tperp.a, Vperp.a, Vzperp.a, Vxyperp.a, Jperp.a).plain<double>());
        
        
        // Nearest-neighbour terms: t, V, J

		if (!P.HAS("tFull") and !P.HAS("Vzfull") and !P.HAS("Vxyfull") and !P.HAS("Jfull"))
		{
			param2d tpara = P.fill_array2d<double>("t", "tPara", {orbitals, next_orbitals}, loc%Lcell);
			param2d Vpara = P.fill_array2d<double>("V", "Vpara", {orbitals, next_orbitals}, loc%Lcell);
			param2d Vzpara = P.fill_array2d<double>("Vz", "Vzpara", {orbitals, next_orbitals}, loc%Lcell);
			param2d Vxypara = P.fill_array2d<double>("Vxy", "Vxypara", {orbitals, next_orbitals}, loc%Lcell);
			param2d Jpara = P.fill_array2d<double>("J", "Jpara", {orbitals, next_orbitals}, loc%Lcell);
        
			Terms.save_label(loc, tpara.label);
			Terms.save_label(loc, Vpara.label);
			Terms.save_label(loc, Vzpara.label);
			Terms.save_label(loc, Vxypara.label);
			Terms.save_label(loc, Jpara.label);
        
			if (loc < N_sites-1 || !P.get<bool>("OPEN_BC"))
			{
				for (std::size_t alpha=0; alpha<orbitals; ++alpha)
				{
					for (std::size_t beta=0; beta<next_orbitals; ++beta)
					{
						SiteOperator<Symmetry, double> c_sign_local = OperatorType::prod(F[loc].c(alpha), F[loc].sign(), {2,-1}).plain<double>();
						SiteOperator<Symmetry, double> cdag_sign_local = OperatorType::prod(F[loc].cdag(alpha), F[loc].sign(), {2,1}).plain<double>();
                    
						SiteOperator<Symmetry, double> c_tight = F[(loc+1)%N_sites].c(beta).plain<double>();
						SiteOperator<Symmetry, double> cdag_tight = F[(loc+1)%N_sites].cdag(beta).plain<double>();
                    
						SiteOperator<Symmetry, double> n_local = F[loc].n(alpha).plain<double>();
						SiteOperator<Symmetry, double> n_tight = F[(loc+1)%N_sites].n(beta).plain<double>();

						SiteOperator<Symmetry, double> tz_local = F[loc].Tz(alpha).plain<double>();
						SiteOperator<Symmetry, double> tz_tight = F[(loc+1)%N_sites].Tz(beta).plain<double>();

						SiteOperator<Symmetry, double> tp_local = pow(-1,loc)*F[loc].Tp(alpha).plain<double>();
						SiteOperator<Symmetry, double> tm_tight = pow(-1,loc+1)*F[(loc+1)%N_sites].Tm(beta).plain<double>();

						SiteOperator<Symmetry, double> tm_local = pow(-1,loc)*F[loc].Tm(alpha).plain<double>();
						SiteOperator<Symmetry, double> tp_tight = pow(-1,loc+1)*F[(loc+1)%N_sites].Tp(beta).plain<double>();
                    
						SiteOperator<Symmetry, double> Sdag_local = F[loc].Sdag(alpha).plain<double>();
						SiteOperator<Symmetry, double> S_tight = F[(loc+1)%N_sites].S(beta).plain<double>();

						//nn hopping
						Terms.push_tight(loc, -tpara(alpha, beta) * std::sqrt(2.), cdag_sign_local, c_tight);
						Terms.push_tight(loc, -tpara(alpha, beta) * std::sqrt(2.), c_sign_local, cdag_tight);
						//nn density interaction
						Terms.push_tight(loc, Vpara(alpha, beta), n_local, n_tight);
						//nn isospin-isopsin interaction
						Terms.push_tight(loc, Vzpara(alpha, beta), tz_local, tz_tight);
						Terms.push_tight(loc, 0.5*Vxypara(alpha, beta), tp_local, tm_tight);
						Terms.push_tight(loc, 0.5*Vxypara(alpha, beta), tm_local, tp_tight);
						//nn spin-spin interaction
						Terms.push_tight(loc, Jpara(alpha, beta) * std::sqrt(3.), Sdag_local, S_tight);
					}
				}
			}
        }
        
        // Next-nearest-neighbour terms: t'
        if (!P.HAS("tFull"))
		{
			param2d tprime = P.fill_array2d<double>("tPrime", "tPrime_array", {orbitals, nextn_orbitals}, loc%Lcell);
			Terms.save_label(loc, tprime.label);
        
			if (loc < N_sites-2 || !P.get<bool>("OPEN_BC"))
			{
				for (std::size_t alpha=0; alpha<orbitals; ++alpha)
				{
					for (std::size_t beta=0; beta<nextn_orbitals; ++beta)
					{
						SiteOperator<Symmetry, double> c_sign_local = OperatorType::prod(F[loc].c(alpha), F[loc].sign(), {2,-1}).plain<double>();
						SiteOperator<Symmetry, double> cdag_sign_local = OperatorType::prod(F[loc].cdag(alpha), F[loc].sign(), {2,1}).plain<double>();
                    
						SiteOperator<Symmetry, double> sign_tight = F[(loc+1)%N_sites].sign().plain<double>();
                    
						SiteOperator<Symmetry, double> c_nextn = F[(loc+2)%N_sites].c(beta).plain<double>();
						SiteOperator<Symmetry, double> cdag_nextn = F[(loc+2)%N_sites].cdag(beta).plain<double>();
                    
						Terms.push_nextn(loc, tprime(alpha, beta) * std::sqrt(2.), cdag_sign_local, sign_tight, c_nextn);
						Terms.push_nextn(loc, tprime(alpha, beta) * std::sqrt(2.), c_sign_local,    sign_tight, cdag_nextn);
					}
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
	ss << name << "(" << locx << "," << locy << ")";
	
	Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > Mout(N_sites, Op.Q(), ss.str(), HERMITIAN);
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	
	(FERMIONIC)? 
		Mout.setLocal(locx, (factor * Op).plain<double>(), F[0].sign().plain<double>()) //* pow(-1.,locx+1)
	:Mout.setLocal(locx, Op.plain<double>());
	
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
n (size_t locx, size_t locy)
{
	return make_local("n", locx,locy, F[locx].n(locy), 1., false, true);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
d (size_t locx, size_t locy)
{
	return make_local("d", locx,locy, F[locx].d(locy), 1., false, true);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
c (size_t locx, size_t locy, double factor)
{
	return make_local("c", locx,locy, F[locx].c(locy), factor, true, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
cdag (size_t locx, size_t locy, double factor)
{
	return make_local("c†", locx,locy, F[locx].cdag(locy), factor, true, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
S (size_t locx, size_t locy)
{
	return make_local("S", locx,locy, F[locx].S(locy), 1., false, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
Sdag (size_t locx, size_t locy, double factor)
{
	return make_local("S†", locx,locy, F[locx].Sdag(locy), factor, false, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
cc (size_t locx, size_t locy)
{
	stringstream ss;
	ss << "c" << UP << "c" << DN;
	return make_local(ss.str(), locx,locy, F[locx].Eta(locy), 1., false, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
cdagcdag (size_t locx, size_t locy)
{
	stringstream ss;
	ss << "c†" << DN << "c†" << UP;
	return make_local(ss.str(), locx,locy, F[locx].Etadag(locy), 1., false, false);
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
cdagc (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
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
nn (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
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
SdagS (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
{
	assert(locx1<this->N_sites and locx2<this->N_sites);
	stringstream ss;
	ss << "S†(" << locx1 << "," << locy1 << ")" << "S(" << locx2 << "," << locy2 << ")";
	
	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
	for (size_t l=0; l<this->N_sites; l++) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	
	auto Op1 = F[locx1].Sdag(locy1);
	auto Op2 = F[locx2].Sdag(locy2);
	
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

//Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
//SSdag (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
//{
//	assert(locx1<this->N_sites and locx2<this->N_sites);
//	stringstream ss;
//	ss << "S†(" << locx1 << "," << locy1 << ")" << "S(" << locx2 << "," << locy2 << ")";

//	Mpo<Symmetry> Mout(N_sites, N_legs);
//	for(size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F[l].get_basis(),l); }

//	auto Sdag = F.Sdag(locy1);
//	auto S = F.S(locy2);
//	Mout.label = ss.str();
//	Mout.setQtarget(Symmetry::qvacuum());
//	Mout.qlabel = HubbardSU2xU1::Slabel;
//	if(locx1 == locx2)
//	{
//		auto product = sqrt(3.)*Operator::prod(Sdag,S,Symmetry::qvacuum());
//		Mout.setLocal(locx1,product,Symmetry::qvacuum());
//		return Mout;
//	}
//	else
//	{
//		Mout.setLocal({locx1, locx2}, {sqrt(3.)*Sdag, S}, {{3,0},{3,0}});
//		return Mout;
//	}
//}

//Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > > HubbardSU2xU1::
//EtaEtadag (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
//{
//	assert(locx1<this->N_sites and locx2<this->N_sites);
//	stringstream ss;
//	ss << "η†(" << locx1 << "," << locy1 << ")" << "η(" << locx2 << "," << locy2 << ")";

//	Mpo<Symmetry> Mout(N_sites, N_legs);
//	for(size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F[l].get_basis(),l); }

//	auto Etadag = F.Etadag(locy1);
//	auto Eta = F.Eta(locy2);
//	Mout.label = ss.str();
//	Mout.setQtarget(Symmetry::qvacuum());
//	Mout.qlabel = HubbardSU2xU1::Slabel;
//	if(locx1 == locx2)
//	{
//		auto product = Operator::prod(Etadag,Eta,Symmetry::qvacuum());
//		Mout.setLocal(locx1,product,Symmetry::qvacuum());
//		return Mout;
//	}
//	else
//	{
//		Mout.setLocal({locx1, locx2}, {Etadag, Eta}, {{1,2},{1,-2}});
//		return Mout;
//	}
//}

// Mpo<SymSU2<double> > HubbardSU2xU1::
// SSdag (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
// {
// 	assert(locx1 < N_sites and locx2 < N_sites and locy1 < N_legs and locy2 < N_legs);
// 	stringstream ss;
// 	ss << "S†S(" << locx1 << "," << locy1 << ")" << "Sz(" << locx2 << "," << locy2 << ")";
// 	vector<vector<qType> > qOptmp(N_sites);
// 	for (size_t l=0; l<N_sites; l++)
// 	{
// 		qOptmp[l].resize(1);
// 		qOptmp[l][0] = (l == locx1 or l == locx2) ? 3 : 1;
// 	}

// 	Mpo<Symmetry> Mout(N_sites, Mpo<Symmetry>::qloc, qOptmp, {1}, HubbardSU2xU1::Slabel, ss.str());
// 	Mout.setLocal({locx1,locx2}, {F.S(locy1),F.Sdag(locy2)});
// 	return Mout;
// }

// Mpo<SymSU2<double> > HubbardSU2xU1::
// triplon (SPIN_INDEX sigma, size_t locx, size_t locy)
// {
// 	assert(locx<N_sites and locy<F[locx].dim());
// 	stringstream ss;
// 	ss << "triplon(" << locx << ")" << "c(" << locx+1 << ",σ=" << sigma << ")";
// 	qstd::array<2> qdiff;
// 	(sigma==UP) ? qdiff = {-2,-1} : qdiff = {-1,-2};
	
// 	vector<SuperMatrix<double> > M(N_sites);
// 	for (size_t l=0; l<locx; ++l)
// 	{
// 		M[l].setMatrix(1,F.dim());
// 		M[l](0,0) = F.sign();
// 	}
// 	// c(locx,UP)*c(locx,DN)
// 	M[locx].setMatrix(1,F.dim());
// 	M[locx](0,0) = F.c(UP,locy)*F.c(DN,locy);
// 	// c(locx+1,UP|DN)
// 	M[locx+1].setMatrix(1,F.dim());
// 	M[locx+1](0,0) = (sigma==UP)? F.c(UP,locy) : F.c(DN,locy);
// 	for (size_t l=locx+2; l<N_sites; ++l)
// 	{
// 		M[l].setMatrix(1,F.dim());
// 		M[l](0,0).setIdentity();
// 	}
	
// 	return Mpo<Symmetry>(N_sites, M, Mpo<Symmetry>::qloc, qdiff, HubbardSU2xU1::Nlabel, ss.str());
// }

// Mpo<SymSU2<double> > HubbardSU2xU1::
// antitriplon (SPIN_INDEX sigma, size_t locx, size_t locy)
// {
// 	assert(locx<N_sites and locy<F[locx].dim());
// 	stringstream ss;
// 	ss << "antitriplon(" << locx << ")" << "c(" << locx+1 << ",σ=" << sigma << ")";
// 	qstd::array<2> qdiff;
// 	(sigma==UP) ? qdiff = {+2,+1} : qdiff = {+1,+2};
	
// 	vector<SuperMatrix<double> > M(N_sites);
// 	for (size_t l=0; l<locx; ++l)
// 	{
// 		M[l].setMatrix(1,F.dim());
// 		M[l](0,0) = F.sign();
// 	}
// 	// c†(locx,DN)*c†(locx,UP)
// 	M[locx].setMatrix(1,F.dim());
// 	M[locx](0,0) = F.cdag(DN,locy)*F.cdag(UP,locy);
// 	// c†(locx+1,UP|DN)
// 	M[locx+1].setMatrix(1,F.dim());
// 	M[locx+1](0,0) = (sigma==UP)? F.cdag(UP,locy) : F.cdag(DN,locy);
// 	for (size_t l=locx+2; l<N_sites; ++l)
// 	{
// 		M[l].setMatrix(1,F.dim());
// 		M[l](0,0).setIdentity();
// 	}
	
// 	return Mpo<Symmetry>(N_sites, M, Mpo<Symmetry>::qloc, qdiff, HubbardSU2xU1::Nlabel, ss.str());
// }

// Mpo<SymSU2<double> > HubbardSU2xU1::
// quadruplon (size_t locx, size_t locy)
// {
// 	assert(locx<N_sites and locy<F[locx].dim());
// 	stringstream ss;
// 	ss << "Auger(" << locx << ")" << "Auger(" << locx+1 << ")";
	
// 	vector<SuperMatrix<double> > M(N_sites);
// 	for (size_t l=0; l<locx; ++l)
// 	{
// 		M[l].setMatrix(1,F.dim());
// 		M[l](0,0).setIdentity();
// 	}
// 	// c(loc,UP)*c(loc,DN)
// 	M[locx].setMatrix(1,F.dim());
// 	M[locx](0,0) = F.c(UP,locy)*F.c(DN,locy);
// 	// c(loc+1,UP)*c(loc+1,DN)
// 	M[locx+1].setMatrix(1,F.dim());
// 	M[locx+1](0,0) = F.c(UP,locy)*F.c(DN,locy);
// 	for (size_t l=locx+2; l<N_sites; ++l)
// 	{
// 		M[l].setMatrix(1,4);
// 		M[l](0,0).setIdentity();
// 	}
	
// 	return Mpo<Symmetry>(N_sites, M, Mpo<Symmetry>::qloc, {-2,-2}, HubbardSU2xU1::Nlabel, ss.str());
// }

} // end namespace VMPS::models

#endif
