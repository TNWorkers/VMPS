#ifndef HUBBARDMODELSU2XSU2_H_
#define HUBBARDMODELSU2XSU2_H_

#include "symmetry/S1xS2.h"
#include "symmetry/SU2.h"
#include "bases/FermionBaseSU2xSU2.h"
#include "Mpo.h"
#include "ParamReturner.h"
#include "Geometry2D.h" // from TOOLS

namespace VMPS
{

/** 
 * \class HubbardSU2xSU2
 * \ingroup Hubbard
 *
 * \brief Hubbard Model
 *
 * MPO representation of 
 * 
 * \f[
 * H = -t \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
 *     +U \sum_i \left[\left(n_{i\uparrow}-\frac{1}{2}\right)\left(n_{i\downarrow}-\frac{1}{2}\right) -\frac{1}{4}\right]
 *     +V \sum_{<ij>} \mathbf{T}_i \mathbf{T}_j
 *     +J \sum_{<ij>} \mathbf{S}_i \mathbf{S}_j
 * \f]
 * with \f$T^+_i = (-1)^i c^{\dagger}_{i\uparrow} c^{\dagger}_{i\downarrow}\f$, \f$Q^-_i = (T^+_i)^{\dagger}\f$, \f$T^z_i = 0.5(n_{i}-1)\f$
 *
 * \note Makes use of the spin-SU(2) symmetry and the charge-SU(2) symmetry.
 * \warning Bipartite hopping structure is mandatory (particle-hole symmetry)!
 * \warning \f$J>0\f$ is antiferromagnetic
 */
class HubbardSU2xSU2 : public Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > ,double>, public ParamReturner
{
public:
	typedef Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > Symmetry;
	MAKE_TYPEDEFS(HubbardSU2xSU2)
	
private:
	
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	
public:
	
	HubbardSU2xSU2() : Mpo(){};
	HubbardSU2xSU2 (const size_t &L, const vector<Param> &params);
	
	static void set_operators(const std::vector<FermionBase<Symmetry>> &F, const ParamHandler &P, HamiltonianTermsXd<Symmetry> &Terms);
	
	Mpo<Symmetry> c (size_t locx, size_t locy=0, double factor=sqrt(2.));
	Mpo<Symmetry> cdag (size_t locx, size_t locy=0, double factor=sqrt(2.));
	
	Mpo<Symmetry> cdagc (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	Mpo<Symmetry> nh (size_t locx, size_t locy=0);
	Mpo<Symmetry> ns (size_t locx, size_t locy=0);
	
	Mpo<Symmetry> S (size_t locx, size_t locy=0);
	Mpo<Symmetry> Sdag (size_t locx, size_t locy=0, double factor=sqrt(3.));
	Mpo<Symmetry> SdagS (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	
	Mpo<Symmetry> T (size_t locx, size_t locy=0);
	Mpo<Symmetry> Tdag (size_t locx, size_t locy=0, double factor=1.);
	Mpo<Symmetry> TdagT (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	
	static const map<string,any> defaults;
	static const map<string,any> sweep_defaults;
	
protected:
	
	vector<FermionBase<Symmetry> > F;
	
	Mpo<Symmetry> make_local (string name, size_t locx, size_t locy, const OperatorType &Op, double factor, bool FERMIONIC, bool HERMITIAN) const;
	Mpo<Symmetry> make_corr  (string name1, string name2, size_t locx1, size_t locx2, size_t locy1, size_t locy2,
	                          const OperatorType &Op1, const OperatorType &Op2, qarray<Symmetry::Nq> Qtot,
	                          double factor, bool FERMIONIC, bool HERMITIAN) const;
};

const map<string,any> HubbardSU2xSU2::defaults = 
{
	{"t",1.}, {"tRung",1.}, {"tPrimePrime",0.}, 
	{"U",0.},
	{"V",0.}, {"Vrung",0.},
	{"J",0.}, {"Jrung",0.},
	{"CALC_SQUARE",false}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}
};

const map<string,any> HubbardSU2xSU2::sweep_defaults = 
{
	{"max_alpha",100.}, {"min_alpha",1e-11}, {"lim_alpha",10ul}, {"eps_svd",1e-7},
	{"Dincr_abs", 2ul}, {"Dincr_per", 2ul}, {"Dincr_rel", 1.1},
	{"min_Nsv",0ul}, {"max_Nrich",-1},
	{"max_halfsweeps",20ul}, {"min_halfsweeps",6ul},
	{"Dinit",4ul}, {"Qinit",10ul}, {"Dlimit",500ul},
	{"tol_eigval",1e-6}, {"tol_state",1e-5},
	{"savePeriod",0ul}, {"CALC_S_ON_EXIT", true}, {"CONVTEST", DMRG::CONVTEST::VAR_2SITE}
};

HubbardSU2xSU2::
HubbardSU2xSU2 (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, qarray<Symmetry::Nq>({1,1}), "", PROP::HERMITIAN, PROP::NON_UNITARY, PROP::HAMILTONIAN),
 ParamReturner(HubbardSU2xSU2::sweep_defaults)
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
//	assert(Lcell > 1 and "You need to set a unit cell with at least Lcell=2 for the charge-SU(2) symmetry!");
	HamiltonianTermsXd<Symmetry> Terms(N_sites, P.get<bool>("OPEN_BC"));
	F.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		
		F[l] = (l%2 == 0) ? FermionBase<Symmetry>(P.get<size_t>("Ly",l%Lcell),SUB_LATTICE::A):
		                    FermionBase<Symmetry>(P.get<size_t>("Ly",l%Lcell),SUB_LATTICE::B);
		setLocBasis(F[l].get_basis().qloc(),l);
	}
	
	set_operators(F, P, Terms);
	
	this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
	this->precalc_TwoSiteData();
}

void HubbardSU2xSU2::
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
		std::size_t next3_orbitals = F[lp3].orbitals();
		
		stringstream ss;
		ss << "Ly=" << P.get<size_t>("Ly",loc%Lcell);
		Terms.save_label(loc, ss.str());
		
		if (P.HAS("tFull"))
		{
			ArrayXXd Full = P.get<Eigen::ArrayXXd>("tFull");
			vector<vector<std::pair<size_t,double> > > R = Geometry2D::rangeFormat(Full);
			
			if (P.get<bool>("OPEN_BC")) {assert(R.size() ==   N_sites and "Use an (N_sites)x(N_sites) hopping matrix for open BC!");}
			else                        {assert(R.size() >= 2*N_sites and "Use at least a (2N_sites)x(2N_sites) hopping matrix for infinite BC!");}
			
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
					auto PsiDag_loc = F[loc].cdag(0);
					auto Sign_loc   = F[loc].sign();
					auto Psi_range  = F[(loc+range)%N_sites].c(0);
					auto PsiDagSign_loc = OperatorType::prod(PsiDag_loc, Sign_loc, {2,2});
					
//					cout << "loc=" << loc << ", pushing at range=" << range << ", value=" << value << endl;
					Terms.push(range, loc, -2.*value, // std::sqrt(2.) * std::sqrt(2.)
					           PsiDagSign_loc.plain<double>(), TransOps, Psi_range.plain<double>());
				}
			}
			
			stringstream ss;
			ss << "tᵢⱼ(avg=" << Geometry2D::avg(Full) << ",σ=" << Geometry2D::sigma(Full) << ",max=" << Geometry2D::max(Full) << ")";
			Terms.save_label(loc,ss.str());
		}
		
		if (P.HAS("Vfull"))
		{
			ArrayXXd Full = P.get<Eigen::ArrayXXd>("Vfull");
			vector<vector<std::pair<size_t,double> > > R = Geometry2D::rangeFormat(Full);
			
			if (P.get<bool>("OPEN_BC")) {assert(R.size() ==   N_sites and "Use an (N_sites)x(N_sites) hopping matrix for open BC!");}
			else                        {assert(R.size() >= 2*N_sites and "Use at least a (2N_sites)x(2N_sites) hopping matrix for infinite BC!");}
			
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
					auto Tdag_loc = F[loc].Tdag(0);
					auto T_hop    = F[(loc+range)%N_sites].T(0);
					
					Terms.push(range, loc, std::sqrt(3.) * value,
					           Tdag_loc.plain<double>(), TransOps, T_hop.plain<double>());
				}
			}
			
			stringstream ss;
			ss << "Vᵢⱼ(avg=" << Geometry2D::avg(Full) << ",σ=" << Geometry2D::sigma(Full) << ",max=" << Geometry2D::max(Full) << ")";
			Terms.save_label(loc,ss.str());
		}
		
		if (P.HAS("Jfull"))
		{
			ArrayXXd Full = P.get<Eigen::ArrayXXd>("Jfull");
			vector<vector<std::pair<size_t,double> > > R = Geometry2D::rangeFormat(Full);
			
			if (P.get<bool>("OPEN_BC")) {assert(R.size() ==   N_sites and "Use an (N_sites)x(N_sites) hopping matrix for open BC!");}
			else                        {assert(R.size() >= 2*N_sites and "Use at least a (2N_sites)x(2N_sites) hopping matrix for infinite BC!");}
			
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
		
		// Local terms: Hubbard-U, t⟂, V⟂, J⟂
		
		param1d U = P.fill_array1d<double>("U", "Uorb", orbitals, loc%Lcell);
		param2d tperp = P.fill_array2d<double>("tRung", "t", "tPerp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Vperp = P.fill_array2d<double>("Vrung", "V", "Vperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		param2d Jperp = P.fill_array2d<double>("Jrung", "J", "Jperp", orbitals, loc%Lcell, P.get<bool>("CYLINDER"));
		
		Terms.save_label(loc, U.label);
		Terms.save_label(loc, tperp.label);
		Terms.save_label(loc, Vperp.label);
		Terms.save_label(loc, Jperp.label);
		
		Terms.push_local(loc, 1., F[loc].HubbardHamiltonian(U.a, tperp.a, Vperp.a, Jperp.a).plain<double>());
		
		// Nearest-neighbour terms: t, V, J
		
		if (!P.HAS("tFull"))
		{
			param2d tpara = P.fill_array2d<double>("t", "tPara", {orbitals, next_orbitals}, loc%Lcell);
			Terms.save_label(loc, tpara.label);
			
			if (loc < N_sites-1 or !P.get<bool>("OPEN_BC"))
			{
				for (std::size_t alfa=0; alfa<orbitals;      ++alfa)
				for (std::size_t beta=0; beta<next_orbitals; ++beta)
				{
					SiteOperator<Symmetry,double> cdag_sign_loc = OperatorType::prod(F[loc].cdag(alfa), F[loc].sign(), {2,2}).plain<double>();
					SiteOperator<Symmetry,double> c_tight       = F[lp1].c(beta).plain<double>();
					
					Terms.push_tight(loc, -tpara(alfa,beta) * 2., cdag_sign_loc, c_tight); // std::sqrt(2.) * std::sqrt(2.)
				}
			}
		}
		
		if (!P.HAS("Vfull"))
		{
			param2d Vpara = P.fill_array2d<double>("V", "Vpara", {orbitals, next_orbitals}, loc%Lcell);
			Terms.save_label(loc, Vpara.label);
			
			if (loc < N_sites-1 or !P.get<bool>("OPEN_BC"))
			{
				for (std::size_t alfa=0; alfa<orbitals;      ++alfa)
				for (std::size_t beta=0; beta<next_orbitals; ++beta)
				{
					SiteOperator<Symmetry,double> Tdag_loc = F[loc].Tdag(alfa).plain<double>();
					SiteOperator<Symmetry,double> T_tight  = F[lp1].T(beta).plain<double>();
					
					Terms.push_tight(loc,  Vpara(alfa,beta) * std::sqrt(3.), Tdag_loc, T_tight);
				}
			}
		}
		
		if (!P.HAS("Jfull"))
		{
			param2d Jpara = P.fill_array2d<double>("J", "Jpara", {orbitals, next_orbitals}, loc%Lcell);
			Terms.save_label(loc, Jpara.label);
			
			if (loc < N_sites-1 or !P.get<bool>("OPEN_BC"))
			{
				for (std::size_t alfa=0; alfa<orbitals;      ++alfa)
				for (std::size_t beta=0; beta<next_orbitals; ++beta)
				{
					SiteOperator<Symmetry,double> Sdag_loc = F[loc].Sdag(alfa).plain<double>();
					SiteOperator<Symmetry,double> S_tight  = F[lp1].S(beta).plain<double>();
					
					Terms.push_tight(loc,  Jpara(alfa,beta) * std::sqrt(3.), Sdag_loc, S_tight);
				}
			}
		}
		
		if (!P.HAS("tFull"))
		{
			// tPrimePrime
			param2d tPrimePrime = P.fill_array2d<double>("tPrimePrime", "tPrimePrime_array", {orbitals, next3_orbitals}, loc%Lcell);
			Terms.save_label(loc, tPrimePrime.label);
			
			if (loc < N_sites-2 or !P.get<bool>("OPEN_BC"))
			{
				vector<SiteOperator<Symmetry,double> > TransOps(2);
				TransOps[0] = F[lp1].sign().plain<double>();
				TransOps[1] = F[lp2].sign().plain<double>();
				
				for (std::size_t alfa=0; alfa<orbitals;       ++alfa)
				for (std::size_t beta=0; beta<next3_orbitals; ++beta)
				{
					SiteOperator<Symmetry, double> cdag_loc_sign = OperatorType::prod(F[loc].cdag(alfa), F[loc].sign(), {2,2}).plain<double>();
					SiteOperator<Symmetry, double> c_lp3         = F[lp3].c(beta).plain<double>();
					
					Terms.push(3, loc, -tPrimePrime(alfa,beta) * 2., cdag_loc_sign, TransOps, c_lp3); // std::sqrt(2.) * std::sqrt(2.)
				}
			}
		}
	}
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > HubbardSU2xSU2::
make_local (string name, size_t locx, size_t locy, const OperatorType &Op, double factor, bool FERMIONIC, bool HERMITIAN) const
{
	assert(locx<F.size() and locy<F[locx].dim());
	stringstream ss;
	ss << name << "(" << locx << "," << locy << ")";
	
	Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > Mout(N_sites, Op.Q(), ss.str(), HERMITIAN);
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	
	(FERMIONIC)? Mout.setLocal(locx, (factor * Op).plain<double>(), F[0].sign().plain<double>())
	           : Mout.setLocal(locx, Op.plain<double>());
	
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > HubbardSU2xSU2::
make_corr (string name1, string name2, size_t locx1, size_t locx2, size_t locy1, size_t locy2,
		   const OperatorType &Op1, const OperatorType &Op2, qarray<Symmetry::Nq> Qtot,
		   double factor, bool FERMIONIC, bool HERMITIAN) const
{
	assert(locx1<F.size() and locy1<F[locx1].dim());
	assert(locx2<F.size() and locy2<F[locx2].dim());
	
	stringstream ss;
	ss << name1 << "(" << locx1 << "," << locy1 << ")"
	   << name2 << "(" << locx2 << "," << locy2 << ")";
	
	Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > Mout(N_sites, Qtot, ss.str(), HERMITIAN);
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	
	if (FERMIONIC)
	{
		if (locx1 == locx2)
		{
			//The diagonal element is actually 2*unity by the symmetry. But we may leave this as a check.
			Mout.setLocal(locx1, factor * OperatorType::prod(Op1,Op2,Qtot).plain<double>());
		}
		else if (locx1<locx2)
		{
			Mout.setLocal({locx1, locx2}, {factor * OperatorType::prod(Op1, F[locx1].sign(), Op1.Q()).plain<double>(), 
										   Op2.plain<double>()}, 
				F[0].sign().plain<double>());
		}
		else if (locx1>locx2)
		{
			Mout.setLocal({locx2, locx1}, {factor * OperatorType::prod(Op2, F[locx2].sign(), Op2.Q()).plain<double>(), 
										   -1. * Op1.plain<double>()}, 
				F[0].sign().plain<double>());
		}
	}
	else
	{
		if (locx1 == locx2)
		{
			auto product = factor*OperatorType::prod(Op1, Op2, Qtot);
			Mout.setLocal(locx1, product.plain<double>());
		}
		else
		{
			Mout.setLocal({locx1, locx2}, {(factor*Op1).plain<double>(), Op2.plain<double>()});
		}
	}
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > HubbardSU2xSU2::
c (size_t locx, size_t locy, double factor)
{
	return make_local("c", locx,locy, F[locx].c(locy), factor, true, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > HubbardSU2xSU2::
cdag (size_t locx, size_t locy, double factor)
{
	return make_local("c†", locx,locy, F[locx].cdag(locy), factor, true, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > HubbardSU2xSU2::
cdagc (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
{
	return make_corr("c†", "c", locx1, locx2, locy1, locy2, F[locx1].cdag(locy1), F[locx2].c(locy2), Symmetry::qvacuum(), 2., PROP::FERMIONIC, PROP::HERMITIAN);
	// 2 = sqrt(2)*sqrt(2)
	// assert(locx1<this->N_sites and locx2<this->N_sites);
	// stringstream ss;
	// ss << "c†(" << locx1 << "," << locy1 << ")" << "c(" << locx2 << "," << locy2 << ")";
	
	// Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
	// for (size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F[l].get_basis().qloc(),l); }
	
	// auto cdag = F[locx1].cdag(locy1);
	// auto c    = F[locx2].c   (locy2);
	
	// if (locx1 == locx2)
	// {
	// 	//The diagonal element is actually 2*unity by the symmetry. But we may leave this as a check.
	// 	Mout.setLocal(locx1, sqrt(2.) * sqrt(2.) * OperatorType::prod(cdag,c,Symmetry::qvacuum()).plain<double>());
	// }
	// else if (locx1<locx2)
	// {
	// 	Mout.setLocal({locx1, locx2}, {sqrt(2.) * sqrt(2.) * OperatorType::prod(cdag, F[locx1].sign(), {2,2}).plain<double>(), 
	// 	                               c.plain<double>()}, 
	// 	                               F[0].sign().plain<double>());
	// }
	// else if (locx1>locx2)
	// {
	// 	Mout.setLocal({locx2, locx1}, {sqrt(2.) * sqrt(2.) * OperatorType::prod(c, F[locx2].sign(), {2,2}).plain<double>(), 
	// 	                               -1. * cdag.plain<double>()}, 
	// 	                               F[0].sign().plain<double>());
	// }
	// return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > HubbardSU2xSU2::
nh (size_t locx, size_t locy)
{
	return make_local("nh", locx,locy, F[locx].nh(locy), 1., false, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > HubbardSU2xSU2::
ns (size_t locx, size_t locy)
{
	return make_local("ns", locx,locy, F[locx].ns(locy), 1., false, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > HubbardSU2xSU2::
S (size_t locx, size_t locy)
{
	return make_local("S", locx,locy, F[locx].S(locy), 1., false, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > HubbardSU2xSU2::
Sdag (size_t locx, size_t locy, double factor)
{
	return make_local("S†", locx,locy, F[locx].Sdag(locy), factor, false, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > HubbardSU2xSU2::
T (size_t locx, size_t locy)
{
	return make_local("T", locx,locy, F[locx].T(locy), 1., false, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > HubbardSU2xSU2::
Tdag (size_t locx, size_t locy, double factor)
{
	return make_local("T†", locx,locy, F[locx].Tdag(locy), factor, false, false);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > HubbardSU2xSU2::
SdagS (std::size_t locx1, std::size_t locx2, std::size_t locy1, std::size_t locy2)
{
	return make_corr("S†", "S", locx1, locx2, locy1, locy2, F[locx1].Sdag(locy1), F[locx2].S(locy2), 
	                 Symmetry::qvacuum(), std::sqrt(3.), PROP::NON_FERMIONIC, PROP::HERMITIAN);
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > HubbardSU2xSU2::
TdagT (std::size_t locx1, std::size_t locx2, std::size_t locy1, std::size_t locy2)
{
	return make_corr("T†", "T", locx1, locx2, locy1, locy2, F[locx1].Tdag(locy1), F[locx2].T(locy2), 
	                 Symmetry::qvacuum(), std::sqrt(3.), PROP::NON_FERMIONIC, PROP::HERMITIAN);
}

} //end namespace VMPS
#endif
