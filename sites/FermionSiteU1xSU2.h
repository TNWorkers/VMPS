#ifndef FERMIONSITEU1xSU2_H_
#define FERMIONSITEU1xSU2_H_

#include "symmetry/kind_dummies.h"

#include "DmrgTypedefs.h"

#include "symmetry/S1xS2.h"
#include "symmetry/SU2.h"
#include "symmetry/U1.h"

#include "tensors/SiteOperatorQ.h"

template <typename Symmetry> class FermionSite;

template <>
class FermionSite<Sym::S1xS2<Sym::U1<Sym::SpinU1>, Sym::SU2<Sym::ChargeSU2> > >
{
	typedef double Scalar;
	typedef Sym::S1xS2<Sym::U1<Sym::SpinU1>, Sym::SU2<Sym::ChargeSU2> > Symmetry;
	typedef SiteOperatorQ<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > OperatorType;
public:
	FermionSite() {};
	FermionSite(bool U_IS_INFINITE, bool UPH_IS_INFINITE);
	
	OperatorType Id_1s() const {return Id_1s_;}
	OperatorType F_1s() const {return F_1s_;}
	
	OperatorType c_1s(SPIN_INDEX sigma, SUB_LATTICE G) const
		{
			if (sigma == UP and G == A) {return cupA_1s_;}
			else if (sigma == UP and G == B) {return cupB_1s_;}
			else if (sigma == DN and G == A) {return cdnA_1s_;}
			return cdnB_1s_; //else if sigma==DN and G==B
		}
	OperatorType cdag_1s(SPIN_INDEX sigma, SUB_LATTICE G) const {return c_1s(sigma, G).adjoint();}

	OperatorType n_1s() const {return n_1s(UP) + n_1s(DN);}
	OperatorType n_1s(SPIN_INDEX sigma) const { if (sigma == UP) {return nup_1s_;} return ndn_1s_; }
	OperatorType ns_1s() const {return Id_1s()-nh_1s();}
	OperatorType nh_1s() const {return nh_1s_;}

	OperatorType Sz_1s() const {return Sz_1s_;}
	OperatorType Sp_1s() const {return Sp_1s_;}
	OperatorType Sm_1s() const {return Sm_1s_;}
	
	OperatorType T_1s() const {return T_1s_;}

	Qbasis<Symmetry> basis_1s() const {return basis_1s_;}
protected:
	Qbasis<Symmetry> basis_1s_;

	OperatorType Id_1s_; //identity
	OperatorType F_1s_; //Fermionic sign

	OperatorType cupA_1s_; //annihilation
	OperatorType cdnA_1s_; //annihilation
	OperatorType cupB_1s_; //annihilation
	OperatorType cdnB_1s_; //annihilation

	OperatorType nup_1s_; //particle number
	OperatorType ndn_1s_; //particle number
	OperatorType nh_1s_; //holon number
	OperatorType Sz_1s_; //orbital spin
	OperatorType Sp_1s_; //orbital spin
	OperatorType Sm_1s_; //orbital spin
	OperatorType T_1s_; //orbital isospin
};

FermionSite<Sym::S1xS2<Sym::U1<Sym::SpinU1>, Sym::SU2<Sym::ChargeSU2> > >::
FermionSite(bool U_IS_INFINITE, bool UPH_IS_INFINITE)
{

	//create basis for one Fermionic Site
	typename Symmetry::qType Q; //empty occupied state
	Eigen::Index inner_dim;
	std::vector<std::string> ident;
	assert(!U_IS_INFINITE and "For charge SU2, U is not allowed to be infinity. This breaks the Charge-SU2 Symmetry.");
	
	if (!UPH_IS_INFINITE)
	{
		Q = {0,2};
		inner_dim = 1;
		ident.push_back("holon");
		basis_1s_.push_back(Q,inner_dim,ident);
		ident.clear();
	}
	
	Q={+1,1}; //singly occupied state up
	inner_dim = 1;
	ident.push_back("up");
	basis_1s_.push_back(Q,inner_dim,ident);
	ident.clear();

	Q={-1,1}; //singly occupied state down
	inner_dim = 1;
	ident.push_back("dn");
	basis_1s_.push_back(Q,inner_dim,ident);
	ident.clear();
		
	Id_1s_ = OperatorType({0,1},basis_1s_);
	F_1s_ = OperatorType({0,1},basis_1s_);
	cupA_1s_ = OperatorType({-1,2},basis_1s_);
	cupB_1s_ = OperatorType({-1,2},basis_1s_);
	cdnA_1s_ = OperatorType({+1,2},basis_1s_);
	cdnB_1s_ = OperatorType({+1,2},basis_1s_);
	nh_1s_ = OperatorType({0,1},basis_1s_);
	T_1s_ = OperatorType({0,3},basis_1s_);
	
	// create operators one orbitals	
	if (!UPH_IS_INFINITE) Id_1s_("holon", "holon") = 1.;
	Id_1s_("up", "up") = 1.;
	Id_1s_("dn", "dn") = 1.;
	
	if (!UPH_IS_INFINITE) F_1s_("holon", "holon") = 1.;
	F_1s_("up", "up") = -1.;
	F_1s_("dn", "dn") = -1.;

	if (!UPH_IS_INFINITE) nh_1s_("holon","holon") = 1.;

	if (!UPH_IS_INFINITE) T_1s_( "holon", "holon" ) = std::sqrt(0.75);
	
	if (!UPH_IS_INFINITE) cupA_1s_( "dn", "holon" ) = sqrt(2.);
	if (!UPH_IS_INFINITE) cupA_1s_( "holon", "up" ) = -1.;
	
	if (!UPH_IS_INFINITE) cdnA_1s_( "up", "holon" ) = sqrt(2.);
	if (!UPH_IS_INFINITE) cdnA_1s_( "holon", "dn" ) = 1.;

	if (!UPH_IS_INFINITE) cupB_1s_( "dn", "holon" ) = -1.*sqrt(2.);
	if (!UPH_IS_INFINITE) cupB_1s_( "holon", "up" ) = -1.;
	
	if (!UPH_IS_INFINITE) cdnB_1s_( "up", "holon" ) = -1.*sqrt(2.);
	if (!UPH_IS_INFINITE) cdnB_1s_( "holon", "dn" ) = 1.;

	nup_1s_  = std::sqrt(0.5) * OperatorType::prod(cupA_1s_.adjoint(),cupA_1s_,{0,1});
	ndn_1s_	 = std::sqrt(0.5) * OperatorType::prod(cdnA_1s_.adjoint(),cdnA_1s_,{0,1});
	
	Sz_1s_ = 0.5 * (std::sqrt(0.5) * OperatorType::prod(cupA_1s_.adjoint(),cupA_1s_,{0,1}) - std::sqrt(0.5) * OperatorType::prod(cdnA_1s_.adjoint(),cdnA_1s_,{0,1}));
	Sp_1s_ = -std::sqrt(0.5) * OperatorType::prod(cupA_1s_.adjoint(),cdnA_1s_,{+2,1});
	Sm_1s_ = Sp_1s_.adjoint();
}

#endif //FERMIONSITESU2xU1_H_
