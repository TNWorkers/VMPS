#ifndef FERMIONSITESU2xSU2_H_
#define FERMIONSITESU2xSU2_H_

#include "symmetry/kind_dummies.h"

#include "DmrgTypedefs.h"

#include "symmetry/S1xS2.h"
#include "symmetry/SU2.h"

#include "tensors/SiteOperatorQ.h"

template <typename Symmetry> class FermionSite;

template <>
class FermionSite<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > >
{
	typedef double Scalar;
	typedef Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > Symmetry;
	typedef SiteOperatorQ<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > OperatorType;
public:
	FermionSite() {};
	FermionSite(bool U_IS_INFINITE, bool UPH_IS_INFINITE);
	
	OperatorType Id_1s() const {return Id_1s_;}
	OperatorType F_1s() const {return F_1s_;}
	
	OperatorType c_1s(SUB_LATTICE G) const { if(G == A) {return cA_1s_;} return cB_1s_; }
	OperatorType cdag_1s(SUB_LATTICE G) const {return c_1s(G).adjoint();}

	OperatorType ns_1s() const {return Id_1s()-nh_1s();}
	OperatorType nh_1s() const {return nh_1s_;}

	OperatorType S_1s() const {return S_1s_;}
	
	OperatorType T_1s() const {return T_1s_;}

	Qbasis<Symmetry> basis_1s() const {return basis_1s_;}
protected:
	Qbasis<Symmetry> basis_1s_;

	OperatorType Id_1s_; //identity
	OperatorType F_1s_; //Fermionic sign

	OperatorType cA_1s_; //annihilation
	OperatorType cB_1s_; //annihilation

	OperatorType nh_1s_; //holon number
	OperatorType S_1s_; //orbital spin
	OperatorType T_1s_; //orbital isospin
};

FermionSite<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > >::
FermionSite(bool U_IS_INFINITE, bool UPH_IS_INFINITE)
{

	//create basis for one Fermionic Site
	typename Symmetry::qType Q; //empty occupied state
	Eigen::Index inner_dim;
	std::vector<std::string> ident;
	assert(!U_IS_INFINITE and "For charge SU2, U is not allowed to be infinity. This breaks the Charge-SU2 Symmetry.");
	
	if (!UPH_IS_INFINITE)
	{
		Q = {1,2};
		inner_dim = 1;
		ident.push_back("holon");
		basis_1s_.push_back(Q,inner_dim,ident);
		ident.clear();
	}
	
	Q={2,1}; //spinon state
	inner_dim = 1;
	ident.push_back("spinon");
	basis_1s_.push_back(Q,inner_dim,ident);
	ident.clear();
	
	Id_1s_ = OperatorType({1,1},basis_1s_);
	F_1s_ = OperatorType({1,1},basis_1s_);
	cA_1s_ = OperatorType({2,2},basis_1s_);
	cB_1s_ = OperatorType({2,2},basis_1s_);
	nh_1s_ = OperatorType({1,1},basis_1s_);
	T_1s_ = OperatorType({1,3},basis_1s_);
	S_1s_ = OperatorType({3,1},basis_1s_);
	
	// create operators one orbitals	
	if (!UPH_IS_INFINITE) Id_1s_("holon", "holon") = 1.;
	Id_1s_("spinon", "spinon") = 1.;
	
	if (!UPH_IS_INFINITE) F_1s_("holon", "holon") = 1.;
	F_1s_("spinon", "spinon") = -1.;

	if (!UPH_IS_INFINITE) nh_1s_("holon","holon") = 1.;

	if (!UPH_IS_INFINITE) T_1s_( "holon", "holon" ) = std::sqrt(0.75);

	S_1s_( "spinon", "spinon" ) = std::sqrt(0.75);
	
	if (!UPH_IS_INFINITE) cA_1s_( "spinon", "holon" ) = std::sqrt(2.);
	if (!UPH_IS_INFINITE) cA_1s_( "holon", "spinon" ) = std::sqrt(2.);
	
	if (!UPH_IS_INFINITE) cB_1s_( "spinon", "holon" ) = std::sqrt(2.);
	if (!UPH_IS_INFINITE) cB_1s_( "holon", "spinon" ) = -1.*std::sqrt(2.);
}

#endif //FERMIONSITESU2xSU2_H_
