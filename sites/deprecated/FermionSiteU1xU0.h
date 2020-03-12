#ifndef FERMIONSITEU1xU0_H_
#define FERMIONSITEU1xU0_H_

#include "DmrgTypedefs.h"

#include "symmetry/S1xS2.h"
#include "symmetry/SU2.h"
#include "symmetry/U1.h"

#include "tensors/SiteOperatorQ.h"

template <typename Symmetry> class FermionSite;

// template <>
// class FermionSite<Sym::U1<Sym::SpinU1> >
// {
// 	typedef double Scalar;
// 	typedef Sym::U1<Sym::SpinU1> Symmetry;
// 	typedef SiteOperatorQ<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > OperatorType;
// public:
// 	FermionSite() {};
// 	FermionSite(bool U_IS_INFINITE, bool UPH_IS_INFINITE);
	
// 	OperatorType Id_1s() const {return Id_1s_;}
// 	OperatorType F_1s() const {return F_1s_;}
	
// 	OperatorType c_1s(SPIN_INDEX sigma) const { if (sigma == UP) {return cup_1s_;} return cdn_1s_;}
// 	OperatorType cdag_1s(SPIN_INDEX sigma) const { if (sigma == UP) {return cup_1s_.adjoint();} return cdn_1s_.adjoint();}

// 	OperatorType n_1s() const {return n_1s(UP) + n_1s(DN);}
// 	OperatorType n_1s(SPIN_INDEX sigma) const { if (sigma == UP) {return nup_1s_;} return ndn_1s_;}
// 	OperatorType ns_1s() const {return n_1s() - 2.*d_1s();}
// 	OperatorType nh_1s() const {return 2.*d_1s() - n_1s() + Id_1s();}
// 	OperatorType d_1s() const {return d_1s_;}

// 	OperatorType Sz_1s() const {return Sz_1s_;}
// 	OperatorType Sp_1s() const {return Sp_1s_;}
// 	OperatorType Sm_1s() const {return Sm_1s_;}

// 	OperatorType Tz_1s() const {return 0.5*(n_1s() - Id_1s());}
// 	OperatorType cc_1s() const {return cc_1s_;}
// 	OperatorType cdagcdag_1s() const  {return cdagcdag_1s_;}

// 	Qbasis<Symmetry> basis_1s() const {return basis_1s_;}
// protected:
	
// 	Qbasis<Symmetry> basis_1s_;

// 	OperatorType Id_1s_; //identity
// 	OperatorType F_1s_; //Fermionic sign
// 	OperatorType cup_1s_; //annihilation
// 	OperatorType cdn_1s_; //annihilation
	
// 	OperatorType n_1s_; //particle number
// 	OperatorType nup_1s_; //particle number
// 	OperatorType ndn_1s_; //particle number
// 	OperatorType d_1s_; //double occupancy

// 	OperatorType Sz_1s_; //orbital spin
// 	OperatorType Sp_1s_; //orbital spin
// 	OperatorType Sm_1s_; //orbital spin
	
// 	OperatorType Tz_1s_; //orbital pseude spin
// 	OperatorType cc_1s_; //pairing
// 	OperatorType cdagcdag_1s_; //pairing adjoint
// };

// FermionSite<Sym::U1<Sym::SpinU1> >::
// FermionSite(bool U_IS_INFINITE, bool UPH_IS_INFINITE)
// {

// 	//create basis for one Fermionic Site
// 	typename Symmetry::qType Q; //empty and doubly occupied state
// 	Eigen::Index inner_dim;
// 	std::vector<std::string> ident;
	
// 	if (!UPH_IS_INFINITE and U_IS_INFINITE)
// 	{
// 		ident.push_back("empty");
// 		inner_dim = 1;
// 		Q = {0};
// 		basis_1s_.push_back(Q,inner_dim,ident);
// 		ident.clear();
// 	}
// 	else if (!U_IS_INFINITE and !UPH_IS_INFINITE)
// 	{
// 		Q={0}; //doubly occupied state
// 		inner_dim = 2;
// 		ident.push_back("empty");
// 		ident.push_back("double");
// 		basis_1s_.push_back(Q,inner_dim,ident);
// 		ident.clear();
// 	}

// 	Q={+1}; //up spin state
// 	inner_dim = 1;
// 	ident.push_back("up");
// 	basis_1s_.push_back(Q,inner_dim,ident);
// 	ident.clear();

// 	Q={-1}; //down spin state
// 	inner_dim = 1;
// 	ident.push_back("dn");
// 	basis_1s_.push_back(Q,inner_dim,ident);
// 	ident.clear();
	
// 	Id_1s_  = OperatorType({0},basis_1s_);
// 	F_1s_   = OperatorType({0},basis_1s_);
// 	cup_1s_ = OperatorType({-1},basis_1s_);
// 	cdn_1s_ = OperatorType({+1},basis_1s_);
// 	d_1s_   =  OperatorType({0},basis_1s_);
// 	Sz_1s_  = OperatorType({0},basis_1s_);
// 	Sp_1s_  = OperatorType({+2},basis_1s_);
// 	Sm_1s_  = OperatorType({-2},basis_1s_);
	
// 	// create operators one orbitals	
// 	if (!UPH_IS_INFINITE) Id_1s_("empty", "empty") = 1.;
	
// 	if (!U_IS_INFINITE and !UPH_IS_INFINITE) Id_1s_("double", "double") = 1.;
// 	Id_1s_("up", "up") = 1.;
// 	Id_1s_("dn", "dn") = 1.;
	
// 	if (!UPH_IS_INFINITE) F_1s_("empty", "empty") = 1.;
// 	if (!U_IS_INFINITE and !UPH_IS_INFINITE) F_1s_("double", "double") = 1.;
// 	F_1s_("up", "up") = -1.;
// 	F_1s_("dn", "dn") = -1.;
	
// 	if (!UPH_IS_INFINITE) cup_1s_("empty", "up")  = 1.;
// 	if (!U_IS_INFINITE and !UPH_IS_INFINITE) cup_1s_("dn", "double") = 1.;

// 	if (!UPH_IS_INFINITE) cdn_1s_("empty", "dn")  = 1.;
// 	if (!U_IS_INFINITE and !UPH_IS_INFINITE) cdn_1s_("up", "double") = -1.;
	
// 	nup_1s_ = cup_1s_.adjoint() * cup_1s_;
// 	ndn_1s_ = cdn_1s_.adjoint() * cdn_1s_;
// 	if (!U_IS_INFINITE and !UPH_IS_INFINITE) d_1s_( "double", "double" ) = 1.;
	
// 	Sz_1s_ = 0.5*(nup_1s_ - ndn_1s_);
// 	Sp_1s_ = cup_1s_.adjoint() * cdn_1s_;
// 	Sm_1s_ = Sp_1s_.adjoint();
// 	cc_1s_ = cdn_1s_ * cup_1s_; //The sign convention corresponds to c_DN c_UP
// 	cdagcdag_1s_ = cc_1s_.adjoint(); //The sign convention corresponds to (c_DN c_UP)†=c_UP† c_DN†

// }

#endif //FERMIONSITESU2xU1_H_
