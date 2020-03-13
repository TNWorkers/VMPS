#ifndef FERMIONSITESU2xU0_H_
#define FERMIONSITESU2xU0_H_

#include "symmetry/kind_dummies.h"

#include "DmrgTypedefs.h"

#include "symmetry/S1xS2.h"
#include "symmetry/SU2.h"
#include "symmetry/U1.h"

#include "tensors/SiteOperatorQ.h"

template <typename Symmetry> class FermionSite;

template <>
class FermionSite<Sym::SU2<Sym::SpinSU2> >
{
	typedef double Scalar;
	typedef Sym::SU2<Sym::SpinSU2> Symmetry;
	typedef SiteOperatorQ<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > OperatorType;
public:
	FermionSite() {};
	FermionSite(bool U_IS_INFINITE, bool UPH_IS_INFINITE);
	
	OperatorType Id_1s() const {return Id_1s_;}
	OperatorType F_1s() const {return F_1s_;}
	
	OperatorType c_1s() const {return c_1s_;}
	OperatorType cdag_1s() const {return cdag_1s_;}

	OperatorType n_1s() const {return n_1s_;}
	OperatorType ns_1s() const {return n_1s() - 2.*d_1s();}
	OperatorType nh_1s() const {return 2.*d_1s() - n_1s() + Id_1s();}
	OperatorType d_1s() const {return d_1s_;}

	OperatorType S_1s() const {return S_1s_;}
	
	OperatorType Tz_1s() const {return 0.5*(n_1s() - Id_1s());}
	OperatorType cc_1s() const {return p_1s_;}
	OperatorType cdagcdag_1s() const {return pdag_1s_;}

	Qbasis<Symmetry> basis_1s() const {return basis_1s_;}
protected:
	Qbasis<Symmetry> basis_1s_;

	OperatorType Id_1s_; //identity
	OperatorType F_1s_; //Fermionic sign
	OperatorType c_1s_; //annihilation
	OperatorType cdag_1s_; //creation
	OperatorType n_1s_; //particle number
	OperatorType d_1s_; //double occupancy
	OperatorType S_1s_; //orbital spin
	OperatorType p_1s_; //pairing
	OperatorType pdag_1s_; //pairing adjoint
};

FermionSite<Sym::SU2<Sym::SpinSU2> >::
FermionSite(bool U_IS_INFINITE, bool UPH_IS_INFINITE)
{

	//create basis for one Fermionic Site
	typename Symmetry::qType Q; //empty occupied state
	Eigen::Index inner_dim;
	std::vector<std::string> ident;
	
	if (!UPH_IS_INFINITE and U_IS_INFINITE)
	{
		Q = {1};
		inner_dim = 1;
		ident.push_back("empty");
		basis_1s_.push_back(Q,inner_dim,ident);
		ident.clear();
	}
	else if (!UPH_IS_INFINITE and !U_IS_INFINITE)
	{
		Q = {1};
		inner_dim = 2;
		ident.push_back("empty");
		ident.push_back("double");
		basis_1s_.push_back(Q,inner_dim,ident);
		ident.clear();
	}
	
	Q={2}; //singly occupied state
	inner_dim = 1;
	ident.push_back("single");
	basis_1s_.push_back(Q,inner_dim,ident);
	ident.clear();
	
	Id_1s_ = OperatorType({1},basis_1s_);
	F_1s_ = OperatorType({1},basis_1s_);
	c_1s_ = OperatorType({2},basis_1s_);
	d_1s_ = OperatorType({1},basis_1s_);
	S_1s_ = OperatorType({3},basis_1s_);
	
	// create operators one orbitals	
	if (!UPH_IS_INFINITE) Id_1s_("empty", "empty") = 1.;
	
	if (!U_IS_INFINITE and !UPH_IS_INFINITE) Id_1s_("double", "double") = 1.;
	Id_1s_("single", "single") = 1.;
	
	if (!UPH_IS_INFINITE) F_1s_("empty", "empty") = 1.;
	if (!U_IS_INFINITE and !UPH_IS_INFINITE) F_1s_("double", "double") = 1.;
	F_1s_("single", "single") = -1.;
	
	if (!UPH_IS_INFINITE) c_1s_("empty", "single")  = std::sqrt(2.);
	if (!U_IS_INFINITE and !UPH_IS_INFINITE) c_1s_("single", "double") = 1.;
	
	cdag_1s_ = c_1s_.adjoint();
	n_1s_ = std::sqrt(2.) * OperatorType::prod(cdag_1s_,c_1s_,{1});
	if (!U_IS_INFINITE and !UPH_IS_INFINITE) d_1s_( "double", "double" ) = 1.;
	S_1s_("single", "single") = std::sqrt(0.75);
	p_1s_ = -std::sqrt(0.5) * OperatorType::prod(c_1s_,c_1s_,{1}); //The sign convention corresponds to c_DN c_UP
	pdag_1s_ = p_1s_.adjoint(); //The sign convention corresponds to (c_DN c_UP)†=c_UP† c_DN†

}

#endif //FERMIONSITESU2xU0_H_
