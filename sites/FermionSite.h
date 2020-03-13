#ifndef FERMIONSITE_H_
#define FERMIONSITE_H_

#include "symmetry/U0.h"

#include "sites/FermionSiteSU2xU1.h"
#include "sites/FermionSiteU1xSU2.h"
#include "sites/FermionSiteSU2xU0.h"
#include "sites/FermionSiteU0xSU2.h"
#include "sites/FermionSiteSU2xSU2.h"

template <typename Symmetry_>
class FermionSite
{
	typedef double Scalar;
	typedef Symmetry_ Symmetry;
	typedef SiteOperatorQ<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > OperatorType;
public:
	FermionSite() {};
	FermionSite(bool U_IS_INFINITE, bool UPH_IS_INFINITE);
	
	OperatorType Id_1s() const {return Id_1s_;}
	OperatorType F_1s() const {return F_1s_;}
	
	OperatorType c_1s(SPIN_INDEX sigma) const { if (sigma == UP) {return cup_1s_;} return cdn_1s_;}
	OperatorType cdag_1s(SPIN_INDEX sigma) const { if (sigma == UP) {return cup_1s_.adjoint();} return cdn_1s_.adjoint();}

	OperatorType n_1s() const {return n_1s(UP) + n_1s(DN);}
	OperatorType n_1s(SPIN_INDEX sigma) const { if (sigma == UP) {return nup_1s_;} return ndn_1s_;}
	OperatorType ns_1s() const {return n_1s() - 2.*d_1s();}
	OperatorType nh_1s() const {return 2.*d_1s() - n_1s() + Id_1s();}
	OperatorType d_1s() const {return d_1s_;}

	OperatorType Sz_1s() const {return Sz_1s_;}
	OperatorType Sp_1s() const {return Sp_1s_;}
	OperatorType Sm_1s() const {return Sm_1s_;}

	OperatorType Tz_1s() const {return 0.5*(n_1s() - Id_1s());}
	OperatorType cc_1s() const {return cc_1s_;}
	OperatorType cdagcdag_1s() const {return cdagcdag_1s_;}

	Qbasis<Symmetry> basis_1s() const {return basis_1s_;}

protected:
	void fill_basis (bool U_IS_INFINITE, bool UPH_IS_INFINITE);
	void fill_SiteOps (bool U_IS_INFINITE, bool UPH_IS_INFINITE);
	
	typename Symmetry_::qType getQ (SPIN_INDEX sigma, int Delta) const;
	typename Symmetry_::qType getQ (SPINOP_LABEL Sa) const;
		
	Qbasis<Symmetry> basis_1s_;

	OperatorType Id_1s_; //identity
	OperatorType F_1s_; //Fermionic sign
	OperatorType cup_1s_; //annihilation
	OperatorType cdn_1s_; //annihilation
	
	OperatorType n_1s_; //particle number
	OperatorType nup_1s_; //particle number
	OperatorType ndn_1s_; //particle number
	OperatorType d_1s_; //double occupancy

	OperatorType Sz_1s_; //orbital spin
	OperatorType Sp_1s_; //orbital spin
	OperatorType Sm_1s_; //orbital spin
	
	OperatorType Tz_1s_; //orbital pseude spin
	OperatorType cc_1s_; //pairing
	OperatorType cdagcdag_1s_; //pairing adjoint
};

template<typename Symmetry_>
FermionSite<Symmetry_>::
FermionSite(bool U_IS_INFINITE, bool UPH_IS_INFINITE)
{
	//create basis for one Fermionic Site
	fill_basis(U_IS_INFINITE, UPH_IS_INFINITE);

	// cout << "single site basis" << endl << this->basis_1s_ << endl;

	fill_SiteOps(U_IS_INFINITE, UPH_IS_INFINITE);
}

template <typename Symmetry_>
void FermionSite<Symmetry_>::
fill_SiteOps(bool U_IS_INFINITE, bool UPH_IS_INFINITE)
{
   	Id_1s_  = OperatorType(Symmetry::qvacuum(),basis_1s_);
	F_1s_   = OperatorType(Symmetry::qvacuum(),basis_1s_);
	cup_1s_ = OperatorType(getQ(UP,-1),basis_1s_);
	cdn_1s_ = OperatorType(getQ(DN,-1),basis_1s_);
	d_1s_   =  OperatorType(Symmetry::qvacuum(),basis_1s_);
	Sz_1s_  = OperatorType(getQ(SZ),basis_1s_);
	Sp_1s_  = OperatorType(getQ(SP),basis_1s_);
	Sm_1s_  = OperatorType(getQ(SM),basis_1s_);
	
	// create operators for one orbital
	if (!UPH_IS_INFINITE) Id_1s_("empty", "empty") = 1.;
	
	if (!U_IS_INFINITE and !UPH_IS_INFINITE) Id_1s_("double", "double") = 1.;
	Id_1s_("up", "up") = 1.;
	Id_1s_("dn", "dn") = 1.;
	
	if (!UPH_IS_INFINITE) F_1s_("empty", "empty") = 1.;
	if (!U_IS_INFINITE and !UPH_IS_INFINITE) F_1s_("double", "double") = 1.;
	F_1s_("up", "up") = -1.;
	F_1s_("dn", "dn") = -1.;
	
	if (!UPH_IS_INFINITE) cup_1s_("empty", "up")  = 1.;
	if (!U_IS_INFINITE and !UPH_IS_INFINITE) cup_1s_("dn", "double") = 1.;

	if (!UPH_IS_INFINITE) cdn_1s_("empty", "dn")  = 1.;
	if (!U_IS_INFINITE and !UPH_IS_INFINITE) cdn_1s_("up", "double") = -1.;
	
	nup_1s_ = cup_1s_.adjoint() * cup_1s_;
	ndn_1s_ = cdn_1s_.adjoint() * cdn_1s_;
	if (!U_IS_INFINITE and !UPH_IS_INFINITE) d_1s_( "double", "double" ) = 1.;
	
	Sz_1s_ = 0.5*(nup_1s_ - ndn_1s_);
	Sp_1s_ = cup_1s_.adjoint() * cdn_1s_;
	Sm_1s_ = Sp_1s_.adjoint();
	cc_1s_ = cdn_1s_ * cup_1s_; //The sign convention corresponds to c_DN c_UP
	cdagcdag_1s_ = cc_1s_.adjoint(); //The sign convention corresponds to (c_DN c_UP)†=c_UP† c_DN†
	return;
}

template<typename Symmetry_>
void FermionSite<Symmetry_>::
fill_basis (bool U_IS_INFINITE, bool UPH_IS_INFINITE)
{
	if
		constexpr (std::is_same<Symmetry, Sym::S1xS2<Sym::U1<Sym::SpinU1>,Sym::U1<Sym::ChargeU1> > >::value) //U1xU1
				  {
					  typename Symmetry::qType Q={0,0}; //empty occupied state
					  Eigen::Index inner_dim = 1;
					  std::vector<std::string> ident;
	
					  if (!UPH_IS_INFINITE)
					  {
						  ident.push_back("empty");
						  this->basis_1s_.push_back(Q,inner_dim,ident);
						  ident.clear();
					  }
	
					  Q={+1,1}; //up spin state
					  inner_dim = 1;
					  ident.push_back("up");
					  this->basis_1s_.push_back(Q,inner_dim,ident);
					  ident.clear();

					  Q={-1,1}; //down spin state
					  inner_dim = 1;
					  ident.push_back("dn");
					  this->basis_1s_.push_back(Q,inner_dim,ident);
					  ident.clear();
	
					  if (!U_IS_INFINITE and !UPH_IS_INFINITE)
					  {
						  Q={0,2}; //doubly occupied state
						  inner_dim = 1;
						  ident.push_back("double");
						  this->basis_1s_.push_back(Q,inner_dim,ident);
						  ident.clear();
					  }
				  }
	else if
		constexpr (std::is_same<Symmetry, Sym::U0>::value) //U0
				  {
					  typename Symmetry::qType Q = {}; //empty and doubly occupied state
					  Eigen::Index inner_dim;
					  std::vector<std::string> ident;
	
					  if (!UPH_IS_INFINITE and U_IS_INFINITE)
					  {
						  ident.push_back("empty");
						  ident.push_back("up");
						  ident.push_back("dn");
						  inner_dim = 3;
						  basis_1s_.push_back(Q,inner_dim,ident);
						  ident.clear();
					  }
					  else if (!U_IS_INFINITE and !UPH_IS_INFINITE)
					  {
						  ident.push_back("empty");
						  ident.push_back("up");
						  ident.push_back("dn");
						  ident.push_back("double");
						  inner_dim = 4;
						  basis_1s_.push_back(Q,inner_dim,ident);
						  ident.clear();
					  }
					  else
					  {
						  ident.push_back("up");
						  ident.push_back("dn");
						  inner_dim = 2;
						  basis_1s_.push_back(Q,inner_dim,ident);
						  ident.clear();		
					  }
				  }
	else if
		constexpr (std::is_same<Symmetry, Sym::U1<Sym::SpinU1> >::value) //spin U1
				  {
					  typename Symmetry::qType Q; //empty and doubly occupied state
					  Eigen::Index inner_dim;
					  std::vector<std::string> ident;
	
					  if (!UPH_IS_INFINITE and U_IS_INFINITE)
					  {
						  ident.push_back("empty");
						  inner_dim = 1;
						  Q = {0};
						  basis_1s_.push_back(Q,inner_dim,ident);
						  ident.clear();
					  }
					  else if (!U_IS_INFINITE and !UPH_IS_INFINITE)
					  {
						  Q={0}; //doubly occupied state
						  inner_dim = 2;
						  ident.push_back("empty");
						  ident.push_back("double");
						  basis_1s_.push_back(Q,inner_dim,ident);
						  ident.clear();
					  }

					  Q={+1}; //up spin state
					  inner_dim = 1;
					  ident.push_back("up");
					  basis_1s_.push_back(Q,inner_dim,ident);
					  ident.clear();

					  Q={-1}; //down spin state
					  inner_dim = 1;
					  ident.push_back("dn");
					  basis_1s_.push_back(Q,inner_dim,ident);
					  ident.clear();
				  }
	else if
		constexpr (std::is_same<Symmetry, Sym::U1<Sym::ChargeU1> >::value) //charge U1
				  {
					  typename Symmetry::qType Q; //empty and doubly occupied state
					  Eigen::Index inner_dim;
					  std::vector<std::string> ident;

					  if (!UPH_IS_INFINITE)
					  {
						  Q = {0};
						  ident.push_back("empty");
						  inner_dim = 1;
						  basis_1s_.push_back(Q,inner_dim,ident);
						  ident.clear();
					  }

					  Q={1}; //single occupied states
					  inner_dim = 2;
					  ident.push_back("up");
					  ident.push_back("dn");
					  basis_1s_.push_back(Q,inner_dim,ident);
					  ident.clear();

					  if (!U_IS_INFINITE and !UPH_IS_INFINITE)
					  {
						  Q={2}; //doubly occupied state
						  inner_dim = 1;
						  ident.push_back("double");
						  basis_1s_.push_back(Q,inner_dim,ident);
						  ident.clear();
					  }
				  }
}

template<typename Symmetry_>
typename Symmetry_::qType FermionSite<Symmetry_>::
getQ (SPIN_INDEX sigma, int Delta) const
{
	if constexpr (Symmetry::IS_TRIVIAL) {return {};}
	else if constexpr (Symmetry::Nq == 1) 
	{
		if constexpr (Symmetry::kind()[0] == Sym::KIND::N) //return particle number as good quantum number.
		{
			typename Symmetry::qType out;
			if      (sigma==UP)     {out = {Delta};}
			else if (sigma==DN)     {out = {Delta};}
			else if (sigma==UPDN)   {out = {2*Delta};}
			else if (sigma==NOSPIN) {out = Symmetry::qvacuum();}
			return out;
		}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::M) //return magnetization as good quantum number.
		{
			typename Symmetry::qType out;
			if      (sigma==UP)     {out = {Delta};}
			else if (sigma==DN)     {out = {-Delta};}
			else if (sigma==UPDN)   {out = Symmetry::qvacuum();}
			else if (sigma==NOSPIN) {out = Symmetry::qvacuum();}
			return out;
		}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::Z2) //return parity as good quantum number.
		{
			typename Symmetry::qType out;
			if      (sigma==UP)     {out = {posmod<2>(Delta)};}
			else if (sigma==DN)     {out = {posmod<2>(-Delta)};}
			else if (sigma==UPDN)   {out = Symmetry::qvacuum();}
			else if (sigma==NOSPIN) {out = Symmetry::qvacuum();}
			return out;
		}
		else {assert(false and "Ill defined KIND of the used Symmetry.");}
	}
	else if constexpr (Symmetry::Nq == 2)
	{
		typename Symmetry::qType out;
		if constexpr (Symmetry::kind()[0] == Sym::KIND::N and Symmetry::kind()[1] == Sym::KIND::M)
		{
			if      (sigma==UP)     {out = {Delta,Delta};}
			else if (sigma==DN)     {out = {Delta,-Delta};}
			else if (sigma==UPDN)   {out = {2*Delta,0};}
			else if (sigma==NOSPIN) {out = Symmetry::qvacuum();}
		}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::M and Symmetry::kind()[1] == Sym::KIND::N)
		{
			if      (sigma==UP)     {out = {Delta,Delta};}
			else if (sigma==DN)     {out = {-Delta,Delta};}
			else if (sigma==UPDN)   {out = {0,2*Delta};}
			else if (sigma==NOSPIN) {out = Symmetry::qvacuum();}
		}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::Nup and Symmetry::kind()[1] == Sym::KIND::Ndn)
		{
			if      (sigma==UP)     {out = {Delta,0};}
			else if (sigma==DN)     {out = {0,Delta};}
			else if (sigma==UPDN)   {out = {Delta,Delta};}
			else if (sigma==NOSPIN) {out = Symmetry::qvacuum();}
		}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::Ndn and Symmetry::kind()[1] == Sym::KIND::Nup)
		{
			if      (sigma==UP)     {out = {0,Delta};}
			else if (sigma==DN)     {out = {Delta,0};}
			else if (sigma==UPDN)   {out = {Delta,Delta};}
			else if (sigma==NOSPIN) {out = Symmetry::qvacuum();}
		}
		return out;
	}
	static_assert("You inserted a Symmetry which can not be handled by FermionBase.");
}

template<typename Symmetry_>
typename Symmetry_::qType FermionSite<Symmetry_>::
getQ (SPINOP_LABEL Sa) const
{
	if constexpr (Symmetry::IS_TRIVIAL) {return {};}
	else if constexpr (Symmetry::Nq == 1)
	{
		if constexpr (Symmetry::kind()[0] == Sym::KIND::N or 
		              Symmetry::kind()[0] == Sym::KIND::Z2) //return particle number as good quantum number.
		{
			return Symmetry::qvacuum();
		}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::M) //return magnetization as good quantum number.
		{
			assert(Sa != SX and Sa != iSY);
			
			typename Symmetry::qType out;
			if      (Sa==SZ) {out = {0};}
			else if (Sa==SP) {out = {+2};}
			else if (Sa==SM) {out = {-2};}
			return out;
		}
		else {assert(false and "Ill defined KIND of the used Symmetry.");}
	}
	else if constexpr (Symmetry::Nq == 2)
	{
		assert(Sa != SX and Sa != iSY);
		
		typename Symmetry::qType out;
		if constexpr (Symmetry::kind()[0] == Sym::KIND::N and Symmetry::kind()[1] == Sym::KIND::M)
		{
			if      (Sa==SZ) {out = {0,0};}
			else if (Sa==SP) {out = {0,+2};}
			else if (Sa==SM) {out = {0,-2};}
		}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::M and Symmetry::kind()[1] == Sym::KIND::N)
		{
			if      (Sa==SZ) {out = {0,0};}
			else if (Sa==SP) {out = {+2,0};}
			else if (Sa==SM) {out = {-2,0};}
		}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::Nup and Symmetry::kind()[1] == Sym::KIND::Ndn)
		{
			if      (Sa==SZ) {out = {0,0};}
			else if (Sa==SP) {out = {+1,-1};}
			else if (Sa==SM) {out = {-1,+1};}
		}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::Ndn and Symmetry::kind()[1] == Sym::KIND::Nup)
		{
			if      (Sa==SZ) {out = {0,0};}
			else if (Sa==SP) {out = {-1,+1};}
			else if (Sa==SM) {out = {+1,-1};}
		}
		return out;
	}
	static_assert("You inserted a Symmetry which can not be handled by FermionBase.");
}

#endif //FERMIONSITE_H_
