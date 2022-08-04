#ifndef SPINLESSFERMIONSITE_H_
#define SPINLESSFERMIONSITE_H_

#include "symmetry/kind_dummies.h"
#include "symmetry/U0.h"
#include "symmetry/U1.h"

template <typename Symmetry_>
class SpinlessFermionSite
{
	typedef double Scalar;
	typedef Symmetry_ Symmetry;
	typedef SiteOperatorQ<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > OperatorType;
	
public:
	
	SpinlessFermionSite();
	
	OperatorType Id_1s() const {return Id_1s_;}
	OperatorType F_1s() const {return F_1s_;}
	
	OperatorType c_1s() const { return c_1s_;}
	OperatorType cdag_1s() const { return cdag_1s_;}
	OperatorType n_1s() const {return n_1s_;}
	OperatorType nph_1s() const {return nph_1s_;}
	
	Qbasis<Symmetry> basis_1s() const {return basis_1s_;}
	
protected:
	
	void fill_basis();
	void fill_SiteOps();
	
	typename Symmetry_::qType getQ (int Delta) const;
	
	Qbasis<Symmetry> basis_1s_;
	
	OperatorType Id_1s_; //identity
	OperatorType F_1s_; //Fermionic sign
	
	OperatorType c_1s_; //annihilation
	OperatorType cdag_1s_; //creation
	OperatorType n_1s_; //particle number
	OperatorType nph_1s_; //particle number-1/2 (particle-hole symmetric)
};

template<typename Symmetry_>
SpinlessFermionSite<Symmetry_>::
SpinlessFermionSite()
{
	fill_basis();
	fill_SiteOps();
}

template <typename Symmetry_>
void SpinlessFermionSite<Symmetry_>::
fill_SiteOps()
{
	// create operators for one site
	Id_1s_ = OperatorType(Symmetry::qvacuum(),basis_1s_);
	F_1s_  = OperatorType(Symmetry::qvacuum(),basis_1s_);
	
	c_1s_    = OperatorType(getQ(-1),basis_1s_);
	cdag_1s_ = OperatorType(getQ(+1),basis_1s_);
	n_1s_    = OperatorType(Symmetry::qvacuum(),basis_1s_);
	nph_1s_  = OperatorType(Symmetry::qvacuum(),basis_1s_);
	
	Id_1s_("empty","empty") = 1.;
	Id_1s_("occup","occup") = 1.;
	
	F_1s_("empty","empty") = 1.;
	F_1s_("occup","occup") = -1.;
	
	c_1s_("empty","occup")  = 1.;
	cdag_1s_("occup","empty")  = 1.;
	
//	cout << "c_1s_=" << MatrixXd(c_1s_.template plain<double>().data) << endl;
//	cout << "cdag_1s_=" << MatrixXd(cdag_1s_.template plain<double>().data) << endl;
	
	n_1s_("occup","occup") = 1.;
	
	nph_1s_("occup","occup") = +0.5;
	nph_1s_("empty","empty") = -0.5;
	
	return;
}

template<typename Symmetry_>
void SpinlessFermionSite<Symmetry_>::
fill_basis()
{
	if constexpr (std::is_same<Symmetry, Sym::U0>::value) //U0
	{
		typename Symmetry::qType Q = {};
		Eigen::Index inner_dim;
		std::vector<std::string> ident;
		
		ident.push_back("empty");
		ident.push_back("occup");
		inner_dim = 2;
		basis_1s_.push_back(Q,inner_dim,ident);
		ident.clear();
	}
	else if constexpr (std::is_same<Symmetry, Sym::U1<Sym::ChargeU1> >::value) //charge U1
	{
		typename Symmetry::qType Q;
		Eigen::Index inner_dim;
		std::vector<std::string> ident;
		
		Q = {0};
		ident.push_back("empty");
		inner_dim = 1;
		basis_1s_.push_back(Q,inner_dim,ident);
		ident.clear();
		
		Q = {1};
		inner_dim = 1;
		ident.push_back("occup");
		basis_1s_.push_back(Q,inner_dim,ident);
		ident.clear();
	}
}

template<typename Symmetry_>
typename Symmetry_::qType SpinlessFermionSite<Symmetry_>::
getQ (int Delta) const
{
	if constexpr (Symmetry::IS_TRIVIAL) {return {};}
	else if constexpr (Symmetry::Nq == 1) 
	{
		if constexpr (Symmetry::kind()[0] == Sym::KIND::N) //return particle number as good quantum number.
		{
			typename Symmetry::qType out;
			out = {Delta};
			return out;
		}
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::Z2) //return parity as good quantum number.
		{
			typename Symmetry::qType out;
			out = {posmod<2>(Delta)};
			return out;
		}
		else {assert(false and "Ill-defined KIND of the used Symmetry.");}
	}
	static_assert("You've inserted a symmetry which can not be handled by SpinlessFermionSite.");
}

#endif //FERMIONSITE_H_
