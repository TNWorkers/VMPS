#ifndef SPINSITE_H_
#define SPINSITE_H_

#include "symmetry/U0.h"
#include "symmetry/U1.h"
#include "symmetry/SU2.h"
#include "symmetry/S1xS2.h"

#include "DmrgTypedefs.h"
#include "sites/SpinSiteXxSU2.h"
#include "sites/SpinSiteSU2.h"
#include "sites/SpinSiteSU2xX.h"

template <typename Symmetry_, size_t order=0ul>
class SpinSite
{
	typedef double Scalar;
	typedef Symmetry_ Symmetry;
	typedef SiteOperatorQ<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > OperatorType;
public:
	SpinSite() {};
	SpinSite(std::size_t D_input);
	
	OperatorType Id_1s() const {return Id_1s_;}
	OperatorType F_1s() const {return F_1s_;}
	
	OperatorType n_1s() const {return n_1s(UP) + n_1s(DN);}

	OperatorType Sz_1s() const {return Sz_1s_;}
	OperatorType Sp_1s() const {return Sp_1s_;}
	OperatorType Sm_1s() const {return Sm_1s_;}

	Qbasis<Symmetry> basis_1s() const {return basis_1s_;}

protected:
	std::size_t D;
	
	void fill_basis();
	void fill_SiteOps();

	/**Returns the quantum numbers of the operators for the different combinations of U1 symmetries.*/
	typename Symmetry_::qType getQ (SPINOP_LABEL Sa) const;
	
	Qbasis<Symmetry> basis_1s_;
	
	OperatorType Id_1s_; //identity
	OperatorType F_1s_; //Fermionic sign
	
	OperatorType n_1s_; //particle number
	
	OperatorType Sz_1s_; //orbital spin
	OperatorType Sp_1s_; //orbital spin
	OperatorType Sm_1s_; //orbital spin	
};

template<typename Symmetry_, size_t order>
SpinSite<Symmetry_,order>::
SpinSite(std::size_t D_input)
:D(D_input)
{
	//create basis for one spin site
	fill_basis();
	
	// cout << "single site basis" << endl << this->basis_1s_ << endl;
	fill_SiteOps();
}

template<typename Symmetry_, size_t order>
void SpinSite<Symmetry_,order>::
fill_SiteOps()
{
	Id_1s_  = OperatorType(Symmetry::qvacuum(),basis_1s_);
	Id_1s_.setIdentity();
	
	F_1s_  = OperatorType(Symmetry::qvacuum(),basis_1s_);
	Sz_1s_ = OperatorType(getQ(SZ),basis_1s_);
	Sp_1s_ = OperatorType(getQ(SP),basis_1s_);
	Sm_1s_ = OperatorType(getQ(SM),basis_1s_);
	
	OperatorType Sbase  = OperatorType(getQ(SP),basis_1s_);
	
	double S = 0.5*(D-1);
	size_t Sx2 = D-1;
	
	for (size_t i=0; i<D-1; ++i)
	{
		double m = -S + static_cast<double>(i);
		int Q = -static_cast<int>(Sx2) + 2*static_cast<int>(i);
		int Qplus1 = Q +2; //note spacing of m is 2 because we deal with 2*m instead of m
		stringstream ssQ; ssQ << Q;
		stringstream ssQplus1; ssQplus1 << Qplus1;
		Sbase(ssQplus1.str(),ssQ.str()) = 0.5*sqrt(S*(S+1.)-m*(m+1.));
	}
	
	Sp_1s_ = 2.*Sbase;
	Sm_1s_ = Sp_1s_.adjoint();
	Sz_1s_ = 0.5 * (Sp_1s_ * Sm_1s_ - Sm_1s_*Sp_1s_);
	
	F_1s_ = 0.5*Id_1s_ - Sz_1s_;
	return;
}

template<typename Symmetry_, size_t order>
void SpinSite<Symmetry_,order>::
fill_basis()
{
	if constexpr (Symmetry::NO_SPIN_SYM()) //U0
	{
		typename Symmetry::qType Q=Symmetry::qvacuum();
		Eigen::Index inner_dim=D;
		std::vector<std::string> ident;
		
		assert(D >= 1);
		double S = 0.5*(D-1);
		size_t Sx2 = D-1;
		for (size_t i=0; i<D; ++i)
		{
			int Qint = -static_cast<int>(Sx2) + 2*static_cast<int>(i);
			inner_dim=1;
			stringstream ss; ss << Qint;
			ident.push_back(ss.str());
		}
		basis_1s_.push_back(Q,inner_dim,ident);
	}
	else if constexpr (Symmetry::IS_SPIN_U1()) //spin U1
	{
		typename Symmetry::qType Q;
		Eigen::Index inner_dim;
		std::vector<std::string> ident;
		
		assert(D >= 1);
		double S = 0.5*(D-1);
		size_t Sx2 = D-1;
		
		for (size_t i=0; i<D; ++i)
		{
			int Qint = -static_cast<int>(Sx2) + 2*static_cast<int>(i);
			if constexpr (Symmetry::Nq>1)
			{
				for (size_t q=0; q<Symmetry::Nq; q++)
				{
					Q[q] = (Symmetry::kind()[q] == Sym::KIND::M and q==order)? Qint:0;
				}
			}
			else
			{
				Q[0] = Qint;
			}
			inner_dim=1;
			stringstream ss; ss << Qint;
			ident.push_back(ss.str());
			basis_1s_.push_back(Q, inner_dim, ident);
			ident.clear();
		}
	}
}

template<typename Symmetry_, size_t order>
typename Symmetry_::qType SpinSite<Symmetry_,order>::
getQ (SPINOP_LABEL Sa) const
{
	if constexpr (Symmetry::NO_SPIN_SYM()) {return Symmetry::qvacuum();}
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
		else if constexpr (Symmetry::kind()[0] == Sym::KIND::M and Symmetry::kind()[1] == Sym::KIND::M)
		{
			if (order == 0ul)
			{
				if      (Sa==SZ) {out = {0,0};}
				else if (Sa==SP) {out = {+2,0};}
				else if (Sa==SM) {out = {-2,0};}
			}
			else
			{
				if      (Sa==SZ) {out = {0,0};}
				else if (Sa==SP) {out = {0,+2};}
				else if (Sa==SM) {out = {0,-2};}
			}
		}
//		cout << "order=" << order << ", Sa=" << Sa << ", out=" << out << endl;
		return out;
	}
	static_assert("You inserted a Symmetry which can not be handled by FermionBase.");
}

#endif //FERMIONSITE_H_
