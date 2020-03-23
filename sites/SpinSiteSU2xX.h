#ifndef SPINSITESU2xX_H_
#define SPINSITESU2xX_H_

#include "symmetry/kind_dummies.h"

#include "DmrgTypedefs.h"

#include "symmetry/S1xS2.h"
#include "symmetry/SU2.h"
#include "symmetry/U1.h"

#include "tensors/SiteOperatorQ.h"

template <typename Symmetry> class SpinSite;

template <typename OtherSym>
class SpinSite<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>, OtherSym > >
{
	typedef double Scalar;
	typedef typename Sym::S1xS2<Sym::SU2<Sym::SpinSU2>, OtherSym > Symmetry;
	typedef SiteOperatorQ<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > OperatorType;
public:
	SpinSite() {};
	SpinSite(std::size_t D_input);
	
	OperatorType Id_1s() const {return Id_1s_;}
	
	OperatorType S_1s() const {return S_1s_;}

	OperatorType Q_1s() const {return Q_1s_;}

	Qbasis<Symmetry> basis_1s() const {return basis_1s_;}
protected:
	std::size_t D;
	
	Qbasis<Symmetry> basis_1s_;

	OperatorType Id_1s_; //identity
	OperatorType S_1s_; //spin
	OperatorType Sdag_1s_; //spin
	OperatorType Q_1s_; //quadrupole moment
};

template<typename OtherSym>
SpinSite<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>, OtherSym> >::
SpinSite(std::size_t D_input)
:D(D_input)
{
	//create basis for one Spin Site
	typename Symmetry::qType Q = join(qarray<1>{static_cast<int>(D)},OtherSym::qvacuum());
	Eigen::Index inner_dim = 1;
	std::vector<std::string> ident;
	ident.push_back("spin");
	
	basis_1s_.push_back(Q,inner_dim,ident);
	
	Id_1s_ = OperatorType(Symmetry::qvacuum(),basis_1s_,"id");
	S_1s_ = OperatorType(join(qarray<1>{3},OtherSym::qvacuum()),basis_1s_,"S");

	Scalar locS = 0.5*static_cast<double>(D-1);
	S_1s_( "spin", "spin" ) = std::sqrt(locS*(locS+1.));
	Sdag_1s_ = S_1s_.adjoint();
	Q_1s_ = OperatorType::prod(S_1s_,S_1s_,join(qarray<1>{5},OtherSym::qvacuum()));
	Id_1s_.setIdentity();
}

#endif //SPINSITESU2xX_H_
