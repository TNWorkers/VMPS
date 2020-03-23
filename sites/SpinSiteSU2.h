#ifndef SPINSITESU2_H_
#define SPINSITESU2_H_

#include "symmetry/kind_dummies.h"

#include "DmrgTypedefs.h"

#include "symmetry/S1xS2.h"
#include "symmetry/SU2.h"
#include "symmetry/U1.h"

#include "tensors/SiteOperatorQ.h"

template <typename Symmetry> class SpinSite;

template <>
class SpinSite<Sym::SU2<Sym::SpinSU2> >
{
	typedef double Scalar;
	typedef Sym::SU2<Sym::SpinSU2> Symmetry;
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

SpinSite<Sym::SU2<Sym::SpinSU2> >::
SpinSite(std::size_t D_input)
:D(D_input)
{
	//create basis for one Spin Site
	typename Symmetry::qType Q = {static_cast<int>(D)};
	Eigen::Index inner_dim = 1;
	std::vector<std::string> ident;
	ident.push_back("spin");
	
	basis_1s_.push_back(Q,inner_dim,ident);
	
	Id_1s_ = OperatorType({1},basis_1s_,"id");
	S_1s_ = OperatorType({3},basis_1s_,"S");

	Scalar locS = 0.5*static_cast<double>(D-1);
	S_1s_( "spin", "spin" ) = std::sqrt(locS*(locS+1.));
	Sdag_1s_ = S_1s_.adjoint();
	Q_1s_ = std::sqrt(2.)*OperatorType::prod(S_1s_,S_1s_,{5});
	Id_1s_.setIdentity();
}

#endif //SPINSITESU2_H_
