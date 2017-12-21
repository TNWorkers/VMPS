#ifndef BASESU2XU1_H_
#define BASESU2XU1_H_

#include "symmetry/SU2xU1.h"
#include "symmetry/qbasis.h"
#include "tensors/SiteOperatorQ.h"

namespace spins {
	
/** \class BaseSU2xU1
  * \ingroup Spins
  *
  * This class provides the local operators for spins (magnitude \p D) in a SU(2) block representation for \p N_Orbitals sites.
  *
  * \describe_Scalar
  *
  */
template<typename Scalar=double>
class BaseSU2xU1
{
	typedef Eigen::Index Index;
	typedef typename Sym::SU2xU1<Scalar> Symmetry;
	typedef SiteOperatorQ<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > Operator;
	typedef typename Symmetry::qType qType;
	
public:
	BaseSU2xU1(){};
	
	/**
	\param L_input : amount of sites
	\param D_input : \f$D=2S+1\f$*/
	BaseSU2xU1 (std::size_t L_input, std::size_t D_input);

	/**amount of states = \f$D^L\f$*/
	inline std::size_t dim() const {return N_states;}
	
	/**\f$D=2S+1\f$*/
	inline std::size_t get_D() const {return D;}
	
	/**amount of orbitals*/
	inline std::size_t orbitals() const  {return N_orbitals;}

	Operator S( std::size_t orbital=0 ) const;
	Operator Sdag( std::size_t orbital=0 ) const;
	Operator Id() const;
	
	Operator HeisenbergHamiltonian( double J, bool PERIODIC=false ) const;

	/**Returns the basis. 
	   \note Use this as input for Mps, Mpo classes.*/ 
	std::vector<typename Symmetry::qType> qloc() { return TensorBasis.qloc(); }

	/**Returns the degeneracy vector of the basis. 
	   \note Use this as input for Mps, Mpo classes.*/ 
	std::vector<Eigen::Index> qlocDeg() { return TensorBasis.qlocDeg(); }

	Qbasis<Symmetry> get_basis() const { return TensorBasis; }
	
private:

	Qbasis<Symmetry> basis_1s; //basis for one site
	Qbasis<Symmetry> TensorBasis; //Final basis for N_orbital sites

	//operators defined on one orbital
	Operator Id_1s; //identity
	Operator S_1s; //spin
	Operator Sdag_1s; 
	Operator Q_1s; //Quadrupled operator (prod(S,S,{5}))

	std::size_t N_orbitals;
	std::size_t N_states;
	std::size_t D;
};

template<typename Scalar>
BaseSU2xU1<Scalar>::
BaseSU2xU1 (std::size_t L_input, std::size_t D_input)
:N_orbitals(L_input), D(D_input)
{
	assert(N_orbitals>=1 and D>=1);
	N_states = std::pow(D,N_orbitals);

	//create basis for one spin with quantumnumber D
	qType Q = {static_cast<int>(D),0};
	Scalar locS = 0.5*static_cast<Scalar>(Q[0]-1);
	Index inner_dim = 1;
	std::vector<std::string> ident; ident.push_back("spin");
	basis_1s.push_back(Q,inner_dim,ident);
	Id_1s = Operator({1,0},basis_1s);
	S_1s = Operator({3,0},basis_1s);

	//create operators for one orbital
	Id_1s( "spin", "spin" ) = 1.;
	S_1s( "spin", "spin" ) = std::sqrt(locS*(locS+1.));
	Sdag_1s = S_1s.adjoint();
	Q_1s = Operator::prod(S_1s,S_1s,{5,0});

	//create basis for N_orbitals spin sites
	if (N_orbitals == 1) { TensorBasis = basis_1s; }
	else
	{
		TensorBasis = basis_1s.combine(basis_1s);
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			TensorBasis = TensorBasis.combine(basis_1s);
		}
	}
}

template<typename Scalar>
SiteOperatorQ<Sym::SU2xU1<Scalar>,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > BaseSU2xU1<Scalar>::
S( std::size_t orbital ) const
{
	if(N_orbitals == 1) { return S_1s; }
	else
	{
		Operator out;
		bool TOGGLE=false;
		if(orbital == 0) { out = Operator::outerprod(S_1s,Id_1s,{3,0}); TOGGLE=true; }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,S_1s,{3,0}); TOGGLE=true; }
			else { out = Operator::outerprod(Id_1s,Id_1s,{1,0}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,S_1s,{3,0}); TOGGLE=true; }
			else if(TOGGLE==false) { out = Operator::outerprod(out,Id_1s,{1,0}); }
			else if(TOGGLE==true) { out = Operator::outerprod(out,Id_1s,{3,0}); }
		}
		return out;
	}
}

template<typename Scalar>
SiteOperatorQ<Sym::SU2xU1<Scalar>,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > BaseSU2xU1<Scalar>::
Sdag( std::size_t orbital ) const
{
	return S(orbital).adjoint();
}

template<typename Scalar>
SiteOperatorQ<Sym::SU2xU1<Scalar>,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > BaseSU2xU1<Scalar>::
Id() const
{
	if(N_orbitals == 1) { return Id_1s; }
	else
	{
		Operator out = Operator::outerprod(Id_1s,Id_1s,{1,0});
		for(std::size_t o=2; o<N_orbitals; o++) { out = Operator::outerprod(out,Id_1s,{1,0}); }
		return out;
	}
}

template<typename Scalar>
SiteOperatorQ<Sym::SU2xU1<Scalar>,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > BaseSU2xU1<Scalar>::
HeisenbergHamiltonian (double J, bool PERIODIC) const
{	
	Operator Mout({1,0},TensorBasis);

	if( N_orbitals >= 2 and J!=0. )
	{
		Mout = -std::sqrt(3)*J * Operator::prod(Sdag(0),S(1),{1,0});
	}

	for (int i=1; i<N_orbitals-1; ++i) // for all bonds
	{
		if (J != 0.)
		{
			Mout += -std::sqrt(3)*J * Operator::prod(Sdag(i),S(i+1),{1,0});
		}
	}
	if (PERIODIC == true and N_orbitals>2)
	{
		if (J != 0.)
		{
			Mout += -std::sqrt(3)*J * Operator::prod(Sdag(N_orbitals-1),S(0),{1,0});
		}
	}	
	return Mout;
}

} //end namespace spins
#endif
