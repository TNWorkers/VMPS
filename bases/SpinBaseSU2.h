#ifndef SPINBASESU2_H_
#define SPINBASESU2_H_

#include "symmetry/SU2.h"
#include "symmetry/qbasis.h"
#include "tensors/SiteOperatorQ.h"

#include "symmetry/kind_dummies.h"
#include "bases/SpinBase.h"

//Note: Don't put a name in this documentation with \class .. because doxygen gets confused with template symbols
/** 
 * \ingroup Bases
 *
 * This class provides the local operators for spins (magnitude \p D) in a SU(2) block representation for \p N_Orbitals sites.
 *
 */
template<>
class SpinBase<Sym::SU2<Sym::SpinSU2> >
{
	typedef Eigen::Index Index;
	typedef double Scalar;
	typedef typename Sym::SU2<Sym::SpinSU2> Symmetry;
	typedef SiteOperatorQ<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > Operator;
	typedef typename Symmetry::qType qType;
	
public:
	SpinBase(){};
	
	/**
	 * \param L_input : amount of sites
	 * \param D_input : \f$D=2S+1\f$
	 */
	SpinBase (std::size_t L_input, std::size_t D_input = 2);

	/**Amount of states.*/
	inline std::size_t dim() const {return N_states;}
	
	/**\f$D=2S+1\f$*/
	inline std::size_t get_D() const {return D;}
	
	/**amount of orbitals*/
	inline std::size_t orbitals() const  {return N_orbitals;}

	///\{
	/**
	 * Quantum spin operator at given orbital.
	 * \param orbital : orbital index
	 */
	Operator S( std::size_t orbital=0 ) const;
	/**
	 * Hermitian conjugate of quantum spin operator at given orbital.
	 * For calculating scalar product \f$\mathbf{S}\cdot\mathbf{S}\f$. 
	 * \param orbital : orbital index
	 */
	Operator Sdag( std::size_t orbital=0 ) const;
	///\}

	/**Identity operator.*/
	Operator Id() const;
	
	/**
	 * Creates the full Heisenberg Hamiltonian on the supersite.
	 * \param J : \f$J\f$
	 * \param PERIODIC: periodic boundary conditions if \p true
	 */
	Operator HeisenbergHamiltonian( double J, bool PERIODIC=false ) const;

	/**
	 * Creates the full Heisenberg Hamiltonian on the supersite.
	 * \param J : \f$J_{ij}\f$
	 */
	Operator HeisenbergHamiltonian( Eigen::MatrixXd J ) const;

	/**Returns the basis.*/
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

SpinBase<Sym::SU2<Sym::SpinSU2> >::
SpinBase (std::size_t L_input, std::size_t D_input)
:N_orbitals(L_input), D(D_input)
{
	assert(N_orbitals>=1 and D>=2);
	N_states = std::pow(D,N_orbitals);

	//create basis for one spin with quantumnumber D
	qType Q = {static_cast<int>(D)};
	Scalar locS = 0.5*static_cast<Scalar>(Q[0]-1);
	Index inner_dim = 1;
	std::vector<std::string> ident; ident.push_back("spin");
	basis_1s.push_back(Q,inner_dim,ident);
	Id_1s = Operator({1},basis_1s);
	S_1s = Operator({3},basis_1s);

	//create operators for one orbital
	Id_1s( "spin", "spin" ) = 1.;
	S_1s( "spin", "spin" ) = std::sqrt(locS*(locS+1.));
	Sdag_1s = S_1s.adjoint();
	Q_1s = Operator::prod(S_1s,S_1s,{5});

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

SiteOperatorQ<Sym::SU2<Sym::SpinSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > SpinBase<Sym::SU2<Sym::SpinSU2> >::
S( std::size_t orbital ) const
{
	if(N_orbitals == 1) { return S_1s; }
	else
	{
		Operator out;
		bool TOGGLE=false;
		if(orbital == 0) { out = Operator::outerprod(S_1s,Id_1s,{3}); TOGGLE=true; }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,S_1s,{3}); TOGGLE=true; }
			else { out = Operator::outerprod(Id_1s,Id_1s,{1}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,S_1s,{3}); TOGGLE=true; }
			else if(TOGGLE==false) { out = Operator::outerprod(out,Id_1s,{1}); }
			else if(TOGGLE==true) { out = Operator::outerprod(out,Id_1s,{3}); }
		}
		return out;
	}
}

SiteOperatorQ<Sym::SU2<Sym::SpinSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > SpinBase<Sym::SU2<Sym::SpinSU2> >::
Sdag( std::size_t orbital ) const
{
	return S(orbital).adjoint();
}

SiteOperatorQ<Sym::SU2<Sym::SpinSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > SpinBase<Sym::SU2<Sym::SpinSU2> >::
Id() const
{
	if(N_orbitals == 1) { return Id_1s; }
	else
	{
		Operator out = Operator::outerprod(Id_1s,Id_1s,{1});
		for(std::size_t o=2; o<N_orbitals; o++) { out = Operator::outerprod(out,Id_1s,{1}); }
		return out;
	}
}

SiteOperatorQ<Sym::SU2<Sym::SpinSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > SpinBase<Sym::SU2<Sym::SpinSU2> >::
HeisenbergHamiltonian (double J, bool PERIODIC) const
{	
	Operator Mout({1},TensorBasis);

	if( N_orbitals >= 2 and J!=0. )
	{
		Mout = -std::sqrt(3)*J * Operator::prod(Sdag(0),S(1),{1});
	}

	for (int i=1; i<N_orbitals-1; ++i) // for all bonds
	{
		if (J != 0.)
		{
			Mout += -std::sqrt(3)*J * Operator::prod(Sdag(i),S(i+1),{1});
		}
	}
	if (PERIODIC == true and N_orbitals>2)
	{
		if (J != 0.)
		{
			Mout += -std::sqrt(3)*J * Operator::prod(Sdag(N_orbitals-1),S(0),{1});
		}
	}	
	return Mout;
}

SiteOperatorQ<Sym::SU2<Sym::SpinSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > SpinBase<Sym::SU2<Sym::SpinSU2> >::
HeisenbergHamiltonian (Eigen::MatrixXd J) const
{	
	Operator Mout({1},TensorBasis);

	for (int i=0; i<N_orbitals; ++i) // for all bonds
	for (int j=i+1; j<N_orbitals; ++j) // for all bonds
	{
		if (J(i,j) != 0.)
		{
			Mout += -std::sqrt(3)*J(i,j) * Operator::prod(Sdag(i),S(j),{1});
		}
	}
	return Mout;
}

#endif
