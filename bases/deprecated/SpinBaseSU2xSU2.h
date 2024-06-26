#ifndef SPINBASESU2XSU2_H_
#define SPINBASESU2XSU2_H_

#include "symmetry/S1xS2.h"
#include "symmetry/SU2.h"
//include "tensors/Qbasis.h"
#include "tensors/SiteOperatorQ.h"

#include "bases/SpinBase.h"

//Note: Don't put a name in this documentation with \class .. because doxygen gets confused with template symbols
/** 
 * \ingroup Bases
 *
 * This class provides the local operators for spins (magnitude \p D) in a SU(2) block representation for \p N_Orbitals sites.
 *
 * \note : The second SU2 quantum number is a dummy, which is present for combining the Spin with SU(2)xSU(2) fermions. (Kondo Model)
 *
 */
template<>
class SpinBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > >
{
	typedef Eigen::Index Index;
	typedef double Scalar;
	typedef typename Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > Symmetry;
	typedef SiteOperatorQ<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > Operator;
	typedef typename Symmetry::qType qType;
	
public:
	
	SpinBase(){};
	
	/**
	 * \param L_input : amount of sites
	 * \param D_input : \f$D=2S+1\f$
	 * \param subLattice_in : sublattice, can be SUB_LATTICE::A or SUB_LATTICE::B
	 */
	SpinBase (std::size_t L_input, std::size_t D_input, SUB_LATTICE subLattice_in = SUB_LATTICE::A);

	/**amount of states*/
	inline std::size_t dim() const {return N_states;}
	
	/**\f$D=2S+1\f$*/
	inline std::size_t get_D() const {return D;}
	
	/**amount of orbitals*/
	inline std::size_t orbitals() const  {return N_orbitals;}
	
	/**Returns the sublattice of orbital 0.*/
	inline SUB_LATTICE sublattice() const {return subLattice;}
	
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
	
	/**Returns an array of size dim() with zeros.*/
	ArrayXd ZeroField() const { return ArrayXd::Zero(N_orbitals); }
	
	/**Returns an array of size dim()xdim() with zeros.*/
	ArrayXXd ZeroHopping() const { return ArrayXXd::Zero(N_orbitals,N_orbitals); }
	
	/**
 	* Creates the full Heisenberg Hamiltonian on the supersite.
 	* \param J : \f$J\f$
 	*/
	Operator HeisenbergHamiltonian (const ArrayXXd &J) const;
	
	/**Returns the basis.*/
	Qbasis<Symmetry> get_basis() const {return TensorBasis;}
	
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
	
	SUB_LATTICE subLattice;
};

SpinBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > >::
SpinBase (std::size_t L_input, std::size_t D_input, SUB_LATTICE subLattice_in)
:N_orbitals(L_input), D(D_input), subLattice(subLattice_in)
{
	assert(N_orbitals>=1 and D>=1);
	N_states = std::pow(D,N_orbitals);
	
	//create basis for one spin with quantumnumber D
	qType Q = {static_cast<int>(D),1};
	Scalar locS = 0.5 * static_cast<Scalar>(Q[0]-1);
	Index inner_dim = 1;
	std::vector<std::string> ident; ident.push_back("spin");
	basis_1s.push_back(Q,inner_dim,ident);
	Id_1s = Operator({1,1},basis_1s);
	S_1s  = Operator({3,1},basis_1s);
	
	//create operators for one orbital
	Id_1s("spin","spin") = 1.;
	S_1s ("spin","spin") = std::sqrt(locS*(locS+1.));
	Sdag_1s = S_1s.adjoint();
	Q_1s = Operator::prod(S_1s,S_1s,{5,1});
	
	//create basis for N_orbitals spin sites
	if (N_orbitals == 1) {TensorBasis = basis_1s;}
	else
	{
		TensorBasis = basis_1s.combine(basis_1s);
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			TensorBasis = TensorBasis.combine(basis_1s);
		}
	}
}

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > SpinBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > >::
S( std::size_t orbital ) const
{
	if (N_orbitals == 1) {return S_1s; }
	else
	{
		Operator out;
		bool TOGGLE=false;
		if (orbital == 0) { out = Operator::outerprod(S_1s,Id_1s,{3,1}); TOGGLE=true; }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,S_1s,{3,1}); TOGGLE=true; }
			else { out = Operator::outerprod(Id_1s,Id_1s,{1,1}); }
		}
		for (std::size_t o=2; o<N_orbitals; o++)
		{
			if (orbital == o) { out = Operator::outerprod(out,S_1s,{3,1}); TOGGLE=true; }
			else if(TOGGLE==false) { out = Operator::outerprod(out,Id_1s,{1,1}); }
			else if(TOGGLE==true) { out = Operator::outerprod(out,Id_1s,{3,1}); }
		}
		return out;
	}
}

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > SpinBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > >::
Sdag( std::size_t orbital ) const
{
	return S(orbital).adjoint();
}

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > SpinBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > >::
Id() const
{
	if(N_orbitals == 1) { return Id_1s; }
	else
	{
		Operator out = Operator::outerprod(Id_1s,Id_1s,{1,1});
		for(std::size_t o=2; o<N_orbitals; o++) { out = Operator::outerprod(out,Id_1s,{1,1}); }
		return out;
	}
}

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > SpinBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > >::
HeisenbergHamiltonian (const ArrayXXd &J) const
{
	Operator Mout({1,1},TensorBasis);
	
	for (int i=0; i<N_orbitals; ++i)
	for (int j=0; j<i; ++j)
	{
		if (J(i,j) != 0.)
		{
			Mout += J(i,j) * std::sqrt(3) * Operator::prod(Sdag(i),S(j),{1,1});
		}
	}
	
	return Mout;
}

#endif
