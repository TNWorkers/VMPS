#ifndef FERMIONBASESU2XU0_H_
#define FERMIONBASESU2XU0_H_

#include <algorithm>
#include <iterator>

#include "symmetry/SU2.h"
#include "tensors/SiteOperator.h"
#include "tensors/Qbasis.h"

#include "bases/FermionBase.h"

//Note: Don't put a name in this documentation with \class .. because doxygen gets confused with template symbols
/**
 * \ingroup Bases
 *
 * This class provides the local operators for fermions in a spin-SU(2) block representation for \p N_Orbitals fermionic sites.
 *
 */
template<>
class FermionBase<Sym::SU2<Sym::SpinSU2> >
{
	typedef Eigen::Index Index;
	typedef double Scalar;
	typedef typename ym::SU2<Sym::SpinSU2> Symmetry;
	typedef SiteOperator<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > Operator;
	typedef typename Symmetry::qType qType;
public:
	
	FermionBase(){};
	
	/**
	 * \param L_input : the amount of orbitals
	 * \param U_IS_INFINITE : if \p true, eliminates doubly-occupied sites from the basis
	 */
	FermionBase (std::size_t L_input, bool U_IS_INFINITE=false);
	
	/**amount of states.*/
	inline Index dim() const {return static_cast<Index>(N_states);}
	
	/**amount of orbitals*/
	inline std::size_t orbitals() const  {return N_orbitals;}
	
	///\{
	/**
	 * Annihilation operator
	 * \param orbital : orbital index
	 */
	Operator c (std::size_t orbital=0) const;
	
	/**
	 * Creation operator.
	 * \param orbital : orbital index
	 */
	Operator cdag (std::size_t orbital=0) const;

	/**
	 * Fermionic sign for the hopping between two orbitals of nearest-neighbour supersites of a ladder.
	 * \param orb1 : orbital on supersite i
	 * \param orb2 : orbital on supersite i+1
	 */
	Operator sign (std::size_t orb1=0, std::size_t orb2=0) const;

	/**
	 * Fermionic sign for one orbital of a supersite.
	 * \param orbital : orbital index
	 */
	Operator sign_local (std::size_t orbital=0) const;

	/**
	 * Occupation number operator
	 * \param orbital : orbital index
	 */
	Operator n (std::size_t orbital=0) const;
		
	/**
	 * Double occupation
	 * \param orbital : orbital index
	 */
	Operator d (std::size_t orbital=0) const;
	///\}
	
	///\{
	/**
	 * Orbital spin
	 * \param orbital : orbital index
	 */
	Operator S (std::size_t orbital=0) const;
	
	/**
	 * Orbital spin† 
	 * \param orbital : orbital index
	 */
	Operator Sdag (std::size_t orbital=0) const;
	///\}
	
	/**Returns an array of size dim() with zeros.*/
	ArrayXd ZeroField() const { return ArrayXd::Zero(N_orbitals); }
	
	/**Returns an array of size dim()xdim() with zeros.*/
	ArrayXXd ZeroHopping() const { return ArrayXXd::Zero(N_orbitals,N_orbitals); }
	
	/**
	 * Creates the full Hubbard Hamiltonian on the supersite with orbital-dependent U and arbitrary hopping matrix.
	 * \param U : \f$U\f$ for each orbital
	 * \param Eorb : energy of the orbital
	 * \param t : \f$t_{ij}\f$ (hopping matrix)
	 * \param V : \f$V_{ij}\f$ (nn Density Interaction matrix)
	 * \param J : \f$J_{ij}\f$ (nn Spin interaction matrix)
	 */
	Operator HubbardHamiltonian (const ArrayXd &U, const ArrayXd &Eorb, const ArrayXXd &t, const ArrayXXd &V, const ArrayXXd &J) const;
	
	/**Identity*/
	Operator Id() const;

	/**
	 * Returns the basis. 
	 * \note Use this as input for Mps, Mpo classes.
	 */ 
	std::vector<typename Symmetry::qType> qloc() const { return TensorBasis.qloc(); }

	/**
	 * Returns the degeneracy vector of the basis. 
	 * \note Use this as input for Mps, Mpo classes.
	 */ 
	std::vector<Eigen::Index> qlocDeg() const { return TensorBasis.qlocDeg(); }

	Qbasis<Symmetry> get_basis() const { return TensorBasis; }
private:
	
	std::size_t N_orbitals;
	std::size_t N_states;
	
	Qbasis<Symmetry> basis_1s; //basis for one site
	Qbasis<Symmetry> TensorBasis; //Final basis for N_orbital sites

	//operators defined on one orbital
	Operator Id_1s; //identity
	Operator F_1s; //Fermionic sign
	Operator c_1s; //annihilation
	Operator cdag_1s; //creation
	Operator n_1s; //particle number
	Operator d_1s; //double occupancy
	Operator S_1s; //orbital spin
	Operator p_1s; //pairing
	Operator pdag_1s; //pairing adjoint
};

FermionBase<Sym::SU2<Sym::SpinSU2> >::
FermionBase (std::size_t L_input, bool U_IS_INFINITE)
:N_orbitals(L_input)
{
	assert(N_orbitals>=1);
	
	std::size_t locdim = (U_IS_INFINITE)? 2 : 3;
	N_states = std::pow(locdim,N_orbitals);

	//create basis for one Fermionic Site
	typename Symmetry::qType Q={1}; //singlet states
	Eigen::Index inner_dim;
	(U_IS_INFINITE)? inner_dim = 1 : inner_dim = 2;
	std::vector<std::string> ident;
	if (!U_IS_INFINITE) {ident.push_back("double");}
	ident.push_back("empty");
	basis_1s.push_back(Q,inner_dim,ident);
	ident.clear();	
	Q={2}; //doublet states
	inner_dim = 1;
	ident.push_back("single");
	basis_1s.push_back(Q,inner_dim,ident);
	ident.clear();

	Id_1s = Operator({1},basis_1s);
	F_1s = Operator({1},basis_1s);
	c_1s = Operator({2},basis_1s);
	d_1s = Operator({1},basis_1s);
	S_1s = Operator({3},basis_1s);

	//create operators for one orbital
	Id_1s( "empty", "empty" ) = 1.;
	Id_1s( "double", "double" ) = 1.;
	Id_1s( "single", "single" ) = 1.;

	F_1s( "empty", "empty" ) = 1.;
	F_1s( "double", "double" ) = 1.;
	F_1s( "single", "single" ) = -1.;

	c_1s( "empty", "single" ) = std::pow(2.,0.5);
	c_1s( "single", "double" ) = 1.;
	
	cdag_1s = c_1s.adjoint();
	
	n_1s = std::sqrt(2.) * Operator::prod(cdag_1s,c_1s,{1});
	
	d_1s( "double", "double" ) = 1.;
	
	S_1s( "single", "single" ) = std::pow(0.75,0.5);
	
	p_1s = -std::sqrt(0.5) * Operator::prod(c_1s,c_1s,{1}); //The sign convention corresponds to c_DN c_UP
	pdag_1s = p_1s.adjoint(); //The sign convention corresponds to (c_DN c_UP)†=c_UP† c_DN†

	//create basis for N_orbitals fermionic sites
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

SiteOperator<Sym::SU2<Sym::SpinSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::SpinSU2> >::
c (std::size_t orbital) const
{
	if(N_orbitals == 1) { return c_1s; }
	else
	{
		Operator out;
		bool TOGGLE=false;
		if(orbital == 0) { out = Operator::outerprod(c_1s,Id_1s,{2}); TOGGLE=true; }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(F_1s,c_1s,{2}); TOGGLE=true; }
			else { out = Operator::outerprod(F_1s,F_1s,{1}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,c_1s,{2}); TOGGLE=true;}
			else if(TOGGLE==false) { out = Operator::outerprod(out,F_1s,{1}); }
			else if(TOGGLE==true) { out = Operator::outerprod(out,Id_1s,{2}); }
		}
		return out;
	}
}

SiteOperator<Sym::SU2<Sym::SpinSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::SpinSU2> >::
cdag (std::size_t orbital) const
{
	return c(orbital).adjoint();
}

SiteOperator<Sym::SU2<Sym::SpinSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::SpinSU2> >::
sign (std::size_t orb1, std::size_t orb2) const
{
	if(N_orbitals == 1) { return F_1s; }
	else
	{
		Operator out = Id();
		for (int i=orb1; i<N_orbitals; ++i)
		{
			// out = Operator::prod(out,sign_local(i),{1}); // * (Id-2.*n(UP,i))*(Id-2.*n(DN,i));
			out = Operator::prod(out, (Id()-2.*n(i)+4.*d(i)),{1});
		}
		for (int i=0; i<orb2; ++i)
		{
			// out = Operator::prod(out,sign_local(i),{1}); // * (Id-2.*n(UP,i))*(Id-2.*n(DN,i));
			out = Operator::prod(out, (Id()-2.*n(i)+4.*d(i)),{1});
		}

		return out;
	}
}

SiteOperator<Sym::SU2<Sym::SpinSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::SpinSU2> >::
sign_local (std::size_t orbital) const
{
	if(N_orbitals == 1) { return F_1s; }
	else
	{
		Operator out;
		if(orbital == 0) { out = Operator::outerprod(F_1s,Id_1s,{1}); }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,F_1s,{1}); }
			else { out = Operator::outerprod(Id_1s,Id_1s,{1}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,F_1s,{1}); }
			else { out = Operator::outerprod(out,Id_1s,{1}); }
		}
		return out;
	}
}

SiteOperator<Sym::SU2<Sym::SpinSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::SpinSU2> >::
n (std::size_t orbital) const
{
	if(N_orbitals == 1) { return n_1s; }
	else
	{
		Operator out;
		if(orbital == 0) { out = Operator::outerprod(n_1s,Id_1s,{1}); }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,n_1s,{1}); }
			else { out = Operator::outerprod(Id_1s,Id_1s,{1}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,n_1s,{1}); }
			else { out = Operator::outerprod(out,Id_1s,{1}); }
		}
		return out;
	}
}

SiteOperator<Sym::SU2<Sym::SpinSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::SpinSU2> >::
d (std::size_t orbital) const
{
	if(N_orbitals == 1) { return d_1s; }
	else
	{
		Operator out;
		if(orbital == 0) { out = Operator::outerprod(d_1s,Id_1s,{1}); }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,d_1s,{1}); }
			else { out = Operator::outerprod(Id_1s,Id_1s,{1}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,d_1s,{1}); }
			else { out = Operator::outerprod(out,Id_1s,{1}); }
		}
		return out;
	}
}

SiteOperator<Sym::SU2<Sym::SpinSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::SpinSU2> >::
S (std::size_t orbital) const
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

SiteOperator<Sym::SU2<Sym::SpinSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::SpinSU2> >::
Sdag (std::size_t orbital) const
{
	return S(orbital).adjoint();
}

SiteOperator<Sym::SU2<Sym::SpinSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::SpinSU2> >::
Id () const
{
	if(N_orbitals == 1) { return Id_1s; }
	else
	{
		Operator out = Operator::outerprod(Id_1s,Id_1s,{1});
		for(std::size_t o=2; o<N_orbitals; o++) { out = Operator::outerprod(out,Id_1s,{1}); }
		return out;
	}
}

SiteOperator<Sym::SU2<Sym::SpinSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::SpinSU2> >::
HubbardHamiltonian (const ArrayXd &U, const ArrayXd &Eorb, const ArrayXXd &t, const ArrayXXd &V, const ArrayXXd &J) const
{
	Operator Mout({1},TensorBasis);
	Mout.setZero();
	
	for (int i=0; i<N_orbitals; ++i)
	for (int j=0; j<i; ++j)
	{
		if (t(i,j) != 0.)
		{
			Mout += -t(i,j)*std::sqrt(2.) * (Operator::prod(cdag(i),c(j),{1,0}) + Operator::prod(c(i),cdag(j),{1,0}));
		}
		if (V(i,j) != 0.)
		{
			Mout += V(i,j) * (Operator::prod(n(i),n(j),{1,0}));
		}
		if (J(i,j) != 0.)
		{
			Mout += J(i,j)*std::sqrt(3.)*(Operator::prod(Sdag(i),S(j),{1,0}));
		}
	}
	
	for (int i=0; i<N_orbitals; ++i)
	{
		if (U(i) != 0. and U(i) != std::numeric_limits<double>::infinity())
		{
			Mout += U(i) * d(i);
		}
		if (Eorb(i) != 0.)
		{
			Mout += Eorb(i) * n(i);
		}
	}
	return Mout;
}

#endif
