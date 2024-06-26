#ifndef FERMIONBASESU2XSU2_H_
#define FERMIONBASESU2XSU2_H_

#include "symmetry/S1xS2.h"
#include "symmetry/SU2.h"
#include "bases/FermionBase.h"
#include "tensors/SiteOperatorQ.h"
//include "tensors/Qbasis.h"




//Note: Don't put a name in this documentation with \class .. because doxygen gets confused with template symbols
/** 
 * \ingroup Bases
 *
 * This class provides the local operators for fermions in a SU(2)⊗SU(2)~SO(4) block representation.
 *
 * \note The SU(2)-charge symmetry is only present for a bipartite lattice.
 *       Using this class in DMRG requires that the SUB_LATTICE parameter is toggled every site.
 *
 */
template<>
class FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > >
{
	typedef Eigen::Index Index;
	typedef typename Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > Symmetry;
	typedef SiteOperatorQ<Symmetry,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > Operator;
	typedef typename Symmetry::qType qType;
public:
	
	FermionBase(){};
	
	/**
	 * \param L_input : the amount of orbitals
	 * \param subLattice_in : The SUB_LATTICE (Either A or B) of orbital 0. SUB_LATTICE of orbital i: \f$\sim (-1)^i \f$
	 */
	FermionBase(std::size_t L_input, SUB_LATTICE subLattice_in = SUB_LATTICE::A);
	
	/**amount of states = \f$4^L\f$*/
	inline Index dim() const {return static_cast<Index>(N_states);}
	
	/**amount of orbitals*/
	inline std::size_t orbitals() const  {return N_orbitals;}

	/**Returns the sublattice of orbital 0.*/
	inline SUB_LATTICE sublattice() const {return subLattice;}
	
	// \{
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

	/**Fermionic sign for one orbital of a supersite.
	   \param orbital : orbital index
	*/
	Operator sign_local (std::size_t orbital=0) const;

	///\{
	/**
	 * Holon occupation number operator
	 * \param orbital : orbital index
	 */
	Operator nh (std::size_t orbital=0) const;

	/**
	 * Spinon occupation number operator
	 * \param orbital : orbital index
	 */
	Operator ns (std::size_t orbital=0) const;
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

	///\{
	/**
	 * Orbital pseudo-spin
	 * \param orbital : orbital index
	 */
	Operator T (std::size_t orbital=0) const;
	
	/**
	 * Orbital pseudo-spin† 
	 * \param orbital orbital index
	 */
	Operator Tdag (std::size_t orbital=0) const;
	///\}
	
	/**Returns an array of size dim() with zeros.*/
	ArrayXd ZeroField() const { return ArrayXd::Zero(N_orbitals); }
	
	/**Returns an array of size dim()xdim() with zeros.*/
	ArrayXXd ZeroHopping() const { return ArrayXXd::Zero(N_orbitals,N_orbitals); }

	/**
	 * Creates the full Hubbard Hamiltonian on the supersite with orbital-dependent U and with arbitrary hopping matrix (bipartite).
	 * \param U \f$U\f$ for each orbital
	 * \param t \f$t_{ij}\f$ (hopping matrix)
	 * \warning The hopping matrix needs to be bipartite!
	 * \param V
	 * \param J
	 * \todo3 Add a check, that the hopping matrix is really bipartite.
	 */
	Operator HubbardHamiltonian (const ArrayXd &U, const ArrayXXd &t, const ArrayXXd &V, const ArrayXXd &J) const;
	
	/**Identity*/
	Operator Id (std::size_t orbital=0) const;
	
	/**Returns the local basis.*/
	Qbasis<Symmetry> get_basis() const {return TensorBasis;}
	
private:

	std::size_t N_orbitals;
	std::size_t N_states;

	SUB_LATTICE subLattice;
	
	Qbasis<Symmetry> basis_1s; //basis for one site
	Qbasis<Symmetry> TensorBasis; //Final basis for N_orbital sites

	//operators defined on one orbital
	Operator Id_1s; //identity
	Operator F_1s; //Fermionic sign
	Operator c_1sA; //annihilation sublattice A
	Operator cdag_1sA; //creation sublattice A
	Operator c_1sB; //annihilation sublattice B
	Operator cdag_1sB; //creation sublattice B
	Operator nh_1s; //holon particle number
	Operator ns_1s; //spinon particle number (ns_1s+nh_1s=Id_1s)
	Operator S_1s; //orbital spin
	Operator T_1s; //orbital pseudo spin
};

FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > >::
FermionBase (std::size_t L_input, SUB_LATTICE subLattice_in)
:N_orbitals(L_input),subLattice(subLattice_in)
{
	assert(N_orbitals>=1);
	
	std::size_t locdim = 2;
	
	//create basis for one Fermionic Site
	typename Symmetry::qType Q={1,2}; //holon state
	Eigen::Index inner_dim = 1;
	std::vector<std::string> ident;
	ident.push_back("holon");
	basis_1s.push_back(Q,inner_dim,ident);
	ident.clear();
	Q={2,1}; //spinon state
	inner_dim = 1;
	ident.push_back("spinon");
	basis_1s.push_back(Q,inner_dim,ident);
	ident.clear();
	
	Id_1s = Operator({1,1},basis_1s);
	F_1s = Operator({1,1},basis_1s);
	c_1sA = Operator({2,2},basis_1s);
	c_1sB = Operator({2,2},basis_1s);
	nh_1s = Operator({1,1},basis_1s);
	ns_1s = Operator({1,1},basis_1s);
	S_1s = Operator({3,1},basis_1s);
	T_1s = Operator({1,3},basis_1s);
	
	//create operators for one orbital
	Id_1s( "holon", "holon" ) = 1.;
	Id_1s( "spinon", "spinon" ) = 1.;
	
	F_1s( "holon", "holon" ) = 1.;
	F_1s( "spinon", "spinon" ) = -1.;
	
	c_1sA( "spinon", "holon" ) = std::sqrt(2.);
	c_1sA( "holon", "spinon" ) = std::sqrt(2.);
	cdag_1sA = c_1sA.adjoint();
	c_1sB( "spinon", "holon" ) = std::sqrt(2.);
	c_1sB( "holon", "spinon" ) = -std::sqrt(2.);
	cdag_1sB = c_1sB.adjoint();
	
	nh_1s( "holon", "holon" ) = 1.;
	ns_1s( "spinon", "spinon" ) = 1.;
	S_1s( "spinon", "spinon" ) = std::sqrt(0.75);
	T_1s( "holon", "holon" )   = std::sqrt(0.75);
	
	//create basis for N_orbitals fermionic sites
	if (N_orbitals == 1) {TensorBasis = basis_1s;}
	else
	{
		TensorBasis = basis_1s.combine(basis_1s);
		for (std::size_t o=2; o<N_orbitals; o++)
		{
			TensorBasis = TensorBasis.combine(basis_1s);
		}
	}
	N_states = TensorBasis.size();
}

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > >::
c (std::size_t orbital) const
{
	Operator Op_1s;
	if(subLattice == SUB_LATTICE::A)
	{
		if(orbital%2 == 0) {Op_1s = c_1sA;}
		else {{Op_1s = c_1sB;}}
	}
	else if(subLattice == SUB_LATTICE::B)
	{
		if(orbital%2 == 0) {Op_1s = c_1sB;}
		else {{Op_1s = c_1sA;}}
	}
	else {assert(1!=1 and "Crazy...");}

	if(N_orbitals == 1) { return Op_1s; }
	else
	{
		Operator out;
		bool TOGGLE=false;
		if(orbital == 0) { out = Operator::outerprod(Op_1s,Id_1s,{2,2}); TOGGLE=true; }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(F_1s,Op_1s,{2,2}); TOGGLE=true; }
			else { out = Operator::outerprod(F_1s,F_1s,{1,1}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,Op_1s,{2,2}); TOGGLE=true; }
			else if(TOGGLE==false) { out = Operator::outerprod(out,F_1s,{1,1}); }
			else if(TOGGLE==true) { out = Operator::outerprod(out,Id_1s,{2,2}); }
		}
		return out;
	}
}

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > >::
cdag (std::size_t orbital) const
{
	return c(orbital).adjoint();
}

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > >::
sign (std::size_t orb1, std::size_t orb2) const
{
	if(N_orbitals == 1) { return F_1s; }
	else
	{
		Operator out = Id();
		for (int i=orb1; i<N_orbitals; ++i)
		{
			out = Operator::prod(out, 2.*nh(i)-Id(i),{1,1});
		}
		for (int i=0; i<orb2; ++i)
		{
			out = Operator::prod(out, 2.*nh(i),{1,1});
		}

		return out;
	}
}

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > >::
sign_local (std::size_t orbital) const
{
	if(N_orbitals == 1) { return Id_1s; }
	else
	{
		Operator out;
		if(orbital == 0) { out = Operator::outerprod(F_1s,Id_1s,{1,1}); }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,F_1s,{1,1}); }
			else { out = Operator::outerprod(Id_1s,Id_1s,{1,1}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,F_1s,{1,1}); }
			else { out = Operator::outerprod(out,Id_1s,{1,1}); }
		}
		return out;
	}
}

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > >::
nh (std::size_t orbital) const
{
	if(N_orbitals == 1) { return nh_1s; }
	else
	{
		Operator out;
		if(orbital == 0) { out = Operator::outerprod(nh_1s,Id_1s,{1,1}); }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,nh_1s,{1,1}); }
			else { out = Operator::outerprod(Id_1s,Id_1s,{1,1}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,nh_1s,{1,1}); }
			else { out = Operator::outerprod(out,Id_1s,{1,1}); }
		}
		return out;
	}
}

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > >::
ns (std::size_t orbital) const
{
	if(N_orbitals == 1) { return ns_1s; }
	else
	{
		Operator out;
		if(orbital == 0) { out = Operator::outerprod(ns_1s,Id_1s,{1,1}); }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,ns_1s,{1,1}); }
			else { out = Operator::outerprod(Id_1s,Id_1s,{1,1}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,ns_1s,{1,1}); }
			else { out = Operator::outerprod(out,Id_1s,{1,1}); }
		}
		return out;
	}
}

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > >::
S (std::size_t orbital) const
{
	if(N_orbitals == 1) { return S_1s; }
	else
	{
		Operator out;
		bool TOGGLE=false;
		if(orbital == 0) { out = Operator::outerprod(S_1s,Id_1s,{3,1}); TOGGLE=true; }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,S_1s,{3,1}); TOGGLE=true; }
			else { out = Operator::outerprod(Id_1s,Id_1s,{1,1}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,S_1s,{3,1}); TOGGLE=true; }
			else if(TOGGLE==false) { out = Operator::outerprod(out,Id_1s,{1,1}); }
			else if(TOGGLE==true) { out = Operator::outerprod(out,Id_1s,{3,1}); }
		}
		return out;
	}
}

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > >::
Sdag (std::size_t orbital) const
{
	return S(orbital).adjoint();
}

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > >::
T (std::size_t orbital) const
{
	if(N_orbitals == 1) { return T_1s; }
	else
	{
		Operator out;
		bool TOGGLE=false;
		if(orbital == 0) { out = Operator::outerprod(T_1s,Id_1s,{1,3}); TOGGLE=true; }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,T_1s,{1,3}); TOGGLE=true; }
			else { out = Operator::outerprod(Id_1s,Id_1s,{1,1}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,T_1s,{1,3}); TOGGLE=true; }
			else if(TOGGLE==false) { out = Operator::outerprod(out,Id_1s,{1,1}); }
			else if(TOGGLE==true) { out = Operator::outerprod(out,Id_1s,{1,3}); }
		}
		return out;
	}
}

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > >::
Tdag (std::size_t orbital) const
{
	return T(orbital).adjoint();
}

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > >::
Id (std::size_t orbital) const
{
	if(N_orbitals == 1) { return Id_1s; }
	else
	{
		Operator out = Operator::outerprod(Id_1s,Id_1s,{1,1});
		for(std::size_t o=2; o<N_orbitals; o++) { out = Operator::outerprod(out,Id_1s,{1,1}); }
		return out;
	}
}

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > >::
HubbardHamiltonian (const ArrayXd &U, const ArrayXXd &t, const ArrayXXd &V, const ArrayXXd &J) const
{
	Operator Mout({1,1},TensorBasis);
	
	for (int i=0; i<N_orbitals; ++i)
	for (int j=0; j<i; ++j)
	{
		if (t(i,j) != 0.)
		{
			Mout += -t(i,j) * std::sqrt(2.)*std::sqrt(2.) * Operator::prod(cdag(i),c(j),{1,1});
		}
		if (V(i,j) != 0.)
		{
			Mout += V(i,j) * std::sqrt(3.) * (Operator::prod(Tdag(i),T(j),{1,1}));
		}
		if (J(i,j) != 0.)
		{
			Mout += J(i,j) * std::sqrt(3.) * (Operator::prod(Sdag(i),S(j),{1,1}));
		}
	}
	
	for (int i=0; i<N_orbitals; ++i)
	{
		if (U(i) != 0. and U(i) != std::numeric_limits<double>::infinity())
		{
			Mout += 0.5 * U(i) * nh(i);
		}
	}
	
	return Mout;
}
#endif
