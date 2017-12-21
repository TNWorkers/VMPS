#ifndef FERMIONBASESU2XSU2_H_
#define FERMIONBASESU2XSU2_H_

#include <algorithm>
#include <iterator>

#include "tensors/SiteOperatorQ.h"
#include "symmetry/qbasis.h"
#include "symmetry/SU2xSU2.h"

#include "FermionBase.h"

enum SUB_LATTICE {A=0,B=1};

	
/** \class FermionBase
  * \ingroup Fermions
  *
  * This class provides the local operators for fermions in a SU(2)⊗U(1) block representation.
  *
  * \describe_Scalar
  *
  * \todo Implement the operators for more than one orbital.
  */
template<>
class FermionBase<Sym::SU2xSU2<double> >
{
	typedef Eigen::Index Index;
	typedef typename Sym::SU2xSU2<double> Symmetry;
	typedef SiteOperatorQ<Symmetry,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > Operator;
	typedef typename Symmetry::qType qType;
public:
	
	FermionBase(){};
	
	/**
	\param L_input : the amount of orbitals
	\param U_IS_INFINITE : if \p true, eliminates doubly-occupied sites from the basis*/
	FermionBase (std::size_t L_input, SUB_LATTICE subLattice_in = SUB_LATTICE::A, bool U_IS_INFINITE=false);
	
	/**amount of states = \f$4^L\f$*/
	inline Index dim() const {return static_cast<Index>(N_states);}
	
	/**amount of orbitals*/
	inline std::size_t orbitals() const  {return N_orbitals;}

	// \{
	/** Annihilation operator
		\param subLattice : Partion of the operator (Either A or B)
		\param orbital : orbital index*/
	Operator c (std::size_t orbital=0) const;
	
	/**Creation operator.
	   \param subLattice : Partion of the operator (Either A or B)
	   \param orbital : orbital index*/
	Operator cdag (std::size_t orbital=0) const;

	/**Fermionic sign for the hopping between two orbitals of nearest-neighbour supersites of a ladder.
	   \param orb1 : orbital on supersite i
	   \param orb2 : orbital on supersite i+1
	*/
	Operator sign (std::size_t orb1=0, std::size_t orb2=0) const;

	/**Fermionic sign for one orbital of a supersite.
	   \param orbital : orbital index
	*/
	Operator sign_local (std::size_t orbital=0) const;

	///\{
	/**Holon occupation number operator
	\param orbital : orbital index*/
	Operator nh (std::size_t orbital=0) const;

	/**Spinon occupation number operator
	\param orbital : orbital index*/
	Operator ns (std::size_t orbital=0) const;
	///\}

	///\{
	/**Orbital spin
	   \param orbital : orbital index*/
	Operator S (std::size_t orbital=0) const;
	
	/**Orbital spin† 
	   \param orbital : orbital index*/
	Operator Sdag (std::size_t orbital=0) const;
	///\}

	///\{
	/**Orbital pseudo-spin
	   \param orbital : orbital index*/
	Operator T (std::size_t orbital=0) const;
	
	/**Orbital pseudo-spin† 
	   \param orbital : orbital index*/
	Operator Tdag (std::size_t orbital=0) const;
	///\}

	/**Creates the full Hubbard Hamiltonian on the supersite.
	\param U : \f$U\f$
	\param t : \f$t\f$
	\param V : \f$V\f$
	\param J : \f$J\f$
	\param PERIODIC: periodic boundary conditions if \p true*/
	Operator HubbardHamiltonian (double U, double t=1., double V=0., double J=0., bool PERIODIC=false) const;
	Operator HubbardHamiltonian (double U, Eigen::ArrayXXd  t) const;
	
	/**Creates the full Hubbard Hamiltonian on the supersite with orbital-dependent U.
	\param Uvec : \f$U\f$ for each orbital
	\param onsite : \f$\varepsilon\f$ onsite energy for each orbital
	\param t : \f$t\f$
	\param V : \f$V\f$
	\param J : \f$J\f$
	\param PERIODIC: periodic boundary conditions if \p true*/
	Operator HubbardHamiltonian (Eigen::ArrayXd Uorb, double t=1., double V=0., double J=0., bool PERIODIC=false) const;

	/**Identity*/
	Operator Id (std::size_t orbital=0) const;

	Qbasis<Symmetry> get_basis() const { return TensorBasis; }

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

FermionBase<Sym::SU2xSU2<double> >::
FermionBase (std::size_t L_input, SUB_LATTICE subLattice_in, bool U_IS_INFINITE)
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
	if (N_orbitals == 1) { TensorBasis = basis_1s; }
	else
	{
		TensorBasis = basis_1s.combine(basis_1s);
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			TensorBasis = TensorBasis.combine(basis_1s);
		}
	}
	N_states = TensorBasis.size();
}

SiteOperatorQ<Sym::SU2xSU2<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xSU2<double> >::
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
	
SiteOperatorQ<Sym::SU2xSU2<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xSU2<double> >::
cdag (std::size_t orbital) const
{
	return c(orbital).adjoint();
}

SiteOperatorQ<Sym::SU2xSU2<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xSU2<double> >::
sign (std::size_t orb1, std::size_t orb2) const
{
	if(N_orbitals == 1) { return F_1s; }
	else
	{
		Operator out = Id();
		for (int i=orb1; i<N_orbitals; ++i)
		{
			// out = Operator::prod(out,sign_local(i),{1}); // * (Id-2.*n(UP,i))*(Id-2.*n(DN,i));
			out = Operator::prod(out, 2.*nh(i),{1,1});
		}
		for (int i=0; i<orb2; ++i)
		{
			// out = Operator::prod(out,sign_local(i),{1}); // * (Id-2.*n(UP,i))*(Id-2.*n(DN,i));
			out = Operator::prod(out, 2.*nh(i),{1,1});
		}

		return out;
	}
}

SiteOperatorQ<Sym::SU2xSU2<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xSU2<double> >::
sign_local (std::size_t orbital) const
{
	if(N_orbitals == 1) { return F_1s; }
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

SiteOperatorQ<Sym::SU2xSU2<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xSU2<double> >::
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

SiteOperatorQ<Sym::SU2xSU2<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xSU2<double> >::
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

SiteOperatorQ<Sym::SU2xSU2<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xSU2<double> >::
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

SiteOperatorQ<Sym::SU2xSU2<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xSU2<double> >::
Sdag (std::size_t orbital) const
{
	return S(orbital).adjoint();
}

SiteOperatorQ<Sym::SU2xSU2<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xSU2<double> >::
T (std::size_t orbital) const
{
	if(N_orbitals == 1) { return T_1s; }
	else
	{
		Operator out;
		bool TOGGLE=false;
		if(orbital == 0) { out = Operator::outerprod(T_1s,Id_1s,{3,1}); TOGGLE=true; }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,T_1s,{3,1}); TOGGLE=true; }
			else { out = Operator::outerprod(Id_1s,Id_1s,{1,1}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,T_1s,{3,1}); TOGGLE=true; }
			else if(TOGGLE==false) { out = Operator::outerprod(out,Id_1s,{1,1}); }
			else if(TOGGLE==true) { out = Operator::outerprod(out,Id_1s,{3,1}); }
		}
		return out;
	}
}

SiteOperatorQ<Sym::SU2xSU2<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xSU2<double> >::
Tdag (std::size_t orbital) const
{
	return T(orbital).adjoint();
}

SiteOperatorQ<Sym::SU2xSU2<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xSU2<double> >::
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

SiteOperatorQ<Sym::SU2xSU2<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xSU2<double> >::
HubbardHamiltonian (double U, double t, double V, double J, bool PERIODIC) const
{
	Operator Mout({1,1},TensorBasis);
	if( N_orbitals >= 2 and t!=0. )
	{
		Mout = -t*std::sqrt(2.)*std::sqrt(2.)*Operator::prod(cdag(0),c(1),{1,1});
	}
	for (int i=1; i<N_orbitals-1; ++i) // for all bonds
	{
		if (t != 0.)
		{
			Mout += -t*std::sqrt(2.)*std::sqrt(2.)*Operator::prod(cdag(i),c(i+1),{1,1});
		}
		// if (V != 0.) {Mout += V*(Operator::prod(n(i),n(i+1),{1,0}));} //what is this term in so(4)?
		if (J != 0.)
		{
			Mout += -J*std::sqrt(3.)*(Operator::prod(Sdag(i),S(i+1),{1,1}));
		}
	}
	if (PERIODIC==true and N_orbitals>2)
	{
		assert(N_orbitals%2==0 and "A ring with an odd number of sites is not bipartite! No SO(4) symmetry");
		if (t != 0.)
		{
			Mout += -t*std::sqrt(2.)*std::sqrt(2.)*Operator::prod(cdag(0),c(N_orbitals-1),{1,1});
		}
		// if (V != 0.) {Mout += V*(Operator::prod(n(0),n(N_orbitals-1),{1,1}));} //what is this term in so(4)?
		if (J != 0.)
		{
			Mout += -J*std::sqrt(3.)*(Operator::prod(Sdag(0),S(N_orbitals-1),{1,1}));
		}
	}
	if (U != 0. and U != std::numeric_limits<double>::infinity())
	{
		for (int i=0; i<N_orbitals; ++i) {Mout += 0.5*U*nh(i);}
	}

	return Mout;
}

SiteOperatorQ<Sym::SU2xSU2<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xSU2<double> >::
HubbardHamiltonian (double U, Eigen::ArrayXXd t) const
{
	Operator Mout({1,1},TensorBasis);
	for (int i=1; i<N_orbitals; ++i) // for all bonds
	for (int j=0; j<N_orbitals; ++j)
	{
		if (t(i,j) != 0.)
		{
			Mout += -t(i,j)*std::sqrt(2.)*std::sqrt(2.)*Operator::prod(cdag(i),c(j),{1,1});
		}
	}
	if (U != 0. and U != std::numeric_limits<double>::infinity())
	{
		for (int i=0; i<N_orbitals; ++i) {Mout += 0.5*U*nh(i);}
	}

}

SiteOperatorQ<Sym::SU2xSU2<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xSU2<double> >::
HubbardHamiltonian (Eigen::ArrayXd Uorb, double t, double V, double J, bool PERIODIC) const
{
	auto Mout = HubbardHamiltonian(0.,t,V,J,PERIODIC);
	
	for (int i=0; i<N_orbitals; ++i)
	{
		if (Uorb.rows() > 0)
		{
			if (Uorb(i) != 0. and Uorb(i) != std::numeric_limits<double>::infinity())
			{
				Mout += 0.5*Uorb(i) * nh(i);
			}
		}
	}
	return Mout;
}

#endif
