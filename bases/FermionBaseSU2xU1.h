#ifndef FERMIONBASESU2XU1_H_
#define FERMIONBASESU2XU1_H_

#include <algorithm>
#include <iterator>

#include "tensors/SiteOperatorQ.h"
#include "symmetry/qbasis.h"
#include "symmetry/SU2xU1.h"

#include "bases/FermionBase.h"

//Note: Don't put a name in this documentation with \class .. because doxygen gets confused with template symbols
/** 
 * \ingroup Bases
 *
 * This class provides the local operators for fermions in a SU(2)⊗U(1) block representation.
 *
 */
template<>
class FermionBase<Sym::SU2xU1<double> >
{
	typedef Eigen::Index Index;
	typedef double Scalar;
	typedef typename Sym::SU2xU1<Scalar> Symmetry;
	typedef SiteOperatorQ<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > Operator;
	typedef typename Symmetry::qType qType;
public:
	
	FermionBase(){};
	
	/**
	 * \param L_input : the amount of orbitals
	 * \param U_IS_INFINITE : if \p true, eliminates doubly-occupied sites from the basis
	 */
	FermionBase (std::size_t L_input, bool U_IS_INFINITE=false);
	
	/**amount of states*/
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
	 * Annihilation operator
	 * \param orbital : orbital index
	 */
	Operator a (std::size_t orbital=0) const;
	
	/**
	 * Creation operator.
	 * \param orbital : orbital index
	 */
	Operator adag (std::size_t orbital=0) const;

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

	///\{
	/**
	 * Orbital pairing η
	 * \param orbital : orbital index
	 */
	Operator Eta (std::size_t orbital=0) const;
	
	/**
	 * Orbital paring η† 
	 * \param orbital : orbital index
	 **/
	Operator Etadag (std::size_t orbital=0) const;
	///\}

	/**
	 * Creates the full Hubbard Hamiltonian on the supersite.
	 * \param U : \f$U\f$
	 * \param t : \f$t\f$
	 * \param V : \f$V\f$
	 * \param J : \f$J\f$
	 * \param PERIODIC: periodic boundary conditions if \p true
	 */
	Operator HubbardHamiltonian (double U, double t=1., double V=0., double J=0., bool PERIODIC=false) const;
	
	/**
	 * Creates the full Hubbard Hamiltonian on the supersite with orbital-dependent U.
	 * \param Uorb : \f$U\f$ for each orbital
	 * \param Eorb : \f$\varepsilon\f$ onsite energy for each orbital
	 * \param t : \f$t\f$
	 * \param V : \f$V\f$
	 * \param J : \f$J\f$
	 * \param PERIODIC: periodic boundary conditions if \p true
	 */
	Operator HubbardHamiltonian (Eigen::ArrayXd Uorb, Eigen::ArrayXd Eorb, double t=1., double V=0., double J=0., bool PERIODIC=false) const;

	/**Identity*/
	Operator Id (std::size_t orbital=0) const;

	/**Returns the basis.*/
	vector<qType> get_basis() const { return TensorBasis.qloc(); }
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
	Operator a_1s; //annihilation
	Operator adag_1s; //creation
	Operator n_1s; //particle number
	Operator d_1s; //double occupancy
	Operator S_1s; //orbital spin
	Operator p_1s; //pairing
	Operator pdag_1s; //pairing adjoint
};

FermionBase<Sym::SU2xU1<double> >::
FermionBase (std::size_t L_input, bool U_IS_INFINITE)
:N_orbitals(L_input)
{
	assert(N_orbitals>=1);
	
	std::size_t locdim = (U_IS_INFINITE)? 2 : 3;
	N_states = std::pow(locdim,N_orbitals);

	//create basis for one Fermionic Site
	typename Symmetry::qType Q={1,0}; //empty occupied state
	Eigen::Index inner_dim = 1;
	std::vector<std::string> ident;
	ident.push_back("empty");
	basis_1s.push_back(Q,inner_dim,ident);
	ident.clear();	
	Q={2,1}; //single occupied state
	inner_dim = 1;
	ident.push_back("single");
	basis_1s.push_back(Q,inner_dim,ident);
	ident.clear();
	Q={1,2}; //double occupied state
	inner_dim = 1;
	ident.push_back("double");
	basis_1s.push_back(Q,inner_dim,ident);
	ident.clear();

	Id_1s = Operator({1,0},basis_1s);
	F_1s = Operator({1,0},basis_1s);
	c_1s = Operator({2,-1},basis_1s);
	a_1s = Operator({2,-1},basis_1s);
	d_1s = Operator({1,0},basis_1s);
	S_1s = Operator({3,0},basis_1s);

	//create operators for one orbital
	Id_1s( "empty", "empty" ) = 1.;
	Id_1s( "double", "double" ) = 1.;
	Id_1s( "single", "single" ) = 1.;

	F_1s( "empty", "empty" ) = 1.;
	F_1s( "double", "double" ) = 1.;
	F_1s( "single", "single" ) = -1.;

	c_1s( "empty", "single" ) = std::sqrt(2.);
	c_1s( "single", "double" ) = 1.;
	a_1s( "empty", "single" ) = std::sqrt(2.);
	a_1s( "single", "double" ) = 1.;

	// cdag_1s = Operator({2,+1},basis_1s);
	// cdag_1s( "single", "empty" ) = 1.;//std::sqrt(2.);
	// cdag_1s( "double", "single" ) = -std::sqrt(2.); //1.;

	cdag_1s = c_1s.adjoint();
	adag_1s = a_1s.adjoint();
	n_1s = std::sqrt(2.) * Operator::prod(cdag_1s,c_1s,{1,0});
	d_1s( "double", "double" ) = 1.;
	S_1s( "single", "single" ) = std::sqrt(0.75);
	p_1s = -std::sqrt(0.5) * Operator::prod(c_1s,c_1s,{1,-2}); //The sign convention corresponds to c_DN c_UP
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

SiteOperatorQ<Sym::SU2xU1<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xU1<double> >::
c (std::size_t orbital) const
{
	if(N_orbitals == 1) { return c_1s; }
	else
	{
		Operator out;
		bool TOGGLE=false;
		if(orbital == 0) { out = Operator::outerprod(c_1s,Id_1s,{2,-1}); TOGGLE=true; }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(F_1s,c_1s,{2,-1}); TOGGLE=true; }
			else { out = Operator::outerprod(F_1s,F_1s,{1,0}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,c_1s,{2,-1}); TOGGLE=true; }
			else if(TOGGLE==false) { out = Operator::outerprod(out,F_1s,{1,0}); }
			else if(TOGGLE==true) { out = Operator::outerprod(out,Id_1s,{2,-1}); }
		}
		return out;
	}
}

SiteOperatorQ<Sym::SU2xU1<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xU1<double> >::
cdag (std::size_t orbital) const
{
	return c(orbital).adjoint();
}

SiteOperatorQ<Sym::SU2xU1<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xU1<double> >::
a (std::size_t orbital) const
{
	if(N_orbitals == 1) { return a_1s; }
	else
	{
		Operator out;
		bool TOGGLE=false;
		if(orbital == 0) { out = Operator::outerprod(c_1s,Id_1s,{2,-1}); TOGGLE=true; }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,c_1s,{2,-1}); TOGGLE=true; }
			else { out = Operator::outerprod(Id_1s,Id_1s,{1,0}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,c_1s,{2,-1}); TOGGLE=true; }
			else if(TOGGLE==false) { out = Operator::outerprod(out,Id_1s,{1,0}); }
			else if(TOGGLE==true) { out = Operator::outerprod(out,Id_1s,{2,-1}); }
		}
		return out;
	}
}

SiteOperatorQ<Sym::SU2xU1<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xU1<double> >::
adag (std::size_t orbital) const
{
	return a(orbital).adjoint();
}

SiteOperatorQ<Sym::SU2xU1<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xU1<double> >::
sign (std::size_t orb1, std::size_t orb2) const
{
	if(N_orbitals == 1) { return F_1s; }
	else
	{
		Operator out = Id();
		for (int i=orb1; i<N_orbitals; ++i)
		{
			// out = Operator::prod(out,sign_local(i),{1}); // * (Id-2.*n(UP,i))*(Id-2.*n(DN,i));
			out = Operator::prod(out, (Id()-2.*n(i)+4.*d(i)),{1,0});
		}
		for (int i=0; i<orb2; ++i)
		{
			// out = Operator::prod(out,sign_local(i),{1}); // * (Id-2.*n(UP,i))*(Id-2.*n(DN,i));
			out = Operator::prod(out, (Id()-2.*n(i)+4.*d(i)),{1,0});
		}

		return out;
	}
}

SiteOperatorQ<Sym::SU2xU1<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xU1<double> >::
sign_local (std::size_t orbital) const
{
	if(N_orbitals == 1) { return F_1s; }
	else
	{
		Operator out;
		if(orbital == 0) { out = Operator::outerprod(F_1s,Id_1s,{1,0}); }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,F_1s,{1,0}); }
			else { out = Operator::outerprod(Id_1s,Id_1s,{1,0}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,F_1s,{1,0}); }
			else { out = Operator::outerprod(out,Id_1s,{1,0}); }
		}
		return out;
	}
}

SiteOperatorQ<Sym::SU2xU1<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xU1<double> >::
n (std::size_t orbital) const
{
	if(N_orbitals == 1) { return n_1s; }
	else
	{
		Operator out;
		if(orbital == 0) { out = Operator::outerprod(n_1s,Id_1s,{1,0}); }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,n_1s,{1,0}); }
			else { out = Operator::outerprod(Id_1s,Id_1s,{1,0}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,n_1s,{1,0}); }
			else { out = Operator::outerprod(out,Id_1s,{1,0}); }
		}
		return out;
	}
}

SiteOperatorQ<Sym::SU2xU1<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xU1<double> >::
d (std::size_t orbital) const
{
	if(N_orbitals == 1) { return d_1s; }
	else
	{
		Operator out;
		if(orbital == 0) { out = Operator::outerprod(d_1s,Id_1s,{1,0}); }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,d_1s,{1,0}); }
			else { out = Operator::outerprod(Id_1s,Id_1s,{1,0}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,d_1s,{1,0}); }
			else { out = Operator::outerprod(out,Id_1s,{1,0}); }
		}
		return out;
	}
}

SiteOperatorQ<Sym::SU2xU1<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xU1<double> >::
S (std::size_t orbital) const
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

SiteOperatorQ<Sym::SU2xU1<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xU1<double> >::
Sdag (std::size_t orbital) const
{
	return S(orbital).adjoint();
}

SiteOperatorQ<Sym::SU2xU1<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xU1<double> >::
Eta (std::size_t orbital) const
{
	if(N_orbitals == 1) { return p_1s; }
	else
	{
		Operator out;
		bool TOGGLE=false;
		if(orbital == 0) { out = Operator::outerprod(p_1s,Id_1s,{1,-2}); TOGGLE=true; }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,p_1s,{1,-2}); TOGGLE=true; }
			else { out = Operator::outerprod(Id_1s,Id_1s,{1,0}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,p_1s,{1,-2}); TOGGLE=true; }
			else if(TOGGLE==false) { out = Operator::outerprod(out,Id_1s,{1,0}); }
			else if(TOGGLE==true) { out = Operator::outerprod(out,Id_1s,{1,-2}); }
		}
		return out;
	}
}

SiteOperatorQ<Sym::SU2xU1<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xU1<double> >::
Etadag (std::size_t orbital) const
{
	return Eta(orbital).adjoint();
}

SiteOperatorQ<Sym::SU2xU1<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xU1<double> >::
Id (std::size_t orbital) const
{
	if(N_orbitals == 1) { return Id_1s; }
	else
	{
		Operator out = Operator::outerprod(Id_1s,Id_1s,{1,0});
		for(std::size_t o=2; o<N_orbitals; o++) { out = Operator::outerprod(out,Id_1s,{1,0}); }
		return out;
	}
}

SiteOperatorQ<Sym::SU2xU1<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xU1<double> >::
HubbardHamiltonian (double U, double t, double V, double J, bool PERIODIC) const
{
	Operator Mout({1,0},TensorBasis);
	if( N_orbitals >= 2 and t!=0. )
	{
		Mout = -t*std::sqrt(2.)*(Operator::prod(cdag(0),c(1),{1,0})+Operator::prod(c(0),cdag(1),{1,0}));
	}
	for (int i=1; i<N_orbitals-1; ++i) // for all bonds
	{
		if (t != 0.)
		{
			Mout += -t*std::sqrt(2.)*(Operator::prod(cdag(i),c(i+1),{1,0})+Operator::prod(c(i),cdag(i+1),{1,0}));
		}
		if (V != 0.) {Mout += V*(Operator::prod(n(i),n(i+1),{1,0}));}
		if (J != 0.)
		{
			Mout += -J*std::sqrt(3.)*(Operator::prod(Sdag(i),S(i+1),{1,0}));
		}
	}
	if (PERIODIC==true and N_orbitals>2)
	{
		if (t != 0.)
		{
			Mout += -t*std::sqrt(2.)*(Operator::prod(cdag(0),c(N_orbitals-1),{1,0})+Operator::prod(cdag(N_orbitals-1),c(0),{1,0}));
		}
		if (V != 0.) {Mout += V*(Operator::prod(n(0),n(N_orbitals-1),{1}));}
		if (J != 0.)
		{
			Mout += -J*std::sqrt(3.)*(Operator::prod(Sdag(0),S(N_orbitals-1),{1,0}));
		}
	}
	if (U != 0. and U != std::numeric_limits<double>::infinity())
	{
		for (int i=0; i<N_orbitals; ++i) {Mout += U*d(i);}
	}

	return Mout;
}

SiteOperatorQ<Sym::SU2xU1<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xU1<double> >::
HubbardHamiltonian (Eigen::ArrayXd Uorb, Eigen::ArrayXd Eorb, double t, double V, double J, bool PERIODIC) const
{
	auto Mout = HubbardHamiltonian(0.,t,V,J,PERIODIC);
	
	for (int i=0; i<N_orbitals; ++i)
	{
		if (Uorb.rows() > 0)
		{
			if (Uorb(i) != 0. and Uorb(i) != std::numeric_limits<double>::infinity())
			{
				Mout += Uorb(i) * d(i);
			}
		}
		if (Eorb.rows() > 0)
		{
			if (Eorb(i) != 0.)
			{
				Mout += Eorb(i) * n(i);
			}
		}
	}
	return Mout;
}

#endif
