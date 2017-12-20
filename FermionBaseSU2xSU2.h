#ifndef FERMIONBASESU2XSU2_H_
#define FERMIONBASESU2XSU2_H_

#include <algorithm>
#include <iterator>

#include "SiteOperatorQ.h"
#include "qbasis.h"
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
	FermionBase (std::size_t L_input, bool U_IS_INFINITE=false);
	
	/**amount of states = \f$4^L\f$*/
	inline Index dim() const {return static_cast<Index>(N_states);}
	
	/**amount of orbitals*/
	inline std::size_t orbitals() const  {return N_orbitals;}

	// \{
	/** Annihilation operator
		\param subLattice : Partion of the operator (Either A or B)
		\param orbital : orbital index*/
	Operator c (SUB_LATTICE subLattice, std::size_t orbital=0) const;
	
	/**Creation operator.
	   \param subLattice : Partion of the operator (Either A or B)
	   \param orbital : orbital index*/
	Operator cdag (SUB_LATTICE subLattice, std::size_t orbital=0) const;

	// /**Fermionic sign for the hopping between two orbitals of nearest-neighbour supersites of a ladder.
	//    \param orb1 : orbital on supersite i
	//    \param orb2 : orbital on supersite i+1
	// */
	// Operator sign (SUB_LATTICE subLattice, std::size_t orb1=0, std::size_t orb2=0) const;

	// /**Fermionic sign for one orbital of a supersite.
	//    \param orbital : orbital index
	// */
	// Operator sign_local (SUB_LATTICE subLattice, std::size_t orbital=0) const;

	/**Holon occupation number operator
	\param orbital : orbital index*/
	Operator nh (std::size_t orbital=0) const;
		
	// /**Double occupation
	// \param orbital : orbital index*/
	// Operator d (std::size_t orbital=0) const;
	// ///\}
	
	// ///\{
	// /**Orbital spin
	//    \param orbital : orbital index*/
	// Operator S (std::size_t orbital=0) const;
	
	// /**Orbital spin† 
	//    \param orbital : orbital index*/
	// Operator Sdag (std::size_t orbital=0) const;
	// ///\}

	// ///\{
	// /**Orbital pairing η
	//    \param orbital : orbital index*/
	// Operator Eta (std::size_t orbital=0) const;
	
	// /**Orbital paring η† 
	//    \param orbital : orbital index*/
	// Operator Etadag (std::size_t orbital=0) const;
	// ///\}

	/**Creates the full Hubbard Hamiltonian on the supersite.
	\param U : \f$U\f$
	\param t : \f$t\f$
	\param V : \f$V\f$
	\param J : \f$J\f$
	\param PERIODIC: periodic boundary conditions if \p true*/
	Operator HubbardHamiltonian (double U, double t=1.) const; //, double V=0., double J=0., bool PERIODIC=false
	
	// /**Creates the full Hubbard Hamiltonian on the supersite with orbital-dependent U.
	// \param Uvec : \f$U\f$ for each orbital
	// \param onsite : \f$\varepsilon\f$ onsite energy for each orbital
	// \param t : \f$t\f$
	// \param V : \f$V\f$
	// \param J : \f$J\f$
	// \param PERIODIC: periodic boundary conditions if \p true*/
	// Operator HubbardHamiltonian (Eigen::ArrayXd Uorb, Eigen::ArrayXd Eorb, double t=1., double V=0., double J=0., bool PERIODIC=false) const;

	/**Identity*/
	Operator Id (std::size_t orbital=0) const;

	// /**Returns the basis. 
	//    \note Use this as input for Mps, Mpo classes.*/ 
	// std::vector<typename Symmetry::qType> qloc() const { return TensorBasis.qloc(); }

	// /**Returns the degeneracy vector of the basis. 
	//    \note Use this as input for Mps, Mpo classes.*/ 
	// std::vector<Eigen::Index> qlocDeg() const { return TensorBasis.qlocDeg(); }

	Qbasis<Symmetry> get_basis() const { return TensorBasis; }

private:

	std::size_t N_orbitals;
	std::size_t N_states;
	
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
	Operator d_1s; //double occupancy
	Operator S_1s; //orbital spin
	Operator Q_1s; //orbital pseudo spin
	Operator p_1s; //pairing
	Operator pdag_1s; //pairing adjoint
};

FermionBase<Sym::SU2xSU2<double> >::
FermionBase (std::size_t L_input, bool U_IS_INFINITE)
:N_orbitals(L_input)
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
	d_1s = Operator({1,1},basis_1s);
	S_1s = Operator({3,1},basis_1s);
	Q_1s = Operator({1,3},basis_1s);

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
	S_1s( "spinon", "spinon" ) = std::sqrt(0.75);
	Q_1s( "holon", "holon" )   = std::sqrt(0.75);

	// p_1s = -std::sqrt(0.5) * Operator::prod(c_1s,c_1s,{1,-2}); //The sign convention corresponds to c_DN c_UP
	// pdag_1s = p_1s.adjoint(); //The sign convention corresponds to (c_DN c_UP)†=c_UP† c_DN†

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
c (SUB_LATTICE subLattice, std::size_t orbital) const
{
	Operator Op_1s;
	if(subLattice == SUB_LATTICE::A) {Op_1s = c_1sA;}
	else if(subLattice == SUB_LATTICE::B) {Op_1s = c_1sB;}

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
cdag (SUB_LATTICE subLattice, std::size_t orbital) const
{
	return c(subLattice,orbital).adjoint();
}

// template<typename Scalar>
// SiteOperatorQ<Sym::SU2xU1<Scalar>,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Scalar>::
// sign (std::size_t orb1, std::size_t orb2) const
// {
// 	if(N_orbitals == 1) { return F_1s; }
// 	else
// 	{
// 		Operator out = Id();
// 		for (int i=orb1; i<N_orbitals; ++i)
// 		{
// 			// out = Operator::prod(out,sign_local(i),{1}); // * (Id-2.*n(UP,i))*(Id-2.*n(DN,i));
// 			out = Operator::prod(out, (Id()-2.*n(i)+4.*d(i)),{1,0});
// 		}
// 		for (int i=0; i<orb2; ++i)
// 		{
// 			// out = Operator::prod(out,sign_local(i),{1}); // * (Id-2.*n(UP,i))*(Id-2.*n(DN,i));
// 			out = Operator::prod(out, (Id()-2.*n(i)+4.*d(i)),{1,0});
// 		}

// 		return out;
// 	}
// }

// template<typename Scalar>
// SiteOperatorQ<Sym::SU2xU1<Scalar>,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Scalar>::
// sign_local (std::size_t orbital) const
// {
// 	if(N_orbitals == 1) { return F_1s; }
// 	else
// 	{
// 		Operator out;
// 		if(orbital == 0) { out = Operator::outerprod(F_1s,Id_1s,{1,0}); }
// 		else
// 		{
// 			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,F_1s,{1,0}); }
// 			else { out = Operator::outerprod(Id_1s,Id_1s,{1,0}); }
// 		}
// 		for(std::size_t o=2; o<N_orbitals; o++)
// 		{
// 			if(orbital == o) { out = Operator::outerprod(out,F_1s,{1,0}); }
// 			else { out = Operator::outerprod(out,Id_1s,{1,0}); }
// 		}
// 		return out;
// 	}
// }

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

// template<typename Scalar>
// SiteOperatorQ<Sym::SU2xU1<Scalar>,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Scalar>::
// d (std::size_t orbital) const
// {
// 	if(N_orbitals == 1) { return d_1s; }
// 	else
// 	{
// 		Operator out;
// 		if(orbital == 0) { out = Operator::outerprod(d_1s,Id_1s,{1,0}); }
// 		else
// 		{
// 			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,d_1s,{1,0}); }
// 			else { out = Operator::outerprod(Id_1s,Id_1s,{1,0}); }
// 		}
// 		for(std::size_t o=2; o<N_orbitals; o++)
// 		{
// 			if(orbital == o) { out = Operator::outerprod(out,d_1s,{1,0}); }
// 			else { out = Operator::outerprod(out,Id_1s,{1,0}); }
// 		}
// 		return out;
// 	}
// }

// template<typename Scalar>
// SiteOperatorQ<Sym::SU2xU1<Scalar>,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Scalar>::
// S (std::size_t orbital) const
// {
// 	if(N_orbitals == 1) { return S_1s; }
// 	else
// 	{
// 		Operator out;
// 		bool TOGGLE=false;
// 		if(orbital == 0) { out = Operator::outerprod(S_1s,Id_1s,{3,0}); TOGGLE=true; }
// 		else
// 		{
// 			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,S_1s,{3,0}); TOGGLE=true; }
// 			else { out = Operator::outerprod(Id_1s,Id_1s,{1,0}); }
// 		}
// 		for(std::size_t o=2; o<N_orbitals; o++)
// 		{
// 			if(orbital == o) { out = Operator::outerprod(out,S_1s,{3,0}); TOGGLE=true; }
// 			else if(TOGGLE==false) { out = Operator::outerprod(out,Id_1s,{1,0}); }
// 			else if(TOGGLE==true) { out = Operator::outerprod(out,Id_1s,{3,0}); }
// 		}
// 		return out;
// 	}
// }

// template<typename Scalar>
// SiteOperatorQ<Sym::SU2xU1<Scalar>,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Scalar>::
// Sdag (std::size_t orbital) const
// {
// 	return S(orbital).adjoint();
// }

// template<typename Scalar>
// SiteOperatorQ<Sym::SU2xU1<Scalar>,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Scalar>::
// Eta (std::size_t orbital) const
// {
// 	if(N_orbitals == 1) { return p_1s; }
// 	else
// 	{
// 		Operator out;
// 		bool TOGGLE=false;
// 		if(orbital == 0) { out = Operator::outerprod(p_1s,Id_1s,{1,-2}); TOGGLE=true; }
// 		else
// 		{
// 			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,p_1s,{1,-2}); TOGGLE=true; }
// 			else { out = Operator::outerprod(Id_1s,Id_1s,{1,0}); }
// 		}
// 		for(std::size_t o=2; o<N_orbitals; o++)
// 		{
// 			if(orbital == o) { out = Operator::outerprod(out,p_1s,{1,-2}); TOGGLE=true; }
// 			else if(TOGGLE==false) { out = Operator::outerprod(out,Id_1s,{1,0}); }
// 			else if(TOGGLE==true) { out = Operator::outerprod(out,Id_1s,{1,-2}); }
// 		}
// 		return out;
// 	}
// }

// template<typename Scalar>
// SiteOperatorQ<Sym::SU2xU1<Scalar>,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Scalar>::
// Etadag (std::size_t orbital) const
// {
// 	return Eta(orbital).adjoint();
// }

SiteOperatorQ<Sym::SU2xSU2<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xSU2<double> >::
Id (std::size_t orbital) const
{
	if(N_orbitals == 1) { return Id_1s; }
	else
	{
		Operator out = Operator::outerprod(Id_1s,Id_1s,{1,0});
		for(std::size_t o=2; o<N_orbitals; o++) { out = Operator::outerprod(out,Id_1s,{1,1}); }
		return out;
	}
}

SiteOperatorQ<Sym::SU2xSU2<double>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2xSU2<double> >::
HubbardHamiltonian (double U, double t) const //double V, double J, bool PERIODIC
{
	Operator Mout({1,1},TensorBasis);
	if( N_orbitals >= 2 and t!=0. )
	{
		cout << "iA=0" << endl; Mout = -t*std::sqrt(2.)*std::sqrt(2.)*Operator::prod(cdag(SUB_LATTICE::A,0),c(SUB_LATTICE::B,1),{1,1}); //std::sqrt(2.)
	}
	for (int i=1; i<N_orbitals-1; ++i) // for all bonds
	{
		if (t != 0.)
		{
			if(i%2 == 0) { Mout += -t*std::sqrt(2.)*std::sqrt(2.)*Operator::prod(cdag(SUB_LATTICE::A,i),c(SUB_LATTICE::B,i+1),{1,1}); }
			else { Mout += -t*std::sqrt(2.)*std::sqrt(2.)*Operator::prod(cdag(SUB_LATTICE::B,i),c(SUB_LATTICE::A,i+1),{1,1}); }
		}
		// if (V != 0.) {Mout += V*(Operator::prod(n(i),n(i+1),{1,0}));}
		// if (J != 0.)
		// {
		// 	Mout += -J*std::sqrt(3.)*(Operator::prod(Sdag(i),S(i+1),{1,0}));
		// }
	}
	// if (PERIODIC==true and N_orbitals>2)
	// {
	// 	if (t != 0.)
	// 	{
	// 		Mout += -t*std::sqrt(2.)*(Operator::prod(cdag(0),c(N_orbitals-1),{1,0})+Operator::prod(cdag(N_orbitals-1),c(0),{1,0}));
	// 	}
	// 	if (V != 0.) {Mout += V*(Operator::prod(n(0),n(N_orbitals-1),{1}));}
	// 	if (J != 0.)
	// 	{
	// 		Mout += -J*std::sqrt(3.)*(Operator::prod(Sdag(0),S(N_orbitals-1),{1,0}));
	// 	}
	// }
	if (U != 0. and U != std::numeric_limits<double>::infinity())
	{
		for (int i=0; i<N_orbitals; ++i) {Mout += 0.5*U*nh(i);}
	}

	return Mout;
}

// template<typename Scalar>
// SiteOperatorQ<Sym::SU2xU1<Scalar>,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Scalar>::
// HubbardHamiltonian (Eigen::ArrayXd Uorb, Eigen::ArrayXd Eorb, double t, double V, double J, bool PERIODIC) const
// {
// 	auto Mout = HubbardHamiltonian(0.,t,V,J,PERIODIC);
	
// 	for (int i=0; i<N_orbitals; ++i)
// 	{
// 		if (Uorb.rows() > 0)
// 		{
// 			if (Uorb(i) != 0. and Uorb(i) != std::numeric_limits<double>::infinity())
// 			{
// 				Mout += Uorb(i) * d(i);
// 			}
// 		}
// 		if (Eorb.rows() > 0)
// 		{
// 			if (Eorb(i) != 0.)
// 			{
// 				Mout += Eorb(i) * n(i);
// 			}
// 		}
// 	}
// 	return Mout;
// }

#endif
