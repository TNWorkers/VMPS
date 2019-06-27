#ifndef FERMIONBASESU2XU1_H_
#define FERMIONBASESU2XU1_H_

#include "symmetry/S1xS2.h"
#include "symmetry/U1.h"
#include "symmetry/SU2.h"
#include "bases/FermionBase.h"
#include "tensors/SiteOperatorQ.h"
//include "tensors/Qbasis.h"

//Note: Don't put a name in this documentation with \class .. because doxygen gets confused with template symbols
/** 
 * \ingroup Bases
 *
 * This class provides the local operators for fermions in a SU(2)⊗U(1) block representation.
 *
 */
template<>
class FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > >
{
	typedef Eigen::Index Index;
	typedef double Scalar;
	
public:
	
	typedef typename Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > Symmetry;
	typedef SiteOperatorQ<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > Operator;
	typedef typename Symmetry::qType qType;
	
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
	 * \note The annihilation spinor is build as follows
	 * \f$c^{1/2} = \left(
	 * \begin{array}{c}
	 * -c_{\downarrow} \\
	 *  c_{\uparrow} \\
	 * \end{array}
	 * \right)\f$
	 * where the upper component corresponds to \f$ m=+1/2\f$ and the lower to \f$ m=-1/2\f$.
	 */
	Operator c (std::size_t orbital=0) const;
	
	/**
	 * Creation operator.
	 * \param orbital : orbital index
	 * \note The creation spinor is computed as \f$ \left(c^{1/2}\right)^\dagger\f$.
	 * The definition of cdag which is consistent with this computation is:
	 * \f$\left(c^{\dagger}\right)^{1/2} = \left(
	 * \begin{array}{c}
	 *  c^\dagger_{\uparrow} \\
	 *  c^\dagger_{\downarrow} \\
	 * \end{array}
	 * \right)\f$
	 * where the upper component corresponds to \f$ m=+1/2\f$ and the lower to \f$ m=-1/2\f$.
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
	
	/**
	 * Spinon density \f$n_s=n-2d\f$
	 * \param orbital : orbital index
	 */
	Operator ns (std::size_t orbital=0) const
	{
		return n(orbital)-2.*d(orbital);
	};
	
	/**
	 * Holon density \f$n_h=2d-n-1=1-n_s\f$
	 * \param orbital : orbital index
	 */
	Operator nh (std::size_t orbital=0) const
	{
		return 2.*d(orbital)-n(orbital)+Id(orbital);
	};
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
	 * Isospin z-component
	 * \param orbital : orbital index
	 */
	Operator Tz (std::size_t orbital=0) const;
	
//	/**
//	 * Isospin ladder operator +
//	 * \param orbital : orbital index
//	 */
//	Operator Tp (std::size_t orbital=0) const;
//	
//	/**
//	 * Isospin ladder operator -
//	 * \param orbital : orbital index
//	 */
//	Operator Tm (std::size_t orbital=0) const;
//	///\}
//	
	///\{
	/**
	 * Orbital pairing η
	 * \param orbital : orbital index
	 */
	Operator cc (std::size_t orbital=0) const;
	
	/**
	 * Orbital paring η† 
	 * \param orbital : orbital index
	 **/
	Operator cdagcdag (std::size_t orbital=0) const;
	///\}
	
	/**Returns an array of size dim() with zeros.*/
	ArrayXd ZeroField() const { return ArrayXd::Zero(N_orbitals); }
	
	/**Returns an array of size dim()xdim() with zeros.*/
	ArrayXXd ZeroHopping() const { return ArrayXXd::Zero(N_orbitals,N_orbitals); }
	
	/**
	 * Creates the full Hubbard Hamiltonian on the supersite with orbital-dependent U.
	 * \param U : \f$U\f$ for each orbital
	 * \param Uph : particle-hole symmetric \f$U\f$ for each orbital (times \f$(n_{\uparrow}-1/2)(n_{\downarrow}-1/2)+1/4\f$)
	 * \param Eorb : \f$\varepsilon\f$ onsite energy for each orbital
	 * \param t : \f$t\f$
	 * \param V : \f$V\f$
	 * \param Vz : \f$V_z\f$
	 * \param Vxy : \f$V_{xy}\f$
	 * \param J : \f$J\f$
	 */
	Operator HubbardHamiltonian (const ArrayXd &U, const ArrayXd &Uph, const ArrayXd &Eorb, const ArrayXXd &t, const ArrayXXd &V,
	                             const ArrayXXd &Vz, const ArrayXXd &Vxy, const ArrayXXd &J) const;
	
	/**Identity*/
	Operator Id (std::size_t orbital=0) const;
	
	/**Returns the basis.*/
	Qbasis<Symmetry> get_basis() const { return TensorBasis; }
	
private:
	
	std::size_t N_orbitals;
	std::size_t N_states;
	
	Qbasis<Symmetry> basis_1s; //basis for one site
	Qbasis<Symmetry> TensorBasis; //Final basis for N_orbital sites

	//operators defined on one orbital
	Operator Id_vac, Zero_vac;
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

FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > >::
FermionBase (std::size_t L_input, bool U_IS_INFINITE)
:N_orbitals(L_input)
{
//	assert(N_orbitals>=1);
	
	std::size_t locdim = (U_IS_INFINITE)? 2 : 3;
	N_states = std::pow(locdim,N_orbitals);
	
	//create basis for one Fermionic Site
	typename Symmetry::qType Q={1,0}; //empty occupied state
	Eigen::Index inner_dim = 1;
	std::vector<std::string> ident;
	
	ident.push_back("empty");
	basis_1s.push_back(Q,inner_dim,ident);
	Qbasis<Symmetry> vacuum = basis_1s;
	ident.clear();
	
	Q={2,1}; //singly occupied state
	inner_dim = 1;
	ident.push_back("single");
	basis_1s.push_back(Q,inner_dim,ident);
	ident.clear();
	
	Q={1,2}; //doubly occupied state
	inner_dim = 1;
	ident.push_back("double");
	basis_1s.push_back(Q,inner_dim,ident);
	ident.clear();
	
	Id_vac = Operator({1,0},basis_1s);
	Zero_vac = Operator({1,0},basis_1s);
	
	Id_1s = Operator({1,0},basis_1s);
	F_1s = Operator({1,0},basis_1s);
	c_1s = Operator({2,-1},basis_1s);
	d_1s = Operator({1,0},basis_1s);
	S_1s = Operator({3,0},basis_1s);
	
	//create operators for zero and one orbitals
	Id_vac("empty", "empty") = 1.;
	Zero_vac("empty", "empty") = 0.;
	
	Id_1s("empty", "empty") = 1.;
	Id_1s("double", "double") = 1.;
	Id_1s("single", "single") = 1.;
	
	F_1s("empty", "empty") = 1.;
	F_1s("double", "double") = 1.;
	F_1s("single", "single") = -1.;
	
	c_1s("empty", "single")  = std::sqrt(2.);
	c_1s("single", "double") = 1.;
	
	cdag_1s = c_1s.adjoint();
	n_1s = std::sqrt(2.) * Operator::prod(cdag_1s,c_1s,{1,0});
	d_1s( "double", "double" ) = 1.;
	S_1s( "single", "single" ) = std::sqrt(0.75);
	p_1s = -std::sqrt(0.5) * Operator::prod(c_1s,c_1s,{1,-2}); //The sign convention corresponds to c_DN c_UP
	pdag_1s = p_1s.adjoint(); //The sign convention corresponds to (c_DN c_UP)†=c_UP† c_DN†
	
	if (N_states == 1)
	{
		basis_1s = vacuum;
		N_orbitals = 1;
	}
	
	//create basis for N_orbitals fermionic sites
	if      (N_orbitals == 1) {TensorBasis = basis_1s;}
	else if (N_orbitals == 0) {TensorBasis = vacuum;}
	else
	{
		TensorBasis = basis_1s.combine(basis_1s);
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			TensorBasis = TensorBasis.combine(basis_1s);
		}
	}
}

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > >::
c (std::size_t orbital) const
{
	if (N_orbitals == 1) {return c_1s;}
	else if (N_orbitals == 0) {return Zero_vac;}
	else
	{
		Operator out;
		bool TOGGLE=false;
		if (orbital == 0) {out = Operator::outerprod(c_1s,Id_1s,{2,-1}); TOGGLE=true;}
		else
		{
			if (orbital == 1) {out = Operator::outerprod(F_1s,c_1s,{2,-1}); TOGGLE=true;}
			else {out = Operator::outerprod(F_1s,F_1s,{1,0});}
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if      (orbital == o)  {out = Operator::outerprod(out,c_1s,{2,-1}); TOGGLE=true; }
			else if (TOGGLE==false) {out = Operator::outerprod(out,F_1s,{1,0});}
			else if (TOGGLE==true)  {out = Operator::outerprod(out,Id_1s,{2,-1});}
		}
		return out;
	}
}

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > >::
cdag (std::size_t orbital) const
{
	return c(orbital).adjoint();
}

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > >::
sign (std::size_t orb1, std::size_t orb2) const
{
	if (N_orbitals == 1) {return F_1s;}
	else if (N_orbitals == 0) {return Zero_vac;}
	else
	{
		Operator out = Id();
		for (int i=orb1; i<N_orbitals; ++i)
		{
			out = Operator::prod(out, (Id()-2.*n(i)+4.*d(i)),{1,0});
		}
		for (int i=0; i<orb2; ++i)
		{
			out = Operator::prod(out, (Id()-2.*n(i)+4.*d(i)),{1,0});
		}
		
		return out;
	}
}

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > >::
sign_local (std::size_t orbital) const
{
	if (N_orbitals == 1) {return F_1s;}
	else if (N_orbitals == 0) {return Zero_vac;}
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

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > >::
n (std::size_t orbital) const
{
	if (N_orbitals == 1) {return n_1s;}
	else if (N_orbitals == 0) {return Zero_vac;}
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

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > >::
d (std::size_t orbital) const
{
	if (N_orbitals == 1) {return d_1s;}
	else if (N_orbitals == 0) {return Zero_vac;}
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

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > >::
S (std::size_t orbital) const
{
	if (N_orbitals == 1) {return S_1s;}
	else if (N_orbitals == 0) {return Zero_vac;}
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

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > >::
Sdag (std::size_t orbital) const
{
	return S(orbital).adjoint();
}

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > >::
cc (std::size_t orbital) const
{
	if (N_orbitals == 1) {return p_1s;}
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

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > >::
cdagcdag (std::size_t orbital) const
{
	return cc(orbital).adjoint();
}

//SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > >::
//Tp (std::size_t orbital) const
//{
//	return Eta(orbital);
//}

//SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > >::
//Tm (std::size_t orbital) const
//{
//	return Eta(orbital).adjoint();
//}

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > >::
Tz (std::size_t orbital) const
{
	return 0.5*(n(orbital)-Id());
}

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > >::
Id (std::size_t orbital) const
{
	if (N_orbitals == 1) {return Id_1s;}
	else if (N_orbitals == 0) {return Id_vac;}
	else
	{
		Operator out = Operator::outerprod(Id_1s,Id_1s,{1,0});
		for(std::size_t o=2; o<N_orbitals; o++) { out = Operator::outerprod(out,Id_1s,{1,0}); }
		return out;
	}
}

SiteOperatorQ<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> >,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::U1<Sym::ChargeU1> > >::
HubbardHamiltonian (const ArrayXd &U, const ArrayXd &Uph, const ArrayXd &Eorb, const ArrayXXd &t, const ArrayXXd &V, const ArrayXXd &Vz, const ArrayXXd &Vxy, const ArrayXXd &J) const
{
	Operator Mout({1,0},TensorBasis);
	
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
		if (Vz(i,j) != 0.)
		{
			Mout += Vz(i,j) * (Operator::prod(Tz(i),Tz(j),{1,0}));
		}
		if (Vxy(i,j) != 0.)
		{
			Mout += 0.5 * Vxy(i,j) * pow(-1.,i+j) * (Operator::prod(cc(i),cdagcdag(j),{1,0}) + Operator::prod(cdagcdag(i),cc(j),{1,0}));
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
		if (Uph(i) != 0. and Uph(i) != std::numeric_limits<double>::infinity())
		{
			Mout += Uph(i) * (d(i)-0.5*n(i)+0.5*Id());
		}
		if (Eorb(i) != 0.)
		{
			Mout += Eorb(i) * n(i);
		}
	}
	return Mout;
}

#endif
