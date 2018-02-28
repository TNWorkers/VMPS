#ifndef FERMIONBASEU0XSU2_H_
#define FERMIONBASEU0XSU2_H_

#include <algorithm>
#include <iterator>

#include "symmetry/kind_dummies.h"
#include "symmetry/SU2.h"
#include "tensors/SiteOperatorQ.h"
#include "symmetry/qbasis.h"

#include "bases/FermionBase.h"

//Note: Don't put a name in this documentation with \class .. because doxygen gets confused with template symbols
/**
 * \ingroup Bases
 *
 * This class provides the local operators for fermions in a charge-SU(2) block representation for \p N_Orbitals fermionic sites.
 *
 * \note The SU(2)-charge symmetry is only present for a bipartite lattice.
 *       Using this class in DMRG requires that the SUB_LATTICE parameter is toggled every site.
 */
template<>
class FermionBase<Sym::SU2<Sym::ChargeSU2> >
{
	typedef Eigen::Index Index;
	typedef double Scalar;
	typedef typename Sym::SU2<Sym::ChargeSU2> Symmetry;
	typedef SiteOperatorQ<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > Operator;
	typedef typename Symmetry::qType qType;
public:
	
	FermionBase(){};
	
	/**
	 * \param L_input : the amount of orbitals
	 * \param subLattice_in : The SUB_LATTICE (Either A or B) of orbital 0. SUB_LATTICE of orbital i: \f$\sim (-1)^i \f$
	 */
	FermionBase (std::size_t L_input, SUB_LATTICE subLattice_in = SUB_LATTICE::A);
	
	/**amount of states.*/
	inline Index dim() const {return static_cast<Index>(TensorBasis.size());}
	
	/**amount of orbitals*/
	inline std::size_t orbitals() const  {return N_orbitals;}

	/**Returns the sublattice of orbital 0.*/
	inline SUB_LATTICE sublattice() const {return subLattice;}

	///\{
	/**
	 * Particle-hole spinor
	 * \param sigma : spin index
	 * \param orbital : orbital index
	 *
	 * The operator quantum number is \f$\frac{1}{2} \f$ and the spinor is defined as follows:
	 * \f$\psi_{\sigma} = \left(
	 * \begin{array}{c}
	 * sc^{\dag}_{-\sigma} \\
	 * \sigma c_{\sigma} \\
	 * \end{array}
	 * \right)\f$
	 * Where the upper component has pseudo-spin z quantumnumber \f$+\frac{1}{2} \f$ 
	 * and the sign \f$s\f$ is either +1 or -1 depending on the sublattice.
	 * \f$\sigma\f$ is as usual \f$\uparrow=1\f$ and \f$\downarrow=-1\f$
	 */
	Operator psi (SPIN_INDEX sigma, size_t orbital=0) const;

	/* Adjoint of particle-hole spinor
	 * \param sigma : spin index
	 * \param orbital : orbital index
	 */
	Operator psidag (SPIN_INDEX sigma, size_t orbital=0) const;
	///\}

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
	 * Holon occupation number operator
	 * \param orbital : orbital index
	 */
	Operator nh (std::size_t orbital=0) const;

	///\{
	/**
	 * Sz
	 * \param orbital : orbital index
	 */
	Operator Sz (std::size_t orbital=0) const;

	/**
	 * Sx
	 * \param orbital : orbital index
	 */
	Operator Sx (std::size_t orbital=0) const;

	/**
	 * Sp
	 * \param orbital : orbital index
	 */
	Operator Sp (std::size_t orbital=0) const;

	/**
	 * Sm
	 * \param orbital : orbital index
	 */
	Operator Sm (std::size_t orbital=0) const;
	///\}
	
	///\{
	/**
	 * Orbital pseudo spin
	 * \param orbital : orbital index
	 */
	Operator T (std::size_t orbital=0) const;
	
	/**
	 * Orbital pseudo spin† 
	 * \param orbital : orbital index
	 */
	Operator Tdag (std::size_t orbital=0) const;
	///\}
	
	/**
	 * Creates the full Hubbard Hamiltonian on the supersite.
	 * \param U : \f$U\f$
	 * \param t : \f$t\f$
	 * \param V : \f$V\f$
	 * \param Jz : \f$Jz\f$
	 * \param Jxy : \f$Jxy\f$
	 * \param Bz : \f$B_z\f$
	 * \param Bx : \f$B_x\f$
	 * \param PERIODIC: periodic boundary conditions if \p true
	 */
	Operator HubbardHamiltonian (double U, double t=1., double V=0., double Jz=0., double Jxy=0., double Bz=0., double Bx=0., bool PERIODIC=false) const;
	
	/**
	 * Creates the full Hubbard Hamiltonian on the supersite with inhomogeneuous parameters.
	 * \param Uorb : \f$U\f$ for each orbital
	 * \param t : \f$t_{ij}\f$ (hopping matrix)
	 * \param V : \f$V_{ij}\f$ (nn pseudo-spin pseudo-spin matrix)
	 * \param Jz : \f$J^{z}_{ij}\f$ (nn Spin interaction matrix)
	 * \param Jxy : \f$J^{xy}_{ij}\f$ (nn Spin interaction matrix)
	 * \param Bz : \f$B_z\f$ for each orbital
	 * \param Bx : \f$B_x\f$ for each orbital
	 */
	Operator HubbardHamiltonian (Eigen::VectorXd Uorb, Eigen::MatrixXd t, Eigen::MatrixXd V, Eigen::MatrixXd Jz, Eigen::MatrixXd Jxy,
								 Eigen::VectorXd Bz, Eigen::VectorXd Bx) const;
	
	/**Identity*/
	Operator Id () const;

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

	SUB_LATTICE subLattice;

	Qbasis<Symmetry> basis_1s; //basis for one site
	Qbasis<Symmetry> TensorBasis; //Final basis for N_orbital sites

	//operators defined on one orbital

	Operator psi_1sA(SPIN_INDEX sigma) const; //spinor sublattice A
	Operator psidag_1sA(SPIN_INDEX sigma) const; //adjoint spinor sublattice A
	Operator psi_1sB(SPIN_INDEX sigma) const; //spinor sublattice B
	Operator psidag_1sB(SPIN_INDEX sigma) const; //adjoint spinor sublattice B

	Operator Id_1s; //identity
	Operator F_1s; //Fermionic sign
	Operator nh_1s; //double occupancy
	Operator T_1s; //orbital pseudo spin
	Operator Sz_1s; //spin z operator
	Operator Sx_1s; //spin x operator
	Operator Sp_1s; //raising spin operator
	Operator Sm_1s; //lowering spin operator
	Operator p_1s; //pairing
	Operator pdag_1s; //pairing adjoint
};

FermionBase<Sym::SU2<Sym::ChargeSU2> >::
FermionBase (std::size_t L_input, SUB_LATTICE subLattice_in)
	:N_orbitals(L_input),subLattice(subLattice_in)
{
	assert(N_orbitals>=1);
	
	//create basis for one Fermionic Site with SU2 particle-hole symmetry
	typename Symmetry::qType Q={1}; //pseudo spin singlet states Spin Up and Spin Down
	Eigen::Index inner_dim;
	inner_dim = 2;
	std::vector<std::string> ident;
	ident.push_back("up");
	ident.push_back("down");
	basis_1s.push_back(Q,inner_dim,ident);
	ident.clear();	
	Q={2}; //particle-hole doublet states (empty and full) (the holon)
	inner_dim = 1;
	ident.push_back("holon");
	basis_1s.push_back(Q,inner_dim,ident);
	ident.clear();

	Id_1s = Operator({1},basis_1s);
	F_1s = Operator({1},basis_1s);
	nh_1s = Operator({1},basis_1s);
	T_1s = Operator({3},basis_1s);
	Sz_1s = Operator({1},basis_1s);
	Sx_1s = Operator({1},basis_1s);
	Sp_1s = Operator({1},basis_1s);
	Sm_1s = Operator({1},basis_1s);

	//create operators for one orbital
	Id_1s( "up", "up" ) = 1.;
	Id_1s( "down", "down" ) = 1.;
	Id_1s( "holon", "holon" ) = 1.;

	F_1s( "up", "up" ) = -1.;
	F_1s( "down", "down" ) = -1.;
	F_1s( "holon", "holon" ) = 1.;
		
	nh_1s( "holon", "holon" ) = 1.;
	
	T_1s( "holon", "holon" ) = std::pow(0.75,0.5);

	Sz_1s( "up", "up" ) = +0.5;
	Sz_1s( "down", "down" ) = -0.5;

	Sp_1s( "up", "down" ) = 1.;
	Sm_1s = Sp_1s.adjoint();

	Sx_1s = 0.5*(Sp_1s + Sm_1s);

	// p_1s = -std::sqrt(0.5) * Operator::prod(c_1s,c_1s,{1}); //The sign convention corresponds to c_DN c_UP
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
}
	
SiteOperatorQ<Sym::SU2<Sym::ChargeSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::ChargeSU2> >::
psi_1sA (SPIN_INDEX sigma) const
{
	Operator psi_1sA({2},basis_1s);

	if (sigma == SPIN_INDEX::UP)
	{
		psi_1sA( "down", "holon" ) = sqrt(2.);
		psi_1sA( "holon", "up" ) = -1.;
	}
	else if (sigma == SPIN_INDEX::DN)
	{
		psi_1sA( "up", "holon" ) = sqrt(2.);
		psi_1sA( "holon", "down" ) = 1.;
	}
	else { assert(1!=1 and "Something went wromg here..."); }
	return psi_1sA;
}

SiteOperatorQ<Sym::SU2<Sym::ChargeSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::ChargeSU2> >::
psidag_1sA (SPIN_INDEX sigma) const
{
	return psi_1sA(sigma).adjoint();
}

SiteOperatorQ<Sym::SU2<Sym::ChargeSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::ChargeSU2> >::
psi_1sB (SPIN_INDEX sigma) const
{
	Operator psi_1sB({2},basis_1s);

	if (sigma == SPIN_INDEX::UP)
	{
		psi_1sB( "down", "holon" ) = -1.*sqrt(2.);
		psi_1sB( "holon", "up" ) = -1.;
	}
	else if (sigma == SPIN_INDEX::DN)
	{
		psi_1sB( "up", "holon" ) = -1.*sqrt(2.);
		psi_1sB( "holon", "down" ) = 1.;
	}
	else { assert(1!=1 and "Something went wromg here..."); }

	return psi_1sB;
}

SiteOperatorQ<Sym::SU2<Sym::ChargeSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::ChargeSU2> >::
psidag_1sB (SPIN_INDEX sigma) const
{
	return psi_1sB(sigma).adjoint();
}

SiteOperatorQ<Sym::SU2<Sym::ChargeSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::ChargeSU2> >::
psi (SPIN_INDEX sigma, std::size_t orbital) const
{
	Operator Op_1s;
	if(subLattice == SUB_LATTICE::A)
	{
		if(orbital%2 == 0) {Op_1s = psi_1sA(sigma);}
		else {{Op_1s = psi_1sB(sigma);}}
	}
	else if(subLattice == SUB_LATTICE::B)
	{
		if(orbital%2 == 0) {Op_1s = psi_1sB(sigma);}
		else {{Op_1s = psi_1sA(sigma);}}
	}
	else {assert(1!=1 and "Crazy...");}

	if(N_orbitals == 1) { return Op_1s; }
	else
	{
		Operator out;
		bool TOGGLE=false;
		if(orbital == 0) { out = Operator::outerprod(Op_1s,Id_1s,{2}); TOGGLE=true; }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(F_1s,Op_1s,{2}); TOGGLE=true; }
			else { out = Operator::outerprod(F_1s,F_1s,{1}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,Op_1s,{2}); TOGGLE=true;}
			else if(TOGGLE==false) { out = Operator::outerprod(out,F_1s,{1}); }
			else if(TOGGLE==true) { out = Operator::outerprod(out,Id_1s,{2}); }
		}
		return out;
	}
}

SiteOperatorQ<Sym::SU2<Sym::ChargeSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::ChargeSU2> >::
psidag (SPIN_INDEX sigma, std::size_t orbital) const
{
	return psi(sigma,orbital).adjoint();
}

SiteOperatorQ<Sym::SU2<Sym::ChargeSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::ChargeSU2> >::
sign (std::size_t orb1, std::size_t orb2) const
{
	if(N_orbitals == 1) { return F_1s; }
	else
	{
		Operator out = Id();
		for (int i=orb1; i<N_orbitals; ++i)
		{
			out = Operator::prod(out, 2.*nh(i)-Id(),{1});
		}
		for (int i=0; i<orb2; ++i)
		{
			out = Operator::prod(out, 2.*nh(i),{1});
		}

		return out;
	}
}

SiteOperatorQ<Sym::SU2<Sym::ChargeSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::ChargeSU2> >::
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

SiteOperatorQ<Sym::SU2<Sym::ChargeSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::ChargeSU2> >::
nh (std::size_t orbital) const
{
	if(N_orbitals == 1) { return nh_1s; }
	else
	{
		Operator out;
		if(orbital == 0) { out = Operator::outerprod(nh_1s,Id_1s,{1}); }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,nh_1s,{1}); }
			else { out = Operator::outerprod(Id_1s,Id_1s,{1}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,nh_1s,{1}); }
			else { out = Operator::outerprod(out,Id_1s,{1}); }
		}
		return out;
	}
}

SiteOperatorQ<Sym::SU2<Sym::ChargeSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::ChargeSU2> >::
Sz (std::size_t orbital) const
{
	if(N_orbitals == 1) { return Sz_1s; }
	else
	{
		Operator out;
		if(orbital == 0) { out = Operator::outerprod(Sz_1s,Id_1s,{1}); }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,Sz_1s,{1}); }
			else { out = Operator::outerprod(Id_1s,Id_1s,{1}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,Sz_1s,{1}); }
			else { out = Operator::outerprod(out,Id_1s,{1}); }
		}
		return out;
	}
}

SiteOperatorQ<Sym::SU2<Sym::ChargeSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::ChargeSU2> >::
Sx (std::size_t orbital) const
{
	if(N_orbitals == 1) { return Sx_1s; }
	else
	{
		Operator out;
		if(orbital == 0) { out = Operator::outerprod(Sx_1s,Id_1s,{1}); }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,Sx_1s,{1}); }
			else { out = Operator::outerprod(Id_1s,Id_1s,{1}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,Sx_1s,{1}); }
			else { out = Operator::outerprod(out,Id_1s,{1}); }
		}
		return out;
	}
}

SiteOperatorQ<Sym::SU2<Sym::ChargeSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::ChargeSU2> >::
Sp (std::size_t orbital) const
{
	if(N_orbitals == 1) { return Sp_1s; }
	else
	{
		Operator out;
		if(orbital == 0) { out = Operator::outerprod(Sp_1s,Id_1s,{1}); }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,Sp_1s,{1}); }
			else { out = Operator::outerprod(Id_1s,Id_1s,{1}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,Sp_1s,{1}); }
			else { out = Operator::outerprod(out,Id_1s,{1}); }
		}
		return out;
	}
}

SiteOperatorQ<Sym::SU2<Sym::ChargeSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::ChargeSU2> >::
Sm (std::size_t orbital) const
{
	return Sp(orbital).adjoint();
}

SiteOperatorQ<Sym::SU2<Sym::ChargeSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::ChargeSU2> >::
T (std::size_t orbital) const
{
	if(N_orbitals == 1) { return T_1s; }
	else
	{
		Operator out;
		bool TOGGLE=false;
		if(orbital == 0) { out = Operator::outerprod(T_1s,Id_1s,{3}); TOGGLE=true; }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,T_1s,{3}); TOGGLE=true; }
			else { out = Operator::outerprod(Id_1s,Id_1s,{1}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,T_1s,{3}); TOGGLE=true; }
			else if(TOGGLE==false) { out = Operator::outerprod(out,Id_1s,{1}); }
			else if(TOGGLE==true) { out = Operator::outerprod(out,Id_1s,{3}); }
		}
		return out;
	}
}

SiteOperatorQ<Sym::SU2<Sym::ChargeSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::ChargeSU2> >::
Tdag (std::size_t orbital) const
{
	return T(orbital).adjoint();
}

SiteOperatorQ<Sym::SU2<Sym::ChargeSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::ChargeSU2> >::
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

SiteOperatorQ<Sym::SU2<Sym::ChargeSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::ChargeSU2> >::
HubbardHamiltonian (double U, double t, double V, double Jz, double Jxy, double Bz, double Bx, bool PERIODIC) const
{
	Operator Mout({1},TensorBasis);

	for (int i=0; i<N_orbitals-1; ++i) // for all bonds
	{
		if (t != 0.)
		{
			Mout += -t*std::sqrt(2.)*(Operator::prod(psidag(UP,i),psi(UP,i+1),{1})+Operator::prod(psidag(DN,i),psi(DN,i+1),{1}));
		}
		if (V != 0.) { Mout += -V*std::sqrt(3.)*(Operator::prod(Tdag(i),T(i+1),{1})); }
		if (Jz != 0.) { Mout += -Jz*Operator::prod(Sz(i),Sz(i+1),{1}); }
		if (Jxy != 0.) { Mout += -Jxy*0.5*(Operator::prod(Sp(i),Sm(i+1),{1})+Operator::prod(Sm(i),Sp(i+1),{1})); }
	}
	if (PERIODIC==true and N_orbitals>2)
	{
		if (t != 0.)
		{
			Mout += -t*std::sqrt(2.)*(Operator::prod(psidag(UP,0),psi(UP,N_orbitals-1),{1})+Operator::prod(psidag(DN,0),psi(DN,N_orbitals-1),{1}));
		}
		if (V != 0.) { Mout += -V*std::sqrt(3.)*(Operator::prod(Tdag(0),T(N_orbitals-1),{1})); }
		if (Jz != 0.) { Mout += -Jz*Operator::prod(Sz(0),Sz(N_orbitals-1),{1}); }
		if (Jxy != 0.) { Mout += -Jxy*0.5*(Operator::prod(Sp(0),Sm(N_orbitals-1),{1})+Operator::prod(Sm(0),Sp(N_orbitals-1),{1})); }
	}
	if (U != 0. and U != std::numeric_limits<double>::infinity()) { for (int i=0; i<N_orbitals; ++i) { Mout += 0.5 * U * nh(i); } }
	if (Bz != 0.) { for (int i=0; i<N_orbitals; ++i) { Mout += -1. * Bz * Sz(i); } }
	if (Bx != 0.) { for (int i=0; i<N_orbitals; ++i) { Mout += -1. * Bx * Sx(i); } }

	return Mout;
}

SiteOperatorQ<Sym::SU2<Sym::ChargeSU2>,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > FermionBase<Sym::SU2<Sym::ChargeSU2> >::
HubbardHamiltonian (Eigen::VectorXd Uorb, Eigen::MatrixXd t, Eigen::MatrixXd V, Eigen::MatrixXd Jz, Eigen::MatrixXd Jxy, Eigen::VectorXd Bz, Eigen::VectorXd Bx) const
{
	Operator Mout({1},TensorBasis);
	Mout.setZero();
	for (Eigen::Index i=0; i<N_orbitals-1; ++i)
	{
		if (Uorb.sum() != 0.)
		{
			for (int i=0; i<N_orbitals; ++i) {Mout += 0.5 * Uorb(i) * nh(i);}
		}
		if (Bz.sum() != 0.)
		{
			for (int i=0; i<N_orbitals; ++i) {Mout += -Bz(i) * Sz(i);}
		}
		if (Bx.sum() != 0.)
		{
			for (int i=0; i<N_orbitals; ++i) {Mout += -Bx(i) * Sx(i);}
		}

		for (Eigen::Index j=i+1; j<N_orbitals; ++j)
		{
			if (t(i,j) != 0.)
			{
				Mout += -t(i,j)*std::sqrt(2.)*(Operator::prod(psidag(UP,i),psi(UP,j),{1})+Operator::prod(psidag(DN,i),psi(DN,j),{1}));
			}
			if (V(i,j) != 0.) {Mout += V(i,j)*sqrt(3.)*(Operator::prod(Tdag(i),T(j),{1}));}
			if (Jz(i,j) != 0.) { Mout += -Jz(i,j)*(Operator::prod(Sz(i),Sz(j),{1})); }
			if (Jz(i,j) != 0.) { Mout += -0.5*Jxy(i,j)*(Operator::prod(Sp(i),Sm(j),{1})+Operator::prod(Sm(i),Sp(j),{1})); }
		}
	}

	return Mout;
}

#endif
