#ifndef FERMIONBASESU2
#define FERMIONBASESU2

#include <algorithm>
#include <iterator>

#include "symmetry/SU2.h"
#include "tensors/SiteOperator.h"
#include "symmetry/qbasis.h"

namespace fermions {

/** \class BaseSU2
  * \ingroup Fermions
  *
  * This class provides the local operators for fermions in a SU(2) block representation for \p N_Orbitals fermionic sites.
  *
  * \describe_Scalar
  *
  */
template<typename Scalar = double>
class BaseSU2
{
	typedef Eigen::Index Index;
	typedef typename Sym::SU2<Scalar> Symmetry;
	typedef SiteOperator<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > Operator;
	typedef typename Symmetry::qType qType;
public:
	
	BaseSU2(){};
	
	/**
	\param L_input : the amount of orbitals
	\param U_IS_INFINITE : if \p true, eliminates doubly-occupied sites from the basis*/
	BaseSU2 (std::size_t L_input, bool U_IS_INFINITE=false);
	
	/**amount of states = \f$3^L\f$*/
	inline Index dim() const {return static_cast<Index>(N_states);}
	
	/**amount of orbitals*/
	inline std::size_t orbitals() const  {return N_orbitals;}
	
	///\{
	/**Annihilation operator
	   \param orbital : orbital index*/
	Operator c (std::size_t orbital=0) const;
	
	/**Creation operator.
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

	/**Occupation number operator
	\param orbital : orbital index*/
	Operator n (std::size_t orbital=0) const;
		
	/**Double occupation
	\param orbital : orbital index*/
	Operator d (std::size_t orbital=0) const;
	///\}
	
	///\{
	/**Orbital spin
	   \param orbital : orbital index*/
	Operator S (std::size_t orbital=0) const;
	
	/**Orbital spin† 
	   \param orbital : orbital index*/
	Operator Sdag (std::size_t orbital=0) const;
	///\}
	
	/**Creates the full Hubbard Hamiltonian on the supersite.
	\param U : \f$U\f$
	\param mu : \f$\mu\f$ (chemical potential)
	\param t : \f$t\f$
	\param V : \f$V\f$
	\param J : \f$J\f$
	\param PERIODIC: periodic boundary conditions if \p true*/
	Operator HubbardHamiltonian (double U, double mu=std::numeric_limits<double>::infinity(), double t=1.,
								 double V=0., double J=0., bool PERIODIC=false) const;
	
	/**Creates the full Hubbard Hamiltonian on the supersite with orbital-dependent U.
	\param Uvec : \f$U\f$ for each orbital
	\param mu : \f$\mu\f$ (chemical potential)
	\param t : \f$t\f$
	\param V : \f$V\f$
	\param J : \f$J\f$
	\param PERIODIC: periodic boundary conditions if \p true*/
	Operator HubbardHamiltonian (std::vector<double> Uvec, double mu, double t=1., double V=0., double J=0., bool PERIODIC=false) const;

	/**Creates the full Hubbard Hamiltonian on the supersite with orbital-dependent U and arbitrary hopping matrix.
	\param U : \f$U\f$ for each orbital
	\param mu : \f$\mu\f$ (chemical potential)
	\param t : \f$t\f$ (hopping matrix)
	\param V : \f$V\f$ (nn Density Interaction matrix)
	\param J : \f$J\f$ (nn Spin interaction matrix)*/
	Operator HubbardHamiltonian (Eigen::VectorXd U, double mu, Eigen::MatrixXd t, Eigen::MatrixXd V, Eigen::MatrixXd J) const;
	
	/**Identity*/
	Operator Id () const;

	/**Returns the basis. 
	   \note Use this as input for Mps, Mpo classes.*/ 
	std::vector<typename Symmetry::qType> qloc() const { return TensorBasis.qloc(); }

	/**Returns the degeneracy vector of the basis. 
	   \note Use this as input for Mps, Mpo classes.*/ 
	std::vector<Eigen::Index> qlocDeg() const { return TensorBasis.qlocDeg(); }

	Qbasis<Symmetry> basis() const { return TensorBasis; }
	
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

template<typename Scalar>
BaseSU2<Scalar>::
BaseSU2 (std::size_t L_input, bool U_IS_INFINITE)
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

template<typename Scalar>
SiteOperator<Sym::SU2<Scalar>,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > BaseSU2<Scalar>::
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

template<typename Scalar>
SiteOperator<Sym::SU2<Scalar>,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > BaseSU2<Scalar>::
cdag (std::size_t orbital) const
{
	return c(orbital).adjoint();
}

template<typename Scalar>
SiteOperator<Sym::SU2<Scalar>,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > BaseSU2<Scalar>::
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

template<typename Scalar>
SiteOperator<Sym::SU2<Scalar>,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > BaseSU2<Scalar>::
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

template<typename Scalar>
SiteOperator<Sym::SU2<Scalar>,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > BaseSU2<Scalar>::
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

template<typename Scalar>
SiteOperator<Sym::SU2<Scalar>,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > BaseSU2<Scalar>::
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

template<typename Scalar>
SiteOperator<Sym::SU2<Scalar>,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > BaseSU2<Scalar>::
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

template<typename Scalar>
SiteOperator<Sym::SU2<Scalar>,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > BaseSU2<Scalar>::
Sdag (std::size_t orbital) const
{
	return S(orbital).adjoint();
}

template<typename Scalar>
SiteOperator<Sym::SU2<Scalar>,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > BaseSU2<Scalar>::
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

template<typename Scalar>
SiteOperator<Sym::SU2<Scalar>,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > BaseSU2<Scalar>::
HubbardHamiltonian (double U, double mu, double t, double V, double J, bool PERIODIC) const
{
	if( mu==std::numeric_limits<double>::infinity() ) { mu = U/2.; }
	Operator Mout({1},TensorBasis);
	if( N_orbitals >= 2 and t!=0. )
	{
		Mout = -t*std::sqrt(2.)*(Operator::prod(cdag(0),c(1),{1})+Operator::prod(c(0),cdag(1),{1}));
	}
	for (int i=1; i<N_orbitals-1; ++i) // for all bonds
	{
		if (t != 0.)
		{
			Mout += -t*std::sqrt(2.)*(Operator::prod(cdag(i),c(i+1),{1})+Operator::prod(c(i),cdag(i+1),{1}));
		}
		if (V != 0.) {Mout += V*(Operator::prod(n(i),n(i+1),{1}));}
		if (J != 0.)
		{
			Mout += -J*std::sqrt(3.)*(Operator::prod(Sdag(i),S(i+1),{1}));
		}
	}
	if (PERIODIC==true and N_orbitals>2)
	{
		if (t != 0.)
		{
			Mout += -t*std::sqrt(2.)*(Operator::prod(cdag(0),c(N_orbitals-1),{1})+Operator::prod(cdag(N_orbitals-1),c(0),{1}));
		}
		if (V != 0.) {Mout += V*(Operator::prod(n(0),n(N_orbitals-1),{1}));}
		if (J != 0.)
		{
			Mout += -J*std::sqrt(3.)*(Operator::prod(Sdag(0),S(N_orbitals-1),{1}));
		}
	}
	if (U != 0. and U != std::numeric_limits<double>::infinity())
	{
		for (int i=0; i<N_orbitals; ++i) {Mout += U*d(i);}
	}
	if (mu != 0.)
	{
		for (int i=0; i<N_orbitals; ++i) {Mout += (-mu)*n(i);}
	}

	return Mout;
}

template<typename Scalar>
SiteOperator<Sym::SU2<Scalar>,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > BaseSU2<Scalar>::
HubbardHamiltonian (std::vector<double> Uvec, double mu, double t, double V, double J, bool PERIODIC) const
{
	auto Mout = HubbardHamiltonian(0.,mu,t,V,J,PERIODIC);
	for (int i=0; i<N_orbitals; ++i)
	{
		if (Uvec.size() > 0)
		{
			if (Uvec[i] != 0. and Uvec[i] != std::numeric_limits<double>::infinity())
			{
				Mout += Uvec[i] * d(i);
			}
		}
	}
	return Mout;
}

template<typename Scalar>
SiteOperator<Sym::SU2<Scalar>,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > BaseSU2<Scalar>::
HubbardHamiltonian (Eigen::VectorXd U, double mu, Eigen::MatrixXd t, Eigen::MatrixXd V, Eigen::MatrixXd J) const
{
	Operator Mout({1},TensorBasis);
	Mout.setZero();
	for (Eigen::Index i=0; i<N_orbitals-1; ++i)
	{
		for (Eigen::Index j=i+1; j<N_orbitals; ++j)
		{
			if (t(i,j) != 0.)
			{
				Mout += -t(i,j)*std::sqrt(2.)*(Operator::prod(cdag(i),c(j),{1})+Operator::prod(c(i),cdag(j),{1}));
			}
			if (V(i,j) != 0.) {Mout += V(i,j)*(Operator::prod(n(i),n(j),{1}));}
			if (J(i,j) != 0.)
			{
				Mout += -J(i,j)*std::sqrt(3.)*(Operator::prod(Sdag(i),S(j),{1}));
			}
		}
	}
	if (U.sum() != 0. and U.sum() != std::numeric_limits<double>::infinity())
	{
		for (int i=0; i<N_orbitals; ++i) {Mout += U(i)*d(i);}
	}
	if (mu != 0.)
	{
		for (int i=0; i<N_orbitals; ++i) {Mout += (-mu)*n(i);}
	}

	return Mout;
}

} //end namespace fermions
#endif
