#ifndef BASE_U1_H_
#define BASE_U1_H_

#include "symmetry/U1.h"
#include "symmetry/qbasis.h"
#include "tensors/SiteOperatorQ.h"

namespace spins {
	
/** \class BaseU1
  * \ingroup Spins
  *
  * This class provides the local operators for spins (magnitude \p D) in a U(1) block representation for \p N_Orbitals sites.
  *
  * \describe_Scalar
  *
  */
template<typename Scalar=double>
class BaseU1
{
	typedef Eigen::Index Index;
	typedef typename Sym::U1<Scalar> Symmetry;
	typedef SiteOperatorQ<Symmetry,Scalar> Operator;
	typedef typename Symmetry::qType qType;
	
public:
	BaseU1(){};
	
	/**
	\param L_input : amount of sites
	\param D_input : \f$D=2S+1\f$*/
	BaseU1 (std::size_t L_input, std::size_t D_input);

	/**amount of states = \f$D^L\f$*/
	inline std::size_t dim() const {return N_states;}
	
	/**\f$D=2S+1\f$*/
	inline std::size_t get_D() const {return D;}
	
	/**amount of orbitals*/
	inline std::size_t orbitals() const  {return N_orbitals;}

	Operator Sz( std::size_t orbital=0 ) const;
	Operator Splus( std::size_t orbital=0 ) const;
	Operator Sminus( std::size_t orbital=0 ) const;
	Operator Id() const;
	
	Operator HeisenbergHamiltonian( double J, double Bz, bool PERIODIC=false ) const;

	/**Returns the basis. 
	   \note Use this as input for Mps, Mpo classes.*/ 
	std::vector<typename Symmetry::qType> qloc() const { return TensorBasis.qloc(); }

	/**Returns the degeneracy vector of the basis. 
	   \note Use this as input for Mps, Mpo classes.*/ 
	std::vector<Eigen::Index> qlocDeg() const { return TensorBasis.qlocDeg(); }

	Qbasis<Symmetry> basis() const { return TensorBasis; }
	
private:

	Qbasis<Symmetry> basis_1s; //basis for one site
	Qbasis<Symmetry> TensorBasis; //Final basis for N_orbital sites

	//operators defined on one orbital
	Operator Id_1s; //identity
	Operator Sz_1s; //spin z
	Operator Splus_1s; //spin plus
	Operator Sminus_1s; //spin minus

	std::size_t N_orbitals;
	std::size_t N_states;
	std::size_t D;
};

template<typename Scalar>
BaseU1<Scalar>::
BaseU1 (std::size_t L_input, std::size_t D_input)
:N_orbitals(L_input), D(D_input)
{
	assert(N_orbitals>=1 and D>=2);
	N_states = std::pow(D,N_orbitals);

	//create basis for one spin with quantumnumber D
	qType Q = {static_cast<int>(D)-1};
	Scalar locS = 0.5*static_cast<Scalar>(Q[0]-1);
	Index inner_dim = 1;
	std::vector<std::string> ident; ident.push_back("up");
	basis_1s.push_back(Q,inner_dim,ident);
	ident.clear();
	Q = {-(static_cast<int>(D)-1)};
	inner_dim = 1;
	ident.push_back("down");
	basis_1s.push_back(Q,inner_dim,ident);

	Id_1s = Operator({0},basis_1s);
	Sz_1s = Operator({0},basis_1s);
	Splus_1s = Operator({2},basis_1s);

	//create operators for one orbital
	Id_1s( "up", "up" ) = 1.;
	Id_1s( "down", "down" ) = 1.;
	Sz_1s( "up", "up" ) = 0.5;
	Sz_1s( "down", "down" ) = -0.5;
	Splus_1s( "up", "down" ) = 1.;
	Sminus_1s = Splus_1s.adjoint();

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

template<typename Scalar>
SiteOperatorQ<Sym::U1<Scalar>,Scalar> BaseU1<Scalar>::
Sz( std::size_t orbital ) const
{
	if(N_orbitals == 1) { return Sz_1s; }
	else
	{
		Operator out;
		bool TOGGLE=false;
		if(orbital == 0) { out = Operator::outerprod(Sz_1s,Id_1s,{0}); TOGGLE=true; }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,Sz_1s,{0}); TOGGLE=true; }
			else { out = Operator::outerprod(Id_1s,Id_1s,{0}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,Sz_1s,{0}); TOGGLE=true; }
			else if(TOGGLE==false) { out = Operator::outerprod(out,Id_1s,{0}); }
			else if(TOGGLE==true) { out = Operator::outerprod(out,Id_1s,{0}); }
		}
		return out;
	}
}

template<typename Scalar>
SiteOperatorQ<Sym::U1<Scalar>,Scalar> BaseU1<Scalar>::
Splus( std::size_t orbital ) const
{
	if(N_orbitals == 1) { return Splus_1s; }
	else
	{
		Operator out;
		bool TOGGLE=false;
		if(orbital == 0) { out = Operator::outerprod(Splus_1s,Id_1s,{2}); TOGGLE=true; }
		else
		{
			if( orbital == 1 ) { out = Operator::outerprod(Id_1s,Splus_1s,{2}); TOGGLE=true; }
			else { out = Operator::outerprod(Id_1s,Id_1s,{0}); }
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if(orbital == o) { out = Operator::outerprod(out,Sz_1s,{2}); TOGGLE=true; }
			else if(TOGGLE==false) { out = Operator::outerprod(out,Id_1s,{0}); }
			else if(TOGGLE==true) { out = Operator::outerprod(out,Id_1s,{2}); }
		}
		return out;
	}
}

template<typename Scalar>
SiteOperatorQ<Sym::U1<Scalar>,Scalar> BaseU1<Scalar>::
Sminus( std::size_t orbital ) const
{
	return Splus(orbital).adjoint();
}

template<typename Scalar>
SiteOperatorQ<Sym::U1<Scalar>,Scalar> BaseU1<Scalar>::
Id() const
{
	if(N_orbitals == 1) { return Id_1s; }
	else
	{
		Operator out = Operator::outerprod(Id_1s,Id_1s,{0});
		for(std::size_t o=2; o<N_orbitals; o++) { out = Operator::outerprod(out,Id_1s,{0}); }
		return out;
	}
}

template<typename Scalar>
SiteOperatorQ<Sym::U1<Scalar>,Scalar> BaseU1<Scalar>::
HeisenbergHamiltonian (double J, double Bz, bool PERIODIC) const
{	
	Operator Mout({1},TensorBasis);

	if( N_orbitals >= 2 and J!=0. )
	{
		Mout = -J * (Operator::prod(Sz(0),Sz(1),{0}) + 0.5*(Operator::prod(Splus(0),Sminus(1),{0})+Operator::prod(Sminus(0),Splus(1),{0})));
	}

	for (int i=1; i<N_orbitals-1; ++i) // for all bonds
	{
		if (J != 0.)
		{
			Mout = -J * (Operator::prod(Sz(i),Sz(i+1),{0}) + 0.5*(Operator::prod(Splus(i),Sminus(i+1),{0})+Operator::prod(Sminus(i),Splus(i+1),{0})));
		}
	}
	if (PERIODIC == true and N_orbitals>2)
	{
		if (J != 0.)
		{
			Mout = -J * (Operator::prod(Sz(N_orbitals-1),Sz(0),{0}) +
						  0.5*(Operator::prod(Splus(N_orbitals-1),Sminus(0),{0})+Operator::prod(Sminus(N_orbitals-1),Splus(0),{0})));
		}
	}
	if (Bz != 0.)
	{
		for (int i=0; i<N_orbitals;i++)
		{
			Mout += Bz * Sz(i);
		}
	}
	return Mout;
}

} //end namespace spins
#endif
