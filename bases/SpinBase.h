#ifndef SPINBASE_H_
#define SPINBASE_H_

#include "DmrgTypedefs.h"

#include "tensors/SiteOperatorQ.h"
#include "sites/SpinSite.h"
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
/// \endcond

#include "symmetry/kind_dummies.h"
#include "DmrgTypedefs.h" // for SPIN_INDEX, SPINOP_LABEL
#include "tensors/SiteOperator.h"
#include "DmrgExternal.h" // for posmod

//Note: Don't put a name in this documentation with \class .. because doxygen gets confused with template symbols
/** 
 * \ingroup Bases
 *
 * This class provides the local operators for fermions in a SU(2)⊗U(1) block representation.
 *
 */
template<typename Symmetry_, size_t order=0>
class SpinBase : public SpinSite<Symmetry_,order>
{
	typedef Eigen::Index Index;
	typedef double Scalar;
	
public:
	
	typedef Symmetry_ Symmetry;
	typedef SiteOperatorQ<Symmetry,Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> > OperatorType;
	typedef typename Symmetry::qType qType;
	
	SpinBase(){};
	
	/**
	 * \param L_input : the amount of orbitals
	 * \param U_IS_INFINITE : if \p true, eliminates doubly-occupied sites from the basis
	 */
	SpinBase (std::size_t L_input, std::size_t D_input);
	
	/**amount of states*/
	inline Index dim() const {return static_cast<Index>(N_states);}
	
	/**amount of orbitals*/
	inline std::size_t orbitals() const  {return N_orbitals;}

	/**\f$D=2S+1\f$*/
	inline size_t get_D() const {return this->D;}
	
	/**
	 * Fermionic sign for the hopping between two orbitals of nearest-neighbour supersites of a ladder.
	 * \param orb1 : orbital on supersite i
	 * \param orb2 : orbital on supersite i+1
	 */
	OperatorType sign (std::size_t orb1=0, std::size_t orb2=0) const;

	/**
	 * Occupation number operator
	 * \param orbital : orbital index
	 */
	template<class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),OperatorType>::type n (std::size_t orbital=0) const;
	
	///\{
	/**
	 * Orbital spin
	 * \param orbital : orbital index
	 */
	template<class Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(),OperatorType>::type S (size_t orbital=0) const;
		
	/**
	 * Orbital spin† 
	 * \param orbital : orbital index
	 */
	template<class Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(),OperatorType>::type Sdag (size_t orbital=0) const;
	
	template<class Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(),OperatorType>::type Q (size_t orbital=0) const;
	
	template<class Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(),OperatorType>::type Qdag (size_t orbital=0) const;
	
	template<class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),OperatorType>::type Qz (size_t orbital=0) const;
	
	template<class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),OperatorType>::type Qp (size_t orbital=0) const;
	
	template<class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),OperatorType>::type Qm (size_t orbital=0) const;
	
	template<class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),OperatorType>::type Qpz (size_t orbital=0) const;
	
	template<class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),OperatorType>::type Qmz (size_t orbital=0) const;
	
	template<class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),OperatorType>::type Sz (size_t orbital=0) const;
	
	template<class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),OperatorType>::type Sp (size_t orbital=0) const;
	
	template<class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),OperatorType>::type Sm (size_t orbital=0) const;
	
	template<class Dummy = Symmetry>
	typename std::enable_if<Dummy::NO_SPIN_SYM(),OperatorType>::type Sx (size_t orbital=0) const;
	
	template<class Dummy = Symmetry>
	typename std::enable_if<Dummy::NO_SPIN_SYM(),OperatorType>::type iSy (size_t orbital=0) const;
	
	template<class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),OperatorType>::type Scomp (SPINOP_LABEL Sa, int orbital=0) const
	{
		assert(Sa != SY and Sa != QP and Sa != QM and Sa != QPZ and Sa != QMZ);
		OperatorType out;
		if constexpr (Dummy::NO_SPIN_SYM())
		{
			if      (Sa==SX)  { out = Sx(orbital); }
			else if (Sa==iSY) { out = iSy(orbital); }
			else if (Sa==SZ)  { out = Sz(orbital); }
			else if (Sa==SP)  { out = Sp(orbital); }
			else if (Sa==SM)  { out = Sm(orbital); }
		}
		else
		{
			if      (Sa==SZ)  { out = Sz(orbital); }
			else if (Sa==SP)  { out = Sp(orbital); }
			else if (Sa==SM)  { out = Sm(orbital); }
		}
		return out;
	};
	
	template<class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),OperatorType>::type Qcomp (SPINOP_LABEL Sa, int orbital=0) const
	{
		assert(Sa != SX and Sa != SY and Sa != iSY and Sa != SZ and Sa != SP and Sa != SM);
		OperatorType out;
		if      (Sa==QZ)  {out = Qz(orbital);}
		else if (Sa==QP)  {out = Qp(orbital);}
		else if (Sa==QM)  {out = Qm(orbital);}
		else if (Sa==QPZ) {out = Qpz(orbital);}
		else if (Sa==QMZ) {out = Qmz(orbital);}
		return out;
	};
	///\}

	template<class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),SiteOperatorQ<Symmetry,Eigen::MatrixXcd> >::type Rcomp (SPINOP_LABEL Sa, int orbital=0) const;
	
	template<class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),OperatorType>::type bead (STRING STR, size_t orbital=0) const;
	
	/**Identity*/
	OperatorType Id (std::size_t orbital=0) const;

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
	template<typename Scalar_>
	SiteOperatorQ<Symmetry_,Eigen::Matrix<Scalar_,-1,-1> > HeisenbergHamiltonian (const Array<Scalar_,Dynamic,Dynamic> &J) const;
	
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), OperatorType>::type HeisenbergHamiltonian (const ArrayXXd &Jxy, const ArrayXXd &Jz, 
	                                                                                          const ArrayXd &Bz, const ArrayXd &mu, const ArrayXd &nu,
	                                                                                          const ArrayXd &Kz) const;
	
	/**
	 * Creates the full Heisenberg (XXZ) Hamiltonian on the supersite.
	 * \param Jxy : \f$J^{xy}\f$
	 * \param Jz : \f$J^{z}\f$
	 * \param Bz : \f$B^{z}_i\f$
	 * \param Bx : \f$B^{x}_i\f$
	 * \param mu : \f$\mu\f$ (for spinless fermions, couples to n=1/2-Sz)
	 * \param Kz : \f$K^{z}_i\f$
	 * \param Kx : \f$K^{x}_i\f$
	 * \param Dy : \f$D^{y}\f$
	 */
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::NO_SPIN_SYM(), OperatorType>::type HeisenbergHamiltonian (const ArrayXXd &Jxy, const ArrayXXd &Jz, 
	                                                                                         const ArrayXd &Bz, const ArrayXd &Bx, const ArrayXd &mu, const ArrayXd &nu,
	                                                                                         const ArrayXd &Kz, const ArrayXd &Kx, 
	                                                                                         const ArrayXXd &Dy) const;
	/**
	 * Creates the full Heisenberg (XYZ) Hamiltonian on the supersite.
	 * \param J : \f$J^{\alpha}\f$, \f$\alpha \in \{x,y,z\} \f$
	 * \param B : \f$B^{\alpha}_i\f$, \f$\alpha \in \{x,y,z\} \f$
	 * \param K : \f$K^{\alpha}_i\f$, \f$\alpha \in \{x,y,z\} \f$
	 * \param D : \f$D^{\alpha}\f$, \f$\alpha \in \{x,y,z\} \f$
	 */	
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::NO_SPIN_SYM(),SiteOperatorQ<Symmetry_,Eigen::Matrix<complex<double>,-1,-1> > >::type HeisenbergHamiltonian (const std::array<ArrayXXd,3> &J, 
	                                                                                                                                           const std::array<ArrayXd,3> &B,
	                                                                                                                                           const std::array<ArrayXd,3> &K,
	                                                                                                                                           const std::array<ArrayXXd,3> &D) const;
	template<typename Scalar_, typename Dummy = Symmetry>
	typename std::enable_if<Dummy::NO_SPIN_SYM(),SiteOperatorQ<Symmetry_,Eigen::Matrix<Scalar_,-1,-1> >>::type coupling_Bx (const Array<double,Dynamic,1> &Bx) const;
	
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::NO_SPIN_SYM(),SiteOperatorQ<Symmetry_,Eigen::Matrix<complex<double>,-1,-1> >>::type coupling_By (const Array<double,Dynamic,1> &By) const;
	
	/**Returns the basis.*/
	Qbasis<Symmetry> get_basis() const { return TensorBasis; }
	
private:
	OperatorType make_operator(const OperatorType &Op_1s, size_t orbital=0, string label="") const;
	std::size_t N_orbitals;
	std::size_t N_states;
	
	Qbasis<Symmetry> TensorBasis; //Final basis for N_orbital sites
};

template <typename Symmetry_, size_t order>
SpinBase<Symmetry_,order>::
SpinBase (std::size_t L_input, std::size_t D_input)
:SpinSite<Symmetry,order>(D_input), N_orbitals(L_input)
{
	//create basis for spin 0
	typename Symmetry::qType Q=Symmetry::qvacuum();
	Eigen::Index inner_dim = 1;
	Qbasis<Symmetry_> vacuum;
	vacuum.push_back(Q, inner_dim);
	
	// // create operators for zero orbitals
	// Zero_vac = OperatorType(Symmetry::qvacuum(), vacuum);
	// Zero_vac.setZero();
	// Id_vac = OperatorType(Symmetry::qvacuum(), vacuum);
	// Id_vac.setIdentity();
	
	// create basis for N_orbitals fermionic sites
	if      (N_orbitals == 1) {TensorBasis = this->basis_1s();}
	else if (N_orbitals == 0) {TensorBasis = vacuum;}
	else
	{
		TensorBasis = this->basis_1s().combine(this->basis_1s());
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			TensorBasis = TensorBasis.combine(this->basis_1s());
		}
	}

	N_states = TensorBasis.size();
}

template<typename Symmetry_, size_t order>
SiteOperatorQ<Symmetry_, Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > SpinBase<Symmetry_,order>::
make_operator (const OperatorType &Op_1s, size_t orbital, string label) const
{
	OperatorType out;
	if (N_orbitals == 1) {out = Op_1s; out.label() = label; return out;}
	else
	{
		OperatorType stringOp = this->Id_1s();;
		bool TOGGLE=false;
		if (orbital == 0) {out = OperatorType::outerprod(Op_1s,this->Id_1s(),Op_1s.Q()); TOGGLE=true;}
		else
		{
			if (orbital == 1) {out = OperatorType::outerprod(stringOp,Op_1s,Op_1s.Q()); TOGGLE=true;}
			else {out = OperatorType::outerprod(stringOp,stringOp,Symmetry_::qvacuum());}
		}
		for(std::size_t o=2; o<N_orbitals; o++)
		{
			if      (orbital == o)  {out = OperatorType::outerprod(out,Op_1s,Op_1s.Q()); TOGGLE=true; }
			else if (TOGGLE==false) {out = OperatorType::outerprod(out,stringOp,Symmetry_::qvacuum());}
			else if (TOGGLE==true)  {out = OperatorType::outerprod(out,this->Id_1s(),Op_1s.Q());}
		}
		out.label() = label;
		return out;
	}
}

template<typename Symmetry_, size_t order>
SiteOperatorQ<Symmetry_,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > SpinBase<Symmetry_,order>::
sign (std::size_t orb1, std::size_t orb2) const
{
	OperatorType Oout;
	if (N_orbitals == 1) {Oout = this->F_1s(); Oout.label()="sign"; return Oout;}
	else
	{
		Oout = Id();
		for (int i=orb1; i<N_orbitals; ++i)
		{
			Oout = Oout * (2.*Sz(i));
		}
		for (int i=0; i<orb2; ++i)
		{
			Oout = Oout * (2.*Sz(i));
		}
		Oout.label() = "sign";
		return Oout;
	}
}

template<typename Symmetry_, size_t order>
template <typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), SiteOperatorQ<Symmetry_,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > >::type SpinBase<Symmetry_,order>::
n (std::size_t orbital) const
{
	OperatorType Oout = 0.5*Id() - Sz(orbital);
	Oout.label() = "n";
	return Oout;
}

template<typename Symmetry_, size_t order>
template <typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), SiteOperatorQ<Symmetry_,Eigen::Matrix<complex<double>,Eigen::Dynamic,Eigen::Dynamic> > >::type SpinBase<Symmetry_,order>::
Rcomp (SPINOP_LABEL Sa, int orbital) const
{
	assert(orbital<N_orbitals);
	SiteOperatorQ<Symmetry_,Eigen::Matrix<complex<double>,Eigen::Dynamic,Eigen::Dynamic> > Oout;
	SiteOperatorQ<Symmetry_,Eigen::Matrix<complex<double>,Eigen::Dynamic,Eigen::Dynamic> > Scomp_cmplx = Scomp(Sa,orbital).template cast<complex<double> >();
	if (Sa==iSY)
	{
		Oout = 2.*M_PI*Scomp_cmplx;
	}
	else
	{
		Oout = 2.*1.i*M_PI*Scomp_cmplx;
	}
	Oout.data() = Oout.data().exp(1.);
	
	cout << "Rcomp=" << Oout << endl << endl;
	// cout << "Re=" << Mtmp.exp().real() << endl << endl;
	// cout << "Im=" << Mtmp.exp().imag() << endl << endl;
	// cout << "Op=" << Op << endl << endl;
	
	return Oout; //SiteOperator<Symmetry,complex<double>>(Op,getQ(Sa));
}

template<typename Symmetry_, size_t order>
template <typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), SiteOperatorQ<Symmetry_,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > >::type SpinBase<Symmetry_,order>::
bead (STRING STR, std::size_t orbital) const
{
	if (STR == STRINGZ)
	{
		if (this->D%2==0)
		{
			lout << termcolor::red << "Warning: exp(iπSz) is not correct for half-integer S!" << termcolor::reset << endl;
		}
		return make_operator(this->exp_i_pi_Sz(), orbital, "exp(iπSz)");
	}
	else if (STR == STRINGX)
	{
		if (this->D%2==0)
		{
			lout << termcolor::red << "Warning: exp(iπSx) is not correct for half-integer S!" << termcolor::reset << endl;
		}
		return make_operator(this->exp_i_pi_Sx(), orbital, "exp(iπSx)");
	}
	else
	{
		if (this->D%2==0)
		{
			lout << termcolor::red << "Warning: exp(iπSy) is not correct for half-integer S!" << termcolor::reset << endl;
		}
		return make_operator(this->exp_i_pi_Sy(), orbital, "exp(iπSy)");
	}
}

template<typename Symmetry_, size_t order>
template <typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), SiteOperatorQ<Symmetry_,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > >::type SpinBase<Symmetry_,order>::
S (std::size_t orbital) const
{
	return make_operator(this->S_1s(), orbital,"S");
}

template<typename Symmetry_, size_t order>
template <typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), SiteOperatorQ<Symmetry_,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > >::type SpinBase<Symmetry_,order>::
Sdag (std::size_t orbital) const
{
	return S(orbital).adjoint();
}

template<typename Symmetry_, size_t order>
template <typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), SiteOperatorQ<Symmetry_,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > >::type SpinBase<Symmetry_,order>::
Q (std::size_t orbital) const
{
	return make_operator(this->Q_1s(), orbital,"Q");
}

template<typename Symmetry_, size_t order>
template <typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), SiteOperatorQ<Symmetry_,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > >::type SpinBase<Symmetry_,order>::
Qdag (std::size_t orbital) const
{
	return Q(orbital).adjoint();
}

template<typename Symmetry_, size_t order>
template <typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), SiteOperatorQ<Symmetry_,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > >::type SpinBase<Symmetry_,order>::
Qz (std::size_t orbital) const
{
	return make_operator(this->Qz_1s(), orbital,"Qz");
}

template<typename Symmetry_, size_t order>
template <typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), SiteOperatorQ<Symmetry_,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > >::type SpinBase<Symmetry_,order>::
Qp (std::size_t orbital) const
{
	return make_operator(this->Qp_1s(), orbital,"Qp");
}

template<typename Symmetry_, size_t order>
template <typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), SiteOperatorQ<Symmetry_,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > >::type SpinBase<Symmetry_,order>::
Qm (std::size_t orbital) const
{
	return make_operator(this->Qm_1s(), orbital,"Qm");
}

template<typename Symmetry_, size_t order>
template <typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), SiteOperatorQ<Symmetry_,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > >::type SpinBase<Symmetry_,order>::
Qpz (std::size_t orbital) const
{
	return make_operator(this->Qpz_1s(), orbital,"Qpz");
}

template<typename Symmetry_, size_t order>
template <typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), SiteOperatorQ<Symmetry_,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > >::type SpinBase<Symmetry_,order>::
Qmz (std::size_t orbital) const
{
	return make_operator(this->Qmz_1s(), orbital,"Qmz");
}

template<typename Symmetry_, size_t order>
template <typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), SiteOperatorQ<Symmetry_,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > >::type SpinBase<Symmetry_,order>::
Sz (std::size_t orbital) const
{
	return make_operator(this->Sz_1s(), orbital,"Sz");
}

template<typename Symmetry_, size_t order>
template <typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), SiteOperatorQ<Symmetry_,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > >::type SpinBase<Symmetry_,order>::
Sp (std::size_t orbital) const
{
	return make_operator(this->Sp_1s(), orbital,"Sp");
}

template<typename Symmetry_, size_t order>
template <typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), SiteOperatorQ<Symmetry_,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > >::type SpinBase<Symmetry_,order>::
Sm (std::size_t orbital) const
{
	return make_operator(this->Sm_1s(), orbital,"Sm");
}

template<typename Symmetry_, size_t order>
template <typename Dummy>
typename std::enable_if<Dummy::NO_SPIN_SYM(), SiteOperatorQ<Symmetry_,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > >::type SpinBase<Symmetry_,order>::
Sx (std::size_t orbital) const
{
	OperatorType out = 0.5 * (Sp(orbital) + Sm(orbital));
	out.label() = "Sx";
	return out;
}

template<typename Symmetry_, size_t order>
template <typename Dummy>
typename std::enable_if<Dummy::NO_SPIN_SYM(), SiteOperatorQ<Symmetry_,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > >::type SpinBase<Symmetry_,order>::
iSy (std::size_t orbital) const
{
	OperatorType out = 0.5 * (Sp(orbital) - Sm(orbital));
	out.label() = "Sy";
	return out;
}

template<typename Symmetry_, size_t order>
SiteOperatorQ<Symmetry_,Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> > SpinBase<Symmetry_,order>::
Id (std::size_t orbital) const
{
	return make_operator(this->Id_1s(), orbital,"Id");
}

template<typename Symmetry_, size_t order>
template<typename Scalar_>
SiteOperatorQ<Symmetry_, Eigen::Matrix<Scalar_,Eigen::Dynamic,Eigen::Dynamic> > SpinBase<Symmetry_,order>::
HeisenbergHamiltonian (const Array<Scalar_,Dynamic,Dynamic> &J) const
{
	SiteOperatorQ<Symmetry_, Eigen::Matrix<Scalar_,Eigen::Dynamic,Eigen::Dynamic> > Oout(Symmetry::qvacuum(),TensorBasis);
	
	for (int i=0; i<N_orbitals; ++i)
	for (int j=0; j<N_orbitals; ++j)
	{
		if (J(i,j) != 0.)
		{
			if constexpr (Symmetry::IS_SPIN_SU2())
			{
				Oout += J(i,j)*std::sqrt(3.) * (OperatorType::prod(Sdag(i),S(j),Symmetry::qvacuum())).template cast<Scalar_>();
			}
			else
			{
				Oout += J(i,j) * (OperatorType::prod(Sz(i),Sz(j),Symmetry::qvacuum())).template cast<Scalar_>();
				Oout += 0.5*J(i,j) * (OperatorType::prod(Sp(i),Sm(j),Symmetry::qvacuum()) + OperatorType::prod(Sm(i),Sp(j),Symmetry::qvacuum())).template cast<Scalar_>();
			}
		}
	}
	
	Oout.label() = "Hloc";
	return Oout;
}

template<typename Symmetry_, size_t order>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), SiteOperatorQ<Symmetry_,Eigen::Matrix<double,-1,-1> > >::type SpinBase<Symmetry_,order>::
HeisenbergHamiltonian (const ArrayXXd &Jxy, const ArrayXXd &Jz, const ArrayXd &Bz, const ArrayXd &mu, const ArrayXd &nu, const ArrayXd &Kz) const
{
	assert(Bz.rows() == N_orbitals and Kz.rows() == N_orbitals);
	
	OperatorType Mout(Symmetry::qvacuum(),TensorBasis);
	
	for (int i=0; i<N_orbitals; ++i)
	for (int j=0; j<N_orbitals; ++j)
	{
		if (Jxy(i,j) != 0.)
		{
			Mout += 0.5*Jxy(i,j) * (Scomp(SP,i) * Scomp(SM,j) + Scomp(SM,i) * Scomp(SP,j));
		}
		if (Jz(i,j) != 0.)
		{
			Mout += Jz(i,j) * Scomp(SZ,i)*Scomp(SZ,j);
		}
	}
	
	for (int i=0; i<N_orbitals; ++i)
	{
		if (Bz(i) != 0.) {Mout -= Bz(i) * Scomp(SZ,i);}
	}
	for (int i=0; i<N_orbitals; ++i)
	{
		if (mu(i) != 0.) {Mout += mu(i) * (Scomp(SZ,i)-0.5*Id());} // for Kitaev chain: -mu*n = -mu*(1/2-Sz) = mu*(Sz-1/2)
	}
	for (int i=0; i<N_orbitals; ++i)
	{
		if (nu(i) != 0.) {Mout += nu(i) * (Scomp(SZ,i)+0.5*Id());}
	}
	for (int i=0; i<N_orbitals; ++i)
	{
		if (Kz(i)!=0.) {Mout += Kz(i) * Scomp(SZ,i) * Scomp(SZ,i);}
	}
	Mout.label() = "Hloc";
	return Mout;
}

template<typename Symmetry_, size_t order>
template<typename Dummy>
typename std::enable_if<Dummy::NO_SPIN_SYM(), SiteOperatorQ<Symmetry_,Eigen::Matrix<double,-1,-1> > >::type SpinBase<Symmetry_,order>::
HeisenbergHamiltonian (const ArrayXXd &Jxy, const ArrayXXd &Jz, 
                       const ArrayXd &Bz, const ArrayXd &Bx, 
                       const ArrayXd &mu, const ArrayXd &nu,
                       const ArrayXd &Kz, const ArrayXd &Kx,
                       const ArrayXXd &Dy) const
{
	assert(Bz.rows() == N_orbitals and Bx.rows() == N_orbitals and Kz.rows() == N_orbitals and Kx.rows() == N_orbitals);
	
	OperatorType Mout = HeisenbergHamiltonian(Jxy, Jz, Bz, mu, nu, Kz);
	
	for (int i=0; i<N_orbitals; ++i)
	for (int j=0; j<i; ++j)
	{
		if (Dy(i,j) != 0.) {Mout += Dy(i,j) * (Scomp(SX,i)*Scomp(SZ,j) - Scomp(SZ,i)*Scomp(SX,j));}
	}	
	for (int i=0; i<N_orbitals; ++i)
	{
		if (Bx(i) != 0.) {Mout -= Bx(i) * Scomp(SX,i);}
	}
	for (int i=0; i<N_orbitals; ++i)
	{
		if (Kx(i)!=0.) {Mout += Kx(i) * Scomp(SX,i) * Scomp(SX,i);}
	}
	
	return Mout;
}

template<typename Symmetry_, size_t order>
template<typename Scalar_, typename Dummy>
typename std::enable_if<Dummy::NO_SPIN_SYM(),SiteOperatorQ<Symmetry_,Eigen::Matrix<Scalar_,-1,-1> > >::type SpinBase<Symmetry_,order>::
coupling_Bx (const Array<double,Dynamic,1> &Bx) const
{
	SiteOperatorQ<Symmetry_,Eigen::Matrix<Scalar_,-1,-1> > Mout(Symmetry::qvacuum(), TensorBasis);
	for (int i=0; i<N_orbitals; ++i)
	{
		if (Bx(i) != 0.)
		{
			Mout -= Bx(i) * Sx(i).template cast<Scalar_>();
		}
	}
	return Mout;
}

template<typename Symmetry_, size_t order>
template<typename Dummy>
typename std::enable_if<Dummy::NO_SPIN_SYM(),SiteOperatorQ<Symmetry_,Eigen::Matrix<complex<double>,-1,-1> >>::type SpinBase<Symmetry_,order>::
coupling_By (const Array<double,Dynamic,1> &By) const
{
	SiteOperatorQ<Symmetry_,Eigen::Matrix<complex<double>,-1,-1> > Mout(Symmetry::qvacuum(), TensorBasis);
	for (int i=0; i<N_orbitals; ++i)
	{
		if (By(i) != 0.)
		{
			Mout -= -1i*By(i) * iSy(i).template cast<complex<double> >();
		}
	}
	return Mout;
}

template<typename Symmetry_, size_t order>
template<typename Dummy>
typename std::enable_if<Dummy::NO_SPIN_SYM(),SiteOperatorQ<Symmetry_,Eigen::Matrix<complex<double>,-1,-1> > >::type SpinBase<Symmetry_,order>::
HeisenbergHamiltonian (const std::array<ArrayXXd,3> &J, const std::array<ArrayXd,3> &B, const std::array<ArrayXd,3> &K, const std::array<ArrayXXd,3> &D) const
{
	SiteOperatorQ<Symmetry_,Eigen::Matrix<complex<double>,-1,-1> > Mout(Symmetry::qvacuum(), TensorBasis);
	for (int i=0; i<N_orbitals; ++i)
	for (int j=0; j<N_orbitals; ++j)
	{
		//J
		if (J[0](i,j) != 0.) {Mout += J[0](i,j) * (Sx(i) * Sx(j)).template cast<complex<double>>();}
		if (J[1](i,j) != 0.) {Mout += -J[1](i,j) * (iSy(i) * iSy(j)).template cast<complex<double>>();}
		if (J[2](i,j) != 0.) {Mout += J[2](i,j) * (Sz(i) * Sz(j)).template cast<complex<double>>();}

		//D
		if (D[0](i,j) != 0.) {Mout += D[0](i,j) * (-1.i) * (iSy(i) * Sz(j) - Sz(i)  * iSy(j)).template cast<complex<double> >();}
		if (D[1](i,j) != 0.) {Mout += D[1](i,j) * (Sx(i) * Sz(j) - Sz(i) * Sx(j)).template cast<complex<double> >();}
		if (D[2](i,j) != 0.) {Mout += D[2](i,j) * (-1.i) * (Sx(i) * iSy(j) - iSy(i)  * Sx(j)).template cast<complex<double> >();}
	}

	for (int i=0; i<N_orbitals; ++i)
	{
		// B
		if (B[0](i) != 0.) {Mout -= B[0](i)          *  Sx(i).template cast<complex<double> >();}
		if (B[1](i) != 0.) {Mout -= B[1](i) * (-1.i) * iSy(i).template cast<complex<double> >();}
		if (B[2](i) != 0.) {Mout -= B[2](i)          *  Sz(i).template cast<complex<double> >();}
		
		// K
		if (K[0](i) != 0.) {Mout += K[0](i) * ( Sx(i) *  Sx(i)).template cast<complex<double> >();}
		if (K[1](i) != 0.) {Mout -= K[1](i) * (iSy(i) * iSy(i)).template cast<complex<double> >();}
		if (K[2](i) != 0.) {Mout += K[2](i) * ( Sz(i) *  Sz(i)).template cast<complex<double> >();}
	}
	Mout.label() = "Hloc";
	return Mout;
}
#endif
