#ifndef HUBBARDOBSERVABLES
#define HUBBARDOBSERVABLES

#include "bases/FermionBase.h"
#include "Mpo.h"
#include "ParamHandler.h" // from HELPERS

//include "DmrgLinearAlgebra.h"
//include "DmrgExternal.h"
//include "tensors/SiteOperator.h"

template<typename Symmetry>
class HubbardObservables
{
	typedef SiteOperatorQ<Symmetry,Eigen::MatrixXd> OperatorType;
	
public:
	
	///@{
	HubbardObservables(){};
	HubbardObservables (const size_t &L); // for inheritance purposes
	HubbardObservables (const size_t &L, const vector<Param> &params, const std::map<string,std::any> &defaults);
	///@}
	
	///@{
	template<class Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type c (size_t locx, size_t locy=0, double factor=1.) const;

	template<SPIN_INDEX sigma, class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type c (size_t locx, size_t locy=0, double factor=1.) const;

	// template<SPIN_INDEX sigma, SUB_LATTICE G=A, class Dummy = Symmetry>
	// typename std::enable_if<Dummy::IS_CHARGE_SU2() and !Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type c (size_t locx, size_t locy=0, double factor=1.) const;

	// template<SUB_LATTICE G=A, class Dummy = Symmetry>
	// typename std::enable_if<Dummy::IS_CHARGE_SU2() and Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type c (size_t locx, size_t locy=0, double factor=1.) const;

	template<class Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(),Mpo<Symmetry>>::type cdag (size_t locx, size_t locy=0, double factor=std::sqrt(2.)) const;

	template<SPIN_INDEX sigma, class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type cdag (size_t locx, size_t locy=0, double factor=1.) const;

	// template<SPIN_INDEX sigma, SUB_LATTICE G=A, class Dummy = Symmetry>
	// typename std::enable_if<Dummy::IS_CHARGE_SU2() and !Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type cdag (size_t locx, size_t locy=0, double factor=std::sqrt(2.)) const;

	// template<SUB_LATTICE G=A, class Dummy = Symmetry>
	// typename std::enable_if<Dummy::IS_CHARGE_SU2() and Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type cdag (size_t locx, size_t locy=0, double factor=std::sqrt(2.)*std::sqrt(2.)) const;
	///@}
	
	///@{
	template<class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_CHARGE_SU2(),Mpo<Symmetry> >::type cc (size_t locx, size_t locy=0) const;
	template<class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_CHARGE_SU2(),Mpo<Symmetry> >::type cdagcdag (size_t locx, size_t locy=0) const;

	template<typename Dummy = Symmetry>
	typename std::conditional<Dummy::IS_SPIN_SU2(),Mpo<Symmetry>, vector<Mpo<Symmetry> > >::type cdagc (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	
	template<SPIN_INDEX sigma, typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type cdagc (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;

	// template<SPIN_INDEX sigma, SUB_LATTICE G=A, typename Dummy = Symmetry>
	// typename std::enable_if<Dummy::IS_CHARGE_SU2() and !Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type cdagc (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;

	// template<SUB_LATTICE G=A, typename Dummy = Symmetry>
	// typename std::enable_if<Dummy::IS_CHARGE_SU2() and Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type cdagc (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	
	// Mpo<Symmetry> triplet (size_t locx, size_t locy=0) const;
	///@}

	template<SPIN_INDEX sigma1, SPIN_INDEX sigma2, typename Dummy = Symmetry>
	typename std::enable_if<Dummy::ABELIAN,Mpo<Symmetry> >::type cdagcdag (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;

	template<SPIN_INDEX sigma1, SPIN_INDEX sigma2, typename Dummy = Symmetry>
	typename std::enable_if<Dummy::ABELIAN,Mpo<Symmetry> >::type cc (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;

	template<typename Dummy = Symmetry>
	typename std::conditional<Dummy::IS_SPIN_SU2(),Mpo<Symmetry>, vector<Mpo<Symmetry> > >::type
	cc3 (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	template<typename Dummy = Symmetry>
	typename std::conditional<Dummy::IS_SPIN_SU2(),Mpo<Symmetry>, vector<Mpo<Symmetry> > >::type
	cdagcdag3 (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	
	///@{
	template<class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_CHARGE_SU2(),Mpo<Symmetry> >::type d (size_t locx, size_t locy=0) const;
	template<class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_CHARGE_SU2(),Mpo<Symmetry> >::type dtot () const;
	Mpo<Symmetry> ns (size_t locx, size_t locy=0) const;
	Mpo<Symmetry> nh (size_t locx, size_t locy=0) const;
	Mpo<Symmetry> nssq (size_t locx, size_t locy=0) const;
	Mpo<Symmetry> nhsq (size_t locx, size_t locy=0) const;
	template<class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_CHARGE_SU2(),Mpo<Symmetry> >::type s (size_t locx, size_t locy=0) const;
	
	template<SPIN_INDEX sigma, typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type n (size_t locx, size_t locy=0) const;
	
	Mpo<Symmetry> n (size_t locx, size_t locy=0) const;
	
	template<SPIN_INDEX sigma1, SPIN_INDEX sigma2, typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type nn (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	
	Mpo<Symmetry> nn (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_CHARGE_SU2(),Mpo<Symmetry> >::type hh (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	///@}
	
	///@{
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_CHARGE_SU2(), Mpo<Symmetry> >::type T (size_t locx, size_t locy=0, double factor=1.) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_CHARGE_SU2(), Mpo<Symmetry> >::type Tdag (size_t locx, size_t locy=0, double factor=std::sqrt(3.)) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_CHARGE_SU2(), Mpo<Symmetry> >::type Tp (size_t locx, size_t locy=0) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_CHARGE_SU2(), Mpo<Symmetry> >::type Tm (size_t locx, size_t locy=0) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::NO_CHARGE_SYM(), Mpo<Symmetry> >::type Tx (size_t locx, size_t locy=0) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::NO_CHARGE_SYM(), Mpo<Symmetry> >::type iTy (size_t locx, size_t locy=0) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_CHARGE_SU2(), Mpo<Symmetry> >::type Tz (size_t locx, size_t locy=0) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_CHARGE_SU2(), Mpo<Symmetry> >::type TpTm (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_CHARGE_SU2(), Mpo<Symmetry> >::type TmTp (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_CHARGE_SU2(), Mpo<Symmetry> >::type TzTz (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	template<typename Dummy = Symmetry>
	typename std::conditional<Dummy::IS_CHARGE_SU2(), Mpo<Symmetry>, vector<Mpo<Symmetry> > >::type TdagT (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	///@}
	
	///@{
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type S (size_t locx, size_t locy=0, double factor=1.) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type Sdag (size_t locx, size_t locy=0, double factor=std::sqrt(3.)) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type Scomp (SPINOP_LABEL Sa, size_t locx, size_t locy=0, double factor=1.) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type Sz (size_t locx, size_t locy=0) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type Sp (size_t locx, size_t locy=0) const {return Scomp(SP,locx,locy);};
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type Sm (size_t locx, size_t locy=0) const {return Scomp(SM,locx,locy);};
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type ScompScomp (SPINOP_LABEL Sa1, SPINOP_LABEL Sa2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type SpSm (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const {return ScompScomp(SP,SM,locx1,locx2,locy1,locy2,fac);};
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type SmSp (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const {return ScompScomp(SM,SP,locx1,locx2,locy1,locy2,fac);};
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type SzSz (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const {return ScompScomp(SZ,SZ,locx1,locx2,locy1,locy2,1.);};
	template<typename Dummy = Symmetry>
	typename std::conditional<Dummy::IS_SPIN_SU2(), Mpo<Symmetry>, vector<Mpo<Symmetry> > >::type SdagS (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;

	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry, complex<double> > >::type Rcomp (SPINOP_LABEL Sa, size_t locx, size_t locy=0) const;
	///@}

	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type Stringz (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type StringzDimer (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;

	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,complex<double> > >::type S_ky    (vector<complex<double> > phases) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,complex<double> > >::type Sdag_ky (vector<complex<double> > phases, double factor=sqrt(3.)) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_CHARGE_SU2(),Mpo<Symmetry,complex<double> > >::type T_ky    (vector<complex<double> > phases) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_CHARGE_SU2(),Mpo<Symmetry,complex<double> > >::type Tdag_ky (vector<complex<double> > phases, double factor=sqrt(3.)) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2() and !Dummy::IS_CHARGE_SU2(),Mpo<Symmetry,complex<double> > >::type c_ky    (vector<complex<double> > phases, double factor=1.) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2() and !Dummy::IS_CHARGE_SU2(),Mpo<Symmetry,complex<double> > >::type cdag_ky (vector<complex<double> > phases, double factor=sqrt(2.)) const;
protected:
	
	Mpo<Symmetry> make_local (size_t locx, size_t locy,
	                          const OperatorType &Op,
							  double factor =1.,
	                          bool FERMIONIC=false, bool HERMITIAN=false) const;
	Mpo<Symmetry> make_corr  (size_t locx1, size_t locx2, size_t locy1, size_t locy2,
	                          const OperatorType &Op1, const OperatorType &Op2, qarray<Symmetry::Nq> Qtot,
	                          double factor, bool FERMIONIC, bool HERMITIAN) const;

	Mpo<Symmetry,complex<double> >
	make_FourierYSum (string name, const vector<OperatorType> &Ops, double factor, bool HERMITIAN, const vector<complex<double> > &phases) const;
	
	typename Symmetry::qType getQ_ScompScomp(SPINOP_LABEL Sa1, SPINOP_LABEL Sa2) const;
		
	vector<FermionBase<Symmetry> > F;
};

template<typename Symmetry>
HubbardObservables<Symmetry>::
HubbardObservables (const size_t &L)
{
	F.resize(L);
}

template<typename Symmetry>
HubbardObservables<Symmetry>::
HubbardObservables (const size_t &L, const vector<Param> &params, const std::map<string,std::any> &defaults)
{
	ParamHandler P(params,defaults);
	size_t Lcell = P.size();
	F.resize(L);
	
	for (size_t l=0; l<L; ++l)
	{
		F[l] = FermionBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), !isfinite(P.get<double>("U",l%Lcell)));
	}
}

//-------------

template<typename Symmetry>
Mpo<Symmetry> HubbardObservables<Symmetry>::
make_local (size_t locx, size_t locy, const OperatorType &Op, double factor, bool FERMIONIC, bool HERMITIAN) const
{
	assert(locx<F.size() and locy<F[locx].dim());
	stringstream ss;
	ss << Op.label() << "(" << locx << "," << locy;
	if (factor != 1.) ss << ",factor=" << factor;
	ss << ")";
	
	Mpo<Symmetry> Mout(F.size(), Op.Q(), ss.str(), HERMITIAN);
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	
	if (FERMIONIC)
	{
		vector<SiteOperator<Symmetry,double> > Signs(locx);
		for (size_t l=0; l<locx; ++l) {Signs[l] = F[l].sign().template plain<double>();}
		
		Mout.setLocal(locx, (factor * Op).template plain<double>(), Signs);
	}
	else
	{
		Mout.setLocal(locx, (factor * Op).template plain<double>());
	}
	
	return Mout;
}

template<typename Symmetry>
Mpo<Symmetry> HubbardObservables<Symmetry>::
make_corr (size_t locx1, size_t locx2, size_t locy1, size_t locy2,
           const OperatorType &Op1, const OperatorType &Op2,
		   qarray<Symmetry::Nq> Qtot,
           double factor, bool FERMIONIC, bool HERMITIAN) const	
{
	assert(locx1<F.size() and locy1<F[locx1].dim());
	assert(locx2<F.size() and locy2<F[locx2].dim());
	
	stringstream ss;
	ss << Op1.label() << "(" << locx1 << "," << locy1 << ")"
	   << Op2.label() << "(" << locx2 << "," << locy2 << ")";
	
	Mpo<Symmetry> Mout(F.size(), Qtot, ss.str(), HERMITIAN);
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	
	if (FERMIONIC)
	{
		if (locx1 == locx2)
		{
			Mout.setLocal(locx1, factor * OperatorType::prod(Op1,Op2,Qtot).template plain<double>());
		}
		else if (locx1<locx2)
		{
			Mout.setLocal({locx1, locx2}, {(factor * (Op1 * F[locx1].sign())).template plain<double>(), 
			                               Op2.template plain<double>()}, 
			                               F[0].sign().template plain<double>());
		}
		else if (locx1>locx2)
		{
			Mout.setLocal({locx2, locx1}, {(factor * (Op2 * F[locx2].sign())).template plain<double>(), 
			                               -Symmetry::spinorFactor() * Op1.template plain<double>()}, 
			                               F[0].sign().template plain<double>());
		}
	}
	else
	{
		if (locx1 == locx2)
		{
			auto product = factor*OperatorType::prod(Op1, Op2, Qtot);
			Mout.setLocal(locx1, product.template plain<double>());
		}
		else
		{
			Mout.setLocal({locx1, locx2}, {(factor*Op1).template plain<double>(), Op2.template plain<double>()});
		}
	}
	return Mout;
}

template<typename Symmetry>
Mpo<Symmetry,complex<double> > HubbardObservables<Symmetry>::
make_FourierYSum (string name, const vector<OperatorType> &Ops, 
                  double factor, bool HERMITIAN, const vector<complex<double> > &phases) const
{
	stringstream ss;
	ss << name << "_ky(";
	for (int l=0; l<phases.size(); ++l)
	{
		ss << phases[l];
		if (l!=phases.size()-1) {ss << ",";}
		else                    {ss << ")";}
	}
	
	// all Ops[l].Q() must match
	Mpo<Symmetry,complex<double> > Mout(F.size(), Ops[0].Q(), ss.str(), HERMITIAN);
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	
	vector<complex<double> > phases_x_factor = phases;
	for (int l=0; l<phases.size(); ++l)
	{
		phases_x_factor[l] = phases[l] * factor;
	}
	
	vector<SiteOperator<Symmetry,complex<double> > > OpsPlain(Ops.size());
	for (int l=0; l<OpsPlain.size(); ++l)
	{
		OpsPlain[l] = Ops[l].template plain<double>().template cast<complex<double> >();
	}
	
	Mout.setLocalSum(OpsPlain, phases_x_factor);
	
	return Mout;
}

//-------------

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
c (size_t locx, size_t locy, double factor) const
{
	if constexpr(Dummy::IS_CHARGE_SU2())
				{
					auto Gxy = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx+locy)));
					return make_local(locx,locy, F[locx].c(Gxy,locy), factor, PROP::FERMIONIC);
				}
	else
	{
		return make_local(locx,locy, F[locx].c(locy), factor, PROP::FERMIONIC);
	}
}

template<typename Symmetry>
template<SPIN_INDEX sigma, typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
c (size_t locx, size_t locy, double factor) const
{
	if constexpr(Dummy::IS_CHARGE_SU2())
				{
					auto Gxy = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx+locy)));
					return make_local(locx,locy, F[locx].c(sigma,Gxy,locy), factor, PROP::FERMIONIC);
				}
	else
	{
		return make_local(locx,locy, F[locx].c(sigma,locy), factor, PROP::FERMIONIC);
	}
}

// template<typename Symmetry>
// template<SPIN_INDEX sigma, SUB_LATTICE G, typename Dummy>
// typename std::enable_if<Dummy::IS_CHARGE_SU2() and !Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
// c (size_t locx, size_t locy, double factor) const
// {
// 	return make_local(locx,locy, F[locx].c(sigma,G,locy), factor, PROP::FERMIONIC);
// }

// template<typename Symmetry>
// template<SUB_LATTICE G, typename Dummy>
// typename std::enable_if<Dummy::IS_CHARGE_SU2() and Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
// c (size_t locx, size_t locy, double factor) const
// {
// 	return make_local(locx,locy, F[locx].c(G,locy), factor, PROP::FERMIONIC);
// }

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
cdag (size_t locx, size_t locy, double factor) const
{
	if constexpr(Dummy::IS_CHARGE_SU2())
				{
					auto Gxy = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx+locy)));
					return make_local(locx,locy, F[locx].cdag(Gxy,locy), factor, PROP::FERMIONIC);
				}
	else
	{
		return make_local(locx,locy, F[locx].cdag(locy), factor, PROP::FERMIONIC);
	}
}

template<typename Symmetry>
template<SPIN_INDEX sigma, typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
cdag (size_t locx, size_t locy, double factor) const
{
	if constexpr(Dummy::IS_CHARGE_SU2())
				{
					auto Gxy = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx+locy)));
					return make_local(locx,locy, F[locx].cdag(sigma,Gxy,locy), factor, PROP::FERMIONIC);
				}
	else
	{
		return make_local(locx,locy, F[locx].cdag(sigma,locy), factor, PROP::FERMIONIC);
	}
}

// template<typename Symmetry>
// template<SPIN_INDEX sigma, SUB_LATTICE G, typename Dummy>
// typename std::enable_if<Dummy::IS_CHARGE_SU2() and !Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
// cdag (size_t locx, size_t locy, double factor) const
// {
// 	return make_local(locx,locy, F[locx].cdag(sigma,G,locy), factor, PROP::FERMIONIC);
// }

// template<typename Symmetry>
// template<SUB_LATTICE G, typename Dummy>
// typename std::enable_if<Dummy::IS_CHARGE_SU2() and Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
// cdag (size_t locx, size_t locy, double factor) const
// {
// 	return make_local(locx,locy, F[locx].cdag(G,locy), factor, PROP::FERMIONIC);
// }

template<typename Symmetry>
template<class Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(),Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
cc (size_t locx, size_t locy) const
{
	return make_local(locx,locy, F[locx].cc(locy), 1., PROP::BOSONIC);
}

template<typename Symmetry>
template<class Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(),Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
cdagcdag (size_t locx, size_t locy) const
{
	return make_local(locx,locy, F[locx].cdagcdag(locy), 1., PROP::BOSONIC);
}

template<typename Symmetry>
template<typename Dummy>
typename std::conditional<Dummy::IS_SPIN_SU2(),Mpo<Symmetry>, vector<Mpo<Symmetry> > >::type HubbardObservables<Symmetry>::
cdagc (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	if constexpr (Dummy::IS_SPIN_SU2() and !Dummy::IS_CHARGE_SU2())
				 {
					 return make_corr(locx1, locx2, locy1, locy2, F[locx1].cdag(locy1), F[locx2].c(locy2), Symmetry::qvacuum(), sqrt(2.), PROP::FERMIONIC, PROP::NON_HERMITIAN);
				 }
	else if constexpr (Dummy::IS_SPIN_SU2() and Dummy::IS_CHARGE_SU2())
				 {
					 auto Gx1y1 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx1+locy1)));
					 auto Gx2y2 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx2+locy2)));
					 return make_corr(locx1, locx2, locy1, locy2, F[locx1].cdag(Gx1y1,locy1), F[locx2].c(Gx2y2,locy2), Symmetry::qvacuum(), sqrt(2.)*sqrt(2.), PROP::FERMIONIC, PROP::NON_HERMITIAN);
				 }
	else
	{
		vector<Mpo<Symmetry> > out(2);
		out[0] = cdagc<UP>(locx1,locx2,locy1,locy2);
		out[1] = cdagc<DN>(locx1,locx2,locy1,locy2);
		return out;
	}
}

template<typename Symmetry>
template<SPIN_INDEX sigma, typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
cdagc (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	if constexpr (Dummy::ABELIAN)
				 {
					 return make_corr(locx1, locx2, locy1, locy2, F[locx1].cdag(sigma,locy1), F[locx2].c(sigma,locy2), Symmetry::qvacuum(), 1., PROP::FERMIONIC, PROP::NON_HERMITIAN);
				 }
	else
	{
		auto Gx1y1 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx1+locy1)));
		auto Gx2y2 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx2+locy2)));
		return make_corr(locx1, locx2, locy1, locy2, F[locx1].cdag(sigma,Gx1y1,locy1), F[locx2].c(sigma,Gx2y2,locy2), Symmetry::qvacuum(), 1., PROP::FERMIONIC, PROP::NON_HERMITIAN);
	}
}

// template<typename Symmetry>
// template<SPIN_INDEX sigma, SUB_LATTICE G, typename Dummy>
// typename std::enable_if<Dummy::IS_CHARGE_SU2() and !Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
// cdagc (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
// {
// 	return make_corr(locx1, locx2, locy1, locy2, F[locx1].cdag(sigma,G,locy1), F[locx2].c(sigma,G,locy2), Symmetry::qvacuum(), sqrt(2.), PROP::FERMIONIC, PROP::NON_HERMITIAN);
// }

// template<typename Symmetry>
// template<SUB_LATTICE G, typename Dummy>
// typename std::enable_if<Dummy::IS_CHARGE_SU2() and Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
// cdagc (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
// {
// 	return make_corr(locx1, locx2, locy1, locy2, F[locx1].cdag(G,locy1), F[locx2].c(G,locy2), Symmetry::qvacuum(), sqrt(2.)*sqrt(2.), PROP::FERMIONIC, PROP::NON_HERMITIAN);
// }

template<typename Symmetry>
template<SPIN_INDEX sigma1, SPIN_INDEX sigma2, typename Dummy>
typename std::enable_if<Dummy::ABELIAN,Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
cc (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	return make_corr(locx1, locx2, locy1, locy2, F[locx1].c(sigma1,locy1), F[locx2].c(sigma2,locy2), Symmetry::qvacuum(), 1., PROP::FERMIONIC, PROP::NON_HERMITIAN);
}

template<typename Symmetry>
template<SPIN_INDEX sigma1, SPIN_INDEX sigma2, typename Dummy>
typename std::enable_if<Dummy::ABELIAN,Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
cdagcdag (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	return make_corr(locx1, locx2, locy1, locy2, F[locx1].cdag(sigma1,locy1), F[locx2].cdag(sigma2,locy2), Symmetry::qvacuum(), 1., PROP::FERMIONIC, PROP::NON_HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::conditional<Dummy::IS_SPIN_SU2(),Mpo<Symmetry>, vector<Mpo<Symmetry> > >::type HubbardObservables<Symmetry>::
cc3 (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	if constexpr (Dummy::IS_SPIN_SU2())
				 {
					 //Determine Qtot to the spin triplet quantumnumber. 
					 auto Qtots = Symmetry::reduceSilent(F[locx1].c(locy1).Q(), F[locx2].c(locy2).Q());
					 typename Symmetry::qType Qtot;
					 for (const auto Q : Qtots) {if (Q[0] == 3) {Qtot = Q;}}
					 return make_corr(locx1, locx2, locy1, locy2, F[locx1].c(locy1), F[locx2].c(locy2), Qtot, sqrt(2.), PROP::FERMIONIC, PROP::NON_HERMITIAN);
				 }
	else
	{
		static_assert(!Dummy::IS_CHARGE_SU2(),"cc with a spin-triplet coupling cannot be computed for a charge SU2 symmetry. Use U1 charge symmetry instead.");
		vector<Mpo<Symmetry> > out(2);
		out[0] = cc<UP,DN>(locx1,locx2,locy1,locy2);
		out[1] = cc<DN,UP>(locx1,locx2,locy1,locy2);
		return out;
	}
}

template<typename Symmetry>
template<typename Dummy>
typename std::conditional<Dummy::IS_SPIN_SU2(),Mpo<Symmetry>, vector<Mpo<Symmetry> > >::type HubbardObservables<Symmetry>::
cdagcdag3 (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	if constexpr (Dummy::IS_SPIN_SU2())
				 {
					 //Determine Qtot to the spin triplet quantumnumber. 
					 auto Qtots = Symmetry::reduceSilent(F[locx1].cdag(locy1).Q(), F[locx2].cdag(locy2).Q());
					 typename Symmetry::qType Qtot;
					 for (const auto Q : Qtots) {if (Q[0] == 3) {Qtot = Q;}}
					 return make_corr(locx1, locx2, locy1, locy2, F[locx1].cdag(locy1), F[locx2].cdag(locy2), Qtot, sqrt(2.), PROP::FERMIONIC, PROP::NON_HERMITIAN);
				 }
	else
	{
		static_assert(!Dummy::IS_CHARGE_SU2(),"cc with a spin-triplet coupling cannot be computed for a charge SU2 symmetry. Use U1 charge symmetry instead.");
		vector<Mpo<Symmetry> > out(2);
		out[0] = cdagcdag<UP,DN>(locx1,locx2,locy1,locy2);
		out[1] = cdagcdag<DN,UP>(locx1,locx2,locy1,locy2);
		return out;
	}
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
Stringz (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	assert(locx1<F.size() and locx2<F.size());
	stringstream ss;
	ss << "Sz" << "(" << locx1 << "," << locy1 << "," << ")" 
	   << "Sz" << "(" << locx2 << "," << locy2 << "," << ")";
	
	auto Sz1 = F[locx1].Sz(locy1);
	auto Sz2 = F[locx2].Sz(locy2);
	
	Mpo<Symmetry> Mout(F.size(), Sz1.Q()+Sz2.Q(), ss.str());
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	
	if (locx1 == locx2)
	{
		Mout.setLocal(locx1, Sz1*Sz2);
	}
	else if (locx1<locx2)
	{
		Mout.setLocal({locx1,locx2}, {Sz1,Sz2}, F[0].nh()-F[0].ns());
//		Mout.setLocal({locx1,locx2}, {F[0].Id(),F[0].Id()}, F[0].nh()-F[0].ns());
	}
	else if (locx1>locx2)
	{
		throw;
//		Mout.setLocal({locx2, locx1}, {c*F[locx2].sign(), -1.*cdag}, F[0].sign());
	}
	
	return Mout;
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
StringzDimer (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	assert(locx1<F.size() and locx2<F.size());
	stringstream ss;
	ss << "Sz" << "(" << locx1 << "," << locy1 << "," << ")" 
	   << "Sz" << "(" << locx2 << "," << locy2 << "," << ")";
	
	auto Sz1 = F[locx1].Sz(locy1);
	auto Sz2 = F[locx2].Sz(locy2);
	
	Mpo<Symmetry> Mout(F.size(), Sz1.Q()+Sz2.Q(), ss.str());
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	
	if (locx1 == locx2)
	{
		throw;
//		Mout.setLocal(locx1, Sz1*Sz2);
	}
	else if (locx1<locx2)
	{
		Mout.setLocal({locx1,locx2}, {-pow(-4,(locx2-(locx1-1)-2)/2)*Sz1,Sz2}, F[0].Sz());
	}
	else if (locx1>locx2)
	{
		throw;
	}
	
	return Mout;
}

//-------------

template<typename Symmetry>
template<class Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(),Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
d (size_t locx, size_t locy) const
{
	return make_local(locx,locy, F[locx].d(locy), 1., PROP::BOSONIC, PROP::HERMITIAN);
}

template<typename Symmetry>
template<class Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(),Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
dtot() const
{
	for (size_t l=0; l<F.size(); ++l) {assert(F[l].orbitals()==1);}
	
	OperatorType Op = F[0].d();
	
	Mpo<Symmetry> Mout(F.size(), Op.Q, "double_occ_total", PROP::HERMITIAN);
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	
	Mout.setLocalSum(Op.template plain<double>());
	return Mout;
}

template<typename Symmetry>
template<class Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(),Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
s (size_t locx, size_t locy) const
{
	return make_local(locx,locy,  F[locx].n(locy)-2.*F[locx].d(locy), 1., PROP::BOSONIC, PROP::HERMITIAN);
}

template<typename Symmetry>
template<SPIN_INDEX sigma, typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
n (size_t locx, size_t locy) const
{
	return make_local(locx,locy, F[locx].n(sigma,locy), 1., PROP::BOSONIC, PROP::HERMITIAN);
}

template<typename Symmetry>
Mpo<Symmetry> HubbardObservables<Symmetry>::
n (size_t locx, size_t locy) const
{
	return make_local(locx,locy, F[locx].n(locy), 1., PROP::BOSONIC, PROP::HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::IS_CHARGE_SU2(), Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
T (size_t locx, size_t locy, double factor) const
{
	return make_local(locx,locy, F[locx].T(locy), factor, PROP::BOSONIC, PROP::NON_HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::IS_CHARGE_SU2(), Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
Tdag (size_t locx, size_t locy, double factor) const
{
	return make_local(locx,locy, F[locx].Tdag(locy), factor, PROP::BOSONIC, PROP::NON_HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(), Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
Tz (size_t locx, size_t locy) const
{
	return make_local(locx,locy, F[locx].Tz(locy), 1., PROP::BOSONIC, PROP::HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::NO_CHARGE_SYM(), Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
Tx (size_t locx, size_t locy) const
{
	auto G = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx+locy)));
	return make_local(locx,locy, F[locx].Tx(locy,G), 1., PROP::BOSONIC, PROP::HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::NO_CHARGE_SYM(), Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
iTy (size_t locx, size_t locy) const
{
	auto G = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx+locy)));
	return make_local(locx,locy, F[locx].iTy(locy,G), 1. , PROP::BOSONIC, PROP::HERMITIAN); //0.5*pow(-1,locx+locy)*(F[locx].cdagcdag(locy)-F[locx].cc(locy))
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(), Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
Tm (size_t locx, size_t locy) const
{
	auto G = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx+locy)));
	return make_local(locx,locy, F[locx].Tm(locy,G), 1., PROP::BOSONIC, PROP::NON_HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(), Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
Tp (size_t locx, size_t locy) const
{
	auto G = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx+locy)));
	return make_local(locx,locy, F[locx].Tp(locy,G), 1., PROP::BOSONIC, PROP::NON_HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(), Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
TpTm (size_t locx1, size_t locx2, size_t locy1, size_t locy2, double fac) const
{
	auto G1 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx1+locy1)));
	auto G2 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx2+locy2)));
	return make_corr(locx1,locx2,locy1,locy2, fac*F[locx1].Tp(locy1,G1), F[locx2].Tm(locy2,G2), Symmetry::qvacuum(), 1., PROP::NON_FERMIONIC, PROP::NON_HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(), Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
TmTp (size_t locx1, size_t locx2, size_t locy1, size_t locy2, double fac) const
{
	auto G1 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx1+locy1)));
	auto G2 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx2+locy2)));
	return make_corr(locx1,locx2,locy1,locy2, fac*F[locx1].Tm(locy1,G1), F[locx2].Tp(locy2,G2), Symmetry::qvacuum(), 1., PROP::NON_FERMIONIC, PROP::NON_HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(), Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
TzTz (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	return make_corr(locx1,locx2,locy1,locy2, F[locx1].Tz(locy1), F[locx2].Tz(locy2), Symmetry::qvacuum(), 1., PROP::NON_FERMIONIC, PROP::HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::conditional<Dummy::IS_CHARGE_SU2(), Mpo<Symmetry>, vector<Mpo<Symmetry> > >::type HubbardObservables<Symmetry>::
TdagT (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	if constexpr (Symmetry::IS_CHARGE_SU2())
				 {
					 return make_corr(locx1, locx2, locy1, locy2, F[locx1].Tdag(locy1), F[locx2].T(locy2), Symmetry::qvacuum(), sqrt(3.), PROP::BOSONIC, PROP::HERMITIAN);
					 // return make_corr("T†", "T", locx1, locx2, locy1, locy2, F[locx1].Tdag(locy1), F[locx2].T(locy2), Symmetry::qvacuum(), std::sqrt(3.), PROP::NON_FERMIONIC, PROP::HERMITIAN);
				 }
	else
	{
		vector<Mpo<Symmetry> > out(3);
		out[0] = TzTz(locx1,locx2,locy1,locy2);
		out[1] = TpTm(locx1,locx2,locy1,locy2,0.5);
		out[2] = TmTp(locx1,locx2,locy1,locy2,0.5);
		return out;
	}
}

template<typename Symmetry>
Mpo<Symmetry> HubbardObservables<Symmetry>::
ns (size_t locx, size_t locy) const
{
	return make_local(locx,locy, F[locx].ns(locy), 1., PROP::BOSONIC, PROP::HERMITIAN);
}

template<typename Symmetry>
Mpo<Symmetry> HubbardObservables<Symmetry>::
nh (size_t locx, size_t locy) const
{
	return make_local(locx,locy, F[locx].nh(locy), 1., PROP::BOSONIC, PROP::HERMITIAN);
}

template<typename Symmetry>
Mpo<Symmetry> HubbardObservables<Symmetry>::
nssq (size_t locx, size_t locy) const
{
	return make_local(locx,locy, F[locx].ns(locy) * F[locx].ns(locy), 1., PROP::BOSONIC, PROP::HERMITIAN);
}

template<typename Symmetry>
Mpo<Symmetry> HubbardObservables<Symmetry>::
nhsq (size_t locx, size_t locy) const
{
	return make_local(locx,locy, F[locx].nh(locy) * F[locx].nh(locy), 1., PROP::BOSONIC, PROP::HERMITIAN);
}

template<typename Symmetry>
template<SPIN_INDEX sigma1, SPIN_INDEX sigma2, typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
nn (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	return make_corr (locx1, locx2, locy1, locy2, F[locx1].n(sigma1,locy1), F[locx2].n(sigma2,locy2), Symmetry::qvacuum(), 1., PROP::NON_FERMIONIC, PROP::HERMITIAN);
}

template<typename Symmetry>
Mpo<Symmetry> HubbardObservables<Symmetry>::
nn (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	return make_corr (locx1, locx2, locy1, locy2, F[locx1].n(locy1), F[locx2].n(locy2), Symmetry::qvacuum(), 1., PROP::NON_FERMIONIC, PROP::HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(),Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
hh (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	return make_corr(locx1,locx2,locy1,locy2, 
	                 F[locx1].d(locy1)-F[locx1].n(locy1)+F[locx1].Id(),
	                 F[locx2].d(locy2)-F[locx2].n(locy2)+F[locx2].Id(),
					 Symmetry::qvacuum(), 1., PROP::NON_FERMIONIC, PROP::HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
Scomp (SPINOP_LABEL Sa, size_t locx, size_t locy, double factor) const
{
	bool HERMITIAN = (Sa==SX or Sa==SZ)? true:false;
	return make_local(locx,locy, F[locx].Scomp(Sa,locy), factor, PROP::BOSONIC, HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,complex<double> > >::type HubbardObservables<Symmetry>::
Rcomp (SPINOP_LABEL Sa, size_t locx, size_t locy) const
{
	stringstream ss;
	if (Sa==iSY)
	{
		ss << "exp[2π" << Sa << "(" << locx << "," << locy << ")]";
	}
	else
	{
		ss << "exp[2πi" << Sa << "(" << locx << "," << locy << ")]";
	}
	
	auto Op = F[locx].Rcomp(Sa,locy);
	
	Mpo<Symmetry,complex<double>> Mout(F.size(), Op.Q, ss.str(), false);
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	
	Mout.setLocal(locx, Op);
	
	return Mout;
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
ScompScomp (SPINOP_LABEL Sa1, SPINOP_LABEL Sa2, size_t locx1, size_t locx2, size_t locy1, size_t locy2, double fac) const
{
	return make_corr(locx1,locx2,locy1,locy2, F[locx1].Scomp(Sa1,locy1), F[locx2].Scomp(Sa2,locy2), getQ_ScompScomp(Sa1,Sa2), fac, PROP::NON_FERMIONIC, PROP::HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
S (size_t locx, size_t locy, double factor) const
{
	return make_local(locx,locy, F[locx].S(locy), factor, PROP::BOSONIC, PROP::NON_HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
Sdag (size_t locx, size_t locy, double factor) const
{
	return make_local(locx,locy, F[locx].Sdag(locy), factor, PROP::BOSONIC, PROP::NON_HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type HubbardObservables<Symmetry>::
Sz (size_t locx, size_t locy) const
{
	return Scomp(SZ,locx,locy);
}

template<typename Symmetry>
template<typename Dummy>
typename std::conditional<Dummy::IS_SPIN_SU2(), Mpo<Symmetry>, vector<Mpo<Symmetry> > >::type HubbardObservables<Symmetry>::
SdagS (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	if constexpr (Symmetry::IS_SPIN_SU2())
				 {
					 return make_corr(locx1, locx2, locy1, locy2, F[locx1].Sdag(locy1), F[locx2].S(locy2), Symmetry::qvacuum(), sqrt(3.), PROP::BOSONIC, PROP::HERMITIAN);
					 // return make_corr("T†", "T", locx1, locx2, locy1, locy2, F[locx1].Tdag(locy1), F[locx2].T(locy2), Symmetry::qvacuum(), std::sqrt(3.), PROP::NON_FERMIONIC, PROP::HERMITIAN);
				 }
	else
	{
		vector<Mpo<Symmetry> > out(3);
		out[0] = SzSz(locx1,locx2,locy1,locy2);
		out[1] = SpSm(locx1,locx2,locy1,locy2,0.5);
		out[2] = SmSp(locx1,locx2,locy1,locy2,0.5);
		return out;
	}
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,complex<double> > >::type HubbardObservables<Symmetry>::
S_ky (vector<complex<double> > phases) const
{
	vector<OperatorType> Ops(F.size());
	for (size_t l=0; l<F.size(); ++l)
	{
		Ops[l] = F[l].S(0);
	}
	return make_FourierYSum("S", Ops, 1., false, phases);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,complex<double> > >::type HubbardObservables<Symmetry>::
Sdag_ky (vector<complex<double> > phases, double factor) const
{
	vector<OperatorType> Ops(F.size());
	for (size_t l=0; l<F.size(); ++l)
	{
		Ops[l] = F[l].Sdag(0);
	}
	return make_FourierYSum("S†", Ops, 1., false, phases);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::IS_CHARGE_SU2(),Mpo<Symmetry,complex<double> > >::type HubbardObservables<Symmetry>::
T_ky (vector<complex<double> > phases) const
{
	vector<OperatorType> Ops(F.size());
	for (size_t l=0; l<F.size(); ++l)
	{
		Ops[l] = F[l].T(0);
	}
	return make_FourierYSum("T", Ops, 1., false, phases);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::IS_CHARGE_SU2(),Mpo<Symmetry,complex<double> > >::type HubbardObservables<Symmetry>::
Tdag_ky (vector<complex<double> > phases, double factor) const
{
	vector<OperatorType> Ops(F.size());
	for (size_t l=0; l<F.size(); ++l)
	{
		Ops[l] = F[l].Tdag(0);
	}
	return make_FourierYSum("T†", Ops, 1., false, phases);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2() and !Dummy::IS_CHARGE_SU2(),Mpo<Symmetry,complex<double> > >::type HubbardObservables<Symmetry>::
c_ky (vector<complex<double> > phases, double factor) const
{
	vector<OperatorType> Ops(F.size());
	for (size_t l=0; l<F.size(); ++l)
	{
		Ops[l] = F[l].c(0);
	}
	return make_FourierYSum("c", Ops, 1., false, phases);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2() and !Dummy::IS_CHARGE_SU2(),Mpo<Symmetry,complex<double> > >::type HubbardObservables<Symmetry>::
cdag_ky (vector<complex<double> > phases, double factor) const
{
	vector<OperatorType> Ops(F.size());
	for (size_t l=0; l<F.size(); ++l)
	{
		Ops[l] = F[l].cdag(0);
	}
	return make_FourierYSum("c†", Ops, 1., false, phases);
}

template<typename Symmetry>
typename Symmetry::qType HubbardObservables<Symmetry>::getQ_ScompScomp(SPINOP_LABEL Sa1, SPINOP_LABEL Sa2) const
{
	typename Symmetry::qType out;
	if ( (Sa1 == SZ and Sa2 == SZ) or (Sa1 == SP and Sa2 == SM) or (Sa1 == SM and Sa2 == SP) or (Sa1 == SX or Sa1 == iSY) ) {out = Symmetry::qvacuum();}
	else {assert(false and "Quantumnumber for the chosen ScompScomp is not computed. Add in HubbardObservables::getQ_ScompScomp");}
	
}
#endif
