#ifndef HUBBARDOBSERVABLES
#define HUBBARDOBSERVABLES

#include "bases/FermionBase.h"
#include "Mpo.h"
#include "ParamHandler.h" // from HELPERS

//include "DmrgLinearAlgebra.h"
//include "DmrgExternal.h"
//include "tensors/SiteOperator.h"

template<typename Symmetry, typename Scalar=double>
class HubbardObservables
{
//	typedef SiteOperatorQ<Symmetry,Eigen::Matrix<Scalar,Dynamic,Dynamic>> OperatorType;
	typedef SiteOperatorQ<Symmetry,Eigen::MatrixXd> OperatorType;
	
public:
	
	///@{
	HubbardObservables(){};
	HubbardObservables (const size_t &L); // for inheritance purposes
	HubbardObservables (const size_t &L, const vector<Param> &params, const std::map<string,std::any> &defaults);
	///@}
	
	///@{
	template<class Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar> >::type c (size_t locx, size_t locy=0, double factor=1.) const;

	template<SPIN_INDEX sigma, class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar> >::type c (size_t locx, size_t locy=0, double factor=1.) const;

	template<class Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar>>::type cdag (size_t locx, size_t locy=0, double factor=std::sqrt(2.)) const;

	template<SPIN_INDEX sigma, class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar> >::type cdag (size_t locx, size_t locy=0, double factor=1.) const;
	///@}
	
	///@{
	template<class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_CHARGE_SU2(),Mpo<Symmetry,Scalar> >::type cc (size_t locx, size_t locy=0) const;
	
	template<class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_CHARGE_SU2(),Mpo<Symmetry,Scalar> >::type cdagcdag (size_t locx, size_t locy=0) const;
	
	template<typename Dummy = Symmetry>
	typename std::conditional<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar>, vector<Mpo<Symmetry,Scalar> > >::type cdagc (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	
	template<typename Dummy = Symmetry>
	typename std::conditional<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar>, vector<Mpo<Symmetry,Scalar> > >::type cdag_nc (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	
	template<typename Dummy = Symmetry>
	typename std::conditional<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar>, vector<Mpo<Symmetry,Scalar> > >::type cdagn_c (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	
	template<typename Dummy = Symmetry>
	typename std::conditional<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar>, vector<Mpo<Symmetry,Scalar> > >::type cdagc3 (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	
	template<SPIN_INDEX sigma1, SPIN_INDEX sigma2, typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar> >::type cdagc (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	
	// Mpo<Symmetry,Scalar> triplet (size_t locx, size_t locy=0) const;
	///@}
	
	template<SPIN_INDEX sigma1, SPIN_INDEX sigma2, typename Dummy = Symmetry>
	typename std::enable_if<Dummy::ABELIAN,Mpo<Symmetry,Scalar> >::type cdagcdag (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	
	template<SPIN_INDEX sigma1, SPIN_INDEX sigma2, typename Dummy = Symmetry>
	typename std::enable_if<Dummy::ABELIAN,Mpo<Symmetry,Scalar> >::type cc (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	
	template<typename Dummy = Symmetry>
	typename std::conditional<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar>, vector<Mpo<Symmetry,Scalar> > >::type
	cc3 (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	
	template<typename Dummy = Symmetry>
	typename std::conditional<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar>, vector<Mpo<Symmetry,Scalar> > >::type
	cdagcdag3 (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	
	///@{
	template<class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_CHARGE_SU2(),Mpo<Symmetry,Scalar> >::type d (size_t locx, size_t locy=0) const;
	template<class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_CHARGE_SU2(),Mpo<Symmetry,Scalar> >::type dtot() const;
	Mpo<Symmetry,Scalar> ns (size_t locx, size_t locy=0) const;
	Mpo<Symmetry,Scalar> nh (size_t locx, size_t locy=0) const;
	Mpo<Symmetry,Scalar> nssq (size_t locx, size_t locy=0) const;
	Mpo<Symmetry,Scalar> nhsq (size_t locx, size_t locy=0) const;
	template<class Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_CHARGE_SU2(),Mpo<Symmetry,Scalar> >::type s (size_t locx, size_t locy=0) const;
	
	template<SPIN_INDEX sigma, typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar> >::type n (size_t locx, size_t locy=0) const;
	
	Mpo<Symmetry,Scalar> n (size_t locx, size_t locy=0) const;
	
	template<SPIN_INDEX sigma1, SPIN_INDEX sigma2, typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar> >::type nn (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	
	Mpo<Symmetry,Scalar> nn (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_CHARGE_SU2(),Mpo<Symmetry,Scalar> >::type hh (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	///@}
	
	///@{
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_CHARGE_SU2(), Mpo<Symmetry,Scalar> >::type T (size_t locx, size_t locy=0, double factor=1.) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_CHARGE_SU2(), Mpo<Symmetry,Scalar> >::type Tdag (size_t locx, size_t locy=0, double factor=std::sqrt(3.)) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_CHARGE_SU2(), Mpo<Symmetry,Scalar> >::type Tp (size_t locx, size_t locy=0) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_CHARGE_SU2(), Mpo<Symmetry,Scalar> >::type Tm (size_t locx, size_t locy=0) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::NO_CHARGE_SYM(), Mpo<Symmetry,Scalar> >::type Tx (size_t locx, size_t locy=0) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::NO_CHARGE_SYM(), Mpo<Symmetry,Scalar> >::type iTy (size_t locx, size_t locy=0) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_CHARGE_SU2(), Mpo<Symmetry,Scalar> >::type Tz (size_t locx, size_t locy=0) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_CHARGE_SU2(), Mpo<Symmetry,Scalar> >::type TpTm (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_CHARGE_SU2(), Mpo<Symmetry,Scalar> >::type TmTp (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_CHARGE_SU2(), Mpo<Symmetry,Scalar> >::type TzTz (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	template<typename Dummy = Symmetry>
	typename std::conditional<Dummy::IS_CHARGE_SU2(), Mpo<Symmetry,Scalar>, vector<Mpo<Symmetry,Scalar> > >::type TdagT (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	///@}
	
	///@{
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type S (size_t locx, size_t locy=0, double factor=1.) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type Sdag (size_t locx, size_t locy=0, double factor=std::sqrt(3.)) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type Stot (size_t locy, double factor, int dLphys) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type Sdagtot (size_t locy, double factor, int dLphys) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type Scomp (SPINOP_LABEL Sa, size_t locx, size_t locy=0, double factor=1.) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type Sz (size_t locx, size_t locy=0) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type Sp (size_t locx, size_t locy=0) const {return Scomp(SP,locx,locy);};
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type Sm (size_t locx, size_t locy=0) const {return Scomp(SM,locx,locy);};
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type ScompScomp (SPINOP_LABEL Sa1, SPINOP_LABEL Sa2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type SpSm (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const {return ScompScomp(SP,SM,locx1,locx2,locy1,locy2,fac);};
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type SmSp (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const {return ScompScomp(SM,SP,locx1,locx2,locy1,locy2,fac);};
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type SzSz (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const {return ScompScomp(SZ,SZ,locx1,locx2,locy1,locy2,1.);};
	template<typename Dummy = Symmetry>
	typename std::conditional<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar>, vector<Mpo<Symmetry,Scalar> > >::type SdagS (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	
	template<class Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type CanonicalEntangler (int dLphys, double factor=1.) const;
	
	template<class Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_U1(), Mpo<Symmetry,Scalar> >::type CanonicalEntangler (int dLphys, double factor=1.) const;
	
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry, complex<double> > >::type Rcomp (SPINOP_LABEL Sa, size_t locx, size_t locy=0) const;
	///@}

	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar> >::type Stringz (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar> >::type StringzDimer (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;

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
	
	Mpo<Symmetry,Scalar> make_local (size_t locx, size_t locy, const OperatorType &Op, double factor =1., bool FERMIONIC=false, bool HERMITIAN=false) const;
	Mpo<Symmetry,Scalar> make_localSum (const vector<OperatorType> &Op, vector<double> factor, bool HERMITIAN) const;
	Mpo<Symmetry,Scalar> make_corr  (size_t locx1, size_t locx2, size_t locy1, size_t locy2,
	                                 const OperatorType &Op1, const OperatorType &Op2, qarray<Symmetry::Nq> Qtot,
	                                 double factor, bool FERMIONIC, bool HERMITIAN) const;
	
	Mpo<Symmetry,complex<double> >
	make_FourierYSum (string name, const vector<OperatorType> &Ops, double factor, bool HERMITIAN, const vector<complex<double> > &phases) const;
	
	typename Symmetry::qType getQ_ScompScomp(SPINOP_LABEL Sa1, SPINOP_LABEL Sa2) const;
	
	vector<FermionBase<Symmetry> > F;
};

template<typename Symmetry, typename Scalar>
HubbardObservables<Symmetry,Scalar>::
HubbardObservables (const size_t &L)
{
	F.resize(L);
}

template<typename Symmetry, typename Scalar>
HubbardObservables<Symmetry,Scalar>::
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

template<typename Symmetry, typename Scalar>
Mpo<Symmetry,Scalar> HubbardObservables<Symmetry,Scalar>::
make_local (size_t locx, size_t locy, const OperatorType &Op, double factor, bool FERMIONIC, bool HERMITIAN) const
{
	assert(locx<F.size() and locy<F[locx].dim());
	stringstream ss;
	ss << Op.label() << "(" << locx << "," << locy;
	if (factor != 1.) ss << ",factor=" << factor;
	ss << ")";
	
	Mpo<Symmetry,Scalar> Mout(F.size(), Op.Q(), ss.str(), HERMITIAN);
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	
	if (FERMIONIC)
	{
		vector<SiteOperator<Symmetry,Scalar> > Signs(locx);
		for (size_t l=0; l<locx; ++l) {Signs[l] = F[l].sign().template plain<double>().template cast<Scalar>();}
		
		Mout.setLocal(locx, (factor * Op).template plain<double>().template cast<Scalar>(), Signs);
	}
	else
	{
		Mout.setLocal(locx, (factor * Op).template plain<double>().template cast<Scalar>());
	}
	
	return Mout;
}

template<typename Symmetry, typename Scalar>
Mpo<Symmetry,Scalar> HubbardObservables<Symmetry,Scalar>::
make_localSum (const vector<OperatorType> &Op, vector<double> factor, bool HERMITIAN) const
{
	assert(Op.size()==F.size() and factor.size()==F.size());
	stringstream ss;
	ss << Op[0].label() << "localSum";
	
	Mpo<Symmetry,Scalar> Mout(F.size(), Op[0].Q(), ss.str(), HERMITIAN);
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	
	vector<SiteOperator<Symmetry,Scalar>> Op_plain;
	for (int i=0; i<Op.size(); ++i)
	{
		Op_plain.push_back(Op[i].template plain<double>().template cast<Scalar>());
	}
	vector<Scalar> factor_cast(factor.size());
	for (int l=0; l<factor.size(); ++l)
	{
		factor_cast[l] = static_cast<Scalar>(factor[l]);
	}
	Mout.setLocalSum(Op_plain, factor_cast);
	
	return Mout;
}

template<typename Symmetry, typename Scalar>
Mpo<Symmetry,Scalar> HubbardObservables<Symmetry,Scalar>::
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
	
	Mpo<Symmetry,Scalar> Mout(F.size(), Qtot, ss.str(), HERMITIAN);
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	
	if (FERMIONIC)
	{
		if (locx1 == locx2)
		{
			Mout.setLocal(locx1, factor * OperatorType::prod(Op1,Op2,Qtot).template plain<double>().template cast<Scalar>());
		}
		else if (locx1<locx2)
		{
			Mout.setLocal({locx1, locx2}, {(factor * (Op1 * F[locx1].sign())).template plain<double>().template cast<Scalar>(), 
			                               Op2.template plain<double>().template cast<Scalar>()}, 
			                               F[0].sign().template plain<double>().template cast<Scalar>());
		}
		else if (locx1>locx2)
		{
			Mout.setLocal({locx2, locx1}, {(factor * (Op2 * F[locx2].sign())).template plain<double>().template cast<Scalar>(), 
			                               -Symmetry::spinorFactor() * Op1.template plain<double>().template cast<Scalar>()}, 
			                               F[0].sign().template plain<double>().template cast<Scalar>());
		}
	}
	else
	{
		if (locx1 == locx2)
		{
			auto product = factor*OperatorType::prod(Op1, Op2, Qtot);
			Mout.setLocal(locx1, product.template plain<double>().template cast<Scalar>());
		}
		else
		{
			Mout.setLocal({locx1, locx2}, {(factor*Op1).template plain<double>().template cast<Scalar>(), 
			                                        Op2.template plain<double>().template cast<Scalar>()});
		}
	}
	return Mout;
}

template<typename Symmetry, typename Scalar>
Mpo<Symmetry,complex<double> > HubbardObservables<Symmetry,Scalar>::
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

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
CanonicalEntangler (int dLphys, double factor) const
{
	assert(dLphys==2 and "Only dLphys=2 is implemented!");
	Mpo<Symmetry,Scalar> Mout(F.size(), Symmetry::qvacuum(), "CanonicalEntangler", PROP::HERMITIAN, false, BC::OPEN, DMRG::VERBOSITY::HALFSWEEPWISE);
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	
	std::vector<typename Symmetry::qType> qList(F.size()+1);
	std::vector<SiteOperator<Symmetry,Scalar>> opList(F.size());
	
	for (int i=0; i<qList.size(); ++i) {qList[i] = Symmetry::qvacuum();}
	for (int i=0; i<opList.size(); ++i) {opList[i] = F[i].Id().template plain<double>();}
	
	for (int i=0; i<F.size(); i+=dLphys)
	for (int j=0; j<F.size(); j+=dLphys)
	{
		if (i<j)
		{
			auto qListWork = qList;
			auto opListWork = opList;
			
			qListWork[1+i]   = qarray<Symmetry::Nq>{2,1}; // cdag
			qListWork[1+i+1] = qarray<Symmetry::Nq>{1,2}; // cdag
			for (int k=i+3; k<j+1; ++k) qListWork[k] = qarray<Symmetry::Nq>{1,2};
			qListWork[1+j]   = qarray<Symmetry::Nq>{2,1}; // c
			qListWork[1+j+1] = qarray<Symmetry::Nq>{1,0}; // c
			
//			qListWork[1+i]   = qarray<Symmetry::Nq>{2,1}; // cdag
//			qListWork[1+i+1] = qarray<Symmetry::Nq>{1,0}; // c
//			for (int k=i+3; k<j+1; ++k) qListWork[k] = qarray<Symmetry::Nq>{1,0};
//			qListWork[1+j]   = qarray<Symmetry::Nq>{2,1}; // cdag
//			qListWork[1+j+1] = qarray<Symmetry::Nq>{1,0}; // c
			
			opListWork[i]   = (F[i].cdag(0) * F[i].sign()).template plain<double>();
			opListWork[i+1] = F[i+1].cdag(0).template plain<double>();
			opListWork[j]   = (F[j].c(0) * F[j].sign()).template plain<double>();
			opListWork[j+1] = F[j+1].c(0).template plain<double>();
			
			std::vector<SiteOperator<Symmetry,Scalar>> opRes = std::vector<SiteOperator<Symmetry,Scalar>>(opListWork.begin()+i, opListWork.begin()+j+2);
			std::vector<typename Symmetry::qType> qRes = std::vector<typename Symmetry::qType>(qListWork.begin()+i, qListWork.begin()+j+3);
			
			// sign: global minus; another global minus of unclear origin
			Mout.push_qpath(i, opRes, qRes, +1.*factor);
//			Mout.push_qpath(i, opRes, qRes, +2.*factor);
			
			qListWork = qList;
			opListWork = opList;
			
			qListWork[1+i]   = qarray<Symmetry::Nq>{2,-1}; // c
			qListWork[1+i+1] = qarray<Symmetry::Nq>{1,-2}; // c
			for (int k=i+3; k<j+1; ++k) qListWork[k] = qarray<Symmetry::Nq>{1,-2};
			qListWork[1+j]   = qarray<Symmetry::Nq>{2,-1}; // cdag
			qListWork[1+j+1] = qarray<Symmetry::Nq>{1,0}; // cdag
			
//			qListWork[1+i]   = qarray<Symmetry::Nq>{2,-1}; // c
//			qListWork[1+i+1] = qarray<Symmetry::Nq>{1,0}; // cdag
//			for (int k=i+3; k<j+1; ++k) qListWork[k] = qarray<Symmetry::Nq>{1,0};
//			qListWork[1+j]   = qarray<Symmetry::Nq>{2,-1}; // c
//			qListWork[1+j+1] = qarray<Symmetry::Nq>{1,0}; // cdag
			
			opListWork[i]   = (F[i].c(0) * F[i].sign()).template plain<double>();
			opListWork[i+1] = F[i+1].c(0).template plain<double>();
			opListWork[j]   = (F[j].cdag(0) * F[j].sign()).template plain<double>();
			opListWork[j+1] = F[j+1].cdag(0).template plain<double>();
			
			opRes = std::vector<SiteOperator<Symmetry,Scalar>>(opListWork.begin()+i, opListWork.begin()+j+2);
			qRes = std::vector<typename Symmetry::qType>(qListWork.begin()+i, qListWork.begin()+j+3);
			
			Mout.push_qpath(i, opRes, qRes, 1.*factor);
//			Mout.push_qpath(i, opRes, qRes, 2.*factor);
		}
	}
	
	Mout.N_phys = F.size()/dLphys;
	Mout.finalize(PROP::COMPRESS, 1); // power=1
	Mout.precalc_TwoSiteData(true);
	
	return Mout;
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_U1(), Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
CanonicalEntangler (int dLphys, double factor) const
{
	assert(dLphys==2 and "Only dLphys=2 is implemented!");
	Mpo<Symmetry,Scalar> Mout(F.size(), Symmetry::qvacuum(), "CanonicalEntangler", PROP::HERMITIAN);
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	
	std::vector<typename Symmetry::qType> qList(F.size()+1);
	std::vector<SiteOperator<Symmetry,Scalar>> opList(F.size());
	
	for (int i=0; i<qList.size(); ++i) {qList[i] = Symmetry::qvacuum();}
	for (int i=0; i<opList.size(); ++i) {opList[i] = F[i].Id().template plain<double>();}
	
	auto add_term = [&qList, &opList, &Mout, this] (int i, int j, const OperatorType &Sysi, const OperatorType &Bathi, const OperatorType &Sysj, const OperatorType &Bathj, double sign)
	{
		auto qListWork = qList;
		auto opListWork = opList;
		
		opListWork[i]   = (Sysi * F[i].sign()).template plain<double>();
		opListWork[i+1] = Bathi.template plain<double>();
		opListWork[j]   = (Sysj * F[j].sign()).template plain<double>();
		opListWork[j+1] = Bathj.template plain<double>();
		
		qListWork[1+i]   = Sysi.Q();
		qListWork[1+i+1] = Sysi.Q()+Bathi.Q();
		for (int k=i+3; k<j+1; ++k) qListWork[k] = Sysi.Q()+Bathi.Q();
		qListWork[1+j]   = Sysi.Q()+Bathi.Q()+Sysj.Q();
		qListWork[1+j+1] = Sysi.Q()+Bathi.Q()+Sysj.Q()+Bathj.Q();
		
		assert(Sysi.Q()+Bathi.Q()+Sysj.Q()+Bathj.Q() == Symmetry::qvacuum());
		
		std::vector<SiteOperator<Symmetry,Scalar>> opRes = std::vector<SiteOperator<Symmetry,Scalar>>(opListWork.begin()+i, opListWork.begin()+j+2);
		std::vector<typename Symmetry::qType> qRes = std::vector<typename Symmetry::qType>(qListWork.begin()+i, qListWork.begin()+j+3);
		
		Mout.push_qpath(i, opRes, qRes, sign);
	};
	
	for (int i=0; i<F.size(); i+=2)
	for (int j=0; j<F.size(); j+=2)
	{
		if (i<j)
		{
			// variant 1: singlet-singlet coupling: strange behaviour, maybe only good for tJ model
			// sign: global minus; one transposition
//			add_term(i, j, F[i].cdag(UP), F[i+1].cdag(DN), F[j].c(UP),    F[j+1].c(DN),    +0.5); // ↑↓*↑↓
//			add_term(i, j, F[i].c(UP),    F[i+1].c(DN),    F[j].cdag(UP), F[j+1].cdag(DN), +0.5);
//			
//			add_term(i, j, F[i].cdag(DN), F[i+1].cdag(UP), F[j].c(UP),    F[j+1].c(DN),    -0.5); // ↓↑*↑↓
//			add_term(i, j, F[i].c(DN),    F[i+1].c(UP),    F[j].cdag(UP), F[j+1].cdag(DN), -0.5);
//			
//			add_term(i, j, F[i].cdag(UP), F[i+1].cdag(DN), F[j].c(DN),    F[j+1].c(UP),    -0.5); // ↑↓*↓↑
//			add_term(i, j, F[i].c(UP),    F[i+1].c(DN),    F[j].cdag(DN), F[j+1].cdag(UP), -0.5);
//			
//			add_term(i, j, F[i].cdag(DN), F[i+1].cdag(UP), F[j].c(DN),    F[j+1].c(UP),    +0.5); // ↓↑*↓↑
//			add_term(i, j, F[i].c(DN),    F[i+1].c(UP),    F[j].cdag(DN), F[j+1].cdag(UP), +0.5);
//			
//			// variant 2: conserves N and Sz
//			add_term(i, j, F[i].cdag(UP), F[i+1].cdag(UP), F[j].c(UP),    F[j+1].c(UP),    +1.*factor); // ↑↑*↑↑
//			add_term(i, j, F[i].c(UP),    F[i+1].c(UP),    F[j].cdag(UP), F[j+1].cdag(UP), +1.*factor);
//			
//			add_term(i, j, F[i].cdag(DN), F[i+1].cdag(DN), F[j].c(DN),    F[j+1].c(DN),    +1.*factor); // ↓↓*↓↓
//			add_term(i, j, F[i].c(DN),    F[i+1].c(DN),    F[j].cdag(DN), F[j+1].cdag(DN), +1.*factor);
//			
			// variant 3: conserves N and Sz
			add_term(i, j, F[i].cdag(UP), F[i+1].cdag(DN), F[j].c(UP),    F[j+1].c(DN),    +1.*factor); // ↑↓*↑↓
			add_term(i, j, F[i].c(UP),    F[i+1].c(DN),    F[j].cdag(UP), F[j+1].cdag(DN), +1.*factor);
			
			add_term(i, j, F[i].cdag(DN), F[i+1].cdag(UP), F[j].c(DN),    F[j+1].c(UP),    +1.*factor); // ↓↑*↓↑
			add_term(i, j, F[i].c(DN),    F[i+1].c(UP),    F[j].cdag(DN), F[j+1].cdag(UP), +1.*factor);
//			
//			// variant 4: bad
//			add_term(i, j, F[i].cdag(UP), F[i+1].cdag(DN), F[j].c(DN),    F[j+1].c(UP),    +1.*factor); // ↑↓*↓↑
//			add_term(i, j, F[i].c(UP),    F[i+1].c(DN),    F[j].cdag(DN), F[j+1].cdag(UP), +1.*factor);
//			
//			add_term(i, j, F[i].cdag(DN), F[i+1].cdag(UP), F[j].c(UP),    F[j+1].c(DN),    +1.*factor); // ↓↑*↑↓
//			add_term(i, j, F[i].c(DN),    F[i+1].c(UP),    F[j].cdag(UP), F[j+1].cdag(DN), +1.*factor);
//			
//			// test variant
//			add_term(i, j, F[i].Sp(), F[i+1].Sm(), F[j].Sm(),    F[j+1].Sp(),    -1.*factor);
//			add_term(i, j, F[i].Sm(), F[i+1].Sp(), F[j].Sp(),    F[j+1].Sm(),    -1.*factor);
//			add_term(i, j, F[i].Tp(), F[i+1].Tm(), F[j].Tm(),    F[j+1].Tp(),    -1.*factor);
//			add_term(i, j, F[i].Tm(), F[i+1].Tp(), F[j].Tp(),    F[j+1].Tm(),    -1.*factor);
		}
	}
	
	Mout.N_phys = F.size()/dLphys;
	Mout.finalize(PROP::COMPRESS, 1); // power=1
	Mout.precalc_TwoSiteData(true);
	
	return Mout;
}

//-------------

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
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

template<typename Symmetry, typename Scalar>
template<SPIN_INDEX sigma, typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
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

// template<typename Symmetry, typename Scalar>
// template<SPIN_INDEX sigma, SUB_LATTICE G, typename Dummy>
// typename std::enable_if<Dummy::IS_CHARGE_SU2() and !Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
// c (size_t locx, size_t locy, double factor) const
// {
// 	return make_local(locx,locy, F[locx].c(sigma,G,locy), factor, PROP::FERMIONIC);
// }

// template<typename Symmetry, typename Scalar>
// template<SUB_LATTICE G, typename Dummy>
// typename std::enable_if<Dummy::IS_CHARGE_SU2() and Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
// c (size_t locx, size_t locy, double factor) const
// {
// 	return make_local(locx,locy, F[locx].c(G,locy), factor, PROP::FERMIONIC);
// }

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
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

template<typename Symmetry, typename Scalar>
template<SPIN_INDEX sigma, typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
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

template<typename Symmetry, typename Scalar>
template<class Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(),Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
cc (size_t locx, size_t locy) const
{
	return make_local(locx,locy, F[locx].cc(locy), 1., PROP::BOSONIC);
}

template<typename Symmetry, typename Scalar>
template<class Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(),Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
cdagcdag (size_t locx, size_t locy) const
{
	return make_local(locx,locy, F[locx].cdagcdag(locy), 1., PROP::BOSONIC);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::conditional<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar>, vector<Mpo<Symmetry,Scalar> > >::type HubbardObservables<Symmetry,Scalar>::
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
		vector<Mpo<Symmetry,Scalar> > out(2);
		out[0] = cdagc<UP>(locx1,locx2,locy1,locy2);
		out[1] = cdagc<DN>(locx1,locx2,locy1,locy2);
		return out;
	}
}

// n(j)*cdag(i)*c(j)
// = cdag(i)*n(j)*c(j) for i!=0
// = n(i)^2 for i==j
template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::conditional<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar>, vector<Mpo<Symmetry,Scalar> > >::type HubbardObservables<Symmetry,Scalar>::
cdag_nc (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	if constexpr (Dummy::IS_SPIN_SU2() and !Dummy::IS_CHARGE_SU2())
	{
		if (locx1 == locx2)
		{
			assert(locy1 == locy2);
			return make_local (locx1, locy1, F[locx1].n(locy1)*F[locx1].n(locy1), 1., PROP::BOSONIC, PROP::HERMITIAN);
		}
		else
		{
			return make_corr(locx1, locx2, locy1, locy2, F[locx1].cdag(locy1), F[locx2].n(locy2)*F[locx2].c(locy2), Symmetry::qvacuum(), sqrt(2.), PROP::FERMIONIC, PROP::NON_HERMITIAN);
		}
	}
	else if constexpr (Dummy::IS_SPIN_SU2() and Dummy::IS_CHARGE_SU2())
	{
		throw;
	}
	else
	{
		throw;
	}
}

// cdag(i)*c(j)*n(i)
// = cdag(i)*n(i)*c(j) for i!=0
// = n(i)^2 for i==j
template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::conditional<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar>, vector<Mpo<Symmetry,Scalar> > >::type HubbardObservables<Symmetry,Scalar>::
cdagn_c (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	if constexpr (Dummy::IS_SPIN_SU2() and !Dummy::IS_CHARGE_SU2())
	{
		if (locx1 == locx2)
		{
			assert(locy1 == locy2);
			return make_local (locx1, locy1, F[locx1].n(locy1)*F[locx1].n(locy1), 1., PROP::BOSONIC, PROP::HERMITIAN);
		}
		else
		{
			return make_corr(locx1, locx2, locy1, locy2, F[locx1].cdag(locy1)*F[locx1].n(locy1), F[locx2].c(locy2), Symmetry::qvacuum(), sqrt(2.), PROP::FERMIONIC, PROP::NON_HERMITIAN);
		}
	}
	else if constexpr (Dummy::IS_SPIN_SU2() and Dummy::IS_CHARGE_SU2())
	{
		throw;
	}
	else
	{
		throw;
	}
}

template<typename Symmetry, typename Scalar>
template<SPIN_INDEX sigma1, SPIN_INDEX sigma2, typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
cdagc (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	if constexpr (Dummy::ABELIAN)
	{
		auto Qtot = F[locx1].cdag(sigma1,locy1).Q() + F[locx2].c(sigma2,locy2).Q();
		return make_corr(locx1, locx2, locy1, locy2, F[locx1].cdag(sigma1,locy1), F[locx2].c(sigma2,locy2), Qtot, 1., PROP::FERMIONIC, PROP::NON_HERMITIAN);
	}
	else
	{
		auto Gx1y1 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx1+locy1)));
		auto Gx2y2 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx2+locy2)));
		return make_corr(locx1, locx2, locy1, locy2, F[locx1].cdag(sigma1,Gx1y1,locy1), F[locx2].c(sigma2,Gx2y2,locy2), Symmetry::qvacuum(), 1., PROP::FERMIONIC, PROP::NON_HERMITIAN);
	}
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::conditional<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar>, vector<Mpo<Symmetry,Scalar> > >::type HubbardObservables<Symmetry,Scalar>::
cdagc3 (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	if constexpr (Dummy::IS_SPIN_SU2() and !Dummy::IS_CHARGE_SU2())
	{
		return make_corr(locx1, locx2, locy1, locy2, F[locx1].cdag(locy1), F[locx2].c(locy2), {3,0}, 1./sqrt(3.), PROP::FERMIONIC, PROP::NON_HERMITIAN);
	}
//	else if constexpr (Dummy::IS_SPIN_SU2() and Dummy::IS_CHARGE_SU2())
//	{
//		auto Gx1y1 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx1+locy1)));
//		auto Gx2y2 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx2+locy2)));
//		return make_corr(locx1, locx2, locy1, locy2, F[locx1].cdag(Gx1y1,locy1), F[locx2].c(Gx2y2,locy2), Symmetry::qvacuum(), sqrt(2.)*sqrt(2.), PROP::FERMIONIC, PROP::NON_HERMITIAN);
//	}
//	else
//	{
//		vector<Mpo<Symmetry,Scalar> > out(2);
//		out[0] = cdagc<UP>(locx1,locx2,locy1,locy2);
//		out[1] = cdagc<DN>(locx1,locx2,locy1,locy2);
//		return out;
//	}
}

template<typename Symmetry, typename Scalar>
template<SPIN_INDEX sigma1, SPIN_INDEX sigma2, typename Dummy>
typename std::enable_if<Dummy::ABELIAN,Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
cc (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	return make_corr(locx1, locx2, locy1, locy2, F[locx1].c(sigma1,locy1), F[locx2].c(sigma2,locy2), Symmetry::qvacuum(), 1., PROP::FERMIONIC, PROP::NON_HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<SPIN_INDEX sigma1, SPIN_INDEX sigma2, typename Dummy>
typename std::enable_if<Dummy::ABELIAN,Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
cdagcdag (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	return make_corr(locx1, locx2, locy1, locy2, F[locx1].cdag(sigma1,locy1), F[locx2].cdag(sigma2,locy2), Symmetry::qvacuum(), 1., PROP::FERMIONIC, PROP::NON_HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::conditional<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar>, vector<Mpo<Symmetry,Scalar> > >::type HubbardObservables<Symmetry,Scalar>::
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
		vector<Mpo<Symmetry,Scalar> > out(2);
		out[0] = cc<UP,DN>(locx1,locx2,locy1,locy2);
		out[1] = cc<DN,UP>(locx1,locx2,locy1,locy2);
		return out;
	}
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::conditional<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar>, vector<Mpo<Symmetry,Scalar> > >::type HubbardObservables<Symmetry,Scalar>::
cdagcdag3 (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	if constexpr (Dummy::IS_SPIN_SU2())
	{
		// Set Qtot to the spin triplet quantum number. 
		auto Qtots = Symmetry::reduceSilent(F[locx1].cdag(locy1).Q(), F[locx2].cdag(locy2).Q());
		typename Symmetry::qType Qtot;
		for (const auto Q : Qtots) {if (Q[0] == 3) {Qtot = Q;}}
		return make_corr(locx1, locx2, locy1, locy2, F[locx1].cdag(locy1), F[locx2].cdag(locy2), Qtot, sqrt(2.), PROP::FERMIONIC, PROP::NON_HERMITIAN);
	 }
	else
	{
		static_assert(!Dummy::IS_CHARGE_SU2(),"cc with a spin-triplet coupling cannot be computed for a charge-SU(2) symmetry. Use U(1) charge symmetry instead.");
		vector<Mpo<Symmetry,Scalar> > out(2);
		out[0] = cdagcdag<UP,DN>(locx1,locx2,locy1,locy2);
		out[1] = cdagcdag<DN,UP>(locx1,locx2,locy1,locy2);
		return out;
	}
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
Stringz (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	assert(locx1<F.size() and locx2<F.size());
	stringstream ss;
	ss << "Sz" << "(" << locx1 << "," << locy1 << "," << ")" 
	   << "Sz" << "(" << locx2 << "," << locy2 << "," << ")";
	
	auto Sz1 = F[locx1].Sz(locy1);
	auto Sz2 = F[locx2].Sz(locy2);
	
	Mpo<Symmetry,Scalar> Mout(F.size(), Sz1.Q()+Sz2.Q(), ss.str());
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

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
StringzDimer (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	assert(locx1<F.size() and locx2<F.size());
	stringstream ss;
	ss << "Sz" << "(" << locx1 << "," << locy1 << "," << ")" 
	   << "Sz" << "(" << locx2 << "," << locy2 << "," << ")";
	
	auto Sz1 = F[locx1].Sz(locy1);
	auto Sz2 = F[locx2].Sz(locy2);
	
	Mpo<Symmetry,Scalar> Mout(F.size(), Sz1.Q()+Sz2.Q(), ss.str());
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

template<typename Symmetry, typename Scalar>
template<class Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(),Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
d (size_t locx, size_t locy) const
{
	return make_local(locx,locy, F[locx].d(locy), 1., PROP::BOSONIC, PROP::HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<class Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(),Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
dtot() const
{
	for (size_t l=0; l<F.size(); ++l) {assert(F[l].orbitals()==1);}
	
	OperatorType Op = F[0].d();
	
	Mpo<Symmetry,Scalar> Mout(F.size(), Op.Q, "double_occ_total", PROP::HERMITIAN);
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	
	Mout.setLocalSum(Op.template plain<double>());
	return Mout;
}

template<typename Symmetry, typename Scalar>
template<class Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(),Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
s (size_t locx, size_t locy) const
{
	return make_local(locx,locy,  F[locx].n(locy)-2.*F[locx].d(locy), 1., PROP::BOSONIC, PROP::HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<SPIN_INDEX sigma, typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
n (size_t locx, size_t locy) const
{
	return make_local(locx,locy, F[locx].n(sigma,locy), 1., PROP::BOSONIC, PROP::HERMITIAN);
}

template<typename Symmetry, typename Scalar>
Mpo<Symmetry,Scalar> HubbardObservables<Symmetry,Scalar>::
n (size_t locx, size_t locy) const
{
	return make_local(locx,locy, F[locx].n(locy), 1., PROP::BOSONIC, PROP::HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<Dummy::IS_CHARGE_SU2(), Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
T (size_t locx, size_t locy, double factor) const
{
	return make_local(locx,locy, F[locx].T(locy), factor, PROP::BOSONIC, PROP::NON_HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<Dummy::IS_CHARGE_SU2(), Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
Tdag (size_t locx, size_t locy, double factor) const
{
	return make_local(locx,locy, F[locx].Tdag(locy), factor, PROP::BOSONIC, PROP::NON_HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(), Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
Tz (size_t locx, size_t locy) const
{
	return make_local(locx,locy, F[locx].Tz(locy), 1., PROP::BOSONIC, PROP::HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<Dummy::NO_CHARGE_SYM(), Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
Tx (size_t locx, size_t locy) const
{
	auto G = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx+locy)));
	return make_local(locx,locy, F[locx].Tx(locy,G), 1., PROP::BOSONIC, PROP::HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<Dummy::NO_CHARGE_SYM(), Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
iTy (size_t locx, size_t locy) const
{
	auto G = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx+locy)));
	return make_local(locx,locy, F[locx].iTy(locy,G), 1. , PROP::BOSONIC, PROP::HERMITIAN); //0.5*pow(-1,locx+locy)*(F[locx].cdagcdag(locy)-F[locx].cc(locy))
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(), Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
Tm (size_t locx, size_t locy) const
{
	auto G = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx+locy)));
	return make_local(locx,locy, F[locx].Tm(locy,G), 1., PROP::BOSONIC, PROP::NON_HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(), Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
Tp (size_t locx, size_t locy) const
{
	auto G = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx+locy)));
	return make_local(locx,locy, F[locx].Tp(locy,G), 1., PROP::BOSONIC, PROP::NON_HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(), Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
TpTm (size_t locx1, size_t locx2, size_t locy1, size_t locy2, double fac) const
{
	auto G1 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx1+locy1)));
	auto G2 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx2+locy2)));
	return make_corr(locx1,locx2,locy1,locy2, fac*F[locx1].Tp(locy1,G1), F[locx2].Tm(locy2,G2), Symmetry::qvacuum(), 1., PROP::NON_FERMIONIC, PROP::NON_HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(), Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
TmTp (size_t locx1, size_t locx2, size_t locy1, size_t locy2, double fac) const
{
	auto G1 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx1+locy1)));
	auto G2 = static_cast<SUB_LATTICE>(static_cast<int>(pow(-1,locx2+locy2)));
	return make_corr(locx1,locx2,locy1,locy2, fac*F[locx1].Tm(locy1,G1), F[locx2].Tp(locy2,G2), Symmetry::qvacuum(), 1., PROP::NON_FERMIONIC, PROP::NON_HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(), Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
TzTz (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	return make_corr(locx1,locx2,locy1,locy2, F[locx1].Tz(locy1), F[locx2].Tz(locy2), Symmetry::qvacuum(), 1., PROP::NON_FERMIONIC, PROP::HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::conditional<Dummy::IS_CHARGE_SU2(), Mpo<Symmetry,Scalar>, vector<Mpo<Symmetry,Scalar> > >::type HubbardObservables<Symmetry,Scalar>::
TdagT (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	if constexpr (Symmetry::IS_CHARGE_SU2())
				 {
					 return make_corr(locx1, locx2, locy1, locy2, F[locx1].Tdag(locy1), F[locx2].T(locy2), Symmetry::qvacuum(), sqrt(3.), PROP::BOSONIC, PROP::HERMITIAN);
					 // return make_corr("T†", "T", locx1, locx2, locy1, locy2, F[locx1].Tdag(locy1), F[locx2].T(locy2), Symmetry::qvacuum(), std::sqrt(3.), PROP::NON_FERMIONIC, PROP::HERMITIAN);
				 }
	else
	{
		vector<Mpo<Symmetry,Scalar> > out(3);
		out[0] = TzTz(locx1,locx2,locy1,locy2);
		out[1] = TpTm(locx1,locx2,locy1,locy2,0.5);
		out[2] = TmTp(locx1,locx2,locy1,locy2,0.5);
		return out;
	}
}

template<typename Symmetry, typename Scalar>
Mpo<Symmetry,Scalar> HubbardObservables<Symmetry,Scalar>::
ns (size_t locx, size_t locy) const
{
	return make_local(locx,locy, F[locx].ns(locy), 1., PROP::BOSONIC, PROP::HERMITIAN);
}

template<typename Symmetry, typename Scalar>
Mpo<Symmetry,Scalar> HubbardObservables<Symmetry,Scalar>::
nh (size_t locx, size_t locy) const
{
	return make_local(locx,locy, F[locx].nh(locy), 1., PROP::BOSONIC, PROP::HERMITIAN);
}

template<typename Symmetry, typename Scalar>
Mpo<Symmetry,Scalar> HubbardObservables<Symmetry,Scalar>::
nssq (size_t locx, size_t locy) const
{
	return make_local(locx,locy, F[locx].ns(locy) * F[locx].ns(locy), 1., PROP::BOSONIC, PROP::HERMITIAN);
}

template<typename Symmetry, typename Scalar>
Mpo<Symmetry,Scalar> HubbardObservables<Symmetry,Scalar>::
nhsq (size_t locx, size_t locy) const
{
	return make_local(locx,locy, F[locx].nh(locy) * F[locx].nh(locy), 1., PROP::BOSONIC, PROP::HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<SPIN_INDEX sigma1, SPIN_INDEX sigma2, typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
nn (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	return make_corr (locx1, locx2, locy1, locy2, F[locx1].n(sigma1,locy1), F[locx2].n(sigma2,locy2), Symmetry::qvacuum(), 1., PROP::NON_FERMIONIC, PROP::HERMITIAN);
}

template<typename Symmetry, typename Scalar>
Mpo<Symmetry,Scalar> HubbardObservables<Symmetry,Scalar>::
nn (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	return make_corr (locx1, locx2, locy1, locy2, F[locx1].n(locy1), F[locx2].n(locy2), Symmetry::qvacuum(), 1., PROP::NON_FERMIONIC, PROP::HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_CHARGE_SU2(),Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
hh (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	return make_corr(locx1,locx2,locy1,locy2, 
	                 F[locx1].d(locy1)-F[locx1].n(locy1)+F[locx1].Id(),
	                 F[locx2].d(locy2)-F[locx2].n(locy2)+F[locx2].Id(),
	                 Symmetry::qvacuum(), 1., PROP::NON_FERMIONIC, PROP::HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
Scomp (SPINOP_LABEL Sa, size_t locx, size_t locy, double factor) const
{
	bool HERMITIAN = (Sa==SX or Sa==SZ)? true:false;
	return make_local(locx,locy, F[locx].Scomp(Sa,locy), factor, PROP::BOSONIC, HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,complex<double> > >::type HubbardObservables<Symmetry,Scalar>::
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
	
	auto Op = F[locx].Rcomp(Sa,locy).template plain<complex<double> >();
	
	Mpo<Symmetry,complex<double>> Mout(F.size(), Op.Q, ss.str(), false);
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	
	Mout.setLocal(locx, Op);
	
	return Mout;
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
ScompScomp (SPINOP_LABEL Sa1, SPINOP_LABEL Sa2, size_t locx1, size_t locx2, size_t locy1, size_t locy2, double fac) const
{
	return make_corr(locx1,locx2,locy1,locy2, F[locx1].Scomp(Sa1,locy1), F[locx2].Scomp(Sa2,locy2), getQ_ScompScomp(Sa1,Sa2), fac, PROP::NON_FERMIONIC, PROP::HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
S (size_t locx, size_t locy, double factor) const
{
	return make_local(locx,locy, F[locx].S(locy), factor, PROP::BOSONIC, PROP::NON_HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
Sdag (size_t locx, size_t locy, double factor) const
{
	return make_local(locx,locy, F[locx].Sdag(locy), factor, PROP::BOSONIC, PROP::NON_HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
Stot (size_t locy, double factor, int dLphys) const
{
	vector<OperatorType> Ops(F.size());
	vector<double> factors(F.size());
	for (int l=0; l<F.size(); ++l)
	{
		Ops[l] = F[l].S(locy);
		factors[l] = 0.;
	}
	for (int l=0; l<F.size(); l+=dLphys)
	{
		factors[l] = factor;
	}
	return make_localSum(Ops, factors, PROP::NON_HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
Sdagtot (size_t locy, double factor, int dLphys) const
{
	vector<OperatorType> Ops(F.size());
	vector<double> factors(F.size());
	for (int l=0; l<F.size(); ++l)
	{
		Ops[l] = F[l].Sdag(locy);
		factors[l] = 0.;
	}
	for (int l=0; l<F.size(); l+=dLphys)
	{
		factors[l] = factor;
	}
	return make_localSum(Ops, factors, PROP::NON_HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type HubbardObservables<Symmetry,Scalar>::
Sz (size_t locx, size_t locy) const
{
	return Scomp(SZ,locx,locy);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::conditional<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar>, vector<Mpo<Symmetry,Scalar> > >::type HubbardObservables<Symmetry,Scalar>::
SdagS (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	if constexpr (Symmetry::IS_SPIN_SU2())
	{
		return make_corr(locx1, locx2, locy1, locy2, F[locx1].Sdag(locy1), F[locx2].S(locy2), Symmetry::qvacuum(), sqrt(3.), PROP::BOSONIC, PROP::HERMITIAN);
		// return make_corr("T†", "T", locx1, locx2, locy1, locy2, F[locx1].Tdag(locy1), F[locx2].T(locy2), Symmetry::qvacuum(), std::sqrt(3.), PROP::NON_FERMIONIC, PROP::HERMITIAN);
	}
	else
	{
		vector<Mpo<Symmetry,Scalar> > out(3);
		out[0] = SzSz(locx1,locx2,locy1,locy2);
		out[1] = SpSm(locx1,locx2,locy1,locy2,0.5);
		out[2] = SmSp(locx1,locx2,locy1,locy2,0.5);
		return out;
	}
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,complex<double> > >::type HubbardObservables<Symmetry,Scalar>::
S_ky (vector<complex<double> > phases) const
{
	vector<OperatorType> Ops(F.size());
	for (size_t l=0; l<F.size(); ++l)
	{
		Ops[l] = F[l].S(0);
	}
	return make_FourierYSum("S", Ops, 1., false, phases);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,complex<double> > >::type HubbardObservables<Symmetry,Scalar>::
Sdag_ky (vector<complex<double> > phases, double factor) const
{
	vector<OperatorType> Ops(F.size());
	for (size_t l=0; l<F.size(); ++l)
	{
		Ops[l] = F[l].Sdag(0);
	}
	return make_FourierYSum("S†", Ops, 1., false, phases);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<Dummy::IS_CHARGE_SU2(),Mpo<Symmetry,complex<double> > >::type HubbardObservables<Symmetry,Scalar>::
T_ky (vector<complex<double> > phases) const
{
	vector<OperatorType> Ops(F.size());
	for (size_t l=0; l<F.size(); ++l)
	{
		Ops[l] = F[l].T(0);
	}
	return make_FourierYSum("T", Ops, 1., false, phases);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<Dummy::IS_CHARGE_SU2(),Mpo<Symmetry,complex<double> > >::type HubbardObservables<Symmetry,Scalar>::
Tdag_ky (vector<complex<double> > phases, double factor) const
{
	vector<OperatorType> Ops(F.size());
	for (size_t l=0; l<F.size(); ++l)
	{
		Ops[l] = F[l].Tdag(0);
	}
	return make_FourierYSum("T†", Ops, 1., false, phases);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2() and !Dummy::IS_CHARGE_SU2(),Mpo<Symmetry,complex<double> > >::type HubbardObservables<Symmetry,Scalar>::
c_ky (vector<complex<double> > phases, double factor) const
{
	vector<OperatorType> Ops(F.size());
	for (size_t l=0; l<F.size(); ++l)
	{
		Ops[l] = F[l].c(0);
	}
	return make_FourierYSum("c", Ops, 1., false, phases);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2() and !Dummy::IS_CHARGE_SU2(),Mpo<Symmetry,complex<double> > >::type HubbardObservables<Symmetry,Scalar>::
cdag_ky (vector<complex<double> > phases, double factor) const
{
	vector<OperatorType> Ops(F.size());
	for (size_t l=0; l<F.size(); ++l)
	{
		Ops[l] = F[l].cdag(0);
	}
	return make_FourierYSum("c†", Ops, 1., false, phases);
}

template<typename Symmetry, typename Scalar>
typename Symmetry::qType HubbardObservables<Symmetry,Scalar>::getQ_ScompScomp(SPINOP_LABEL Sa1, SPINOP_LABEL Sa2) const
{
	typename Symmetry::qType out;
	if ( (Sa1 == SZ and Sa2 == SZ) or (Sa1 == SP and Sa2 == SM) or (Sa1 == SM and Sa2 == SP) or (Sa1 == SX or Sa1 == iSY) ) {out = Symmetry::qvacuum();}
	else {assert(false and "Quantumnumber for the chosen ScompScomp is not computed. Add in HubbardObservables::getQ_ScompScomp");}
	return out;
}
#endif
