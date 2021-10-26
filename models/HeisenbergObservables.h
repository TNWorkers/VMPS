#ifndef HEISENBERGOBSERVABLES
#define HEISENBERGOBSERVABLES

#include "Mpo.h"
#include "ParamHandler.h" // from HELPERS
#include "bases/SpinBase.h"
//include "DmrgLinearAlgebra.h"
//include "DmrgExternal.h"
#include "Permutations.h"

template<typename Symmetry, typename Scalar=double>
class HeisenbergObservables
{
	typedef SiteOperatorQ<Symmetry,Eigen::Matrix<Scalar,Dynamic,Dynamic> > OperatorType;

public:
	
	///@{
	HeisenbergObservables(){};
	HeisenbergObservables (const size_t &L); // for inheritance purposes
	HeisenbergObservables (const size_t &L, const vector<Param> &params, const std::map<string,std::any> &defaults);
	///@}

		///@{
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type S (size_t locx, size_t locy=0, double factor=1.) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type Sdag (size_t locx, size_t locy=0, double factor=std::sqrt(3.)) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type Scomp (SPINOP_LABEL Sa, size_t locx, size_t locy=0, double factor=1.) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type Qcomp (SPINOP_LABEL Sa, size_t locx, size_t locy=0, double factor=1.) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type Sz (size_t locx, size_t locy=0) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type Sp (size_t locx, size_t locy=0) const {return Scomp(SP,locx,locy);};
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type Sm (size_t locx, size_t locy=0) const {return Scomp(SM,locx,locy);};
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type ScompScomp (SPINOP_LABEL Sa1, SPINOP_LABEL Sa2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type QcompQcomp (SPINOP_LABEL Sa1, SPINOP_LABEL Sa2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type SpSm (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const {return ScompScomp(SP,SM,locx1,locx2,locy1,locy2,fac);}
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type SmSp (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const {return ScompScomp(SM,SP,locx1,locx2,locy1,locy2,fac);}
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type SzSz (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const {return ScompScomp(SZ,SZ,locx1,locx2,locy1,locy2,1.);}
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::NO_SPIN_SYM(), Mpo<Symmetry,Scalar> >::type SxSx (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const {return ScompScomp(SX,SX,locx1,locx2,locy1,locy2,1.);}
	template<typename Dummy = Symmetry>
	typename std::conditional<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar>, vector<Mpo<Symmetry,Scalar> > >::type SdagS (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	template<typename Dummy = Symmetry>
	typename std::conditional<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar>, vector<Mpo<Symmetry,Scalar> > >::type QdagQ (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	template<typename Dummy = Symmetry>
	typename std::conditional<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar>, vector<Mpo<Symmetry,Scalar> > >::type SdagSxS (size_t locx1, size_t locx2, size_t locx3, size_t locy1=0, size_t locy2=0, size_t locy3=0) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type Stot (size_t locy1=0, double factor=1., int dLphys=1) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type Sdagtot (size_t locy1=0, double factor=std::sqrt(3.), int dLphys=1) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type Scomptot (SPINOP_LABEL Sa, size_t locy1=0, double factor=1., int dLphys=1) const;
	
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type Q (size_t locx, size_t locy=0, double factor=1.) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type Qdag (size_t locx, size_t locy=0, double factor=std::sqrt(5.)) const;

	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry, complex<double> > >::type Rcomp (SPINOP_LABEL Sa, size_t locx, size_t locy=0) const;
	///@}
	
	// ///@{
	// Mpo<Symmetry,Scalar> Scomp (SPINOP_LABEL Sa, size_t locx, size_t locy=0, double factor=1.) const;
	// Mpo<Symmetry,Scalar> ScompScomp (SPINOP_LABEL Sa1, SPINOP_LABEL Sa2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const;
	// Mpo<Symmetry,complex<double>> Rcomp (SPINOP_LABEL Sa, size_t locx, size_t locy=0) const;
	// ///@}
	
	// ///@{
	// Mpo<Symmetry,Scalar> Sz (size_t locx, size_t locy=0) const {return Scomp(SZ,locx,locy);};
	// Mpo<Symmetry,Scalar> Sx (size_t locx, size_t locy=0) const {return Scomp(SX,locx,locy);};
	// Mpo<Symmetry,Scalar> n  (size_t locx, size_t locy=0) const;
	// Mpo<Symmetry,Scalar> SzSz (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const {return ScompScomp(SZ,SZ,locx1,locx2,locy1,locy2,1.);};
	// Mpo<Symmetry,Scalar> SpSm (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const {return ScompScomp(SP,SM,locx1,locx2,locy1,locy2,fac);};
	// Mpo<Symmetry,Scalar> SmSp (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const {return ScompScomp(SM,SP,locx1,locx2,locy1,locy2,fac);};
	// Mpo<Symmetry,Scalar> SxSx (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const {return ScompScomp(SX,SX,locx1,locx2,locy1,locy2,fac);};
	// vector<Mpo<Symmetry,Scalar> > SdagS (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	// ///@}

	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,complex<double> > >::type S_ky    (vector<complex<double> > phases) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,complex<double> > >::type Sdag_ky (vector<complex<double> > phases, double factor=sqrt(3.)) const;
	
	///@{
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar> >::type String (STRING STR, size_t locx, size_t locy=0) const;
	
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar> >::type StringCorr (STRING STR, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	///@}
	
	typename Symmetry::qType getQ_ScompScomp(SPINOP_LABEL Sa1, SPINOP_LABEL Sa2) const
	{
		if (Symmetry::IS_TRIVIAL) {return Symmetry::qvacuum();}
		typename Symmetry::qType out;
		if ((Sa1 == SZ and Sa2 == SZ) or 
		    (Sa1 == SP and Sa2 == SM) or 
		    (Sa1 == SM and Sa2 == SP) or 
		    (Sa1 == SX and Sa1 == iSY) or 
		    (Sa1 == QZ and Sa2 == QZ) or 
		    (Sa1 == QP and Sa2 == QM) or 
		    (Sa1 == QM and Sa2 == QP) or 
		    (Sa1 == QPZ and Sa2 == QMZ) or 
		    (Sa1 == QMZ and Sa2 == QPZ)
		   )
		{
			out = Symmetry::qvacuum();
		}
		else
		{
			lout << "Sa1=" << Sa1 << ", Sa2=" << Sa2 << endl;
			assert(false and "Quantum number for the chosen ScompScomp is not computed. Add in HubbardObservables::getQ_ScompScomp");
		}
		return out;
	}
	
    std::vector<Mpo<Symmetry,Scalar>> make_spinPermutation (const Permutation& permutations) const;

    
    MpoTerms<Symmetry,double> spin_swap_operator_D2 (const std::size_t locx1, const std::size_t locx2, const std::size_t locy1=0ul, const std::size_t locy2=0ul) const;

    MpoTerms<Symmetry,double> spin_swap_operator_D3 (const std::size_t locx1, const std::size_t locx2, const std::size_t locy1=0ul, const std::size_t locy2=0ul) const;

protected:
	
	Mpo<Symmetry,Scalar> make_local (size_t locx, size_t locy,
	                          const OperatorType &Op,
	                          double factor =1.,
	                          bool HERMITIAN=false,
	                          STRING STR=NOSTRING) const;
	Mpo<Symmetry,Scalar> make_localSum (const vector<OperatorType> &Op, vector<Scalar> factor, bool HERMITIAN) const;
	Mpo<Symmetry,Scalar> make_corr  (size_t locx1, size_t locx2, size_t locy1, size_t locy2,
	                          const OperatorType &Op1, const OperatorType &Op2, qarray<Symmetry::Nq> Qtot,
	                          double factor, bool HERMITIAN,
	                          STRING STR=NOSTRING) const;

	Mpo<Symmetry,complex<double> >
	make_FourierYSum (string name, const vector<OperatorType> &Ops, double factor, bool HERMITIAN, const vector<complex<double> > &phases) const;
	
	vector<SpinBase<Symmetry> > B;
};

template<typename Symmetry, typename Scalar>
HeisenbergObservables<Symmetry,Scalar>::
HeisenbergObservables (const size_t &L)
{
	B.resize(L);
}

template<typename Symmetry, typename Scalar>
HeisenbergObservables<Symmetry,Scalar>::
HeisenbergObservables (const size_t &L, const vector<Param> &params, const std::map<string,std::any> &defaults)
{
	ParamHandler P(params,defaults);
	size_t Lcell = P.size();
	B.resize(L);
	
	for (size_t l=0; l<L; ++l)
	{
		B[l] = SpinBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), P.get<size_t>("D",l%Lcell));
	}
}

template<typename Symmetry, typename Scalar>
Mpo<Symmetry,Scalar> HeisenbergObservables<Symmetry,Scalar>::
make_local (size_t locx, size_t locy, const OperatorType &Op, double factor, bool HERMITIAN, STRING STR) const
{
	assert(locx<B.size() and locy<B[locx].dim());
	stringstream ss;
	ss << Op.label() << "(" << locx << "," << locy;
	if (factor != 1.) ss << ",factor=" << factor;
	ss << ")";
	
	Mpo<Symmetry,Scalar> Mout(B.size(), Op.Q(), ss.str(), HERMITIAN);
	for (size_t l=0; l<B.size(); ++l) {Mout.setLocBasis(B[l].get_basis().qloc(),l);}
	
	if (STR==NOSTRING)
	{
		Mout.setLocal(locx, (factor * Op).template cast<Scalar>(). template plain<Scalar>());
	}
	else
	{
		Mout.setLocal(locx, (factor * Op).template cast<Scalar>(). template plain<Scalar>(), 
		              B[0].bead(STR).template cast<Scalar>(). template plain<Scalar>());
	}
	
	return Mout;
}

template<typename Symmetry, typename Scalar>
Mpo<Symmetry,Scalar> HeisenbergObservables<Symmetry,Scalar>::
make_localSum (const vector<OperatorType> &Op, vector<Scalar> factor, bool HERMITIAN) const
{
	assert(Op.size()==B.size() and factor.size()==B.size());
	stringstream ss;
	ss << Op[0].label() << "localSum";
	
	Mpo<Symmetry,Scalar> Mout(B.size(), Op[0].Q(), ss.str(), HERMITIAN);
	for (size_t l=0; l<B.size(); ++l) {Mout.setLocBasis(B[l].get_basis().qloc(),l);}
	
	vector<SiteOperator<Symmetry,Scalar>> Op_plain;
	for (int i=0; i<Op.size(); ++i)
	{
		Op_plain.push_back(Op[i].template cast<Scalar>().template plain<Scalar>());
	}
	Mout.setLocalSum(Op_plain, factor);
	
	return Mout;
}

template<typename Symmetry, typename Scalar>
Mpo<Symmetry,Scalar> HeisenbergObservables<Symmetry,Scalar>::
make_corr (size_t locx1, size_t locx2, size_t locy1, size_t locy2,
           const OperatorType &Op1, const OperatorType &Op2,
           qarray<Symmetry::Nq> Qtot,
           double factor, bool HERMITIAN, 
           STRING STR) const
{
	assert(locx1<B.size() and locy1<B[locx1].dim());
	assert(locx2<B.size() and locy2<B[locx2].dim());
	
	stringstream ss;
	ss << Op1.label() << "(" << locx1 << "," << locy1 << ")"
	   << Op2.label() << "(" << locx2 << "," << locy2;
	if (factor != 1.) {ss << ",fac=" << factor;}
	ss << ")";
	
	Mpo<Symmetry,Scalar> Mout(B.size(), Qtot, ss.str(), HERMITIAN);
	for (size_t l=0; l<B.size(); ++l) {Mout.setLocBasis(B[l].get_basis().qloc(),l);}
	
	if (locx1 == locx2)
	{
		auto product = factor*OperatorType::prod(Op1, Op2, Qtot);
		Mout.setLocal(locx1, product.template cast<Scalar>().template plain<Scalar>());
	}
	else
	{
		if (STR==NOSTRING)
		{
			// Set dummySign carefully because basis might be site-dependent
			if (int(locx2)-int(locx1)-1 != 0)
			{
				vector<SiteOperator<Symmetry,Scalar> > dummySign(max(locx1,locx2)-min(locx1,locx2)-1);
				for (int i=0; i<dummySign.size(); ++i)
				{
					dummySign[i] = B[min(locx1,locx2)+i+1].Id().template cast<Scalar>().template plain<Scalar>();
				}
				Mout.setLocal(vector<size_t>{locx1, locx2}, 
				              vector<SiteOperator<Symmetry,Scalar> >{
				                  (factor*Op1).template cast<Scalar>().template plain<Scalar>(), 
				                          Op2 .template cast<Scalar>().template plain<Scalar>()},
				              dummySign);
			}
			else
			{
				Mout.setLocal(vector<size_t>{locx1, locx2}, 
				              vector<SiteOperator<Symmetry,Scalar> >{
				                  (factor*Op1).template cast<Scalar>().template plain<Scalar>(), 
				                          Op2 .template cast<Scalar>().template plain<Scalar>()});
			}
		}
		else
		{
			Mout.setLocal({locx1, locx2}, {(factor*Op1).template cast<Scalar>().template plain<Scalar>(), 
			                                       Op2 .template cast<Scalar>().template plain<Scalar>()}, 
			              B[0].bead(STR).template cast<Scalar>().template plain<Scalar>());
		}
	}
	
	return Mout;
}

template<typename Symmetry, typename Scalar>
Mpo<Symmetry,complex<double> > HeisenbergObservables<Symmetry,Scalar>::
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
	Mpo<Symmetry,complex<double> > Mout(B.size(), Ops[0].Q(), ss.str(), HERMITIAN);
	for (size_t l=0; l<B.size(); ++l) {Mout.setLocBasis(B[l].get_basis().qloc(),l);}
	
	vector<complex<double> > phases_x_factor = phases;
	for (int l=0; l<phases.size(); ++l)
	{
		phases_x_factor[l] = phases[l] * factor;
	}
	
	vector<SiteOperator<Symmetry,complex<double> > > OpsPlain(Ops.size());
	for (int l=0; l<OpsPlain.size(); ++l)
	{
		OpsPlain[l] = Ops[l].template cast<complex<double> >(). template plain<double>();
	}
	
	Mout.setLocalSum(OpsPlain, phases_x_factor);
	
	return Mout;
}

//-------------

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type HeisenbergObservables<Symmetry,Scalar>::
Scomp (SPINOP_LABEL Sa, size_t locx, size_t locy, double factor) const
{
	bool HERMITIAN = (Sa==SX or Sa==SZ)? true:false;
	return make_local(locx,locy, B[locx].Scomp(Sa,locy).template cast<Scalar>(), factor, HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type HeisenbergObservables<Symmetry,Scalar>::
Qcomp (SPINOP_LABEL Sa, size_t locx, size_t locy, double factor) const
{
	bool HERMITIAN = (Sa==QZ)? true:false;
	return make_local(locx,locy, B[locx].Qcomp(Sa,locy).template cast<Scalar>(), factor, HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,complex<double> > >::type HeisenbergObservables<Symmetry,Scalar>::
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
	
	auto Op = B[locx].Rcomp(Sa,locy).template plain<complex<double> >();
	
	Mpo<Symmetry,complex<double>> Mout(B.size(), Op.Q, ss.str(), false);
	for (size_t l=0; l<B.size(); ++l) {Mout.setLocBasis(B[l].get_basis().qloc(),l);}
	
	Mout.setLocal(locx, Op);
	
	return Mout;
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type HeisenbergObservables<Symmetry,Scalar>::
ScompScomp (SPINOP_LABEL Sa1, SPINOP_LABEL Sa2, size_t locx1, size_t locx2, size_t locy1, size_t locy2, double fac) const
{
	bool HERMITIAN = false;
	if ((Sa1 == SZ and Sa2 == SZ) or (Sa1 == SX and Sa2 == SX)) HERMITIAN = true;
	return make_corr(locx1,locx2,locy1,locy2, B[locx1].Scomp(Sa1,locy1).template cast<Scalar>(), B[locx2].Scomp(Sa2,locy2).template cast<Scalar>(), getQ_ScompScomp(Sa1,Sa2), fac, HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type HeisenbergObservables<Symmetry,Scalar>::
QcompQcomp (SPINOP_LABEL Sa1, SPINOP_LABEL Sa2, size_t locx1, size_t locx2, size_t locy1, size_t locy2, double fac) const
{
	bool HERMITIAN = false;
	if (Sa1 == QZ and Sa2 == QZ) HERMITIAN = true;
	return make_corr(locx1,locx2,locy1,locy2, B[locx1].Qcomp(Sa1,locy1).template cast<Scalar>(), B[locx2].Qcomp(Sa2,locy2).template cast<Scalar>(), getQ_ScompScomp(Sa1,Sa2), fac, HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type HeisenbergObservables<Symmetry,Scalar>::
S (size_t locx, size_t locy, double factor) const
{
	return make_local(locx,locy, B[locx].S(locy), factor, PROP::NON_HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type HeisenbergObservables<Symmetry,Scalar>::
Sdag (size_t locx, size_t locy, double factor) const
{
	return make_local(locx,locy, B[locx].Sdag(locy), factor, PROP::NON_HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type HeisenbergObservables<Symmetry,Scalar>::
Q (size_t locx, size_t locy, double factor) const
{
	return make_local(locx,locy, B[locx].Q(locy), factor, PROP::NON_HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type HeisenbergObservables<Symmetry,Scalar>::
Qdag (size_t locx, size_t locy, double factor) const
{
	return make_local(locx,locy, B[locx].Qdag(locy), factor, PROP::NON_HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type HeisenbergObservables<Symmetry,Scalar>::
Sz (size_t locx, size_t locy) const
{
	return Scomp(SZ,locx,locy);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type HeisenbergObservables<Symmetry,Scalar>::
Scomptot (SPINOP_LABEL Sa, size_t locy, double factor, int dLphys) const
{
	vector<OperatorType> Ops(B.size());
	vector<Scalar> factors(B.size());
	for (int l=0; l<B.size(); ++l)
	{
		Ops[l] = B[l].Scomp(Sa,locy).template cast<Scalar>();
		factors[l] = 0.;
	}
	for (int l=0; l<B.size(); l+=dLphys)
	{
		factors[l] = factor;
	}
	return make_localSum(Ops, factors, (Sa==SZ)?PROP::HERMITIAN:PROP::NON_HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type HeisenbergObservables<Symmetry,Scalar>::
Stot (size_t locy, double factor, int dLphys) const
{
	vector<OperatorType> Ops(B.size());
	vector<Scalar> factors(B.size());
	for (int l=0; l<B.size(); ++l)
	{
		Ops[l] = B[l].S(locy).template cast<Scalar>();
		factors[l] = 0.;
	}
	for (int l=0; l<B.size(); l+=dLphys)
	{
		factors[l] = factor;
	}
	return make_localSum(Ops, factors, PROP::NON_HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar> >::type HeisenbergObservables<Symmetry,Scalar>::
Sdagtot (size_t locy, double factor, int dLphys) const
{
	vector<OperatorType> Ops(B.size());
	vector<double> factors(B.size());
	for (int l=0; l<B.size(); ++l)
	{
		Ops[l] = B[l].Sdag(locy);
		factors[l] = 0.;
	}
	for (int l=0; l<B.size(); l+=dLphys)
	{
		factors[l] = factor;
	}
	return make_localSum(Ops, factors, PROP::NON_HERMITIAN);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::conditional<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar>, vector<Mpo<Symmetry,Scalar> > >::type HeisenbergObservables<Symmetry,Scalar>::
SdagS (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	if constexpr (Symmetry::IS_SPIN_SU2())
	{
		return make_corr(locx1, locx2, locy1, locy2, B[locx1].Sdag(locy1), B[locx2].S(locy2), Symmetry::qvacuum(), sqrt(3.), PROP::HERMITIAN);
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
typename std::conditional<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar>, vector<Mpo<Symmetry,Scalar> > >::type HeisenbergObservables<Symmetry,Scalar>::
QdagQ (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	if constexpr (Symmetry::IS_SPIN_SU2())
	{
		return make_corr(locx1, locx2, locy1, locy2, B[locx1].Qdag(locy1), B[locx2].Q(locy2), Symmetry::qvacuum(), sqrt(5.), PROP::HERMITIAN);
	}
	else
	{
		vector<Mpo<Symmetry,Scalar> > out(5);
		out[0] = QcompQcomp(QZ,QZ,locx1,locx2,locy1,locy2);
		out[1] = QcompQcomp(QP,QM,locx1,locx2,locy1,locy2,0.5);
		out[2] = QcompQcomp(QM,QP,locx1,locx2,locy1,locy2,0.5);
		out[3] = QcompQcomp(QPZ,QMZ,locx1,locx2,locy1,locy2,0.5);
		out[4] = QcompQcomp(QMZ,QPZ,locx1,locx2,locy1,locy2,0.5);
		return out;
	}
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::conditional<Dummy::IS_SPIN_SU2(), Mpo<Symmetry,Scalar>, vector<Mpo<Symmetry,Scalar> > >::type HeisenbergObservables<Symmetry,Scalar>::
SdagSxS (size_t locx1, size_t locx2, size_t locx3, size_t locy1, size_t locy2, size_t locy3) const
{
	if constexpr (Symmetry::IS_SPIN_SU2())
	{
		Mpo<Symmetry,double> Mout(B.size(), Symmetry::qvacuum(), "SdagSxS", PROP::NON_HERMITIAN, false, BC::OPEN, DMRG::VERBOSITY::HALFSWEEPWISE);
		for (size_t l=0; l<B.size(); ++l) {Mout.setLocBasis(B[l].get_basis().qloc(),l);}
		
		std::vector<typename Symmetry::qType> qList(B.size()+1);
		std::vector<SiteOperator<Symmetry,double>> opList(B.size());
		
		for (int i=0; i<qList.size(); ++i) {qList[i] = Symmetry::qvacuum();}
		for (int i=0; i<opList.size(); ++i) {opList[i] = B[i].Id().template plain<double>();}
		
		for (int i=0; i<B.size(); ++i)
		{
			if (i>=locx1 and i<locx3)
			{
				qList[1+i] = qarray<Symmetry::Nq>{3};
				if (i==locx1)
				{
					opList[i] = (B[i].S(i)).template plain<double>();
				}
				else if (i==locx2)
				{
					opList[i] = (B[i].S(i)).template plain<double>();
				}
			}
			else
			{
				opList[i] = (B[i].S(i)).template plain<double>();
			}
		}
		
		for (int i=0; i<qList.size(); ++i)
		{
			cout << "i=" << i << ", q=" << qList[i] << endl;
		}
		
		Mout.push_qpath(locx1, opList, qList, 1.);
		
		Mout.N_phys = B.size();
		Mout.finalize(PROP::COMPRESS, 1); // power=1
		Mout.precalc_TwoSiteData(true);
		
		return Mout;
	}
	else
	{
		lout << "SdagSxS is not implemented for this symmetry!" << endl;
		throw;
	}
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,complex<double> > >::type HeisenbergObservables<Symmetry,Scalar>::
S_ky (vector<complex<double> > phases) const
{
	vector<OperatorType> Ops(B.size());
	for (size_t l=0; l<B.size(); ++l)
	{
		Ops[l] = B[l].S(0);
	}
	return make_FourierYSum("S", Ops, 1., false, phases);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,complex<double> > >::type HeisenbergObservables<Symmetry,Scalar>::
Sdag_ky (vector<complex<double> > phases, double factor) const
{
	vector<OperatorType> Ops(B.size());
	for (size_t l=0; l<B.size(); ++l)
	{
		Ops[l] = B[l].Sdag(0);
	}
	return make_FourierYSum("S†", Ops, 1., false, phases);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar> >::type HeisenbergObservables<Symmetry,Scalar>::
String (STRING STR, size_t locx, size_t locy) const
{
	return make_local(locx, locy, B[locx].Scomp(STRING_TO_SPINOP(STR),locy), 1., false, STR);
}

template<typename Symmetry, typename Scalar>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry,Scalar> >::type HeisenbergObservables<Symmetry,Scalar>::
StringCorr (STRING STR, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	SPINOP_LABEL Sa = STRING_TO_SPINOP(STR);
	auto Qtot = B[locx1].Scomp(Sa,locy1).Q() + B[locx2].Scomp(Sa,locy2).Q();
	// factor -1 because of Sa*exp(i*pi*Sa) = -Sa
	return make_corr(locx1, locx2, locy1, locy2, B[locx1].Scomp(Sa,locy1), B[locx2].Scomp(Sa,locy2), Qtot, -1., false, STR);
}

// template<typename Symmetry, typename Scalar>
// Mpo<Symmetry,Scalar> HeisenbergObservables<Symmetry,Scalar>::
// Scomp (SPINOP_LABEL Sa, size_t locx, size_t locy, double factor) const
// {
// 	assert(locx<B.size() and locy<B[locx].dim());
// 	stringstream ss;
// 	ss << Sa << "(" << locx << "," << locy << ")";
	
// 	OperatorType Op = factor * B[locx].Scomp(Sa,locy);
	
// 	bool HERMITIAN = (Sa==SX or Sa==SZ)? true:false;
	
// 	Mpo<Symmetry,Scalar> Mout(B.size(), Op.Q, ss.str(), HERMITIAN);
// 	for (size_t l=0; l<B.size(); ++l) {Mout.setLocBasis(B[l].get_basis(),l);}
	
// 	Mout.setLocal(locx,Op);
// 	return Mout;
// }

// template<typename Symmetry, typename Scalar>
// Mpo<Symmetry,complex<double>> HeisenbergObservables<Symmetry,Scalar>::
// Rcomp (SPINOP_LABEL Sa, size_t locx, size_t locy) const
// {
// 	assert(locx<B.size() and locy<B[locx].dim());
// 	stringstream ss;
// 	if (Sa==iSY)
// 	{
// 		ss << "exp[1/S*π*" << Sa << "](" << locx << "," << locy << ")]";
// 	}
// 	else
// 	{
// 		ss << "exp[1/S*π*i" << Sa << "](" << locx << "," << locy << ")]";
// 	}
	
// 	auto Op = B[locx].Rcomp(Sa,locy);
	
// 	Mpo<Symmetry,complex<double>> Mout(B.size(), Op.Q, ss.str(), false);
// 	for (size_t l=0; l<B.size(); ++l) {Mout.setLocBasis(B[l].get_basis(),l);}
	
// 	Mout.setLocal(locx,Op);
// 	return Mout;
// }

// template<typename Symmetry, typename Scalar>
// Mpo<Symmetry,Scalar> HeisenbergObservables<Symmetry,Scalar>::
// n (size_t locx, size_t locy) const
// {
// 	assert(locx<B.size() and locy<B[locx].dim());
// 	stringstream ss;
// 	ss << "n(" << locx << "," << locy << ")";
	
// 	OperatorType Op = B[locx].n(locy);
	
// 	Mpo<Symmetry,Scalar> Mout(B.size(), Op.Q, ss.str(), true);
// 	for (size_t l=0; l<B.size(); ++l) {Mout.setLocBasis(B[l].get_basis(),l);}
	
// 	Mout.setLocal(locx,Op);
// 	return Mout;
// }

// template<typename Symmetry, typename Scalar>
// Mpo<Symmetry,Scalar> HeisenbergObservables<Symmetry,Scalar>::
// ScompScomp (SPINOP_LABEL Sa1, SPINOP_LABEL Sa2, size_t locx1, size_t locx2, size_t locy1, size_t locy2, double fac) const
// {
// 	assert(locx1<B.size() and locx2<B.size() and locy1<B[locx1].dim() and locy2<B[locx2].dim());
// 	stringstream ss;
// 	ss << Sa1 << "(" << locx1 << "," << locy1 << ")" << Sa2 << "(" << locx2 << "," << locy2 << ")";
	
// 	OperatorType Op1 = B[locx1].Scomp(Sa1,locy1);
// 	OperatorType Op2 = B[locx2].Scomp(Sa2,locy2);
	
// 	bool HERMITIAN = (Sa1==Sa2 and (Sa1==SZ or Sa1==SX) and locx1==locx2 and locy1==locy2)? true:false;
	
// 	Mpo<Symmetry,Scalar> Mout(B.size(), Op1.Q+Op2.Q, ss.str(), HERMITIAN);
// 	for (size_t l=0; l<B.size(); ++l) {Mout.setLocBasis(B[l].get_basis(),l);}
	
// 	Mout.setLocal({locx1,locx2}, {fac*Op1,Op2});
// 	return Mout;
// }

// template<typename Symmetry, typename Scalar>
// vector<Mpo<Symmetry,Scalar> >HeisenbergObservables<Symmetry,Scalar>::
// SdagS (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
// {
// 	vector<Mpo<Symmetry,Scalar> > out(3);
// 	out[0] = SzSz(locx1,locx2,locy1,locy2);
// 	out[1] = SpSm(locx1,locx2,locy1,locy2,0.5);
// 	out[2] = SmSp(locx1,locx2,locy1,locy2,0.5);
// 	return out;
// }


template<typename Symmetry, typename Scalar>
std::vector<Mpo<Symmetry,Scalar>> HeisenbergObservables<Symmetry,Scalar>::
make_spinPermutation (const Permutation& permutations) const
{
    auto check_locs = [this](std::size_t& locx1, std::size_t& locx2, std::size_t& locy1, std::size_t& locy2)
    {
        assert(locx1 < B.size() and locx2 < B.size());
        assert(locy1 < B[locx1].dim() and locy2 < B[locx2].dim());
        assert(locx1 != locx2 or locy1 != locy2);
        if(locx1 > locx2)
        {
            std::swap(locx1,locx2);
            std::swap(locy1,locy2);
        }
    };
    std::size_t D = B[0].get_D();
    assert(D == 2ul or D == 3ul);
    for (size_t loc=0; loc<B.size(); ++loc)
    {
        assert(B[loc].get_D() == D);
    }
    Stopwatch<> watch;
    std::vector<std::vector<Transposition>> transpositions = permutations.independentTranspositions();
    std::size_t divisions = transpositions.size();
    std::vector<Mpo<Symmetry,Scalar>> Mout(divisions);
    for(std::size_t div=0; div<divisions; ++div)
    {
        std::size_t locx1 = transpositions[div][0].source;
        std::size_t locx2 = transpositions[div][0].target;
        std::size_t locy1 = 0ul;
        std::size_t locy2 = 0ul;
        check_locs(locx1,locx2,locy1,locy2);
        MpoTerms<Symmetry,double> terms = (D == 2ul ? spin_swap_operator_D2(locx1,locx2,locy1,locy2) : spin_swap_operator_D3(locx1,locx2,locy1,locy2));
        for (std::size_t t=1; t<transpositions[div].size(); ++t)
        {
            std::size_t locx1 = transpositions[div][t].source;
            std::size_t locx2 = transpositions[div][t].target;
            std::size_t locy1 = 0ul;
            std::size_t locy2 = 0ul;
            check_locs(locx1,locx2,locy1,locy2);
            terms = (D == 2ul ? MpoTerms<Symmetry,double>::prod(spin_swap_operator_D2(locx1,locx2,locy1,locy2),terms,Symmetry::qvacuum()) : MpoTerms<Symmetry,double>::prod(spin_swap_operator_D3(locx1,locx2,locy1,locy2),terms,Symmetry::qvacuum()));
        }
        std::stringstream ss;
        ss << "Spin permutation";
        if(divisions>1)
        {
            ss << " " << div+1 << "/" << divisions;
        }
        ss << ": " << terms.get_name();
        terms.set_name(ss.str());
        Mout[div] = terms;
        Mout[div].UNITARY = true;
    }
    lout << "Construction of spin permutation operator: " << watch.info("Time") << std::endl;
    return Mout;
}

template<typename Symmetry, typename Scalar>
MpoTerms<Symmetry,double> HeisenbergObservables<Symmetry,Scalar>::
spin_swap_operator_D2 (const std::size_t locx1, const std::size_t locx2, const std::size_t locy1, const std::size_t locy2) const
{
    MpoTerms<Symmetry,double> Tout(B.size());
    for (size_t loc=0; loc<B.size(); ++loc)
    {
        Tout.setLocBasis(B[loc].get_basis().qloc(),loc);
    }
    if(locx1 == locx2)
    {
        SiteOperator<Symmetry,double> identity = B[locx1].Id().template plain<double>();
        Tout.push(locx1,{identity},0.5);
        if constexpr (Symmetry::IS_SPIN_SU2())
        {
            SiteOperator<Symmetry,double> SdagS = (OperatorType::prod(B[locx1].Sdag(locy1), B[locx1].S(locy2), {1})).template plain<double>();
            Tout.push(locx1,{SdagS},2.*std::sqrt(3.));
        }
        else
        {
            SiteOperator<Symmetry,double> SzSz = (OperatorType::prod(B[locx1].Sz(locy1), B[locx1].Sz(locy2), {0})).template plain<double>();
            SiteOperator<Symmetry,double> SpSm = (OperatorType::prod(B[locx1].Sp(locy1), B[locx1].Sm(locy2), {0})).template plain<double>();
            SiteOperator<Symmetry,double> SmSp = (OperatorType::prod(B[locx1].Sm(locy1), B[locx1].Sp(locy2), {0})).template plain<double>();
            Tout.push(locx1,{SzSz},2.);
            Tout.push(locx1,{SpSm},1.);
            Tout.push(locx1,{SmSp},1.);
        }
    }
    else
    {
        std::vector<SiteOperator<Symmetry,double>> opList(locx2-locx1+1);
        for(int j=0; j<-1+locx2-locx1; ++j)
        {
            opList[j+1] = (B[j].Id().template plain<double>());
        }
        SiteOperator<Symmetry,double> &first_op = opList[0];
        SiteOperator<Symmetry,double> &last_op = opList[locx2-locx1];
        first_op = B[locx1].Id().template plain<double>();
        last_op = B[locx2].Id().template plain<double>();
        Tout.push(locx1,opList,0.5);
        
        if constexpr (Symmetry::IS_SPIN_SU2())
        {
            first_op = (B[locx1].Sdag(locy1).template plain<double>());
            last_op = (B[locx2].S(locy2).template plain<double>());
            Tout.push(locx1,opList,2.*std::sqrt(3));
        }
        else
        {
            first_op = (B[locx1].Sz(locy1).template plain<double>());
            last_op = (B[locx2].Sz(locy2).template plain<double>());
            Tout.push(locx1,opList,2.);
            
            first_op = (B[locx1].Sp(locy1).template plain<double>());
            last_op = (B[locx2].Sm(locy2).template plain<double>());
            Tout.push(locx1,opList,1.);
            
            first_op = (B[locx1].Sm(locy1).template plain<double>());
            last_op = (B[locx2].Sp(locy2).template plain<double>());
            Tout.push(locx1,opList,1.);
        }
    }
    Tout.finalize(true,1);
    std::stringstream ss;
    ss << "(" << locx1 << "<->" << locx2 << ")";
    Tout.set_name(ss.str());
    return Tout;
}

template<typename Symmetry, typename Scalar>
MpoTerms<Symmetry,double> HeisenbergObservables<Symmetry,Scalar>::
spin_swap_operator_D3 (const std::size_t locx1, const std::size_t locx2, const std::size_t locy1, const std::size_t locy2) const
{
    MpoTerms<Symmetry,double> Tout(B.size());
    for (size_t loc=0; loc<B.size(); ++loc)
    {
        Tout.setLocBasis(B[loc].get_basis().qloc(),loc);
    }
    if(locx1 == locx2)
    {
        SiteOperator<Symmetry,double> identity = B[locx1].Id().template plain<double>();
        Tout.push(locx1,{identity},-1.);
        if constexpr (Symmetry::IS_SPIN_SU2())
        {
            SiteOperator<Symmetry,double> SdagS = (OperatorType::prod(B[locx1].Sdag(locy1), B[locx1].S(locy2), {1})).template plain<double>();
            Tout.push(locx1,{SdagS},std::sqrt(3.));
            
            OperatorType SdagSdag_singl = OperatorType::prod(B[locx1].Sdag(locy1), B[locx1].Sdag(locy1), {1});
            OperatorType SS_singl = OperatorType::prod(B[locx2].S(locy2), B[locx2].S(locy2), {1});
            SiteOperator<Symmetry,double> SdagSdagSS_singl = (OperatorType::prod(SdagSdag_singl, SS_singl, {1})).template plain<double>();
            Tout.push(locx1,{SdagSdagSS_singl},1.);
            
            OperatorType SdagSdag_tripl = OperatorType::prod(B[locx1].Sdag(locy1), B[locx1].Sdag(locy1), {3});
            OperatorType SS_tripl = OperatorType::prod(B[locx2].S(locy2), B[locx2].S(locy2), {3});
            SiteOperator<Symmetry,double> SdagSdagSS_tripl = (OperatorType::prod(SdagSdag_tripl, SS_tripl, {1})).template plain<double>();
            Tout.push(locx1,{SdagSdagSS_tripl},std::sqrt(3.));
            
            OperatorType SdagSdag_quint = OperatorType::prod(B[locx1].Sdag(locy1), B[locx1].Sdag(locy1), {5});
            OperatorType SS_quint = OperatorType::prod(B[locx2].S(locy2), B[locx2].S(locy2), {5});
            SiteOperator<Symmetry,double> SdagSdagSS_quint = (OperatorType::prod(SdagSdag_quint, SS_quint, {1})).template plain<double>();
            Tout.push(locx1,{SdagSdagSS_quint},std::sqrt(5.));
        }
        else
        {
            SiteOperator<Symmetry,double> SzSz = (OperatorType::prod(B[locx1].Sz(locy1), B[locx1].Sz(locy2), {0})).template plain<double>();
            SiteOperator<Symmetry,double> SpSm = (OperatorType::prod(B[locx1].Sp(locy1), B[locx1].Sm(locy2), {0})).template plain<double>();
            SiteOperator<Symmetry,double> SmSp = (OperatorType::prod(B[locx1].Sm(locy1), B[locx1].Sp(locy2), {0})).template plain<double>();
            Tout.push(locx1,{SzSz},1.);
            Tout.push(locx1,{SpSm},0.5);
            Tout.push(locx1,{SmSp},0.5);
            
            OperatorType SzSz_1 = OperatorType::prod(B[locx1].Sz(locy1), B[locx1].Sz(locy1), {0});
            OperatorType SzSp_1 = OperatorType::prod(B[locx1].Sz(locy1), B[locx1].Sp(locy1), {2});
            OperatorType SzSm_1 = OperatorType::prod(B[locx1].Sz(locy1), B[locx1].Sm(locy1), {-2});
            OperatorType SpSz_1 = OperatorType::prod(B[locx1].Sp(locy1), B[locx1].Sz(locy1), {2});
            OperatorType SpSp_1 = OperatorType::prod(B[locx1].Sp(locy1), B[locx1].Sp(locy1), {4});
            OperatorType SpSm_1 = OperatorType::prod(B[locx1].Sp(locy1), B[locx1].Sm(locy1), {0});
            OperatorType SmSz_1 = OperatorType::prod(B[locx1].Sm(locy1), B[locx1].Sz(locy1), {-2});
            OperatorType SmSp_1 = OperatorType::prod(B[locx1].Sm(locy1), B[locx1].Sp(locy1), {0});
            OperatorType SmSm_1 = OperatorType::prod(B[locx1].Sm(locy1), B[locx1].Sm(locy1), {-4});
            OperatorType SzSz_2 = OperatorType::prod(B[locx2].Sz(locy2), B[locx2].Sz(locy2), {0});
            OperatorType SzSp_2 = OperatorType::prod(B[locx2].Sz(locy2), B[locx2].Sp(locy2), {2});
            OperatorType SzSm_2 = OperatorType::prod(B[locx2].Sz(locy2), B[locx2].Sm(locy2), {-2});
            OperatorType SpSz_2 = OperatorType::prod(B[locx2].Sp(locy2), B[locx2].Sz(locy2), {2});
            OperatorType SpSp_2 = OperatorType::prod(B[locx2].Sp(locy2), B[locx2].Sp(locy2), {4});
            OperatorType SpSm_2 = OperatorType::prod(B[locx2].Sp(locy2), B[locx2].Sm(locy2), {0});
            OperatorType SmSz_2 = OperatorType::prod(B[locx2].Sm(locy2), B[locx2].Sz(locy2), {-2});
            OperatorType SmSp_2 = OperatorType::prod(B[locx2].Sm(locy2), B[locx2].Sp(locy2), {0});
            OperatorType SmSm_2 = OperatorType::prod(B[locx2].Sm(locy2), B[locx2].Sm(locy2), {-4});
            
            SiteOperator<Symmetry,double> SzSzSzSz = (OperatorType::prod(SzSz_1,SzSz_2,{0})).template plain<double>();
            SiteOperator<Symmetry,double> SzSpSzSm = (OperatorType::prod(SzSp_1,SzSm_2,{0})).template plain<double>();
            SiteOperator<Symmetry,double> SpSzSmSz = (OperatorType::prod(SpSz_1,SmSz_2,{0})).template plain<double>();
            SiteOperator<Symmetry,double> SzSmSzSp = (OperatorType::prod(SzSm_1,SzSp_2,{0})).template plain<double>();
            SiteOperator<Symmetry,double> SmSzSpSz = (OperatorType::prod(SmSz_1,SpSz_2,{0})).template plain<double>();
            SiteOperator<Symmetry,double> SpSpSmSm = (OperatorType::prod(SpSp_1,SmSm_2,{0})).template plain<double>();
            SiteOperator<Symmetry,double> SmSmSpSp = (OperatorType::prod(SmSm_1,SpSp_2,{0})).template plain<double>();
            SiteOperator<Symmetry,double> SpSmSmSp = (OperatorType::prod(SpSm_1,SmSp_2,{0})).template plain<double>();
            SiteOperator<Symmetry,double> SmSpSpSm = (OperatorType::prod(SmSp_1,SpSm_2,{0})).template plain<double>();
            
            Tout.push(locx1,{SzSzSzSz},1.);
            Tout.push(locx1,{SzSpSzSm},0.5);
            Tout.push(locx1,{SzSmSzSp},0.5);
            Tout.push(locx1,{SpSzSmSz},0.5);
            Tout.push(locx1,{SmSzSpSz},0.5);
            Tout.push(locx1,{SpSmSmSp},0.25);
            Tout.push(locx1,{SmSpSpSm},0.25);
            Tout.push(locx1,{SpSpSmSm},0.25);
            Tout.push(locx1,{SmSmSpSp},0.25);
        }
    }
    else
    {
        std::vector<SiteOperator<Symmetry,double>> opList(locx2-locx1+1);
        for(int j=0; j<-1+locx2-locx1; ++j)
        {
            opList[j+1] = (B[j].Id().template plain<double>());
        }
        SiteOperator<Symmetry,double> &first_op = opList[0];
        SiteOperator<Symmetry,double> &last_op = opList[locx2-locx1];
        first_op = B[locx1].Id().template plain<double>();
        last_op = B[locx2].Id().template plain<double>();
        Tout.push(locx1,opList,-1.);
        
        if constexpr (Symmetry::IS_SPIN_SU2())
        {
            first_op = (B[locx1].Sdag(locy1).template plain<double>());
            last_op = (B[locx2].S(locy2).template plain<double>());
            Tout.push(locx1,opList,std::sqrt(3.));
            
            first_op = (OperatorType::prod(B[locx1].Sdag(locy1), B[locx1].Sdag(locy1), {1})).template plain<double>();
            last_op = (OperatorType::prod(B[locx2].S(locy2), B[locx2].S(locy2), {1})).template plain<double>();
            Tout.push(locx1,opList,1.);
            
            first_op = (OperatorType::prod(B[locx1].Sdag(locy1), B[locx1].Sdag(locy1), {3})).template plain<double>();
            last_op = (OperatorType::prod(B[locx2].S(locy2), B[locx2].S(locy2), {3})).template plain<double>();
            Tout.push(locx1,opList,std::sqrt(3.));
            
            first_op = (OperatorType::prod(B[locx1].Sdag(locy1), B[locx1].Sdag(locy1), {5})).template plain<double>();
            last_op = (OperatorType::prod(B[locx2].S(locy2), B[locx2].S(locy2), {5})).template plain<double>();
            Tout.push(locx1,opList,std::sqrt(5.));
        }
        else
        {
            first_op = (B[locx1].Sz(locy1).template plain<double>());
            last_op = (B[locx2].Sz(locy2).template plain<double>());
            Tout.push(locx1,opList,1.);
            
            first_op = (B[locx1].Sp(locy1).template plain<double>());
            last_op = (B[locx2].Sm(locy2).template plain<double>());
            Tout.push(locx1,opList,0.5);
            
            first_op = (B[locx1].Sm(locy1).template plain<double>());
            last_op = (B[locx2].Sp(locy2).template plain<double>());
            Tout.push(locx1,opList,0.5);
            
            first_op = (OperatorType::prod(B[locx1].Sz(locy1), B[locx1].Sz(locy1), {0})).template plain<double>();
            last_op = (OperatorType::prod(B[locx2].Sz(locy2), B[locx2].Sz(locy2), {0})).template plain<double>();
            Tout.push(locx1,opList,1.);
            
            first_op = (OperatorType::prod(B[locx1].Sz(locy1), B[locx1].Sp(locy1), {2})).template plain<double>();
            last_op = (OperatorType::prod(B[locx2].Sz(locy2), B[locx2].Sm(locy2), {-2})).template plain<double>();
            Tout.push(locx1,opList,0.5);
            
            first_op = (OperatorType::prod(B[locx1].Sz(locy1), B[locx1].Sm(locy1), {-2})).template plain<double>();
            last_op = (OperatorType::prod(B[locx2].Sz(locy2), B[locx2].Sp(locy2), {2})).template plain<double>();
            Tout.push(locx1,opList,0.5);
            
            first_op = (OperatorType::prod(B[locx1].Sp(locy1), B[locx1].Sz(locy1), {2})).template plain<double>();
            last_op = (OperatorType::prod(B[locx2].Sm(locy2), B[locx2].Sz(locy2), {-2})).template plain<double>();
            Tout.push(locx1,opList,0.5);
            
            first_op = (OperatorType::prod(B[locx1].Sp(locy1), B[locx1].Sp(locy1), {4})).template plain<double>();
            last_op = (OperatorType::prod(B[locx2].Sm(locy2), B[locx2].Sm(locy2), {-4})).template plain<double>();
            Tout.push(locx1,opList,0.25);
            
            first_op = (OperatorType::prod(B[locx1].Sp(locy1), B[locx1].Sm(locy1), {0})).template plain<double>();
            last_op = (OperatorType::prod(B[locx2].Sm(locy2), B[locx2].Sp(locy2), {0})).template plain<double>();
            Tout.push(locx1,opList,0.25);
            
            first_op = (OperatorType::prod(B[locx1].Sm(locy1), B[locx1].Sz(locy1), {-2})).template plain<double>();
            last_op = (OperatorType::prod(B[locx2].Sp(locy2), B[locx2].Sz(locy2), {2})).template plain<double>();
            Tout.push(locx1,opList,0.5);
            
            first_op = (OperatorType::prod(B[locx1].Sm(locy1), B[locx1].Sp(locy1), {0})).template plain<double>();
            last_op = (OperatorType::prod(B[locx2].Sp(locy2), B[locx2].Sm(locy2), {0})).template plain<double>();
            Tout.push(locx1,opList,0.25);
            
            first_op = (OperatorType::prod(B[locx1].Sm(locy1), B[locx1].Sm(locy1), {-4})).template plain<double>();
            last_op = (OperatorType::prod(B[locx2].Sp(locy2), B[locx2].Sp(locy2), {4})).template plain<double>();
            Tout.push(locx1,opList,0.25);
        }
    }
    Tout.finalize(true,1);
    std::stringstream ss;
    ss << "(" << locx1 << "<->" << locx2 << ")";
    Tout.set_name(ss.str());
    return Tout;
}
#endif
