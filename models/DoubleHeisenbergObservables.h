#ifndef DOUBLEHEISENBERGOBSERVABLES
#define DOUBLEHEISENBERGOBSERVABLES

#include "Mpo.h"
#include "ParamHandler.h" // from HELPERS
#include "bases/SpinBase.h"
//include "DmrgLinearAlgebra.h"
//include "DmrgExternal.h"

template<typename Symmetry>
class DoubleHeisenbergObservables
{
	typedef SiteOperatorQ<Symmetry,Eigen::MatrixXd> OperatorType;

public:
	
	///@{
	DoubleHeisenbergObservables(){};
	DoubleHeisenbergObservables (const size_t &L); // for inheritance purposes
	DoubleHeisenbergObservables (const size_t &L, const vector<Param> &params, const std::map<string,std::any> &defaults);
	///@}
	
	///@{
	template<size_t order = 0ul, typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type S (size_t locx, size_t locy=0, double factor=1.) const;
	
	template<size_t order = 0ul, typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type Sdag (size_t locx, size_t locy=0, double factor=std::sqrt(3.)) const;
	
	template<size_t order = 0ul, typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type Stot (size_t locy=0, double factor=1.) const;
	
	template<size_t order = 0ul, typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type Sdagtot (size_t locy=0, double factor=1.) const;
	
	template<size_t order = 0ul, typename Dummy = Symmetry>
	typename std::conditional<Dummy::IS_SPIN_SU2(), Mpo<Symmetry>, vector<Mpo<Symmetry> > >::type 
	SdagS (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	///@}
	
protected:
	
	Mpo<Symmetry> make_local (size_t locx, size_t locy,
	                          const OperatorType &Op,
	                          double factor =1.,
	                          bool HERMITIAN=false) const;
	template<size_t order=0ul> Mpo<Symmetry> make_localSum (const vector<OperatorType> &Op, vector<double> factor, bool HERMITIAN) const;
	template<size_t order=0ul> Mpo<Symmetry> make_corr  (size_t locx1, size_t locx2, size_t locy1, size_t locy2,
	                          const OperatorType &Op1, const OperatorType &Op2, qarray<Symmetry::Nq> Qtot,
	                          double factor, bool HERMITIAN) const;

//	Mpo<Symmetry,complex<double> >
//	make_FourierYSum (string name, const vector<OperatorType> &Ops, double factor, bool HERMITIAN, const vector<complex<double> > &phases) const;
//	
//	typename Symmetry::qType getQ_ScompScomp(SPINOP_LABEL Sa1, SPINOP_LABEL Sa2) const;
	
	vector<SpinBase<Symmetry,0ul>> B0;
	vector<SpinBase<Symmetry,1ul>> B1;
};

template<typename Symmetry>
DoubleHeisenbergObservables<Symmetry>::
DoubleHeisenbergObservables (const size_t &L)
{
	B0.resize(L);
	B1.resize(L);
}

template<typename Symmetry>
DoubleHeisenbergObservables<Symmetry>::
DoubleHeisenbergObservables (const size_t &L, const vector<Param> &params, const std::map<string,std::any> &defaults)
{
	ParamHandler P(params,defaults);
	size_t Lcell = P.size();
	B0.resize(L);
	B1.resize(L);
	
	for (size_t l=0; l<L; ++l)
	{
		B0[l] = SpinBase<Symmetry,0ul>(P.get<size_t>("Ly",l%Lcell), P.get<size_t>("D",l%Lcell));
		B1[l] = SpinBase<Symmetry,1ul>(P.get<size_t>("Ly",l%Lcell), P.get<size_t>("D",l%Lcell));
	}
}

template<typename Symmetry>
Mpo<Symmetry> DoubleHeisenbergObservables<Symmetry>::
make_local (size_t locx, size_t locy, const OperatorType &Op, double factor, bool HERMITIAN) const
{
	assert(locx<B0.size() and locy<B0[locx].dim());
	stringstream ss;
	ss << Op.label() << "(" << locx << "," << locy;
	if (factor != 1.) ss << ",factor=" << factor;
	ss << ")";
	
	Mpo<Symmetry> Mout(B0.size(), Op.Q(), ss.str(), HERMITIAN);
	for (size_t l=0; l<B0.size(); ++l) {Mout.setLocBasis(B0[l].get_basis().combine(B1[l].get_basis()).qloc(),l);}
	
	Mout.setLocal(locx, (factor * Op).template plain<double>());
	
	return Mout;
}

template<typename Symmetry>
template<size_t order>
Mpo<Symmetry> DoubleHeisenbergObservables<Symmetry>::
make_localSum (const vector<OperatorType> &Op, vector<double> factor, bool HERMITIAN) const
{
	assert(Op.size()==B0.size() and factor.size()==B0.size());
	stringstream ss;
	ss << Op[0].label() << "localSum";
	
	vector<OperatorType> OpExt(Op.size());
	for (int i=0; i<Op.size(); ++i)
	{
		if (order == 0ul)
		{
			OpExt[i] = kroneckerProduct(Op[i], B1[i].Id());
		}
		else if (order == 1ul)
		{
			OpExt[i] = kroneckerProduct(B0[i].Id(), Op[i]);
		}
	}
	
	Mpo<Symmetry> Mout(B0.size(), OpExt[0].Q(), ss.str(), HERMITIAN);
	for (size_t l=0; l<B0.size(); ++l) {Mout.setLocBasis(B0[l].get_basis().combine(B1[l].get_basis()).qloc(),l);}
	
	vector<SiteOperator<Symmetry,double>> Op_plain;
	for (int i=0; i<Op.size(); ++i)
	{
		Op_plain.push_back(OpExt[i].template plain<double>());
	}
	Mout.setLocalSum(Op_plain, factor);
	
	return Mout;
}

template<typename Symmetry>
template<size_t order>
Mpo<Symmetry> DoubleHeisenbergObservables<Symmetry>::
make_corr (size_t locx1, size_t locx2, size_t locy1, size_t locy2,
           const OperatorType &Op1, const OperatorType &Op2,
           qarray<Symmetry::Nq> Qtot,
           double factor, bool HERMITIAN) const
{
	assert(locx1<B0.size() and locy1<B0[locx1].dim());
	assert(locx2<B0.size() and locy2<B0[locx2].dim());
	
	stringstream ss;
	ss << Op1.label() << "(" << locx1 << "," << locy1 << ")"
	   << Op2.label() << "(" << locx2 << "," << locy2 << ")";
	
	Mpo<Symmetry> Mout(B0.size(), Qtot, ss.str(), HERMITIAN);
	for (size_t l=0; l<B0.size(); ++l) {Mout.setLocBasis(B0[l].get_basis().combine(B1[l].get_basis()).qloc(),l);}
	
	OperatorType Op1Ext;
	OperatorType Op2Ext;
	
	if (order == 0ul)
	{
		Op1Ext = kroneckerProduct(Op1, B1[locx1].Id());
		Op2Ext = kroneckerProduct(Op2, B1[locx2].Id());
	}
	else if (order == 1ul)
	{
		Op1Ext = kroneckerProduct(B0[locx1].Id(), Op1);
		Op2Ext = kroneckerProduct(B0[locx2].Id(), Op2);
	}
	
	if (locx1 == locx2)
	{
		auto product = factor * OperatorType::prod(Op1Ext, Op2Ext, Qtot);
		Mout.setLocal(locx1, product.template plain<double>());
	}
	else
	{
		Mout.setLocal({locx1, locx2}, {(factor*Op1Ext).template plain<double>(), Op2Ext.template plain<double>()});
	}
	
	return Mout;
}

template<typename Symmetry>
template<size_t order, typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type DoubleHeisenbergObservables<Symmetry>::
S (size_t locx, size_t locy, double factor) const
{
	return (order==0ul)?
		make_local(locx,locy, kroneckerProduct(B0[locx].S(locy),B1[locx].Id()), factor, PROP::NON_HERMITIAN):
		make_local(locx,locy, kroneckerProduct(B0[locx].Id(),B1[locx].S(locy)), factor, PROP::NON_HERMITIAN);
}

template<typename Symmetry>
template<size_t order, typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type DoubleHeisenbergObservables<Symmetry>::
Sdag (size_t locx, size_t locy, double factor) const
{
	return (order==0ul)? 
		make_local(locx,locy, kroneckerProduct(B0[locx].Sdag(locy),B1[locx].Id()), factor, PROP::NON_HERMITIAN):
		make_local(locx,locy, kroneckerProduct(B1[locx].Id(),B0[locx].Sdag(locy)), factor, PROP::NON_HERMITIAN);
}

template<typename Symmetry>
template<size_t order, typename Dummy>
typename std::conditional<Dummy::IS_SPIN_SU2(), Mpo<Symmetry>, vector<Mpo<Symmetry> > >::type DoubleHeisenbergObservables<Symmetry>::
SdagS (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	if constexpr (Symmetry::IS_SPIN_SU2())
	{
		return (order==0ul)? 
			make_corr<order>(locx1, locx2, locy1, locy2, B0[locx1].Sdag(locy1), B0[locx2].S(locy2), Symmetry::qvacuum(), sqrt(3.), PROP::HERMITIAN):
			make_corr<order>(locx1, locx2, locy1, locy2, B1[locx1].Sdag(locy1), B1[locx2].S(locy2), Symmetry::qvacuum(), sqrt(3.), PROP::HERMITIAN);
	}
//	else
//	{
//		vector<Mpo<Symmetry> > out(3);
//		out[0] = SzSz(locx1,locx2,locy1,locy2);
//		out[1] = SpSm(locx1,locx2,locy1,locy2,0.5);
//		out[2] = SmSp(locx1,locx2,locy1,locy2,0.5);
//		return out;
//	}
}

template<typename Symmetry>
template<size_t order, typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type DoubleHeisenbergObservables<Symmetry>::
Stot (size_t locy, double factor) const
{
	vector<OperatorType> Ops(B0.size());
	vector<double> factors(B0.size());
	for (int l=0; l<B0.size(); ++l)
	{
		if (order==0)
		{
			Ops[l] = B0[l].S(locy);
		}
		else
		{
			Ops[l] = B1[l].S(locy);
		}
		factors[l] = factor;
	}
	return make_localSum<order>(Ops, factors, PROP::NON_HERMITIAN);
}

template<typename Symmetry>
template<size_t order, typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type DoubleHeisenbergObservables<Symmetry>::
Sdagtot (size_t locy, double factor) const
{
	vector<OperatorType> Ops(B0.size());
	vector<double> factors(B0.size());
	for (int l=0; l<B0.size(); ++l)
	{
		if (order==0)
		{
			Ops[l] = B0[l].Sdag(locy);
		}
		else
		{
			Ops[l] = B1[l].Sdag(locy);
		}
		factors[l] = factor;
	}
	return make_localSum<order>(Ops, factors, PROP::NON_HERMITIAN);
}

#endif
