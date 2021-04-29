#ifndef HEISENBERGOBSERVABLES
#define HEISENBERGOBSERVABLES

#include "Mpo.h"
#include "ParamHandler.h" // from HELPERS
#include "bases/SpinBase.h"
//include "DmrgLinearAlgebra.h"
//include "DmrgExternal.h"


template<typename Symmetry>
class HeisenbergObservables
{
	typedef SiteOperatorQ<Symmetry,Eigen::MatrixXd> OperatorType;

public:
	
	///@{
	HeisenbergObservables(){};
	HeisenbergObservables (const size_t &L); // for inheritance purposes
	HeisenbergObservables (const size_t &L, const vector<Param> &params, const std::map<string,std::any> &defaults);
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
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type SpSm (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const {return ScompScomp(SP,SM,locx1,locx2,locy1,locy2,fac);}
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type SmSp (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const {return ScompScomp(SM,SP,locx1,locx2,locy1,locy2,fac);}
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type SzSz (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const {return ScompScomp(SZ,SZ,locx1,locx2,locy1,locy2,1.);}
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::NO_SPIN_SYM(), Mpo<Symmetry> >::type SxSx (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const {return ScompScomp(SX,SX,locx1,locx2,locy1,locy2,1.);}
	template<typename Dummy = Symmetry>
	typename std::conditional<Dummy::IS_SPIN_SU2(), Mpo<Symmetry>, vector<Mpo<Symmetry> > >::type SdagS (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	template<typename Dummy = Symmetry>
	typename std::conditional<Dummy::IS_SPIN_SU2(), Mpo<Symmetry>, vector<Mpo<Symmetry> > >::type SdagSxS (size_t locx1, size_t locx2, size_t locx3, size_t locy1=0, size_t locy2=0, size_t locy3=0) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type Stot (size_t locy1=0, double factor=1., int dLphys=1) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type Sdagtot (size_t locy1=0, double factor=std::sqrt(3.), int dLphys=1) const;
	
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type Q (size_t locx, size_t locy=0, double factor=1.) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type Qdag (size_t locx, size_t locy=0, double factor=std::sqrt(5.)) const;

	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry, complex<double> > >::type Rcomp (SPINOP_LABEL Sa, size_t locx, size_t locy=0) const;
	///@}
	
	// ///@{
	// Mpo<Symmetry> Scomp (SPINOP_LABEL Sa, size_t locx, size_t locy=0, double factor=1.) const;
	// Mpo<Symmetry> ScompScomp (SPINOP_LABEL Sa1, SPINOP_LABEL Sa2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const;
	// Mpo<Symmetry,complex<double>> Rcomp (SPINOP_LABEL Sa, size_t locx, size_t locy=0) const;
	// ///@}
	
	// ///@{
	// Mpo<Symmetry> Sz (size_t locx, size_t locy=0) const {return Scomp(SZ,locx,locy);};
	// Mpo<Symmetry> Sx (size_t locx, size_t locy=0) const {return Scomp(SX,locx,locy);};
	// Mpo<Symmetry> n  (size_t locx, size_t locy=0) const;
	// Mpo<Symmetry> SzSz (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const {return ScompScomp(SZ,SZ,locx1,locx2,locy1,locy2,1.);};
	// Mpo<Symmetry> SpSm (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const {return ScompScomp(SP,SM,locx1,locx2,locy1,locy2,fac);};
	// Mpo<Symmetry> SmSp (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const {return ScompScomp(SM,SP,locx1,locx2,locy1,locy2,fac);};
	// Mpo<Symmetry> SxSx (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const {return ScompScomp(SX,SX,locx1,locx2,locy1,locy2,fac);};
	// vector<Mpo<Symmetry> > SdagS (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	// ///@}

	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,complex<double> > >::type S_ky    (vector<complex<double> > phases) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,complex<double> > >::type Sdag_ky (vector<complex<double> > phases, double factor=sqrt(3.)) const;
	
	///@{
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type Stringz (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	///@}
	
	typename Symmetry::qType getQ_ScompScomp(SPINOP_LABEL Sa1, SPINOP_LABEL Sa2) const
	{
		typename Symmetry::qType out;
		if ( (Sa1 == SZ and Sa2 == SZ) or (Sa1 == SP and Sa2 == SM) or (Sa1 == SM and Sa2 == SP) or (Sa1 == SX or Sa1 == iSY) ) {out = Symmetry::qvacuum();}
		else {assert(false and "Quantumnumber for the chosen ScompScomp is not computed. Add in HubbardObservables::getQ_ScompScomp");}
		return out;
	}
	
protected:
	
	Mpo<Symmetry> make_local (size_t locx, size_t locy,
	                          const OperatorType &Op,
							  double factor =1.,
	                          bool HERMITIAN=false) const;
	Mpo<Symmetry> make_localSum (const vector<OperatorType> &Op, vector<double> factor, bool HERMITIAN) const;
	Mpo<Symmetry> make_corr  (size_t locx1, size_t locx2, size_t locy1, size_t locy2,
	                          const OperatorType &Op1, const OperatorType &Op2, qarray<Symmetry::Nq> Qtot,
	                          double factor, bool HERMITIAN) const;

	Mpo<Symmetry,complex<double> >
	make_FourierYSum (string name, const vector<OperatorType> &Ops, double factor, bool HERMITIAN, const vector<complex<double> > &phases) const;
	
	vector<SpinBase<Symmetry> > B;
};

template<typename Symmetry>
HeisenbergObservables<Symmetry>::
HeisenbergObservables (const size_t &L)
{
	B.resize(L);
}

template<typename Symmetry>
HeisenbergObservables<Symmetry>::
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

template<typename Symmetry>
Mpo<Symmetry> HeisenbergObservables<Symmetry>::
make_local (size_t locx, size_t locy, const OperatorType &Op, double factor, bool HERMITIAN) const
{
	assert(locx<B.size() and locy<B[locx].dim());
	stringstream ss;
	ss << Op.label() << "(" << locx << "," << locy;
	if (factor != 1.) ss << ",factor=" << factor;
	ss << ")";
	
	Mpo<Symmetry> Mout(B.size(), Op.Q(), ss.str(), HERMITIAN);
	for (size_t l=0; l<B.size(); ++l) {Mout.setLocBasis(B[l].get_basis().qloc(),l);}
	
	Mout.setLocal(locx, (factor * Op).template plain<double>());
	
	return Mout;
}

template<typename Symmetry>
Mpo<Symmetry> HeisenbergObservables<Symmetry>::
make_localSum (const vector<OperatorType> &Op, vector<double> factor, bool HERMITIAN) const
{
	assert(Op.size()==B.size() and factor.size()==B.size());
	stringstream ss;
	ss << Op[0].label() << "localSum";
	
	Mpo<Symmetry> Mout(B.size(), Op[0].Q(), ss.str(), HERMITIAN);
	for (size_t l=0; l<B.size(); ++l) {Mout.setLocBasis(B[l].get_basis().qloc(),l);}
	
	vector<SiteOperator<Symmetry,double>> Op_plain;
	for (int i=0; i<Op.size(); ++i)
	{
		Op_plain.push_back(Op[i].template plain<double>());
	}
	Mout.setLocalSum(Op_plain, factor);
	
	return Mout;
}

template<typename Symmetry>
Mpo<Symmetry> HeisenbergObservables<Symmetry>::
make_corr (size_t locx1, size_t locx2, size_t locy1, size_t locy2,
           const OperatorType &Op1, const OperatorType &Op2,
		   qarray<Symmetry::Nq> Qtot,
           double factor, bool HERMITIAN) const	
{
	assert(locx1<B.size() and locy1<B[locx1].dim());
	assert(locx2<B.size() and locy2<B[locx2].dim());
	
	stringstream ss;
	ss << Op1.label() << "(" << locx1 << "," << locy1 << ")"
	   << Op2.label() << "(" << locx2 << "," << locy2 << ")";
	
	Mpo<Symmetry> Mout(B.size(), Qtot, ss.str(), HERMITIAN);
	for (size_t l=0; l<B.size(); ++l) {Mout.setLocBasis(B[l].get_basis().qloc(),l);}
	
	if (locx1 == locx2)
	{
		auto product = factor*OperatorType::prod(Op1, Op2, Qtot);
		Mout.setLocal(locx1, product.template plain<double>());
	}
	else
	{
		Mout.setLocal({locx1, locx2}, {(factor*Op1).template plain<double>(), Op2.template plain<double>()});
	}
	
	return Mout;
}

template<typename Symmetry>
Mpo<Symmetry,complex<double> > HeisenbergObservables<Symmetry>::
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
		OpsPlain[l] = Ops[l].template plain<double>().template cast<complex<double> >();
	}
	
	Mout.setLocalSum(OpsPlain, phases_x_factor);
	
	return Mout;
}

//-------------

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type HeisenbergObservables<Symmetry>::
Scomp (SPINOP_LABEL Sa, size_t locx, size_t locy, double factor) const
{
	bool HERMITIAN = (Sa==SX or Sa==SZ)? true:false;
	return make_local(locx,locy, B[locx].Scomp(Sa,locy), factor, HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry,complex<double> > >::type HeisenbergObservables<Symmetry>::
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

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type HeisenbergObservables<Symmetry>::
ScompScomp (SPINOP_LABEL Sa1, SPINOP_LABEL Sa2, size_t locx1, size_t locx2, size_t locy1, size_t locy2, double fac) const
{
	return make_corr(locx1,locx2,locy1,locy2, B[locx1].Scomp(Sa1,locy1), B[locx2].Scomp(Sa2,locy2), getQ_ScompScomp(Sa1,Sa2), fac, PROP::HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type HeisenbergObservables<Symmetry>::
S (size_t locx, size_t locy, double factor) const
{
	return make_local(locx,locy, B[locx].S(locy), factor, PROP::NON_HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type HeisenbergObservables<Symmetry>::
Sdag (size_t locx, size_t locy, double factor) const
{
	return make_local(locx,locy, B[locx].Sdag(locy), factor, PROP::NON_HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type HeisenbergObservables<Symmetry>::
Q (size_t locx, size_t locy, double factor) const
{
	return make_local(locx,locy, B[locx].Q(locy), factor, PROP::NON_HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type HeisenbergObservables<Symmetry>::
Qdag (size_t locx, size_t locy, double factor) const
{
	return make_local(locx,locy, B[locx].Qdag(locy), factor, PROP::NON_HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type HeisenbergObservables<Symmetry>::
Sz (size_t locx, size_t locy) const
{
	return Scomp(SZ,locx,locy);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type HeisenbergObservables<Symmetry>::
Stot (size_t locy, double factor, int dLphys) const
{
	vector<OperatorType> Ops(B.size());
	vector<double> factors(B.size());
	for (int l=0; l<B.size(); ++l)
	{
		Ops[l] = B[l].S(locy);
		factors[l] = 0.;
	}
	for (int l=0; l<B.size(); l+=dLphys)
	{
		factors[l] = factor;
	}
	return make_localSum(Ops, factors, PROP::NON_HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type HeisenbergObservables<Symmetry>::
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

template<typename Symmetry>
template<typename Dummy>
typename std::conditional<Dummy::IS_SPIN_SU2(), Mpo<Symmetry>, vector<Mpo<Symmetry> > >::type HeisenbergObservables<Symmetry>::
SdagS (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	if constexpr (Symmetry::IS_SPIN_SU2())
	 {
		 return make_corr(locx1, locx2, locy1, locy2, B[locx1].Sdag(locy1), B[locx2].S(locy2), Symmetry::qvacuum(), sqrt(3.), PROP::HERMITIAN);
		 // return make_corr("T†", "T", locx1, locx2, locy1, locy2, B[locx1].Tdag(locy1), B[locx2].T(locy2), Symmetry::qvacuum(), std::sqrt(3.), PROP::NON_FERMIONIC, PROP::HERMITIAN);
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
typename std::conditional<Dummy::IS_SPIN_SU2(), Mpo<Symmetry>, vector<Mpo<Symmetry> > >::type HeisenbergObservables<Symmetry>::
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

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,complex<double> > >::type HeisenbergObservables<Symmetry>::
S_ky (vector<complex<double> > phases) const
{
	vector<OperatorType> Ops(B.size());
	for (size_t l=0; l<B.size(); ++l)
	{
		Ops[l] = B[l].S(0);
	}
	return make_FourierYSum("S", Ops, 1., false, phases);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,complex<double> > >::type HeisenbergObservables<Symmetry>::
Sdag_ky (vector<complex<double> > phases, double factor) const
{
	vector<OperatorType> Ops(B.size());
	for (size_t l=0; l<B.size(); ++l)
	{
		Ops[l] = B[l].Sdag(0);
	}
	return make_FourierYSum("S†", Ops, 1., false, phases);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(),Mpo<Symmetry> >::type HeisenbergObservables<Symmetry>::
Stringz (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	assert(locx1<B.size() and locx2<B.size());
	stringstream ss;
	ss << "Sz" << "(" << locx1 << "," << locy1 << "," << ")" 
	   << "Sz" << "(" << locx2 << "," << locy2 << "," << ")";
	
	auto Sz1 = B[locx1].Sz(locy1);
	auto Sz2 = B[locx2].Sz(locy2);
	
	Mpo<Symmetry> Mout(B.size(), Sz1.Q()+Sz2.Q(), ss.str());
	for (size_t l=0; l<B.size(); ++l) {Mout.setLocBasis(B[l].get_basis().qloc(),l);}
	
	if (locx1 == locx2)
	{
		Mout.setLocal(locx1, Sz1*Sz2);
	}
	else if (locx1<locx2)
	{
		Mout.setLocal({locx1,locx2}, {Sz1,Sz2}, B[0].nh()-B[0].ns());
//		Mout.setLocal({locx1,locx2}, {B[0].Id(),B[0].Id()}, B[0].nh()-B[0].ns());
	}
	else if (locx1>locx2)
	{
		throw;
//		Mout.setLocal({locx2, locx1}, {c*B[locx2].sign(), -1.*cdag}, B[0].sign());
	}
	
	return Mout;
}

// template<typename Symmetry>
// Mpo<Symmetry> HeisenbergObservables<Symmetry>::
// Scomp (SPINOP_LABEL Sa, size_t locx, size_t locy, double factor) const
// {
// 	assert(locx<B.size() and locy<B[locx].dim());
// 	stringstream ss;
// 	ss << Sa << "(" << locx << "," << locy << ")";
	
// 	OperatorType Op = factor * B[locx].Scomp(Sa,locy);
	
// 	bool HERMITIAN = (Sa==SX or Sa==SZ)? true:false;
	
// 	Mpo<Symmetry> Mout(B.size(), Op.Q, ss.str(), HERMITIAN);
// 	for (size_t l=0; l<B.size(); ++l) {Mout.setLocBasis(B[l].get_basis(),l);}
	
// 	Mout.setLocal(locx,Op);
// 	return Mout;
// }

// template<typename Symmetry>
// Mpo<Symmetry,complex<double>> HeisenbergObservables<Symmetry>::
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

// template<typename Symmetry>
// Mpo<Symmetry> HeisenbergObservables<Symmetry>::
// n (size_t locx, size_t locy) const
// {
// 	assert(locx<B.size() and locy<B[locx].dim());
// 	stringstream ss;
// 	ss << "n(" << locx << "," << locy << ")";
	
// 	OperatorType Op = B[locx].n(locy);
	
// 	Mpo<Symmetry> Mout(B.size(), Op.Q, ss.str(), true);
// 	for (size_t l=0; l<B.size(); ++l) {Mout.setLocBasis(B[l].get_basis(),l);}
	
// 	Mout.setLocal(locx,Op);
// 	return Mout;
// }

// template<typename Symmetry>
// Mpo<Symmetry> HeisenbergObservables<Symmetry>::
// ScompScomp (SPINOP_LABEL Sa1, SPINOP_LABEL Sa2, size_t locx1, size_t locx2, size_t locy1, size_t locy2, double fac) const
// {
// 	assert(locx1<B.size() and locx2<B.size() and locy1<B[locx1].dim() and locy2<B[locx2].dim());
// 	stringstream ss;
// 	ss << Sa1 << "(" << locx1 << "," << locy1 << ")" << Sa2 << "(" << locx2 << "," << locy2 << ")";
	
// 	OperatorType Op1 = B[locx1].Scomp(Sa1,locy1);
// 	OperatorType Op2 = B[locx2].Scomp(Sa2,locy2);
	
// 	bool HERMITIAN = (Sa1==Sa2 and (Sa1==SZ or Sa1==SX) and locx1==locx2 and locy1==locy2)? true:false;
	
// 	Mpo<Symmetry> Mout(B.size(), Op1.Q+Op2.Q, ss.str(), HERMITIAN);
// 	for (size_t l=0; l<B.size(); ++l) {Mout.setLocBasis(B[l].get_basis(),l);}
	
// 	Mout.setLocal({locx1,locx2}, {fac*Op1,Op2});
// 	return Mout;
// }

// template<typename Symmetry>
// vector<Mpo<Symmetry> >HeisenbergObservables<Symmetry>::
// SdagS (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
// {
// 	vector<Mpo<Symmetry> > out(3);
// 	out[0] = SzSz(locx1,locx2,locy1,locy2);
// 	out[1] = SpSm(locx1,locx2,locy1,locy2,0.5);
// 	out[2] = SmSp(locx1,locx2,locy1,locy2,0.5);
// 	return out;
// }

#endif
