#ifndef KONDOONECKLACEBSERVABLES
#define KONDOONECKLACEBSERVABLES

#include "bases/SpinBase.h"
#include "Mpo.h"
#include "ParamHandler.h" // from TOOLS
#include "Geometry2D.h" // from TOOLS

template<typename Symmetry>
class KondoNecklaceObservables
{
	typedef SiteOperatorQ<Symmetry,Eigen::MatrixXd> OperatorType;
	
public:
	
	///@{
	KondoNecklaceObservables(){};
	KondoNecklaceObservables (const size_t &L); // for inheritance purposes
	KondoNecklaceObservables (const size_t &L, const vector<Param> &params, const map<string,any> &defaults);
	///@}	
	
	///@{
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type Simp (size_t locx, size_t locy=0, double factor=1.) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type Simpdag (size_t locx, size_t locy=0, double factor=std::sqrt(3.)) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type Ssub (size_t locx, size_t locy=0, double factor=1.) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type Ssubdag (size_t locx, size_t locy=0, double factor=std::sqrt(3.)) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type Simp (SPINOP_LABEL Sa, size_t locx, size_t locy=0, double factor=1.) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type Ssub (SPINOP_LABEL Sa, size_t locx, size_t locy=0, double factor=1.) const;
	
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type S ( size_t locx, size_t locy=0, double factor=1.) const
	{return Simp(locx,locy,factor);};
	
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type Scomp (SPINOP_LABEL Sa, size_t locx, size_t locy=0, double factor=1.) const 
	{return Simp(SZ,locx,locy,factor);}
	
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type Sz (size_t locx, size_t locy=0) const {return Simp(SZ,locx,locy,1.);}
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type SimpSimp (SPINOP_LABEL Sa1, SPINOP_LABEL Sa2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type SsubSsub (SPINOP_LABEL Sa1, SPINOP_LABEL Sa2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type SimpSsub (SPINOP_LABEL Sa1, SPINOP_LABEL Sa2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const;
	template<typename Dummy = Symmetry>
	typename std::conditional<Dummy::IS_SPIN_SU2(), Mpo<Symmetry>, vector<Mpo<Symmetry> > >::type SimpdagSimp (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	template<typename Dummy = Symmetry>
	typename std::conditional<Dummy::IS_SPIN_SU2(), Mpo<Symmetry>, vector<Mpo<Symmetry> > >::type SsubdagSsub (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	template<typename Dummy = Symmetry>
	typename std::conditional<Dummy::IS_SPIN_SU2(), Mpo<Symmetry>, vector<Mpo<Symmetry> > >::type SimpdagSsub (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	template<typename Dummy = Symmetry>
	typename std::conditional<Dummy::IS_SPIN_SU2(), Mpo<Symmetry>, vector<Mpo<Symmetry> > >::type SdagS (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const {return SimpdagSimp(locx1,locx2,locy1,locy2);}	
	// template<typename Dummy = Symmetry>
	// typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry, complex<double> > >::type Rcomp (SPINOP_LABEL Sa, size_t locx, size_t locy=0) const;
	///@}

	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,complex<double> > >::type Simp_ky    (vector<complex<double> > phases) const;
	template<typename Dummy = Symmetry>
	typename std::enable_if<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,complex<double> > >::type Simpdag_ky (vector<complex<double> > phases, double factor=sqrt(3.)) const;
	

	///@{ not implemented
//	Mpo<Symmetry> SimpSsubSimpSimp (size_t locx1, SPINOP_LABEL SOP1, size_t locx2, SPINOP_LABEL SOP2, 
//	                          size_t loc3x, SPINOP_LABEL SOP3, size_t loc4x, SPINOP_LABEL SOP4,
//	                          size_t locy1=0, size_t locy2=0, size_t loc3y=0, size_t loc4y=0) const;
//	Mpo<Symmetry> SimpSsubSimpSsub (size_t locx1, SPINOP_LABEL SOP1, size_t locx2, SPINOP_LABEL SOP2, 
//	                          size_t loc3x, SPINOP_LABEL SOP3, size_t loc4x, SPINOP_LABEL SOP4,
//	                          size_t locy1=0, size_t locy2=0, size_t loc3y=0, size_t loc4y=0) const;
	///@}
		
protected:

	Mpo<Symmetry> make_local (KONDO_SUBSYSTEM SUBSYS, size_t locx, size_t locy,
	                          const OperatorType &Op,
							  double factor =1.,
	                          bool HERMITIAN=false) const;
	Mpo<Symmetry> make_corr  (KONDO_SUBSYSTEM SUBSYS, size_t locx1, size_t locx2, size_t locy1, size_t locy2,
	                          const OperatorType &Op1, const OperatorType &Op2, qarray<Symmetry::Nq> Qtot,
	                          double factor, bool HERMITIAN) const;
	
	Mpo<Symmetry,complex<double> >
	make_FourierYSum (KONDO_SUBSYSTEM SUBSYS, string name, const vector<OperatorType> &Ops, double factor, bool HERMITIAN, const vector<complex<double> > &phases) const;

	typename Symmetry::qType getQ_ScompScomp(SPINOP_LABEL Sa1, SPINOP_LABEL Sa2) const;
	
	vector<SpinBase<Symmetry> > Bimp;
	vector<SpinBase<Symmetry> > Bsub;
};

template<typename Symmetry>
KondoNecklaceObservables<Symmetry>::
KondoNecklaceObservables (const size_t &L)
{
	Bimp.resize(L);
	Bsub.resize(L);
}

template<typename Symmetry>
KondoNecklaceObservables<Symmetry>::
KondoNecklaceObservables (const size_t &L, const vector<Param> &params, const map<string,any> &defaults)
{
	ParamHandler P(params,defaults);
	size_t Lcell = P.size();
	Bimp.resize(L); Bsub.resize(L);
	
	for (size_t l=0; l<L; ++l)
	{
		Bsub[l] = SpinBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), P.get<size_t>("Dsub",l%Lcell));
		Bimp[l] = SpinBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), P.get<size_t>("Dimp",l%Lcell));
	}
}

//-------------

template<typename Symmetry>
Mpo<Symmetry> KondoNecklaceObservables<Symmetry>::
make_local (KONDO_SUBSYSTEM SUBSYS, size_t locx, size_t locy, const OperatorType &Op, double factor, bool HERMITIAN) const
{
	assert(locx<Bimp.size() and locy<Bimp[locx].dim());
	assert(SUBSYS != IMPSUB);
	stringstream ss;
	ss << Op.label() << "(" << locx << "," << locy;
	if (factor != 1.) ss << ",factor=" << factor;
	ss << ")";
	
	Mpo<Symmetry> Mout(Bimp.size(), Op.Q(), ss.str(), HERMITIAN);
	for (size_t l=0; l<Bimp.size(); ++l) {Mout.setLocBasis(Bsub[l].get_basis().combine(Bimp[l].get_basis()).qloc(),l);}
	
	OperatorType OpExt;

	if (SUBSYS == SUB)
	{
		OpExt   = kroneckerProduct(Op, Bimp[locx].Id());
	}
	else if (SUBSYS == IMP)
	{
		OpExt = kroneckerProduct(Bsub[locx].Id(), Op);
	}

	Mout.setLocal(locx, (factor * OpExt).template plain<double>());
	return Mout;
}

template<typename Symmetry>
Mpo<Symmetry> KondoNecklaceObservables<Symmetry>::
make_corr  (KONDO_SUBSYSTEM SUBSYS, size_t locx1, size_t locx2, size_t locy1, size_t locy2,
			const OperatorType &Op1, const OperatorType &Op2, qarray<Symmetry::Nq> Qtot,
			double factor, bool HERMITIAN) const
{
	assert(locx1<Bimp.size() and locx2<Bimp.size() and locy1<Bimp[locx1].dim() and locy2<Bimp[locx2].dim());
	stringstream ss;
	ss << Op1.label() << "(" << locx1 << "," << locy1 << ")"
	   << Op2.label() << "(" << locx2 << "," << locy2 << ")";
	
	Mpo<Symmetry> Mout(Bimp.size(), Qtot, ss.str(), HERMITIAN);
	for (size_t l=0; l<Bimp.size(); ++l) {Mout.setLocBasis(Bsub[l].get_basis().combine(Bimp[l].get_basis()).qloc(),l);}
	
	OperatorType Op1Ext;
	OperatorType Op2Ext;
	
	if (SUBSYS == SUB)
	{
		Op1Ext = kroneckerProduct(Op1, Bimp[locx1].Id());
		Op2Ext = kroneckerProduct(Op2, Bimp[locx2].Id());
	}
	else if (SUBSYS == IMP)
	{
		Op1Ext = kroneckerProduct(Bsub[locx1].Id(), Op1);
		Op2Ext = kroneckerProduct(Bsub[locx2].Id(), Op2);
	}
	else if (SUBSYS == IMPSUB)
	{
		Op1Ext = kroneckerProduct(Bsub[locx1].Id(), Op1);
		Op2Ext = kroneckerProduct(Op2, Bimp[locx2].Id());
	}
	
	if (locx1 == locx2)
	{
		auto product = factor*OperatorType::prod(Op1Ext, Op2Ext, Qtot);
		Mout.setLocal(locx1, product.template plain<double>());
	}
	else
	{
		Mout.setLocal({locx1, locx2}, {(factor*Op1Ext).template plain<double>(), Op2Ext.template plain<double>()});
	}
		
	return Mout;
}

template<typename Symmetry>
Mpo<Symmetry,complex<double> > KondoNecklaceObservables<Symmetry>::
make_FourierYSum (KONDO_SUBSYSTEM SUBSYS, string name, const vector<OperatorType> &Ops, 
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
	Mpo<Symmetry,complex<double> > Mout(Bimp.size(), Ops[0].Q(), ss.str(), HERMITIAN);
	for (size_t l=0; l<Bimp.size(); ++l) {Mout.setLocBasis(Bsub[l].get_basis().combine(Bimp[l].get_basis()).qloc(),l);}

	vector<complex<double> > phases_x_factor = phases;
	for (int l=0; l<phases.size(); ++l)
	{
		phases_x_factor[l] = phases[l] * factor;
	}
	
	vector<SiteOperator<Symmetry,complex<double> > > OpsPlain(Ops.size());
	for (int l=0; l<OpsPlain.size(); ++l)
	{
		if (SUBSYS == SUB)
		{
			OpsPlain[l] = kroneckerProduct(Ops[l], Bimp[l].Id()).template plain<double>().template cast<complex<double> >();
		}
		else if (SUBSYS == IMP)
		{
			OpsPlain[l] = kroneckerProduct(Bsub[l].Id(),Ops[l]).template plain<double>().template cast<complex<double> >();
		}	
	}
	
	Mout.setLocalSum(OpsPlain, phases_x_factor);
	return Mout;
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type KondoNecklaceObservables<Symmetry>::
Simp (SPINOP_LABEL Sa, size_t locx, size_t locy, double factor) const
{
	bool HERMITIAN = (Sa==SX or Sa==SZ)? true:false;
	return make_local(IMP,locx,locy, Bimp[locx].Scomp(Sa,locy), factor, HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type KondoNecklaceObservables<Symmetry>::
Ssub (SPINOP_LABEL Sa, size_t locx, size_t locy, double factor) const
{
	bool HERMITIAN = (Sa==SX or Sa==SZ)? true:false;
	return make_local(SUB,locx,locy, Bsub[locx].Scomp(Sa,locy), factor, HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type KondoNecklaceObservables<Symmetry>::
SimpSimp (SPINOP_LABEL Sa1, SPINOP_LABEL Sa2, size_t locx1, size_t locx2, size_t locy1, size_t locy2, double fac) const
{
	return make_corr(IMP,locx1,locx2,locy1,locy2, Bimp[locx1].Scomp(Sa1,locy1), Bimp[locx2].Scomp(Sa2,locy2), getQ_ScompScomp(Sa1,Sa2), fac, PROP::HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type KondoNecklaceObservables<Symmetry>::
SsubSsub (SPINOP_LABEL Sa1, SPINOP_LABEL Sa2, size_t locx1, size_t locx2, size_t locy1, size_t locy2, double fac) const
{
	return make_corr(SUB,locx1,locx2,locy1,locy2, Bsub[locx1].Scomp(Sa1,locy1), Bsub[locx2].Scomp(Sa2,locy2), getQ_ScompScomp(Sa1,Sa2), fac, PROP::HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<!Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type KondoNecklaceObservables<Symmetry>::
SimpSsub (SPINOP_LABEL Sa1, SPINOP_LABEL Sa2, size_t locx1, size_t locx2, size_t locy1, size_t locy2, double fac) const
{
	return make_corr(IMPSUB,locx1,locx2,locy1,locy2, Bimp[locx1].Scomp(Sa1,locy1), Bsub[locx2].Scomp(Sa2,locy2), getQ_ScompScomp(Sa1,Sa2), fac, PROP::HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type KondoNecklaceObservables<Symmetry>::
Simp (size_t locx, size_t locy, double factor) const
{
	return make_local(IMP,locx,locy, Bimp[locx].S(locy), factor, PROP::NON_HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type KondoNecklaceObservables<Symmetry>::
Simpdag (size_t locx, size_t locy, double factor) const
{
	return make_local(IMP,locx,locy, Bimp[locx].Sdag(locy), factor, PROP::NON_HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type KondoNecklaceObservables<Symmetry>::
Ssub(size_t locx, size_t locy, double factor) const
{
    return make_local(SUB,locx,locy, Bsub[locx].S(locy), factor, PROP::NON_HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(), Mpo<Symmetry> >::type KondoNecklaceObservables<Symmetry>::
Ssubdag (size_t locx, size_t locy, double factor) const
{
    return make_local(SUB,locx,locy, Bsub[locx].Sdag(locy), factor, PROP::NON_HERMITIAN);
}

template<typename Symmetry>
template<typename Dummy>
typename std::conditional<Dummy::IS_SPIN_SU2(), Mpo<Symmetry>, vector<Mpo<Symmetry> > >::type KondoNecklaceObservables<Symmetry>::
SimpdagSimp (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	if constexpr (Symmetry::IS_SPIN_SU2())
				 {
					 return make_corr(IMP,locx1, locx2, locy1, locy2, Bimp[locx1].Sdag(locy1), Bimp[locx2].S(locy2), Symmetry::qvacuum(), sqrt(3.), PROP::HERMITIAN);
					 // return make_corr("T†", "T", locx1, locx2, locy1, locy2, F[locx1].Tdag(locy1), F[locx2].T(locy2), Symmetry::qvacuum(), std::sqrt(3.), PROP::HERMITIAN);
				 }
	else
	{
		vector<Mpo<Symmetry> > out(3);
		out[0] = SimpSimp(SZ,SZ,locx1,locx2,locy1,locy2);
		out[1] = SimpSimp(SP,SM,locx1,locx2,locy1,locy2,0.5);
		out[2] = SimpSimp(SM,SP,locx1,locx2,locy1,locy2,0.5);
		return out;
	}
}

template<typename Symmetry>
template<typename Dummy>
typename std::conditional<Dummy::IS_SPIN_SU2(), Mpo<Symmetry>, vector<Mpo<Symmetry> > >::type KondoNecklaceObservables<Symmetry>::
SsubdagSsub (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	if constexpr (Symmetry::IS_SPIN_SU2())
				 {
					 return make_corr(SUB,locx1, locx2, locy1, locy2, Bsub[locx1].Sdag(locy1), Bsub[locx2].S(locy2), Symmetry::qvacuum(), sqrt(3.), PROP::HERMITIAN);
					 // return make_corr("T†", "T", locx1, locx2, locy1, locy2, F[locx1].Tdag(locy1), F[locx2].T(locy2), Symmetry::qvacuum(), std::sqrt(3.), PROP::HERMITIAN);
				 }
	else
	{
		vector<Mpo<Symmetry> > out(3);
		out[0] = SsubSsub(SZ,SZ,locx1,locx2,locy1,locy2);
		out[1] = SsubSsub(SP,SM,locx1,locx2,locy1,locy2,0.5);
		out[2] = SsubSsub(SM,SP,locx1,locx2,locy1,locy2,0.5);
		return out;
	}
}

template<typename Symmetry>
template<typename Dummy>
typename std::conditional<Dummy::IS_SPIN_SU2(), Mpo<Symmetry>, vector<Mpo<Symmetry> > >::type KondoNecklaceObservables<Symmetry>::
SimpdagSsub (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	if constexpr (Symmetry::IS_SPIN_SU2())
				 {
					 return make_corr(IMPSUB,locx1, locx2, locy1, locy2, Bimp[locx1].Sdag(locy1), Bsub[locx2].S(locy2), Symmetry::qvacuum(), sqrt(3.), PROP::HERMITIAN);
					 // return make_corr("T†", "T", locx1, locx2, locy1, locy2, F[locx1].Tdag(locy1), F[locx2].T(locy2), Symmetry::qvacuum(), std::sqrt(3.), PROP::HERMITIAN);
				 }
	else
	{
		vector<Mpo<Symmetry> > out(3);
		out[0] = SimpSsub(SZ,SZ,locx1,locx2,locy1,locy2);
		out[1] = SimpSsub(SP,SM,locx1,locx2,locy1,locy2,0.5);
		out[2] = SimpSsub(SM,SP,locx1,locx2,locy1,locy2,0.5);
		return out;
	}
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,complex<double> > >::type KondoNecklaceObservables<Symmetry>::
Simp_ky (vector<complex<double> > phases) const
{
	vector<OperatorType> Ops(Bimp.size());
	for (size_t l=0; l<Bimp.size(); ++l)
	{
		Ops[l] = Bimp[l].S(0);
	}
	return make_FourierYSum(IMP,"S", Ops, 1., false, phases);
}

template<typename Symmetry>
template<typename Dummy>
typename std::enable_if<Dummy::IS_SPIN_SU2(),Mpo<Symmetry,complex<double> > >::type KondoNecklaceObservables<Symmetry>::
Simpdag_ky (vector<complex<double> > phases, double factor) const
{
	vector<OperatorType> Ops(Bimp.size());
	for (size_t l=0; l<Bimp.size(); ++l)
	{
		Ops[l] = Bimp[l].Sdag(0);
	}
	return make_FourierYSum(IMP,"S†", Ops, 1., false, phases);
}

template<typename Symmetry>
typename Symmetry::qType KondoNecklaceObservables<Symmetry>::getQ_ScompScomp(SPINOP_LABEL Sa1, SPINOP_LABEL Sa2) const
{
	typename Symmetry::qType out;
	if ( (Sa1 == SZ and Sa2 == SZ) or (Sa1 == SP and Sa2 == SM) or (Sa1 == SM and Sa2 == SP) or (Sa1 == SX or Sa1 == iSY) ) {out = Symmetry::qvacuum();}
	else {assert(false and "Quantumnumber for the chosen ScompScomp is not computed. Add in KondoNecklaceObservables::getQ_ScompScomp");}
	return out;
}

// template<typename Symmetry>
// template<SPIN_INDEX sigma>
// Mpo<Symmetry> KondoNecklaceObservables<Symmetry>::
// c (size_t locx, size_t locy) const
// {
// 	stringstream ss;
// 	ss << "c" << sigma;
// 	return make_local(SUB, ss.str(), locx,locy, F[locx].c(sigma,locy), true);
// }

// template<typename Symmetry>
// template<SPIN_INDEX sigma>
// Mpo<Symmetry> KondoNecklaceObservables<Symmetry>::
// cdag (size_t locx, size_t locy) const
// {
// 	stringstream ss;
// 	ss << "c†" << sigma;
// 	return make_local(SUB, ss.str(), locx,locy, F[locx].cdag(sigma,locy), true);
// }

// template<typename Symmetry>
// Mpo<Symmetry> KondoNecklaceObservables<Symmetry>::
// n (size_t locx, size_t locy) const
// {
// 	return make_local(SUB, "n", locx,locy, F[locx].n(locy), false, true);
// }

// template<typename Symmetry>
// template<SPIN_INDEX sigma>
// Mpo<Symmetry> KondoNecklaceObservables<Symmetry>::
// n (size_t locx, size_t locy) const
// {
// 	return make_local(SUB, "n", locx,locy, F[locx].n(sigma,locy), false, true);
// 	// FERMIONIC=false, HERMITIAN=true
// }

// template<typename Symmetry>
// Mpo<Symmetry> KondoNecklaceObservables<Symmetry>::
// d (size_t locx, size_t locy) const
// {
// 	return make_local(SUB, "d", locx,locy, F[locx].d(locy), false, true);
// }

// template<typename Symmetry>
// Mpo<Symmetry> KondoNecklaceObservables<Symmetry>::
// cc (size_t locx, size_t locy) const
// {
// 	stringstream ss;
// 	ss << "c" << UP << "c" << DN;
// 	return make_local(SUB, ss.str(), locx,locy, F[locx].c(UP,locy)*F[locx].c(DN,locy), false, false);
// }

// template<typename Symmetry>
// Mpo<Symmetry> KondoNecklaceObservables<Symmetry>::
// cdagcdag (size_t locx, size_t locy) const
// {
// 	stringstream ss;
// 	ss << "c†" << DN << "c†" << UP;
// 	return make_local(SUB, ss.str(), locx,locy, F[locx].cdag(DN,locy)*F[locx].cdag(UP,locy), false, false);
// }

// template<typename Symmetry>
// Mpo<Symmetry> KondoNecklaceObservables<Symmetry>::
// nn (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
// {
// 	return make_corr(SUB, "n","n", locx1,locx2,locy1,locy2, F[locx1].n(UPDN,locy1), F[locx2].n(UPDN,locy2));
// }

// template<typename Symmetry>
// Mpo<Symmetry> KondoNecklaceObservables<Symmetry>::
// Simp (SPINOP_LABEL Sa, size_t locx, size_t locy, double factor) const
// {
// 	stringstream ss;
// 	ss << Sa << "imp";
// 	bool HERMITIAN = (Sa==SX or Sa==SZ)? true:false;
// 	return make_local(IMP, ss.str(), locx, locy, factor * B[locx].Scomp(Sa,locy), false, HERMITIAN);
// }

// template<typename Symmetry>
// Mpo<Symmetry> KondoNecklaceObservables<Symmetry>::
// Ssub (SPINOP_LABEL Sa, size_t locx, size_t locy, double factor) const
// {
// 	stringstream ss;
// 	ss << Sa << "sub";
// 	bool HERMITIAN = (Sa==SX or Sa==SZ)? true:false;
// 	return make_local(SUB, ss.str(), locx,locy, factor * F[locx].Scomp(Sa,locy), false, HERMITIAN);
// }

// template<typename Symmetry>
// Mpo<Symmetry> KondoNecklaceObservables<Symmetry>::
// SimpSimp (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
// {
// 	stringstream ss1; ss1 << SOP1 << "imp";
// 	stringstream ss2; ss2 << SOP2 << "imp";
	
// 	return make_corr(IMP, ss1.str(),ss2.str(), locx1,locx2,locy1,locy2, B[locx1].Scomp(SOP1,locy1), B[locx2].Scomp(SOP2,locy2));
// }

// template<typename Symmetry>
// Mpo<Symmetry> KondoNecklaceObservables<Symmetry>::
// SsubSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
// {
// 	stringstream ss1; ss1 << SOP1 << "sub";
// 	stringstream ss2; ss2 << SOP2 << "sub";
	
// 	return make_corr(SUB, ss1.str(),ss2.str(), locx1,locx2,locy1,locy2, F[locx1].Scomp(SOP1,locy1), F[locx2].Scomp(SOP2,locy2));
// }

// template<typename Symmetry>
// Mpo<Symmetry> KondoNecklaceObservables<Symmetry>::
// SimpSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
// {
// 	stringstream ss1; ss1 << SOP1 << "imp";
// 	stringstream ss2; ss2 << SOP2 << "sub";
	
// 	return make_corr(IMPSUB, ss1.str(),ss2.str(), locx1,locx2,locy1,locy2, B[locx1].Scomp(SOP1,locy1), F[locx2].Scomp(SOP2,locy2));
// }

// template<typename Symmetry>
// template<typename MpsType>
// double KondoNecklaceObservables<Symmetry>::
// SvecSvecAvgImpSub (const MpsType &Psi, size_t locx1, size_t locx2, size_t locy1, size_t locy2)
// {
// 	return isReal(avg(Psi,SimpSsub(SZ,SZ,locx1,locx2,locy1,locy2),Psi))+
// 	       isReal(avg(Psi,SimpSsub(SP,SM,locx1,locx2,locy1,locy2),Psi));
// }

// template<typename Symmetry>
// template<SPIN_INDEX sigma>
// Mpo<Symmetry> KondoNecklaceObservables<Symmetry>::
// cdagc (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
// {
// 	assert(locx1<F.size() and locx2<F.size() and locy1<F[locx1].dim() and locy2<F[locx2].dim());
// 	stringstream ss;
// 	ss << "c†" << sigma << "(" << locx1 << "," << locy1 << "," << ")" 
// 	   << "c " << sigma << "(" << locx2 << "," << locy2 << "," << ")";
	
// 	auto cdag = kroneckerProduct(B[locx1].Id(),F[locx1].cdag(sigma,locy1));
// 	auto c    = kroneckerProduct(B[locx2].Id(),F[locx2].c   (sigma,locy2));
// 	auto sign = kroneckerProduct(B[0].Id(),F[0].sign());
	
// 	Mpo<Symmetry> Mout(F.size(), cdag.Q+c.Q, ss.str());
// 	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l);}
	
// 	if (locx1 == locx2)
// 	{
// 		Mout.setLocal(locx1, cdag*c);
// 	}
// 	else if (locx1<locx2)
// 	{
// 		Mout.setLocal({locx1, locx2}, {cdag*sign, c}, sign);
// 	}
// 	else if (locx1>locx2)
// 	{
// 		Mout.setLocal({locx2, locx1}, {c*sign, -1.*cdag}, sign);
// 	}
	
// 	return Mout;
// }

// Mpo<Sym::U1xU1<double> > KondoU1xU1::
// SimpSsubSimpSimp (size_t locx1, SPINOP_LABEL SOP1, size_t locx2, SPINOP_LABEL SOP2, size_t loc3x, SPINOP_LABEL SOP3, size_t loc4x, SPINOP_LABEL SOP4,
//                   size_t locy1, size_t locy2, size_t loc3y, size_t loc4y)
// {
// 	assert(locx1<this->F.size() and locx2<this->F.size() and loc3x<this->F.size() and loc4x<this->F.size());
// 	stringstream ss;
// 	ss << SOP1 << "(" << locx1 << "," << locy1 << ")" << SOP2 << "(" << locx2 << "," << locy2 << ")" <<
// 	      SOP3 << "(" << loc3x << "," << loc3y << ")" << SOP4 << "(" << loc4x << "," << loc4y << ")";
// 	Mpo<Symmetry> Mout(this->F.size(), this->N_legs, locBasis(), {0,0}, KondoU1xU1::NMlabel, ss.str());
// 	Mout.setLocal({locx1, locx2, loc3x, loc4x},
// 				  {kroneckerProduct(B.Scomp(SOP1,locy1),F.Id()), 
// 				   kroneckerProduct(B.Id(),F.Scomp(SOP2,locy2)),
// 				   kroneckerProduct(B.Scomp(SOP3,loc3y),F.Id()),
// 				   kroneckerProduct(B.Scomp(SOP4,loc4y),F.Id())});
// 	return Mout;
// }

// Mpo<Sym::U1xU1<double> > KondoU1xU1::
// SimpSsubSimpSsub (size_t locx1, SPINOP_LABEL SOP1, size_t locx2, SPINOP_LABEL SOP2, size_t loc3x, SPINOP_LABEL SOP3, size_t loc4x, SPINOP_LABEL SOP4,
//                   size_t locy1, size_t locy2, size_t loc3y, size_t loc4y)
// {
// 	assert(locx1<this->F.size() and locx2<this->F.size() and loc3x<this->F.size() and loc4x<this->F.size());
// 	stringstream ss;
// 	ss << SOP1 << "(" << locx1 << "," << locy1 << ")" << SOP2 << "(" << locx2 << "," << locy2 << ")" <<
// 	      SOP3 << "(" << loc3x << "," << loc3y << ")" << SOP4 << "(" << loc4x << "," << loc4y << ")";
// 	Mpo<Symmetry> Mout(this->F.size(), this->N_legs, locBasis(), {0,0}, KondoU1xU1::NMlabel, ss.str());
// 	SparseMatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
// 	SparseMatrixXd IdImp(Mpo<Symmetry>::qloc[locx2].size()/F.dim(), Mpo<Symmetry>::qloc[locx2].size()/F.dim()); IdImp.setIdentity();
// 	Mout.setLocal({locx1, locx2, loc3x, loc4x},
// 				  {kroneckerProduct(B.Scomp(SOP1,locy1),F.Id()), 
// 				   kroneckerProduct(B.Id(),F.Scomp(SOP2,locy2)),
// 				   kroneckerProduct(B.Scomp(SOP3,loc3y),F.Id()),
// 				   kroneckerProduct(B.Id(),F.Scomp(SOP4,loc4y))});
// 	return Mout;
// }

#endif
