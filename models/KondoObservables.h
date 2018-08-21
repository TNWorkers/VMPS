#ifndef KONDOOBSERVABLES
#define KONDOOBSERVABLES

#include "bases/FermionBase.h"
#include "bases/SpinBase.h"
#include "Mpo.h"
#include "ParamHandler.h" // from TOOLS

enum KONDO_SUBSYSTEM {IMP, SUB, IMPSUB};

template<typename Symmetry>
class KondoObservables
{
typedef SiteOperator<Symmetry,double> OperatorType;
	
public:
	
	///@{
	KondoObservables(){};
	KondoObservables (const size_t &L); // for inheritance purposes
	KondoObservables (const size_t &L, const vector<Param> &params, const map<string,any> &defaults);
	///@}
	
	///@{
	template<SPIN_INDEX sigma> Mpo<Symmetry> c    (size_t locx, size_t locy=0) const;
	template<SPIN_INDEX sigma> Mpo<Symmetry> cdag (size_t locx, size_t locy=0) const;
	Mpo<Symmetry> cc   (size_t locx, size_t locy=0) const;
	Mpo<Symmetry> cdagcdag (size_t locx, size_t locy=0) const;
	///@}
	
	///@{
	Mpo<Symmetry> n (size_t locx, size_t locy=0) const;
	Mpo<Symmetry> d (size_t locx, size_t locy=0) const;
	template<SPIN_INDEX sigma> Mpo<Symmetry> cdagc (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	Mpo<Symmetry> nn (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	///@}
	
	///@{
	Mpo<Symmetry> Simp (SPINOP_LABEL SOP, size_t locx, size_t locy=0, double factor=1.) const;
	Mpo<Symmetry> Ssub (SPINOP_LABEL SOP, size_t locx, size_t locy=0, double factor=1.) const;
	// for compatibility:
	Mpo<Symmetry> Scomp (SPINOP_LABEL SOP, size_t locx, size_t locy=0, double factor=1.) const {return Simp(SOP,locx,locy,factor);};
	Mpo<Symmetry> Sz (size_t locx, size_t locy=0, double factor=1.) const {return Simp(SZ,locx,locy,factor);};
	///@}
	
	///@{
	Mpo<Symmetry> SimpSimp (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	Mpo<Symmetry> SsubSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	Mpo<Symmetry> SimpSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	///@}
	
	///@{ not implemented
//	Mpo<Symmetry> SimpSsubSimpSimp (size_t locx1, SPINOP_LABEL SOP1, size_t locx2, SPINOP_LABEL SOP2, 
//	                          size_t loc3x, SPINOP_LABEL SOP3, size_t loc4x, SPINOP_LABEL SOP4,
//	                          size_t locy1=0, size_t locy2=0, size_t loc3y=0, size_t loc4y=0) const;
//	Mpo<Symmetry> SimpSsubSimpSsub (size_t locx1, SPINOP_LABEL SOP1, size_t locx2, SPINOP_LABEL SOP2, 
//	                          size_t loc3x, SPINOP_LABEL SOP3, size_t loc4x, SPINOP_LABEL SOP4,
//	                          size_t locy1=0, size_t locy2=0, size_t loc3y=0, size_t loc4y=0) const;
	///@}
	
	template<typename MpsType> double SvecSvecAvgImpSub (const MpsType &Psi, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	
protected:
	
	Mpo<Symmetry> make_local (KONDO_SUBSYSTEM SUBSYS, string name, 
	                          size_t locx, size_t locy, 
	                          const OperatorType &Op,
	                          bool FERMIONIC=false, bool HERMITIAN=false) const;
	
	Mpo<Symmetry> make_corr  (KONDO_SUBSYSTEM SUBSYS, string name1, string name2, 
	                          size_t locx1, size_t locx2, size_t locy1, size_t locy2, 
	                          const OperatorType &Op1, const OperatorType &Op2,
	                          bool BOTH_HERMITIAN=false) const;
	
	vector<SpinBase   <Symmetry> > B;
	vector<FermionBase<Symmetry> > F;
};

template<typename Symmetry>
KondoObservables<Symmetry>::
KondoObservables (const size_t &L)
{
	F.resize(L);
}

template<typename Symmetry>
KondoObservables<Symmetry>::
KondoObservables (const size_t &L, const vector<Param> &params, const map<string,any> &defaults)
{
	ParamHandler P(params,defaults);
	size_t Lcell = P.size();
	B.resize(L); F.resize(L);
	
	for (size_t l=0; l<L; ++l)
	{
		B[l] = SpinBase<Symmetry> (P.get<size_t>("Ly",l%Lcell), P.get<size_t>("D",l%Lcell));
		F[l] = FermionBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), !isfinite(P.get<double>("U",l%Lcell)));
	}
}

//-------------

template<typename Symmetry>
Mpo<Symmetry> KondoObservables<Symmetry>::
make_local (KONDO_SUBSYSTEM SUBSYS, string name, size_t locx, size_t locy, const OperatorType &Op, bool FERMIONIC, bool HERMITIAN) const
{
	assert(locx<F.size() and locy<F[locx].dim());
	assert(SUBSYS != IMPSUB);
	stringstream ss;
	ss << name << "(" << locx << "," << locy << ")";
	
	Mpo<Symmetry> Mout(F.size(), Op.Q, ss.str(), HERMITIAN);
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l);}
	
	OperatorType OpExt, SignExt;
	
	if (SUBSYS == SUB)
	{
		OpExt   = kroneckerProduct(B[locx].Id(), Op);
		SignExt = kroneckerProduct(B[locx].Id(), F[locx].sign());
	}
	else if (SUBSYS == IMP)
	{
		assert(!FERMIONIC and "Impurity cannot be fermionic!");
		OpExt = kroneckerProduct(Op, F[locx].Id());
	}
	
	if (FERMIONIC)
	{
		Mout.setLocal(locx, OpExt, SignExt);
	}
	else
	{
		Mout.setLocal(locx, OpExt);
	}
	return Mout;
}

template<typename Symmetry>
Mpo<Symmetry> KondoObservables<Symmetry>::
make_corr (KONDO_SUBSYSTEM SUBSYS, string name1, string name2, 
           size_t locx1, size_t locx2, size_t locy1, size_t locy2, 
           const OperatorType &Op1, const OperatorType &Op2, 
           bool BOTH_HERMITIAN) const
{
	assert(locx1<F.size() and locx2<F.size() and locy1<F[locx1].dim() and locy2<F[locx2].dim());
	stringstream ss;
	ss << name1 << "(" << locx1 << "," << locy1 << ")"
	   << name2 << "(" << locx2 << "," << locy2 << ")";
	
	bool HERMITIAN = (BOTH_HERMITIAN and locx1==locx2 and locy1==locy2)? true:false;
	
	OperatorType Op1Ext;
	OperatorType Op2Ext;
	
	if (SUBSYS == SUB)
	{
		Op1Ext = kroneckerProduct(B[locx1].Id(), Op1);
		Op2Ext = kroneckerProduct(B[locx2].Id(), Op2);
	}
	else if (SUBSYS == IMP)
	{
		Op1Ext = kroneckerProduct(Op1, F[locx1].Id());
		Op2Ext = kroneckerProduct(Op2, F[locx2].Id());
	}
	else if (SUBSYS = IMPSUB)
	{
		Op2Ext = kroneckerProduct(Op1, F[locx1].Id());
		Op1Ext = kroneckerProduct(B[locx2].Id(), Op2);
	}
	
	Mpo<Symmetry> Mout(F.size(), Op1.Q+Op2.Q, ss.str(), HERMITIAN);
	for (size_t l=0; l<F.size(); ++l)  {Mout.setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l);}
	
	Mout.setLocal({locx1,locx2}, {Op1Ext,Op2Ext});
	return Mout;
}

//-------------

template<typename Symmetry>
template<SPIN_INDEX sigma>
Mpo<Symmetry> KondoObservables<Symmetry>::
c (size_t locx, size_t locy) const
{
	stringstream ss;
	ss << "c" << sigma;
	return make_local(SUB, ss.str(), locx,locy, F[locx].c(sigma,locy), true);
}

template<typename Symmetry>
template<SPIN_INDEX sigma>
Mpo<Symmetry> KondoObservables<Symmetry>::
cdag (size_t locx, size_t locy) const
{
	stringstream ss;
	ss << "c†" << sigma;
	return make_local(SUB, ss.str(), locx,locy, F[locx].cdag(sigma,locy), true);
}

template<typename Symmetry>
Mpo<Symmetry> KondoObservables<Symmetry>::
n (size_t locx, size_t locy) const
{
	return make_local(SUB, "n", locx,locy, F[locx].n(locy), false, true);
}

template<typename Symmetry>
Mpo<Symmetry> KondoObservables<Symmetry>::
d (size_t locx, size_t locy) const
{
	return make_local(SUB, "d", locx,locy, F[locx].d(locy), false, true);
}

template<typename Symmetry>
Mpo<Symmetry> KondoObservables<Symmetry>::
cc (size_t locx, size_t locy) const
{
	stringstream ss;
	ss << "c" << UP << "c" << DN;
	return make_local(SUB, ss.str(), locx,locy, F[locx].c(UP,locy)*F[locx].c(DN,locy), false, false);
}

template<typename Symmetry>
Mpo<Symmetry> KondoObservables<Symmetry>::
cdagcdag (size_t locx, size_t locy) const
{
	stringstream ss;
	ss << "c†" << DN << "c†" << UP;
	return make_local(SUB, ss.str(), locx,locy, F[locx].cdag(DN,locy)*F[locx].cdag(UP,locy), false, false);
}

template<typename Symmetry>
Mpo<Symmetry> KondoObservables<Symmetry>::
nn (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	return make_corr(SUB, "n","n", locx1,locx2,locy1,locy2, F[locx1].n(UPDN,locy1), F[locx2].n(UPDN,locy2));
}

template<typename Symmetry>
Mpo<Symmetry> KondoObservables<Symmetry>::
Simp (SPINOP_LABEL Sa, size_t locx, size_t locy, double factor) const
{
	stringstream ss;
	ss << Sa << "imp";
	bool HERMITIAN = (Sa==SX or Sa==SZ)? true:false;
	return make_local(IMP, ss.str(), locx, locy, factor * B[locx].Scomp(Sa,locy), false, HERMITIAN);
}

template<typename Symmetry>
Mpo<Symmetry> KondoObservables<Symmetry>::
Ssub (SPINOP_LABEL Sa, size_t locx, size_t locy, double factor) const
{
	stringstream ss;
	ss << Sa << "sub";
	bool HERMITIAN = (Sa==SX or Sa==SZ)? true:false;
	return make_local(SUB, ss.str(), locx,locy, factor * F[locx].Scomp(Sa,locy), false, HERMITIAN);
}

template<typename Symmetry>
Mpo<Symmetry> KondoObservables<Symmetry>::
SimpSimp (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	stringstream ss1; ss1 << SOP1 << "imp";
	stringstream ss2; ss2 << SOP2 << "imp";
	
	return make_corr(IMP, ss1.str(),ss2.str(), locx1,locx2,locy1,locy2, B[locx1].Scomp(SOP1,locy1), B[locx2].Scomp(SOP2,locy2));
}

template<typename Symmetry>
Mpo<Symmetry> KondoObservables<Symmetry>::
SsubSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	stringstream ss1; ss1 << SOP1 << "sub";
	stringstream ss2; ss2 << SOP2 << "sub";
	
	return make_corr(SUB, ss1.str(),ss2.str(), locx1,locx2,locy1,locy2, F[locx1].Scomp(SOP1,locy1), F[locx2].Scomp(SOP2,locy2));
}

template<typename Symmetry>
Mpo<Symmetry> KondoObservables<Symmetry>::
SimpSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	stringstream ss1; ss1 << SOP1 << "imp";
	stringstream ss2; ss2 << SOP2 << "sub";
	
	return make_corr(IMPSUB, ss1.str(),ss2.str(), locx1,locx2,locy1,locy2, B[locx1].Scomp(SOP1,locy1), F[locx2].Scomp(SOP2,locy2));
}

template<typename Symmetry>
template<typename MpsType>
double KondoObservables<Symmetry>::
SvecSvecAvgImpSub (const MpsType &Psi, size_t locx1, size_t locx2, size_t locy1, size_t locy2)
{
	return isReal(avg(Psi,SimpSsub(SZ,SZ,locx1,locx2,locy1,locy2),Psi))+
	       isReal(avg(Psi,SimpSsub(SP,SM,locx1,locx2,locy1,locy2),Psi));
}

template<typename Symmetry>
template<SPIN_INDEX sigma>
Mpo<Symmetry> KondoObservables<Symmetry>::
cdagc (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	assert(locx1<F.size() and locx2<F.size() and locy1<F[locx1].dim() and locy2<F[locx2].dim());
	stringstream ss;
	ss << "c†" << sigma << "(" << locx1 << "," << locy1 << "," << ")" 
	   << "c " << sigma << "(" << locx2 << "," << locy2 << "," << ")";
	
	auto cdag = kroneckerProduct(B[locx1].Id(),F[locx1].cdag(sigma,locy1));
	auto c    = kroneckerProduct(B[locx2].Id(),F[locx2].c   (sigma,locy2));
	auto sign = kroneckerProduct(B[0].Id(),F[0].sign());
	
	Mpo<Symmetry> Mout(F.size(), cdag.Q+c.Q, ss.str());
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l);}
	
	if (locx1 == locx2)
	{
		Mout.setLocal(locx1, cdag*c);
	}
	else if (locx1<locx2)
	{
		Mout.setLocal({locx1, locx2}, {cdag*sign, c}, sign);
	}
	else if (locx1>locx2)
	{
		Mout.setLocal({locx2, locx1}, {c*sign, -1.*cdag}, sign);
	}
	
	return Mout;
}

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
