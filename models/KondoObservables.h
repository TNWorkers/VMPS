#ifndef KONDOOBSERVABLES
#define KONDOOBSERVABLES

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
	MpoQ<Symmetry> c    (SPIN_INDEX sigma, size_t locx, size_t locy=0) const;
	MpoQ<Symmetry> cdag (SPIN_INDEX sigma, size_t locx, size_t locy=0) const;
	///@}
	
	///@{
	MpoQ<Symmetry> n (size_t locx, size_t locy=0) const;
	MpoQ<Symmetry> d (size_t locx, size_t locy=0) const;
	MpoQ<Symmetry> cdagc (SPIN_INDEX sigma, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	///@}
	
	///@{
	MpoQ<Symmetry> Simp (SPINOP_LABEL SOP, size_t locx, size_t locy=0) const;
	MpoQ<Symmetry> Ssub (SPINOP_LABEL SOP, size_t locx, size_t locy=0) const;
	///@}
	
	///@{
	MpoQ<Symmetry> SimpSimp (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	MpoQ<Symmetry> SsubSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	MpoQ<Symmetry> SimpSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	///@}
	
	///@{ not implemented
//	MpoQ<Symmetry> SimpSsubSimpSimp (size_t locx1, SPINOP_LABEL SOP1, size_t locx2, SPINOP_LABEL SOP2, 
//	                          size_t loc3x, SPINOP_LABEL SOP3, size_t loc4x, SPINOP_LABEL SOP4,
//	                          size_t locy1=0, size_t locy2=0, size_t loc3y=0, size_t loc4y=0) const;
//	MpoQ<Symmetry> SimpSsubSimpSsub (size_t locx1, SPINOP_LABEL SOP1, size_t locx2, SPINOP_LABEL SOP2, 
//	                          size_t loc3x, SPINOP_LABEL SOP3, size_t loc4x, SPINOP_LABEL SOP4,
//	                          size_t locy1=0, size_t locy2=0, size_t loc3y=0, size_t loc4y=0) const;
	///@}
	
	template<typename MpsType> double SvecSvecAvgImpSub (const MpsType &Psi, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	
protected:
	
	MpoQ<Symmetry> make_local (string name, size_t locx, size_t locy, const OperatorType &Oimp, const OperatorType &Osub, bool FERMIONIC=false) const;
	
	MpoQ<Symmetry> make_corr  (string name1, string name2, 
	                          size_t locx1, size_t locx2, size_t locy1, size_t locy2, 
	                          const OperatorType &Oimp1, const OperatorType &Osub1,
	                          const OperatorType &Oimp2, const OperatorType &Osub2) const;
	
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
		B[l] = SpinBase<Symmetry> (P.get<size_t>("Ly",l%Lcell), P.get<size_t>("D",l%Lcell), true);
		// True means that N is the good quantum number. Hence, only a dummy-0 is returned.
		F[l] = FermionBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), !isfinite(P.get<double>("U",l%Lcell)), true);
		// True means that N,M are the good quantum numbers.
	}
}

//-------------

template<typename Symmetry>
MpoQ<Symmetry> KondoObservables<Symmetry>::
make_local (string name, size_t locx, size_t locy, const OperatorType &Oimp, const OperatorType &Osub, bool FERMIONIC) const
{
	assert(locx<F.size() and locy<F[locx].dim());
	stringstream ss;
	ss << name << "(" << locx << "," << locy << ")";
	
	MpoQ<Symmetry> Mout(F.size(), Oimp.Q+Osub.Q, defaultQlabel<Symmetry::Nq>(), ss.str());
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l);}
	
	if (FERMIONIC)
	{
		Mout.setLocal(locx, kroneckerProduct(Oimp,Osub), kroneckerProduct(B[locx].Id(),F[0].sign()));
	}
	else
	{
		Mout.setLocal(locx, kroneckerProduct(Oimp,Osub));
	}
	return Mout;
}

template<typename Symmetry>
MpoQ<Symmetry> KondoObservables<Symmetry>::
make_corr  (string name1, string name2, size_t locx1, size_t locx2, size_t locy1, size_t locy2, 
            const OperatorType &Oimp1, const OperatorType &Osub1, const OperatorType &Oimp2, const OperatorType &Osub2) const
{
	assert(locx1<F.size() and locx2<F.size() and locy1<F[locx1].dim() and locy2<F[locx2].dim());
	stringstream ss;
	ss << name1 << "(" << locx1 << "," << locy1 << ")"
	   << name2 << "(" << locx2 << "," << locy2 << ")";
	
	MpoQ<Symmetry> Mout(F.size(), Oimp1.Q+Oimp2.Q+Osub1.Q+Osub2.Q, defaultQlabel<Symmetry::Nq>(), ss.str());
	for (size_t l=0; l<F.size(); ++l)  {Mout.setLocBasis(Symmetry::reduceSilent(B[l].get_basis(),F[l].get_basis()),l);}
	
	Mout.setLocal({locx1,locx2}, {kroneckerProduct(Oimp1,Osub1), kroneckerProduct(Oimp2,Osub2)});
	return Mout;
}

//-------------

template<typename Symmetry>
MpoQ<Symmetry> KondoObservables<Symmetry>::
c (SPIN_INDEX sigma, size_t locx, size_t locy) const
{
	stringstream ss;
	ss << "c" << sigma;
	return make_local(ss.str(), locx,locy, B[locx].Id(),F[locx].c(sigma,locy), true);
}

template<typename Symmetry>
MpoQ<Symmetry> KondoObservables<Symmetry>::
cdag (SPIN_INDEX sigma, size_t locx, size_t locy) const
{
	stringstream ss;
	ss << "c†" << sigma;
	return make_local(ss.str(), locx,locy, B[locx].Id(),F[locx].cdag(sigma,locy), true);
}

//-------------

template<typename Symmetry>
MpoQ<Symmetry> KondoObservables<Symmetry>::
n (size_t locx, size_t locy) const
{
	return make_local("n", locx,locy, B[locx].Id(),F[locx].n(locy));
}

template<typename Symmetry>
MpoQ<Symmetry> KondoObservables<Symmetry>::
d (size_t locx, size_t locy) const
{
	return make_local("double_occ", locx,locy, B[locx].Id(),F[locx].d(locy));
}

template<typename Symmetry>
MpoQ<Symmetry> KondoObservables<Symmetry>::
cdagc (SPIN_INDEX sigma, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	assert(locx1<F.size() and locx2<F.size() and locy1<F[locx1].dim() and locy2<F[locx2].dim());
	stringstream ss;
	ss << "c†" << sigma << "(" << locx1 << "," << locy1 << "," << ")" 
	   << "c " << sigma << "(" << locx2 << "," << locy2 << "," << ")";
	
	auto cdag = kroneckerProduct(B[locx1].Id(),F[locx1].cdag(sigma,locy1));
	auto c    = kroneckerProduct(B[locx2].Id(),F[locx2].c   (sigma,locy2));
	auto sign = kroneckerProduct(B[0].Id(),F[0].sign());
	
	MpoQ<Symmetry> Mout(F.size(), cdag.Q+c.Q, defaultQlabel<Symmetry::Nq>(), ss.str());
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

//-------------

template<typename Symmetry>
MpoQ<Symmetry> KondoObservables<Symmetry>::
Simp (SPINOP_LABEL Sa, size_t locx, size_t locy) const
{
	stringstream ss;
	ss << Sa << "imp";
	return make_local(ss.str(), locx,locy, B[locx].Scomp(Sa,locy), F[locx].Id());
}

template<typename Symmetry>
MpoQ<Symmetry> KondoObservables<Symmetry>::
Ssub (SPINOP_LABEL Sa, size_t locx, size_t locy) const
{
	stringstream ss;
	ss << Sa << "sub";
	return make_local(ss.str(), locx,locy, B[locx].Id(), F[locx].Scomp(Sa,locy));
}

//-------------

template<typename Symmetry>
MpoQ<Symmetry> KondoObservables<Symmetry>::
SimpSimp (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	stringstream ss1; ss1 << SOP1 << "imp";
	stringstream ss2; ss2 << SOP2 << "imp";
	
	return make_corr(ss1.str(), ss2.str(), locx1,locx2,locy1,locy2, B[locx1].Scomp(SOP1,locy1), F[locx1].Id(), 
	                                                                B[locx2].Scomp(SOP2,locy2), F[locx2].Id());
}

template<typename Symmetry>
MpoQ<Symmetry> KondoObservables<Symmetry>::
SsubSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	stringstream ss1; ss1 << SOP1 << "sub";
	stringstream ss2; ss2 << SOP2 << "sub";
	
	return make_corr(ss1.str(), ss2.str(), locx1,locx2,locy1,locy2, B[locx1].Id(), F[locx1].Scomp(SOP1,locy1), 
	                                                                B[locx1].Id(), F[locx2].Scomp(SOP2,locy2));
}

template<typename Symmetry>
MpoQ<Symmetry> KondoObservables<Symmetry>::
SimpSsub (SPINOP_LABEL SOP1, SPINOP_LABEL SOP2, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	stringstream ss1; ss1 << SOP1 << "imp";
	stringstream ss2; ss2 << SOP2 << "sub";
	
	return make_corr(ss1.str(), ss2.str(), locx1,locx2,locy1,locy2, B[locx1].Scomp(SOP1,locy1), F[locx1].Id(), 
	                                                                B[locx2].Id(), F[locx2].Scomp(SOP2,locy2));
}

template<typename Symmetry>
template<typename MpsType>
double KondoObservables<Symmetry>::
SvecSvecAvgImpSub (const MpsType &Psi, size_t locx1, size_t locx2, size_t locy1, size_t locy2)
{
	return isReal(avg(Psi,SimpSsub(SZ,SZ,locx1,locx2,locy1,locy2),Psi))+
	       isReal(avg(Psi,SimpSsub(SP,SM,locx1,locx2,locy1,locy2),Psi));
}

// MpoQ<Sym::U1xU1<double> > KondoU1xU1::
// SimpSsubSimpSimp (size_t locx1, SPINOP_LABEL SOP1, size_t locx2, SPINOP_LABEL SOP2, size_t loc3x, SPINOP_LABEL SOP3, size_t loc4x, SPINOP_LABEL SOP4,
//                   size_t locy1, size_t locy2, size_t loc3y, size_t loc4y)
// {
// 	assert(locx1<this->F.size() and locx2<this->F.size() and loc3x<this->F.size() and loc4x<this->F.size());
// 	stringstream ss;
// 	ss << SOP1 << "(" << locx1 << "," << locy1 << ")" << SOP2 << "(" << locx2 << "," << locy2 << ")" <<
// 	      SOP3 << "(" << loc3x << "," << loc3y << ")" << SOP4 << "(" << loc4x << "," << loc4y << ")";
// 	MpoQ<Symmetry> Mout(this->F.size(), this->N_legs, locBasis(), {0,0}, KondoU1xU1::NMlabel, ss.str());
// 	Mout.setLocal({locx1, locx2, loc3x, loc4x},
// 				  {kroneckerProduct(B.Scomp(SOP1,locy1),F.Id()), 
// 				   kroneckerProduct(B.Id(),F.Scomp(SOP2,locy2)),
// 				   kroneckerProduct(B.Scomp(SOP3,loc3y),F.Id()),
// 				   kroneckerProduct(B.Scomp(SOP4,loc4y),F.Id())});
// 	return Mout;
// }

// MpoQ<Sym::U1xU1<double> > KondoU1xU1::
// SimpSsubSimpSsub (size_t locx1, SPINOP_LABEL SOP1, size_t locx2, SPINOP_LABEL SOP2, size_t loc3x, SPINOP_LABEL SOP3, size_t loc4x, SPINOP_LABEL SOP4,
//                   size_t locy1, size_t locy2, size_t loc3y, size_t loc4y)
// {
// 	assert(locx1<this->F.size() and locx2<this->F.size() and loc3x<this->F.size() and loc4x<this->F.size());
// 	stringstream ss;
// 	ss << SOP1 << "(" << locx1 << "," << locy1 << ")" << SOP2 << "(" << locx2 << "," << locy2 << ")" <<
// 	      SOP3 << "(" << loc3x << "," << loc3y << ")" << SOP4 << "(" << loc4x << "," << loc4y << ")";
// 	MpoQ<Symmetry> Mout(this->F.size(), this->N_legs, locBasis(), {0,0}, KondoU1xU1::NMlabel, ss.str());
// 	SparseMatrixXd IdSub(F.dim(),F.dim()); IdSub.setIdentity();
// 	SparseMatrixXd IdImp(MpoQ<Symmetry>::qloc[locx2].size()/F.dim(), MpoQ<Symmetry>::qloc[locx2].size()/F.dim()); IdImp.setIdentity();
// 	Mout.setLocal({locx1, locx2, loc3x, loc4x},
// 				  {kroneckerProduct(B.Scomp(SOP1,locy1),F.Id()), 
// 				   kroneckerProduct(B.Id(),F.Scomp(SOP2,locy2)),
// 				   kroneckerProduct(B.Scomp(SOP3,loc3y),F.Id()),
// 				   kroneckerProduct(B.Id(),F.Scomp(SOP4,loc4y))});
// 	return Mout;
// }

#endif
