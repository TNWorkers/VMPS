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
typedef SiteOperator<Symmetry,double> OperatorType;
	
public:
	
	///@{
	HubbardObservables(){};
	HubbardObservables (const size_t &L); // for inheritance purposes
	HubbardObservables (const size_t &L, const vector<Param> &params, const std::map<string,std::any> &defaults);
	///@}
	
	///@{
	template<SPIN_INDEX sigma>
	Mpo<Symmetry> c (size_t locx, size_t locy=0) const;
	
	template<SPIN_INDEX sigma>
	Mpo<Symmetry> cdag (size_t locx, size_t locy=0) const;
	///@}
	
	///@{
	Mpo<Symmetry> cc (size_t locx, size_t locy=0) const;
	Mpo<Symmetry> cdagcdag (size_t locx, size_t locy=0) const;
	template<SPIN_INDEX sigma> Mpo<Symmetry> cdagc (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	
	Mpo<Symmetry> TpTm (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	Mpo<Symmetry> TmTp (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	Mpo<Symmetry> TzTz (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	///@}
	
	///@{
	Mpo<Symmetry> d (size_t locx, size_t locy=0) const;
	Mpo<Symmetry> dtot() const;
	Mpo<Symmetry> ns (size_t locx, size_t locy=0) const;
	Mpo<Symmetry> nh (size_t locx, size_t locy=0) const;
	Mpo<Symmetry> s (size_t locx, size_t locy=0) const;
	template<SPIN_INDEX sigma> Mpo<Symmetry> n (size_t locx, size_t locy=0) const;
	Mpo<Symmetry> n (size_t locx, size_t locy=0) const;
	template<SPIN_INDEX sigma1, SPIN_INDEX sigma2>
	Mpo<Symmetry> nn (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	Mpo<Symmetry> nn (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	Mpo<Symmetry> hh (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	///@}
	
	///@{
	Mpo<Symmetry> Scomp (SPINOP_LABEL Sa, size_t locx, size_t locy=0, double factor=1.) const;
	Mpo<Symmetry> ScompScomp (SPINOP_LABEL Sa1, SPINOP_LABEL Sa2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	Mpo<Symmetry> SpSm (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const {return ScompScomp(SP,SM,locx1,locx2,locy1,locy2);};
	Mpo<Symmetry> Sz (size_t locx, size_t locy=0) const;
	Mpo<Symmetry> SzSz (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	///@}
	
	///@{
//	Mpo<Symmetry,complex<double> > doublonPacket (complex<double> (*f)(int)) const;
//	Mpo<Symmetry,complex<double> > electronPacket (complex<double> (*f)(int)) const;
//	Mpo<Symmetry,complex<double> > holePacket (complex<double> (*f)(int)) const;
	///@}
	
	///@{
//	Mpo<Symmetry> triplon (SPIN_INDEX sigma, size_t locx, size_t locy=0) const;
//	Mpo<Symmetry> antitriplon (SPIN_INDEX sigma, size_t locx, size_t locy=0) const;
//	Mpo<Symmetry> quadruplon (size_t locx, size_t locy=0) const;
	///@}
	
protected:
	
	Mpo<Symmetry> make_local (string name, 
	                          size_t locx, size_t locy, 
	                          const OperatorType &Op, 
	                          bool FERMIONIC=false, bool HERMITIAN=false) const;
	Mpo<Symmetry> make_corr  (string name1, string name2, 
	                          size_t locx1, size_t locx2, size_t locy1, size_t locy2, 
	                          const OperatorType &Op1, const OperatorType &Op2,
	                          bool BOTH_HERMITIAN=false) const;
	
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
make_local (string name, size_t locx, size_t locy, const OperatorType &Op, bool FERMIONIC, bool HERMITIAN) const
{
	assert(locx<F.size() and locy<F[locx].dim());
	stringstream ss;
	ss << name << "(" << locx << "," << locy << ")";
	
	Mpo<Symmetry> Mout(F.size(), Op.Q, ss.str(), HERMITIAN);
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
	
	(FERMIONIC)? Mout.setLocal(locx, Op, F[0].sign())
	           : Mout.setLocal(locx, Op);
	return Mout;
}

template<typename Symmetry>
Mpo<Symmetry> HubbardObservables<Symmetry>::
make_corr (string name1, string name2, 
           size_t locx1, size_t locx2, size_t locy1, size_t locy2, 
           const OperatorType &Op1, const OperatorType &Op2,
           bool BOTH_HERMITIAN) const
{
	assert(locx1<F.size() and locx2<F.size() and locy1<F[locx1].dim() and locy2<F[locx2].dim());
	stringstream ss;
	ss << name1 << "(" << locx1 << "," << locy1 << ")"
	   << name2 << "(" << locx2 << "," << locy2 << ")";
	
	bool HERMITIAN = (BOTH_HERMITIAN and locx1==locx2 and locy1==locy2)? true:false;
	
	Mpo<Symmetry> Mout(F.size(), Op1.Q+Op2.Q, ss.str(), HERMITIAN);
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
	
	Mout.setLocal({locx1,locx2}, {Op1,Op2});
	return Mout;
}

//-------------

template<typename Symmetry>
template<SPIN_INDEX sigma>
Mpo<Symmetry> HubbardObservables<Symmetry>::
c (size_t locx, size_t locy) const
{
	stringstream ss;
	ss << "c" << sigma;
	return make_local("c", locx,locy, F[locx].c(sigma,locy), true); // FERMIONIC = true
}

template<typename Symmetry>
template<SPIN_INDEX sigma>
Mpo<Symmetry> HubbardObservables<Symmetry>::
cdag (size_t locx, size_t locy) const
{
	stringstream ss;
	ss << "c†" << sigma;
	return make_local(ss.str(), locx,locy, F[locx].cdag(sigma,locy), true); // FERMIONIC = true
}

template<typename Symmetry>
Mpo<Symmetry> HubbardObservables<Symmetry>::
cc (size_t locx, size_t locy) const
{
	stringstream ss;
	ss << "c" << UP << "c" << DN;
	return make_local(ss.str(), locx,locy, F[locx].c(UP,locy)*F[locx].c(DN,locy), false);
}

template<typename Symmetry>
Mpo<Symmetry> HubbardObservables<Symmetry>::
cdagcdag (size_t locx, size_t locy) const
{
	stringstream ss;
	ss << "c†" << DN << "c†" << UP;
	return make_local(ss.str(), locx,locy, F[locx].cdag(DN,locy)*F[locx].cdag(UP,locy), false);
}

template<typename Symmetry>
template<SPIN_INDEX sigma>
Mpo<Symmetry> HubbardObservables<Symmetry>::
cdagc (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	assert(locx1<F.size() and locx2<F.size());
	stringstream ss;
	ss << "c†" << sigma << "(" << locx1 << "," << locy1 << "," << ")" 
	   << "c " << sigma << "(" << locx2 << "," << locy2 << "," << ")";
	
	auto cdag = F[locx1].cdag(sigma,locy1);
	auto c    = F[locx2].c   (sigma,locy2);
	
	Mpo<Symmetry> Mout(F.size(), cdag.Q+c.Q, ss.str());
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
	
	if (locx1 == locx2)
	{
		Mout.setLocal(locx1, cdag*c);
	}
	else if (locx1<locx2)
	{
		Mout.setLocal({locx1, locx2}, {cdag*F[locx1].sign(), c}, F[0].sign());
	}
	else if (locx1>locx2)
	{
		Mout.setLocal({locx2, locx1}, {c*F[locx2].sign(), -1.*cdag}, F[0].sign());
	}
	
	return Mout;
}

template<typename Symmetry>
Mpo<Symmetry> HubbardObservables<Symmetry>::
TpTm (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	stringstream ss;
	return make_corr("T+","T-", locx1,locx2,locy1,locy2, pow(-1.,locx1+locy1)*F[locx1].cc(locy1), pow(-1.,locx2+locy2)*F[locx2].cdagcdag(locy2), false);
}

template<typename Symmetry>
Mpo<Symmetry> HubbardObservables<Symmetry>::
TmTp (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	stringstream ss;
	return make_corr("T+","T-", locx1,locx2,locy1,locy2, pow(-1.,locx2+locy2)*F[locx1].cdagcdag(locy1), pow(-1.,locx1+locy1)*F[locx2].cc(locy2), false);
}

template<typename Symmetry>
Mpo<Symmetry> HubbardObservables<Symmetry>::
TzTz (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	stringstream ss;
	return make_corr("Tz","Tz", locx1,locx2,locy1,locy2, F[locx1].Tz(locy1), F[locx2].Tz(locy2), true);
}

//-------------

template<typename Symmetry>
Mpo<Symmetry> HubbardObservables<Symmetry>::
d (size_t locx, size_t locy) const
{
	return make_local("double_occ", locx,locy, F[locx].d(locy), false, true);
	// FERMIONIC=false, HERMITIAN=true
}

template<typename Symmetry>
Mpo<Symmetry> HubbardObservables<Symmetry>::
dtot() const
{
	for (size_t l=0; l<F.size(); ++l) {assert(F.orbitals()==1);}
	
	OperatorType Op = F[0].d();
	
	Mpo<Symmetry> Mout(F.size(), Op.Q, "double_occ_total", true); // HERMITIAN=true
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
	
	Mout.setLocalSum(Op);
	return Mout;
}

template<typename Symmetry>
Mpo<Symmetry> HubbardObservables<Symmetry>::
s (size_t locx, size_t locy) const
{
	return make_local("single_occ", locx,locy,  F[locx].n(UP,locy)+F[locx].n(DN,locy)-2.*F[locx].d(locy), false, true);
	// FERMIONIC=false, HERMITIAN=true
}

template<typename Symmetry>
template<SPIN_INDEX sigma>
Mpo<Symmetry> HubbardObservables<Symmetry>::
n (size_t locx, size_t locy) const
{
	return make_local("n", locx,locy, F[locx].n(sigma,locy), false, true);
	// FERMIONIC=false, HERMITIAN=true
}

template<typename Symmetry>
Mpo<Symmetry> HubbardObservables<Symmetry>::
n (size_t locx, size_t locy) const
{
	return make_local("n", locx,locy, F[locx].n(locy), false, true);
	// FERMIONIC=false, HERMITIAN=true
}

template<typename Symmetry>
Mpo<Symmetry> HubbardObservables<Symmetry>::
ns (size_t locx, size_t locy) const
{
	return make_local("ns", locx,locy, F[locx].ns(locy), false, true);
	// FERMIONIC=false, HERMITIAN=true
}

template<typename Symmetry>
Mpo<Symmetry> HubbardObservables<Symmetry>::
nh (size_t locx, size_t locy) const
{
	return make_local("nh", locx,locy, F[locx].nh(locy), false, true);
	// FERMIONIC=false, HERMITIAN=true
}

template<typename Symmetry>
template<SPIN_INDEX sigma1, SPIN_INDEX sigma2>
Mpo<Symmetry> HubbardObservables<Symmetry>::
nn (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	return make_corr ("n","n", locx1,locx2,locy1,locy2, F[locx1].n(sigma1,locy1), F[locx2].n(sigma2,locy2), true);
}

template<typename Symmetry>
Mpo<Symmetry> HubbardObservables<Symmetry>::
nn (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	return make_corr ("n","n", locx1,locx2,locy1,locy2, F[locx1].n(locy1), F[locx2].n(locy2), true);
}

template<typename Symmetry>
Mpo<Symmetry> HubbardObservables<Symmetry>::
hh (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	return make_corr("h","h", locx1,locx2,locy1,locy2, 
	                 F[locx1].d(locy1)-F[locx1].n(locy1)+F[locx1].Id(),
	                 F[locx2].d(locy2)-F[locx2].n(locy2)+F[locx2].Id(),
	                 true);
}

template<typename Symmetry>
Mpo<Symmetry> HubbardObservables<Symmetry>::
Scomp (SPINOP_LABEL Sa, size_t locx, size_t locy, double factor) const
{
	stringstream ss; ss << Sa;
	bool HERMITIAN = (Sa==SX or Sa==SZ)? true:false;
	return make_local(ss.str(), locx,locy, factor*F[locx].Scomp(Sa,locy), false, HERMITIAN);
}

template<typename Symmetry>
Mpo<Symmetry> HubbardObservables<Symmetry>::
ScompScomp (SPINOP_LABEL Sa1, SPINOP_LABEL Sa2, size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	stringstream ss1; ss1 << Sa1;
	stringstream ss2; ss2 << Sa2;
	bool HERMITIAN = false;
	return make_corr(ss1.str(),ss2.str(), locx1,locx2,locy1,locy2, F[locx1].Scomp(Sa1,locy1), F[locx2].Scomp(Sa2,locy2), HERMITIAN);
}

template<typename Symmetry>
Mpo<Symmetry> HubbardObservables<Symmetry>::
Sz (size_t locx, size_t locy) const
{
	return Scomp(SZ,locx,locy);
}

template<typename Symmetry>
Mpo<Symmetry> HubbardObservables<Symmetry>::
SzSz (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	return ScompScomp(SZ,SZ,locx1,locx2,locy1,locy2);
}

#endif
