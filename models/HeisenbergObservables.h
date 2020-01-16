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
typedef SiteOperator<Symmetry,double> OperatorType;

public:
	
	///@{
	HeisenbergObservables(){};
	HeisenbergObservables (const size_t &L); // for inheritance purposes
	HeisenbergObservables (const size_t &L, const vector<Param> &params, const std::map<string,std::any> &defaults);
	///@}
	
	///@{
	Mpo<Symmetry> Scomp (SPINOP_LABEL Sa, size_t locx, size_t locy=0, double factor=1.) const;
	Mpo<Symmetry> ScompScomp (SPINOP_LABEL Sa1, SPINOP_LABEL Sa2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const;
	///@}
	
	///@{
	Mpo<Symmetry> Sz (size_t locx, size_t locy=0) const {return Scomp(SZ,locx,locy);};
	Mpo<Symmetry> Sx (size_t locx, size_t locy=0) const {return Scomp(SX,locx,locy);};
	Mpo<Symmetry> n  (size_t locx, size_t locy=0) const;
	Mpo<Symmetry> SzSz (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const {return ScompScomp(SZ,SZ,locx1,locx2,locy1,locy2,1.);};
	Mpo<Symmetry> SpSm (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const {return ScompScomp(SP,SM,locx1,locx2,locy1,locy2,fac);};
	Mpo<Symmetry> SmSp (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const {return ScompScomp(SM,SP,locx1,locx2,locy1,locy2,fac);};
	Mpo<Symmetry> SxSx (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0, double fac=1.) const {return ScompScomp(SX,SX,locx1,locx2,locy1,locy2,fac);};
	vector<Mpo<Symmetry> > SdagS (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	///@}
	
	///@{
	Mpo<Symmetry> Stringz (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	///@}
	
protected:
	
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
Scomp (SPINOP_LABEL Sa, size_t locx, size_t locy, double factor) const
{
	assert(locx<B.size() and locy<B[locx].dim());
	stringstream ss;
	ss << Sa << "(" << locx << "," << locy << ")";
	
	OperatorType Op = factor * B[locx].Scomp(Sa,locy);
	
	bool HERMITIAN = (Sa==SX or Sa==SZ)? true:false;
	
	Mpo<Symmetry> Mout(B.size(), Op.Q, ss.str(), HERMITIAN);
	for (size_t l=0; l<B.size(); ++l) {Mout.setLocBasis(B[l].get_basis(),l);}
	
	Mout.setLocal(locx,Op);
	return Mout;
}

template<typename Symmetry>
Mpo<Symmetry> HeisenbergObservables<Symmetry>::
n (size_t locx, size_t locy) const
{
	assert(locx<B.size() and locy<B[locx].dim());
	stringstream ss;
	ss << "n(" << locx << "," << locy << ")";
	
	OperatorType Op = B[locx].n(locy);
	
	Mpo<Symmetry> Mout(B.size(), Op.Q, ss.str(), true);
	for (size_t l=0; l<B.size(); ++l) {Mout.setLocBasis(B[l].get_basis(),l);}
	
	Mout.setLocal(locx,Op);
	return Mout;
}

template<typename Symmetry>
Mpo<Symmetry> HeisenbergObservables<Symmetry>::
ScompScomp (SPINOP_LABEL Sa1, SPINOP_LABEL Sa2, size_t locx1, size_t locx2, size_t locy1, size_t locy2, double fac) const
{
	assert(locx1<B.size() and locx2<B.size() and locy1<B[locx1].dim() and locy2<B[locx2].dim());
	stringstream ss;
	ss << Sa1 << "(" << locx1 << "," << locy1 << ")" << Sa2 << "(" << locx2 << "," << locy2 << ")";
	
	OperatorType Op1 = B[locx1].Scomp(Sa1,locy1);
	OperatorType Op2 = B[locx2].Scomp(Sa2,locy2);
	
	bool HERMITIAN = (Sa1==Sa2 and (Sa1==SZ or Sa1==SX) and locx1==locx2 and locy1==locy2)? true:false;
	
	Mpo<Symmetry> Mout(B.size(), Op1.Q+Op2.Q, ss.str(), HERMITIAN);
	for (size_t l=0; l<B.size(); ++l) {Mout.setLocBasis(B[l].get_basis(),l);}
	
	Mout.setLocal({locx1,locx2}, {fac*Op1,Op2});
	return Mout;
}

template<typename Symmetry>
Mpo<Symmetry> HeisenbergObservables<Symmetry>::
Stringz (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	assert(locx1<B.size() and locx2<B.size() and locy1<B[locx1].dim() and locy2<B[locx2].dim());
	stringstream ss;
	ss << "Sz" << "(" << locx1 << "," << locy1 << ")" << "Sz" << "(" << locx2 << "," << locy2 << ")";
	
	OperatorType Op1 = B[locx1].Scomp(SZ,locy1);
	OperatorType Op2 = B[locx2].Scomp(SZ,locy2);
	
	Mpo<Symmetry> Mout(B.size(), Op1.Q+Op2.Q, ss.str(), false);
	for (size_t l=0; l<B.size(); ++l) {Mout.setLocBasis(B[l].get_basis(),l);}
	
	Mout.setLocal({locx1,locx2}, {Op1,Op2}, B[0].beadz());
	return Mout;
}

template<typename Symmetry>
vector<Mpo<Symmetry> >HeisenbergObservables<Symmetry>::
SdagS (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	vector<Mpo<Symmetry> > out(3);
	out[0] = SzSz(locx1,locx2,locy1,locy2);
	out[1] = SpSm(locx1,locx2,locy1,locy2,0.5);
	out[2] = SmSp(locx1,locx2,locy1,locy2,0.5);
	return out;
}

#endif
