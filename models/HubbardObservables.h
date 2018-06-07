#ifndef HUBBARDOBSERVABLES
#define HUBBARDOBSERVABLES

#include "Mpo.h"
#include "ParamHandler.h" // from HELPERS
#include "bases/SpinBase.h"
#include "DmrgLinearAlgebra.h"
#include "DmrgExternal.h"
#include "tensors/SiteOperator.h"

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
	Mpo<Symmetry> eta() const;
	///@}
	
	///@{
	Mpo<Symmetry> d (size_t locx, size_t locy=0) const;
	Mpo<Symmetry> dtot() const;
	Mpo<Symmetry> s (size_t locx, size_t locy=0) const;
	template<SPIN_INDEX sigma> Mpo<Symmetry> n (size_t locx, size_t locy=0) const;
	Mpo<Symmetry> n (size_t locx, size_t locy=0) const;
	template<SPIN_INDEX sigma1, SPIN_INDEX sigma2>
	Mpo<Symmetry> nn (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	Mpo<Symmetry> hh (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	///@}
	
	///@{
	Mpo<Symmetry> Scomp (SPINOP_LABEL Sa, size_t locx, size_t locy=0, double factor=1.) const;
	Mpo<Symmetry> ScompScomp (SPINOP_LABEL Sa1, SPINOP_LABEL Sa2, size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
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
	return make_local("c", locx,locy, F[locx].c(sigma,locy), true);
}

template<typename Symmetry>
template<SPIN_INDEX sigma>
Mpo<Symmetry> HubbardObservables<Symmetry>::
cdag (size_t locx, size_t locy) const
{
	stringstream ss;
	ss << "c†" << sigma;
	return make_local(ss.str(), locx,locy, F[locx].cdag(sigma,locy), true);
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
eta() const
{
	for (size_t l=0; l<F.size(); ++l) {assert(F.orbitals()==1);}
	
	OperatorType Op = F[0].c(UP)*F[0].c(DN);
	
	Mpo<Symmetry> Mout(F.size(), Op.Q, "eta");
	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
	
	Mout.setLocalSum(Op,stagger);
	return Mout;
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
template<SPIN_INDEX sigma1, SPIN_INDEX sigma2>
Mpo<Symmetry> HubbardObservables<Symmetry>::
nn (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
{
	return make_corr ("n","n", locx1,locx2,locy1,locy2, F[locx1].n(sigma1,locy1), F[locx2].n(sigma2,locy2), true);
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
	bool HERMITIAN = ((Sa1==SX or Sa2==SZ) and (Sa2==SX or Sa2==SZ))? true:false;
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





//template<typename Symmetry>
//Mpo<Sym::U1xU1<double>,complex<double> > HubbardObservables<Symmetry>::
//doublonPacket (complex<double> (*f)(int))
//{
//	stringstream ss;
//	ss << "doublonPacket";
//	
//	Mpo<Symmetry,complex<double> > Mout(F.size(), qarray<Symmetry::Nq>({-1,-1}), HubbardU1xU1::Nlabel, ss.str());
//	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F.get_basis(),l);}
//	
//	Mout.setLocalSum(F.c(UP)*F.c(DN), f);
//	return Mout;
//}

//template<typename Symmetry>
//Mpo<Sym::U1xU1<double>,complex<double> > HubbardObservables<Symmetry>::
//electronPacket (complex<double> (*f)(int))
//{
//	assert(N_legs==1);
//	stringstream ss;
//	ss << "electronPacket";
//	
//	qarray<2> qdiff = {+1,0};
//	
//	vector<SuperMatrix<Symmetry,complex<double> > > M(F.size());
//	M[0].setRowVector(2,F.dim());
////	M[0](0,0) = f(0) * F.cdag(UP);
//	M[0](0,0).data = f(0) * F.cdag(UP).data; M[0](0,0).Q = F.cdag(UP).Q;
//	M[0](0,1) = F.Id();
//	
//	for (size_t l=1; l<F.size()-1; ++l)
//	{
//		M[l].setMatrix(2,F.dim());
////		M[l](0,0) = complex<double>(1.,0.) * F.sign();
//		M[l](0,0).data = complex<double>(1.,0.) * F.sign().data; M[l](0,0).Q = F.sign().Q;
////		M[l](1,0) = f(l) * F.cdag(UP);
//		M[l](1,0).data = f(l) * F.cdag(UP).data; M[l](1,0).Q = F.cdag(UP).Q;
//		M[l](0,1).setZero();
//		M[l](1,1) = F.Id();
//	}
//	
//	M[F.size()-1].setColVector(2,F.dim());
////	M[F.size()-1](0,0) = complex<double>(1.,0.) * F.sign();
//	M[F.size()-1](0,0).data = complex<double>(1.,0.) * F.sign().data; M[F.size()-1](0,0).Q = F.sign().Q;
////	M[F.size()-1](1,0) = f(F.size()-1) * F.cdag(UP);
//	M[F.size()-1](1,0).data = f(F.size()-1) * F.cdag(UP).data; M[F.size()-1](1,0).Q = F.cdag(UP).Q;
//	
//	Mpo<Symmetry,complex<double> > Mout(F.size(), M, qarray<Symmetry::Nq>(qdiff), HubbardU1xU1::Nlabel, ss.str());
//	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F.get_basis(),l);}
//	return Mout;
//}

//template<typename Symmetry>
//Mpo<Sym::U1xU1<double>,complex<double> > HubbardObservables<Symmetry>::
//holePacket (complex<double> (*f)(int))
//{
//	assert(N_legs==1);
//	stringstream ss;
//	ss << "holePacket";
//	
//	qarray<2> qdiff = {-1,0};
//	
//	vector<SuperMatrix<Symmetry,complex<double> > > M(F.size());
//	M[0].setRowVector(2,F.dim());
//	M[0](0,0) = f(0) * F.c(UP);
//	M[0](0,1) = F.Id();
//	
//	for (size_t l=1; l<F.size()-1; ++l)
//	{
//		M[l].setMatrix(2,F.dim());
//		M[l](0,0) = complex<double>(1.,0.) * F.sign();
//		M[l](1,0) = f(l) * F.c(UP);
//		M[l](0,1).setZero();
//		M[l](1,1) = F.Id();
//	}
//	
//	M[F.size()-1].setColVector(2,F.dim());
//	M[F.size()-1](0,0) = complex<double>(1.,0.) * F.sign();
//	M[F.size()-1](1,0) = f(F.size()-1) * F.c(UP);
//	
//	Mpo<Symmetry,complex<double> > Mout(F.size(), M, qarray<Symmetry::Nq>(qdiff), HubbardU1xU1::Nlabel, ss.str());
//	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F.get_basis(),l);}
//	return Mout;
//}

//template<typename Symmetry>
//Mpo<Symmetry> HubbardObservables<Symmetry>::
//triplon (SPIN_INDEX sigma, size_t locx, size_t locy)
//{
//	assert(locx<F.size() and locy<F[locx].dim());
//	stringstream ss;
//	ss << "triplon(" << locx << ")" << "c(" << locx+1 << ",σ=" << sigma << ")";
//	
//	qarray<2> qdiff;
//	(sigma==UP) ? qdiff = {-2,-1} : qdiff = {-1,-2};
//	
//	vector<SuperMatrix<Symmetry,double> > M(F.size());
//	for (size_t l=0; l<locx; ++l)
//	{
//		M[l].setMatrix(1,F[l].dim());
//		M[l](0,0) = F[l].sign();
//	}
//	// c(locx,UP)*c(locx,DN)
//	M[locx].setMatrix(1,F[locx].dim());
//	M[locx](0,0) = F[locx].c(UP,locy)*F[locx].c(DN,locy);
//	// c(locx+1,UP|DN)
//	M[locx+1].setMatrix(1,F[locx+1].dim());
//	M[locx+1](0,0) = (sigma==UP)? F[locx+1].c(UP,locy) : F[locx+1].c(DN,locy);
//	for (size_t l=locx+2; l<F.size(); ++l)
//	{
//		M[l].setMatrix(1,F[l].dim());
//		M[l](0,0) = F[l].Id();
//	}
//	
//	Mpo<Symmetry> Mout(F.size(), M, qarray<Symmetry::Nq>(qdiff), HubbardU1xU1::Nlabel, ss.str());
//	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
//	return Mout;
//}

//template<typename Symmetry>
//Mpo<Symmetry> HubbardObservables<Symmetry>::
//antitriplon (SPIN_INDEX sigma, size_t locx, size_t locy)
//{
//	assert(locx<F.size() and locy<F[locx].dim());
//	stringstream ss;
//	ss << "antitriplon(" << locx << ")" << "c(" << locx+1 << ",σ=" << sigma << ")";
//	
//	qarray<2> qdiff;
//	(sigma==UP) ? qdiff = {+2,+1} : qdiff = {+1,+2};
//	
//	vector<SuperMatrix<Symmetry,double> > M(F.size());
//	for (size_t l=0; l<locx; ++l)
//	{
//		M[l].setMatrix(1,F[l].dim());
//		M[l](0,0) = F[l].sign();
//	}
//	// c†(locx,DN)*c†(locx,UP)
//	M[locx].setMatrix(1,F[locx].dim());
//	M[locx](0,0) = F[locx].cdag(DN,locy)*F[locx].cdag(UP,locy);
//	// c†(locx+1,UP|DN)
//	M[locx+1].setMatrix(1,F[locx+1].dim());
//	M[locx+1](0,0) = (sigma==UP)? F[locx+1].cdag(UP,locy) : F[locx+1].cdag(DN,locy);
//	for (size_t l=locx+2; l<F.size(); ++l)
//	{
//		M[l].setMatrix(1,F[l].dim());
//		M[l](0,0) = F[l].Id();
//	}
//	
//	Mpo<Symmetry> Mout(F.size(), M, qarray<Symmetry::Nq>(qdiff), HubbardU1xU1::Nlabel, ss.str());
//	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
//	return Mout;
//}

//template<typename Symmetry>
//Mpo<Symmetry> HubbardObservables<Symmetry>::
//quadruplon (size_t locx, size_t locy)
//{
//	assert(locx<F.size() and locy<F[locx].dim());
//	stringstream ss;
//	ss << "Auger(" << locx << ")" << "Auger(" << locx+1 << ")";
//	
//	vector<SuperMatrix<Symmetry,double> > M(F.size());
//	for (size_t l=0; l<locx; ++l)
//	{
//		M[l].setMatrix(1,F[l].dim());
//		M[l](0,0) = F[l].Id();
//	}
//	// c(loc,UP)*c(loc,DN)
//	M[locx].setMatrix(1,F[locx].dim());
//	M[locx](0,0) = F[locx].c(UP,locy)*F[locx].c(DN,locy);
//	// c(loc+1,UP)*c(loc+1,DN)
//	M[locx+1].setMatrix(1,F[locx+1].dim());
//	M[locx+1](0,0) = F[locx+1].c(UP,locy)*F[locx+1].c(DN,locy);
//	for (size_t l=locx+2; l<F.size(); ++l)
//	{
//		M[l].setMatrix(1,4);
//		M[l](0,0) = F[l].Id();
//	}
//	
//	Mpo<Symmetry> Mout(F.size(), M, qarray<Symmetry::Nq>({-2,-2}), HubbardU1xU1::Nlabel, ss.str());
//	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
//	return Mout;
//}


#endif
