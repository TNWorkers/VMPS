#ifndef SPINLESSFERMIONSOBSERVABLES
#define SPINLESSFERMIONSOBSERVABLES

#include "bases/SpinlessFermionBase.h"
#include "Mpo.h"
#include "ParamHandler.h" // from HELPERS

template<typename Symmetry, typename Scalar=double>
class SpinlessFermionsObservables
{
	typedef SiteOperatorQ<Symmetry,Eigen::MatrixXd> OperatorType;
	
public:
	
	///@{
	SpinlessFermionsObservables(){};
	SpinlessFermionsObservables (const size_t &L); // for inheritance purposes
	SpinlessFermionsObservables (const size_t &L, const vector<Param> &params, const std::map<string,std::any> &defaults);
	///@}
	
	///@{
	Mpo<Symmetry,Scalar> c (size_t locx, size_t locy=0) const;
	Mpo<Symmetry,Scalar> cdag (size_t locx, size_t locy=0) const;
	/**1/√2*(c+c†), only possible with Z(2) symmetry or without symmetries*/
//	Mpo<Symmetry,Scalar> c_plus_cdag (size_t locx, size_t locy=0) const;
	///@}
	
	///@{
//	Mpo<Symmetry,Scalar> cdagc (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	///@}
	
	///@{
	/**Is of course zero for spinless fermions, good to have for consistency.*/
//	Mpo<Symmetry,Scalar> d (size_t locx, size_t locy=0) const;
	Mpo<Symmetry,Scalar> n (size_t locx, size_t locy=0) const;
	Mpo<Symmetry,Scalar> nph (size_t locx, size_t locy=0) const;
//	Mpo<Symmetry,Scalar> nn (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0) const;
	///@}
	
protected:
	
	Mpo<Symmetry,Scalar> make_local (size_t locx, size_t locy, const OperatorType &Op, double factor, bool FERMIONIC=false, bool HERMITIAN=false) const;
//	Mpo<Symmetry,Scalar> make_corr  (string name1, string name2, 
//	                          size_t locx1, size_t locx2, size_t locy1, size_t locy2, 
//	                          const OperatorType &Op1, const OperatorType &Op2,
//	                          bool BOTH_HERMITIAN=false) const;
	
	vector<SpinlessFermionBase<Symmetry> > F;
};

template<typename Symmetry, typename Scalar>
SpinlessFermionsObservables<Symmetry,Scalar>::
SpinlessFermionsObservables (const size_t &L)
{
	F.resize(L);
}

template<typename Symmetry, typename Scalar>
SpinlessFermionsObservables<Symmetry,Scalar>::
SpinlessFermionsObservables (const size_t &L, const vector<Param> &params, const std::map<string,std::any> &defaults)
{
	ParamHandler P(params,defaults);
	size_t Lcell = P.size();
	F.resize(L);
	
	for (size_t l=0; l<L; ++l)
	{
		F[l] = SpinlessFermionBase<Symmetry>(P.get<size_t>("Ly",l%Lcell));
	}
}

//-------------

template<typename Symmetry, typename Scalar>
Mpo<Symmetry,Scalar> SpinlessFermionsObservables<Symmetry,Scalar>::
make_local (size_t locx, size_t locy, const OperatorType &Op, double factor, bool FERMIONIC, bool HERMITIAN) const
{
	assert(locx<F.size() and locy<F[locx].dim());
	stringstream ss;
	ss << Op.label() << "(" << locx << "," << locy;
	if (factor != 1.) ss << ",factor=" << factor;
	ss << ")";
	
	Mpo<Symmetry,Scalar> Mout(F.size(), Op.Q(), ss.str(), HERMITIAN);
	for (size_t l=0; l<F.size(); ++l) Mout.setLocBasis(F[l].get_basis().qloc(),l);
	
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

//template<typename Symmetry, typename Scalar>
//Mpo<Symmetry,Scalar> SpinlessFermionsObservables<Symmetry,Scalar>::
//make_corr (string name1, string name2, 
//           size_t locx1, size_t locx2, size_t locy1, size_t locy2, 
//           const OperatorType &Op1, const OperatorType &Op2,
//           bool BOTH_HERMITIAN) const
//{
//	assert(locx1<F.size() and locx2<F.size() and locy1<F[locx1].dim() and locy2<F[locx2].dim());
//	stringstream ss;
//	ss << name1 << "(" << locx1 << "," << locy1 << ")"
//	   << name2 << "(" << locx2 << "," << locy2 << ")";
//	
//	bool HERMITIAN = (BOTH_HERMITIAN and locx1==locx2 and locy1==locy2)? true:false;
//	
//	Mpo<Symmetry,Scalar> Mout(F.size(), Op1.Q+Op2.Q, ss.str(), HERMITIAN);
//	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
//	
//	Mout.setLocal({locx1,locx2}, {Op1,Op2});
//	return Mout;
//}

//-------------

template<typename Symmetry, typename Scalar>
Mpo<Symmetry,Scalar> SpinlessFermionsObservables<Symmetry,Scalar>::
c (size_t locx, size_t locy) const
{
	return make_local(locx,locy, F[locx].c(locy), 1., true);
}

template<typename Symmetry, typename Scalar>
Mpo<Symmetry,Scalar> SpinlessFermionsObservables<Symmetry,Scalar>::
cdag (size_t locx, size_t locy) const
{
	return make_local(locx,locy, F[locx].cdag(locy), 1., true);
}

//template<typename Symmetry, typename Scalar>
//Mpo<Symmetry,Scalar> SpinlessFermionsObservables<Symmetry,Scalar>::
//c_plus_cdag (size_t locx, size_t locy) const
//{
//	assert(Symmetry::name() != "U(1)");
//	return make_local("c+c†", locx,locy, 1./sqrt(2.)*(F[locx].c(locy)+F[locx].cdag(locy)), true);
//}

template<typename Symmetry, typename Scalar>
Mpo<Symmetry,Scalar> SpinlessFermionsObservables<Symmetry,Scalar>::
n (size_t locx, size_t locy) const
{
	return make_local(locx,locy, F[locx].n(locy), 1., false, true);
}

template<typename Symmetry, typename Scalar>
Mpo<Symmetry,Scalar> SpinlessFermionsObservables<Symmetry,Scalar>::
nph (size_t locx, size_t locy) const
{
	return make_local(locx,locy, F[locx].nph(locy), 1., false, true);
}

//template<typename Symmetry, typename Scalar>
//Mpo<Symmetry,Scalar> SpinlessFermionsObservables<Symmetry,Scalar>::
//d (size_t locx, size_t locy) const
//{
//	return make_local("d", locx,locy, F[locx].Zero(), false, true);
//}

//template<typename Symmetry, typename Scalar>
//Mpo<Symmetry,Scalar> SpinlessFermionsObservables<Symmetry,Scalar>::
//cdagc (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
//{
//	assert(locx1<F.size() and locx2<F.size());
//	stringstream ss;
//	ss << "c†" << "(" << locx1 << "," << locy1 << "," << ")" 
//	   << "c " << "(" << locx2 << "," << locy2 << "," << ")";
//	
//	auto cdag = F[locx1].cdag(locy1);
//	auto c    = F[locx2].c   (locy2);
//	
//	Mpo<Symmetry,Scalar> Mout(F.size(), cdag.Q+c.Q, ss.str());
//	for (size_t l=0; l<F.size(); ++l) {Mout.setLocBasis(F[l].get_basis(),l);}
//	
//	if (locx1 == locx2)
//	{
//		Mout.setLocal(locx1, cdag*c);
//	}
//	else if (locx1<locx2)
//	{
//		Mout.setLocal({locx1, locx2}, {cdag*F[locx1].sign(), c}, F[0].sign());
//	}
//	else if (locx1>locx2)
//	{
//		Mout.setLocal({locx2, locx1}, {c*F[locx2].sign(), -1.*cdag}, F[0].sign());
//	}
//	
//	return Mout;
//}

//-------------

//template<typename Symmetry, typename Scalar>
//Mpo<Symmetry,Scalar> SpinlessFermionsObservables<Symmetry,Scalar>::
//nn (size_t locx1, size_t locx2, size_t locy1, size_t locy2) const
//{
//	return make_corr ("n","n", locx1,locx2,locy1,locy2, F[locx1].n(locy1), F[locx2].n(locy2), true);
//}

#endif
