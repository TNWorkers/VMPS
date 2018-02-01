#ifndef HUBBARDMODELSU2XSU2_H_
#define HUBBARDMODELSU2XSU2_H_

#include "bases/FermionBaseSU2xSU2.h"
#include "symmetry/S1xS2.h"
#include "Mpo.h"
#include "DmrgExternal.h"
#include "ParamHandler.h"

namespace VMPS
{

/** 
 * \class HubbardSU2xSU2
 * \ingroup Hubbard
 *
 * \brief Hubbard Model
 *
 * MPO representation of 
 * 
 * \f[
 * H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
 * + U \sum_i \left[\left(n_{i\uparrow}-\frac{1}{2}\right)\left(n_{i\downarrow}-\frac{1}{2}\right) -\frac{1}{4}\right]
 * \f]
 *
 * \note Take use of the Spin SU(2) symmetry and SU(2) charge symmetry.
 * \warning Bipartite hopping structure is mandatory! (Particle-hole symmetry)
 * \warning \f$J>0\f$ is antiferromagnetic
 * \todo Implement spin and pseudo spin observables.
 */
class HubbardSU2xSU2 : public Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > ,double>
{
public:
	typedef Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > Symmetry;
	
private:
	
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	
public:
	
	HubbardSU2xSU2() : Mpo(){};
	HubbardSU2xSU2 (const size_t &L, const vector<Param> &params);
	
	static HamiltonianTermsXd<Symmetry> set_operators (const vector<FermionBase<Symmetry> > &F, const ParamHandler &P, size_t loc=0);
		
//	Mpo<Symmetry> Auger (size_t locx, size_t locy=0);
//	Mpo<Symmetry> eta(size_t locx, size_t locy=0);
//	Mpo<Symmetry> Aps (size_t locx, size_t locy=0);
	Mpo<Symmetry> c (size_t locx, size_t locy=0);
	Mpo<Symmetry> cdag (size_t locx, size_t locy=0);
	
	Mpo<Symmetry> cdagc (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	Mpo<Symmetry> nh (size_t locx, size_t locy=0);
	Mpo<Symmetry> ns (size_t locx, size_t locy=0);
	
	// MpoQ<Symmetry> S (size_t locx, size_t locy=0);
	// MpoQ<Symmetry> Sdag (size_t locx, size_t locy=0);
	// MpoQ<Symmetry> SSdag (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	
	// MpoQ<Symmetry> T (size_t locx, size_t locy=0);
	// MpoQ<Symmetry> Tdag (size_t locx, size_t locy=0);
	// MpoQ<Symmetry> TTdag (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);
	
//	MpoQ<Symmetry> EtaEtadag (size_t locx1, size_t locx2, size_t locy1=0, size_t locy2=0);	
//	MpoQ<Symmetry> triplon (size_t locx, size_t locy=0);
//	MpoQ<Symmetry> antitriplon (size_t locx, size_t locy=0);
//	MpoQ<Symmetry> quadruplon (size_t locx, size_t locy=0);
	
	static const map<string,any> defaults;
	
protected:
	
	vector<FermionBase<Symmetry> > F;
};

const map<string,any> HubbardSU2xSU2::defaults = 
{
	{"t",1.}, {"tPerp",0.},
	{"U",0.}, {"J",0.}, {"Jperp",0.},
	{"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"subL",SUB_LATTICE::A}, {"Ly",1ul}
};

HubbardSU2xSU2::
HubbardSU2xSU2 (const size_t &L, const vector<Param> &params)
	:Mpo<Symmetry> (L, qarray<Symmetry::Nq>({1,1}), "", SfromD_SfromD)
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
	vector<SuperMatrix<Symmetry,double> > G;
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	F.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		F[l] = FermionBase<Symmetry>(P.get<size_t>("Ly",l%Lcell),P.get<SUB_LATTICE>("subL",l%Lcell));
		setLocBasis(F[l].get_basis().qloc(),l);
	}
	for (size_t l=0; l<N_sites; ++l)
	{
		Terms[l] = set_operators(F,P,l%Lcell);
		this->Daux = Terms[l].auxdim();
		
		G.push_back(Generator(Terms[l]));
		setOpBasis(G[l].calc_qOp(),l);
	}
	
	this->generate_label(Terms[0].name,Terms,Lcell);
	this->construct(G, this->W, this->Gvec, false, P.get<bool>("OPEN_BC"));
	// false: For SU(2) symmetries, the squared Hamiltonian cannot be calculated in advance.
}

HamiltonianTermsXd<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > HubbardSU2xSU2::
set_operators (const vector<FermionBase<Symmetry> > &F, const ParamHandler &P, size_t loc)
{
	HamiltonianTermsXd<Symmetry> Terms;
	
	auto save_label = [&Terms] (string label)
	{
		if (label!="") {Terms.info.push_back(label);}
	};

	param0d subL = P.fill_array0d<SUB_LATTICE>("subL","subL",loc);
	save_label(subL.label);

	// NN terms
	
	auto [t,tPara,tlabel] = P.fill_array2d<double>("t","tPara",F[loc].orbitals(),loc);
	save_label(tlabel);
		
	auto [J,Jpara,Jlabel] = P.fill_array2d<double>("J","Jpara",F[loc].orbitals(),loc);
	save_label(Jlabel);
	
	for (int i=0; i<F[loc%2].orbitals(); ++i)
	for (int j=0; j<F[(loc+1)%2].orbitals(); ++j)
	{
		if (tPara(i,j) != 0.)
		{
			auto cdagF = OperatorType::prod(F[loc%2].cdag(i),F[loc%2].sign(),{2,2});
			Terms.tight.push_back(make_tuple(-tPara(i,j)*sqrt(2.)*sqrt(2.), cdagF.plain<double>(), F[(loc)%2].c(j).plain<double>()));
		}
		
		// if (Vpara(i,j) != 0.)
		// {
		// 	Terms.tight.push_back(make_tuple(Vpara(i,j), F.n(i).plain<double>(), F.n(j).plain<double>()));
		// }
		
		if (Jpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(-sqrt(3)*Jpara(i,j), F[loc].Sdag(i).plain<double>(), F[(loc+1)%2].S(j).plain<double>()));
		}
	}
	
	// NNN terms
	
	// param0d tPrime = P.fill_array0d<double>("tPrime","tPrime",loc);
	// save_label(tPrime.label);
	
//	if (tPrime.x != 0.)
//	{
//		assert(F.orbitals() == 1 and "Cannot do a ladder with t'!");
//		
//		Terms.nextn.push_back(make_tuple(tPrime.x*sqrt(2.), F.cdag(), Operator::prod(F.sign(),F.c(),{2,-1}), F.sign()));
//		Terms.nextn.push_back(make_tuple(tPrime.x*sqrt(2.), F.c(), Operator::prod(F.sign(),F.cdag(),{2,1}), F.sign()));
//	}
	
	// local terms
	
	// Hubbard-U
	auto [U,Uorb,Ulabel] = P.fill_array1d<double>("U","Uorb",F[loc].orbitals(),loc);
	save_label(Ulabel);
		
	// t⟂
	param0d tPerp = P.fill_array0d<double>("t","tPerp",loc);
	save_label(tPerp.label);

	// J⟂
	param0d Jperp = P.fill_array0d<double>("J","Jperp",loc);
	save_label(Jperp.label);
	
	Terms.local.push_back(make_tuple(1.,F[loc].HubbardHamiltonian(Uorb,tPerp.x,0.,Jperp.x, P.get<bool>("CYLINDER")).plain<double>()));
	Terms.name = "Hubbard SU(2)⊗SU(2)";
	
	return Terms;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > HubbardSU2xSU2::
c (size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<F[locx].dim());
	stringstream ss;
	ss << "c(" << locx << "," << locy << ")";
	
	Mpo<Symmetry> Mout(N_sites, {2,2}, ss.str(), SfromD_SfromD);
	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	Mout.setLocal(locx, F[locx].c(locy).plain<double>(), F[0].sign().plain<double>());
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > HubbardSU2xSU2::
cdag (size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<F[locx].dim());
	stringstream ss;
	ss << "c†(" << locx << "," << locy << ")";
	
	Mpo<Symmetry> Mout(N_sites, {2,2}, ss.str(), SfromD_SfromD);
	for (size_t l=0; l<N_sites; ++l) {Mout.setLocBasis(F[l].get_basis().qloc(),l);}
	Mout.setLocal(locx, F[locx].cdag(locy).plain<double>(), F[0].sign().plain<double>());
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > HubbardSU2xSU2::
cdagc (size_t locx1, size_t locx2, size_t locy1, size_t locy2)
{
	assert(locx1<this->N_sites and locx2<this->N_sites);
	stringstream ss;
	ss << "c†(" << locx1 << "," << locy1 << ")" << "c(" << locx2 << "," << locy2 << ")";
	
	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str(), SfromD_SfromD);
	for (size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F[l].get_basis().qloc(),l); }
	
	auto cdag = F[locx1].cdag(locy1);
	auto c    = F[locx2].c   (locy2);
	
	if (locx1 == locx2)
	{
		//The diagonal element is actually 2*unity by the symmetry. But we may leave this as a check.
		Mout.setLocal(locx1, sqrt(2.) * sqrt(2.) * OperatorType::prod(cdag,c,Symmetry::qvacuum()).plain<double>());
	}
	else if (locx1<locx2)
	{
		Mout.setLocal({locx1, locx2}, {sqrt(2.)*sqrt(2.)*OperatorType::prod(cdag, F[locx1].sign(), {2,2}).plain<double>(), 
					                   c.plain<double>()}, 
		                               F[0].sign().plain<double>());
	}
	else if (locx1>locx2)
	{
		Mout.setLocal({locx2, locx1}, {sqrt(2.)*sqrt(2.)*OperatorType::prod(c, F[locx2].sign(), {2,2}).plain<double>(), 
		                               -1.*cdag.plain<double>()}, 
		                               F[0].sign().plain<double>());
	}
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > HubbardSU2xSU2::
nh (size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<F[locx].dim());
	stringstream ss;
	ss << "holon_occ(" << locx << "," << locy << ")";
	
	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str(), SfromD_SfromD);
	for (size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F[l].get_basis().qloc(),l); }
	
	Mout.setLocal(locx, F[locx].nh(locy).plain<double>());
	return Mout;
}

Mpo<Sym::S1xS2<Sym::SU2<Sym::SpinSU2>,Sym::SU2<Sym::ChargeSU2> > > HubbardSU2xSU2::
ns (size_t locx, size_t locy)
{
	assert(locx<N_sites and locy<F[locx].dim());
	stringstream ss;
	ss << "spinon_occ(" << locx << "," << locy << ")";
	
	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str(), SfromD_SfromD);
	for (size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis(F[l].get_basis().qloc(),l); }
	
	Mout.setLocal(locx, F[locx].ns(locy).plain<double>());
	return Mout;
}

} //end namespace VMPS
#endif
