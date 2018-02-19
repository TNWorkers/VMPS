#ifndef STRAWBERRY_HEISENBERGSU2
#define STRAWBERRY_HEISENBERGSU2

#include "symmetry/SU2.h"
#include "bases/SpinBaseSU2.h"
#include "Mpo.h"
#include "DmrgExternal.h"
#include "ParamHandler.h" // from HELPERS

namespace VMPS
{

/** \class HeisenbergSU2
  * \ingroup Heisenberg
  *
  * \brief Heisenberg Model
  *
  * MPO representation of 
  * \f[
  * H = -J \sum_{<ij>} \left(\mathbf{S_i}\mathbf{S_j}\right) -J' \sum_{<<ij>>} \left(\mathbf{S_i}\mathbf{S_j}\right)
  * \f]
  *
  * \note Take use of the Spin SU(2) symmetry, which implies no magnetic fields. For using B-fields see VMPS::HeisenbergU1.
  * \note The default variable settings can be seen in \p HeisenbergSU2::defaults.
  * \note \f$J<0\f$ is antiferromagnetic
  */
class HeisenbergSU2 : public Mpo<Sym::SU2<Sym::SpinSU2>,double>
{
public:
	typedef Sym::SU2<Sym::SpinSU2> Symmetry;
	
private:
	
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	typedef Eigen::SparseMatrix<double> SparseMatrixType;
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	
	typedef SiteOperatorQ<Symmetry,MatrixType> Operator;
	
public:
	
	//---constructors---
	///\{
	/**Do nothing.*/
	HeisenbergSU2() : Mpo<Symmetry>() {};
	
	/**
	   \param L : chain length
	   \describe_params
	*/
	HeisenbergSU2 (const size_t &L, const vector<Param> &params={});
	///\}
	
	/**
	 * \describe_set_operators
	 *
	 * \param B : Base class from which the local operators are received
	 * \param P : The parameters
	 * \param loc : The location in the chain
	*/
	static HamiltonianTermsXd<Symmetry> set_operators (const SpinBase<Symmetry> &B, const ParamHandler &P, size_t loc=0);
		
	///@{
	/**Observables.*/
	Mpo<Symmetry,double> S (std::size_t locx, std::size_t locy=0);
	Mpo<Symmetry,double> Sdag (std::size_t locx, std::size_t locy=0);
	Mpo<Symmetry,double> SS (std::size_t locx1, std::size_t locx2, std::size_t locy1=0, std::size_t locy2=0);
	///@}
	
	/**Validates whether a given total quantum number \p qnum is a possible target quantum number for an MpsQ.
	\returns \p true if valid, \p false if not*/
	bool validate (qarray<1> qnum) const;
	
protected:
	
	const std::map<string,std::any> defaults = 
	{
		{"J",-1.}, {"Jprime",0.}, {"Jperp",0.}, {"D",2ul},
		{"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}
	};
	
	vector<SpinBase<Symmetry> > B;
};

HeisenbergSU2::
HeisenbergSU2 (const size_t &L, const vector<Param> &params)
:Mpo<Symmetry> (L, qarray<Symmetry::Nq>({1}), "", true)
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
	vector<SuperMatrix<Symmetry,double> > G;
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	B.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		
		B[l] = SpinBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), P.get<size_t>("D",l%Lcell));
		setLocBasis(B[l].get_basis().qloc(),l);
		
		Terms[l] = set_operators(B[l],P,l%Lcell);
		this->Daux = Terms[l].auxdim();
		
		G.push_back(Generator(Terms[l]));
		setOpBasis(G[l].calc_qOp(),l);
	}
	
	this->generate_label(Terms[0].name,Terms,Lcell);
	this->construct(G, this->W, this->Gvec, false, P.get<bool>("OPEN_BC"));
	// false: For SU(2) symmetries, the squared Hamiltonian cannot be calculated in advance.
}

Mpo<Sym::SU2<Sym::SpinSU2> > HeisenbergSU2::
S (std::size_t locx, std::size_t locy)
{
	assert(locx<this->N_sites);
	std::stringstream ss;
	ss << "S(" << locx << "," << locy << ")";

	SiteOperator Op = B[locx].S(locy).plain<double>();

	Mpo<Symmetry> Mout(N_sites, Op.Q, ss.str());
	for (std::size_t l=0; l<N_sites; l++) { Mout.setLocBasis(B[l].get_basis().qloc(),l); }

	Mout.setLocal(locx,Op);
	return Mout;
}

Mpo<Sym::SU2<Sym::SpinSU2> > HeisenbergSU2::
Sdag (std::size_t locx, std::size_t locy)
{
	assert(locx<this->N_sites);
	std::stringstream ss;
	ss << "Sdag(" << locx << "," << locy << ")";

	SiteOperator Op = B[locx].Sdag(locy).plain<double>();

	Mpo<Symmetry> Mout(N_sites, Op.Q, ss.str());
	for (std::size_t l=0; l<N_sites; l++) { Mout.setLocBasis(B[l].get_basis().qloc(),l); }

	Mout.setLocal(locx,Op);
	return Mout;
}

Mpo<Sym::SU2<Sym::SpinSU2> > HeisenbergSU2::
SS (std::size_t locx1, std::size_t locx2, std::size_t locy1, std::size_t locy2)
{
	assert(locx1<this->N_sites and locx2<this->N_sites);
	std::stringstream ss;
	ss << "S(" << locx1 << "," << locy1 << ")" << "S(" << locx2 << "," << locy2 << ")";
	
	Mpo<Symmetry> Mout(N_sites,Symmetry::qvacuum(),ss.str());
	for (std::size_t l=0; l<N_sites; l++) { Mout.setLocBasis(B[l].get_basis().qloc(),l); }
	
	if (locx1 == locx2)
	{
//		auto product = std::sqrt(3.)*Operator::prod(B[locx1].Sdag(locy1),B[locx2].S(locy2),Symmetry::qvacuum());
		auto product = Operator::prod(B[locx1].Sdag(locy1),B[locx2].S(locy2),Symmetry::qvacuum());
		Mout.setLocal(locx1,product.plain<double>());
		return Mout;
	}
	else
	{
//		Mout.setLocal({locx1, locx2}, {(std::sqrt(3.)*B[locx1].Sdag(locy1)).plain<double>(), B[locx2].S(locy2).plain<double>()});
		Mout.setLocal({locx1, locx2}, {(B[locx1].Sdag(locy1)).plain<double>(), B[locx2].S(locy2).plain<double>()});
		return Mout;
	}
}

bool HeisenbergSU2::
validate (qarray<1> qnum) const
{
	frac Smax(0,1);
	frac q_in(qnum[0]-1,2);
	for (size_t l=0; l<N_sites; ++l) { Smax+=frac(B[l].get_D()-1,2); }
	if(Smax.denominator()==q_in.denominator() and q_in <= Smax) {return true;}
	else {return false;}
}

HamiltonianTermsXd<Sym::SU2<Sym::SpinSU2> > HeisenbergSU2::
set_operators (const SpinBase<Symmetry> &B, const ParamHandler &P, size_t loc)
{
	HamiltonianTermsXd<Symmetry> Terms;
	Terms.name = "Heisenberg";
	
	frac S = frac(B.get_D()-1,2);
	stringstream Slabel;
	Slabel << "S=" << print_frac_nice(S);
	Terms.info.push_back(Slabel.str());
	
	// J-terms
	
	auto [J,Jpara,Jlabel] = P.fill_array2d<double>("J","Jpara",B.orbitals(),loc);
	Terms.info.push_back(Jlabel);
	
	for (int i=0; i<B.orbitals(); ++i)
	for (int j=0; j<B.orbitals(); ++j)
	{
		if (Jpara(i,j) != 0.)
		{
			Terms.tight.push_back(make_tuple(-std::sqrt(3)*Jpara(i,j),
			                                 B.Sdag(i).plain<double>(),
			                                 B.S(j).plain<double>()));
		}
	}
	
	// J'-terms
	
	param0d Jprime = P.fill_array0d<double>("Jprime","Jprime",loc);
	if(!Jprime.label.empty()) {Terms.info.push_back(Jprime.label);}
	assert((B.orbitals() == 1 or Jprime.x == 0) and "Cannot interpret Ly>1 and J'!=0");
	if (Jprime.x != 0)
	{
		Terms.nextn.push_back(make_tuple(-std::sqrt(3)*Jprime.x, B.Sdag(0).plain<double>(),
		                                 B.S(0).plain<double>(),
		                                 B.Id().plain<double>()));
	}
	
	// local terms
	
	param0d Jperp = P.fill_array0d<double>("J","Jperp",loc);
	if(!Jperp.label.empty()) {Terms.info.push_back(Jperp.label);}
	if (B.orbitals() > 1)
	{
		Terms.local.push_back(make_tuple(1., B.HeisenbergHamiltonian(Jperp.x).plain<double>()));
	}
	
	return Terms;
}

} //end namespace VMPS

#endif
