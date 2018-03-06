#ifndef KONDOMODEL_U0XU1_H_
#define KONDOMODEL_U0XU1_H_

#include "ParamHandler.h" // from HELPERS

#include "bases/SpinBase.h"
#include "bases/FermionBaseU0xSU2.h"
#include "Mpo.h"

namespace VMPS
{
/** \class KondoU0xSU2
  * \ingroup Kondo
  *
  * \brief Kondo Model
  *
  * MPO representation of
  * \f[
  * H = - \sum_{<ij>\sigma} c^\dagger_{i\sigma}c_{j\sigma} -t^{\prime} \sum_{<<ij>>\sigma} c^\dagger_{i\sigma}c_{j\sigma} 
  * - J \sum_{i \in I} \mathbf{S}_i \cdot \mathbf{s}_i
  * \f].
  *
  * where further parameters from HubbardSU2xU1 and HeisenbergSU2 are possible.
  * \note The default variable settings can be seen in \p KondoU0xSU2::defaults.
  * \note Take use of the Spin SU(2) symmetry and U(1) charge symmetry.
  * \note If the nnn-hopping is positive, the ground state energy is lowered.
  * \warning \f$J<0\f$ is antiferromagnetic
  */
class KondoU0xSU2 : public Mpo<Sym::SU2<Sym::ChargeSU2>,double>
{
public:
	typedef Sym::SU2<Sym::ChargeSU2> Symmetry;
private:
	typedef Eigen::Index Index;
	typedef Symmetry::qType qType;
	typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> MatrixType;
	typedef SiteOperatorQ<Symmetry,MatrixType> OperatorType;
	
public:
	///@{
	KondoU0xSU2() : Mpo() {};
	KondoU0xSU2 (const size_t &L, const vector<Param> &params);
	///@}
	
	/**
	 * \describe_set_operators
	 *
	 * \param B : Base class from which the local spin-operators are received
	 * \param F : Base class from which the local fermion-operators are received
	 * \param P : The parameters
	 * \param loc : The location in the chain
	 */
	static HamiltonianTermsXd<Symmetry> set_operators (const SpinBase<Symmetry> &B, const vector<FermionBase<Symmetry> > &F,
	                                                    const ParamHandler &P, size_t loc=0);
	
	static const std::map<string,std::any> defaults;
	
protected:
	
	vector<FermionBase<Symmetry> > F;
	vector<SpinBase<Symmetry> > B;
};

const std::map<string,std::any> KondoU0xSU2::defaults =
{
	{"t",1.}, {"tPerp",0.},
	{"J",-1.}, 
	{"U",0.}, 
	{"Bz",0.}, {"Bx",0.}, {"Bzsub",0.}, {"Bxsub",0.}, {"Kz",0.}, {"Kx",0.},
	{"D",2ul},
	{"CALC_SQUARE",true}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}
};

KondoU0xSU2::
KondoU0xSU2 (const size_t &L, const vector<Param> &params)
	:Mpo<Symmetry> (L, Symmetry::qvacuum(), "", true)
{
	ParamHandler P(params,defaults);
	
	size_t Lcell = P.size();
	vector<HamiltonianTermsXd<Symmetry> > Terms(N_sites);
	B.resize(N_sites);
	F.resize(N_sites);
	
	for (size_t l=0; l<N_sites; ++l)
	{
		N_phys += P.get<size_t>("Ly",l%Lcell);
		
		F[l] = FermionBase<Symmetry>(P.get<size_t>("Ly",l%Lcell),P.get<SUB_LATTICE>("subL",l%Lcell));
		B[l] = SpinBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), P.get<size_t>("D",l%Lcell));
		
		setLocBasis((B[l].get_basis().combine(F[l].get_basis())).qloc(),l);
		
		Terms[l] = set_operators(B[l],F[l],P,l%Lcell);
	}
	for (size_t l=0; l<N_sites; ++l)
	{
		Terms[l] = set_operators(B[l],F,P,l%Lcell);
	}
	
	this->construct_from_Terms(Terms, Lcell, false, P.get<bool>("OPEN_BC"));
	// false: For SU(2) symmetries, the squared Hamiltonian cannot be calculated in advance.
}

HamiltonianTermsXd<Sym::SU2<Sym::ChargeSU2> > KondoU0xSU2::
set_operators (const SpinBase<Symmetry> &B, const vector<FermionBase<Symmetry> > &F, const ParamHandler &P, size_t loc)
{
	HamiltonianTermsXd<Symmetry> Terms;

	frac S = frac(B.get_D()-1,2);
	stringstream Slabel;
	Slabel << "S=" << print_frac_nice(S);
	Terms.info.push_back(Slabel.str());
	
	auto save_label = [&Terms] (string label)
	{
		if (label!="") {Terms.info.push_back(label);}
	};
	
	// NN terms
	
	auto [t,tPara,tlabel] = P.fill_array2d<double>("t","tPara",F[loc].orbitals(),loc);
	save_label(tlabel);
	
	auto [V,Vpara,Vlabel] = P.fill_array2d<double>("V","Vpara",F[loc].orbitals(),loc);
	save_label(Vlabel);
	
	for (int i=0; i<F[loc%2].orbitals(); ++i)
	for (int j=0; j<F[(loc+1)%2].orbitals(); ++j)
	{
		if (tPara(i,j) != 0.)
		{
			auto cdagF = OperatorType::prod(F[loc%2].psidag(UP),F[loc%2].sign(),{2,2});
			Terms.tight.push_back(make_tuple(-tPara(i,j)*sqrt(2.)*sqrt(2.), cdagF.plain<double>(), F[(loc)%2].psi(UP).plain<double>()));
			
			cdagF = OperatorType::prod(F[loc%2].psidag(DN),F[loc%2].sign(),{2,2});
			Terms.tight.push_back(make_tuple(-tPara(i,j)*sqrt(2.)*sqrt(2.), cdagF.plain<double>(), F[(loc)%2].psi(DN).plain<double>()));
		}
	}
	
	// local terms
	
	// t⟂
	param0d tPerp = P.fill_array0d<double>("t","tPerp",loc);
	save_label(tPerp.label);
	
	// Hubbard U
	auto [U,Uorb,Ulabel] = P.fill_array1d<double>("U","Uorb",F[loc].orbitals(),loc);
	save_label(Ulabel);
	
	OperatorType KondoHamiltonian({1,0},B.get_basis().combine(F[loc].get_basis()));
	
	//set Hubbard part of Kondo Hamiltonian
	KondoHamiltonian = OperatorType::outerprod(B.Id(), F[loc].HubbardHamiltonian(U,tPerp.x), {1});
	
	//set Heisenberg part of Hamiltonian
//	KondoHamiltonian += OperatorType::outerprod(B.HeisenbergHamiltonian(0.,P.get<bool>("CYLINDER")),F[loc].Id(),{1,0});
	
	// Kondo-J
	auto [J,Jorb,Jlabel] = P.fill_array1d<double>("J","Jorb",F[loc].orbitals(),loc);
	save_label(Jlabel);
	
	//set interaction part of Hamiltonian.
	for (int i=0; i<F[loc].orbitals(); ++i)
	{
		if (Jorb(i) != 0.)
		{
			KondoHamiltonian += -Jorb(i)    * OperatorType::outerprod(B.Scomp(SZ,i), F[loc].Sz(i), {1});
			KondoHamiltonian += -0.5*Jorb(i)* OperatorType::outerprod(B.Scomp(SP,i), F[loc].Sm(i), {1});
			KondoHamiltonian += -0.5*Jorb(i)* OperatorType::outerprod(B.Scomp(SM,i), F[loc].Sp(i), {1});
		}
	}
	
	Terms.name = "Kondo SU(2)⊗U(1)";
	Terms.local.push_back(make_tuple(1.,KondoHamiltonian.plain<double>()));
	
	return Terms;
}

} //end namespace VMPS

#endif
