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
	static HamiltonianTermsXd<Symmetry> set_operators (const vector<SpinBase<Symmetry> > &B, const vector<FermionBase<Symmetry> > &F,
	                                                    const ParamHandler &P, size_t loc=0);
	
	Mpo<Symmetry> Simp (SPINOP_LABEL Sa, size_t locx, size_t locy=0) const;
	
	static const std::map<string,std::any> defaults;
	
protected:
	
	vector<FermionBase<Symmetry> > F;
	vector<SpinBase<Symmetry> > B;
};

const std::map<string,std::any> KondoU0xSU2::defaults =
{
	{"t",1.}, {"tPerp",0.},
	{"J",-1.}, 
	{"Bz",0.}, {"Bx",0.}, {"Bzsub",0.}, {"Bxsub",0.}, {"Kz",0.}, {"Kx",0.},
	{"D",2ul},
	{"U",0.}, {"V",0.},
	{"CALC_SQUARE",false}, {"CYLINDER",false}, {"OPEN_BC",true}, {"Ly",1ul}
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

		F[l] = (l%2 == 0) ? FermionBase<Symmetry>(P.get<size_t>("Ly",l%Lcell),SUB_LATTICE::A) 
		                  : FermionBase<Symmetry>(P.get<size_t>("Ly",l%Lcell),SUB_LATTICE::B);
		B[l] = SpinBase<Symmetry>(P.get<size_t>("Ly",l%Lcell), P.get<size_t>("D",l%Lcell));
		
		setLocBasis((B[l].get_structured_basis().combine(F[l].get_basis())).qloc(),l);
		
		// Terms[l] = set_operators(B,F,P,l%Lcell);
	}
	for (size_t l=0; l<N_sites; ++l)
	{
		Terms[l] = set_operators(B,F,P,l%Lcell);
	}
	
	this->construct_from_Terms(Terms, Lcell, P.get<bool>("CALC_SQUARE"), P.get<bool>("OPEN_BC"));
}

HamiltonianTermsXd<Sym::SU2<Sym::ChargeSU2> > KondoU0xSU2::
set_operators (const vector<SpinBase<Symmetry> > &B, const vector<FermionBase<Symmetry> > &F, const ParamHandler &P, size_t loc)
{
	HamiltonianTermsXd<Symmetry> Terms;

	frac S = frac(B[loc].get_D()-1,2);
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
			//-------------------------------------------------------------------------------------------------------------------------------------//
			// Terms.tight.push_back(make_tuple(-tPara(i,j),
			//                                  kroneckerProduct(B.Id(), F.cdag(UP,i) * F.sign()),
			//                                  kroneckerProduct(B.Id(), F.c(UP,j))));
			// Terms.tight.push_back(make_tuple(-tPara(i,j),
			//                                  kroneckerProduct(B.Id(), F.cdag(DN,i) * F.sign()),
			//                                  kroneckerProduct(B.Id(), F.c(DN,j))));
			// Terms.tight.push_back(make_tuple(-tPara(i,j),
			//                                  kroneckerProduct(B.Id(), -1.*F.c(UP,i) * F.sign()),
			//                                  kroneckerProduct(B.Id(), F.cdag(UP,j))));
			// Terms.tight.push_back(make_tuple(-tPara(i,j),
			//                                  kroneckerProduct(B.Id(), -1.*F.c(DN,i) * F.sign()),
			//                                  kroneckerProduct(B.Id(), F.cdag(DN,j))));

			// Mout += -t*std::sqrt(2.)*(Operator::prod(psidag(UP,i),psi(UP,i+1),{1})+Operator::prod(psidag(DN,i),psi(DN,i+1),{1}));
			//-------------------------------------------------------------------------------------------------------------------------------------//

			//c†UPcUP

			auto Otmp = OperatorType::prod(OperatorType::outerprod(B[loc].Id().structured(),F[loc].psidag(UP,i),{2}),
										   OperatorType::outerprod(B[loc].Id().structured(),F[loc].sign()      ,{1}),
										   {2});
			Terms.tight.push_back(make_tuple(-tPara(i,j)*sqrt(2.),
											 Otmp.plain<double>(),
											 OperatorType::outerprod(B[loc].Id().structured(),F[loc].psi(UP,j),{2}).plain<double>()));

			//c†DNcDN
			Otmp = OperatorType::prod(OperatorType::outerprod(B[loc].Id().structured(),F[loc].psidag(DN,i),{2}),
									  OperatorType::outerprod(B[loc].Id().structured(),F[loc].sign()      ,{1}),
									  {2});
			Terms.tight.push_back(make_tuple(-tPara(i,j)*sqrt(2.),
											 Otmp.plain<double>(),
											 OperatorType::outerprod(B[loc].Id().structured(),F[loc].psi(DN,j),{2}).plain<double>()));

			//-cUPc†UP
			// Otmp = OperatorType::prod(OperatorType::outerprod(B.Id().structured(),F.psi(UP,i),{2}),
			// 						  OperatorType::outerprod(B.Id().structured(),F.sign()   ,{1}),
			// 						  {2});
			// Terms.tight.push_back(make_tuple(tPara(i,j)*sqrt(2.),
			// 								 Otmp.plain<double>(),
			// 								 OperatorType::outerprod(B.Id().structured(),F.psidag(UP,j),{2}).plain<double>()));

			//-cDNc†DN
			// Otmp = OperatorType::prod(OperatorType::outerprod(B.Id().structured(),F.psi(DN,i),{2}),
			// 						  OperatorType::outerprod(B.Id().structured(),F.sign()   ,{1}),
			// 						  {2});
			// Terms.tight.push_back(make_tuple(tPara(i,j)*sqrt(2.),
			// 								 Otmp.plain<double>(),
			// 								 OperatorType::outerprod(B.Id().structured(),F.psidag(DN,j),{2}).plain<double>()));

			//-------------------------------------------------------------------------------------------------------------------------------------//
		}
	}
	
	// local terms
	
	// t⟂
	param0d tPerp = P.fill_array0d<double>("t","tPerp",loc);
	save_label(tPerp.label);
	
	// Hubbard U
	auto [U,Uorb,Ulabel] = P.fill_array1d<double>("U","Uorb",F[loc].orbitals(),loc);
	save_label(Ulabel);

	// Bx substrate
	auto [Bxsub,Bxsuborb,Bxsublabel] = P.fill_array1d<double>("Bxsub","Bxsuborb",F[loc].orbitals(),loc);
	save_label(Bxsublabel);
	
	// Bx impurities
	auto [Bx,Bxorb,Bxlabel] = P.fill_array1d<double>("Bx","Bxorb",F[loc].orbitals(),loc);
	save_label(Bxlabel);
	
	// Kx anisotropy
	auto [Kx,Kxorb,Kxlabel] = P.fill_array1d<double>("Kx","Kxorb",B[loc].orbitals(),loc);
	save_label(Kxlabel);
	
	// Bz substrate
	auto [Bzsub,Bzsuborb,Bzsublabel] = P.fill_array1d<double>("Bzsub","Bzsuborb",F[loc].orbitals(),loc);
	save_label(Bzsublabel);
	
	// Bz impurities
	auto [Bz,Bzorb,Bzlabel] = P.fill_array1d<double>("Bz","Bzorb",F[loc].orbitals(),loc);
	save_label(Bzlabel);
	
	// Kx anisotropy
	auto [Kz,Kzorb,Kzlabel] = P.fill_array1d<double>("Kz","Kzorb",B[loc].orbitals(),loc);
	save_label(Kzlabel);

	// OperatorType KondoHamiltonian({1},B[loc].get_structured_basis().combine(F[loc].get_basis()));

	//set Heisenberg part of Kondo Hamiltonian
	auto KondoHamiltonian = OperatorType::outerprod(B[loc].HeisenbergHamiltonian(0.,0.,Bzorb,Bxorb,Kzorb,Kxorb,0.,
																			P.get<bool>("CYLINDER")).structured(),
													F[loc].Id(),
													{1});

	//set Hubbard part of Kondo Hamiltonian
	KondoHamiltonian += OperatorType::outerprod(B[loc].Id().structured(),
												F[loc].HubbardHamiltonian(Uorb,tPerp.x,0.,0.,0.,Bzsuborb,Bxsuborb),
												{1});


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
			KondoHamiltonian += -Jorb(i)    * OperatorType::outerprod(B[loc].Scomp(SZ,i).structured(), F[loc].Sz(i), {1});
			KondoHamiltonian += -0.5*Jorb(i)* OperatorType::outerprod(B[loc].Scomp(SP,i).structured(), F[loc].Sm(i), {1});
			KondoHamiltonian += -0.5*Jorb(i)* OperatorType::outerprod(B[loc].Scomp(SM,i).structured(), F[loc].Sp(i), {1});
		}
	}
	
	Terms.name = "Kondo U(0)⊗SU(2)";
	Terms.local.push_back(make_tuple(1.,KondoHamiltonian.plain<double>()));
	
	return Terms;
}

Mpo<Sym::SU2<Sym::ChargeSU2> > KondoU0xSU2::
Simp (SPINOP_LABEL Sa, size_t locx, size_t locy) const
{
	assert(locx < this->N_sites);
	std::stringstream ss;
//	ss << "S(" << loc1x << "," << loc1y << ")" << "S(" << loc2x << "," << loc2y << ")";
	
	Mpo<Symmetry> Mout(N_sites, Symmetry::qvacuum(), ss.str());
	for (std::size_t l=0; l<this->N_sites; l++) { Mout.setLocBasis((B[l].get_structured_basis().combine(F[l].get_basis())).qloc(),l); }
	
	auto Sop = OperatorType::outerprod(B[locx].Scomp(Sa,locy).structured(), F[locx].Id(), {1});
	
	Mout.setLocal(locx, Sop.plain<double>());
	return Mout;
}

} //end namespace VMPS

#endif