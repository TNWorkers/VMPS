#ifndef VUMPSLINEARALGEBRA
#define VUMPSLINEARALGEBRA

#include "tensors/Multipede.h"

/**Calculates the matrix element between two Umps and an Mpo. Goes from the left and uses \f$A_C\f$ and \f$A_R\f$.*/
template<typename Symmetry, typename MpoScalar, typename Scalar>
Scalar avg (const Umps<Symmetry,Scalar> &Vbra, 
            const Mpo<Symmetry,MpoScalar> &O, 
            const Umps<Symmetry,Scalar> &Vket)
{
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Bnext;
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > B;
	size_t amount_of_cells = 1;
	auto Obs = O;
	if(Obs.length() != Vket.length())
	{
		amount_of_cells = static_cast<size_t>(Obs.length()/Vket.length());
		qarray<Symmetry::Nq> transformed_Qtot = ::adjustQN<Symmetry>(Vket.Qtarget(),amount_of_cells);
		Obs.transform_base(transformed_Qtot,false);
	}
	else
	{
		Obs.transform_base(Vket.Qtarget(),false);
	}
	auto Vbra_copy = Vbra;
	auto Vket_copy = Vket;
	if(Obs.length() != Vket.length())
	{
		Vbra_copy.adjustQN(amount_of_cells);
		Vket_copy.adjustQN(amount_of_cells);
	}
	
	// for(size_t g=0; g<1; g++)
	// for(size_t l=0; l<Vket.length(); l++)
	// for(size_t s=0; s<Vket.A[g][l].size(); s++)
	// {
	// 	cout << "g=" << g << ", l=" << l << ", s=" << s << endl;
	// 	cout << "qs=" << Vket.locBasis(l)[s] << endl;
	// 	cout << Vket.A[g][l][s].print(false) << endl;
	// }
	
	B.setIdentity(Obs.auxdim(), 1, Vket_copy.inBasis(0));
	for (size_t l=0; l<Obs.length(); ++l)
	{
		GAUGE::OPTION g = (l==0)? GAUGE::C : GAUGE::R;
		contract_L(B,
				   Vbra_copy.A_at(g,l%Vket.length()), Obs.W_at(l), O.IS_HAMILTONIAN(),
				   Vket_copy.A_at(g,l%Vket.length()), Obs.locBasis(l), Obs.opBasis(l), Bnext);
		
		B.clear();
		B = Bnext;
		Bnext.clear();
	}
	

	// Obs.transform_base(Vket.Qtarget(),false);

	// B.setIdentity(Obs.auxdim(), 1, Vket.inBasis(0));
	// for (size_t l=0; l<Obs.length(); ++l)
	// {
	// 	GAUGE::OPTION g = (l==0)? GAUGE::C : GAUGE::R;
	// 	contract_L(B, Vbra.A_at(g,l%Vket.length()), Obs.W_at(l), O.IS_HAMILTONIAN(), Vket.A_at(g,l%Vket.length()), Obs.locBasis(l), Obs.opBasis(l), Bnext);
		
	// 	B.clear();
	// 	B = Bnext;
	// 	Bnext.clear();
	// }
	
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > IdR;
	// IdR.setIdentity(Obs.auxdim(), 1, Vket.outBasis((Obs.length()-1)%Vket.length()));
	IdR.setIdentity(Obs.auxdim(), 1, Vket_copy.outBasis((Obs.length()-1)%Vket.length()));

	// cout << IdR.print(false) << endl << B.print(false) << endl;
	return contract_LR(B,IdR);
	
//	if (B.dim == 1)
//	{
//		return B.block[0][0][0].trace();
//	}
//	else
//	{
//		lout << "Warning: Result of VUMPS-contraction in <φ|O|ψ> has several blocks, returning 0!" << endl;
//		lout << "UMPS in question: " << Vket.info() << endl;
//		lout << "MPO in question: "  << O.info() << endl;
//		return 0;
//	}
}

/**Calculates the matrix element for a vector of MPOs, summing up the result.*/
template<typename Symmetry, typename MpoScalar, typename Scalar>
Scalar avg (const Umps<Symmetry,Scalar> &Vbra, 
            const vector<Mpo<Symmetry,MpoScalar> > &O, 
            const Umps<Symmetry,Scalar> &Vket)
{
	Scalar out = 0;
	
	for (int t=0; t<O.size(); ++t)
	{
//		cout << "partial val=" << avg(Vbra,O[t],Vket) << endl;
		out += avg(Vbra,O[t],Vket);
	}
	return out;
}

#endif
