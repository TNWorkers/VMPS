#ifndef VUMPSLINEARALGEBRA
#define VUMPSLINEARALGEBRA

#include "tensors/Multipede.h"

/**Calculates the matrix element between two UMPS and an MPO. Goes from the left and uses \f$A_C\f$ and \f$A_R\f$.*/
template<typename Symmetry, typename MpoScalar, typename Scalar>
Scalar avg (const Umps<Symmetry,Scalar> &Vbra, 
            const Mpo<Symmetry,MpoScalar> &O, 
            const Umps<Symmetry,Scalar> &Vket)
{
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > Bnext;
	Tripod<Symmetry,Matrix<Scalar,Dynamic,Dynamic> > B;
	
	B.setIdentity(Vbra.get_frst_rows(),Vbra.get_frst_rows(),1,1);
	for (size_t l=0; l<O.length(); ++l)
	{
		GAUGE::OPTION g = (l==0)? GAUGE::C : GAUGE::R;
		contract_L(B, Vbra.A_at(g,l%Vket.length()), O.W_at(l), Vket.A_at(g,l%Vket.length()), O.locBasis(l), O.opBasis(l), Bnext);
		
		B.clear();
		B = Bnext;
		Bnext.clear();
	}
	
	if (B.dim == 1)
	{
		return B.block[0][0][0].trace();
	}
	else
	{
		lout << "Warning: Result of VUMPS-contraction in <φ|O|ψ> has several blocks, returning 0!" << endl;
		lout << "UMPS in question: " << Vket.info() << endl;
		lout << "MPO in question: "  << O.info() << endl;
		return 0;
	}
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
